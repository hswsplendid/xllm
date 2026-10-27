# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen recurrent rollback and device-length attention under adaptive graphs."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.python.qwen3_5_test_utils import make_linear_config
from tests.python.test_prepared_acl_graph import _adaptive_inputs
from xllm.python.attention.backend import LayerCache
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.model_executor.forward_context import ForwardContext, forward_context, get_execution_context
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner
from xllm.python.models.qwen3_5 import Qwen3_5Config


def _config() -> Qwen3_5Config:
    return make_linear_config(
        hidden_size=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        rms_norm_eps=1e-6,
    )


@pytest.mark.parametrize("draft_width", [5, 7])
def test_adaptive_recurrent_indices_follow_live_request_identity(draft_width: int) -> None:
    _, _, _, inputs = _adaptive_inputs(torch.device("cpu"), graph_rows=8, draft_width=draft_width)
    inputs = replace(
        inputs,
        linear_state_indices=torch.tensor([3, 1, 4], dtype=torch.int32),
        accepted_lengths=torch.tensor([2, 1, 3], dtype=torch.int32),
    )
    builder = Qwen3_5GdnMetadataBuilder(_config())
    builder._state_caches, builder._checkpoint_stride = {}, draft_width + 1
    inputs.control.fill_(2)
    inputs.owners.copy_(torch.tensor([0, 0, 0, 1, 1, -1, -1, -1], dtype=torch.int32))
    inputs.columns.copy_(torch.tensor([0, 1, 2, 0, 1, 0, 0, 0], dtype=torch.int32))
    result = builder._build_adaptive_verify(inputs)
    assert result.state_indices.tolist() == [3, 1, 0]
    assert result.num_accepted_tokens.tolist() == [2, 1, 1]
    assert result.packed_to_dense[:5].tolist() == [0, 1, 2, draft_width + 1, draft_width + 2]
    assert result.packed_to_dense[5:].tolist() == list(range(3 * (draft_width + 1) + 5, 3 * (draft_width + 1) + 8))
    # Same borrowed addresses, different request and previous acceptance.
    inputs.control.fill_(1)
    inputs.linear_state_indices[0] = 2
    inputs.accepted_lengths[0] = draft_width + 1
    result = builder._build_adaptive_verify(inputs)
    assert result.state_indices.tolist() == [2, 0, 0]
    assert result.num_accepted_tokens.tolist() == [draft_width + 1, 1, 1]
    inputs.control.fill_(1 | (1 << 32))
    result = builder._build_adaptive_verify(inputs)
    assert result.state_indices.tolist() == [0, 0, 0]
    assert result.num_accepted_tokens.tolist() == [1, 1, 1]
    builder._checkpoint_stride = 1
    with pytest.raises(ValueError, match="checkpoint"):
        builder._build_adaptive_verify(inputs)


def test_qwen_prefill_accepts_prepared_cumulative_ends() -> None:
    builder = Qwen3_5GdnMetadataBuilder(_config())
    cache = LayerCache(
        None,
        None,
        conv=torch.zeros(4, 8, 384, dtype=torch.bfloat16),
        ssm=torch.zeros(24, 1, 128, 128, dtype=torch.float32),
    )
    builder.bind_layer_caches([cache])
    metadata = SimpleNamespace(
        slot_mapping=torch.zeros(5, dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([3, 2], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=[3, 5],
        has_initial_state=torch.tensor([False, True]),
        is_dummy=False,
        is_spec_verify=False,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    result = builder.build(metadata)
    assert result.cu_seqlens.tolist() == [0, 3, 5]
    assert result.conv_read_indices.tolist() == [-1, 2]
    assert result.ssm_read_indices.tolist() == [-1, 12]
    assert result.ssm_write_indices.tolist() == [6, 12]
    metadata.q_cu_seq_lens_host_values = [3, 6]
    with pytest.raises(ValueError, match="cumulative query lengths"):
        builder.build(metadata)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_qwen_adaptive_graph_matches_ragged_recurrent_reference(draft_width: int) -> None:
    device = torch.device("npu:0")
    cfg = _config()
    width = draft_width + 1
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    torch.manual_seed(173)
    cache = LayerCache(
        None,
        None,
        conv=torch.randn(5, width + 2, 384, dtype=torch.bfloat16, device=device) * 0.01,
        ssm=torch.randn(5 * width, 1, 128, 128, dtype=torch.float32, device=device) * 0.01,
    )

    class RecurrentTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.execution_metadata_builders = (builder,)
            self.embedding = torch.nn.Embedding(512, 128, dtype=torch.bfloat16, device=device)
            self.gdn = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
            for parameter in self.parameters():
                parameter.normal_(std=0.02)

        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return self.gdn._verify(self.embedding(tokens), get_execution_context(GdnMetadata))

    target = RecurrentTarget()
    backend = SimpleNamespace(
        prepare=lambda *args, **kwargs: None,
        prepare_graph_replay=lambda metadata: None,
        is_mla=True,
        logical_page_size=128,
    )
    runner = PreparedAclGraphRunner(target, backend, device, 8)
    runner.bind_layer_caches([cache])
    tokens, positions, metadata, inputs = _adaptive_inputs(device, table_width=1, graph_rows=8, draft_width=draft_width)
    inputs = replace(
        inputs,
        linear_state_indices=torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        accepted_lengths=torch.ones(3, dtype=torch.int32, device=device),
    )
    inputs.control.fill_(1 | (1 << 32))
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    before_capture_conv, before_capture_ssm = cache.conv.clone(), cache.ssm.clone()
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    torch.npu.current_stream(device).wait_stream(stream)
    torch.testing.assert_close(cache.conv[1:], before_capture_conv[1:], rtol=0, atol=0)
    torch.testing.assert_close(cache.ssm[width:], before_capture_ssm[width:], rtol=0, atol=0)
    for phase, batch in enumerate((3, 2, 1, 3)):
        inputs.control.fill_(batch)
        inputs.linear_state_indices.copy_(torch.tensor([3, 1, 2], dtype=torch.int32, device=device))
        inputs.accepted_lengths.fill_(1 if phase == 0 else 2)
        proposals = inputs.arena[: 8 * batch * draft_width].view(torch.int64).view(batch, draft_width)
        confidence = inputs.arena[8 * batch * draft_width : 12 * batch * draft_width].view(torch.float32)
        proposals.copy_(torch.arange(batch * draft_width, device=device).view(batch, draft_width) + 5)
        confidence.copy_(torch.linspace(0.98, 0.4, batch * draft_width, device=device))
        before_conv, before_ssm = cache.conv.clone(), cache.ssm.clone()
        actual = runner.execute(tokens, positions, metadata, dspark_adaptive=inputs).clone()
        torch.npu.synchronize()
        graph_conv, graph_ssm = cache.conv.clone(), cache.ssm.clone()
        live = int((inputs.owners >= 0).sum().item())
        lengths = torch.bincount(inputs.owners[:live].to(torch.int64), minlength=batch).cpu().to(torch.int32)
        cache.conv.copy_(before_conv)
        cache.ssm.copy_(before_ssm)
        reference_metadata = builder._build_verify(
            SimpleNamespace(
                q_seq_lens_host=lengths,
                slot_mapping=tokens[:live],
                num_accepted_tokens=inputs.accepted_lengths[:batch],
                num_accepted_tokens_host_values=inputs.accepted_lengths[:batch].cpu().tolist(),
            ),
            inputs.linear_state_indices[:batch],
        )
        expected = target.gdn._verify(target.embedding(tokens[:live]), reference_metadata)
        torch.npu.synchronize()
        torch.testing.assert_close(actual[:live], expected, rtol=0, atol=0)
        ids = inputs.linear_state_indices[:batch].to(torch.int64)
        checkpoints = (ids[:, None] * width + torch.arange(width, device=device)).flatten()
        torch.testing.assert_close(graph_conv.index_select(0, ids), cache.conv.index_select(0, ids), rtol=0, atol=0)
        torch.testing.assert_close(
            graph_ssm.index_select(0, checkpoints), cache.ssm.index_select(0, checkpoints), rtol=0, atol=0
        )
        inactive_ids = sorted(set(range(1, 5)) - set(ids.cpu().tolist()))
        for inactive in inactive_ids:
            torch.testing.assert_close(graph_conv[inactive], before_conv[inactive], rtol=0, atol=0)
            torch.testing.assert_close(
                graph_ssm[inactive * width : (inactive + 1) * width],
                before_ssm[inactive * width : (inactive + 1) * width],
                rtol=0,
                atol=0,
            )
        cache.conv.copy_(graph_conv)
        cache.ssm.copy_(graph_ssm)
    assert len(runner._prepared_graphs) == 1
    assert runner.prepared_replays == 4


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("heads,kv_heads,head_dim,table_width", [(2, 1, 128, 1), (12, 2, 256, 1), (12, 2, 256, 2048)])
@torch.inference_mode()
def test_prepared_dense_graph_respects_live_causal_lengths(
    heads: int, kv_heads: int, head_dim: int, table_width: int
) -> None:
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata

    device = torch.device("npu:0")
    torch.manual_seed(211)
    backend = NpuPagedAttentionBackend(heads, kv_heads, head_dim, head_dim**-0.5, -1, False, device, torch.bfloat16)
    keys = torch.randn(4, 128, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    backend.bind_kv_caches([LayerCache(keys, values)])
    queries = torch.randn(3, heads, head_dim, dtype=torch.bfloat16, device=device)
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=torch.zeros(3, dtype=torch.int32, device=device),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        block_table=torch.tensor([[1], [2], [3]], dtype=torch.int32, device=device),
        q_seq_lens=torch.ones(3, dtype=torch.int32, device=device),
        q_cu_seq_lens=torch.arange(1, 4, dtype=torch.int32, device=device),
        q_cu_seq_lens_host_values=[1, 2, 3],
        kv_seq_lens=torch.tensor([1, 5, 23], dtype=torch.int32, device=device),
        kv_seq_lens_host_values=[128, 128, 128],
    )
    if table_width > 1:
        tables = torch.zeros(3, table_width, dtype=torch.int32, device=device)
        tables[:, :1].copy_(metadata.block_table)
        metadata.block_table = tables
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with (
        forward_context(ForwardContext(backend, device, metadata, [LayerCache(keys, values)])),
        torch.npu.stream(stream),
    ):
        for _ in range(2):
            backend._decode(queries, keys, values, metadata, 3)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            backend._block_attention_masks.clear()
            output = backend._decode(queries, keys, values, metadata, 3)
    torch.npu.current_stream(device).wait_stream(stream)
    for lengths in ([1, 5, 23], [64, 17, 127], [7, 2, 11]):
        metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        graph.replay()
        torch.npu.synchronize()
        expected = []
        for row, length in enumerate(lengths):
            q = queries[row].cpu().float()
            k = keys[row + 1, :length].cpu().float().repeat_interleave(heads // kv_heads, dim=1)
            v = values[row + 1, :length].cpu().float().repeat_interleave(heads // kv_heads, dim=1)
            probabilities = torch.softmax(torch.einsum("hd,lhd->hl", q, k) * (head_dim**-0.5), dim=-1)
            expected.append(torch.einsum("hl,lhd->hd", probabilities, v))
        torch.testing.assert_close(
            output.cpu().view(3, heads, head_dim).float(), torch.stack(expected), rtol=0.02, atol=0.01
        )
        assert metadata.kv_seq_lens_host_values == [128, 128, 128]


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("use_fia_v2", [False, True])
@torch.inference_mode()
def test_qwen_hybrid_executor_captures_adaptive_dense_attention(use_fia_v2: bool) -> None:
    from xllm.python.model_executor.executor import ModelExecutor
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
    from xllm.python.models.qwen3_5 import Qwen3_5ForCausalLM

    device = torch.device("npu:0")
    config = {
        "model_type": "qwen3_5_text",
        "device": "npu:0",
        "dtype": "bfloat16",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "intermediate_size": 256,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "vocab_size": 512,
        "python_graph_backend": "aclgraph",
        "enable_task_pipeline": True,
        "num_speculative_tokens": 5,
    }
    torch.manual_seed(443)
    model = Qwen3_5ForCausalLM(config)
    for parameter in model.parameters():
        parameter.normal_(std=0.02)
    cache = LayerCache(
        None,
        None,
        conv=torch.randn(4, 8, 384, dtype=torch.bfloat16, device=device) * 0.01,
        ssm=torch.randn(24, 1, 128, 128, dtype=torch.float32, device=device) * 0.01,
    )
    full_cache = LayerCache(
        torch.randn(4, 128, 1, 128, dtype=torch.bfloat16, device=device) * 0.01,
        torch.randn(4, 128, 1, 128, dtype=torch.bfloat16, device=device) * 0.01,
    )
    executor = ModelExecutor(model, config, max_seqs_per_batch=3, num_decoding_tokens=6)
    executor.bind_kv_caches([cache, full_cache])
    executor.attention_backend._use_fia_v2 = use_fia_v2
    tokens, positions, metadata, inputs = _adaptive_inputs(device, table_width=1, graph_rows=8, draft_width=5)
    inputs.base_block_table.copy_(torch.tensor([[1], [2], [3]], dtype=torch.int32, device=device))
    inputs = replace(
        inputs,
        linear_state_indices=torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        accepted_lengths=torch.ones(3, dtype=torch.int32, device=device),
    )
    metadata.kv_seq_lens_host_values = [128] * 8
    inputs.control.fill_(2 | (1 << 32))
    executor.prepare_metadata(metadata)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        executor.warmup_prepared_graph(tokens, positions, metadata, dspark_adaptive=inputs)
    torch.npu.current_stream(device).wait_stream(stream)
    for bound in (16, 64, 128):
        metadata.kv_seq_lens_host_values = [bound] * 8
        inputs.base_kv_lengths.fill_(bound - 6)
        inputs.base_positions.fill_(bound - 7)
        executor.prepare_metadata(metadata)
        inputs.control.fill_(2)
        inputs.arena[:80].view(torch.int64).copy_(torch.arange(5, 15, device=device))
        inputs.arena[80:120].view(torch.float32).fill_(0.8)
        before_conv, before_ssm = cache.conv.clone(), cache.ssm.clone()
        actual = executor.execute(tokens, positions, metadata, enable_graph=True, dspark_adaptive=inputs).clone()
        torch.npu.synchronize()
        live = int((inputs.owners >= 0).sum().item())
        lengths = torch.bincount(inputs.owners[:live].to(torch.int64), minlength=2).cpu().to(torch.int32)
        cache.conv.copy_(before_conv)
        cache.ssm.copy_(before_ssm)
        ends = lengths.cumsum(0)
        reference = StaticGraphAttentionMetadata(
            slot_mapping=metadata.slot_mapping[:live],
            paged_kv_indptr=None,
            paged_kv_indices=None,
            paged_kv_last_page_len=None,
            block_table=inputs.base_block_table[:2],
            q_seq_lens=lengths.to(device),
            q_seq_lens_host=lengths,
            q_cu_seq_lens=torch.cat((torch.zeros(1, dtype=torch.int32), ends)).to(device),
            q_cu_seq_lens_host_values=ends.tolist(),
            kv_seq_lens=(inputs.base_kv_lengths[:2] + lengths.to(device) - 1),
            kv_seq_lens_host_values=(inputs.base_kv_lengths[:2].cpu() + lengths - 1).tolist(),
            linear_state_indices=inputs.linear_state_indices[:2],
            num_accepted_tokens=inputs.accepted_lengths[:2],
            num_accepted_tokens_host_values=(1, 1),
            is_spec_verify=True,
            is_chunked_prefill=True,
        )
        expected = executor.eager_runner.execute(tokens[:live], positions[:live], reference)
        torch.npu.synchronize()
        torch.testing.assert_close(actual[:live], expected, rtol=0.03, atol=0.03)
    assert len(executor.prepared_graph_runner._prepared_graphs) == 1
    assert executor.prepared_graph_runner.prepared_replays == 3
