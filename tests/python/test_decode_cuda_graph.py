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

"""CUDA graph output contracts and Qwen3 auxiliary-output regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner
from xllm.python.models.qwen3 import Qwen3ForCausalLM
from xllm.python.platform import current_platform


@pytest.mark.parametrize("kind", ["tensor", "aux", "none_none", "aux_none", "none_topk", "aux_topk"])
def test_cuda_graph_output_snapshots_preserve_structure_and_ownership(kind: str) -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    aux_hidden = hidden + 100
    topk = torch.arange(8, dtype=torch.int64).reshape(4, 1, 2)
    outputs: dict[str, ModelExecutionOutput] = {
        "tensor": hidden,
        "aux": (hidden, aux_hidden),
        "none_none": (hidden, None, None),
        "aux_none": (hidden, aux_hidden, None),
        "none_topk": (hidden, None, topk),
        "aux_topk": (hidden, aux_hidden, topk),
    }
    output = outputs[kind]
    actual = DecodeCudaGraphRunner._slice_output(output, 3)
    assert isinstance(actual, tuple) == isinstance(output, tuple)
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = output if isinstance(output, tuple) else (output,)
    assert len(actual_items) == len(expected_items)
    for result, reference in zip(actual_items, expected_items):
        if reference is None:
            assert result is None
            continue
        assert result is not None
        assert result.shape == (3, *reference.shape[1:])
        assert result.dtype == reference.dtype
        assert result.device == reference.device
        expected = reference[:3].clone()
        reference.add_(1000)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layers_to_capture", [(), (1, 0)])
@pytest.mark.parametrize("profile_costs", [False, True])
@torch.inference_mode()
def test_qwen3_cuda_graph_preserves_outputs_across_replays(
    layers_to_capture: tuple[int, ...], profile_costs: bool
) -> None:
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("flashinfer")
    device = torch.device("cuda", torch.cuda.current_device())
    config = {
        "model_type": "qwen3",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "layers_to_capture": layers_to_capture,
        "dtype": "bfloat16",
        "device": str(device),
        "adaptive_speculative_cost_model": "graph" if profile_costs else "linear",
    }
    model = Qwen3ForCausalLM(config).eval()
    generator = torch.Generator(device=device).manual_seed(20261020)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.02, generator=generator)
    executors = []
    for backend in ("off", "cudagraphs"):
        executor = ModelExecutor(model, {**config, "python_graph_backend": backend}, max_seqs_per_batch=4)
        executor.bind_kv_caches(
            [
                LayerCache(
                    torch.zeros(4, 128, 2, 64, dtype=torch.bfloat16, device=device),
                    torch.zeros(4, 128, 2, 64, dtype=torch.bfloat16, device=device),
                )
                for _ in range(2)
            ]
        )
        executors.append(executor)
    graph_runner = executors[1].decode_graph_runner
    assert isinstance(graph_runner, DecodeCudaGraphRunner)
    costs = executors[1].decode_graph_costs()
    if profile_costs:
        assert [rows for rows, _ in costs] == [1, 2, 4]
        assert all(milliseconds > 0 for _, milliseconds in costs)
    else:
        assert costs == []
    lengths = [0, 0, 0]
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step, requests in enumerate(((0, 1, 2), (0,), (2, 0), (1, 2, 0), (2, 1, 0))):
        batch_size = len(requests)
        tokens = torch.tensor([11 + request + step for request in requests], dtype=torch.int32, device=device)
        positions = torch.tensor([lengths[request] for request in requests], dtype=torch.int32, device=device)
        cumulative_lengths = [0]
        for request in requests:
            lengths[request] += 1
            cumulative_lengths.append(cumulative_lengths[-1] + lengths[request])
        # Give each request a separate reversed physical page; rows may reorder.
        pages = torch.tensor([3 - request for request in requests], dtype=torch.int32, device=device)
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            slot_mapping=pages * 128 + positions,
            kv_cu_seq_lens=torch.tensor(cumulative_lengths, dtype=torch.int32, device=device),
            kv_seq_lens_host=torch.tensor(cumulative_lengths, dtype=torch.int32),
            paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indices=pages,
            paged_kv_last_page_len=positions + 1,
        )
        assert graph_runner.can_execute(tokens, metadata)
        expected = executors[0].execute(tokens, positions, metadata)
        actual = executors[1].execute(tokens, positions, metadata)
        if layers_to_capture:
            assert isinstance(expected, tuple) and isinstance(actual, tuple)
            assert len(actual) == 2
            assert actual[1].shape == (batch_size, config["hidden_size"] * len(layers_to_capture))
            pairs = zip(actual, expected)
        else:
            assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
            pairs = [(actual, expected)]
        for result, reference in pairs:
            assert result.shape[0] == batch_size
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            retained.append((result, reference.clone()))
    # Reusing the same bucket must not overwrite either previously returned tensor.
    for result, reference in retained:
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def _verification_metadata(
    widths: tuple[int, ...], prefixes: tuple[int, ...], device: torch.device, expanded: bool
) -> SimpleNamespace:
    slots: list[int] = []
    positions: list[int] = []
    pages: list[int] = []
    page_indptr = [0]
    last_page_lengths: list[int] = []
    cumulative_kv = [0]
    cumulative_q = [0]
    for request, (width, prefix) in enumerate(zip(widths, prefixes)):
        first_page = 1 + 2 * request  # Page zero is reserved for startup capture.
        for offset in range(width):
            position = prefix + offset
            positions.append(position)
            slots.append(first_page * 128 + position)
            if expanded:
                length = position + 1
                block_count = (length + 127) // 128
                pages.extend(range(first_page, first_page + block_count))
                page_indptr.append(len(pages))
                last_page_lengths.append((length - 1) % 128 + 1)
                cumulative_kv.append(cumulative_kv[-1] + length)
                cumulative_q.append(cumulative_q[-1] + 1)
        if not expanded:
            length = prefix + width
            pages.extend(range(first_page, first_page + (length + 127) // 128))
            page_indptr.append(len(pages))
            last_page_lengths.append((length - 1) % 128 + 1)
            cumulative_kv.append(cumulative_kv[-1] + length)
            cumulative_q.append(cumulative_q[-1] + width)
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=not expanded,
        slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        kv_cu_seq_lens=torch.tensor(cumulative_kv, dtype=torch.int32, device=device),
        kv_seq_lens_host=torch.tensor(cumulative_kv, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor(cumulative_q, dtype=torch.int32, device=device),
        qo_indptr=torch.tensor(cumulative_q, dtype=torch.int32, device=device),
        paged_kv_indptr=torch.tensor(page_indptr, dtype=torch.int32, device=device),
        paged_kv_indices=torch.tensor(pages, dtype=torch.int32, device=device),
        paged_kv_last_page_len=torch.tensor(last_page_lengths, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int32, device=device),
        linear_state_indices=torch.arange(len(widths), dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("widths", [(1, 8), (8, 2), (4, 6)])
@torch.inference_mode()
def test_cuda_graph_varlen_verification_matches_causal_chunked_prefill(widths: tuple[int, ...]) -> None:
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("flashinfer")
    device = torch.device("cuda", torch.cuda.current_device())
    config = dict(
        model_type="qwen3",
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=256,
        max_position_embeddings=256,
        layers_to_capture=[0, 1],
        dtype="bfloat16",
        device=str(device),
        num_speculative_tokens=7,
    )
    model = Qwen3ForCausalLM(config).eval()
    generator = torch.Generator(device=device).manual_seed(20261022)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.02, generator=generator)
    cache_values = [
        torch.randn(5, 128, 2, 64, dtype=torch.bfloat16, device=device, generator=generator) * 0.02 for _ in range(4)
    ]
    executors = []
    for backend in ("off", "cudagraphs", "off"):
        executor = ModelExecutor(model, {**config, "python_graph_backend": backend}, max_seqs_per_batch=2)
        executor.bind_kv_caches(
            [LayerCache(cache_values[2 * layer].clone(), cache_values[2 * layer + 1].clone()) for layer in range(2)]
        )
        executors.append(executor)
    reference_metadata = _verification_metadata(widths, (126, 123), device, False)
    expanded_metadata = _verification_metadata(widths, (126, 123), device, True)
    tokens = torch.arange(sum(widths), dtype=torch.int32, device=device) + 11
    expected = executors[0].execute(tokens, reference_metadata.positions, reference_metadata)
    runner = executors[1].decode_graph_runner
    assert isinstance(runner, DecodeCudaGraphRunner)
    assert runner.can_execute(tokens, expanded_metadata)
    actual = executors[1].execute(tokens, expanded_metadata.positions, expanded_metadata)
    eager_expanded = executors[2].execute(tokens, expanded_metadata.positions, expanded_metadata)
    assert isinstance(expected, tuple) and isinstance(actual, tuple)
    assert isinstance(eager_expanded, tuple)
    for result, reference in zip(actual, eager_expanded):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    # Chunked prefill and expanded decode use different attention kernels;
    # BF16 roundoff is permitted, but model predictions must agree.
    for result, reference in zip(actual, expected):
        tolerance = 4 * torch.finfo(result.dtype).eps
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)
    actual_logits = model.lm_head(actual[0])
    expected_logits = model.lm_head(expected[0])
    torch.testing.assert_close(actual_logits.argmax(-1), expected_logits.argmax(-1), rtol=0, atol=0)


class _StateIndexModel(torch.nn.Module):
    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        state_indices = get_forward_context().metadata.linear_state_indices
        return tokens + positions + state_indices


@torch.inference_mode()
def test_cuda_graph_linear_state_indices_survive_warmup_and_row_reordering() -> None:
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("flashinfer")
    from xllm.python.attention.flashinfer import FlashInferBackend

    device = torch.device("cuda", torch.cuda.current_device())
    backend = FlashInferBackend(4, 2, 64, 0.125, -1, device, torch.bfloat16)
    caches = [
        LayerCache(
            torch.zeros(5, 128, 2, 64, dtype=torch.bfloat16, device=device),
            torch.zeros(5, 128, 2, 64, dtype=torch.bfloat16, device=device),
            conv=torch.zeros(4, 2, 2, dtype=torch.bfloat16, device=device),
        )
    ]
    backend.bind_kv_caches(caches)
    runner = DecodeCudaGraphRunner(_StateIndexModel(), backend, device, max_batch=3, max_model_len=256)
    runner.bind_layer_caches(caches)
    runner.warmup(device, torch.bfloat16)
    metadata = _verification_metadata((1, 1), (3, 5), device, True)
    tokens = torch.tensor([11, 12], dtype=torch.int32, device=device)
    for indices in ([3, 1], [1, 3]):
        metadata.linear_state_indices = torch.tensor(indices, dtype=torch.int32, device=device)
        actual = runner.execute(tokens, metadata.positions, metadata)
        expected = tokens + metadata.positions + metadata.linear_state_indices
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metadata.linear_state_indices = metadata.linear_state_indices[:1]
    with pytest.raises(ValueError, match="one state index per query row"):
        runner.execute(tokens, metadata.positions, metadata)
