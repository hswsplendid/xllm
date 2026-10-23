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

"""Tests for the FlashInfer attention backend."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("FlashInfer tests require CUDA", allow_module_level=True)
pytest.importorskip("flashinfer", reason="FlashInfer is not installed")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend, _should_use_tensor_core_decode
from xllm.python.layers.attention import Attention


def test_tensor_core_decode_for_large_gqa_groups():
    assert _should_use_tensor_core_decode(torch.bfloat16, 24, 4)
    assert _should_use_tensor_core_decode(torch.float16, 32, 8)


def test_cuda_core_decode_for_small_gqa_groups_or_float32():
    assert not _should_use_tensor_core_decode(torch.bfloat16, 8, 4)
    assert not _should_use_tensor_core_decode(torch.float32, 24, 4)


@pytest.mark.parametrize("rows", [512, 1024])
@pytest.mark.parametrize("num_heads,num_kv_heads,override", [(32, 8, None), (2, 2, True)])
@torch.inference_mode()
def test_large_verification_graph_reserves_split_kv_workspace(
    rows: int, num_heads: int, num_kv_heads: int, override: bool | None
) -> None:
    device = torch.device("cuda")
    backend = FlashInferBackend(
        num_heads,
        num_kv_heads,
        128,
        128**-0.5,
        -1,
        device,
        torch.bfloat16,
        max_decode_rows=rows,
        use_tensor_core_decode=override,
    )
    layer = Attention(num_heads, num_kv_heads, 128, 128**-0.5, -1, 0)
    generator = torch.Generator(device=device).manual_seed(20261022)
    keys = torch.randn(rows, num_kv_heads, 128, device=device, dtype=torch.bfloat16, generator=generator)
    values = torch.randn(rows, num_kv_heads, 128, device=device, dtype=torch.bfloat16, generator=generator)
    queries = torch.randn(rows, num_heads, 128, device=device, dtype=torch.bfloat16, generator=generator)
    backend.bind_kv_caches([LayerCache(torch.zeros_like(keys).unsqueeze(1), torch.zeros_like(values).unsqueeze(1))])
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        slot_mapping=torch.arange(rows, device=device, dtype=torch.int32),
        paged_kv_indptr=torch.arange(rows + 1, device=device, dtype=torch.int32),
        paged_kv_indices=torch.arange(rows, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(rows, device=device, dtype=torch.int32),
        paged_kv_indptr_host=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_last_page_len_host=torch.ones(rows, dtype=torch.int32),
    )
    backend.prepare(metadata, graph_mode=True)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.execute(queries, keys, values, layer)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = backend.execute(queries, keys, values, layer)
    graph.replay()
    expected = values.repeat_interleave(num_heads // num_kv_heads, dim=1).reshape(rows, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("paged", [True, False])
def test_block_attention_matches_dense_reference(causal: bool, paged: bool) -> None:
    """Check future-token visibility and isolation across paged requests."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(123)
    query_lengths = [5, 7]
    context_lengths = [3, 13] if paged else [0, 0]
    kv_lengths = [query + context for query, context in zip(query_lengths, context_lengths)]
    num_heads, num_kv_heads, head_dim, page_size = 8, 2, 64, 16
    scale = head_dim**-0.5
    backend = FlashInferBackend(num_heads, num_kv_heads, head_dim, scale, -1, device, torch.bfloat16, causal=causal)
    layer = Attention(num_heads, num_kv_heads, head_dim, scale, -1, 0, causal=causal)
    keys = [
        torch.randn(length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator)
        for length in kv_lengths
    ]
    values = [torch.randn(key.shape, device=device, dtype=key.dtype, generator=generator) for key in keys]
    queries = [
        torch.randn(length, num_heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator)
        for length in query_lengths
    ]
    page_counts = [(length + page_size - 1) // page_size for length in kv_lengths]
    page_offsets = [0, page_counts[0], sum(page_counts)]
    key_cache = torch.zeros(sum(page_counts), page_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping: list[int] = []
    for index, context in enumerate(context_lengths):
        offset = page_offsets[index] * page_size
        key_cache.view(-1, num_kv_heads, head_dim)[offset : offset + context].copy_(keys[index][:context])
        value_cache.view(-1, num_kv_heads, head_dim)[offset : offset + context].copy_(values[index][:context])
        slot_mapping.extend(range(offset + context, offset + kv_lengths[index]))
    backend.bind_kv_caches([LayerCache(key_cache, value_cache)])

    def indices(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.int32)

    metadata = SimpleNamespace(
        is_prefill=not paged,
        is_chunked_prefill=paged,
        slot_mapping=indices(slot_mapping),
        q_cu_seq_lens=indices([0, query_lengths[0], sum(query_lengths)]),
        kv_cu_seq_lens=indices([0, kv_lengths[0], sum(kv_lengths)]),
        qo_indptr=indices([0, query_lengths[0], sum(query_lengths)]),
        paged_kv_indptr=indices(page_offsets),
        paged_kv_indices=indices(list(range(sum(page_counts)))),
        paged_kv_last_page_len=indices([(length - 1) % page_size + 1 for length in kv_lengths]),
    )
    backend.prepare(metadata)
    actual = backend.execute(
        torch.cat(queries),
        torch.cat([key[context:] for key, context in zip(keys, context_lengths)]),
        torch.cat([value[context:] for value, context in zip(values, context_lengths)]),
        layer,
    )
    references: list[torch.Tensor] = []
    for query, key, value, context in zip(queries, keys, values, context_lengths):
        mask = None
        if causal:
            mask = torch.arange(key.size(0), device=device)[None, :] <= (
                context + torch.arange(query.size(0), device=device)[:, None]
            )
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).float(),
            key.repeat_interleave(num_heads // num_kv_heads, dim=1).transpose(0, 1).float(),
            value.repeat_interleave(num_heads // num_kv_heads, dim=1).transpose(0, 1).float(),
            attn_mask=mask,
            scale=scale,
        )
        references.append(output.transpose(0, 1).reshape(query.size(0), -1))
    torch.testing.assert_close(actual.float(), torch.cat(references), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("override,expected", [(None, False), (False, False), (True, True)])
def test_mha_decode_precision_choice_is_explicit(override: bool | None, expected: bool) -> None:
    backend = FlashInferBackend(
        2, 2, 128, 128**-0.5, 0, torch.device("cuda:0"), torch.bfloat16, use_tensor_core_decode=override
    )
    assert backend._decode_use_tensor_cores is expected
    assert backend._decode_wrapper.use_tensor_cores is expected


def test_float32_tensor_core_decode_is_rejected_before_allocation() -> None:
    with pytest.raises(ValueError, match="requires float16 or bfloat16"):
        FlashInferBackend(2, 2, 128, 128**-0.5, 0, torch.device("cuda:0"), torch.float32, use_tensor_core_decode=True)
