# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Real-device coverage for output gating, view addressing and graph replay."""

from __future__ import annotations

import pytest
import torch

from xllm.python import kernels

pytestmark = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required")


def _assert_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    tolerance = (1e-2, 2e-2) if actual.dtype == torch.bfloat16 else (2e-3, 2e-3)
    torch.testing.assert_close(actual, reference, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("fused_qkv,enabled", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_attention_forward_gate_before_projection(
    monkeypatch: pytest.MonkeyPatch, fused_qkv: bool, enabled: bool, tokens: int, graph_mode: bool
) -> None:
    """Exercise the real layer forward, with controlled QKV/attention inputs.

    QKV normalization/attention are replaced to isolate the gate boundary;
    the NPU gate operator and output projection execute on the device.
    """
    from types import SimpleNamespace

    from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention

    heads, head_dim = 4, 256
    width = heads * head_dim
    dtype, device = torch.bfloat16, "npu:0"
    q = torch.randn(tokens, heads, head_dim, dtype=dtype, device=device)
    gate = torch.randn_like(q)
    key = torch.randn_like(q)
    value = torch.randn_like(q)
    attention_output = torch.randn(tokens, width, dtype=dtype, device=device)
    qg = torch.cat((q, gate), dim=-1).reshape(tokens, 2 * width) if enabled else q.reshape(tokens, width)
    qkv = torch.cat((qg, key.reshape(tokens, width), value.reshape(tokens, width)), dim=-1)
    saved_qkv = qkv.clone()

    class FixedAttention(torch.nn.Module):
        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return attention_output

    class IdentityRotary(torch.nn.Module):
        def forward(self, positions: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return value

        def build_mrope_cos_sin(self, positions: torch.Tensor) -> torch.Tensor:
            return torch.zeros(tokens, 64, device=device, dtype=dtype)

    layer = NpuQwen3_5Attention.__new__(NpuQwen3_5Attention)
    torch.nn.Module.__init__(layer)
    layer.use_fused_qkv = fused_qkv
    layer.attn_output_gate = enabled
    layer.num_heads = layer.num_kv_heads = heads
    layer.head_dim = head_dim
    layer.q_size = layer.kv_size = width
    layer.qkv_proj = torch.nn.Identity()
    layer.q_norm = torch.nn.Identity()
    layer.k_norm = torch.nn.Identity()
    layer.q_norm.weight = torch.nn.Parameter(torch.zeros(head_dim, device=device, dtype=dtype))
    layer.k_norm.weight = torch.nn.Parameter(torch.zeros(head_dim, device=device, dtype=dtype))
    layer.rotary = IdentityRotary()
    layer.attn = FixedAttention()
    layer.cfg = SimpleNamespace(rms_norm_eps=1e-6)
    layer._mrope_gather_pattern = torch.zeros(64, device=device, dtype=torch.int32)
    layer.o_proj = torch.nn.Linear(width, 64, bias=False, device=device, dtype=dtype)

    def split_qkv(*_args: object) -> tuple[torch.Tensor, ...]:
        return q, key, value, gate

    monkeypatch.setattr(kernels, "split_qkv_rmsnorm_mrope", split_qkv)
    from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context

    saved_gate = gate.clone()
    gate_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    fused_gate = kernels.fused_sigmoid_mul

    def record_fused_gate(output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        gate_calls.append((output, value))
        return fused_gate(output, value)

    monkeypatch.setattr(kernels, "fused_sigmoid_mul", record_fused_gate)
    context = ForwardContext(
        None, torch.device(device), None, [], execution_state=AclGraphExecutionState({}) if graph_mode else None
    )
    expected_input = attention_output
    if enabled:
        expected_input = (
            (attention_output.float() * torch.sigmoid(gate.float().reshape(tokens, width))).to(dtype)
            if graph_mode
            else attention_output * torch.sigmoid(gate.reshape(tokens, width))
        )
    with torch.inference_mode(), forward_context(context):
        expected = layer.o_proj(expected_input)
        actual = layer(torch.arange(tokens, device=device), qkv)
    _assert_close(actual, expected)
    assert len(gate_calls) == int(enabled and graph_mode)
    torch.testing.assert_close(qkv, saved_qkv, rtol=0, atol=0)
    if fused_qkv and not graph_mode:
        torch.testing.assert_close(gate, torch.sigmoid(saved_gate), rtol=0, atol=0)
    else:
        torch.testing.assert_close(gate, saved_gate, rtol=0, atol=0)
