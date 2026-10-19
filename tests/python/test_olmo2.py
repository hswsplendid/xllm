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

"""OLMo2 model semantics, strict checkpoint loading and registry contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Olmo2Config as HfOlmo2Config
from transformers import Olmo2ForCausalLM as HfOlmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm as HfOlmo2RMSNorm

from xllm import xllm_export
from xllm.pybind.utils import _infer_model_type_and_backend
from xllm.python import distributed, registry
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.olmo2 import Olmo2Attention, Olmo2Config, Olmo2ForCausalLM, Olmo2RMSNorm
from xllm.python.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="OLMo2 currently supports the CUDA runtime only")


def _config(**overrides: Any) -> dict[str, Any]:
    values = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "intermediate_size": 24,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "use_cache": False,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.reads: set[str] = set()

    def keys(self) -> list[str]:
        return list(self.tensors)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        self.reads.add(name)
        return self.tensors[name]


def _reference() -> HfOlmo2ForCausalLM:
    torch.manual_seed(20261019)
    cfg = HfOlmo2Config(**{key: value for key, value in _config().items() if key not in ("device", "dtype")})
    cfg._attn_implementation = "eager"
    model = HfOlmo2ForCausalLM(cfg).eval()
    # Nonuniform, signed gamma makes the full-projection norm axis observable.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    return model


def test_official_configuration_and_cpp_field_aliases() -> None:
    fixture = Path(__file__).parents[1] / "models/fixtures/olmo2/config.json"
    cfg = Olmo2Config.from_dict(json.loads(fixture.read_text()))
    assert (cfg.hidden_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (2048, 16, 16, 16, 128)
    assert (cfg.intermediate_size, cfg.vocab_size, cfg.max_position_embeddings) == (8192, 100352, 4096)
    assert cfg.rms_norm_eps == 1e-6 and cfg.rope_theta == 500000.0
    config = _config()
    for hf_name, cpp_name in (
        ("num_hidden_layers", "n_layers"),
        ("num_attention_heads", "n_heads"),
        ("num_key_value_heads", "n_kv_heads"),
    ):
        config[cpp_name] = config.pop(hf_name)
    assert Olmo2Config.from_dict(config) == Olmo2Config.from_dict(_config())


def test_default_rope_parameters_preserve_theta() -> None:
    cfg = Olmo2Config.from_dict(_config(rope_parameters={"rope_type": "default", "rope_theta": 123456.0}))
    assert cfg.rope_theta == 123456.0


def test_native_backend_discovery_for_formal_entrypoint() -> None:
    assert xllm_export.get_model_backend("olmo2") == "llm"
    fixture = Path(__file__).parents[1] / "models/fixtures/olmo2"
    assert _infer_model_type_and_backend(str(fixture)) == ("olmo2", "llm")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"hidden_size": 15}, "divisible"),
        ({"head_dim": 3}, "even head_dim"),
        ({"num_key_value_heads": 2}, "MHA"),
        ({"tp_size": 3}, "TP=1/2"),
        ({"tp_size": 2, "tp_rank": 2}, "rank"),
        ({"dp_size": 2}, "DP=CP=PP=1"),
        ({"cp_size": 2}, "DP=CP=PP=1"),
        ({"world_size": 2}, "DP=CP=PP=1"),
        ({"intermediate_size": 25, "tp_size": 2}, "divide evenly"),
        ({"rms_norm_eps": float("nan")}, "positive and finite"),
        ({"rope_theta": 0}, "positive and finite"),
        ({"hidden_act": "gelu"}, "SiLU"),
        ({"attention_bias": True}, "bias-free"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"rope_scaling": {"factor": 2}}, "unscaled"),
        ({"rope_parameters": {"rope_type": "linear", "factor": 2}}, "unscaled"),
        ({"rope_parameters": []}, "must be an object"),
        ({"quant_method": "awq"}, "unquantized"),
    ],
)
def test_unsupported_configuration_is_rejected(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Olmo2Config.from_dict(_config(**change))


def test_missing_required_dimension_is_rejected() -> None:
    config = _config()
    del config["num_hidden_layers"]
    with pytest.raises(ValueError, match="requires config field: n_layers"):
        Olmo2Config.from_dict(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_matches_reference_without_intermediate_low_precision_cast(dtype: torch.dtype) -> None:
    torch.manual_seed(13)
    value = (torch.randn(3, 16) * 3.0).to(dtype)
    weight = torch.randn(16).to(dtype)
    norm = Olmo2RMSNorm(16, 1e-6, dtype, torch.device("cpu"))
    reference = HfOlmo2RMSNorm(16, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        reference.weight.copy_(weight)
        torch.testing.assert_close(norm(value), reference(value), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_qk_norm_reduces_separate_global_statistics(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    cfg = Olmo2Config.from_dict(_config(tp_size=2, tp_rank=rank))
    attention = Olmo2Attention(cfg, 0, torch.float32, torch.device("cpu"))
    full_q = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    full_k = torch.arange(101, 133, dtype=torch.float32).reshape(2, 16)
    q, k = full_q.chunk(2, dim=-1)[rank], full_k.chunk(2, dim=-1)[rank]
    remote_q, remote_k = full_q.chunk(2, dim=-1)[1 - rank], full_k.chunk(2, dim=-1)[1 - rank]
    calls: list[torch.Tensor] = []

    def _sum(stats: torch.Tensor) -> None:
        calls.append(stats.clone())
        stats.add_(torch.stack((remote_q.square().sum(-1), remote_k.square().sum(-1)), dim=-1))

    monkeypatch.setattr(distributed, "tp_all_reduce", _sum)
    gamma = torch.linspace(-1.3, 2.1, 16)
    with torch.no_grad():
        attention.q_norm.weight.copy_(gamma.chunk(2)[rank])
        attention.k_norm.weight.copy_(gamma.flip(0).chunk(2)[rank])
        actual_q, actual_k = attention._normalize_qk(q, k)
        for full, local, weight in ((full_q, actual_q, gamma), (full_k, actual_k, gamma.flip(0))):
            reference = HfOlmo2RMSNorm(16, eps=cfg.rms_norm_eps)
            reference.weight.copy_(weight)
            torch.testing.assert_close(local, reference(full).chunk(2, dim=-1)[rank])
    assert len(calls) == 1 and calls[0].shape == (2, 2) and calls[0].dtype == torch.float32


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_checkpoint_loading_consumes_all_keys_and_preserves_logical_shards(tp_size: int, rank: int) -> None:
    state = _StateDict(dict(_reference().state_dict()))
    model = Olmo2ForCausalLM(_config(tp_size=tp_size, tp_rank=rank))
    model.load_weights([state], rank, tp_size)
    tensors = state.tensors
    assert set(model.loaded_checkpoint_keys) == set(tensors) == state.reads
    assert len(state.reads) == 25
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        expected_qkv = torch.cat(
            [tensors[prefix + f"self_attn.{name}_proj.weight"].chunk(tp_size, dim=0)[rank] for name in ("q", "k", "v")]
        )
        torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv, rtol=0, atol=0)
        for name in ("q", "k"):
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{name}_norm").weight,
                tensors[prefix + f"self_attn.{name}_norm.weight"].chunk(tp_size)[rank],
                rtol=0,
                atol=0,
            )
        expected_mlp = torch.cat(
            [tensors[prefix + f"mlp.{name}_proj.weight"].chunk(tp_size, dim=0)[rank] for name in ("gate", "up")]
        )
        torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected_mlp, rtol=0, atol=0)
        torch.testing.assert_close(
            layer.self_attn.o_proj.weight, tensors[prefix + "self_attn.o_proj.weight"].chunk(tp_size, dim=1)[rank]
        )
        torch.testing.assert_close(
            layer.mlp.down_proj.weight, tensors[prefix + "mlp.down_proj.weight"].chunk(tp_size, dim=1)[rank]
        )
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            torch.testing.assert_close(getattr(layer, name).weight, tensors[prefix + name + ".weight"])
    torch.testing.assert_close(
        model.model.embed_tokens.weight, tensors["model.embed_tokens.weight"].chunk(tp_size, 1)[rank]
    )
    torch.testing.assert_close(model.lm_head.weight, tensors["lm_head.weight"].chunk(tp_size, 0)[rank])
    torch.testing.assert_close(model.model.norm.weight, tensors["model.norm.weight"])
    assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "wrong_shape"])
def test_invalid_checkpoint_fails_without_ignoring_tensors(fault: str) -> None:
    tensors = dict(_reference().state_dict())
    states = [_StateDict(tensors)]
    if fault == "missing":
        del tensors["lm_head.weight"]
    elif fault == "extra":
        tensors["model.layers.0.input_layernorm.weight"] = torch.ones(16)
    elif fault == "duplicate":
        states.append(_StateDict({"lm_head.weight": tensors["lm_head.weight"]}))
    else:
        tensors["model.layers.0.self_attn.q_norm.weight"] = torch.ones(4)
    model = Olmo2ForCausalLM(_config())
    with pytest.raises(ValueError, match="checkpoint"):
        model.load_weights(states, 0, 1)
    assert model.loaded_checkpoint_keys == ()


class _DenseAttentionBackend:
    """Unit reference for attention only; the model and CUDA MLP remain real."""

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        values = [value.view(1, -1, layer.num_heads, layer.head_dim).transpose(1, 2) for value in (q, k, v)]
        output = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True, scale=layer.scale)
        return output.transpose(1, 2).reshape(-1, layer.num_heads * layer.head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OLMo2 CUDA composition test requires a GPU")
def test_cuda_model_matches_tiny_transformers_at_nonzero_positions() -> None:
    device = torch.device("cuda:0")
    reference = _reference().to(device)
    model = Olmo2ForCausalLM(_config(device=str(device))).eval()
    model.load_weights([_StateDict(dict(reference.state_dict()))], 0, 1)
    tokens = torch.tensor([3, 5, 7], device=device)
    positions = torch.tensor([2, 5, 9], device=device)
    context = ForwardContext(
        attention_backend=cast(AttentionBackend, _DenseAttentionBackend()),
        device=device,
        metadata=cast(AttentionMetadata, SimpleNamespace()),
        layer_caches=[],
    )
    with torch.inference_mode(), forward_context(context):
        # Explicitly select one causal sequence: HF otherwise treats position gaps as packed sequence boundaries.
        expected = reference(
            tokens.unsqueeze(0),
            attention_mask=torch.ones_like(tokens.unsqueeze(0)),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).logits[0]
        actual = model.compute_logits(model.model(tokens, positions), None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert model.model.rotary.cos_sin_cache.dtype == torch.float32


def test_registry_resolves_both_names_and_rejects_unverified_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    assert registry.get_model_class("olmo2") is Olmo2ForCausalLM
    assert registry.get_model_class("Olmo2ForCausalLM") is Olmo2ForCausalLM
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    with pytest.raises(NotImplementedError, match="olmo2.*npu"):
        registry.get_model_class("olmo2")
