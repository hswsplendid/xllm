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

"""Regression checks for the CUDA service acceptance tool's failure signals."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from tools import validate_olmo2_cuda as validation


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("tp", [1, 2])
def test_loaded_configuration_matches_requested_dtype_and_all_ranks(dtype: str, tp: int) -> None:
    loaded = [
        f"Loaded 179 OLMo2 checkpoint tensors; tp_size={tp}, tp_rank={rank}, dtype=torch.{dtype}" for rank in range(tp)
    ]
    validation._validate_loaded_configuration(loaded, tp, dtype)


@pytest.mark.parametrize("fault", ["dtype", "missing_rank", "duplicate_rank", "wrong_size", "incomplete"])
def test_loaded_configuration_rejects_incorrect_dtype_or_topology(fault: str) -> None:
    loaded = [f"tp_rank={rank}, tp_size=2, dtype=torch.bfloat16" for rank in range(2)]
    if fault == "dtype":
        loaded[1] = loaded[1].replace("bfloat16", "float16")
    elif fault == "missing_rank":
        loaded.pop()
    elif fault == "duplicate_rank":
        loaded[1] = loaded[0]
    elif fault == "wrong_size":
        loaded[1] = loaded[1].replace("tp_size=2", "tp_size=20")
    else:
        loaded[1] = "tp_rank=1, dtype=torch.bfloat16"
    with pytest.raises(ValueError, match="OLMo2 loaded"):
        validation._validate_loaded_configuration(loaded, 2, "bfloat16")


@pytest.mark.parametrize("architecture", [None, "8.0"])
@pytest.mark.parametrize("memory,cache", [(0.4, 1073741824), (0.99, 536870912)])
def test_launch_preserves_architecture_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, architecture: str | None, memory: float, cache: int
) -> None:
    if architecture is None:
        monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", architecture)
    binary = tmp_path / "xllm"
    binary.write_bytes(b"test executable; never launched")
    monkeypatch.setattr(validation.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation.Tokenizer, "from_file", lambda *args, **kwargs: object())
    captured: dict[str, str] = {}

    def _stop_before_launch(*args: Any, env: dict[str, str], **kwargs: Any) -> None:
        captured.update(env)
        raise RuntimeError("stop before launching a GPU server")

    monkeypatch.setattr(validation.subprocess, "Popen", _stop_before_launch)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validate_olmo2_cuda.py",
            "--model",
            str(tmp_path),
            "--binary",
            str(binary),
            "--output",
            str(tmp_path / "output"),
            "--tp",
            "1",
            "--max-memory-utilization",
            str(memory),
            "--max-cache-size",
            str(cache),
        ],
    )
    with pytest.raises(RuntimeError, match="stop before launching"):
        validation.main()
    assert captured.get("FLASHINFER_CUDA_ARCH_LIST") == architecture
    command = json.loads((tmp_path / "output" / "launch.json").read_text())["command"]
    assert f"--max_memory_utilization={memory}" in command
    assert f"--max_cache_size={cache}" in command


@pytest.mark.parametrize(
    "option,value",
    [
        ("--max-memory-utilization", "0"),
        ("--max-memory-utilization", "1.1"),
        ("--max-memory-utilization", "nan"),
        ("--max-cache-size", "0"),
        ("--max-cache-size", "-1"),
    ],
)
def test_invalid_memory_budget_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, option: str, value: str
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validate_olmo2_cuda.py",
            "--model",
            str(tmp_path),
            "--binary",
            str(tmp_path / "xllm"),
            "--output",
            str(output),
            "--tp",
            "1",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        validation.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("fault", ["missing_done", "missing_finish", "missing_usage", "stream_error"])
def test_incomplete_stream_cannot_pass_service_acceptance(fault: str) -> None:
    result = {"text": "你好", "finish_reason": "length", "usage": {"completion_tokens": 12}}
    events = [{"time": 0.1, "line": "data: [DONE]"}]
    if fault == "missing_done":
        events.clear()
    elif fault == "missing_finish":
        result["finish_reason"] = None
    elif fault == "missing_usage":
        result["usage"] = None
    else:
        events.insert(0, {"time": 0.05, "line": 'data: {"error": {"message": "decode failed"}}'})
    with pytest.raises(ValueError, match="stream"):
        validation._validate_stream(result, events, include_usage=True)


@pytest.mark.parametrize("include_usage", [False, True])
def test_finished_stream_is_accepted_with_requested_usage(include_usage: bool) -> None:
    result = {"text": "你好", "finish_reason": "stop", "usage": {"completion_tokens": 1} if include_usage else None}
    events = [
        {"time": 0.1, "line": "data: " + json.dumps({"choices": [{"finish_reason": "stop"}]})},
        {"time": 0.2, "line": "data: [DONE]"},
    ]
    validation._validate_stream(result, events, include_usage=include_usage)


def test_main_rejects_truncated_chat_stream_after_matching_all_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the real entrypoint so an unused stream checker cannot pass."""
    binary = tmp_path / "xllm"
    binary.write_bytes(b"test executable; never launched")
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [3])
    monkeypatch.setattr(validation.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(validation.Tokenizer, "from_file", lambda *args, **kwargs: object())
    completion = {"choices": [{"text": "Hello", "finish_reason": "length"}], "usage": {"completion_tokens": 12}}
    chat = {"choices": [{"message": {"content": "Hello"}, "finish_reason": "length"}]}
    full_stream = (
        'data: {"choices": [{"text": "Hello"}]}\n\n'
        'data: {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 12}}\n\n'
        "data: [DONE]\n\n"
    )
    truncated_chat = 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    bodies = iter([{"data": [{"id": tmp_path.name}]}, completion, full_stream, chat, truncated_chat])

    class _Session:
        trust_env = False

        def __enter__(self) -> _Session:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, *args: Any, **kwargs: Any) -> requests.Response:
            body = next(bodies)
            response = requests.Response()
            response.status_code = 200
            response.encoding = "utf-8"
            if isinstance(body, str):
                response.raw = io.BytesIO(body.encode())
            else:
                response._content = json.dumps(body).encode()
            return response

        post = get

    def _server(*args: Any, stdout: Any, **kwargs: Any) -> SimpleNamespace:
        stdout.write("Loaded 179 OLMo2 checkpoint tensors; tp_rank=0, tp_size=1, dtype=torch.float16\n")
        stdout.flush()
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: None, wait=lambda **kwargs: 0)

    monkeypatch.setattr(validation.requests, "Session", _Session)
    monkeypatch.setattr(validation.subprocess, "Popen", _server)
    monkeypatch.setattr(validation.os, "killpg", lambda *args: None)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validate_olmo2_cuda.py",
            "--model",
            str(tmp_path),
            "--binary",
            str(binary),
            "--output",
            str(tmp_path / "output"),
            "--tp",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="stream ended without"):
        validation.main()
    saved = json.loads((tmp_path / "output/chat-stream.json").read_text())
    assert saved["result"]["text"] == "Hello" and saved["result"]["finish_reason"] is None
    assert not (tmp_path / "output/result.json").exists()
