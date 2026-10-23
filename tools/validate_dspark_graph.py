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

"""Reproducible HTTP correctness and performance runs for DSpark graph policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import logger


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _snapshot(url: str, device: str, output: Path) -> None:
    """Record serving counters and device state outside the timed interval."""
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(url + "/brpc_metrics", timeout=10)
        response.raise_for_status()
        output.with_suffix(".prom").write_text(response.text)
    snapshot = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + device,
            "--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output.with_suffix(".gpu.csv").write_text(snapshot.stdout)


def _completion(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(url + "/v1/completions", json=payload, stream=payload.get("stream", False), timeout=600)
        response.raise_for_status()
        if not payload.get("stream"):
            return {"elapsed": time.perf_counter() - start, "response": response.json()}
        events: list[dict[str, Any]] = []
        text = ""
        usage = None
        finish = None
        first = None
        done = False
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            elapsed = time.perf_counter() - start
            if line == "data: [DONE]":
                done = True
                continue
            item = json.loads(line[6:])
            events.append({"time": elapsed, "data": item})
            usage = item.get("usage") or usage
            for choice in item.get("choices", []):
                piece = choice.get("text", "")
                if piece and first is None:
                    first = elapsed
                text += piece
                finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - start
        if not done or first is None or usage is None:
            raise RuntimeError(f"Incomplete streamed response: done={done}, first={first}, usage={usage}")
        tokens = usage["completion_tokens"]
        return {
            "elapsed": elapsed,
            "ttft": first,
            "tpot": (elapsed - first) / max(tokens - 1, 1),
            "text": text,
            "usage": usage,
            "finish_reason": finish,
            "events": events,
        }


def _prompts(model: Path, count: int, length: int) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    questions = [
        "Explain how binary search works and provide Python code.",
        "Describe how photosynthesis converts sunlight into chemical energy.",
        "Write a Python function that computes Fibonacci numbers iteratively.",
        "Explain the difference between TCP and UDP with examples.",
        "Calculate the sum of the integers from 1 to 100 and explain the method.",
        "Describe the role of a compiler and the stages of compilation.",
        "Explain why the sky appears blue during the day.",
        "Write a short guide to unit testing a sorting function.",
    ]
    context = tokenizer.encode(
        "This is background information for a detailed technical explanation. ", add_special_tokens=False
    )
    result: list[list[int]] = []
    for index in range(count):
        suffix = tokenizer.encode(
            "\nQuestion: " + questions[index % len(questions)] + "\nAnswer:", add_special_tokens=False
        )
        if len(suffix) >= length:
            raise ValueError("input length is too short for the benchmark question")
        padding = (context * ((length + len(context) - 1) // len(context)))[: length - len(suffix)]
        result.append(padding + suffix)
    return result


def _token_lookup(model: Path) -> dict[str, list[int]]:
    """Invert LogprobState's rendering without assuming decode is injective.

    OpenAI JSON omits the protobuf token_ids field. Ambiguous renderings must
    fail correctness validation, rather than silently retokenizing the text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    lookup: dict[str, list[int]] = {}
    for piece, token_id in tokenizer.get_vocab().items():
        rendered = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if rendered.endswith("\ufffd"):
            rendered = piece
        lookup.setdefault(rendered, []).append(token_id)
    return lookup


def _token_ids(choice: dict[str, Any], lookup: dict[str, list[int]]) -> list[int]:
    result: list[int] = []
    for token in choice["logprobs"]["tokens"]:
        candidates = lookup.get(token, [])
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot recover an unambiguous token ID for {token!r}: {candidates}")
        result.append(candidates[0])
    return result


def _benchmark_workload(
    url: str,
    mode: str,
    device: str,
    out: Path,
    prompts: list[list[int]],
    output_tokens: int,
    concurrency: int,
    rounds: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Two unmeasured waves warm kernels, request geometry, and the
    # cost estimator. Identical waves are used in every mode.
    for round_index in range(-2, rounds):
        payloads = [
            dict(
                model="graph-validation",
                prompt=prompt,
                temperature=0,
                max_tokens=output_tokens,
                ignore_eos=True,
                stream=True,
                stream_options={"include_usage": True},
            )
            for prompt in prompts
        ]
        _snapshot(url, device, out / f"round-{round_index}-before")
        started_unix = time.time()
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            responses = list(executor.map(lambda payload: _completion(url, payload), payloads))
        elapsed = time.perf_counter() - start
        _snapshot(url, device, out / f"round-{round_index}-after")
        if any(response["usage"]["completion_tokens"] != output_tokens for response in responses):
            raise RuntimeError("Performance requests returned an unexpected token count")
        record = dict(
            round=round_index,
            started_unix=started_unix,
            finished_unix=started_unix + elapsed,
            elapsed=elapsed,
            output_tokens_per_second=sum(response["usage"]["completion_tokens"] for response in responses) / elapsed,
            ttft_median=float(np.median([response["ttft"] for response in responses])),
            ttft_p95=float(np.percentile([response["ttft"] for response in responses], 95)),
            ttft_p99=float(np.percentile([response["ttft"] for response in responses], 99)),
            tpot_median=float(np.median([response["tpot"] for response in responses])),
            tpot_p95=float(np.percentile([response["tpot"] for response in responses], 95)),
            tpot_p99=float(np.percentile([response["tpot"] for response in responses], 99)),
            responses=responses,
        )
        _save(out / f"round-{round_index}.json", record)
        if round_index >= 0:
            records.append({key: value for key, value in record.items() if key != "responses"})
        logger.info(
            "mode=%s round=%s output_tokens_per_second=%.2f",
            mode,
            round_index,
            record["output_tokens_per_second"],
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["target", "fixed", "linear", "graph", "graph_feedback", "graph_shadow"], required=True
    )
    parser.add_argument("--model", type=Path, default=Path("/root/share/models/Qwen3-8B"))
    parser.add_argument("--draft", type=Path, default=Path("/root/share/models/dspark_qwen3_8b_block7"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18740)
    parser.add_argument("--device", default="1")
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workloads", type=Path, help="JSON list of named workloads; reuse one service process")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--trace-plans", action="store_true")
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Emit CUDA profiler range after correctness checks; launch this tool under nsys",
    )
    parser.add_argument("--random-regression", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.max_seqs <= 128 or args.requests < args.concurrency:
        parser.error("require 1 <= concurrency <= max-seqs <= 128 and requests >= concurrency")
    workloads = (
        json.loads(args.workloads.read_text())
        if args.workloads
        else [
            dict(
                name="performance",
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                concurrency=args.concurrency,
                requests=args.requests,
            )
        ]
    )
    names: set[str] = set()
    for case in workloads:
        name = case["name"]
        if not isinstance(name, str) or not name or not all(char.isalnum() or char in "_-" for char in name):
            parser.error("workload names must be nonempty alphanumeric names, optionally with _ or -")
        if name in names:
            parser.error("workload names must be unique")
        names.add(name)
        if not 1 <= case["concurrency"] <= args.max_seqs or case["requests"] < case["concurrency"]:
            parser.error("workload request/concurrency counts exceed server limits")
        if case["input_tokens"] < 32 or case["output_tokens"] < 2:
            parser.error("workloads require at least 32 input and 2 output tokens")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = args.source_root.resolve()
    binary = args.binary.resolve()
    command = [
        str(binary),
        f"--model={args.model.resolve()}",
        "--model_id=graph-validation",
        "--model_impl=python",
        "--host=127.0.0.1",
        f"--port={args.port}",
        f"--master_node_addr=127.0.0.1:{args.port + 1000}",
        "--max_memory_utilization=0.90",
        "--max_cache_size=8589934592",
        "--max_tokens_per_batch=4096",
        f"--max_seqs_per_batch={args.max_seqs}",
        "--enable_graph=true",
        "--python_graph_backend=cudagraphs",
        "--enable_schedule_overlap=false",
        "--enable_task_pipeline=false",
        "--disable_graph_warmup=false",
        "--enable_prefix_cache=false",
        "--enable_in_batch_prefix_cache=false",
        "--random_seed=20261022",
    ]
    if args.mode != "target":
        command += [
            f"--draft_model={args.draft.resolve()}",
            "--speculative_algorithm=DSpark",
            "--num_speculative_tokens=7",
            "--draft_sampling_mode=greedy",
        ]
    if args.mode not in ("target", "fixed"):
        command += ["--enable_adaptive_speculative_decode=true", f"--adaptive_speculative_cost_model={args.mode}"]
    if args.mode == "linear":
        # The legacy controller otherwise uses a constant validate-time
        # fallback. Give the comparison its actual startup-fitted predictor.
        command += ["--enable_profile_step_time=true"]
    if args.trace_plans:
        command += ["--v=1"]
    if args.profile_kernels:
        command += ["--enable_online_profile=true", "--profile_backend=cuda"]
    overrides = dict(
        CUDA_VISIBLE_DEVICES=args.device,
        PYTHONPATH=str(root),
        XLLM_PYTHON_MODEL_PATH=str(root),
        XLLM_PYTHON_EXECUTABLE=sys.executable,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        FLASHINFER_CUDA_ARCH_LIST="8.9",
        FLASHINFER_OPS_PATH=os.environ.get("FLASHINFER_OPS_PATH", "/flashinfer_ops"),
        MAX_JOBS="4",
        # Native and Python FlashInfer must resolve the same TVM FFI runtime.
        LD_LIBRARY_PATH=os.pathsep.join(
            [str(Path(sysconfig.get_path("platlib")) / "tvm_ffi" / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        ),
    )
    env = os.environ.copy()
    env.update(overrides)
    with binary.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _save(out / "launch.json", {"command": command, "env_overrides": overrides, "binary_sha256": digest})
    python_sources = {}
    for path in sorted((root / "xllm" / "python").rglob("*.py")):
        python_sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save(out / "python-source-sha256.json", python_sources)
    prompts = _prompts(args.model, args.requests, args.input_tokens)
    token_lookup = _token_lookup(args.model)
    _save(out / "inputs.json", prompts)
    url = f"http://127.0.0.1:{args.port}"
    records: list[dict[str, Any]] = []
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        _save(out / "pid.json", {"pid": process.pid})
        try:
            deadline = time.monotonic() + 600
            with requests.Session() as session:
                session.trust_env = False
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited during startup: {process.returncode}")
                    try:
                        ready = session.get(url + "/v1/models", timeout=2)
                        if ready.status_code == 200:
                            break
                    except requests.ConnectionError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Service readiness deadline exceeded")
                    time.sleep(1)
            logger.info("Service ready: mode=%s", args.mode)
            correctness: list[dict[str, Any]] = []
            for index, prompt in enumerate(prompts[:3]):
                payload = dict(
                    model="graph-validation",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=64,
                    ignore_eos=True,
                    logprobs=1,
                    skip_special_tokens=False,
                )
                record = _completion(url, payload)
                _save(out / f"correctness-{index}.json", record)
                choice = record["response"]["choices"][0]
                token_ids = _token_ids(choice, token_lookup)
                if len(token_ids) != 64 or record["response"]["usage"]["completion_tokens"] != 64:
                    raise RuntimeError("Incorrect output token count")
                correctness.append({"token_ids": token_ids, "text": choice["text"]})
            _save(out / "correctness.json", correctness)
            if args.random_regression:
                if args.mode != "target":
                    raise ValueError("Seeded sampling regression requires target-only mode")
                payload = dict(
                    model="graph-validation",
                    prompt=prompts[0],
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    seed=1234,
                    max_tokens=64,
                    ignore_eos=True,
                    logprobs=1,
                    skip_special_tokens=False,
                )
                samples = [_completion(url, payload)["response"] for _ in range(2)]
                _save(out / "random-regression.json", samples)
                first_ids = _token_ids(samples[0]["choices"][0], token_lookup)
                second_ids = _token_ids(samples[1]["choices"][0], token_lookup)
                _save(out / "random-regression-token-ids.json", [first_ids, second_ids])
                if first_ids != second_ids:
                    raise RuntimeError("Seeded sampling changed across identical requests")
            # Check stream/regular equivalence and stopping at every small
            # verification width, including anchor-only and bonus boundaries.
            boundaries: list[dict[str, Any]] = []
            for length in (1, 2, 7, 8, 9):
                payload = dict(
                    model="graph-validation",
                    prompt=prompts[0],
                    temperature=0,
                    max_tokens=length,
                    ignore_eos=True,
                    logprobs=1,
                    skip_special_tokens=False,
                )
                regular = _completion(url, payload)["response"]
                choice = regular["choices"][0]
                if _token_ids(choice, token_lookup) != correctness[0]["token_ids"][:length]:
                    raise RuntimeError(f"max_tokens={length} changed the greedy prefix")
                streamed = _completion(url, {**payload, "stream": True, "stream_options": {"include_usage": True}})
                if streamed["text"] != choice["text"] or streamed["usage"]["completion_tokens"] != length:
                    raise RuntimeError(f"Stream mismatch at max_tokens={length}")
                boundaries.append({"length": length, "regular": regular, "streamed": streamed})
            _save(out / "boundaries.json", boundaries)
            stopping: list[dict[str, Any]] = []
            reference_ids = correctness[0]["token_ids"]
            for position in (4, 8, 11):
                stop_id = reference_ids[position]
                expected_length = reference_ids.index(stop_id) + 1
                response = _completion(
                    url,
                    dict(
                        model="graph-validation",
                        prompt=prompts[0],
                        temperature=0,
                        max_tokens=64,
                        ignore_eos=True,
                        logprobs=1,
                        skip_special_tokens=False,
                        stop_token_ids=[stop_id],
                        include_stop_str_in_output=True,
                    ),
                )["response"]
                choice = response["choices"][0]
                if choice["finish_reason"] != "stop" or response["usage"]["completion_tokens"] != expected_length:
                    raise RuntimeError("Stop token failed to truncate speculative output")
                if _token_ids(choice, token_lookup) != reference_ids[:expected_length]:
                    raise RuntimeError("Stop token changed the greedy prefix")
                stopping.append(response)
            _save(out / "stop-tokens.json", stopping)
            eos_id = AutoTokenizer.from_pretrained(args.model, local_files_only=True).eos_token_id
            if eos_id is None:
                raise ValueError("Correctness validation requires a tokenizer EOS token")
            eos_response = _completion(
                url,
                dict(
                    model="graph-validation",
                    prompt=prompts[0],
                    temperature=0,
                    max_tokens=64,
                    ignore_eos=False,
                    allowed_token_ids=[eos_id],
                    logprobs=1,
                    skip_special_tokens=False,
                    include_stop_str_in_output=True,
                ),
            )["response"]
            _save(out / "eos.json", eos_response)
            eos_choice = eos_response["choices"][0]
            if (
                eos_choice["finish_reason"] != "stop"
                or eos_response["usage"]["completion_tokens"] != 1
                or _token_ids(eos_choice, token_lookup) != [eos_id]
            ):
                raise RuntimeError("EOS did not terminate the request at its first generated token")
            mixed_payloads = [
                dict(
                    model="graph-validation",
                    prompt=prompts[index],
                    temperature=0,
                    max_tokens=length,
                    ignore_eos=True,
                    logprobs=1,
                    skip_special_tokens=False,
                )
                for index, length in enumerate((1, 7, 64)[: len(correctness)])
            ]
            with ThreadPoolExecutor(max_workers=min(3, args.max_seqs)) as executor:
                mixed = list(executor.map(lambda payload: _completion(url, payload), mixed_payloads))
            _save(out / "mixed-lengths.json", mixed)
            for index, (payload, response) in enumerate(zip(mixed_payloads, mixed)):
                actual_ids = _token_ids(response["response"]["choices"][0], token_lookup)
                if actual_ids != correctness[index]["token_ids"][: payload["max_tokens"]]:
                    raise RuntimeError("Mixed batch changed a request's greedy token sequence")
            if args.mode.startswith("graph"):
                with requests.Session() as session:
                    session.trust_env = False
                    rejected = session.post(
                        url + "/v1/completions",
                        json=dict(model="graph-validation", prompt=prompts[0], temperature=0.7, max_tokens=8),
                        timeout=30,
                    )
                    _save(out / "unsupported-random.json", {"status": rejected.status_code, "body": rejected.text})
                    if rejected.status_code != 400:
                        raise RuntimeError("Unsupported random request was not rejected before execution")

            if args.reference is not None:
                reference = json.loads(args.reference.read_text())
                if [row["token_ids"] for row in correctness] != [row["token_ids"] for row in reference]:
                    raise RuntimeError("Greedy token IDs differ from the reference")
            if args.profile_kernels:
                with requests.Session() as session:
                    session.trust_env = False
                    started = session.post(url + "/start_profile", timeout=120)
                    started.raise_for_status()
                    _save(out / "profile-start.json", {"status": started.status_code, "body": started.text})
                    for case in workloads:
                        profile_prompts = _prompts(args.model, case["concurrency"], case["input_tokens"])
                        payloads = [
                            dict(
                                model="graph-validation",
                                prompt=prompt,
                                temperature=0,
                                max_tokens=32,
                                ignore_eos=True,
                            )
                            for prompt in profile_prompts
                        ]
                        with ThreadPoolExecutor(max_workers=case["concurrency"]) as executor:
                            profiled = list(executor.map(lambda payload: _completion(url, payload), payloads))
                        _save(out / f"profile-{case['name']}.json", profiled)
                    stopped = session.post(url + "/stop_profile", timeout=120)
                    stopped.raise_for_status()
                    _save(out / "profile-stop.json", {"status": stopped.status_code, "body": stopped.text})
                # CUDA profiling uses cudaProfilerStart/Stop. The surrounding
                # nsys process writes its report after this process exits;
                # successful HTTP calls alone do not validate that artifact.
                _save(out / "profile-range.json", {"backend": "cuda", "nsys_report_validation": "external"})
            if not args.smoke_only:
                for case in workloads:
                    case_out = out / case["name"]
                    case_out.mkdir()
                    case_prompts = _prompts(args.model, case["requests"], case["input_tokens"])
                    _save(case_out / "inputs.json", case_prompts)
                    measurements = _benchmark_workload(
                        url,
                        args.mode,
                        args.device,
                        case_out,
                        case_prompts,
                        case["output_tokens"],
                        case["concurrency"],
                        args.rounds,
                    )
                    records.append({"workload": case, "measurements": measurements})
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            _save(out / "shutdown.json", {"exit_code": process.returncode, "pid": process.pid})
    if process.returncode != 0:
        raise RuntimeError(f"Service shutdown returned {process.returncode}")
    _save(out / "result.json", {"status": "PASS", "mode": args.mode, "workloads": records})


if __name__ == "__main__":
    main()
