---
title: "OLMo2 CUDA validation preview"
sidebar:
  order: 1
---

<!-- Copyright 2026 The xLLM Authors. -->

This preview covers `allenai/OLMo-2-0425-1B-Instruct`, revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`, with the Python executor,
FlashInfer paged attention, FP16, eager execution and TP=1/2. It depends on
the OLMo2 C++ configuration loader (Z25, PR #2518). It is not a completed
platform support announcement. NPU, quantization, graph execution and
DP/CP/PP greater than one have not been accepted for this model.
This contribution targets NVIDIA CUDA. Ascend adaptation is a separate task.

The checkpoint has 16 layers, hidden size 2048, 16 MHA heads of size 128,
intermediate size 8192, vocabulary size 100352 and a 4096-token context.
Q/K RMSNorm spans the complete projection before head reshaping and RoPE.
TP=2 sums separate FP32 Q/K squared statistics across ranks, then applies
the sharded norm weights. Attention/MLP post-norm outputs are added to an
FP32 residual stream; projection weights and KV cache retain FP16. The
embedding and LM head are independent weights.

## Prepare a separate FP16 configuration

Keep the downloaded checkpoint unchanged. Set these paths to the fixed
checkpoint and a new, nonexistent directory; this creates config-only
overrides and symlinks the original weights and tokenizer files.

```bash
export OLMO2_SOURCE=/path/to/OLMo-2-0425-1B-Instruct
export OLMO2_FP16=/path/to/OLMo-2-0425-1B-Instruct-fp16
python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["OLMO2_SOURCE"]).resolve()
target = Path(os.environ["OLMO2_FP16"])
target.mkdir()
for item in source.iterdir():
    if item.name != "config.json":
        (target / item.name).symlink_to(item)
config = json.loads((source / "config.json").read_text())
config.update(dtype="float16", torch_dtype="float16")
(target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

The source weights are BF16. Casting to FP16 changes some very small values;
this is an explicit compute configuration, not bitwise-preserved BF16 execution.
Use the checkpoint's serialized `tokenizer.json` for token-ID references.
Transformers 5.12.1 `AutoTokenizer` reconstructs a GPT2 tokenizer with a
different newline pre-tokenizer. HF template rendering followed by
`tokenizers.Tokenizer.from_file(...).encode(..., add_special_tokens=False)`
preserves the checkpoint tokenization used by the service.

## Reproduce service acceptance

Build the branch's CUDA executable and matching Python extension first.
Run from the repository root with its Python dependencies installed. Set
`XLLM_BINARY` to that executable; preserve the runtime library paths required
by your CUDA build. The local validation used Python 3.12.3, Torch 2.10.0+cu130,
Transformers 5.12.1, FlashInfer 0.6.14 and TVM-FFI 0.1.13.

```bash
export XLLM_BINARY=/path/to/built/xllm
python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-tp1

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 2 --devices 0,1 --port 18000 --output /tmp/olmo2-tp2
```

Use new output directories for each run. The tool starts and stops its own
servers, saves commands, requests and SSE events, and fails on mismatches.
TP=2 uses two local server processes with `--nnodes=2 --node_rank=0/1` and a
shared rendezvous address. Exposing two GPUs to one server process alone
does not create TP=2 in this engine revision. It checks the actual loaded
TP topology and FP16 dtype, template token count, stream equivalence,
EOS/ignore_eos, explicit stop tokens/strings, length, invalid inputs and
mixed-length concurrent isolation. These are service checks; they do not
replace numerical comparison of logits.
Every stream must report a finish reason and end with `[DONE]`; requested
usage must also be present. Truncated chat streams and in-band errors fail
acceptance even when all expected text has already arrived.

The tool preserves `FLASHINFER_CUDA_ARCH_LIST` when set by the caller;
otherwise FlashInfer detects the GPU architecture. It does not force the
RTX 4090's architecture on other NVIDIA devices. The effective override is
recorded in `launch.json`; validation on other GPU models is still required.

For this server's allocator/FFI environment, the successful runs additionally
set `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4` and placed the
matching Python TVM-FFI `lib` directory first in `LD_LIBRARY_PATH`.

## Tests and measurements

```bash
python -m pytest tests/python/test_olmo2.py tests/python/test_registry.py \
  tests/python/test_flashinfer_attention.py tests/python/test_validate_olmo2_cuda.py -q

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-benchmark \
  --benchmark --build-label "record your compiler flags here"
```

The optional benchmark uses prompt lengths 64/512/2048, 32 output tokens,
concurrency 1/4, one warmup and three measured repetitions per group. Both
prefix caches and graphs are off; per-rank KV cache is capped at 1 GiB.
Raw SSE, memory snapshots and per-request timings are retained. TTFT is the
first nonempty text arrival; effective TPOT is `(request_end - TTFT)/(N-1)`
using reported completion tokens, including invisible special tokens. It is
a client metric, not an individual kernel or token-arrival measurement.
Record other GPU users and cache pressure before interpreting throughput.

The current local native build uses `XLLM_DEV_MODE=ON` (O1). Its measurements
are development baselines, not release performance or a speedup claim.


The local development run used two RTX 4090 devices reporting 48537 MiB each.
The following are aggregate output throughput medians and ranges across three
repetitions; the raw requests are retained with the contribution evidence.
These are separate TP baselines, not an optimization comparison.

| TP | Prompt tokens | Concurrency | Output tokens/s, median (range) |
| --- | --- | --- | --- |
| 1 | 64 | 1 | 24.18 (22.93–24.38) |
| 1 | 64 | 4 | 90.64 (84.99–93.50) |
| 1 | 512 | 1 | 21.22 (21.12–22.27) |
| 1 | 512 | 4 | 59.63 (56.75–67.72) |
| 1 | 2048 | 1 | 17.06 (14.88–17.46) |
| 1 | 2048 | 4 | 34.03 (32.11–34.64) |
| 2 | 64 | 1 | 19.30 (19.01–19.56) |
| 2 | 64 | 4 | 67.22 (63.69–68.49) |
| 2 | 512 | 1 | 16.96 (16.80–17.43) |
| 2 | 512 | 4 | 51.36 (49.65–52.22) |
| 2 | 2048 | 1 | 14.76 (13.36–15.26) |
| 2 | 2048 | 4 | 33.96 (32.81–34.73) |

TP1 has 8192 cache slots in this configuration. The 2048 × 4 case
needs 8320 slots including output and therefore includes cache pressure.
TP2 has a different sharded cache capacity. These rows cannot establish
a release speedup or isolate collective overhead. Profiling separately
covered prefill512 and 16 same-history decode steps: 272 Q/K norm calls
and 544 separate Q/K sums. Numerous small Torch operations and conversions
make CPU dispatch a follow-up investigation; no kernel optimization or
post-optimization speedup is claimed.

## Numerical acceptance still under review

FP16 passed the fixed original B06/B07/B08 matrices, including real paged
decode, mixed batches, total length 4096 and real NCCL TP=2. This does not
establish BF16 correctness: the BF16 run still fails 1 B06 node, 19 B07
comparisons and 13 B08 logits comparisons per rank. Additional cross-dtype
diagnostics also have failures, retained in the review evidence. FP32
residual accumulation changes low-precision rounding and needs an agreed
precision contract. The supported-model matrix remains unchanged pending
that review, independent reproduction and platform acceptance.

## Reproduce full-checkpoint numerical acceptance

`tools/validate_olmo2_numerics.py` carries the original B06/B07 inputs and
unchanged v2 budgets in `tests/models/fixtures/olmo2/numerics.json`. It uses
the original BF16 checkpoint/config at revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`; SHA256 checks reject a different
checkpoint. Pass `--dtype float16` explicitly to convert weights in memory
(normal FP16 rounding, including subnormal values); the source checkpoint is never rewritten. HF reference
generation requires Transformers 5.12.1 and compares against unmodified eager
BF16. FP32 references are diagnostics. No budget is fitted to xLLM output.

```bash
# Run from a built checkout with its matching xllm_export/runtime environment.
MODEL=/path/to/OLMo-2-0425-1B-Instruct
OUT=/path/to/new-validation
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py reference \
  --model "$MODEL" --output "$OUT/reference"
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py validate \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/fp16-tp1" \
  --dtype float16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/validate_olmo2_numerics.py validate --tp 2 --dtype float16 \
  --model "$MODEL" --reference "$OUT/reference" --baseline "$OUT/fp16-tp1" \
  --output "$OUT/fp16-tp2"
# This is a diagnostic control; a threshold failure deliberately exits 1.
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py hf-control \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/hf-control"
```

Use new output directories for every run. Keep the generated tensor files and
manifests outside Git. TP2 requires the same-dtype TP1 evidence and verifies
checkpoint, runtime source and tensor-file hashes. Every loaded weight/shard is checked exactly against the explicit dtype
conversion of its checkpoint tensor; conversion error is recorded separately.
Both ranks use real xLLM
NCCL collectives and paged FlashInfer execution, compare full vocabulary logits
against HF/TP1, and check rank equality. Cache starts with NaNs; nonmonotonic
pages, 127/128/129 boundaries, chunked prefill, mixed batches and total length
4096 are included. TP1 runs 591 comparisons; TP2 runs 1064 per rank, extending
the original B08 length coverage. Missing data, non-finite values, shape errors
and numeric failures cannot produce a successful exit code. Reports preserve
all failures; TP2 shares a failing status across ranks. Runtime exceptions
propagate and torchrun terminates the worker group.

Repeat both validation commands with `--dtype bfloat16` and distinct output
paths to retain the known failures. The `hf-control` command reproduces the
official eager BF16 cache/batch variation against its own full-prefill reference;
it does not waive xLLM failures or change the frozen criteria. Actual automated
reproduction is distinct from review/reproduction by another contributor.
