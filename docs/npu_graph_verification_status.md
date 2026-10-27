<!-- Copyright 2026 The xLLM Authors. -->

# NPU graph verification development snapshot

Branch: `feat/npu-graph-cost-aware-verification`.
Base: upstream `8549560f121bcb5793341542633d0b05121aeb50`.

This branch implements DeepInfer-inspired exact startup graph costs and adapts
Qwen3.5-family MTP verification to the NPU task pipeline. It preserves
request-owned recurrent checkpoints, supports ragged verification, and updates
FIA graph tasks using conservative host bounds plus exact device length masks.

Validation completed:

- Native service and Python extension builds on Ascend 910B2.
- 101 cost-model tests, 12 input tests, 13 context tests, and 100 speculative
  pipeline tests.
- 414 Python tests, including real NPU graph replay and Qwen hybrid attention.
- Qwen3.8-27B TP2 loading, startup graph calibration, and functional requests
  with concurrency 1/4, mixed lengths, and request-slot reuse.

Acceptance remains incomplete. Concurrent real-model output differences against
fixed-width MTP are under investigation. One single-request divergence was
traced to a BF16 logit tie; this does not explain all concurrent differences.
Targeted optimization performance comparisons remain pending. This is a backup
snapshot, not an upstream-ready change or a verified performance claim.

After updating main, rebuild but skip performance tests bundled with the build.
Changes to the optimization still require their own functional and performance
acceptance tests.
