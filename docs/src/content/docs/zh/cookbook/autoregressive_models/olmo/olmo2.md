---
title: "OLMo2 CUDA 验证预览"
sidebar:
  order: 1
---

<!-- Copyright 2026 The xLLM Authors. -->

本预览针对 `allenai/OLMo-2-0425-1B-Instruct`，固定 revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`，使用 Python executor、
FlashInfer 分页 attention、FP16、eager、TP=1/2。依赖 OLMo2 C++ 配置
加载前置 Z25（PR #2518）。当前不是完整平台支持声明；NPU、量化、图执行
以及大于1的 DP/CP/PP 尚未完成本模型验收。

模型为16层、hidden2048、16个128维 MHA head、intermediate8192、vocab100352、
context4096。Q/K RMSNorm 在 reshape head 和 RoPE 前覆盖完整投影维度；
TP=2 跨 rank 分别归约 FP32 Q/K 平方和，再应用分片 norm 权重。
attention/MLP 的 post-norm 输出加入 FP32 残差流，投影权重和 KV cache
保持 FP16；embedding 与 LM head 使用独立权重。

## 创建独立 FP16 配置

保持原始 checkpoint 不变。源路径指向固定下载目录，目标路径必须不存在；
以下只创建独立 config，并符号链接原权重和 tokenizer。

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

原权重为 BF16，转为 FP16 时部分极小值会改变。这是显式计算配置，不是
BF16 逐位不变的执行。token ID 参考使用 checkpoint 自带的 `tokenizer.json`；
Transformers5.12.1 的 `AutoTokenizer` 会重建 GPT2 tokenizer，换行预切分规则
与原文件不同。先用 HF 渲染模板字符串，再用
`tokenizers.Tokenizer.from_file(...).encode(..., add_special_tokens=False)`
编码，可保持与服务相同的原始 tokenization。

## 复现正式服务验收

先构建本分支 CUDA executable 和匹配 Python extension，安装 Python 依赖，
在仓库根目录运行。设置 `XLLM_BINARY` 为实际产物，并保留该 CUDA 构建所需
的运行库路径。本机验证环境为 Python3.12.3、Torch2.10.0+cu130、
Transformers5.12.1、FlashInfer0.6.14、TVM-FFI0.1.13。

```bash
export XLLM_BINARY=/path/to/built/xllm
python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-tp1

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 2 --devices 0,1 --port 18000 --output /tmp/olmo2-tp2
```

每次使用新的输出目录。工具只启动和停止自己的服务，保存命令、请求与 SSE，
遇到不一致即失败。TP=2 使用两个本机进程，分别设置
`--nnodes=2 --node_rank=0/1` 并共享 rendezvous 地址；当前引擎只让一个进程
可见两卡不会自动形成 TP=2。工具核对实际加载的 TP/dtype、模板 token 数、
流式一致性、EOS/ignore_eos、stop token/string、长度、错误输入及混合并发隔离。
这些属于服务验收，不能代替 logits 数值对照。

本机成功命令额外设置
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4`，并在
`LD_LIBRARY_PATH` 首位放置配套 Python TVM-FFI 的 `lib` 目录，以匹配本机分配器和 FFI 环境。

## 测试与性能测量

```bash
python -m pytest tests/python/test_olmo2.py tests/python/test_registry.py \
  tests/python/test_flashinfer_attention.py -q

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-benchmark \
  --benchmark --build-label "填写实际编译选项"
```

可选性能矩阵为 prompt64/512/2048、output32、并发1/4，每组预热1次后正式
重复3次。关闭两种 prefix cache 和图，每rank KV cache上限1GiB，保留原始
SSE、显存快照及逐请求时间。TTFT 为首个非空文本到达时间；有效 TPOT 为
`(请求结束时间−TTFT)/(N−1)`，N取实际completion tokens，包含不可见special
tokens。这是客户端指标，不是逐token到达间隔或kernel时间。分析吞吐时需核对
其他GPU使用者及cache压力。

当前本机 native 为 `XLLM_DEV_MODE=ON`（O1）开发构建；其测量只属于开发基线，
不代表发布性能，也没有据此声明加速。


本机开发基线使用两张各报告48537MiB的RTX4090。下表为三次重复的总输出吞吐
中位数及范围，原始请求保留在贡献证据中。两种TP配置分别记录，不视为优化对照。

| TP | Prompt tokens | 并发 | 输出tokens/s，中位数（范围） |
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

此配置TP1只有8192个cache slots，2048×4加输出需8320个，因此包含cache压力；
TP2分片后的cache容量也不同。不能据此声明发布加速或量化collective开销。
独立profiling覆盖prefill512及16步同历史decode，产生272次Q/K norm和544次
独立Q/K求和。多个小Torch算子及转换提示后续可调查CPU发射开销；本轮未新增
kernel优化，也没有声明优化后加速。

## 数值验收仍有待评审

FP16通过固定原 B06/B07/B08 矩阵，包含真实分页decode、混合batch、总长度
4096及真实NCCL TP=2。BF16结论仍为失败：B06有1节点、B07有19比较、B08
每rank有13个logits比较超限；新增跨dtype诊断也有失败，均保留在审核证据中。
FP32残差累加改变低精度舍入，需约定精度契约。支持矩阵暂不修改，待精度评审、
他人复现及对应平台验收完成后再声明支持。
