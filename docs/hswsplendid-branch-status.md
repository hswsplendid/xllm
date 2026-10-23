# hswsplendid 开发分支说明

同步日期：2026-10-23。完整开发分支：`feat/cuda-graph-aware-verification`。

## 包含的工作

- 原 `/root/xllm` 中 CUDA Graph / dSpark 的 44 个修改与新增文件，原样快照提交为 `a59ccc83`。
- G13 OLMo2 CUDA 实现、FP16 / BF16 精度修复、模型配置、测试、验证工具和双语使用说明。
- Z25 最新嵌套 RoPE 缩放字段与显式空 dtype 修复，并补齐 Python 配置入口的相应校验；G13 来源提交为 `1740790b`。
- J02 `compare_tensor` 相对容差及非有限值策略的最新代码与提交历史。合入时当前基线已经包含相同实现。
- 原官方基线已有的 CUDA 编译、平台保护、共享内存生命周期和计数器修复。

合并保留了 OLMo2 的 Tensor Core 解码精度选项，以及 Graph 工作区容量、因果注意力与成本测量功能，并增加大批量 MHA 的组合回归测试。

历史 CUDA 工作也已推送到 `fix/cuda-layer-synchronizer`（`0bebde95`）和 `bugfix/cuda-unit-test-install`（`8acdb2c5`）。其中旧架构的链接依赖调整没有用来覆盖当前重构后的依赖图；需要历史原代码时应检出对应分支。

## 在另一台服务器获取

```bash
git clone --recurse-submodules --branch feat/cuda-graph-aware-verification \
  https://github.com/hswsplendid/xllm.git
cd xllm
git status --short --branch
```

这是完整源码分支。新服务器需要按仓库 CUDA 构建说明准备依赖并重新构建、安装；本机编译产物、模型权重和服务器上的实验原始日志不属于 Git 源码。

独立的 G13 / Z25 分支继续保留，各自的 PR 不包含本分支的 Graph 实验代码。

## 验证边界

此前 Graph / feedback 实验的严格跨形状输出一致性与性能验收尚未通过；这些策略保持显式选择，保存到远端不表示实验验收完成。本分支用于迁移、继续开发和复现。

## 本次验证记录

代码合并提交：`fccce38e`。以下验证使用该提交的源码；后续提交仅增加本说明文档。

| 项目 | 结果 |
| --- | --- |
| 原工作树快照 | 44 文件均已备份并提交；其中 38 个在合并后字节一致，6 个保留双方修改 |
| G13 文件完整性 | 33 文件全部保留；26 个与 G13 分支字节一致，7 个组合合并 |
| 格式与静态检查 | clang-format 20.1.6、Ruff、codespell、whitespace 检查通过 |
| CUDA 服务、扩展和相关测试目标构建 | 通过，467.78 秒 |
| Python 合并回归 | 508 通过、1 跳过、0 失败；跳过项需要 `torch_npu` |
| C++ 合并回归 | 130 通过、10 跳过、0 失败；跳过项需要当前平台不支持的 MTP / token-bucket warmup 能力 |
| G13 独立 RoPE / dtype 同步回归 | 24 项 C++ 配置测试与 217 项 Python 测试通过 |

Python 回归覆盖 CUDA Graph、FlashInfer、ModelExecutor、OLMo2 模型和验收工具；C++ 回归覆盖 OLMo2 配置、CUDA Graph 执行器、输入打包、warmup、dSpark 控制器和运行成本模型。使用本次新构建的服务 / 扩展记录 SHA256，GPU 为本机 NVIDIA GPU 1，CUDA 架构为 SM89。

本次验收范围是同步及合并相关回归。此前完整权重、双卡及性能实验的结论仍按各自原始记录解释。
