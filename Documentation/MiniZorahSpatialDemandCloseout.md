# MiniZorah 空间 Demand 实验收尾（2026-10-25）

已按要求停止优化并撤回所有未达标的运行时空间候选，恢复原 ordered Demand 路径。保留原有 distributed Demand / refill8、Wave32 Frontier、LDS active-bit 优化和既有 LOD、流送、驻留及上传契约，没有继续寻找新候选或追加性能 A/B。

leaf8 / task32 曾独立确认 Demand 局部下降 6.27%，但伴随 Seed / Frontier 回退，未确认遍历总收益；该局部收益不能从整套实现中独立保留。独立存储 v3 和任务内 forest v4 也未通过验收。因此没有把任何空间候选作为已接受的生产优化留下。

## 撤回与保留

- `GPUDrivenStreamAsset.slang`、`MeshletLOD.cpp/.h`、候选测试注册恢复为本轮前 HEAD 内容。移除空间 buffer、节点/引用/range-map、构建器、环境开关、任务映射和 GPU 参数槽改名；CPU/Slang ABI 恢复原 `blasInstanceReusePadding`。
- Runtime 的两 buffer immutable upload、copy offsets、lease、submission、取消/ready、总字节计量均恢复原契约。移除候选专用 CPU/GPU tests，保留原有测试。
- 独立保留 streaming NvPerf 三范围、实际 dispatch / shader 身份、前后语义/图像快照及失败 drain/cancel。保留通用 metadata hash、task count、GPU 进程计数器有界重试，以及 WG32 编译失败时返回实际 diagnostics。这些是诊断与验证能力，不声称为新的生产性能收益。
- 两个空间 A/B runner 的 `run` 在任何文件、锁或进程操作前立即报错，明确候选已撤回。历史 `verify`、decoder、统计规则和绘图保持；不能把已移除的 B 静默当作 A。
- 全部历史 captures、图表、原始 counter images、离线解码与源码归档保留。四份历史报告只增加收尾状态说明；没有重写实验结果。未跟踪的 `ucode.elf` 等无关内容未删除。

## 恢复验证

1. 普通 `build-release` sample、`build-spatial-nvperf` sample、`build-pass-stages-nrd` RHI tests 均沿用现有配置重建通过。
2. 六项聚焦测试全部通过，0 skipped：RT demand、Frontier32、Frontier LDS、initial loading、reference cut/metadata、ordered tile hierarchy。前三项共 552 帧 GPU oracle（104 + 160 + 288），上传测试另覆盖 Device → Host → Device 生命周期。
3. 只运行一次恢复后 MiniZorah 正确性捕获：相同场景/相机、1920×1080、相同配置、seeds 0/1。六类图像共 12 份均有限，逐位等于已封存的 ordered A 参考，maxAbs / RMSE 为 0；原始 topology/primitive hashes、任务和 group/node 计数、14-buffer 语义也相同。已打开同一显示变换下的预览检查。
4. 十项纯 CPU 工具退役检查通过：立即拒绝新采集且无副作用；AST 核对其余历史解析/验证/统计逻辑完全不变。当前 verifier 重验 v4 普通和 counter v2 完整证据通过；没有修改历史判定。
5. Shader module audit 通过：375 files / 263 modules / 112 implementation units。diff whitespace 检查通过；运行 Shader 与原层级构建器的 Git 内容与 HEAD 完全相同。

本次只验证撤回后的正确性，没有新性能结论，也没有重复硬件计数器采集。NvPerf 构建同步重建，指定 EXE 仍为 Medium 完整性。没有新提交、发布或系统配置修改。

仓库已有 `.git/index.lock` 阻止了只读内容恢复后的 Git stat-cache 刷新；没有删除该锁或改动暂存区。部分 `git status` 条目可能仍使用旧状态，以上恢复文件已通过内容 hash 与 `git diff --exit-code` 核对。

## 记录

- [收尾前源码备份与精确变更](../build/spatial-demand-closeout-20261025/SourceChanges.json)、[完整撤回前 diff](../build/spatial-demand-closeout-20261025/BeforeDiff.patch)、[最终状态](../build/spatial-demand-closeout-20261025/FinalState.json)。
- [普通构建](../build/spatial-demand-closeout-20261025/BuildOrdinary.log)、[NvPerf 构建](../build/spatial-demand-closeout-20261025/BuildNvPerf.log)、[RHI 构建](../build/spatial-demand-closeout-20261025/BuildRHI.log)、[六项测试](../build/spatial-demand-closeout-20261025/RHI.log)。
- [恢复画面与语义验证](../build/spatial-demand-closeout-20261025/RestorationQuality.json)、[验证脚本](../build/spatial-demand-closeout-20261025/ValidateRestoration.py)、[工具退役检查](../build/spatial-demand-closeout-20261025/RetirementChecks.log)。
- [历史普通证据复核](../build/spatial-demand-closeout-20261025/VerifyV4.json)、[历史 counter 证据复核](../build/spatial-demand-closeout-20261025/VerifyCounterV2.json)。

![Restored ordered output equals archived A](../build/spatial-demand-closeout-20261025/RestorationPreview.png)
