# #61 RenderGraph 计时文档对账（2026-11-14）

## 范围

目标：[GitHub #61](https://github.com/af8a2a/metallic/issues/61) 的三文档一致性子项。
源码核查基线 `6fed3680a0ba7cac02894446dcaa033faedc3f6d`，本轮仅修改文档；
不修改 timestamp 预算、队列/生命周期、Tracy 发布逻辑或任何测试断言。
不是整票 GPU 验收，不关闭 #61，也不改变 #10 的 GPU profiling 余额。

## 当前实现与修正

| 契约 | 源码依据 | 文档修正 |
| --- | --- | --- |
| native 计时支持 Graphics/Compute/Copy | [queue index、pool、interval](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1764-L1826) | 不再称 self-submission 没有 GPU 计时 |
| completion 与 query availability 后才回填 frame/node/section | [resolveGpuTimings](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1828-L1869) | unavailable 不等于 measured zero；CPU recording 与 GPU interval 分开 |
| 三个 timing slots；无 frame completion、graphics pool 或可复用槽位则无 sample | [beginGpuTiming](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1903-L1949) | raw external command buffer 不能仅凭 query availability 收集 GPU 结果 |
| query/section 容量有限，记录 overflow | [interval budget](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1802-L1814)、[section capacity](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L2434-L2476) | 区分设备不支持、槽忙与 `profilingOverflow`；不把不足变成 rendering fail |
| self-submit 的 graphics frame envelope 在 epilogue join 后结束 | [beginSegment](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L3786-L3825)、[timing-only prologue](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4066-L4079)、[epilogue](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4311-L4334) | 不累加各队列时间，不宣称这是校准后的多队列时间轴；external envelope 从 subsystem prologue 后开始并包含 epilogue |
| Tracy 只接受 frame timer 与所有 node timer 均为 graphics | [publication predicate](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1872-L1880) | 混合 node queue 时整帧不发布，native 仍可有结果；section 不参与谓词，也不导出为 GPU zone |
| 已有跨队列资源依赖与 opt-in aliasing | [多队列生命周期](../RenderFrameLifetime.md#self-submitting-graphs-and-multiple-queues)、[compile options](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.h#L34-L43) | 架构约束不再声称跨队列边、transient aliasing 尚不存在；保留 ownership transfer、审计与导出/持久资源边界 |

三份同步文档：
[RenderFrameLifetime](../RenderFrameLifetime.md#gpu-timing-and-completion)、
[TracyGpuProfiling](../TracyGpuProfiling.md#timing-coverage-and-limitations)、
[ProjectArchitecture](../ProjectArchitecture.md#15-当前架构约束)。

## 验收边界

以下是**已有测试源码入口**，不是本轮新增测试或通过结果：

- [frame_cross_queue_graph_dependencies](../../tests/rhi/FrameContextTests.cpp#L1587-L1664)：六轮 mixed-queue self-submit、completion 收集、native frame/node/section availability 检查；源码没有对每个 node 强制 `gpuMilliseconds > 0`。
- [gpu_profiling_scope_budget](../../tests/rhi/GPUProfilingTests.cpp#L216-L257)：self-submit 交替触发 overflow、检查 section 上限及槽位复用。
- [gpu_profiling_external_requires_completion](../../tests/rhi/GPUProfilingTests.cpp#L328-L415)：区分 self-submit 与无 completion 的 raw external 记录。

文档子项的 oracle 是源码调用链、三份叙述一致、本地目标/锚点/行范围有效及独立审查。
静态验证：`python -B build/issue61-docs-20261114/ValidateDocLinks.py` 检查四份变更文档的
44 个 Markdown 链接（含本地 heading / source line range），并确认缺失目标、缺失锚点、
越界行号三个反例被拒绝，0 errors；`git diff --check` 通过。检查脚本仅为本地 build 输出，
不纳入源码或声称为 renderer 回归。独立只读审查 `d2-audit-docs` 对四份文档结论为
APPROVE，无阻断意见；复核了计时 envelope、completion、Tracy 谓词、aliasing 和测试入口。

没有为纯文档修正重新构建 renderer 或运行 GPU/editor；不存在本轮新增的 GPU passed/skipped/failed 计数。
#61 的跨队列正毫秒/section 区间、实际 overflow 回归和 editor Statistics 非零列等验收，
仍需在兼容构建及指定设备上执行并保留结果。已有测试存在不等于已经通过，CPU 静态核查不能代替这些证据。
