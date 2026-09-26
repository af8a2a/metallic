# 并行录制与提交：MiniZorah 测量与决策（2026-09-26）

本轮保留当前提交 coordinator，完成按需诊断、可重复对照和批完成等待修复。独立提交线程、跨调用异步录制、单 rendering scope 分片均暂不引入；当前负载没有给出足够收益证据。此前的 batch seal、接收回执、frame completion 与 GPU fork/join 契约保持不变。

## 环境与负载

- Ryzen 7 7800X3D，8 核 / 16 线程；RTX 5070 Ti 16 GB，驱动 616.92；Windows、MSVC Release `/O2`、Ninja。
- 基于 `b7b9ac3a9a587abed3de152cb3789a39c83ac5ac`，使用独立 `build-scheduling-release` 配置：`cmake --preset metallic-release -B build-scheduling-release -DMETALLIC_BUILD_TESTS=ON`。不改变已有构建树或默认 SDK 配置；本树未启用 NRD。
- 按要求使用 MiniZorah，没有启动 ZorahFull。其他程序保持运行；MiniZorah 三次测量前后 GPU 利用率为 18%–76%，显存占用约 9.3–10.0 GB。这里只采集运行前后快照，没有连续 GPU 调度轨迹。
- MiniZorah 使用 `gpu-driven-minizorah-vbuffer`，1920×1080、LOD 1.5 px、CLAS 开启；geometry 预算 1 GiB、CLAS 512 MiB，保持既有上传预算和完成门控。实际配置的 software raster async 为关闭，不把传入 compute queue 等同于实际异步分支。
- 每次新进程运行 1500 帧：0–299 为预热并从统计剔除；300–599 固定视角；600–899 为原路线前 5 秒移动；900–1499 保持第 899 帧视角。保留原来的帧步进，未做实时配速。每次有效样本 1200 帧。
- 缓存条件为已有 cooked asset、OS 文件缓存、持久 shader / PSO 缓存；GPU residency 每进程重建。运行后进程本地 GPU 分配约 2.88–3.01 GiB，是结束快照而非峰值。
- 场景为离屏 VisibilityBuffer + MaterialResolve，包含 streaming / CLAS，未覆盖完整 realtime / DLSS / UI / present。计时帧没有输出读回；结束后仍检查真实相机、LOD cut、容量、加载失败、请求溢出、CLAS 收敛和统计层级。

外部 GPU 工作明显改变 frame drain。因此所有 CPU execute 绝对值都带有竞争条件，不能由本报告推导生产帧率或 GPU 空闲时间。

## 诊断接口和统计边界

`RenderGraphSubmitDesc::schedulingDiagnostics` 或进程启动前的 `METALLIC_RENDER_SCHEDULING_DIAGNOSTICS=1` 开启 `executionStats().scheduling`。环境开关在进程内缓存；默认关闭，不读取诊断时钟。worker 使用独立 TLS capture 和本地计数，汇合后合并，不在录制热路径增加共享计数锁。

| 字段 | 测量范围 |
| --- | --- |
| `executeNs` / `frameWaitNs` | self-submit execute 入口到结果发布；其中的 prior-frame drain 和 frame-slot wait |
| `prepareNs` | coordinator 的 node 准备、`prepareExecution`，包含纯准备任务的汇合等待 |
| `scenePrepareNs` | scene begin 与 traversal；不代表全部 streaming CPU 时间 |
| `serialExecuteNs` / `workerExecuteNs` | 串行 pass execute / CPU batch pass execute 之和，后者也包括内联 batch |
| `waitNs` | recording wave 的条件等待与 TaskSystem join |
| `sealNs` / `submitNs` / `nativeSubmitNs` | batch seal / tracker 完整接收路径 / 其中的 `vkQueueSubmit2` 调用 |
| `firstSubmitNs` / `firstPassSubmitNs` | 从 execute 入口到成功接收批次的提交调用开始；后者排除仅含计时 prologue 的批次 |
| `recordingEndNs` | 最后 epilogue 封口完成，尚未结束整个提交窗口 |
| `readyDelayNs` / `maxReadyDelayNs` | ready 到提交调用开始；包含合法图顺序等待及 Joined 模式的整帧等待，不等同于调度器空转 |
| `renderingNs` / `maxRenderingNs` | native beginRendering 返回至 endRendering 返回之间的 CPU 录制时间 / 每帧最大值 |
| `drawCalls` / `maxScopeDrawCalls` | native RHI draw 调用数 / 每帧单 scope 最多调用数；一个 indirect draw 只计一次 |

时间域嵌套且包含并行求和，不能相加得出总耗时。`submitNs` 包含验证、保留包移交和 coordinator 回调，不能假设全部可移到另一线程；`nativeSubmitNs` 不覆盖脱离当前 capture 的后台上传或 SDK 内部提交。rendering scope 指 Vulkan rendering 区间，不是整 pass 或 GPU 执行时长。

`Tools/AnalyzeSchedulingDiagnostics.py` 输出 mean / P50 / P95 / P99 / max，保留每次运行、路线阶段和采样数量；拒绝失败场景、丢帧和时间域不变量错误。旧采集缺少 drain 原因字段时输出空列表，表示未采集而非没有阻塞。

## 已修复：批完成通知丢失后的定时等待

原实现扫描完成 batch 并提交后，无条件执行 `condition_variable::wait_for(1ms)`。如果 worker 在扫描/提交期间已经通知，coordinator 仍会休眠；Windows 定时等待还可能超过请求的 1 ms。线程更多、批次更小会增加这种额外等待的机会。

修复将完成计数和 done 发布放在同一 mutex 下，扫描前保存完成计数，等待谓词检查计数变化或 TaskGraph 完成。全部 batch 完成后直接 join，避免依赖最后一次任务记账再发送通知。1 ms 有界检查仍用于 TaskSystem 取消未启动任务的退出路径；失败、异常和已接收前缀仍总是汇合后清理。

小型基准是 32×32 clear-red + copy 链，16 / 64 pass。每配置预热 30 帧、测 120 帧，重复 3 轮并轮换/反转配置顺序；每配置结束后等待 GPU 并逐像素校验。实际执行顺序为改动前 A、改动后 B、B 重复、A 重复。表内范围是两次独立进程的均值范围，不是置信区间。

| 配置（诊断开启） | 修复前 wait 均值 ms | 修复后 wait 均值 ms | 修复前 execute 扣除 frame wait ms | 修复后 execute 扣除 frame wait ms |
| --- | ---: | ---: | ---: | ---: |
| 16 pass，4 worker，workload 4，Pipelined | 1.011–1.078 | 0.026–0.030 | 1.303–1.379 | 0.285–0.309 |
| 64 pass，4 worker，workload 4，Pipelined | 4.380–4.690 | 0.113–0.343 | 5.578–5.818 | 1.158–1.783 |
| 64 pass，4 worker，workload 16，Pipelined | 0.995–1.117 | 0.069–0.077 | 1.843–2.013 | 0.851–0.978 |

64-pass、workload 4 的完整 execute wall 均值从 5.598–5.889 ms 到 1.963–2.380 ms，P95 从 8.844–8.866 ms 到 4.931–5.599 ms；这些是小型基准的 CPU 调用时间，不是 MiniZorah 的帧率提升。扣除 frame wait 也不能消除 CPU 调度竞争。

修复后 64-pass workload 16 比 workload 4 有更低的 native-submit 总成本（约 0.108–0.111 ms 对 0.256–0.297 ms）。1 worker 适配路径和并行路径的 native batch 数也不同，例如 64 pass 的单 worker 为 66 次提交，4 worker / workload 4 为 18 次；不能把这组比较归因于纯 CPU 核数扩展。没有据此改变所有生产场景的默认批次权重。

基准同时测诊断关闭的相同 Pipelined 配置：64 pass 的 wall 均值从 5.719–6.017 ms 到 1.849–1.969 ms，修复收益并不依赖诊断。诊断开启/关闭的 wall 差值受帧等待和运行顺序噪声影响，本次不能可靠给出诊断的净开销；它仍为默认关闭的测量工具。

## MiniZorah 的结果与三项决策

以下为两次 4-worker 运行的均值范围，每次剔除前 300 帧：

| CPU 指标 | 均值范围 ms |
| --- | ---: |
| execute | 5.620–7.085 |
| frame wait / prior-frame drain | 3.463–5.504 |
| prepare | 0.411–0.640 |
| scene begin + traversal | 0.513–0.704 |
| serial pass execute | 0.198–0.252 |
| tracker 提交总时间 | 0.073–0.091 |
| 原生提交总时间 | 0.067–0.083 |
| 每帧最大 rendering scope | 0.0040–0.0053 |
| 最后录制结束至 execute 返回 | 0.075–0.093 |

每帧 4 次 native submit、6 个 rendering scope、4 次 native draw，单 scope 最多 1 次 draw。每帧原生提交总耗时 P95 为 0.081–0.131 ms，每帧最大 scope 的 P95 为 0.0055–0.0075 ms。单个 indirect draw 内的 GPU cluster 数不构成 CPU 录制分片机会。

所有场景样本实际为 Joined，阻止提前提交的 pass 是 `GPUDriven`、`MaterialResolve`；CPU recording task 数为 0，纯准备任务数为 2。最后一次采集还确认每帧 `drainReasonMask=1`，frame-overlap blocker 为 `MaterialResolve`。这解释了为何修复小型流水负载的条件等待不会直接加速当前 MiniZorah 路径。

| 方案 | 本轮决定 | 重新评估所需证据与前提 |
| --- | --- | --- |
| 独立提交线程 | 暂不增加 | 当前 native submit 每帧约 0.07–0.08 ms，转线程的理想可隐藏量也很小。需要证明持续的 coordinator 提交瓶颈，并先拆清 callback 所属线程、队列顺序与 frame ledger 唯一 owner；仅增加线程不能让未审计的 scene pass 提前提交。 |
| 跨调用异步录制 | 保持调用内汇合 | 主要等待受 MaterialResolve 的 frame-overlap 契约约束。当前编辑器 execute 返回后立即读取 stats/history 并 transition 输出，尚未测得可隐藏录制的独立调用方工作。需要 frame job 持有稳定的 scene/view/history/输出快照、取消/重编译等待契约，以及调用方实际重叠区间；返回一个 future 本身不构成收益。 |
| 单 scope 分片 | 暂不引入 secondary command buffer 或并行 scope 编码 | 当前每 scope 最多一个 native draw，CPU 区间只有几微秒。需要出现包含大量独立 CPU draw 编码、且 scope 录制占据关键路径的负载，再比较 secondary buffer 的 inheritance、动态状态和合并成本。 |

额外的 1-worker 同路线对照通过，准备任务数为 0，prepare 均值 0.347 ms，4-worker 两次为 0.411 / 0.640 ms。纯相机和资源包准备任务本身仅约十几微秒总和；但 prepare 还包含 GPUScene、host 写入和调度，且外部负载变化，不能把整个差值归因于任务调度，也不足以修改通用默认策略。后续应优先细分这个 coordinator 阶段与审核 MaterialResolve 的 overlap 契约，而非继续增加线程。

## 验证、失败记录与复现

Release 构建 `Metallic`、`MetallicRhiTests`、`MetallicTaskTests` 成功。21 项相关 RHI 回归全部通过，无跳过、无 Vulkan 验证层错误；覆盖 pipeline GPU 前缀进度、接收失败/异常恢复、资源包寿命、参数追加与串行/并行准备像素一致性。检查了 `scheduling-regression/VisibilityPreparedMaterial.png` 的 Bunny 输出。TaskSystem CTest 与编辑器一次 acquire / submit / present smoke 通过。

三次 1500 帧 MiniZorah 性能运行全部通过，另有 600 帧静态视角 Vulkan 验证运行通过。尚未进行完整 realtime / NRD / DLSS、长时间路线、无外部负载条件下的端到端 A/B 或真实 non-coherent 内存验证。

保留而不混入有效结果的尝试：

- 最初使用了错误的关闭验证参数，长基准仍带验证；在启动后续基准时发现重叠，终止了本任务的旧测试进程。`scheduling-before.log` 和 `scheduling-before-release` 均排除，随后使用正确的 `--rhi-no-validation` 重新采集 `scheduling-before-clean`。
- MiniZorah 旧断言要求 Stream Begin 至少 7 个直接 CPU 子项，但维护阶段已移到 Residency completion 下，无上传的 CLAS 帧只有 6 项。诊断关闭时也复现；改成检查稳定的必需阶段名称，保留父子关系、非负时长和不重复计时检查，并在失败报告中保存原始 sections。
- `minizorah-scheduling-workers4` 的 1200 帧 / 300 帧末尾驻留尝试未通过 CLAS 最终收敛：pending 为 0，但仍有 retiring 页面。没有放宽收敛断言；延长到 1500 帧 / 600 帧末尾驻留后重新采集三次有效运行。分析脚本拒绝纳入失败记录。

所有日志、JSONL、图像和保留的旧测试二进制都在被忽略的 `build-scheduling-release/` 内。总结果为 `scheduling-summary.json`。改动前二进制 `tests/MetallicRhiTests-before-wait.exe` SHA256 为 `820069FA305DAE245700C6577974EBFA275D7EC04429546B89FFF9D3D6B8759D`；最终二进制为 `CD6028E4ED7CD02F3F6A82C673ECC51D8B9199F80D71608686CD3FAAACD2AD21`。

在 x64 VS developer shell 中复现，依次运行以避免测试自身竞争：

```powershell
cmake --build build-scheduling-release --target Metallic MetallicRhiTests MetallicTaskTests --parallel 8

# 小型对照；进程环境开关不能覆盖其中的 diagnostics-off 对照。
Remove-Item Env:METALLIC_RENDER_SCHEDULING_DIAGNOSTICS -ErrorAction SilentlyContinue
$env:METALLIC_TEST_SCHEDULING_BENCHMARK = '1'
.\build-scheduling-release\tests\MetallicRhiTests.exe --rhi-no-validation --rhi-bindless --rhi-async-compute '--gtest_filter=*scheduling_diagnostics*' --output-dir build-scheduling-release/scheduling-repro
Remove-Item Env:METALLIC_TEST_SCHEDULING_BENCHMARK

$env:METALLIC_TEST_MINIZORAH = '1'
$env:METALLIC_MINIZORAH_BENCH_FRAMES = '1500'
$env:METALLIC_MINIZORAH_BENCH_FINAL_HOLD = '600'
$env:METALLIC_MINIZORAH_RECORDING_WORKERS = '4' # 对照改为 1，其余不变
$env:METALLIC_MINIZORAH_BENCH_CLAS = '1'
$env:METALLIC_MINIZORAH_BENCH_REALTIME = '0'
$env:METALLIC_MINIZORAH_BENCH_QUALITY = '0'
$env:METALLIC_RENDER_SCHEDULING_DIAGNOSTICS = '1'
.\build-scheduling-release\tests\MetallicRhiTests.exe --rhi-no-validation --rhi-bindless --rhi-async-compute '--gtest_filter=*minizorah_fixed_baseline*' --output-dir build-scheduling-release/minizorah-repro
python -X utf8 Tools/AnalyzeSchedulingDiagnostics.py --benchmark build-scheduling-release/scheduling-repro/SchedulingBenchmark.json --minizorah build-scheduling-release/minizorah-repro/Frames.jsonl --warmup 300 --output build-scheduling-release/repro-summary.json

Remove-Item Env:METALLIC_RENDER_SCHEDULING_DIAGNOSTICS
.\build-scheduling-release\tests\MetallicRhiTests.exe --rhi-bindless --rhi-async-compute '--gtest_filter=*parallel_*:*pipelined*:*scheduling_diagnostics*:*prepared_*:*visibility_preparation*:*registry_*:*submission*' --output-dir build-scheduling-release/scheduling-regression
ctest --test-dir build-scheduling-release -R '^MetallicTaskTests$' --output-on-failure
.\build-scheduling-release\Source\Metallic.exe --smoke-test
```

`FINAL_HOLD` 和 `RECORDING_WORKERS` 默认均为 0，保留既有路线和自动 worker 上限。末尾驻留参数不能覆盖外部 camera replay。正式测量前应清除其他 MiniZorah override，并检查 `Baseline.json` 中的实际配置。
