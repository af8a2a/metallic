# M2：可信工作负载

2026-09-24。**已实现固定帧内 WorkloadCase、生产 shader/dispatch 身份、输入输出检查、独立进程 A/A runner 和 SDK 触发边界；M2 状态仍为 partial。** MiniZorah 的三次独立运行均通过身份与输出验证，但整图时间稳定性未过门槛。Full Zorah 在冻结前遇到 CLAS 显存预算不足。不能将这些结果用于自动接受优化候选。

## 本轮实现

- [WorkloadCase 工具与契约](../Tools/Perf/WorkloadCase.md)：显式选择 MiniZorah/Full Zorah、注册 variant 和 `in-frame-early-late` 范围；case 固定输出/内部渲染尺寸、预热、恢复帧及 A/A 阈值。源码、Shader、Pipeline 副本、运行库 hash、实际配置、日志、逐帧数据与读回输出统一归档，可离线 `verify`。
- [冻结 harness](../Source/Editor/EditorRasterComparison.cpp)：复用真实生产帧；`ProfileReady` 可命中显式 variant，不再仅限 legacy mode 10。diagnostic/正常 timing 分开标识。
- [生产绑定证据](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)：在绑定实际 raster pipeline 后记录 module、entry、SPIR-V FNV1a64、early/late、graphics/compute 分支及 indirect offset。诊断快照增加稳定软件 bin 列表指纹和 indirect dimensions。
- [Nsight SDK 边界](../Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp)：支持由外部 `ngfx` 注入后的 start/stop；冻结与恢复帧之后开始，GPU drain 后停止。未注入、SDK 不可用或与 Graphics Capture 冲突均失败；不把 API 调用成功当作 trace 文件或源码导出已验证。
- runner 对零软件工作量、错 shader/queue、输入/输出漂移、时间戳缺失、诊断混入计时、证据文件变化失败关闭。完整报告写出后仍不退出的进程会在 20 秒后被回收，并明确记录 `teardownReclaimed`。

## 实测：MiniZorah WorkControl

正式证据目录：[workload-m2-mini-20260924-02](../build/workload-m2-mini-20260924-02/Manifest.json)。三次独立进程，每次 3 轮 × 64 measured frames，共 576 帧；计时数据不包含诊断读回。预热为 ready 后 5 秒，恢复帧 16。统计单位是独立进程的运行中位数，而非把所有连续帧当作独立实验。

| 项目 | 结果 |
| --- | --- |
| GPU / driver | RTX 5070 Ti / 616.92 |
| 输出 / 内部渲染 | 1797×660 / 1198×440 |
| 生产入口 | `streamClusterRasterWorkControlMain`，internal raster mode 5 |
| Module | `Features/GPUDriven/GPUDrivenStreamWorkRaster` |
| SPIR-V FNV1a64 | `14699073418322314354`；这是引擎指纹，不冒充 Nsight module hash |
| 队列 | 同步 graphics；early/late 分开记录 |
| Early 软件 clusters / dispatch | 18,542 / `[18542, 1, 1]` |
| Late 软件 clusters / dispatch | 0 / `[0, 1, 1]`；此 case 没有有效 late 软件工作 |
| Cut / page mappings | `17253534412451599641` / `692342298463438016`，三次一致 |
| Before/after、轮间、进程间输出 | depth/visibility 字节一致 |
| Shader、bin 列表、驻留和参数状态 | 三次一致，检查通过 |
| 进程退出 | 三次完整报告均已写出，但退出清理卡住；runner 分别回收自有进程 |

[A/A 结果](../build/workload-m2-mini-20260924-02/AA.json)：

| 时间指标 | 三次运行中位数（ms） | 相对极差 | 预设门槛 |
| --- | --- | --- | --- |
| Software early + late | 0.139360 / 0.139456 / 0.139424 | 0.0689% | ≤10%，通过 |
| RenderGraph GPU envelope | 4.965904 / 4.540752 / 5.157552 | 12.4207% | ≤10%，未通过 |

最终判定是 **`inconclusive`**，`candidateAccepted=false`。没有调高阈值使其过关。软件 kernel 的局部重复性好，不代表完整图的稳定性已合格。Late 的时间包含空 dispatch/范围开销，不是有效 late 光栅吞吐；后续必须另选有非零 late 工作的 case。

本轮 depth/visibility 验证是固定 workload 的重复一致性，不是新算法与参考实现的正确性验收，也不是恢复全部可写资源的 isolated replay。

## 失败与未验收项

1. **Full Zorah 资源不足。** [首次运行](../build/workload-m2-20260924-01/run1/Capture.json)在 warmup 中报告 `CLAS build: OutOfMemory`。日志记录设备 heap usage 10,991,505,408 bytes、budget 11,008,159,744 bytes，并有 268,435,456 bytes safety reservation；新分配被预算管理拒绝。尚未生成有效工作负载样本。没有降低画质、关闭 CLAS 或放松预算来伪造同一 case 通过。
2. **退出清理异常。** [MiniZorah 首轮探索](../build/workload-m2-mini-20260924-01/run1/Capture.json)已完成采集，但旧 runner 等待正常退出后超时/回收；正式 runner 将报告完成与退出完成分别记录。根因尚未定位，仍是无人值守可靠性缺口。
3. **资产身份不足。** Full Zorah StreamAsset 约 205 GB，MiniZorah 约 61 GB；本版只记录路径、长度与修改时间，没有内容 hash、完整依赖闭包或跨机器可移植输入快照。`portableInputSnapshotComplete=false`，不能把 bin/cut 指纹当成全部资源内容指纹。
4. **环境干扰未排除。** 归档了 GPU clocks、utilization、temperature 和 power，但没有与测量区间对齐的逐进程竞争证明。`externalProfilerAbsenceVerified=false`，整图噪声原因尚未确定。
5. **Nsight 对齐未完成。** [SDK 试验](../build/workload-m2-sdk-20260924-02/Probe.json)通过 CLI 请求外部注入、SDK start/stop、单帧 shader profiler 与 metrics 自动导出；180 秒内未观察到 Metallic 子进程，也没有应用报告或 trace 产物，已限时清理此次 ngfx 及其 console 子进程。启动停滞的根因未定，不能归因于 SDK start/stop 调用，因为尚未执行到那里。第一次预检因输出子目录不存在退出，修正目录后才进行了此次试验。引擎侧身份不等于 Nsight shader/queue/range 已对齐；M1 的 Source/IL UI 依赖也未消除。

## 验证

- Release `MetallicGPUDrivenSample` 构建通过；日志 [m2-build-02.log](../build/m2-build-02.log)。
- 三进程数据对应上述构建及 manifest 归档的源码/二进制 hash。之后增加了 SDK 重复启动/未启动停止的状态保护；不能把历史测量标签自动转移到后来重建的二进制。
- 最终 `MetallicGPUDrivenSample` 与 `MetallicRhiTests` 构建通过；[构建日志](../build/m2-build-final.log)。[RHI SDK 状态测试](../build/m2-rhi-state.log) 1/1 通过：未启动即停止、未注入即启动、失败后再次停止均安全返回失败。
- [12 项 WorkloadCase 证据测试](../tests/perf/TestWorkloadCase.py)通过；覆盖错配置、shader/queue、零工作量、bin 漂移、读回篡改、非有限时间戳、诊断计时、跨运行身份差异、噪声判定、路径越界与失败 manifest 不得重新认证。
- [Perf CTest](../build/m2-perf-ctest.log) 4/4 通过，含之前 38 项 M0/M1 测试，总计 50 项 Python 测试。
- 正式证据包的 `verify` 通过完整性和语义检查，重新计算的 A/A 仍为 `inconclusive`；命令以退出码 2 表示此结果，而非成功资格。

## 下一步退出条件

优先定位 Full Zorah 的资源预算和 teardown 问题，补逐进程 GPU 活动及测量窗口，保持原 10% 门槛重测整图 A/A。随后完成有非零 late 工作的 case、资产/依赖内容身份、SDK trace 产物及 Nsight 范围核对。全部通过后才能把 M2 标为完成；此时仍不能省略 M3 的候选正确性与交错 A/B 验收。
