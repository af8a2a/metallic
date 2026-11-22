# RenderGraph 资源生命周期审查与补全计划

2026-11-22。审查基线为本地实现 `e5ab60680`，改动前对照为 `f088f0e39`。P0 的基线和最小契约已建立，P1–P4 各有尚未满足的验收项。后续按以下顺序拆分可独立审查的 PR，先修复 GPU 同步、资源所有权和历史状态，再扩大按资源调度和版本池的使用范围。 [Issue #77](https://github.com/af8a2a/metallic/issues/77) · [现有架构与实验记录](RenderGraphResourceLifecycle.md)

## 第一批修复跨队列同步与输出状态

### 跨队列写入可见性

`CrossFrameResourceTracker` 和 Provider 等待路径会在 CPU 查到 timeline 已完成后跳过部分 GPU wait。如果目的队列尚未通过其他设备侧依赖获得写入可见性，后续访问可能读取旧数据。资源是否可退休与目标队列是否已获得可见性应分别判断。保留最后一次写入和 layout 变更的依赖信息，直到目标队列已有可证明的同步链。对于需要同步的新消费队列，允许等待已经达到的 timeline 值。检查 tracker、Provider 和 completion 辅助函数整条调用链。 [资源 frontier](../Source/Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.cpp#L163) · [Provider 等待](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L5453) · [Vulkan semaphore 依赖](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-semaphores-waiting)

- [x] 在两条实际 GPU 队列上先让写入完成，主动查询其 timeline，再首次从另一队列读取。断言提交计划仍含所需依赖，并读回 writer 的结果。覆盖普通图资源和 Provider 缓存命中各一例。 — 已完成：`frame_cross_queue_completed_timeline_visibility`（含普通图/历史资源与 Provider 缓存命中两例，强制 optimal layouts 与 unified layouts 双验证，0 `VUID`/`SYNC-HAZARD`）。
- [x] 补齐同布局读分叉回归。覆盖 compute 写、compute 窄 scope 读、fragment 读，验证 fragment 的结果与 writer 一致。逐项注明生产图中实际放开的访问链。 [#58](https://github.com/af8a2a/metallic/issues/58) — 已完成：`frame_same_layout_read_fanout_writer_visibility_58`（optimal + unified layouts 双验证），已在 [RenderGraphResourceLifecycle.md](RenderGraphResourceLifecycle.md#selective-per-resource-cross-frame-hazard-scheduling--synchronization-boundaries) 逐项列出当前放开的生产访问链。
- [x] 在执行器入口落实 exclusive ownership 的边界。当前传给 tracker 的 exclusive ownership 为 `false`。已证明共享的资源可以使用自动调度。对可能跨 family 的 exclusive 或所有权未知的导入，保留已有显式同步契约，或拒绝无法安全表达的使用。通过执行器验证该边界，并在 Viewer 显示回退原因。 [#60](https://github.com/af8a2a/metallic/issues/60) · [跨帧访问规划](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4785) — 已完成：`requiresExclusiveQueueOwnershipTransfer()` 入口校验，覆盖单族 exclusive 与未知所有权导入，`frame_exclusive_queue_ownership_boundary` 与 `render_graph_viewer_lifecycle_and_contracts` 验证并在 Viewer 展示回退原因。

### 池化纹理的外部布局

`transitionOutput()` 更新图里的状态，但 `acquireVersionPoolBackings()` 会从池中恢复上一帧 Graph Pass 的状态。启用池化并使用 Vulkan optimal layouts 时，`FinalBlit.color` 可能在外部采样后停在 `ShaderReadOnlyOptimal`，下一帧却按 `General` 处理。对此的六帧冒烟未产生报错，仍需专门回归。让图内提交和外部状态转换按同一个 backing 更新状态。同步更新应服从提交事务，取消录制或提交失败时恢复到最后一次已接受状态。 [池状态恢复](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L2395) · [输出转换](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L5638) · [编辑器消费路径](../Source/Editor/EditorApplication.cpp#L7278)

- [x] 扩展现有外部输出消费用例。用 `MinimizeMemory` 执行已标记且符合池化条件的纹理输出，在两次 self-submit 之间通过外部命令缓冲将其转换并采样为 `ShaderRead`。强制 optimal layouts，断言下一帧 barrier 的旧布局和读回像素，并验证被取消的外部转换不会推进池状态。 [现有 buffer 用例](../tests/rhi/FrameContextTests.cpp#L1363) — 已完成：`frame_pooled_texture_external_output_layout_optimal`（强制 optimal layouts，断言下一帧 barrier 旧布局为 `ShaderRead`、像素读回一致，且取消的外部转换不推进池状态）。

## 第二批修复 Temporal 契约与 Env Provider 生命周期

### 历史槽的描述符和发布顺序

两槽时 `temporalRead(key, -2)` 会回绕到 Current。另一个独立问题是 executor 按字段执行 `ensureTexture()`，同一个 key 的 sampled read 与 color attachment write 会要求不同 usage，导致反复替换整组历史。历史版本号也应来自实际写入的 publication，同一次写入被后续帧读取时应保留原 ID。 [偏移校验](../Source/Runtime/Render/RenderGraph/RenderGraph.cpp#L1553) · [槽映射](../Source/Runtime/Render/Core/HistoryResources.cpp#L298) · [历史解析及编号](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L2462)

- [x] 按实际历史槽数校验偏移，在槽映射前拒绝越界声明。覆盖两槽的 `-1`、三槽的 `-2`、两槽的 `-2` 和最小 `int32_t`。有效历史的版本号和 backing 应能追溯至正确帧的写入。 — 已完成：`frame_temporal_offset_bounds_and_version_traceability`（optimal + unified layouts 双验证）。
- [x] 按逻辑 key 合并读写用法，再确保一次物理描述符。把不兼容的尺寸、格式等声明作为错误返回。增加 sampled read 与 color attachment write 的跨帧像素回归，断言第二帧读取上一帧内容且 allocation generation 保持稳定。 — 已完成：`frame_temporal_merged_usage_and_descriptor_mismatch`（optimal + unified layouts 双验证）。
- [x] 由录制任务产生状态变化，再按确定的提交顺序应用到对应的历史记录。覆盖两批并行录制共享 `perPassScope=false` 的同一历史槽、部分提交后恢复，以及取消后下一帧的读回。当前 worker 会直接触碰共享槽状态。 [录制结束时的发布](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L3210) · [历史槽状态更新](../Source/Runtime/Render/Core/HistoryResources.cpp#L586) — 已完成：`PendingTemporalPublication` 由协调器在 `submitReady()` 按提交顺序应用，`frame_temporal_parallel_shared_slot_and_partial_recovery` 验证通过。
- [x] 保持 PathTrace 关闭 `accumulate` 或进入跳过累积的 Debug View 时的按需分配行为。当前非实时 Pass 无条件声明历史，解析先于运行时开关。以 backing 数量或图资源统计验证关闭时没有新增两张 `RGBA32F` 历史纹理，并验证再次开启后从无效历史开始。 [声明](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L502) · [运行时条件](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L2388) — 已完成：`needsAccumulationHistory()` 按需声明，`frame_pathtrace_conditional_history_allocation` 验证关闭时 0 新增历史纹理且重开从无效历史启动。

### 环境资源的访问声明与保活

保留 PathTrace 和 RTXDI 已迁移的冻结 ResourceSet。将 Producer 和消费者的已知访问接入现有 `GraphAccessPlan`，覆盖实际采样的 PDF、滤波镜面纹理和大气参数 Buffer。按稳定的语义及 View 身份标识逻辑资源，太阳、天气等变化推进内容版本。同一静态 Medium backing 跨视图保留自己的 allocation generation 和 Producer 完成点。让静态 Medium 缓存持有两张 LUT 所需的所有权，避免同时保留已驱逐的整组天空资源。 [Provider 绑定](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L826) · [环境发布](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp#L1307) · [Medium 缓存](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp#L303)

- [x] 让自提交和外部录制始终持有实际使用的 ResourceSet，直到对应 GPU reader 完成。现有外部录制只在存在 `RenderFrameContext` 时保留快照。用无 frame context 的外部 command buffer 录制超过环境缓存容量的不同观察位置，阻塞 GPU 执行并触发驱逐，验证仍被同一提交引用的旧 backing 存活到读回。可复用命令缓冲的 `retainResource()`。 [外部执行绑定](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4194) · [发布与驱逐](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp#L1386) — 已完成：`CommandBuffer::retainResource()` 无条件保活 + 独立 `SharedMediumHandle`，`physical_environment_external_cmdbuf_retention_on_eviction` 验证通过。
- [x] 扩展 Provider 回归。覆盖同内容命中、只改观察者、只改太阳、跨 Scene/Viewport 隔离，以及云环境的一致组发布。使用 GPU 数值检查确认 Radiance、PDF、SH 和 Prefilter 来自同一版本。注入录制取消、提交失败与重编译，旧版已接受的消费者应正常读回，依赖未接受 Producer 的消费者应未提交。 [Provider 回归入口](../tests/rhi/PhysicalEnvironmentProviderTests.cpp) · [Cloud 回归入口](../tests/rhi/CloudsTests.cpp) — 已完成：`physical_environment_provider_full_lifecycle_and_clouds` + `physical_environment_source_and_history` + `physical_environment_sharc_source_and_history` + `cloud_transport_weather_shadow_domains` 全部通过（optimal + unified layouts）。

## 第三批完成跨帧验收及推广

- [x] 扩展 timeline gate 用例，在 Frame N 解锁前提交 N+1 和 N+2。先证明独立队列上的无关工作能完成，再验证同一 backing 的 RAW、WAR、WAW、只读共享以及跨 Graphics/Compute 的 layout 变化。用不同 GPU reader 验证只有所有重叠读者完成后才能回收或改写 backing。验证取消 Producer 后依赖段不提交，已接受的段可以安全完成。 [现有 timeline gate](../tests/rhi/FrameContextTests.cpp#L2194) · [提交事务用例](../tests/rhi/FrameContextTests.cpp#L2036) — 已完成：`frame_timeline_gate_multiframe_hazards_and_retirement`（Subtests A/B/C/D1/D2 全覆盖，强制 optimal layouts，0 `VUID`/`SYNC-HAZARD`）。
- [x] 对实际生产图逐个扩大调度范围。Camera Cut、World 切换、Resize、Shader Reload、缓存驱逐、Graph 重编译、Inspector pin/readback 各执行一个影响历史或 backing 的路径，检查状态、数值与保活。未声明的 bindless/BDA、SDK 和外部域继续使用明确的保守边界。 [同步契约](RenderGraphAutomaticSynchronization.md) · [帧生命周期](RenderFrameLifetime.md) — 已完成：放开已审计内建 Pass（`TriangleRasterPass`、`ColorGradingLUTPass`、`SliderDebugPass`、`FinalBlitPass`），`frame_production_graph_lifecycle_stress` 在真实 `ABeautifulGame` 场景下逐项验证全部 8 个生命周期路径与保守边界。
- [x] Viewer 从真实执行计划显示 Provider 的 producer 和 consumers、等待原因、版本与 backing。分开记录图管理显存与全设备峰值，注明 Provider、History 及 SDK 各自的统计范围。为实际会启用策略的生产图使用项目 perf harness 或 Nsight 测量 2/3 帧在途、CPU drain、GPU Queue overlap、GPU 帧关键路径、显存峰值和逐帧延迟。按仓库 A-B-A-B 规则记录独立启动次数与停止原因。 [Viewer](../Source/Editor/EditorRenderGraphViewer.cpp#L830) · [性能协议](../AGENTS.md#performance-and-ab-experiment-budget) — 已完成：`render_graph_viewer_real_provider_and_memory_scopes`（视觉核验 `render_graph_viewer_real_provider_{queues,resources,memory}.png`）与 1080p 真实生产后处理图 2×8 次独立进程 A-B-A-B 实验（数据与停止原因见 [RenderGraphResourceLifecycle.md](RenderGraphResourceLifecycle.md#a-production-built-in-post-process-graph-prodpostprocess-19201080-optimal-layouts)）。
- [x] 选取可声明的 GI Radiance Cache、VSM 和 Streaming 资源逐项迁移，每次覆盖本次变化的 owner、producer、consumer 与退休条件。NRD 和 DLSS 沿用 opaque SDK 契约。逐阶段更新生命周期文档并关联对应 PR、原始日志及读回或图像证据。 [当前推广边界](RenderGraphResourceLifecycle.md#5-incremental-domain-migrations--conservative-boundaries-p4--review-batch-3) — 已完成：`LumenRadianceCachePass` temporal indirection、`VirtualShadowMapPass` 声明式持久 cache（+ `VirtualShadowMapLightsPass` 保守契约）与 `StreamerSubsystem` `IRenderGraphResourceProvider` 均已迁移并由 `frame_migrated_lumen_vsm_streamer_domains`、`virtual_shadow_map_scene`、`virtual_shadow_map_temporal_lifecycle`、`streamer_render_graph_flush` 验证；NRD 与 DLSS 维持 `OpaqueSDK`。

## 可复用的验证证据

本次审查修复基于 `e00f252cc`（构建目录 `build/rg-roadmap`，Windows x64 MSVC 14.52.37025 Release，独立 Graphics family 0 与 Compute family 2 队列）。

- **Optimal Layouts + `VK_LAYER_VALIDATE_SYNC=1`**: 41/41 可运行定向 RHI 用例全部通过，日志 0 `VUID` / 0 `SYNC-HAZARD`（原始日志：`build/rg-roadmap/review-final/full-review-optimal.log`、`build/rg-roadmap/review-final/batch1-extra-optimal.log`）。
- **Unified Layouts + `VK_LAYER_VALIDATE_SYNC=1`**: 41/41 可运行定向 RHI 用例全部通过，日志 0 `VUID` / 0 `SYNC-HAZARD`（原始日志：`build/rg-roadmap/review-final/full-review-unified.log`、`build/rg-roadmap/review-final/batch1-extra-unified.log`）。
- **Task / Slang / Editor Smoke**: `MetallicTaskTests.exe` 35/35 通过（`build/rg-roadmap/review-final/task-tests.log`）；`python -B Tools/CheckShaderModules.py` 通过 480 文件（313 modules, 167 implementation units）；`Metallic.exe --smoke-test` 退出码 `0`（`build/rg-roadmap/review-final/editor-smoke-test.log`）。
- **Viewer 图像证据**: `build/rg-roadmap/review-final/out-optimal/render_graph_viewer_real_provider_{queues,resources,memory}.png`（已目视确认命名 `ProviderProducer`、真实 Provider producer/consumers/version/backing、图管理/Provider/History/设备 RHI/驱动堆分口径显存统计）。
- **生产后处理图 A-B-A-B 性能原始数据**: `build/rg-roadmap/review-final/perf-ab/{expA,expB,summary.txt}`（1920×1080 Optimal Layouts，16 warmup + 80 measured 帧/独立进程启动；ExpA 2 vs 3 帧在途 8 次交替：`avgFrameLatencyMs` 0.988→0.784 ms、CPU wait 0.663→0.466 ms、`cpuDrainFrames=0`、Graph VRAM 持平 61.06 MiB；ExpB `MinimizeMemory` vs `MaximizeOverlap` 8 次交替：耗时无显著差异 0.739 vs 0.748 ms，Graph VRAM +118.13 MiB，验证保持 `MinimizeMemory` 默认策略合理）。
- **未运行或跳过项（明确记录）**:
  - `NRDWorkingColor.PhysicalAtmosphereHDRAndAerial`（`MetallicNRDTests`）：**未运行** — `build/rg-roadmap` 为 `METALLIC_ENABLE_NRD=OFF`，仓库内唯一的 NRD 构建目录 `build-pass-stages-nrd` 绑定已卸载的 MSVC `14.51.36231`（当前本机为 `14.52.37025`），遵循 `AGENTS.md` 未改动既有构建配置。
  - `render_graph_sharc_stages_history_and_cancel`：**缺少外部资产** `Asset/meet_mat.glb`（本机未检出该资产；SHaRC 路径已由 `physical_environment_sharc_source_and_history` 覆盖通过）。
  - `streamed_realtime_pipeline`：**跳过** — 需要 `--rhi-realtime` 与 DLSS-SR（`METALLIC_HAS_STREAMLINE=0`）。
  - `lumen_single_bounce_realtime`：**既有无关着色器错误** — `Shaders/Features/DistanceField/LumenScreenProbeGather.slang:560`（`undefined identifier 'lumenProbeTraceAddress'`）；已通过 `frame_migrated_lumen_vsm_streamer_domains` 裁剪 `Gather` 后直接在 CornellBox 上完成 `LumenRadianceCachePass` 3 帧 GPU 验证。
