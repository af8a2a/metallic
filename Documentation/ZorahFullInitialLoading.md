# ZorahFull 专用初始加载模式

日期：2026-09-29。

本实现将新 stream 的必需根资源加载从完整渲染帧中拆出，使用独立命令批次上传根页、构建 CLAS 和 fallback BLAS。批次由 GPU 完成点推进，不再需要每次绘制和 Present 后才能继续加载。背景分析见 [ZorahFull 加载实现对比](ZorahFullLoadingComparison20260929.md)；该报告保留为改动前的分析记录。

本次只改变运行时调度和初始上传批次额度，不修改 shader、cook 格式、根集合、法线/TBN 处理或可见性规则。ZorahFull 仍需准备原有完整 terminal fallback 覆盖；未通过删减根页或放宽 readiness 检查减少工作量。

## 配置与回退

支持几何流送的图节点使用布尔属性 `initialLoad`，默认值为 `true`。VisibilityBufferPass 的 preview stream 和 GPUDrivenStreamAssetPass 的 asset stream 均读取此属性。

```json
{
    "initialLoad": true
}
```

设为 `false` 可恢复原有逐渲染帧加载路径，适合保持其他设置不变时进行 A/B 对照：

```json
{
    "initialLoad": false
}
```

该设置不修改普通流送的 `maxPageUploadsPerFrame`、`maxUploadBytesPerFrame`、CLAS 容量或其他预算。初始加载完成后，细节请求、卸载、GPU feedback、动态 BLAS/TLAS 和渲染继续走原有路径。直接创建 `MeshletStreamRuntime` 的调用者仍自行驱动命令；专用加载器接口为 `MeshletStreamInitialLoader`。

## 执行顺序与资源生存期

`StreamerSubsystem::acquireStream()` 初始化新 runtime，并登记弱引用形式的待加载项。实际填充根资源发生在 `StreamerSubsystem::beginFrame()`：

1. 图编译或场景刷新完成各 pass 对新 stream 的引用替换。
2. `collectReleasedStreams()` 回收已无借用者的旧 stream，避免在旧根 CLAS 仍被旧 pass 持有时填充新根 CLAS。
3. `completeInitialLoads()` 完成仍存活的待加载项。
4. 开启普通图上传 frame，再开始该帧的 subsystem 和 pass 命令录制。

图编译和场景刷新沿用已有的已提交工作排空过程。弱引用不会单独延长废弃场景的寿命；仍由其他视图或有效借用者持有的 stream 不会被强制释放。Streamer 是 GPUScene 等 subsystem 的依赖，初始加载在这些 subsystem 的 frame 工作之前执行。

`prepareBeforePacing()` 可能提前对新 runtime 执行一次 CPU maintenance。首次加载批次消费其 `maintenancePrepared_` 状态并清除标志，后续批次继续正常轮询完成状态；此时新 runtime 尚无渲染产生的细节请求或反馈。

## 有界上传与加速结构批次

| 项目 | 初始加载模式 |
|---|---|
| 页面数 | 每批最多 `min(1024, maxUpdatePatches_)` 页 |
| 上传字节 | 每批目标上限 64 MiB，按设备 payload 字节计量 |
| 单个超大页 | 保留已有规则：本批尚无上传时允许一个超过字节上限的页推进，避免永久饥饿 |
| CLAS build | 继续使用 runtime 配置的 `maxClasBuildClusters`；Full 预设为 8192 |
| 常驻内存、CLAS storage、scratch | 沿用已有容量、增长策略和预算检查 |
| I/O | 沿用已有 page loader 并发和在途任务上限，不等待把批次填满 |

页面数和字节上限是独立约束。实际批次可以因 I/O 尚未完成、staging 或存储预算不足而更小。64 MiB 是设备上传计量，不等于压缩文件读取字节，也不包含全部 CLAS 临时内存。

`MeshletStreamInitialLoader` 自己持有 `StreamingUploads`、单个 slot 0 的 `RenderFrameContext`、可复用 command pool/buffer 和 `QueueSubmissionTracker`。它使用 graphics queue，调用者在协调线程串行访问队列；它不占用普通图的 streamer staging slots。

每批都执行独立的 begin、命令录制、提交、关闭提交窗口和 GPU 完成等待，再推进下一批。Compact CLAS 的 build、size readback、MOVE 和发布继续遵循已有完成点约束。不会把所有迭代记录到一个尚未提交完的 frame，也不会为整个根集合扩大 CLAS staging 容量。

初始命令仅包含根上传、页表初始化/补丁、必要 barrier、有界 CLAS 工作和 fallback BLAS。它跳过视图 LOD traversal、active table、动态 BLAS、TLAS、request readback、着色及 Present。首次正常渲染再建立视图相关数据。

`sceneReadiness()` 中 fallback BLAS 的已提交状态表示队列接受，不代表 GPU 完成；加载器在每批提交后等待 GPU，并在交接前排空自身资源，确保正常图不会读到未完成的初始批次。

## 失败处理与当前边界

每批 GPU 等待使用 30 秒超时参数；整次初始加载循环在未就绪达到五分钟时返回失败。这些检查不是严格的 UI 响应时间上限，单批 CPU 工作和错误后的资源排空仍可能占用时间。

初始化、页面读取/解码、命令录制、提交、完成等待或交接失败会记录到该待加载 session。下一帧立即返回相同错误，不会重新进入对同一部分失败状态的长时间加载，也不会跳过失败项继续普通渲染。恢复需要重新创建 stream session。已接受的 GPU 工作在加载器 reset/析构时排空，未接受的命令取消后才释放资源。

当前集成仍在主线程同步完成整次初始加载。内部 `pump()` 的 16 ms 预算只约束一次调用继续追加批次；单批可超过预算，外层循环也会持续调用直到完成或失败。因此，本实现不提供异步 UI、加载期间的交互/取消响应，亦不承诺达到参考程序的三秒。收益需要通过相同工作负载下的实测确定。

## 诊断与计时口径

加载期间约每五秒输出一次 `[StreamInitialLoad]` 进度，完成时输出最终记录：

- `resources`：readiness 的必需资源步骤；启用 cluster RTX 时根页与其 CLAS 各计一步，不是物理根页数量。
- `batches`：加载器接受的独立 GPU 提交批次数。
- `uploadedMiB`：已接受上传的设备 payload 总量。
- `elapsedMs`：专用初始加载循环及交接的 wall time，不包含此前 metadata、纹理准备和 runtime 初始化。
- `gpuWaitMs`：CPU 等待 GPU 完成点的累计 wall time，不是 GPU timestamp 测得的执行时间。

加载器还提供 `pumpCalls` 和累计 `pumpMilliseconds` 供诊断。端到端比较应另保留从加载请求到首次有效画面、根资源就绪及视图细节收敛的时间；专用阶段耗时不能直接称作端到端提速。

## 性能与验证记录

在 RTX 5070 Ti 16 GB、驱动 616.92 上交替执行 normal / initial 两轮。使用同一个 Release 可执行文件、相同 shader、VSync 开、隐藏窗口、输出 960×540 / DLSS render 640×360、就绪后 warmup=0、随后录制 1 秒。已有 cooked cache 和 shader/PSO cache，未清空 OS 文件缓存，因此是缓存已存在的重复加载，不代表磁盘冷启动。

| 模式 | 第一次 `loadingSeconds` | 第二次 `loadingSeconds` | 平均 | 加载绘制次数 |
|---|---:|---:|---:|---:|
| `initialLoad=false` | 19.550 s | 17.686 s | 18.618 s | 398 / 398 |
| `initialLoad=true` | 7.798 s | 7.645 s | 7.722 s | 1 / 1 |

该计时从 benchmark 绘制循环开始，到根资源与有效预览就绪，**包括图编译，排除更早的编辑器启动**。平均减少 58.5%，约 2.41 倍。没有复现用户此前的一分钟，也没有达到参考程序的总计三秒。

专用初始加载阶段分别为 3312.589 / 3377.342 ms，均使用 148 个独立 GPU 提交，上传 3020.5 MiB 设备 payload；CPU 等待完成点累计为 400.036 / 396.956 ms。新路径的图编译另耗时 4276.32 / 4060.16 ms。51,764 个根页和对应 CLAS 都在首个图渲染帧之前准备完成，readiness 的 103,528 步全部完成。

原始证据在 [A/B 汇总](../build-release/initial-load-ab-20260929/Summary.json) 及同目录每个运行的 `Manifest.json`、`run1/Capture.json`、`run1/stdout.log`。四次可执行文件 SHA256 均为 `ADF5F4D0D35E0602E70E9FA745A09344E30F9F9A5685AD38F84FC21FED0A94E4`，shader SHA256 均为 `F766D30213123E60D4BCDA750F4840D28DAECB5FD0A216A837BF0F85729EB5B5`。

新模式在首帧只提供完整根覆盖，首帧图像仍是较粗 LOD；随后才根据视图请求细节。旧路径在加载的数百帧中已经消费过视图反馈。因此该结果不等于“全部视图细节收敛”提速，也不证明漫游帧率提升。两次新路径的就绪后零预热、1 秒短采样各出现一个约 52 / 55 ms 的帧；本次没有做相同收敛状态下的长时间漫游性能比较。

构建通过：

```powershell
cmake --build build-release --target MetallicGPUDrivenSample -j 6
cmake --build build-scheduling-release --target MetallicRhiTests -j 6
```

以下九项真实 Vulkan 测试全部通过，均开启 `--rhi-validation`，没有测试跳过或 `VUID-`：

- `RhiRendering.stream_initial_loading`：独立根/CLAS/fallback 构建、未提交取消重试、GPU gate 前不发布就绪、正常页数和字节预算、reset/reload。初次运行的 refinement 断言误用了 `requestPage()` 返回值；改为检查分配和入队后重跑通过，见 [修正后的结果](../build-scheduling-release/initial-load-regression-fixed/rhi.json)。
- `RhiCommand.streamer_meshlet_upload_completion`、`RhiCommand.streamer_ordered_publication_retry`、`RhiValidation.streamer_meshlet_upload_byte_budget`、`RhiResource.clas_compact_lifecycle`、`RhiRendering.render_graph_scene_binding_contract`、`RhiResource.render_graph_resize_reuses_compiled_passes`：见 [回归结果](../build-scheduling-release/initial-load-regression/rhi.json)，其中仅上述已修正的新测试首跑失败，其他六项通过。
- `RhiRendering.minizorah_vbuffer`：根覆盖、19,144 个实例可见性、resize、释放及重新打开，见 [结果](../build-scheduling-release/initial-load-mini/rhi.json)。已查看输出图像。
- `RhiRendering.zorah_full_first_frame`：`METALLIC_TEST_ZORAH_FULL=1`、`METALLIC_ZORAH_FULL_CYCLES=2`，两轮 Mini→Full 分别使用 asset/world 绑定，首帧就绪后各继续 120 帧，再检查材质、唯一 stream 所有者和释放，见 [完整报告](../build-scheduling-release/initial-load-full/ZorahFullFirstFrame.json)。已查看首帧和持续细化后的输出图像。

Full 验证首帧的 geometryUsedBytes=3,173,527,552、clasUsedBytes=1,499,483,008，均沿用原有预算。后续细化填满 3.5 GiB geometry budget，出现正常容量拒绝计数；page load failure 始终为零，根 readiness 不回退。两轮末尾本地 heap usage 快照分别为 9,039,536,128 / 9,396,051,968 字节；这不是整个进程的显存峰值，本轮没有连续采样峰值。测试前显存受其他应用占用时，原路径基准曾 OOM；用户释放显存后重新完成上述对照。

测试覆盖了实际 Mini→Full、重开和通用 scene refresh 合约；尚未专门运行“同一图自动 scene refresh 同时替换真实 stream”这一组合，也未做长时间漫游、强制 GPU 超时或 UI 异步交互验证。测试输出保留在忽略的 build 目录，未加入源码。

## 实现入口

- [MeshletStreamInitialLoader.h](../Source/Runtime/Render/Streamer/MeshletStreamInitialLoader.h)、[MeshletStreamInitialLoader.cpp](../Source/Runtime/Render/Streamer/MeshletStreamInitialLoader.cpp)：独立提交和完成驱动的加载器。
- [MeshletStreamRuntime.cpp](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp)：`cmdLoadInitialResources()`、共用上传和 CLAS helpers。
- [StreamerSubsystem.cpp](../Source/Runtime/Render/Streamer/StreamerSubsystem.cpp)：待加载项、frame 接入和失败锁存。
- [SceneStreamingConfig.h](../Source/Runtime/Render/Streamer/SceneStreamingConfig.h)：图属性读取。
- [ZorahFull 图预设](../Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json)：Full 的普通流送和 CLAS 容量配置。
