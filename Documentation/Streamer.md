# Streamer 与统一 GPUDriven 实时路径

`StreamerSubsystem`（`render.streamer`）统一管理场景资源和流送，代码集中在 `Source/Runtime/Render/Streamer/`：

- `SceneResourceManager`：场景元数据与 GPU 材质、纹理快照。
- `StreamingUploads`、`UploadStreamer`：按提交帧管理上传环、传输与完成状态。
- `MeshletStreamRuntime`：按视图分配的流式会话，包含页表、GPU 反馈、LOD 遍历及活动 cluster。
- `MeshletStreamResidency`、`MeshletStreamPageLoader`：预算内驻留、异步页面读取、上传与回收。
- `MeshletStreamCLAS`、`MeshletStreamCompactCLASPool`：驻留页面的 CLAS 和压缩池。

VisibilityBuffer 和保留的 StreamAsset 诊断 Pass 通过 `acquireStream()` 借用会话。子系统拥有会话；提交帧保留强引用以保护 GPU 使用中的资源。借用结束且提交帧释放引用后，子系统回收会话。不同视图使用独立遍历反馈；GPUScene 继续负责稳定的场景身份和绘制索引。应用关机仍须先等待 GPU 完成。

默认 `MetallicGPUDrivenSample` 使用 `gpu_driven_realtime.metallic_graph.json`，加载 `Asset/MiniZorah/zorah_main_public.v2.gltf` 的元数据及 `Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin`。场景统一进入几何流送，不再提供 resident/streaming 模式开关。匹配的缓存通过元数据加载；普通可导入场景缺少有效缓存时，CPU 导入结果作为流送烘焙输入。缺少源依赖或格式无法导入时明确失败。

实时链路共用 VisibilityBuffer、LightGrid、OpenPBR Deferred、光追阴影/SIGMA、DLSS-SR、自动曝光及可选 DLSS-NR。默认图不包含路径追踪 Pass。Deferred 的首个交点来自可见性记录，不发射主相机射线查询；所有可见性记录均从 Streamer 页池解码，GPUScene 只保存身份、变换、bounds 和材质映射。主光源阴影必须连接同一可见性 producer 发布的 CLAS TLAS；缺失或不匹配时明确失败。

当前 position-only StreamAsset 格式提供位置与标量材质，流式法线由三角形恢复。此路径要求不透明材质；纹理、真实透明或体积透射不能由该缓存表达，会明确拒绝。材质资源可独立于几何流送会话准备，避免加载 MiniZorah 的全量源几何。流式阴影精度随驻留 LOD 变化；当前流式着色使用主光源阴影信号，其余灯光仍参与 LightGrid 照明。

`MetallicGPUDrivenSample` 仅提供 MiniZorah 与 ZorahFull 两个正式场景；默认 MiniZorah，`--zorah-full` 加载 Full，`--list-scenes` 列出入口。File 菜单直接切换场景；File/Open 或拖入这两个源文件会加载对应的完整预设，包括相机、缓存与预算。保留 `--streamasset-path`、`--smoke-test`、`--debug-control`；`--sample` 仅接受这两个场景的 ID，不再支持任意 `--scene` 覆盖。其他诊断样例仍可在通用 `Metallic` 编辑器的 Built-in Sample 中选择。用法见 [GPUDrivenSample](GPUDrivenSample.md)。

相关回归：`streamed_realtime_pipeline` 覆盖独立材质资源、统一流送资源契约、流式 TLAS、材质分箱一致性、相机 guides、resize 和会话回收；`minizorah_realtime_pipeline` 在 `METALLIC_TEST_MINIZORAH=1` 下验证完整默认场景及 DLSS-SR。两者使用 `--rhi-realtime` 设备。原有 `stream_metadata_contract`、`stream_metadata_vbuffer` 与 Sponza 剔除测试保留。

默认实时预算为 512 MiB 几何页、256 MiB CLAS、256 MiB 动态 BLAS，活动 group 上限 65,536。粗 LOD 回退页保持锁定；这些预算为后处理和 DLSS 留出显存。

光追视图通过 `lodPixelError`（默认 1.5，显示像素）和 `lodBias`（默认 0）控制几何质量；降低误差或使用负 bias 会提高细节需求。路径追踪、材质可视化、光线查询可视化及 RTXDI 共用相机和质量参数传递，按显示高度换算到渲染像素，避免渲染比例改变几何细节标准。

RT 采用到变换后的误差扩展球最近点的距离计算像素误差，不使用视空间深度，因此纯相机旋转不会改变距离指标；正交相机仍使用正交投影比例。屏外按 `rtOffscreenErrorScale`（默认 2，范围 1–16）放宽误差预算，继续流送和构建反射、阴影所需的几何，不强制降为根 LOD；设为 1 使用相同的屏内外预算。VBuffer/GPUDriven 作为共享光栅与 RT producer 时，屏内取光栅投影指标和 RT 指标所要求的更高精度，RT 参数在 producer 节点上设置。

`rtLodHysteresis`（默认 0.2，范围 0–0.9）控制双阈值：无细化需求时误差超过预算 `T` 才细化，已有需求在误差降至 `T * (1 - rtLodHysteresis)` 后才降级。历史按实例和 group 保存 GPU 实际执行的需求，独立于页面与 CLAS 是否就绪；取消录制不推进历史，修改质量参数或切换手动 LOD 会使旧策略历史失效。预取不写历史。调试快照 `rtLod` 提供解析后的阈值与历史分配字节；原 transition telemetry 的 threshold 归因仅用于光栅，RT 不将无法重建滞回历史的变化误记为阈值切换。

RT cut 的子级激活与父级替换使用同一就绪条件：几何页可用且 CLAS 页表已发布 Active。细页先到、CLAS 尚在构建时，保持完整的可用祖先覆盖，避免仅因 CLAS 延迟使整实例回退到根 BLAS。祖先页继续参与需求与预算；几何页和 CLAS 的实际分配不会因为切换 cut 而从统计中消失。几何页回收还要等待引用它的 GPU 提交完成或取消，等待期间仍占用页槽和字节预算。CLAS 页表通过 GPU 顺序发布，取消构建会撤销就绪状态并回收分配，允许原计划重试。

`stream_clas_cut_readiness` 覆盖 compact/direct CLAS 池与 distributed/ordered 遍历的延迟细化、祖先覆盖和预算，并逐组对照 GPU 需求历史与 CPU RT 参考，验证旋转、屏外倍率、双阈值往返、取消帧、参数失效、渲染缩放和共享光栅质量。`meshlet_lod_ray_distance_and_hysteresis` 覆盖距离公式、剪切变换、正交与阈值边界；`streamer_meshlet_geometry_retirement` 覆盖 GPU 未完成时禁止页面复用，以及取消与重置；`clas_compact_lifecycle` 同时验证直接构建的延迟发布、取消回滚和重试。

2026-09-14 验证（RTX 5060 8 GiB，RelWithDebInfo，NRD 开启）：

- `MetallicRHITests --rhi-no-validation --gtest_filter="*streamer*:*streaming_task_queue*:*meshlet_stream_page_load*:*gpu_scene*:*material_binning*:*stream_metadata*:*scene_upload_pipeline:*render_graph_sample*:*gpu_driven_sponza_culling_equivalence"`：39 项通过。
- `METALLIC_TEST_MINIZORAH=1`、`--rhi-realtime --rhi-async-compute --rhi-no-validation`：`streamed_realtime_pipeline`、`minizorah_realtime_pipeline`、`gpu_driven_sponza_realtime_pipeline` 的断言通过。MiniZorah 运行 180 帧并测试相机、resize 和释放；常驻顶点/索引资源为空。
- `METALLIC_SMOKE_TEST_FRAMES=120`、`MetallicGPUDrivenSample --smoke-test`：默认窗口完成 120 帧，退出码 0。日志位于 `.cache/gpu-driven-unify/SampleFinalSmoke.log`。
- 离屏 DLSS 测试仍在 Streamline 关闭时出现 SDK 异常，进程退出码 1；不能将该进程视为完整通过。Vulkan 验证层仍会在描述符堆范围检查中报告异常。原始日志保留于 `.cache/gpu-driven-unify/FinalRealtime.log` 和 `FirstTests.log`。
