# Streamer 与统一 GPUDriven 实时路径

`StreamerSubsystem`（`render.streamer`）统一管理场景资源和流送，代码集中在 `Source/Runtime/Render/Streamer/`：

- `SceneResourceManager`：场景元数据与 GPU 材质/常驻资源快照。
- `StreamingUploads`、`UploadStreamer`：按提交帧管理上传环、传输与完成状态。
- `MeshletStreamRuntime`：按视图分配的流式会话，包含页表、GPU 反馈、LOD 遍历及活动 cluster。
- `MeshletStreamResidency`、`MeshletStreamPageLoader`：预算内驻留、异步页面读取、上传与回收。
- `MeshletStreamClas`、`MeshletStreamCompactClasPool`：驻留页面的 CLAS 和压缩池。

VisibilityBuffer 和保留的 StreamAsset 诊断 Pass 通过 `acquireStream()` 借用会话。子系统拥有会话；提交帧保留强引用以保护 GPU 使用中的资源。借用结束且提交帧释放引用后，子系统回收会话。不同视图使用独立遍历反馈；GPUScene 继续负责稳定的场景身份和绘制索引。应用关机仍须先等待 GPU 完成。

默认 `MetallicGPUDrivenSample` 使用 `gpu_driven_realtime.metallic_graph.json`，加载 `Asset/MiniZorah/zorah_main_public.v2.gltf` 的元数据及 `Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin`。`streamAssetOnly=true`、`enableMeshletStreaming=true`、`autoBuildStreamAsset=false` 强制该入口使用匹配的预构建缓存。未找到缓存、缓存过期或元数据格式不支持时返回错误。

实时链路共用 VisibilityBuffer、LightGrid、OpenPBR Deferred、光追阴影/SIGMA、DLSS-SR、自动曝光及可选 DLSS-NR。默认图不包含路径追踪 Pass。Deferred 的首个交点来自可见性记录，不发射主相机射线查询；stream 记录从 Streamer 页池解码，resident 记录从 GPUScene 读取。两种记录共享材质分箱和照明实现。主光源阴影使用同一帧发布的流式 TLAS；尚未就绪时禁用该帧阴影，TLAS 就绪后重置 SIGMA 历史。

当前 position-only StreamAsset 格式提供位置与标量材质，流式法线由三角形恢复。此路径要求不透明材质；纹理、真实透明或体积透射不能由该缓存表达，会明确拒绝。材质资源可独立于常驻几何和 RTAS 准备，避免加载 MiniZorah 的全量源几何。流式阴影精度随驻留 LOD 变化；当前流式着色使用主光源阴影信号，其余灯光仍参与 LightGrid 照明。

命令行保留 `--scene`、`--streamasset-path`、`--smoke-test`、`--debug-control`，通过 `--sample <id>` 显式访问诊断场景。旧 MiniZorah/StreamAsset 快捷参数归一到默认实时路径。

相关回归：`streamed_realtime_pipeline` 覆盖独立材质资源、禁止 resident 回退、流式 TLAS、材质分箱一致性、相机 guides、resize 和会话回收；`minizorah_realtime_pipeline` 在 `METALLIC_TEST_MINIZORAH=1` 下验证完整默认场景及 DLSS-SR。两者使用 `--rhi-realtime` 设备。原有 `stream_metadata_contract`、`stream_metadata_vbuffer` 与 Sponza 剔除测试保留。

默认实时预算为 512 MiB 几何页、256 MiB CLAS、256 MiB 动态 BLAS，活动 group 上限 65,536。粗 LOD 回退页保持锁定；这些预算为后处理和 DLSS 留出显存。

2026-09-14 验证（RTX 5060 8 GiB，RelWithDebInfo，NRD 开启）：

- `MetallicRhiTests --rhi-no-validation --gtest_filter="*streamer*:*streaming_task_queue*:*meshlet_stream_page_load*:*gpu_scene*:*material_binning*:*stream_metadata*:*scene_upload_pipeline:*render_graph_sample*:*gpu_driven_sponza_culling_equivalence"`：39 项通过。
- `METALLIC_TEST_MINIZORAH=1`、`--rhi-realtime --rhi-async-compute --rhi-no-validation`：`streamed_realtime_pipeline`、`minizorah_realtime_pipeline`、`gpu_driven_sponza_realtime_pipeline` 的断言通过。MiniZorah 运行 180 帧并测试相机、resize 和释放；常驻顶点/索引资源为空。
- `METALLIC_SMOKE_TEST_FRAMES=120`、`MetallicGPUDrivenSample --smoke-test`：默认窗口完成 120 帧，退出码 0。日志位于 `.cache/gpu-driven-unify/SampleFinalSmoke.log`。
- 离屏 DLSS 测试仍在 Streamline 关闭时出现 SDK 异常，进程退出码 1；不能将该进程视为完整通过。Vulkan 验证层仍会在描述符堆范围检查中报告异常。原始日志保留于 `.cache/gpu-driven-unify/FinalRealtime.log` 和 `FirstTests.log`。
