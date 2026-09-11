# 实时延迟渲染

在 Samples → Lighting 中选择 **Real-time / Physical Lighting**（ID：`realtime-lighting`）。图资产为 `Pipelines/Samples/realtime_lighting.metallic_graph.json`。

链路：`VisibilityBuffer → Deferred → DLSS-SR → AutoExposure → DLSS-NR（默认关闭）→ FinalBlit`。

- **可见性与灯光**：mesh/task shader 光栅化主可见性；VisibilityBuffer 内部构建当前视图、当前 frame slot 的 LightGrid。灯光按影响范围与渲染视锥相交筛选；即使冻结几何剔除相机，照明仍跟随渲染相机。开启抖动时，视锥和分簇边界保留一像素余量。延迟照明查询像素对应的 XY/Z 分簇，方向光和无范围灯光额外求值一次。容量溢出或深度越界使用完整候选列表，避免丢光。灯光索引直接引用网格的 GPUScene 槽位，不依赖另一个灯光列表的排序。
- **环境光**：GPU 预计算九个 cosine-convolved SH 系数用于漫反射，另生成八档 GGX 粗糙度过滤的 HDRI 用于 split-sum 镜面反射。镜面端保留原 HDRI 分辨率，水平接缝循环、垂直极点夹取，粗糙度档位之间插值。旋转、强度和材质 AO 在求值时应用；漫反射只除一次 π，并与镜面能量分配。无 HDRI 时使用程序天空。预计算在源环境更换时执行，不随相机或曝光变化重复。
- **实时模式**：`Deferred.lightingMode = realtime` 禁用渐进累积、环境逐像素路径采样和透射路径续追。直接照明保留 OpenPBR BSDF 和 ray-query 阴影。透射使用折射方向的环境近似；场景内多层折射、间接反弹和局部反射遮挡不在此模式中。原 LookDev 的 `reference` 模式仍支持采样环境及体积透射参考。
- **统一视图**：相机由视口持有的 `RenderView` 管理，图资产顶层 `view.camera` / `view.temporalJitter` 保存初始配置。Executor 每帧生成 `ViewConstants`（当前/上一帧相机、抖动、渲染/显示尺寸、切镜与历史状态），按 frame slot 上传同一个只读 GPU buffer。VBuffer 和 LightGrid 使用其当前相机；延迟 shader 直接读取该 buffer；SR 从 execution context 获取相同的相机和历史。切换预览输出不影响相机，也不再通过 `cameraSyncGroup` 或 pass 类型同步相机。
- **DLSS-SR**：默认 Quality。由 SR 查询输入尺寸并反向约束光栅/延迟分辨率，所有消费者使用 ViewConstants 的同一像素抖动。延迟导出 RG16F 的 current-to-previous UV motion 和 R32F 标准 Z（光栅可使用 reversed Z 或标准 Z）。普通平移/旋转保留 DLSS 重投影历史，只重置逐帧累积；切镜、尺寸变化、场景编辑或显式 Reset 仍重置重投影。当前运动矢量支持相机运动，不提供蒙皮或独立物体的逐顶点速度。
- **曝光**：SR 输入保持物理 HDR，AutoExposure 对升频后的 HDR 进行直方图测光、适应和色调映射。曝光设置沿用场景 Physical Lighting 中的自动曝光开关和参数。
- **可选 NR**：开启 `DlssNr.enabled`。NR 接收色调映射后的 RGBA8，以及 SR 生成的显示分辨率运动/深度引导。引导重采样去除当前抖动并在边缘选择前景深度；UV 运动不乘分辨率比例。其深度设置为 `depthInverted = false`。没有可用 NR runtime 时按 `fallbackToInput = true` 透传。SR 需要可用 NVIDIA Streamline/DLSS-SR；NR 依赖项目已有的实验性 runtime 接口。

环境漫反射/镜面分离参考本地 Unreal 的 `Engine/Shaders/Private/ReflectionEnvironmentShared.ush` 和 `BRDF.ush`；实现复用 Metallic 的 SH 与 GGX 积分近似。

视图的所有权参考 Unreal `SceneView.h` 中每个 `FSceneView` 的 `ViewUniformBuffer`：共享范围是一个视图，不是整个进程的单例。运行时可调用 `executor.bindRenderView(&view)` 并在录制下一帧前更新相机；多个 executor 各自保存帧历史。未绑定外部 View 时，具有顶层 `view` 的图会创建自己的 RenderView。编辑器只在加载旧图时从旧 `camera` 属性导入一次；旧 pass 的参数 ABI 通过执行上下文适配，节点原始属性不会随视口运动改变。`sceneBinding: asset` 或 `viewBinding: local` 保留独立相机。`rasterInfo` 继续用于验证场景身份和光栅资源，旧 SR 图仍可启用 `useRasterCamera` 兼容路径。

统一视图回归：`MetallicRhiTests --filter render_view_shared_constants_history --rhi-validation` 检查 GPU 数据共享、纯旋转、上一帧数据、切镜、resize、序列化及多视图隔离。编辑器回归使用环境变量 `METALLIC_SMOKE_TEST_SAMPLE=realtime-lighting`、`METALLIC_SMOKE_TEST_DLSS_CAMERA=1` 运行 `Metallic --smoke-test`，检查 16 帧平移/旋转和显式 DLSS Reset。

验证：`MetallicRhiTests --rhi-realtime --filter realtime_clustered_dlss_pipeline --rhi-validation`，输出 `RealtimePipeline0/1/2.png`，覆盖默认 SR、可选 NR、相机运动、静态抖动和窗口尺寸变化。`--rhi-realtime` 使用单个具备 mesh/task shader、ray query 和 Streamline 能力的设备；普通 RHI 测试入口跳过该硬件测试。另运行 `realtime_environment_prefilter_energy`、LightGrid、photometric、auto exposure 和 DLSS motion 回归。

本机验证记录（2026-09-11）：主程序构建通过；编辑器 SR 相机 smoke test 的 16 帧平移/纯旋转、相机属性隔离、历史保留和显式 Reset 全部通过，进程正常退出。启用 Vulkan 验证的 16 项相关 GPU 回归通过，包括共享 View、历史资源、独立资产视图、LightGrid、曝光及原延迟 OpenPBR 参考对比。完整 SR/NR 链路的逐帧图像、运动、标准 Z 和 resize 断言通过，但完整测试进程仍未通过：启用 Vulkan 验证后，图重建会报告 `VUID-vkCmdBindResourceHeapEXT-pBindInfo-11236`；测试进程中的 Streamline 关闭还会在 `sl.common.dll` 内发生异常并遗留一个信号量。关闭 NR 后此前也复现过关闭异常。受限进程环境中关闭调用还会等待 NGX 遥测线程。上述限制保留在测试日志中，未过滤验证消息或绕过 SDK 关闭。
