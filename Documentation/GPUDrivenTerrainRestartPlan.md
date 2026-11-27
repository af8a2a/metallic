# GPUDriven 地形重启方案

先把现有地形样例恢复为可重复验证的入口，再交付一个**地形与普通网格混排、走 OpenPBR 和 VSM 的场景**。以高度场作为地形源数据，离线烘焙的 meshlet 继续进入现有 GPUScene、Streamer 和 VisibilityBuffer。第一轮用小场景把着色、遮挡和阴影接通，下一轮用多块地形验证接缝与流式预算。

## 当前基线

| 项目 | 2026-11-27 核对结果 |
| --- | --- |
| 源数据与烘焙 | 仓库的 Houdini EXR 为 300×300。现有工具烘焙一个 Y-up glTF primitive，包含外圈裙边和一个常量材质。按现有参数重建得到 92,400 顶点、181,194 三角形，meshstream 有 128 页和 1 个 fallback 页。 [转换器](../Tools/BuildTerrainMesh.py) · [CMake 烘焙参数](../Tools/CMakeLists.txt) |
| P0 与 P1 | P0 走 `GPUDrivenStreamAssetPass` 的 LOD 诊断，P1 走统一 `VisibilityBufferPass` 的 meshlet 诊断。两者使用同一份烘焙资产。P1 的默认图直接显示 `GPUDriven.color`，正式延迟着色和 VSM 需要另行接线。 [P0 图](../Pipelines/Samples/gpu_driven_terrain_p0_streamasset.metallic_graph.json) · [P1 图](../Pipelines/Samples/gpu_driven_terrain_p1_unified.metallic_graph.json) |
| 可运行入口 | 两个 CMake smoke 目标仍传 `--terrain-p0` / `--terrain-p1`，当前专用 `MetallicGPUDrivenSample` 只接受 MiniZorah / ZorahFull。实测旧 P1 参数在启动 Vulkan 前被拒绝。通用 `Metallic.exe --smoke-test` 可通过 `METALLIC_SMOKE_TEST_SAMPLE` 选择两者。 [CMake smoke 目标](../Tools/CMakeLists.txt) · [专用入口](../Source/Samples/GPUDrivenSampleMain.cpp) · [通用 smoke 入口](../Source/Editor/EditorApplication.cpp) |
| 已验证范围 | 用现有 RelWithDebInfo 编辑器分别运行 P0、P1 各 24 帧，均退出 0，最终截图可见地形。现有 CI 测试二进制的 `render_graph_sample_load` 通过，覆盖样例登记和图结构。真实材质、跨块、VSM、RT 和受限驻留留待以下阶段验证。 |

截图和日志位于本地 `build/gpu-driven-terrain-restart/`。以上为样例运行证据，性能比较从正式混排场景开始。

## Unreal 5.7.4 的设计依据

仓库的 [Landscape 调研](UnrealLandscapeResearch.md) 基于 UE 5.6.1，[整合可行性分析](TerrainMeshShaderFeasibility.md) 记录了转向 cluster 管线的历史决策。当前方案依据本机 `E:/UnrealEngine` 的 5.7.4 源码，并结合 Epic 5.7 文档更新实施顺序。

- **源数据与渲染产物。** `LandscapeNaniteComponent.cpp:235,538,562` 异步导出标准静态网格并构建 Nanite，发布前检查内容 ID。RVT、水面等仍消费经典 Landscape 数据，几何页与源纹理可能同时驻留。Metallic 采用相同的所有权边界，按实际消费者分别记账。 [Epic Nanite Landscape](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-nanite-with-landscapes-in-unreal-engine?application_version=5.7)
- **内部边与独立块边界。** UE 的 `NaniteBuilder/Private/Cluster.cpp:660` 在简化时锁外边。启用裙边时，`LandscapeEdit.cpp:4290,4405` 在当前导出网格的外缘生成裙边，包含无相邻组件的地形外缘。经典 `LandscapeVertexFactory.ush:98,655` 用相邻较粗 LOD 和双 mip morph 处理自己的接缝。Metallic 先验证共享高度边界、简化和量化，独立烘焙边缘按需要启用裙边。 [Epic Landscape Technical Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-technical-guide-in-unreal-engine?application_version=5.7) · [Epic Nanite Skirts](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-nanite-with-landscapes-in-unreal-engine?application_version=5.7#nanite-skirts)
- **材质与其它消费者。** UE 的 `LandscapeEdit.cpp:721` 给组件材质直接绑定权重通道，`LandscapeNaniteComponent.cpp:637` 复用组件材质。`LandscapeRender.cpp:2470` 为 RVT、水面和 Lumen capture 组织独立的固定网格批次。`LandscapeCollision.cpp:1097` 创建有独立 mip 的 heightfield。Metallic 先做直接采样的材质层，再分别定义光追、GI 和物理如何取地形数据。 [Epic Landscape Collision](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-collision-guide-in-unreal-engine?application_version=5.7)
- **程序化高度需要独立论证。** UE 经典 Landscape GPU tile culling 的适用范围是 LOD0（`LandscapeCulling.cpp:520,829`）。VHM 用 min/max 高度、虚拟页遍历和双 mip 采样重构几何（`VirtualHeightfieldMesh.usf:260` 和 `VirtualHeightfieldMeshVertexFactory.ush:130`）。Nanite 材质细分则以已有网格为底，支持细节位移。Metallic 的现有细分可单独评估为微地形能力，主线继续使用烘焙高度场。 [Epic Nanite 内容工作流](https://dev.epicgames.com/documentation/en-us/unreal-engine/working-with-naniteenabled-content?application_version=5.7) · [Metallic 细分](GPUDrivenTessellation.md)

## 推进顺序

### 1 恢复基线并打通正式着色

1. 修复 `MetallicTerrainP0Smoke` 和 `MetallicTerrainP1Smoke`，复用通用编辑器的 `METALLIC_SMOKE_TEST_SAMPLE` 路径，保留现有离线烘焙步骤。P0/P1 继续作为诊断基线。
2. 增加一个正式着色样例，地形、少量不透明普通网格和方向光共用 `VisibilityBufferPass → VisibilityBufferDeferredPass`。按现有实时图接入 `VirtualShadowMapLightsPass`，固定分辨率、相机和曝光，先做无需 DLSS 的证据输出。显式设置场景、视图和 LOD 属性。要测预构建资产时，先把地形和道具放在同一个源 glTF。当前 composed scene 会选择会话烘焙，单独记录该入口的成本。 [VSM 参考图](../Pipelines/Samples/gpu_driven_realtime_vsm.metallic_graph.json) · [Streamer 资产准备](../Source/Runtime/Render/Streamer/StreamerSubsystem.cpp)
3. 固定镜头验证地形和道具深度互遮挡、双向投影、一个镜头外投影体及静态 VSM 缓存复用。切换 HZB 对照同一视图的地形深度与覆盖。证据保留最终着色图、深度或材质诊断和 VSM 页统计。

### 2 补多块接缝与流式证据

从全局样本网格生成能独立烘焙的空间块。先做 2×2 合成夹具，覆盖平地、斜坡、跨边尖脊、四块交点和不整除的外缘，再扩展到一份代表性美术高度场。块记录整数坐标、水平间距、高度缩放和偏移。相邻块读取同一全局边界样本，计算法线时共享边外一圈高度。稳定坐标样本与导出设置进入源指纹。

先复用通用简化和 meshstream DAG，验证跨块的拓扑、位置量化和不同 LOD 边界。现有 `clod::lockBoundary` 在单个 primitive 的构建范围内执行。对独立烘焙的边界，显式验收边界采样及解码后的世界坐标。若边界约束不足以消除可见裂缝，只在独立块边缘加可控裙边，并检查掠射视角、交角和 VSM 下的阴影伪影。只有通用简化在代表性高度场上无法满足覆盖、误差或压缩需求时，再比较专用高度误差与拓扑。 [简化边界](../Source/Runtime/Scene/scene.cpp#L1319) · [推荐构建诊断](../Source/Tools/MeshletCookMain.cpp)

用同一组机位覆盖俯视、贴地沿接缝、反复跨 LOD 阈值、冷启动飞行和镜头跳转。在正常预算及故意限制的预算下，逐帧检查目标地表覆盖、粗 cut 保底、队列溢出和流式收敛。先统计完整 terminal cut 的页数、字节和实例组数，再为可流送细节留出至少一个最大页的驻留空间。逐帧记录几何页、请求到可见的延迟、上传量、细化回退、光栅与阴影页队列，区分可见的 LOD 跳变与错误漏画。 [Meshlet 驻留预算](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp#L1400)

### 3 补材质与数据消费

- **材质。** 用两个可辨识的材质层验证直接采样的权重纹理，接入现有材质程序和 OpenPBR 参数。覆盖权重归一化、跨块纹理边界、世界尺度 UV、颜色、法线与粗糙度。按需扩到每张 RGBA 权重纹理四层。测过纹理工作集和材质成本后，再分别评估 SVT 的离线纹理流送和 RVT 的运行时合成需求。
- **RT 与 GI。** 在支持 streamed CLAS 的设备上，用已知世界点的 ray query 验证地形、普通网格和跨块命中，单独测 RT LOD、驻留和内存。VSM 继续走自己的视图与页请求。若需要 Lumen Surface Cache，验证其卡片与地形几何输入，作为独立验收。 [VSM 流式投影体](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.cpp) · [RT 能力](../Source/Runtime/Render/Streamer/StreamerSubsystem.cpp#L159) · [Lumen Surface Cache](../Source/Runtime/Render/RenderPass/BuiltinPass/LumenSurfaceCachePass.cpp#L618)
- **拾取、编辑与碰撞。** 开始时用现有场景节点表达地形并验证选择和变换。需要精确地表拾取时，从源高度提供 CPU 查询和 raycast，并对比固定 LOD 烘焙网格的世界坐标。引入编辑时按脏块和共享边重建，版本号或内容指纹拒绝过期任务，最新产物就绪后一次切换，并同步更新受影响的 VSM 数据及已接入的 RT 表示。物理用独立高度表示和可选降采样，按物理能力与需求排期。 [现有 CPU 拾取](../Source/Runtime/Scene/ScenePicker.cpp)

### 4 根据瓶颈评估程序化地形

当代表性场景证明几何磁盘或显存成本、局部重建延迟超过目标时，先对比 CPU 局部烘焙与 GPU 生成 cluster 页。在一块有保守 min/max 和同边采样约束的地形上验证按版本发布页、退役资源及回退覆盖。接着比较光栅、延迟重建、VSM 与 RT 的几何一致性。连续 morph 和直接从高度页重建顶点适合单独原型。地形材料的微位移可用现有细分路径做对照，并单独统计域缓冲、硬件光栅和 CLAS 成本。 [细分流水线](GPUDrivenTessellation.md) · [CLAS 细分](CLASTessellation.md)

## 验收与记录

| 目标 | 通过条件 |
| --- | --- |
| 工具链 | 空缓存能从 EXR 烘焙并启动 P0/P1。样例与图加载测试通过，截图非空。 |
| 正式场景 | 地形与普通网格共同走深度、OpenPBR、VSM。稳定机位输出可复现，镜头外的投影仍存在。 |
| 多块覆盖 | 合成夹具的共享边和四角通过 CPU 检查。掠射与混合 LOD 序列中无非预期背景像素、法线断带和明显裙边阴影。紧预算仍覆盖，恢复预算后细化收敛，溢出计数为零。 |
| 性能证据 | 固定相机路线、分辨率、LOD 误差和着色设置。分别记录冷启动和预热，报告帧时间、VisibilityBuffer、Deferred、VSM、上传与驻留；有 RT 时单独报告。等基线确认后再定具体延迟和显存目标。对优化候选按仓库 A/B 预算运行并记录停止原因。 |

### 复现当前诊断基线

先用已有构建目录的 `MetallicTerrainP0` 目标生成源 glTF 和 `.meshstream.bin`，或按 [Tools/CMakeLists.txt](../Tools/CMakeLists.txt) 用 `BuildTerrainMesh.py` 和 `Metallic --build-meshstream` 顺序执行。在仓库根运行，示例使用本机 RelWithDebInfo 构建。P0 诊断对应的 sample ID 为 `gpu-driven-terrain-p0`。

```powershell
$env:METALLIC_SMOKE_TEST_SAMPLE = 'gpu-driven-terrain-p1-unified'
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
$env:METALLIC_SMOKE_TEST_FRAMES = '24'
$env:METALLIC_SMOKE_TEST_CAPTURE = 'build/gpu-driven-terrain-p1.ppm'
.\build-relwithdebinfo\Source\Metallic.exe --smoke-test
```
