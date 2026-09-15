# GPUDriven 与 Nanite / vk_lod_clusters 差距审核

审核日期：2026-09-15。Metallic 基线为 `d6fdc59`；本机 `E:/vk_lod_clusters` 基线为 `1febfa7694ebdc8f97005a07897029272ca9e85a`。本次核对源码、默认 graph、既有测试和测量记录，没有修改运行时代码，也没有重新运行同条件性能基准。

本次无法访问旧报告引用的 `E:/UnrealEngine`，Nanite 结论仅采用 Epic 官方文档与技术分享。不能把旧报告检查过的特定 Unreal CLAS 分支，当作当前发行版 Nanite 的默认能力。参考程序录像所用二进制与上述源码是否一致，也未验证。

## 总体判断

Metallic 已有完整的静态几何流送、连续 LOD、安全覆盖和实时渲染链路。距离 vk_lod_clusters 的主要差距是**数据密度、遍历工作组织、CLAS 发布和 BLAS 复用**；距离 Nanite 的主要差距还包括**通用内容、可编程材质光栅、多视图资源共享和工程验收**。

不宜再把以下项目列为未实现：请求与安全绘制集合解耦、基础覆盖首屏条件、同帧几何发布、两阶段保守遮挡剔除、SPD wave operations、HW/SW 混合光栅、实时延迟光照消费流送几何及 RTAS、稳定逻辑 meshlet ID、镜像变换绕序、原始调试预览的 temporal jitter 抑制。

没有足够证据给出“达到 Nanite 的百分之多少”或“慢于参考几倍”。两边渲染器、抗锯齿、LOD 度量、预算与驻留历史不同，当前截图不能作为吞吐或总显存对照。

## 当前能力与剩余边界

| 维度 | Metallic 当前实现 | 剩余差距及影响 |
| --- | --- | --- |
| 内容构建 | meshoptimizer CLOD、多父级 DAG、完整 terminal fallback、可恢复且限制工作内存的完整 MiniZorah cook | 通用属性与材质需要贯穿 cook、流送、光栅和 resolve；没有理由仅因显存大而重写整个 builder |
| GPU LOD | 实例剔除、BVH/tile 误差剪枝、每实例 64 线程 cooperative frontier、稳定 emit | 大实例仍按顺序扫描 tile；全拓扑及每实例全 group 状态常驻，工作量和内存不能仅随可见细节缩放 |
| 可见性 | instance/meshlet 剔除、两阶段保守 HZB、SPD wave ops、HW/SW 分箱与异步光栅 | HZB 还未成为生产 frontier 中的层级节点剪枝；精确分箱和实际光栅重复加载、投影几何 |
| 帧调度 | 帧槽、完成信号、图内异步计算已具备 | 默认图含不支持 frame overlap 的 pass，下一帧执行前会等待前一帧图任务；图内 async 不等于跨帧流水 |
| 流送 | mmap、并发页读取、优先级、预取、冷页回收；需求独立于 drawable cut；同录制上传发布 | 页任务先生成独立 payload，再复制到 staging；缺统一字节/时间准入；CLAS 就绪仍落后于几何 |
| 首屏与运动 | 根几何齐备后展示；RT 开启还等根 CLAS/fallback BLAS；resize 保留会话 | 首屏条件保证粗级完整，不保证初始视图细节收敛；容量压力可能触发全局粗级切换 |
| CLAS | 实际尺寸紧凑分配、MOVE、延迟回收及取消处理 | 尺寸读回 CPU、CPU 分配、MOVE 再确认后发布；固定物理池，无按需增长 |
| BLAS/TLAS | 根 fallback BLAS，动态 cut BLAS/TLAS，实时阴影实际消费 | 未实现动态细节 BLAS 跨帧缓存、跨实例 sharing/merging；CLAS 不齐时整实例退回根 BLAS |
| 材质与光照 | VisibilityBuffer → OpenPBR/LightGrid/环境光照 → DLSS-SR → 自动曝光；可选 DLSS-NR | 纯流送入口仍限定静态无纹理标量材质，stream resolve 使用几何面法线；不能代表完整 PBR 资产支持 |
| 场景集成 | Streamer 子系统管理生命周期；全局 ViewConstant；支持刚体变换和可见性更新 | 多会话尚未共享同资产的页/CLAS 驻留；实例修改触发全表更新；缺流送变形路径 |
| 工程验证 | CPU/GPU cut 对照、取消/重试、镜像覆盖、相机与预览回归 | 场景级验证层运行与 Streamline 正常退出仍未通过；缺当前版本统一双端性能基线 |

默认配置见 [实时 graph](E:/metallic/Pipelines/Samples/gpu_driven_realtime.metallic_graph.json:13)：强制 stream asset，几何/CLAS/BLAS 预算为 512/256/256 MiB，每帧最多 256 页，LOD error 为 1.5 render px。`meshletNormalConeCull` 默认关闭，不能将测试中启用的模式当作默认配置。

## 1. 数据密度：已有量化证据，收益最容易验证

[当前 payload](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.h:160) 与 [CLAS 解码契约](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamClas.cpp:78) 仍要求 Float32x4 位置，即每顶点 16 B。磁盘格式支持 None/ByteRle，并不等于存在紧凑的 GPU 驻留编码。

[2026-09-13 完整缓存审计](E:/metallic/Documentation/MiniZorahMemoryComparison.json:23) 中，同一最高精度模型为 16.27 亿三角形；Metallic 与参考最高精度 cluster 数仅差约 0.2%。全部 LOD 的保留位置 payload 总和分别为 **56.340 GiB / 44.106 GiB**，相差约 **27.7%**。这些是全资产数据总和，不是运行时显存。

Metallic 位置占 payload 的 **78.16%**。单独把 16 B 改为 12 B、保持顶点数和其他数据不变，理论上可减少约 **19.5% payload**；这是布局估算，未计新增对齐，不是整帧显存或性能收益承诺。MiniZorah 此缓存没有法线/UV/tangent 数据，不能把差额归因于额外材质属性。

建议先统一 Float32x3 解码 ABI，再评估量化位置、压缩索引和属性编码。Nanite 使用可配置的位置精度量化，并明确处理模块共享边界的一致性；不能让相邻 cluster 各自独立舍入。[Epic 精度说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-technical-details)

参考 RT 模式还可在 CLAS 构建后释放位置，通过 ray-tracing position fetch 取命中三角形。Metallic 的光栅和材质重建仍需位置，不能直接照搬其 Geometry 图表大小。[参考位置生命周期](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/streaming.md)

## 2. CLAS 发布和 BLAS 复用：当前 RT 路径的明确结构性开销

[CLAS collect](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamCompactClasPool.cpp:129) 在构建完成后读回尺寸；[分配阶段](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamCompactClasPool.cpp:397) 使用 CPU 分配器；MOVE 完成后才发布 Active 地址。因此现在仍是：

`几何完成确认 → 构建 CLAS → 完成确认/尺寸回读 → CPU 分配 → MOVE → 完成确认/发布地址`

这不是渲染线程同步等待，但有多轮完成延迟。最近的同帧发布修复缩短的是几何到 raster drawable，不包含这条 CLAS 链。参考使用 GPU 实际尺寸分配和搬移数据生成，并通过 sparse buffer 增长物理池；其 CPU 仍负责容量准入及状态读回，不能称完全无 CPU 参与。参考持久池也不缩小、不整理存活对象。[参考 CLAS 分配](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/clas_allocation.md)

[动态 BLAS 阶段](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2498) 每帧重新生成输入并构建；[shader reset/count/setup](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2837) 没有判断上一帧的细节 cut 是否可复用。根 fallback BLAS 已缓存，缺的是动态细节 BLAS 的跨帧复用和跨实例共享。参考提供 [sharing](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/blas_sharing.md)、[caching](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/blas_caching.md) 和 [merging](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/blas_merging.md)，减少重复构建及部分遍历。

另外，选中 cut 中只要有页的 CLAS 未 Active，当前 shader 就让整个实例使用根 fallback。即使 raster 已显示细节，阴影仍可能较粗，随后整体更新。应分别统计 geometry-ready、CLAS-ready、BLAS-ready，以及 fallback 覆盖的屏幕比例。

建议先做“cut 未变且 CLAS 地址代数未变则复用”，再扩展同几何的兼容 cut 共享。复用键必须包含几何、cut/驻留代数与相关 ray/material flags，不能只比较 primitive ID。保持退役页及在途 BLAS 引用安全。

## 3. 遍历和元数据：GPU 驱动不等于工作量已按可见性缩放

[buildActiveTable](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:3304) 已有多阶段并行；但 [cooperative frontier](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2430) 仍为一实例一个组，顺序扫描各 LOD tile，每个 tile 后执行组屏障。可见性包含实例视锥和 group 需求判断，已有 BVH 误差剪枝；这里没有参考 persistent traversal 那种跨实例共享节点工作队列。[参考 kernel](E:/vk_lod_clusters/shaders/traversal_run.comp.glsl:657)

[状态分配](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2982) 按各实例 primitive 的**全部 group 数**预留 active/mask/sparse ID，而非当前可见 frontier。既有 [GroundReport](E:/metallic/rhi-test-output/GroundReport.json:92) 记录：

| 内存 | 字节 | MiB |
| --- | ---: | ---: |
| LOD state | 162,274,640 | 154.76 |
| LOD topology | 172,573,172 | 164.58 |
| 合计 | 334,847,812 | 319.34 |

这是镜像覆盖测试留下的 MiniZorah 报告，使用 1 GiB 几何预算且关闭 cluster RTX，不是本次新跑的默认 graph benchmark。该值用于说明页驻留池之外的开销；不能直接加到截图的 Geometry/CLAS 曲线来估算同一次运行总显存。

后续可把状态改为活动节点/组工作集，并引入跨实例 node/tile 队列及更早的层级可见性剪枝。保留多父级依赖和完整覆盖；简单无序追加会破坏安全 cut。先测 tested tiles、有效节点、单组长尾和各阶段 GPU 时间，再决定队列和拓扑分页的具体实现。

## 4. 可感知流送：原有三项问题已修复，剩余应重新归因

[StreamingConvergence](E:/metallic/Documentation/StreamingConvergence.md) 已完成正式细节需求解耦、根覆盖首屏条件及同录制 GPU 页表发布。现在不应再用“必须等父级驻留才能请求细页”解释所有变化。

剩余可见变化来源包括：

- 首屏只保证根级覆盖。是否增加初始视图误差/缺页阈值或可配置的根驻留细节，应根据等待时间与首屏质量权衡；把加载藏在遮罩后不代表吞吐提高。
- [prefixStreamLodFrontier](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2154) 在 active 容量溢出时选择全局 terminal cut。覆盖安全，但可能造成整片粗细跳变；缺预算压力下逐渐调整目标误差的控制。参考有可选 adaptive error，录像并不能证明它已启用。也不能假定 Nanite 自动解决一切容量溢出，Epic 明确记录了候选/可见 cluster buffer 的容量限制。[Epic 容量说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-technical-details)
- 已有扩张视锥/误差预取、收益优先级和冷页保留；尚未形成针对相机速度的预测及 LOD 切换迟滞。Metallic 采用更保守且随视角变化的 [LOD 投影度量](E:/metallic/Shaders/Modules/GPUDriven/MeshletLodMetric.slang:9)，1.5 px 不能与参考 1 px 直接等同。
- [加载任务](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamPageLoader.cpp:40) 先复制到独立 payload，再进入 staging；每帧页数也不代表固定字节量。可优化为有界批次与 staging 直写，并统一上传/CLAS 的时间、字节准入。
- 同帧几何首次可用早于 CPU Resident 确认，旧 `UploadToDrawable` CPU 计时不能直接度量新链路。需要 GPU 阶段时间戳和屏幕细节收敛指标。

原始 meshlet debug 已在全局 View 层抑制采样 jitter，恢复实时输出时重新启用并重置历史；它仍是无时间重建的原始图。参考录像 RT debug 使用 DLSS，不能把所有剩余边缘闪烁当作相机矩阵错误。LOD cut 变化、亚像素覆盖和重建分别检查。[预览回归](E:/metallic/Documentation/VisibilityBufferSample.md:196)

## 5. 通用材质和场景：距离 Nanite 最大的功能边界

[流送 metadata 入口](E:/metallic/Source/Runtime/Scene/scene.cpp:2648) 明确只接受外部 `.gltf` 静态三角形，拒绝非空 images/textures/skins/animations 和 morph targets。[stream surface resolve](E:/metallic/Shaders/Features/VisibilityBuffer/VisibilityBufferDeferred.slang:77) 使用 position-only 三角形计算面法线及标量材质。普通 resident 路径拥有完整 PBR 能力，并不使 pure-stream 自动获得法线/UV/切线、纹理和 masked coverage。

建议先打通静态带纹理资产的 cook 属性、流送解码、导数与材质 resolve，再实现 HW/SW 一致的 alpha mask、双面和必要的可编程光栅。Nanite 的参考价值是材质同时驱动覆盖与最终着色的体系。[Epic GPU-driven materials](https://www.unrealengine.com/blog/take-a-deep-dive-into-nanite-gpu-driven-materials)、[可编程光栅应用](https://www.unrealengine.com/tech-blog/bringing-nanite-to-fortnite-battle-royale-in-chapter-4)

参考程序支持受限纹理 PBR 和 alpha mask，但材质复杂度有意受限，纹理在初始化时载入而非随几何流送。因此纹理流送是 Metallic 的产品需求，不是 vk_lod_clusters 已完整解决的能力。[参考材质边界](E:/vk_lod_clusters/README.md:272)

[syncRuntimeScene](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2662) 支持刚体变换和可见性更新；目前任一修订变化后重建、上传整个实例表，可改为 dirty ranges。皮肤、WPO 等变形需要额外的 bounds、motion 和 RTAS 更新契约，应在静态内容闭环之后按需求推进；不把 Nanite 的所有变形功能视为无条件支持。

[StreamerSubsystem::acquireStream](E:/metallic/Source/Runtime/Render/Streamer/StreamerSubsystem.cpp:35) 每次创建独立 runtime；会话生命周期和 resize 复用已解决，同资产多视图/多会话仍未共享页及 CLAS 驻留，也缺全局预算仲裁。宜分开资产驻留池与 view 的需求/cut，避免编辑器双视图重复占用资源。

## 6. 光栅和渲染消费者：已有闭环，优化方向不必照搬参考

当前默认实时图还有一处帧间吞吐限制：[RayTracedShadowPass](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/ScreenSpaceShadowPass.cpp:14) 显式返回 `supportsFrameOverlap=false`；VisibilityBuffer 等未覆盖该接口的 pass 也继承默认 false。[执行器](E:/metallic/Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:2616) 据此执行 `graph.priorFrameDrain`，先等待先前图任务再录制下一帧。即使已有帧槽及图内 async software raster，仍不能据此声称默认图具备充分的 CPU/GPU 跨帧重叠。这里保护着 singleton upload、历史及共享资源，不能简单改成 true；应先测等待时间，再完成按帧资源和 GPU 依赖管理。此结论来自调度代码，没有新测量证明它当前占多少毫秒。

Metallic 已有两阶段保守遮挡剔除和 SPD wave reduction。参考 README 将其 HiZ 描述为基于上一帧的基础测试并提示快速运动伪影，也将 SW raster 标为特定可视化条件下的基础实现、未经充分调优。因此参考更快不能归因为“它一定拥有更好的 HZB 或 SW raster”。[参考剔除](E:/vk_lod_clusters/README.md:161)、[参考 SW 边界](E:/vk_lod_clusters/README.md:117)

当前 [精确分箱](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3555) 与 [软件光栅](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3590) 重复读取、投影唯一顶点。可评估 cook 最大边长等保守元数据以减少精确分类；无法证明覆盖安全时继续用现有精确判断。没有新的阶段 profile，不能声称它是当前第一 GPU 热点。

实时 graph 已使用统一 View、流送 VBuffer/材质 bin、LightGrid 直接光照、SH/HDRI、RT 阴影、DLSS-SR 和自动曝光，未加入 PT pass。差距是完整阴影/次级可见性需求、更多视图和内容的集成；屏外实例虽保留 fallback BLAS，却没有独立的细节需求保证高精度屏外遮挡。VSM、Lumen 是 Unreal 的相关系统，不是 Nanite 几何核心本身，也不应机械加入当前任务范围。[Epic 集成与 RT 边界](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine?application_version=5.7)

还须区分配置与生效能力：graph 配置 `sigmaDenoise=true`，但本机 `build-release/CMakeCache.txt` 中 `METALLIC_ENABLE_NRD=FALSE`；不能把 SIGMA 列为这份构建已验证的效果。DLSS-NR 节点为可选且默认关闭。

## 推进顺序与验收

| 顺序 | 工作 | 验收重点 |
| --- | --- | --- |
| 先建立基线 | 固定双方提交、资产、相机轨迹、内部尺寸、质量与预算；分别测几何链路和完整实时链路 | 冷/热加载、首屏完整覆盖时间、可见细节收敛时间、P95/P99 帧时、总显存和分项开销；不混用 RT + RR 与 raster + SR 的 FPS |
| 第一批：减少重复工作与等待 | 动态 BLAS cut 缓存；CLAS GPU 分配/发布；位置格式去掉冗余分量；默认图 frame overlap 资源改造 | 静止 cut 的 BLAS 构建量、geometry→CLAS→BLAS 延迟、实际页字节、priorFrameDrain 时间；在途引用/取消回收正确 |
| 第二批：规模和运动稳定 | 稀疏状态与跨实例遍历；预算压力下的误差控制；按字节/时间批量流送 | tested tiles/有效节点、组长尾、元数据显存、缺页与 cut churn；小预算、急转、近裁面、resize 仍完整覆盖 |
| 内容扩展主线 | 静态纹理/法线/UV/切线 → masked/双面可编程覆盖 → 多视图共享驻留 | 相同资产 resident/stream、HW/SW 的材质与覆盖一致，验证导数、镜像、法线贴图和 alpha 边缘 |
| 后续按产品需求 | 变形、更多阴影/次级视图、平台 fallback、资产工具链 | 各自有明确 bounds、运动、驻留与质量契约；不为追求功能清单复制完整 Unreal |

以上排序基于结构性证据，不是实测收益排序。BLAS、CLAS 和格式迁移可分别评估，避免一次重写所有阶段。

既有验收包含 365 组 CPU/GPU frontier 对照、16 组 runtime cut、取消/重试、MiniZorah 镜像地板覆盖，以及最新 138 帧原始预览回归，见 [流送验证](E:/metallic/Documentation/StreamingConvergence.md)、[绕序验证](E:/metallic/Documentation/StreamRasterWinding.md)、[预览验证](E:/metallic/Documentation/VisibilityBufferSample.md:196)。这些来自不同提交和运行，不能合并为本次全套测试通过。

生产验收仍有两个明确未闭环项：场景级 Vulkan validation/descriptor heap 路径异常，以及带 Streamline 的测试渲染检查通过后退出卡住。前者旧协议也可复现，尚不能据此认定为新流送代码缺陷；两者都不能记成 validation-clean 或正常退出。应在优化迭代中持续保留为验收门槛。
