# Metallic 与 Nanite / vk_lod_clusters：当前实现差距

基于 Metallic 提交 `0c2aeccb5`，包含 CLAS MOVE scratch 崩溃修复。本次核对源代码与既有实验记录，未运行新的性能基准，未修改运行时。

参考源码为 `E:/vk_lod_clusters` 和 `E:/UnrealEngine`。后者 `Engine/Build/Build.version` 标识为 **5.7.4**，包含 `Nanite/RayTracing/` 下的 ClusterOps/CLAS 实现。本文将这些能力称为“本机 Nanite CLAS 路径”，不把分支代码的存在等同于所有发行版、平台或默认配置均启用。公开功能与压缩设计同时核对了 [Epic 文档](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-technical-details) 和 [GPU-driven materials 介绍](https://www.unrealengine.com/blog/take-a-deep-dive-into-nanite-gpu-driven-materials?lang=en)。

## 判断

Metallic 已具备可持续流送的 GPU 驱动几何基础：完整 cook、共享层级 DAG、完整回退 cut、屏幕需求、异步上传、两阶段遮挡剔除、并行候选展开、混合光栅、稳定可视化，以及增量 CLAS 构建、实际尺寸搬移和联合冷页回收。

剩余差距主要在 **数据密度、遍历工作组织、CLAS 发布延迟、BLAS 复用及渲染消费者**。继续按早期“单组候选展开、逐三角形重复顶点、CLAS 固定最坏尺寸槽”分配优化精力，已经不符合当前代码。

Nanite 应作为压缩、剔除/光栅和材质集成的参考；vk_lod_clusters 应作为流送 CLAS、BLAS sharing/caching 和光追工作集的参考。两者不应合并成一个统一的性能标杆。

## 当前能力和证据边界

| 能力 | 当前状态 | 仍需区分的边界 |
| --- | --- | --- |
| 完整 MiniZorah | 同一最高精度模型约 16.27 亿三角形；已有全场景 cook | 这不是运行时驻留三角形数，也不是含实例的场景总三角形数 |
| 1.5 px | 三个固定视角已按 cook 误差度量收敛；DAG 完整性有 GPU/CPU 对照 | 不等于原始全精度几何的逐像素误差证明；运动和小预算仍有加载瞬态 |
| 候选和分类 | 多组 count/prefix/scatter；128 候选/组元数据剔除；存活 cluster 才做几何分类 | 完整 cut 的 frontier 仍是一实例一组，与候选展开并行化是不同阶段 |
| 硬件光栅 | 64 线程/cluster，最多 128 个唯一顶点和 128 个三角形 | 分类阶段仍先加载/投影顶点，真正光栅时再次执行 |
| CLAS | 实际尺寸分配、驱动搬移、延迟发布与回收 | 尺寸仍读回 CPU；持久池不是稀疏增长；没有在线整理存活对象 |
| 稳定性 | 修复后 12 次编辑器切换、1200 帧在途提交漫游和 12 项回归通过 | 这是稳定性证据，不是当前整帧性能对照 |

来源：[质量](E:/metallic/Documentation/MiniZorahQuality.md)、[分类](E:/metallic/Documentation/MiniZorahClusterClassification.md)、[唯一顶点](E:/metallic/Documentation/MiniZorahIndexedHardware.md)、[CLAS](E:/metallic/Documentation/MiniZorahClasCompaction.md)、[崩溃修复](E:/metallic/Documentation/MiniZorahClasMoveCrash.md)。这些报告来自不同迭代，不能拼接为一次当前版本的测试结果。

## 1. 几何格式：最确定的数据密度差距

当前 GPU payload 保留 Float32x4 位置、每三角形 3 个 uint8 局部索引、96 B cluster 记录和 112 B page 头。格式可以携带属性，也有 ByteRle 支持，但当前 MiniZorah cook 实际仅含位置和材质标识，所有页未压缩。不能把差距解释为多存了一套法线和 UV。

既有完整缓存审计显示：

| 同一模型的全 LOD 数据 | Metallic | vk_lod_clusters |
| --- | ---: | ---: |
| 最高精度三角形 | 1,627,207,159 | 1,627,207,159 |
| 最高精度 clusters | 16,086,545 | 16,121,034 |
| 保留位置的运行时 payload 总和 | 56.340 GiB | 44.106 GiB |
| 位置记录 | 16 B | 12 B |

两边最高精度 cluster 数只差约 0.2%；布局总和相差约 27.7%。没有证据支持为了内存问题优先重写整个 cluster builder。Metallic 与参考均使用 meshoptimizer CLOD 路线；实际参数和布局并不完全一致。

Metallic 位置占全 payload 的 **78.16%**。若仅移除第四个分量，保持相同顶点数，理论上可减少约 **19.5% 的 payload**，不包含新对齐开销，也不是整机显存下降 19.5% 的承诺。CLAS 输入、HW/SW 光栅、MaterialResolve 和校验代码都写有当前位置格式契约，必须一起迁移。

Nanite 使用按精度量化、按位打包的位置及专用索引/属性解码，并在页面安装阶段 GPU 转码。差距是 GPU 驻留格式和上传格式的密度，而不只是磁盘上是否压缩。量化必须处理共享边界一致性，不能让每个 cluster 独立舍入造成裂缝。[Epic 精度说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-technical-details)

参考光追路径另有一个无法直接移植到当前 VBuffer 的优势：位置仅临时上传用于 CLAS 构建，命中着色通过内建操作获取三角形顶点，动态 Geometry 池可不保留位置。VBuffer 还要读取位置来光栅和重建三角形，不能直接释放。

代码：[payload 格式](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.h:140)、[格式检查](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.cpp:554)、[CLAS 解码契约](E:/metallic/Source/Runtime/Render/MeshletStreamClas.cpp:78)、[当前 cook 配置](E:/metallic/Source/Runtime/Scene/scene.cpp:1204)、[参考位置生命周期](E:/vk_lod_clusters/docs/streaming.md)、[Nanite 位置解码](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteDataDecode.ush:896)。数值来自[缓存审计](E:/metallic/Documentation/MiniZorahMemoryComparison.json)。

## 2. LOD 与遍历：已有并行，但工作粒度仍受实例限制

当前生产路径 `buildActiveTable` 依次执行 reset、frontier、prefix、emit、finalize，frontier/emit 为每实例一个 64 线程组。组内按 LOD 顺序串行扫描 tile，tile 内并行处理最多 64 个 group，并用组内屏障保证全部父级状态就绪。已有 BVH 范围剪枝、稀疏清理和共享父级依赖；不能描述为没有层级遍历。

其结构性限制是：大实例的 tile 链由一个组承担，其余实例完成后无法接手；可见实例仍逐 tile 检查。当前状态内存也按每个实例对应 primitive 的全部 group 预留 active、mask、稀疏 ID 三个 uint32，而不仅按实际 frontier 大小分配。原始 topology、父级反向表、BVH/tile 和细化范围在初始化时组装并全量保留。

vk_lod_clusters 的 persistent traversal 从全局节点任务队列按 subgroup 取任务、批量展开，另有分离的 group traversal。Nanite 同样有节点/cluster 队列，以及 persistent 与分阶段变体。它们的参考价值是跨实例分摊实际工作，不能把“persistent”本身当成必然更快的开关。

建议先测每实例 tested tiles、有效 nodes、组执行长尾、frontier/emit 时间及状态内存。若长尾再次由少数实例主导，再实现跨实例 node/tile 队列；保留完整父级依赖、容量不足时完整回退，以及稳定 emit。简单把当前状态更新改为无序原子追加会破坏现有 DAG 和等深覆盖契约。

相关代码：[host 阶段](E:/metallic/Source/Runtime/Render/MeshletStreamRuntime.cpp:3257)、[tile 循环](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2446)、[状态分配](E:/metallic/Source/Runtime/Render/MeshletStreamRuntime.cpp:2946)、[参考 persistent](E:/vk_lod_clusters/shaders/traversal_run.comp.glsl:535)、[Nanite 队列变体](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Nanite/NaniteCullRaster.cpp:1015)。

**LOD 数字也不能直接对齐。** Metallic 使用到近裁面的深度、球半径、误差扩张和离轴放大项，扩张球碰到近裁面时返回无限大误差。vk_lod_clusters 主要使用 `error / max(near, distance - radius)`；固定 eye 转动视角时，该距离项基本不变。Nanite 的 `GetProjectedEdgeScales` 本身也依赖视角，不能宣称 Nanite LOD 完全不随朝向变化。

因此参考的 1 px 与 Metallic 的 1.5 px 不表示相同 cut 或相同质量。建议对比相同几何组的误差分布，特别是屏幕边缘、近裁面与非均匀缩放；再评估更紧的保守投影界。不能单纯换成更宽松公式，以减少 cluster 数作为效率提升。此前全屏跳色已通过逻辑 page/cluster ID 修复，剩余 cut 变化应按稳定几何身份统计。

来源：[Metallic 度量](E:/metallic/Shaders/Libraries/GPUDriven/MeshletLodMetric.slang:7)、[参考度量](E:/vk_lod_clusters/shaders/traversal.glsl:204)、[Nanite 度量](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:233)、[稳定可视化验收](E:/metallic/Documentation/MiniZorahVisualizationStability.md)。

## 3. 软硬分类与光栅：下一步是避免重复读几何

当前早/晚两阶段剔除和异步 HW/SW 光栅已具备，唯一顶点输出也已完成。剩余明确工作量是：存活 cluster 的分类加载全部唯一顶点并投影，逐三角形执行 `hybridTriangleFits`；实际 HW 或 SW 光栅再次加载/投影顶点。一个三角形需要硬件路径时，整个 cluster 进入 HW 箱。

本机 Nanite 利用 cluster 的 EdgeLength 和投影界做 HW/SW 选择，可在读取全部顶点之前完成一部分判断。值得尝试的是 cook 保存最大边长等保守元数据，用于快速分类；无法证明满足软件覆盖约束的 cluster 继续走精确判断或硬件路径。接近近裁面、jitter、非均匀缩放和巨大三角形必须沿用现有覆盖测试。

当前精确分类已有明显降本，尚无修复后完整新 capture，不能保证这是第一 GPU 热点。候选/稳定分箱的多次 dispatch 与屏障也需单独计时，再决定是否合并。参考程序自己的 README 明确称其软件光栅较基础、未经充分调优，不能默认其 SW raster 比当前实现更高效。

代码：[精确分类](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3549)、[SW 重投影](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3590)、[Nanite EdgeLength 分类](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:315)、[参考光栅边界](E:/vk_lod_clusters/README.md:108)。

## 4. CLAS：尺寸问题已解决，分配和发布仍跨 CPU 往返

当前路径为：

`几何上传完成 → 临时 CLAS 构建 → GPU 完成 → CPU 读取尺寸并分配 → GPU MOVE → GPU 完成 → 发布持久地址`。

这条路径不阻塞渲染线程，但多了完成轮次。每帧一个有界 build batch 和一个 move batch；持久分配仍用 CPU 分配器，构建描述来自上传时提取的 CPU plan。构建预算主要按 cluster 数，尚未根据新鲜 GPU 时间估计动态预算；待 CLAS 队列也没有独立的屏幕收益/光追紧迫度排序。

vk_lod_clusters 在 GPU 上累计组内实际尺寸、从分桶空闲区分配并生成搬移数据；CPU 仍负责流送请求、准入和容量增长，因此不能称整个参考流送系统完全没有读回。其 CLAS 用 sparse buffer 按需增长，持久分配器模式明确**不缩小、不整理存活对象**。本机 Nanite CLAS 路径则包含 GPU page-size/page-offset/compaction 和 shift-left defrag，二者采用不同空间管理策略。

Metallic 的固定 512 MiB 持久 CLAS buffer 与 1 GiB 几何 buffer 仍按预算建立，图表下降主要说明有效分配减少，不表示空闲部分已退还物理显存。构建/搬移工作区约 156 MiB，独立于驻留图表；上次修复额外增加约 192 KiB MOVE scratch。

既有同轨迹 360 帧对照：几何 **363.5 → 250.6 MiB**，CLAS **1140.9 → 256.1 MiB**；同一最终 cluster 集合仅实际尺寸分配贡献 **67.4%** 的 CLAS 降幅。该内存对照发生在 scratch 修复前，修复后未重新跑同一内存实验；这些是历史量化依据，不是本次重测。

建议先补 `geometry drawable → CLAS ready → RT usable` 延迟和 batch 占用率，再移植 GPU 分配/搬移描述生成。若关注进程显存，再单独实现 sparse 或分段增长。是否做存活对象 defrag 应由最大连续空闲块、分配失败率和碎片率决定；它还要求同步更新引用这些地址的 BLAS，不能无条件每帧搬移。

代码：[尺寸读回与地址发布](E:/metallic/Source/Runtime/Render/MeshletStreamCompactClasPool.cpp:120)、[固定池与批次](E:/metallic/Source/Runtime/Render/MeshletStreamCompactClasPool.cpp:239)、[CPU 分配与 MOVE](E:/metallic/Source/Runtime/Render/MeshletStreamCompactClasPool.cpp:361)、[参考 GPU 分配器](E:/vk_lod_clusters/docs/clas_allocation.md)、[Nanite CLAS](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Nanite/RayTracing/NaniteRayTracingCLAS.cpp:291)、[Nanite defrag](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Nanite/RayTracing/NaniteRayTracingCLASDefrag.cpp:219)。

## 5. 光追：已有底层能力，当前样例尚未消费

MiniZorah 默认 `enableClas=true`、`enableClusterRtx=false`，画面为 VBuffer 加标量材质预览。更关键的是，VBuffer 的 runtime adapter **显式写死 `enableClusterRtx=false`**，并非只需改图配置即可启用完整光追。

独立 StreamAsset runtime 已有 CLAS→动态 BLAS→TLAS 和光线查询验证，不能说完全没有 BLAS/TLAS。但当前动态路径每帧重置各实例的构建状态、统计引用并生成 BLAS，没有参考的同几何实例 BLAS sharing、跨帧 caching 和 merging。静止且 cut 不变时也没有动态 BLAS 缓存命中跳过机制；预构建 terminal fallback 是已有的例外。

当前 BLAS 输入发现某个选中页 CLAS 未发布时，会将该实例置为 fallback。因此几何已细化不等于光追已获得相同细节，CPU 尺寸往返会放大这段时间。

此外，VBuffer 的主视角可见性需求不能直接成为阴影、反射、GI 的全部需求。参考光追为屏外/遮挡实例调整 LOD，而不按主可见性直接丢弃所有相关几何。接入消费者时必须定义主可见、次级射线和保底几何的预算，保留被 BLAS 引用的页面，并测试镜面中的屏外物体和背光阴影。

若下一目标是接近参考的光追画面，顺序应为：流式 RTAS 接口与阴影/反射消费者 → 实测 RT 覆盖和构建成本 → unchanged-cut 缓存与同几何 sharing → 按数据决定 merging/更复杂缓存。不要只把 `enableClusterRtx` 打开就认为链路完成。

代码：[VBuffer adapter](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:287)、[每帧 BLAS 路径](E:/metallic/Source/Runtime/Render/MeshletStreamRuntime.cpp:2460)、[缺失 CLAS 的实例回退](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2866)、[参考 BLAS sharing](E:/vk_lod_clusters/docs/blas_sharing.md)、[参考跨帧缓存](E:/vk_lod_clusters/docs/blas_caching.md)。

## 6. 材质与引擎集成：距 Nanite 产品能力仍远

当前 MiniZorah 的 `VisibilityBufferMaterialPass` 输出 Rgba8Unorm，使用几何法线、标量材质和固定方向光；明确拒绝纹理和真实透射。仓库已有其他材质、OpenPBR、光追和材质分箱能力，但当前 metadata/stream 场景并未自动接入这些消费者。`SceneResourceManager` 仍拒绝从 StreamAsset metadata 创建传统 resident/RTAS，避免回到全量几何导入。

Nanite 已有材质驱动的 programmable raster、材质分箱及最终着色，并接入引擎阴影和 GI 系统。这里的差距不适合用单个 VBuffer kernel 的毫秒数表达。[Epic 材质管线介绍](https://www.unrealengine.com/blog/take-a-deep-dive-into-nanite-gpu-driven-materials?lang=en)

对于当前无纹理 MiniZorah，先接入流式几何的真实法线/属性解码、HDR 材质和 RT 阴影，可建立有意义的消费者验证；带纹理资产、masked/WPO、动态几何、多视图阴影等是后续独立验收项。不要把 resident 路径支持某功能等同于 streamed 路径也已覆盖。vk_lod_clusters 也不是完整生产引擎；本地 README 明确记录材质纹理在初始化加载、尚不流送。

代码：[当前材质消费者](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferMaterialPass.cpp:31)、[metadata 限制](E:/metallic/Source/Runtime/Render/SceneResourceManager.cpp:141)、[Nanite 材质分箱](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteShadeBinning.usf)、[参考材质边界](E:/vk_lod_clusters/README.md:303)。

## 7. 流送与观测：下一步应跟踪质量和 RT 可用性

现有屏幕收益/字节排序、同帧准入、完成后发布、受限预取和 85%/70% 联合水位不是待补功能。相比参考 16 帧年龄阈值，Metallic 默认冷页保留 120 帧，压力下最低 16 帧，有意换取回看复用；两者驻留曲线即使相同画面也会不同。预取基于放大视锥和略细阈值，没有相机速度外推，CLAS 临时工作区/发布能力尚未成为完整的前端准入成本模型。

建议保留目前策略，再增加以下诊断：

- 活跃、缓存冷页、保底、退役分别统计字节；池预留/实际绑定、地址表、拓扑和工作区另列。
- 几何请求到可绘制、几何可绘制到 CLAS 可用、RT fallback 持续时间分开统计；同时报告未完成请求年龄和完成样本数。
- 真实 cut churn 与稳定 ID、当前可见误差、页面重载次数关联，避免把调试跳色或候选顺序变化当作 LOD 抖动。
- GPU build/move 分开计时，补最大连续空闲块、批次停滞原因；graphics/compute 的嵌套和重叠时间不可直接相加为整帧。

现有 Profiler 已有 GPU scope、表格排序、折叠图表和流送积压，本轮不是建议重做 UI。

## 推进顺序与验收

| 顺序 | 工作 | 验收重点 |
| --- | --- | --- |
| P0 | 建立修复后基线：固定步进相机、实际渲染分辨率、质量、预算、冷/热状态；VBuffer-only 与 VBuffer+CLAS 分组 | 新 capture/同轨迹 JSON；GPU P50/P95/P99；可见 cut 误差；几何/CLAS 及工作区同口径；不把开启 DLSS 的参考截图直接比较 |
| P1 | 紧凑位置与 cluster 元数据；将父级表、tile、refinement bounds 等派生静态数据纳入 cook 缓存 | 相同 cut/无损格式图像等价、上传/驻留字节下降；冷/热启动初始化时间；后续量化另测共享边界和误差 |
| P2 | 按新 profile 选择 metadata 快速分类或跨实例遍历任务队列 | 保持覆盖与完整 DAG；减少几何重复加载、tested tiles、空工作与阶段长尾；不预先承诺加速比 |
| P3 | GPU CLAS 尺寸归并、分配及搬移描述生成；容量增长单独实现 | 消除尺寸分配的 CPU 往返依赖；RT ready 延迟、暂存峰值、碎片/预算推迟下降；保留取消与重载测试 |
| RT 主线 | 接通流式 RTAS 消费者，再做 BLAS reuse/sharing | 真正渲染阴影/反射；静止 cut 下动态 BLAS 构建下降；屏外射线覆盖、地址生命周期与 fallback 正确 |

若当前主要目标仍是 MiniZorah VBuffer 的效率，P0→P1→P2 最直接；若目标已转为参考程序的光追画面，P0 后应优先开通 RT 主线，并与 P3 配套。完整 Nanite 的材质/变形/多视图生态属于后续阶段，当前不宜用“已完成百分之多少”估算。

旧 Nsight capture 早于候选展开、分类重组、唯一顶点及 CLAS 改动。本次可以确定上述实现差异，尚不能据此给出当前比 Nanite/参考慢多少倍、或者某一优化必然节省多少毫秒。
