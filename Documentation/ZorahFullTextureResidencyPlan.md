# ZorahFull 贴图驻留与显存预算方案

2026-09-20。根据用户 13:06–13:07 日志、Metallic `ae60d3d76`、本地 `E:/vk_lod_clusters` 的 `1febfa7` 调研。此次只读取源码、glTF 和 KTX2 header，未修改运行时/预设，未运行 GPU 场景。可复跑 [分析脚本](E:/metallic/Tools/AnalyzeZorahTextureResidency.py)，输出 [analysis.json](E:/metallic/build-release/zorah-texture-residency/analysis.json)。

## 本次证据改变了优化重点

Full 纹理资源阶段 wall **21578.23 ms**；其中 image/view 创建累计 **9449.31 ms**，上传背压等待 **10524.77 ms**，等待预取结果 **663.26 ms**。单张 image 创建最慢 **1123.32 ms**。4 个 decoder 正常复用，CPU 预取峰值 **4826599 B（4.60 MiB）**。读取/解码是并行累计值，不能与这些时间直接相加。

实际加载全部 4418 张贴图、44140 mip，cap 512；GPU image allocation **1469821440 B（1401.73 MiB / 1.369 GiB）**。配置允许纹理 2048 MiB，规划时不考虑 geometry、CLAS、帧资源、DLSS 或其他进程的压力。并行优化改善准备速度，没有减少最终驻留量。

13:07:27 图编译完成，13:07:31 DLSS 返回 `eWarnOutOfVRAM`，随后被映射成致命 `OutOfMemory`。日志支持预算压力及资源创建/等待长尾，并未提供实际 heap budget/usage、换页事件或分域峰值，因此不能断言具体换页量、其他应用占用或泄漏。DLSS 警告分级仍须处理，但仅修改错误分级不能解决压力。

源码另有两个影响边界：

- Full 几何池按配置初始化分配 **3.5 GiB**，compact CLAS 外层 storage 按最大容量分配 **2 GiB**；加当前贴图已约 **6.87 GiB**，还没计 CLAS build/MOVE scratch、BLAS/TLAS、帧资源与其他分配。只减少贴图不能保证整场景一定不过预算。[几何池](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:911)、[CLAS 池](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamCompactClasPool.cpp:309)。
- `allocationInfoForMemory(Device)` 设置 `VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT`，`createTexture()` 直接沿用，因此这些小 BC 纹理也强制独立分配。需测量 allocation/block 数、创建时间及实际 heap usage；不能把 image requirement 求和当作全部物理开销。[分配标志](E:/metallic/Source/Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp:1142)、[纹理创建](E:/metallic/Source/Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp:8988)。

## vk_lod_clusters 可以借鉴什么

`SceneTextures` 明确说明当前不支持 texture streaming，所有选中的尾链在加载时预载。其方案是先探测每张图片的 mip 大小，然后轮流提高每张图片的 base mip，直到 payload 总量符合预算；创建的 image 本身只包含该尾链，避免先分配完整 image 再限制采样。[声明](E:/vk_lod_clusters/src/scene_textures.hpp:23)、[规划](E:/vk_lod_clusters/src/scene_textures.cpp:172)、[KTX 尾链 image 创建](E:/vk_lod_clusters/src/scene_textures.cpp:515)。

必须保留以下差异：

1. 本地默认 texture budget 是 **4096 MiB**，Full cfg 没有覆盖该值。它并不默认比 Metallic 更省纹理。按当前 glTF 图片顺序复现其规划，4096 MiB 预算选出约 **4095.74 MiB payload**；512 MiB 预算选出约 **511.87 MiB**。这是算法模拟，不是参考程序的实测显存。
2. 它的预算与纹理统计采用 mip payload；Metallic 已有 actual image allocation 查询，应保留这一优势。它的逐图轮转并不按可见性/屏幕收益调度，末轮也受图像顺序影响，不应照搬成长期细化策略。
3. `--maxtexturemegabytes` 支持固定值或 device-local heap 百分比；百分比基于物理 heap 大小，不能替代当前可用预算。[预算参数](E:/vk_lod_clusters/src/lodclusters.cpp:790)。
4. 参考 `initScene()` 先 `deinitScene()`，有助于降低新旧场景重叠。Metallic 应在安全完成点回收旧场景，或把仍在途的旧资源完整纳入 admission 预算，而不是提前释放 GPU 正在使用的数据。[参考切换](E:/vk_lod_clusters/src/lodclusters.cpp:300)。

因此应借鉴“加载前选择可负担尾链”，进一步补上共享预算和视图需求；不能把参考的几何 streaming 等同于贴图 streaming。

## 可以减少多少

重新审计确认 glTF SHA-256 与 Z0 一致。当前选中场景用到 **1514/1514 材质、4418/4418 图片**，没有可通过删除未引用图片直接省下的内容。各槽位分别为基色 1510、金属/粗糙度 1375、法线 1399、自发光 73、specular 61 张。MASK 材质引用了 **110 张基色图**。

下面只统计选定 BC mip 的 decoded payload；不含 fallback、GPU 对齐、VMA block 空闲、staging 或 decoder：

| 首帧策略 | payload MiB | 用途 |
| --- | ---: | --- |
| 全部最高 512（当前） | 1390.71 | 当前材质质量参照 |
| 全部最高 256 | 347.91 | 简单预算降档，需检查 alpha |
| 普通图最高 256，MASK 基色最高 512 | **363.47** | 第一批较保守的首帧候选 |
| 全部最高 128 | 87.11 | 全场景粗底图 |
| 普通图最高 128，MASK 基色最高 512 | **106.56** | 按需细化版本的基础常驻集候选 |
| 全部最高 64 | 21.89 | 极低预算探针，不能直接作为质量验收配置 |

“最高 256/128”均读取该 mip 及以下完整尾链，不是只读一张图层。MASK 基色保留原来的 512 是为了不在本次降档中改变现有 alpha 测试精度；它不代表任何视角下都已满足完整源纹理质量。法线和粗糙度降档会影响近景细节与高光，必须有逐步细化及画面对照。

## 实施路线

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| **T0 共享预算与分配观测** | 接入 `VK_EXT_memory_budget` 的 heapBudget/heapUsage，补 VMA block/allocation 与分域统计；在加载前、池分配后、纹理每批、DLSS 首次求值及切换回收后打点。为尚未创建的帧资源、DLSS、CLAS scratch/growth 预留空间 | 明确整机压力下可供新增资源使用的额度；预算不足时停止细化/降档或明确最小工作集不足，不继续硬分配 |
| **T1 有预算的首帧尾链** | 引入逐 image 的 resident base mip；候选纹理物理预算 512 MiB，普通图 256、MASK 基色 512；仍不足时尝试普通图 128。预留迁移空间，并以 allocation 查询校验全部计划。材质 sampled-image 优先允许 VMA 子分配，保留驱动要求的 dedicated 分配 | 实际 allocation 和 heap 变化都下降；相同相机 MASK 轮廓、阴影、swizzle/sRGB 正确；图编译/DLSS 和多次切换通过。不要只改一个 pass 的预算 |
| **T2 可见需求驱动细化** | 基础常驻候选为 128＋MASK 512；按 VBuffer 材质/纹理需求及 UV 导数反馈选择需要的原始 mip。只给当前画面受益的图片升级，先恢复到当前 512 质量目标；利用已有有界 IO/decode 和上传链 | 面向墙壁/近景时对应纹理升清晰度，转向后不全场景上传高 mip；固定路线下记录目标与驻留 mip 差、请求延迟和图像对照 |
| **T3 冷纹理降档与峰值控制** | 对长期不可见图片退回基础尾链；细化/降档采用不同阈值与冷却期，统一计入旧 image 退休、解码、上传和迁移额度；低预算时先暂停升级，再回收冷内容 | 固定往返/急转/resize、连续 Mini↔Full 后内存回落，无周期性重复加载、悬空描述符或窗口切换峰值超预算 |

建议先实现 **T0＋T1**。512 MiB 是待测的纹理预算候选，不是对 16 GiB 显卡的固定可用空间承诺。后续 T2 的基础集约 107 MiB，可把剩余额度集中给可见细节；不能将“512 MiB 预算”实现为额外再给 512 MiB 细节缓存。

T1 的子分配改动只针对材质纹理，避免直接全局改变所有 Device buffer、RTAS 和外部共享资源的分配行为。VMA 会根据资源需求和驱动建议选择 dedicated allocation；取消强制标志后仍需比较 block 空闲量与创建成本。[VMA 内存选择](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/choosing_memory_type.html)。

## 实现时的约束

**预算读取与预留。** heapBudget/heapUsage 是动态估计，不是显存总量和整卡已用量的简单相减。Metallic 当前在相关设备配置下已启用 memory-budget 扩展，但 allocator flags 未接入 VMA memory-budget 支持，也没有给纹理规划提供实时额度；需同时补查询和消费者。新增允许额度取用户上限与 `heapBudget - heapUsage - 未落实预留 - 安全余量` 的较小值，已纳入 heapUsage 的旧资源不可重复扣除；分配失败仍需降档恢复。[Vulkan 预算说明](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html)、[VMA 预算接入](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/staying_within_budget.html)。

**优先使用普通 image 尾链替换。** 第一版不要求稀疏纹理：创建更高或更低 base mip 的新 image，复制/上传所需尾链，copy/acquire 完成后切换逻辑映射，全部 GPU 消费者结束后销毁旧 image。预算必须容纳完整新 image 与待退休旧 image 的共存，而不是只计两者尺寸之差。仅改 sampler LOD clamp、view baseMip 或停止采样不会回收普通 image 的高 mip 分配；需要实际替换 image。更长期可以评估 sparse image，但涉及格式支持、tile 粒度、mip tail 绑定与缺页处理，不应成为这次首帧修复的前置条件。[Vulkan 资源绑定](https://docs.vulkan.org/spec/latest/chapters/sparsemem.html)。

**反馈必须知道原始尺寸。** 不能只用当前低分辨率 resident image 的 `GetDimensions()` 计算需求，否则粗尾链可能永远不请求更精细 mip。保留 source dimensions、resident firstMip；使用正确 UV transform/导数计算原始 mip，再换算采样 LOD。优先级按像素收益、缺失细节、材质槽位与新增字节共同决定。

**共享与生命周期。** VBuffer 的 MASK、Deferred 和阴影必须共享同一纹理 owner/policy。现在 resource key 含 maxDimension/budget，Full JSON 两个 pass 都写了相同值；只改一个会产生不同缓存身份，可能加载两套纹理。运行时 resident mip 不应进入资源缓存 key，避免升级触发整套材质重建。逻辑 image/texture ID 保持稳定，以每帧映射或版本化描述符切换物理 image，不覆盖仍被 graphics/compute/ray-query 使用的描述符；预取请求带 scene generation，取消后不得发布旧结果。

**离屏需求与质量。** 当前阴影仍可能采样离屏 MASK 图，因此保留基础集，并让 alpha 消费路径参与需求或锁定质量底线。反射/路径追踪若接入，也不能只用主视图的像素反馈。验证包含植被/镂空、玻璃、法线与粗糙度、漫游首见模糊持续时间；降低首帧质量与稳定后的质量分别报告。

本轮结论：先让首帧只驻留可负担的尾链，同时解决总预算与小纹理分配；然后将显存留给可见高 mip。CLAS/几何池的固定大容量以及 DLSS warning 分级仍保留在整体加载计划内，不能用减少贴图掩盖这些独立问题。


2026-09-20 实施更新：T0 的统一预算、分配观测、图/DLSS 预留与加载前 cap 降档已接入，详见 [实现与压力验证](ZorahFullUnifiedMemoryBudget.md)。动态 CLAS 增长额度调度、MASK 分级保真和纹理子分配仍待后续阶段；本文前部保留调研时的实现状态。
