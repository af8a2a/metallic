# Unreal VSM CPU 工作与 Metallic 优化参考

2026-10-28。结论：当前最值得参考的是 **资源寿命分层与复用、紧凑的 CPU→GPU 页变更描述、跨灯面批处理**。Metallic 已有持久深度 atlas、GPU 页分配和 resident caster cache；下一步应重点消除逐面录制中的资源创建/释放，再减少 CPU dense remap。并行录制排在这两项之后。

后续实施记录：[资源复用与池化](VirtualShadowMapsResourcePool.md)、[紧凑页变更上传](VirtualShadowMapsCompactUpdates.md)。下文保留研究时的基线，实际 CPU/GPU 收益及验收限制见对应实现记录。

## 证据范围

- 直接读取 `E:/UnrealEngine/Engine/`。`Build/Build.version` 为 **5.7.4**，`Changelist=0`、`CompatibleChangelist=47537391`。本轮 Git 查询因目录访问权限失败，未用历史记录的 commit 冒充当前提交；16 个相关文件的 SHA256 记录于 [SourceIdentity.json](../.cache/vsm-ue-cpu-research/SourceIdentity.json)。
- Metallic 基线 `ed0fa34797529240f1dae0f188b371471e158c23`；引用[上一轮 CPU 细分](VirtualShadowMapsCPUProfile.md)及其原始证据。本轮只做源码研究、计算和文档更新，没有构建或运行任一引擎，没有 UE/Metallic 同场景性能对比。
- Epic 官方 [5.7 VSM 文档](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine?application_version=5.7)用于确认按需页与缓存概念；[5.7 RDG 文档](https://dev.epicgames.com/documentation/en-us/unreal-engine/render-dependency-graph-in-unreal-engine?application_version=5.7)用于确认逻辑资源、外部资源、池/瞬态资源和上传边界。实现和开关以本地文件为准。
- 下文的收益方向属于源码推导；已测 CPU 数字来自 Metallic 诊断模式的排他均值，不能当作可全部消除的时间或普通计时的加速结论。原 Sponza jittered 图像对照的一次差异与补充检查边界继续保留。

## UE 的 CPU 实际负责什么

| CPU 工作 | 粒度与输出 | 后续 GPU 工作 |
| --- | --- | --- |
| 灯光准入、clipmap/local face 投影与稳定缓存身份 | light / shadow-map entry / clipmap，生成投影、ID、偏移和有效标志 | 使用投影和缓存身份寻址 |
| 场景变化收集 | 更新/删除前的旧实例范围，以及 GPUScene 更新后的新实例范围 | 投影 bounds、标记覆盖页失效 |
| 渲染图设置 | 声明数组级页表、物理池、请求/工作列表和依赖，安排 compute/raster passes | 按 receiver 需求更新、保留、回收、分配、清页、裁剪及绘制 |
| 几何与 view 批次准备 | Nanite primary views；非 Nanite 的 mesh draw / instance culling 工作 | 压缩有效 views/mips，GPU culling 和 indirect raster |
| 参数上传和命令生成 | 批量投影/变更描述、RHI commands；满足任务依赖后消费 | 执行命令，不依赖 CPU 逐页 draw |
| 跨帧资源保留与退役 | persistent cache、前帧数据、当帧 scratch、底层池各有独立寿命 | 按 GPU 使用次序安全复用 |

UE 仍然有 CPU light/view 循环、缓存判定、场景变更收集和每帧 RDG 对象构造。关键在于工作粒度、批处理和资源复用，不能概括为“CPU 不做 VSM 工作”。

## 1. 资源复用：最直接对应当前热点

Metallic 当前状态替换加 buffer 创建为 **8.4912 / 1.1834 ms**，占 LookDev / Sponza Shadows CPU 的 **82.23% / 74.23%**。每帧基础 face buffers 为 **80 / 10 个**，其中 Device buffers 为 **40 / 5 个**。这些数字是优化目标所在的边界，不是预测收益。

UE 的资源策略有多层，必须分开理解：

| 层次 | 已核实的 UE 行为 | 对 Metallic 的实际意义 |
| --- | --- | --- |
| VSM 持久数据 | [`SetPhysicalPoolSize`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:1137) 仅在尺寸、array size、max pages 或 flags 改变时重新申请 physical pool / metadata；[每帧注册 external](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:951) | Metallic atlas 已有此类复用，当前重点是其余十槽 buffer |
| 前帧历史 | [`ExtractFrameData`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:1487) 提取 page table、flags、bounds；缓存开启时还提取 projection、physical lists、request flags | 历史资源与 scratch 分开，不能在新帧覆写仍需读取的旧页表 |
| RDG buffer pool | [`TryFindPooledBuffer`](E:/UnrealEngine/Engine/Source/Runtime/RenderCore/Private/RenderGraphResourcePool.cpp:87) 根据对齐描述寻找可用对象，外部引用仍存在则不复用；[`FindFreeBuffer`](E:/UnrealEngine/Engine/Source/Runtime/RenderCore/Private/RenderGraphResourcePool.cpp:148) 仅 miss 时创建 | 每帧创建 RDG 描述不等于每帧创建 native buffer；先在 VSM 内复用 typed/capacity slots 即可借鉴该原则 |
| Vulkan 子分配 | [`FVulkanBuffer::AllocateMemory`](E:/UnrealEngine/Engine/Source/Runtime/VulkanRHI/Private/VulkanBuffer.cpp:204) 调用 `AllocateBufferPooled`；[used/free suballocator](E:/UnrealEngine/Engine/Source/Runtime/VulkanRHI/Private/VulkanMemory.cpp:2733) 优先找可用区间，miss 才创建较大 VkBuffer/backing memory | UE 逻辑 buffer 数量不等于 VkBuffer 或 native memory allocation 数量；Metallic 当前 Device buffer 的 dedicated 策略不同 |
| 最终回收 | [`FreeVulkanAllocation`](E:/UnrealEngine/Engine/Source/Runtime/VulkanRHI/Private/VulkanMemory.cpp:2470) 默认进入延迟释放，[收集与 end-of-pipe task](E:/UnrealEngine/Engine/Source/Runtime/VulkanRHI/Private/VulkanMemory.cpp:4600) 执行退役；部分 backing memory 继续回池 | 把 free 移到别的线程可改变关键路径，但不自动减少总 CPU；复用应优先于单纯搬迁释放 |

RDG 的 `CreateBuffer/CreateTexture` 首先创建逻辑描述，底层分配由图安排；external extraction 延长寿命，transient 允许证明不重叠的资源共享存储。上传 buffer 不走 transient，indirect / extracted 资源也有策略限制，不能称 UE 所有 VSM 资源都进行瞬态别名。见 [RDG 官方说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/render-dependency-graph-in-unreal-engine?application_version=5.7)和[本地 transient 条件](E:/UnrealEngine/Engine/Source/Runtime/RenderCore/Private/RenderGraphBuilder.cpp:491)。

Metallic 对应代码：[每面 buffer 创建](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:557)、[旧 State 替换](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:807)、[Device dedicated 标志](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:802)、[同步 BufferImpl 析构](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:2424)。它已有 [graph 资源分配/alias plan](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:1244)，但 VSM 的这批私有 buffer 直接在 `record()` 创建，不经过该图资源分配路径。

建议首个候选保持现有 GPU 算法和 pass 顺序，只分离不可变录制快照与可复用 backing storage：

| 数据 | 推荐寿命 |
| --- | --- |
| atlas / 长期缓存 metadata | 场景或 shadow cache 生命周期，已有机制继续使用 |
| previous/current page table | 独立代际与读者引用；槽数按实际寿命确定，不机械固定双缓冲 |
| work / indirect / owners scratch | 容量稳定时复用；跨帧覆盖前确认 GPU 使用完成 |
| statistics / projections | 保留到所有 point-face 投影、publication 和相关 GPU 使用完成，不能在单面 raster 完成后提前回收 |
| material / coverage / view / point-face 上传数据 | 帧级共享或 revision 驱动；上传区间受 completion 约束 |

复用条件同时需要处理 GPU completion、cache 持有者和提交/取消事务；单看 `shared_ptr::use_count()` 不构成 Metallic 的完整复用合同。尺寸、usage、memory location、设备或场景变化时按拥有边界处理。Pool trim 应显式计入成本和内存策略，不照搬 UE 的尺寸对齐或闲置帧阈值。也不建议在这个候选中直接删除全局 dedicated 标志，以免把局部优化变成整个 RHI 的策略变更。

## 2. 页映射：CPU 发变更描述，GPU 应用

UE 在 [`UpdateClipmapLevel`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:202) 判定缓存是否仍有效，计算整数页偏移；[按 shadow entry 更新 next ID](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:1411)。[`FNextVirtualShadowMapData`](E:/UnrealEngine/Engine/Shaders/Shared/VirtualShadowMapDefinitions.h:151) 为 16 B，包含 flags、next ID 和 int2 offset。[CPU 上传 NextData](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:2051)后，dispatch 按物理页容量组织；[GPU 更新 physical metadata](E:/UnrealEngine/Engine/Shaders/Private/VirtualShadowMaps/VirtualShadowMapPhysicalPageManagement.usf:155)，应用偏移、换 ID 并处理越界失效。

Metallic [当前 dense remapping](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:496) 是每虚拟页两个 uint：旧页 index 和 dirty 位。当前配置为 `98,305 × 8 B = 786,440 B/face`：

| 场景 | Remap 上传字节/帧 | 占基础 HostUpload 字节 | 已测 CPU remap |
| --- | ---: | ---: | ---: |
| LookDev，8 faces | 6,291,520 B | 99.856% | 0.5488 ms |
| Sponza，1 face | 786,440 B | 99.419% | 0.1599 ms |

字节占比不等于 upload CPU 时间占比；map/flush 等有固定成本，当前上传阶段还包含其他数据。

可拆成两个独立候选，先做前者：

1. CPU 上传每 level 的 offset/有效性、dirty bounds 和当前已有投影数据；GPU 生成**原有格式**的 remap，继续使用现有 allocator。这样先消除 CPU 逐页构造和大表 HostUpload，避免同时重写页分配协议。
2. 再评估接近 UE 的 resident physical metadata 更新，减少生成 dense 中间表的 GPU 工作；这需要重新审视 owner、eviction、page ID 和缓存代际，范围更大。

Metallic 的 [`vsmResetMain`](../Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang:488) 已按虚拟页并行并消费 remap；通过原格式对照后，还可将 compact 描述的解码融入该 pass，避免独立中间表与额外 dispatch。Dirty rectangles 可保留现有 CPU bounds 投影结果，以减小迁移时浮点/边界差异；动态变化很多时，其 GPU 判定成本需另测。

CPU 数据规模可由 `faces × virtual pages` 转向 `faces × levels + dirty changes`，但会增加 GPU 计算，独立生成方案还需要额外同步，净收益必须实测。UE 是更新旧物理页的地址，Metallic 当前是由新页查旧页；偏移的方向和符号不可直接照搬。必须保持 coarse 有效性、forceRedraw、clipmap 越界、Z guard band、旧/新 bounds、预算变化、取消提交和上一帧读者的语义。

Metallic 已有 GPU 页分配 passes；[`BuildPageAllocations`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:2818) 的参考价值是进一步明确 CPU/GPU 分工，不应把“采用 GPU allocator”重新列成尚未完成的第一步。

## 3. 批量化：先减少每面重复工作

UE 的 [`ProjectionDataRDG + FRDGScatterUploadBuffer`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:2167)覆盖全体 VSM entries；CPU 按 ID 写投影并集中上传。[`CreateNaniteRenderPasses`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Shadows/ShadowSceneRenderer.cpp:1230)把多个光源、point faces 和 clipmaps 装入批次，达到 view 容量才 flush。[整批 primary-view setup](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Shadows/ShadowSceneRenderer.cpp:1207)之后，由 [GPU CompactViewsVSM](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:3844)压缩实际需要的 views/mips，再交给 multiview `DrawGeometry`。

非 Nanite 也有 [merged instance culling 与单个 raster pass](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp:4128)，内部仍循环 shadow draw；不是所有路径都把每一面变成独立 worker task。

Metallic 可先共享一个帧级 view、集中 projection/material/coverage 数据和点光面索引表，用 offsets / slices 分配 statistics、work 和 indirect 区域。第一步不必引入全局共享物理 atlas，也不必改变 light admission / 粗级保底策略。已有 [UploadStreamer completion slots](../Source/Runtime/Render/Streamer/UploadStreamer.cpp:182)可作为上传寿命的现有基础，仍需验证适合 VSM 实际的 buffer usage 与访问方式。

批处理的直接目标是资源数量、descriptor/lease 数和重复 setup；进一步合并 GPU dispatch/raster 批次应作为独立候选。当前 point face 5 的 CPU 较高还包含其他面的 projection/statistics 最后释放，不能凭这一个面较慢就给它单独增加 worker；应比较整盏 point light 或全部 faces 的总成本。

## 4. 增量失效与任务化：针对未来规模

UE [`PreSceneUpdate`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:2055)在更新/删除前收集旧实例；[`PostGPUSceneUpdate`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:2160)在新 GPUScene 发布后处理新增/更新。CPU 组织 instance range / payload 和 [load-balanced batches](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapCacheManager.cpp:1995)，GPU 在 [CacheInvalidation](E:/UnrealEngine/Engine/Shaders/Private/VirtualShadowMaps/VirtualShadowMapCacheInvalidation.ush:41)计算 bounds 覆盖和失效页。

可借鉴 scene change-set / streaming publication 驱动的增量失效，避免灯数和 caster 数增长时每面扫描整份状态。当前静态场景 projections/dirty-bounds 阶段约 0.0228 / 0.0043 ms，优先级较低；旧位置、新位置、删除及未被当帧 receiver 请求的缓存页都必须覆盖。

UE setup task 有显式输入依赖和消费等待：[`FPackedViewArray::CreateWithSetupTask`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Nanite/NaniteShared.cpp:258)、[`GetViews`](E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/Nanite/NaniteShared.h:116)。[`AddSetupTask`](E:/UnrealEngine/Engine/Source/Runtime/RenderCore/Public/RenderGraphBuilder.inl:585)在条件不满足时可 inline。借鉴点是稳定输入快照、任务拥有输出和有界粒度，而不是照抄 worker 数。

当前 demand/raster 命令 CPU 只有 0.4169 / 0.0589 ms；资源创建/释放边界占主要成本，其中包含共享锁与驱动调用，但尚未单独测量锁等待。直接并行逐面录制可能保留全部 native 创建/释放并增加竞争。先做资源复用，再复测剩余热点决定是否并行；帧间 overlap、异步 compute 与 CPU 多线程录制是不同问题，不能通过将 `supportsFrameOverlap` 改成 true 代替资源寿命设计。更完整的任务/提交差异见[并行录制对照](UnrealParallelRecordingComparison.md)，其中旧基线数字不作为本轮实测。

## 建议实施与验收顺序

| 次序 | 单一候选 | 必须新增的证据 |
| --- | --- | --- |
| 1 | VSM 内部资源 slot / 容量复用，保持 shader 和逐面录制顺序 | 稳态 native create/free 次数、pool hit/miss、状态替换与创建 CPU、回收成本、live/peak allocation；取消、resize、budget、scene handoff、在途旧页表与六面引用 |
| 2 | 紧凑页变化上传 + GPU 生成原 remap，再评估融合 reset | CPU 构造和上传字节下降；新增或融合 GPU 工作与整图开销；逐页映射/owner/overflow 对照、滚动与失效后 cached/force-redraw 图像一致 |
| 3 | 帧级共享 view/material/projection 与批量资源布局 | 每帧资源/lease 数、point table 次数、CPU stage 时间与内存；多灯/六面 seam 及所有公开统计保持一致 |
| 4 | 大场景增量失效、独立准备任务和必要的录制并行 | 多灯/动态场景规模曲线、总 worker CPU、关键路径、锁等待、提交数和 GPU idle；不能只展示主线程耗时减少 |

每个候选先做正确性与 CPU 细分定位，再用固定工作量、缓存/历史条件和预先声明预算的独立进程 ABBA 验证普通 CPU、整图 GPU 与内存。对于现有 Sponza temporal history 差异，应显式固定驻留和历史起点，并单列启动/动态检查；不能删除不一致样本。测量总成本时包含回收工作，避免把 free 移出 Shadows node 后误称消除了 CPU 开销。

本轮建议不涉及降低灯光更新频率、关闭 coarse caster、减少阴影质量或新增静/动态双份深度池。那些机制有独立质量、GPU 或内存取舍，不能作为本次同等行为下减少 CPU 创建/释放的附带变更。
