# MiniZorah 流送收敛与可感知变化对比

2026-09-15。对比 Metallic `ae7b3c5567fb4a94f46917fcb9873f51b18c1690` 与本机 `E:/vk_lod_clusters` 的 `1febfa7694ebdc8f97005a07897029272ca9e85a`。本次是源码、配置和用户录像调查，没有重新运行同条件的双端性能基准，也没有修改运行时代码。录像中的可执行文件是否恰好由该参考源码版本构建，尚未验证。

## 结论

差异不仅是磁盘读取速度。参考实现把最低 LOD 准备放在场景展示之前，能沿常驻层级直接产生细节请求，并在 CPU 收到已完成的 GPU 请求后，将上传和 GPU 驻留更新串到当前帧。Metallic 则暴露了根页分批加载过程，正式细化请求受父级可绘制状态限制，上传完成到发布之间还有 CPU 完成确认；CLAS 又有独立的构建、尺寸回读、搬移及发布轮次。这些机制足以解释为什么 Metallic 更容易出现成片、逐层变化，但各自占用多少毫秒仍需分阶段测量。

优先改进应是解耦“期望细节请求”和“安全可绘制集合”、明确首屏就绪条件、缩短上传发布链路。单纯增加加载线程或每帧页数不是当前最有依据的第一步。

## 1. 录像与配置：两边实际每帧上限相同

用户本次 GIF 共 107 帧、8.85 秒。抽取原始帧观察：

| GIF 时间 | 可观察内容 |
| --- | --- |
| 0.00 s | 兔子场景；不能把此时的设置当作 MiniZorah 设置 |
| 2.17 s | 文件选择器位于 `E:/zorah`，高亮 `zorah_main_public.v2.cfg` |
| 4.42 s | MiniZorah 已显示，仍有粗细变化；出现 `Streaming: Few CLAS groups left` 提示，界面显示约 14 FPS |
| 6.58 s | 已有较完整的建筑细节；仍有流送 CPU 工作，界面显示约 60 FPS |
| 8.83 s | 场景细节较稳定；界面显示约 34 FPS，不能据此宣称全程稳定 60 FPS |

这些是录像时间点和界面读数，不是从点击加载起算的精确收敛时间。录像也表明参考实现并非完全没有首屏后的细化。

读取录像中出现的本地 cfg，内容包括：

```text
--maxresidentgroups 131072
--maxframeloadrequests 256
--claspositionbits 8
--blassharing 1
--blascaching 1
--blasmerging 1
--mappedcache 1
--autoloadcache 1
--dlss 1
--visualize 1
```

MiniZorah 显示后，界面为 ray tracing、grey、DLSS-RR Enabled Quality、内部尺寸 1547×471、LOD pixel error 1.0；Adaptive error 看起来未勾选。cfg 与这些设置相符，但没有完整命令行和运行时配置转储，未覆盖项不能直接认定为有效默认值。

两份源 glTF（`E:/zorah/zorah_main_public.v2.gltf`、`Asset/MiniZorah/zorah_main_public.v2.gltf`）均为 10,366,362 字节，SHA-256 相同：`79BAD3A90125618CF2EDD9B3815373E9BAE94CB3CF5A917D346409C8174F7ACD`。这确认了 glTF 描述一致，不代表外部依赖及两套 cooked cache 已逐字节核对。

| 项目 | 参考程序：录像 cfg / 源码 | Metallic：默认实时 graph |
| --- | --- | --- |
| 每帧加载上限 | cfg 覆盖为 256 groups；源码默认 128 不适用于本次比较 | 256 pages；当前 cache 一组一页 |
| 批量传输字节上限 | 源码默认 32 MiB，cfg 未覆盖 | graph 以页数限额，没有对应的统一 32 MiB 配置 |
| 分组容量 | cfg 最大驻留 groups 131072 | 最大 active groups 65536；语义并不等同于驻留上限 |
| 几何 / CLAS 预算 | 源码按设备内存选择预算；运行时有效值未完整导出 | 512 / 256 MiB，另有 BLAS 256 MiB |
| 页读取 | mmap；请求批次复制或解压到 staging | mmap；4 路加载、最多 1024 个在途任务，任务产物再复制到 staging |
| LOD 阈值 | 1.0，经窗口/渲染倍率换算 | 1.5，以渲染高度计算；公式也不同 |
| 显示路径 | RT + grey + DLSS-RR，启用 BLAS sharing/caching/merging | VisibilityBuffer + 延迟光照 + RT 阴影 + DLSS-SR + 自动曝光 |

因此，256 对 256 不能说明有效吞吐相同，也不能解释成参考程序靠更多每帧请求取胜。其末段图表约 70 MiB 是图中 Geometry + CLAS 的量级，不是总显存；不同渲染路径、预算压力和驻留历史均需单独核对。

依据：`Pipelines/Samples/gpu_driven_realtime.metallic_graph.json:14`；参考 `src/scene_streaming_utils.hpp:26`、`src/lodclusters.cpp:136`；录像中的本地 cfg。

## 2. 首屏：参考程序先完成最低 LOD，Metallic 将根页排队

参考 `SceneStreaming::initGeometries()` 为每个 geometry 准备最低 LOD group，要求该 group 只有一个 cluster；根几何批量上传后 `uploader.flush()` 通过 `tempSyncSubmit()` 等待完成。RT 模式的 `initClas()` 也在初始化阶段构建最低 LOD CLAS 并同步提交。场景读取可以在后台进行，但首次可渲染场景仍以基础 GPU 资源就绪为前提。

Metallic 初始化时同样收集并锁定 terminal fallback pages，但 `lockFallbackPages()` 只是 `queueUpload()`；实际数据走后续每帧的 `processUploads()`。锁定表示不会被驱逐，不表示已能绘制。于是基础覆盖也参与渐进显示。

2026-09-12 的仓库内 cache 审计记录有 3163 个 terminal pages。若沿用该 cache、每帧上限 256，仅根页就至少需要 `ceil(3163 / 256) = 13` 个上传批次，尚未计入 I/O、GPU 完成和发布等待。这是根据历史页数计算的下界，不是本次实测耗时。

这部分优势主要是展示时机：把必要工作放在首屏前，会减少可见变化，但不会自动减少总加载时间。改进时应保留响应式加载界面或旧场景，分预算完成基础覆盖；不宜直接把全部等待搬到 UI 线程。

依据：参考 `src/scene_streaming.cpp:218,292,337,2003,2351,2438`，`src/resources.hpp:517`；Metallic `MeshletStreamRuntime.cpp:921`、`MeshletStreamResidency.cpp:657`（均位于 `Source/Runtime/Render/Streamer/`）；[历史根页验证](MiniZorahFirstFrameResult.json)。

## 3. 请求生成：逐级驻留依赖会延长细化过程

参考遍历先读取常驻的 `geometry.nodes` 和误差信息。内部节点可以继续向下，直到 group 才检查实际数据地址；遇到缺失 group 就发出请求。请求目标无需等待所有中间级别几何先变为可绘制。输出 cluster 时再检查细级 generating group 是否驻留，缺失则 `forceCluster` 保留粗 cluster。

Metallic 当前 cooperative 遍历包含：

```text
parentsActive = 所有 parent 的 active state 都已建立
demanded = terminal || (parentsActive && desired && visible)
active = demanded && groupPageDrawable(...)
```

这里把正式需求传播与可绘制状态联系起来。如果粗级未完成驻留，细级正式请求就可能尚未生成；粗级完成之后才开启下一轮。深层 LOD DAG 会放大反馈、上传和发布各阶段的延迟，表现为逐层推进。

不能把它描述为“完全没有前瞻”：当前已有独立的 request-only prefetch，能绕过父级驻留条件扫描潜在细节。但它是受限的补充：

- `availablePrefetchRequests()` 以 `maxPageLoadsInFlight / 4` 为额度，并减去已有排队上传数量。当前配置下，排队数达到 256 就返回 0。
- GPU prefetch 还要求有相机变化或近期请求，且几何使用量低于预算约 75%。
- 视锥扩展为 1.0625，误差阈值乘 0.95；预取优先级低于根页和正式需求，在途数量也受四分之一额度限制。

所以“已经有 prefetch”不能保证请求链已经解耦。在最需要加速的加载高峰，它可能恰好被积压关闭。

建议保留两套逻辑：常驻拓扑决定期望 cut 和需求优先级；已驻留数据决定当前可安全绘制 cut。实现时必须继续满足共享父级 DAG 的完整覆盖约束，不能只删除 `parentsActive` 条件，否则可能重新引入孔洞或重叠。

依据：参考 `shaders/traversal_run.comp.glsl:245`、`shaders/traversal_run_groups.comp.glsl:263`；Metallic `Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:2305,2441`、`MeshletStreamResidency.h:388`、`MeshletStreamResidency.cpp:960`、`MeshletStreamRuntime.cpp:3233`。

## 4. 上传发布与 CLAS：额外的完成确认轮次

两边都需要接收已经完成的 GPU 请求反馈。区别在收到反馈之后，不能把参考程序理解为“本帧 GPU 新发现的缺页也零延迟补齐”。

| 阶段 | 参考默认 immediate 路径 | Metallic 当前路径 |
| --- | --- | --- |
| 已完成反馈 | 消费最新的已完成请求，跳过更旧的已完成快照 | 消费反馈并纳入 residency / 加载任务队列 |
| 数据上传 | 按批将 mmap 数据复制或解压到 staging，上传命令排到当前帧 | 异步任务产物收集后排上传，页面进入 PendingUpload |
| 几何可用 | 当前帧 `cmdPreTraversal` 更新 GPU 地址，后续遍历可使用 | 后续 CPU 帧观察 completion 完成，再发布驻留页表 |
| CLAS 可用 | pre-traversal 构建，post-traversal GPU 分配及搬移，同帧后续 RT 构建使用 | 几何驻留后入 CLAS 队列；构建完成后 CPU 回读尺寸、分配，搬移完成后发布地址 |

参考默认 `useAsyncTransfer=false`。关键是同帧命令依赖与 GPU 发布，并不是默认独立传输队列；可选 async transfer 也区分 immediate 和 decoupled 模式。

Metallic 的 `completionDrivenUploads` 当前默认 **true**，不是固定等待 3 帧。问题在于它需要 CPU 跨帧观察完成状态；不能将资源帧槽数量直接算作固定额外延迟。CLAS 的 Building → Sized → Moving → Active 又增加两次 GPU 完成确认，尺寸和地址处理在 CPU。

尤其要区分：`groupPageDrawable()` 只检查几何页状态和地址，不等待 CLAS。因此 CLAS 延迟主要影响 RT 阴影/TLAS 更新，不能把所有地表细化都归因于 CLAS。延迟光照画面中的变化可能包含先出现几何、后更新阴影的两次变化。

后续可先优化几何的 upload → GPU page-table patch → traversal 依赖，再考虑 CLAS GPU 分配/地址发布。两者都必须保留取消提交、帧在途地址安全和延迟回收协议，不能简单跳过 `isComplete()`。

依据：参考 `src/scene_streaming.cpp:390,428,873,983,1226,1443`；Metallic `MeshletStreamResidency.cpp:405,1159,1190`、`MeshletStreamRuntime.cpp:2346`、`MeshletStreamCompactClasPool.cpp:125`；`Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:1205`；`Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:301`。

## 5. 请求量和画面变化还受误差公式、数据布局影响

Metallic 的投影误差使用球体到近侧的视空间深度 `minZ`，加上离轴斜率项，并将几何误差纳入球半径；靠近或穿过近裁剪面时返回最大误差。参考使用 `length(center - eye) - radius` 的距离项。后者在纯旋转时距离误差项基本不变，前者会随朝向改变，可能让屏幕边缘、近处大包围球产生更多细化需求。

Metallic 的仿射缩放上界也包含剪切等保守处理。不能直接照抄较宽松公式作为性能修复；应先统一像素单位，再对边缘、近裁剪、非均匀缩放的实际屏幕误差做验证。参考的窗口/渲染倍率换算与 Metallic 的渲染高度定义不同，1.0 和 1.5 不能直接比较质量。

两边都使用 mmap，Metallic 并非每页重新打开文件。差别是 Metallic 页任务会把 payload 复制到独立结果 vector，再交给 staging；参考能直接复制或解压到批次 staging。这里有可消除的拷贝/分配开销，但尚无证据证明它就是当前主瓶颈。

历史 2026-09-13 数据布局审计显示，Metallic 全 LOD 解码 payload 为 56.340 GiB，参考保留位置的 group 数据为 44.106 GiB；Metallic 使用 16 字节位置，参考为 12 字节。它描述离线数据布局，不代表本次实际传输字节、当前工作集或总 VRAM。参考 RT 路径还可将构建所需位置放在临时空间，Metallic 的光栅主路径需要持续保留位置，显存数字不能直接等价比较。

依据：`Shaders/Modules/GPUDriven/MeshletLodMetric.slang:9`；参考 `shaders/traversal.glsl:204`、`src/renderer.cpp:610`；`Source/Runtime/Scene/MeshletStreamAsset.cpp:4255`、`MeshletStreamPageLoader.cpp:20`；[历史内存布局审计](MiniZorahMemoryComparison.md)。

## 6. 建议实施顺序和验证口径

| 顺序 | 改进 | 验证重点 |
| --- | --- | --- |
| 1 | 分离期望请求与驻留安全 cut，不让正式细节需求被逐级上传串行化 | 相同画质下缩短请求到目标 cut 的轮次；保持完整覆盖、无重叠，控制多余请求 |
| 2 | Streamer 增加基础覆盖就绪状态，首屏前分预算准备根页；必要时对初始视角做有限预热 | 分开统计加载到首屏、首屏到稳定；UI 持续响应，不伪装成总加载加速 |
| 3 | 几何上传与 GPU 页表发布同帧排序；随后推进 CLAS GPU 分配/发布 | 分别测几何和阴影的延迟；检查资源回收、取消提交和多帧在途正确性 |
| 4 | 统一 LOD 像素定义，审计保守误差造成的过量请求，再优化 staging/位置格式 | 在相同屏幕误差下比较目标工作集和传输量；避免用降画质代替提速 |

性能验证应固定相机、输出尺寸和像素误差口径，先用灰色/LOD 显示观察几何，再加入阴影和最终光照；冷 OS 文件缓存、热文件缓存但空 GPU 驻留、暖驻留漫游分别测试。参考 RT 模式与 Metallic 光栅模式的原始用户体验应保留一组，同时另设尽量对齐显示条件的对照组。

当前已有 `MeshletStreamLatency.h` 的 Feedback、Admission、IoQueue、Decode、ReadyToUpload、UploadToDrawable、DemandToDrawable 等分段统计，可先复用。还需要覆盖“细节在理论上已需要、但尚因父级门控没有发出请求”的时间，以及 CLAS ready 时间；仅测已经发出的请求会漏掉关键等待。

建议新增记录：首个完整基础覆盖时间、固定相机目标 cut 的覆盖收敛时间、每帧可见粗级 fallback 像素占比、几何身份/LOD 变化像素占比、上传字节、预取被抑制帧数、重复加载/驱逐、CPU/GPU 帧时 P95/P99。自动曝光、阴影、DLSS 历史都能引起颜色变化，不能只用最终 RGB 帧差当作流送比例。

此前视口改变重建整个 Streamer 会话的问题已在 `ae7b3c5` 修复，见 [视口调整修复与验证](StreamingViewportResize.md)。本报告针对保留驻留会话后仍然存在的机制差异，不把旧的重复初始化再次当作未修复原因。

参考官方设计说明：[vk_lod_clusters streaming](https://github.com/nvpro-samples/vk_lod_clusters/blob/1febfa7694ebdc8f97005a07897029272ca9e85a/docs/streaming.md)。该说明解释了常驻最低 LOD、批量请求和 GPU CLAS 管理；本报告具体配置和实现判断以本次本地源码、cfg 和录像为准。
