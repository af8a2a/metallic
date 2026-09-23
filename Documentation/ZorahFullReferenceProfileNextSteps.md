# ZorahFull：参考 vk_lod_clusters profile 后的优化顺序

日期：2026-09-23。基于用户提供的两张参考程序截图、Metallic 最新三轮固定漫游和本地两套源码。此次为分析与计划，未修改运行时代码，未重新执行性能测试。

## 决策

下一项实施应为 **请求消费增量化与有界准入**。同时把 Shadows CPU 3.1 ms 拆分定位，作为第二项低风险优化入口。GPU 主线继续围绕可见工作量、SW 光栅和 Deferred；BLAS sharing / RT 主可见性适合作为后续架构实验，暂不替代现有主线。

Preflight 已降至约 0.004 ms，继续优化这一阶段的收益很小。

## 截图事实与对比边界

参考截图：Ray tracing、Path tracing 开启、3 bounces、Super Resolution 4x、DLSS-RR Quality，界面显示 render resolution 2203×715；LOD pixel error 1.0，Adaptive error 开启且当前 used=1；BLAS Sharing/Caching 开启，128T/128V，CLAS mantissa drop bits 7，fast trace。

Metallic 基准：输出 1797×660，DLSS SR Quality，内部 1198×440，LOD 1.5 render px，混合光栅、延迟着色和 RT 阴影，10 秒预热、30 秒漫游。截图无法证明参考应用使用相同相机路线、相同驻留历史或相同 GPU/时钟；截图均值不是漫游 P95/P99。参考显示内部像素数约为 Metallic 的 2.99 倍，但不能按该比例归一化几何、RT、DLSS 或整帧时间。

| 参考截图 GPU scope | 两张截图 ms |
|---|---:|
| Frame | 11.594–11.625 |
| Traversal Preparation | 0.041–0.055 |
| Traversal Run | 0.318–0.337 |
| BLAS Build Preparation | 0.042–0.044 |
| BLAS Build | 0.888–0.914 |
| TLAS Build | 0.037 |
| Render | 6.960–6.972 |
| DLSS | 3.201–3.252 |
| HiZ | 0.011–0.018 |

Render + DLSS 约占参考整帧的 88%。参考 CPU Frame 11.648–11.681 ms 包含整帧节奏，不能当成纯 CPU 计算工作量；Stream Begin CPU 0.007–0.008 ms 也不代表流送成本为零，部分工作在其他 GPU scopes / 队列。

参考 traversal：42.3 K tasks、466 K clusters、460 BLAS builds。不能用它们与我们的 active groups、每次新建 BLAS 引用计数直接相除；统计口径不同。

## Metallic 当前瓶颈

来自 full-readiness-cache-0923 三轮。GPU 行是 timestamp 区间，CPU 行是 host scope；不把父子区间相加，不将 CPU 与 GPU 时间相加推算整帧。Stream traversal 包含 RTAS 构建，不能直接对比参考的 Traversal Run。

| Metallic 分项 | 三轮均值 ms | 含义 |
|---|---:|---|
| 整帧 wall time | 21.03–22.02 | P95 27.26–30.12，4204 帧中 26 帧 >33.33 ms |
| GPU graph envelope | 19.81–20.31 | GPU 区间，可能包含同步空隙 |
| GPU VBuffer | 13.11–13.32 | 包含遍历与 early/late 光栅 |
| GPU early SW | 6.49–6.61 | GPU 优化主目标之一；慢帧条件均值约 8.5–9.6 ms |
| GPU Deferred | 5.56–5.80 | 需要内核/依赖细分后确定带宽、采样或其他归属 |
| GPU Stream traversal | 2.55–2.59 | 不能与参考纯 traversal 一一对应 |
| GPU BLAS build | 0.161–0.178 | 已有 cut 复用，当前时间较小 |
| GPU TLAS build | 0.332–0.355 | 与参考有明显差距，但绝对收益较小 |
| CPU Consume requests | 5.58–5.98 | P95 7.75–9.58，优先压缩重复工作 |
| CPU Shadows | 3.07–3.15 | GPU 同 scope 仅 0.044–0.047；先定位 host 成本 |
| CPU Texture streaming | 1.28–1.37 | 主要为 migration；与 Deferred 父 scope 不重复相加 |

每帧平均约 13.54 K 请求、62.1 K 驻留页、13.46 K 预算重试抑制，但实际上传仅约 21 页/帧。说明分配失败后的昂贵重试已抑制，**失败前后的请求去重、排序、逐项调用和驻留状态刷新仍在大量重复**。计数不能证明这 13.46 K 条目都可直接丢弃；需求保活、优先级变化与未来重试仍需保存。

## 源码定位

1. Metallic MeshletStreamResidency.cpp: consumeGpuRequests 每批 clear/reserve unordered_map requestMarks_，逐请求验证和合并；consumeReadyRequestTasks 再刷新整批优先级、stable_sort，逐条 requestPage。Update resident demand 遍历全部 residentPages_；beginFrame 的 latency pending 也全量巡检。
2. Metallic 已有 GPU requestStreamPageLoad 的同帧去重与 GPU unused 列表生成。不能把计划描述为“从零把去重/冷页筛选搬到 GPU”。真正缺口是 CPU 二次加工、跨帧无变化请求，以及反馈只有当帧 unused 而没有足够的年龄/变化信息。
3. vk_lod_clusters scene_streaming.cpp: handleCompletedRequest 消费已完成的 GPU 请求、检查已有 resident group 并受传输/内存容量限制；cmdPostTraversal 进行 GPU age filter。renderer_raytrace_clusters_lod.cpp 在 GPU 上准备 cluster 引用、间接 BLAS build；sharing/caching 可跳过一部分实例遍历。
4. Metallic ScreenSpaceShadowPass.cpp execute 每帧构造灯光记录并调用 shadows_.record；目前只有父 scope，尚不能把 3.1 ms 归因于灯光构造、资源准备或驱动调用。需用子计时验证后再决定缓存/批处理。
5. BLAS overflowCount 是多原因聚合：非法 group、每实例/全局引用容量、插入阶段 CLAS 状态变化、插入越界等。最新最高 12425，不能直接说是 12425 个不同实例失败，也不能把较低 BLAS 耗时解释为质量等价下更快。

## 驻留差异

参考 UI：Textures 4 GiB；Geometry actual 1.36 GiB / reserved 1.37 GiB；CLAS actual 1.23 GiB / reserved 2.63 GiB；BLAS 25.1 MiB / 55.1 MiB；TLAS 2.86 MiB。图表悬浮值与表格瞬时值略不同，不混为同一采样点。

Metallic 最新路线平均几何约 3.49 GiB，预算 3.5 GiB；CLAS 约 1.69 GiB；纹理约 245–246 MiB。这组纹理驻留远小于参考的 4 GiB，不能按“完整材质已开启”宣称纹理细节等价；后续相同相机应同时核对请求 mip、实际 mip、normal/alpha 可用性和图像。

本地参考 scene_streaming.cpp 明确使用 splitPositions = m_requiresClas：RT 路径只把位置写入 CLAS 构建临时区，常驻 group 不含位置。CHANGELOG 的 2026-8-3 项说明后续使用 ray-tracing position fetch。Metallic HW/SW raster 必须继续读取顶点，不能直接照搬移除常驻位置。近期应先分解位置、法线/UV、索引、页填充与缓存页字节，再做属性压缩/布局实验；不要把 Geometry 直接减到 1.36 GiB 当成优化。

## 实施顺序与验收

### R1：请求消费增量化与有界准入（下一项）

- 将每批节点哈希去重改成可复用的紧凑页索引/epoch 标记；按总 pageCount 计算额外 CPU 内存，必要时分块，不无界展开大 PageEntry。保持 demand 覆盖 prefetch、max benefit 合并和异常反馈检查。
- 保留根页/当前需求保活；把 already resident、in-flight、deferred、可立即准入分开。只对可执行候选做有界优先选择；预算被阻塞的尾部维持 pending 状态和老化，容量/冷页/完成事件变化时恢复调度。
- 用确定性优先级和 aging 防饥饿。Top-K 不能按页数机械截断：不同页面字节数和碎片条件不同，应支持小页补位及有界追加选择。
- 降低 62 K 驻留全表与 latency pending 巡检：先复用反馈 epoch、touched 集合/到期队列，第二阶段才调整 GPU 反馈协议为年龄或状态变化。延迟反馈、新页、截断列表、取消上传和页复用都必须保守保护。
- 目标（待实测）：Consume requests 均值 <2.5 ms、P95 <4 ms；同路线请求尾延迟、质量收敛与回访缺页不恶化，吞吐不因饥饿下降。不能把 CPU 节省直接减到整帧上。

### R2：Shadows / 纹理发布 CPU 尾部

- Shadows 子计时覆盖灯光记录、资源/参数准备、trace/denoise 录制、输出发布；若确认是重复构建，按场景/灯光/纹理资源 generation 缓存，保留生命周期失效。
- 纹理 Prepare images and staging 与 Publish texture generation 分别均值约 0.68–0.72 / 0.46–0.51 ms，P95 可达约 2.6 / 3.1 ms。检查对象/描述符重建范围与批次上限，让单帧提交有界并保持升级公平性。
- 验收看分项 P95 和 >33.33 ms 帧比例，不以延迟必需纹理或关闭阴影换分数。

### R3：GPU 工作量与执行成本

- 新增参考截图相机的配对短测，同时保留现有路线；匹配 render extent、FOV、LOD 与几何/纹理收敛条件。一次做稳定状态、一段做漫游；不要重新以旧冻结结果代替当前漫游。
- 采集 active cut、候选/存活/early-late 重试数、SW 三角形、bbox/覆盖/原子尝试数。先判定是否过细、重复展开或遮挡拒绝太晚；保持 1.5 px 质量约束。
- SW 仍是单项大头；局部分桶之前实测退化，暂不扩大。只在新计数支持时选择三角形剔除、队列组织或扫描改动；图像仍核对覆盖、深度和稳定 ID。
- Deferred 5.6–5.8 ms 是另一条独立 GPU 主线：细分可见性解析/属性读取/纹理采样/光照及依赖后选择优化。现有 interval 不能证明全是材质 shader 算术。

### R4：BLAS 质量诊断与共享架构

- 在性能结论前拆分 overflow 原因，统计动态/缓存/根回退实例覆盖及影响像素；当前很低的 build count/time 可能包含回退效应。
- 现有 exact-cut reuse 继续保留。跨实例 sharing、离散 LOD cache 在质量与二次射线覆盖符合要求时再扩展。优先量化 TLAS 输入数量/重建范围，再判断约 0.35 ms 是否值得先改。
- RT 主可见性作为可选实验：复用 CLAS/TLAS 输出一致的 visibility/depth，再接已有着色，隔离主射线与混合光栅成本；要求 RTAS 覆盖充分、MASK/双面/材质属性与几何误差一致。参考 7 ms Render 包含其整套 tracing，不等于我们切换后可得 7 ms，也不作为当前 30 fps 收敛的前置重写。

## 证据

- 用户截图：codex-clipboard-fe44f335-f383-4a9b-9496-51da0533188e.png、codex-clipboard-bbcf0e0a-544d-416a-b65d-e16eebe0eb82.png。
- Metallic：ZorahFullReadinessCache.md / Result.json；build-release/full-readiness-cache-0923/run1..3/Summary.json、Frames.jsonl、SlowFrames.md。
- 局部分桶旧对照：ZorahFullLocalWorkBins.md（复用不分桶 7.317 ms、分桶 8.215 ms，仅当时冻结状态）。
- 本地参考：E:/vk_lod_clusters/src/scene_streaming.cpp，renderer_raytrace_clusters_lod.cpp，CHANGELOG.md。以本地实现为准，截图未提供对应 commit。
- 官方 BLAS sharing/caching 设计：https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/blas_sharing.md 。官方区分跨实例共享与跨帧离散 LOD 缓存，并指出共享更高细节可能增加 trace 成本；不是无条件收益。
