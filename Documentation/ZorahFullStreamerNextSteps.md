# ZorahFull Streamer 下一步优化方案

2026-09-23。依据刚完成的 Reflex On 两轮 Full 漫游，以及本地 `E:/vk_lod_clusters`（HEAD 1febfa7）的实现。Metallic HEAD 9c733fb94，采样二进制另含帧起始诊断计时。本轮只分析并制定计划，没有改动运行时行为，也没有新增性能跑测。

建议顺序：**低开销延迟跟踪 → 驻留冷热状态增量化 → 跨帧请求调度 → 纹理迁移发布 → GPU 遍历工作量**。目标是让 CPU 工作与本帧状态变化、实际准入数量相关，减少对全部待处理请求和驻留页的反复遍历。

## 当前基准

条件：1797×660 输出，1198×440 渲染，DLSS Quality，LOD 1.5 px，预热 10 秒、固定漫游 30 秒。沿用两轮 Reflex On；不混入 Off 或验证层测试。

| CPU scope | 两轮均值 ms | 两轮 P95 ms | 判断 |
|---|---:|---:|---|
| Stream Begin | 6.708 / 6.828 | 10.202 / 9.743 | 主要 CPU 目标 |
| Consume requests | 5.401 / 5.435 | 8.221 / 7.802 | 嵌套于上项 |
| Reset / latency tracking | 0.872 / 0.918 | 1.472 / 1.460 | 扫 pending 跟踪表并查询页状态 |
| Track merged demand | 1.403 / 1.332 | 2.491 / 2.205 | 逐请求查表、读时钟、更新延迟记录 |
| Update resident demand | 1.023 / 1.071 | 1.455 / 1.452 | 每批反馈刷新全部驻留页 |
| Refresh request priorities | 1.020 / 1.020 | 1.358 / 1.305 | 重新扫描请求并读取页/资产元数据 |
| Filter blocked admission | 0.310 / 0.338 | 0.487 / 0.492 | 整批检查预算/IO 阻塞 |
| Prepare admission heap | 0.125 / 0.122 | 0.154 / 0.153 | 已不值得优先改写排序 |
| Admit demand / prefetch | 1.229 / 1.255 | 2.649 / 2.692 | 含分配、回收及循环内过滤/重建堆 |
| Texture streaming | 1.471 / 1.419 | 3.969 / 3.640 | 独立于几何 Stream Begin |

父子 scope 不相加。`Reset / latency tracking + Track merged demand` 是不重叠区间，合计约 **2.25–2.27 ms/帧**，约占 Stream Begin 的三分之一；其中 Reset 区间还含少量帧状态清理，需对照确认纯计数开销。

每帧平均约 **1.356–1.357 万请求、6.208–6.209 万驻留页**，实际调用准入只有 **42–44 次**；约 1.348 万次预算重试被抑制。不是这些页都执行了加载，而是仍付出多遍扫描与跟踪成本。当前重复合并计数仅 1.6–4.6/帧，不能再把去重当作主要问题。上传约 **987–993 页/秒**，无加载失败或请求溢出。

## 与参考实现的实际差异

| 方面 | vk_lod_clusters | Metallic 与优化含义 |
|---|---|---|
| 驻留年龄 | 遍历在 GPU 更新使用状态，age-filter 输出超过阈值的冷组 | GPU 输出本帧 unused，CPU 再扫描全部驻留页推导冷热与年龄；这是最直接可借鉴的结构差异 |
| 请求 | GPU 用帧标识去重，subgroup 批量追加；CPU 校验已驻留状态及容量 | 已有 GPU 帧内去重、CPU 廉价合并；主要额外成本是优先级、诊断跟踪与阻塞候选反复处理 |
| 生命周期 | request/storage/update 有界队列，完成后回收；仅消费最新已完成请求 | Metallic 已有类似机制与延迟释放，不应作为新功能再实现一遍 |
| CPU 存储 | geometry-group 到 resident-ID 的哈希索引，resident 数据放连续数组 | 参考也使用 unordered_map；关键不是“禁止哈希”，而是避免每帧多次探测和全表刷新 |
| 纹理 | init 阶段按预算选 base mip，批量异步上传；描述符集中建立 | Metallic 有运行时反馈、细化、降级与不可变代际发布，工作范围不同，不能直接套用其近零运行时纹理成本 |

参考源码：[帧首任务处理](E:/vk_lod_clusters/src/scene_streaming.cpp:340)、[GPU 年龄过滤](E:/vk_lod_clusters/shaders/streaming.glsl:6)、[遍历请求](E:/vk_lod_clusters/shaders/traversal_run.comp.glsl:300)、[纹理预算加载](E:/vk_lod_clusters/src/scene_textures.cpp:656)。本地参考并非重新下载的上游快照。

## S1：先降低延迟观测自身的成本

当前默认开启 measurePageLatency。`beginFrame` 遍历 pending 并访问 pages_，`Track merged demand` 又对每个请求查 pages_、pending 哈希表并读取时钟。

- 先以同路线开/关 measurePageLatency 做诊断对照，确认可归因成本；关闭只用于测量，不作为最终优化。
- 每批反馈读取一次时间，传给所有请求；保留 sourceFrame→源时间映射，明确批内统一时间的精度。
- 跟踪记录使用按需分块索引或稳定槽位，合并已有请求查询结果；不要为全部 Full 页面分配庞大稠密对象。
- 过期清理改为到期桶/时间轮，带 request 生命周期代号，避免每帧扫描 pending。重复需求更新、prefetch→demand、取消/重试、已在途页面都必须保持原来的统计语义。
- 不得为了降低计数成本，仅跟踪成功准入的请求；预算阻塞的尾延迟仍需可见。

首阶段验收目标：上述两个区间合计均值 **≤0.5 ms**（目标，不是已实现收益）；核对延迟样本数、abandoned、pending 和 demand→drawable 分布，不能通过漏记实现达标。

入口：[beginFrame](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:378)、[request 计时](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamLatency.h:100)、[合并后跟踪](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:939)。

## S2：驻留冷热状态改为增量维护

直接目标是约 1.02–1.07 ms/帧的 resident demand 全表刷新，以及后续冷候选扫描。

先保留现有反馈协议，用前后 unused 集合的变化更新热→冷、冷→热转换，维护连续冷页集合与到期队列。不能简单跳过热页 lastUsedFrame 更新：需要把“持续热”表达为 epoch/默认状态，转冷时再确定时间，或由 GPU 返回 last-use 元数据。

后续再向参考靠拢：GPU 维护 age/last-use，返回到期冷候选及必要的回热/使用代际信息。CPU 只验证候选并安排联合几何/CLAS 回收。只返回冷候选而完全丢掉回热信息是不完整方案，可能导致陈旧反馈卸载已经重新被使用的页面。

必须保留 sourceFrame、页驻留代际、fallback/BLAS/在途保护；反馈溢出、丢失或不完整时走保守保护/重新同步。旧候选不得跨卸载再加载复用，实际内存仍在 GPU 完成后释放。当前的几何/CLAS 压力阈值、冷保留期限保持不变，以便先验证结构变化。

目标：常态 CPU 工作量从 O(全部驻留页) 降为 O(冷热变化 + 到期候选)，Update resident demand 均值 **≤0.3 ms**。增加 visited/changed/due/revived/staleRejected 计数，静止驻留稳定时不应持续刷新 6 万页。

入口：[驻留全表刷新](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:979)、[冷候选扫描](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:1561)、[GPU unused 输出](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:1836)。

## S3：跨帧保留请求状态，容量变化时恢复准入

每帧 1.35 万请求里只有约 0.3% 执行准入。现有预算门控已有效避免逐项重复 requestPage，但候选元数据、优先级和阻塞条件仍反复扫描。

- 将合并结果、payloadBytes、prefetch LOD、页状态查询结果缓存到稳定请求记录，减少 Track / Refresh / requestPage 之间重复查表。
- 按尺寸类别与阻塞原因维护等待集合：geometry、CLAS、IO、页槽。容量/完成事件、下一冷页到期时重新激活相关候选，不采用固定整批冷却期。
- 相机产生新高收益 demand、prefetch 升级为 demand 时必须立即刷新；小页不能被大页拒绝阻塞，年龄提升必须有界，防止饥饿。
- 使用带代际的持久候选堆或桶队列，合并最新需求并限制每帧维护工作；优先保证公平性与质量，再考虑把候选裁剪移到 GPU。
- 先细分现有 Admit demand / prefetch 中 requestPage、回收、循环内 filter/make_heap 的时间，不能把整个 1.2 ms 归给堆操作。

与 S1/S2 合并验收，目标是 Consume requests 均值 **<2.5 ms、P95 <4 ms**；沿用此前未达成的目标。上传吞吐维持本轮约 990 页/秒量级，需求尾延迟、预算与切换保真不退化。阈值只是验收目标，不能预先承诺整帧改善同等毫秒数。

入口：[刷新候选](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:579)、[准入过滤与堆](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:644)。

## S4：纹理迁移发布与资源准备

Publish texture generation 每次约 2.31–2.40 ms，但仅部分帧发生，摊到全部帧为 **0.59–0.62 ms**。Prepare images and staging 每次约 1.66–1.71 ms，摊到全部帧 **0.68–0.71 ms**。不能把每次发生的均值误报为每帧开销。

发布 scope 包含替换图片、引用计数、全量 generation/snapshot 重建，以及 migration.reset() 销毁上传相关资源。先拆出这些成本，再决定优化对象。

- 若全量代际复制占主导，采用分块不可变快照，仅替换脏块；继续让正在执行的帧持有旧资源。已有 Shadows/Deferred 消费侧代际缓存保留，不重复做同一优化。
- 若资源销毁占主导，转为有界的完成后回收，并复用 command pool、上传 staging 和可复用对象。
- 准备图片的 1 ms 预算允许单个 image 超时；拆分原生分配、创建 view、staging 分配、CPU copy，再决定预分配/池化或 worker 准备的安全边界。
- 纹理动态细化与降级保持，不能照搬参考的加载期固定 mip 方案来换取更低运行时成本。

入口：[全量代际快照](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1413)、[发布和销毁](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1512)。

## S5：GPU 遍历另立工作量对照

当前 Stream traversal GPU 均值约 2.60 ms，但它包含 CLAS、BLAS、TLAS。不能与参考截图的 Traversal Run 0.32 ms 直接相除。当前子区间包括 Detail demand 0.438、LOD frontier 0.426、mask 0.185、prefix 0.184、emit 0.136 ms；CLAS build 0.210、BLAS build 0.191、TLAS build 0.341 ms。

先导出 visited nodes、unique demand pages、重复实例需求、有效 frontier 和 prefix 实际输入量，再做固定 camera/cut/residency 对照。确认热区后再推进唯一几何需求合并、稀疏活跃实例处理或 prefix 并行化。当前 CLAS CPU completion/expiry 仅约 0.07 ms，不优先重写整个 CLAS 分配器。

## 验收约束

所有调度状态仍归 StreamerSubsystem；RenderPass 只提供需求/反馈和消费已准备资源。每阶段分别构建与生命周期回归，再使用相同 Full 路线多轮采样。压力测试覆盖预算耗尽后释放、小页准入、快速往返、反馈溢出、页 ID 再利用、上传取消与场景切换。

性能同时记录 CPU 工作量、请求 P95/P99、上传/回收、显存峰值、图像/LOD 收敛、整帧 P95/P99、帧槽+Reflex Sleep。上一轮已证明 CPU 等待会迁移，不能用单个 scope 下降来宣称稳定 30 fps。

[结构化基准](E:/metallic/Documentation/ZorahFullStreamerNextStepsResult.json)；原始数据位于 `build-release/full-reflex-begin-on-0923/run1` 与 `build-release/full-reflex-begin-on-repeat-0923/run1`。
