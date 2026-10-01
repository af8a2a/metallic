# ZorahFull 峰值 VRAM 降低方案（2026-09-28）

基于 `aafa778a8` 的当前实现、Full 预设、本地 V9 cook 目录和 `E:/vk_lod_clusters` 参考源码。用户报告整卡峰值约 14 GB；本次没有重新运行 Full 捕获该峰值，因此下列容量审计不是对 14 GB 的完整实测归因，也不是已实现的节省。

## 决策

优先 **工作缓冲容量解耦 + 生命周期收紧**，随后 **CLAS 按需物理分配**，长期解决 **根页过重**。继续降低纹理分辨率或仅加强冷页回收不是当前首选。

本次静态探针见 `build-release/ZorahFullVramStatic20260928.json`；脚本 `build-release/ProbeZorahVram.py` 只读取 V9 文件目录，不读取几何 payload，也不申请 GPU 内存。

## 已确认的容量结构

| 项目 | 当前数值/行为 | 结论 |
|---|---|---|
| 几何 pool | Full 配置 3.5 GiB；初始化创建完整 device buffer | 页面卸载回收池内范围，不降低此 buffer 的物理分配 |
| CLAS persistent pool | Full 配置 2 GiB；compact pool 初始化创建完整 device buffer | 已有对象实际尺寸压缩，但外层容量仍预分配 |
| Active group capacity | 1,048,575 | 来源为实例 group 总量与配置上限的最小值，不是当前可见量 |
| Visible cluster capacity | 33,554,400 = group capacity × 32 | 使用每组 cluster 数的资产上限，放大多个下游缓冲 |
| VisibleClusterRecord | 536,870,400 B，约 512 MiB | 每记录 16 B |
| Hybrid classification queue | 1,213,201,344 B，约 1.130 GiB | `64 + C×36 + ceil(C/128)×20`；另有小型 arguments、像素与三角形队列 |
| 上述两个 cluster 工作缓冲合计 | 1.630 GiB | 静态 capacity 对应的 buffer payload，不是有效输出量 |
| 根页 | 51,764 页 / 1,154,901 clusters | 当前 cook、实际被实例引用的 primitive 的 terminal cut |
| 根页实例 groups | 512,130 | 不能简单把 active group 上限调到几万 |
| 当前紧凑根页 | **3,173,527,552 B = 2.956 GiB** | 按现行 position/shading 紧凑规则和 256 B 分配对齐重新计算 |
| 材质纹理 | Full 默认 image allocation 预算 512 MiB，128→512 按需细化 | 相比以上大项，继续压缩的整卡收益受限 |

几何 + CLAS 两个池与上述两个工作 buffer 共约 **7.13 GiB 请求容量**。此外还有全量 LOD topology/group/node 数据、BLAS/TLAS、CLAS 构建/搬移 workspace、渲染图附件/历史、SDK 资源和保留分配。不能将未测的剩余约 7 GB 一概称为“泄漏”或“纹理”。主机上传资源只在实际位于 device-local heap 时计入 VRAM。

代码依据：

- `Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp`：几何分配约 884 行；group/cluster capacity 约 966–1048 行；visible records 约 1824 行；terminal cut 约 918–950 行。
- `Source/Runtime/Render/Streamer/MeshletStreamCompactCLASPool.cpp`：315–350 行，完整 capacity storage 与每 queued frame 一个临时 builder。
- `Source/Runtime/Render/VisibilityHybridRasterizer.cpp`：46 行，cluster queue 字节公式。
- `Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json`：Full 各项预算。

历史 P0 数据 `ZorahFullP0Results.json` 的一个阶段中，geometry used 已达约 3.47–3.50 GiB，CLAS used 约 1.68–1.71 GiB。它不是本次峰值实测，但说明不能预设两个池的大部分都空闲。当前根页紧凑布局已启用，不能把再次“启用 compact attributes”算作新收益。

## V0：先取得同帧的 allocation 峰值证据

复用现有 `Device::memoryBudget()`、分域 accounting、`ResourceMemoryInfo` 和 RenderGraph resource 导出，补充统一快照而非另建计数体系：

- 每个资源的 owner、allocation ID、requested/allocated bytes、heap、generation、用途、创建帧、最后引用完成点、retired 状态。
- 每个 pool 的 used、physical committed、virtual capacity/max budget、free bytes、largest free range。
- 分开 geometry、CLAS persistent、build workspace、RTAS、material textures、work queues、frame attachments/history、upload/readback、其他资源。
- 同帧记录 VMA allocationBytes/blockBytes、driver heap usage/budget 和 NVML 整卡使用量；allocation 域与 block/heap 是不同口径，不能叠加。reservation/safety 是规划额度，不是实际占用。
- 覆盖加载根页、首次启用 DLSS、首次细化、稳定漫游、转身、Mini↔Full、resize 后旧代际退休。

输出总峰值帧的 Top 20 资源与 owner；不能把不同帧各域的最大值相加冒充峰值。跨进程/driver/SDK 的不可归属部分明确标记。

## V1：解耦可见输出容量，优先处理约 1.63 GiB 工作区

保留必要的 group 遍历容量，但为 **可见 cluster records、分类队列和 BLAS references** 分别定义容量与增长策略，避免一律沿用 `maxActiveGroups × maxPageClusters`。BLAS reference 当前还会依据驱动 size query 与 maxBlasBytes 下调，不能直接将理论 C×8 算作它已分配的实际值。

建议流程：

1. 采集 traversal candidates、cull 后 visible、HW/SW queue、late-pass append、BLAS references 的高水位；计数包括两遍剔除，不能只看 early 或 HW。
2. 候选容量与最终可见容量分离，优先把有必要的剔除前移；从候选处理到输出 compaction 使用明确的有界队列。
3. 以真实峰值和余量决定 start/grow/max；容量不足必须能检测并安全重试或保留完整可绘制 cut，不允许 clamp 计数丢几何。
4. 增长使用新 generation，完成后发布并退休旧资源；检查旧新共存峰值。低水位持续一段时间才考虑缩容，避免 resize 抖动。

**条件算例**：若同条件实测证明 4,194,304 个输出 cluster 能覆盖工作集及余量，两块 buffer 可从 1.630 GiB 降至约 208.6 MiB，理论释放约 **1.426 GiB**。这不是建议直接写死 4M：当前根 cut 有大量实例 cluster，实际可见需求必须先测；安全容量可能更高。若无法减少最大合法容量，可研究稀疏物理后备，但同样必须保护写入前已映射范围。

此项不用重 cook、纹理质量可不变，且收益上限比 NTC 替换 512 MiB 纹理更大；因此列为第一实现候选。

## V1b：收紧一次性与重叠生命周期

- **根 fallback BLAS 构建资源**：scratch/reference/buildInfo/destination 当前随 runtime 保留，主要用于首次构建；完成后可研究单独退休。当前 `ready()` 和 `cmdBuildFallbackBlas()` 都检查这些指针，不能直接 reset。必须拆分 loading 与 ready 状态，保留 fallback AS/address；支持取消重试和必要重建时再分配。
- **CLAS workspace**：目前 queued frame 个 builder 各自保留固定 maxBuild 的 temporary CLAS/scratch。让 build batch 数量与实际并发解耦，workspace 按在途 batch 借用；按需求建立较小批次。减少批次可能延长首帧/请求尾延迟，不能只测静止后的显存。
- **场景切换/resize**：保留完成点所需的旧资源，但在新场景 admission 前释放已无消费者的缓存，避免新旧完整场景长时间重叠。保留平滑切换模式；低预算时分阶段切换，不能简单删除仍被 GPU 使用的旧帧。
- **上传**：已有三批有界上传和纯 TransferSource 优先 Host 的实现，不重复当作新优化。只对当前确定位于 local heap 的 staging 或多余滞留批次进一步处理；不要为了少量 staging 显存制造每批 queue-idle。

这些项的收益取决于 V0 owner 数据；在测到 scratch/retired 大项前不承诺具体 MiB。

## V2：CLAS start/grow/max 分离，按物理块增长

参考本地 `E:/vk_lod_clusters/src/scene_streaming_utils.cpp:189` 附近，CLAS 分离 maximum、initial 和 growth，使用 `LargeBuffer`；配置默认 start/grow 128 MiB。官方变更记录明确其 CLAS allocator 使用 sparse buffer 根据需求增长。[官方 changelog](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/CHANGELOG.md)

Metallic 当前 RHI 没有对应 sparse buffer bind 路径，可选择：

- 地址稳定的 sparse buffer：保留虚拟地址范围，按块提交物理内存。增加 feature/queue 能力探测、bindSparse 同步和预算记账。空块仅在所有使用该地址的 AS/帧完成且无活对象后解绑。
- 分段 CLAS arena：每段固定物理 buffer，CLAS 本身已有地址表；让 allocation 携带 segment ID，地址发布使用相应 segment base。无需整个旧池搬到新池，但现有单一 `storageBuffer` 及 RHI/graph 资源声明都要适配。

先增长，再实现冷空块归还。对象空闲分散时可用有界 MOVE 整理，但必须重建/更新引用这些 CLAS 的 BLAS，完成后才退休旧地址。当前 roots invalidation 使场景 readiness 失效，不能随意搬移已被 fallback BLAS 引用的 root CLAS。初期可以 root 专区固定、动态区分块。

不要采用“申请更大连续池→全量拷贝/MOVE→释放旧池”作为降低峰值的默认策略：迁移时两池共存，反而提高峰值。128 MiB 只是参考的块粒度；Full 初始容量需覆盖实际 root CLAS 与在途工作，不能照搬 128 MiB 即可完成 Full 首帧。

预计收益：`原 2 GiB − 实际需要的对齐物理块 − 新增管理开销`。历史 CLAS 工作集已接近 1.7 GiB，忙碌漫游时可能只节省几百 MiB；冷/加载阶段可能更多，需要测量。

## V3：根 cut 与几何物理池联合改进

当前根 cut 即使压缩后也要 2.956 GiB，3.5 GiB 池只剩约 0.544 GiB 细化余量。先减根再减池，顺序不可倒置。

离线审计根字节 Top primitives、简化停止原因、材质/UV seam、透明覆盖、跨材质几何复用。当前前三个 root-heavy primitives 5152/4108/2624 合计约 367.8 MiB，适合先做局部 cook 探针。区分“几何可共享”和“材质/alpha/TBN 语义必须保留”。禁止通过删 terminal branch、破坏纹理 seam 或仅驻留主相机可见 roots 达成数字下降；阴影/反射与视角跳变也需要完整回退表示。

研究独立粗代理、修正过严/错误的简化约束、几何与材质绑定解耦、误差受控的位置/UV 编码。当前 normal/tangent 已紧凑，不重复计算其收益。若 position 做量化，CLAS 输入可能需要解码临时空间，必须纳入净收益。

几何物理池随后改为分块 committed allocation。参考的几何 storage 使用最大 128 MiB 分配块。[参考项目](https://github.com/nvpro-samples/vk_lod_clusters)

目前 shader 以单 page-buffer 基址加 32 位 offset 寻址；分段需要 page table 的地址/segment 语义升级，sparse 则需要 RHI 能力。只调整 CPU allocator 无法释放已完整分配的 3.5 GiB buffer。

## V4：帧资源复用及 SDK/缓存治理

V0 若显示 frame/transient/history 很大，再处理：

- 按真实 GPU 生命周期对不重叠资源使用同一 backing allocation；binding 别名、同 buffer 不同范围不等于物理 memory aliasing。
- 跨 queue、跨 frame、历史读写和 editor debug pin 的存活区间必须纳入分析；不能按 CPU 录制顺序判定可复用。
- DLSS/NRD 功能切换、分辨率变化后及时释放旧 feature/历史，保持当前 DLSS Quality 与效果不变。SDK 内部分配从调用前后 heap 差值观察，不伪装成可精确归属的 RHI allocation。
- VMA block slack 与真正活跃资源分开；空块归还/defrag 只在确有收益且支持地址修复时做。CLAS/AS 指针不能用普通 memcpy 搬迁。

VMA 支持资源别名，但要求兼容内存类型/对齐，并保证 GPU 使用区间不重叠与必要同步。[官方 aliasing 文档](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/resource_aliasing.html)

## 验收与推进建议

首个实施批次建议 **V0 + V1**：取得同帧 Top 20 分配与 cluster 高水位，实施独立输出容量/有界增长；同时把 V1b 中确认的一次性资源按完成点退休。第二批 V2，V3 用独立小资产探针推进。

所有比较保持相机轨迹、输出尺寸、DLSS Quality、LOD 1.5 px、材质/阴影、帧槽和 warmup 相同。分别测冷启动、稳定漫游及切换峰值，不混用首次 shader 编译与热缓存数据。报告至少包括：

- allocation / block / device heap / NVML 四个口径；同帧峰值与 steady P95；各池 used/physical/max。
- visible/candidate/HW/SW/BLAS 高水位、所有 overflow、降级/回退、页反复装卸、细节收敛和请求尾延迟。
- 整帧 P95/P99、加载时间、GPU pass 时间、验证层结果、转身/近墙/MASK/阴影图像。
- 新旧资源共存、取消提交、切换、resize、预算拒绝和恢复。

不把几何池直接调到 2 GiB，不以关闭阴影/DLSS或提高 LOD error 作为等条件优化；不把减少帧槽产生的吞吐/延迟变化隐藏在显存收益内。NTC 保留为高精度纹理路线，当前优先级低于容量放大和根集问题。
