# ZorahFull：压缩实验归档后的优化路线（2026-09-29）

## 结论与证据边界

下一批建议做 **B0：BLAS 准入诊断 + B1：按实际 cut 紧凑分配 BLAS 引用**，随后做逐实例 BLAS 复用。降低显存的独立候选是紧凑 visible records；暂不继续位置/N/T 编码，也不直接切换主可见性渲染路径。

本次核查源码和已有原始采样，没有重新运行 GPU 测试。当前 Metallic 为 `modernization@97ff18d33`；参考本地 `E:/vk_lod_clusters@1febfa7694ebdc8f97005a07897029272ca9e85a`。压缩实验已在 `modernization_/meshlet-compression`，其中的联合几何/CLAS 上限、细分 BLAS overflow 计数不能视为当前主线功能。构建目录可能仍有实验二进制，下一次采样必须重建并记录程序、shader 和配置摘要。

当前主线已有 8M raster candidate 容量、队列溢出的硬件回退、CLAS 实际尺寸搬移与按需分块 backing，以及前序 Streamer 生命周期缓存、增量优先级和 pacing 前维护。不要再次把这些列为待实现优化。

## 1. 当前最值得修正的是 BLAS 容量分配

### 源码差异

Metallic 的 `MeshletStreamRuntime.cpp:1195` 开始按 primitive 的**全部 LOD group** 累加 cluster 数，再按场景实例顺序分配固定引用区间。`planBlasCapacities` 同时受引用数、构建数和 BLAS 字节预算约束；字节预算不足时引用上限和构建上限一起减半。靠后的实例可以始终得到零容量，与该帧实际选中多少 cluster 无关。

`GPUDrivenStreamAsset.slang:3270` 对零容量、超出实例区间或全局区间的实例设置 fallback/overflow。增加驻留页或提高细化质量不能解决这种预分配偏置；直接扩大预算也会放大最坏情况预留。

参考 `vk_lod_clusters/shaders/blas_setup_insertion.comp.glsl:108` 使用遍历产生的 `clusterReferencesCount`，按实际数量累加分配引用空间；随后 `blas_clusters_insert.comp.glsl` 填入当前选择的 CLAS 地址。它的引用缓冲以受限的 render cluster 容量为基础，不向每个实例预留全部 LOD 的引用数。不能脱离参考的遍历容量约束，仅复制原子追加代码。

### 已有主线原始数据

重新读取 `build-release/clas-demand-final-roam/run1/Frames.jsonl` 的 180 条采样，得到：

| 指标 | 观察结果 |
|---|---:|
| 动态 BLAS 构建数最大值 | 219 |
| BLAS cluster 引用数最大值 | 18,781 |
| 聚合 BLAS overflow 最大值 | 12,425 |
| 首条反馈：构建 / 引用 / overflow | 123 / 2,944 / 11,726 |
| 驻留页范围 | 54,160–62,209 |

这些是最近 CLAS 分块实现报告对应的历史主线数据，不是本次运行。反馈有延迟，不能将其与同一 CPU 帧的相机或 pass 时间直接逐帧配对；overflow 是聚合事件计数，不等于去重后的回退实例数。当前计数无法证明全部 overflow 都来自零容量，但固定区间分配提供了明确的可复现机制。

因此，较低的 BLAS 构建耗时可能包含大量实例回退的影响。应先修正准入，再对比构建性能。修复后更完整地构建细化几何，BLAS 时间可能上升，这不自动表示优化失败。

### B0：先恢复独立诊断

- 分开统计：未分配实例区间、实例容量不足、全局引用不足、构建槽不足、非法 group、CLAS 未就绪。计数器按 CPU/Slang/读回布局统一扩展，不整批合入压缩实验。
- 导出实际需求、获准引用数、构建容量、驱动查询的最大单 BLAS cluster 数和总量，以及去重后的动态/回退实例数。
- 区分“合法根回退”和“容量分配造成的回退”，同时记录反馈源帧和发布代际。增加重建原因：cut 变化、CLAS 地址/驻留变化、显式失效。
- 诊断读回单独运行；正常耗时对照关闭强制工作量读回。

### B1：按本帧实际 cut 分配

在 Streamer 所属的 GPU 准备流程中执行：实例选中数量统计 → 有界准入和前缀分配 → 填引用 → 间接 BLAS 构建。移除按全部 LOD 数量分给靠前实例的固定区间。

同时约束总引用、单 BLAS 引用、构建槽和驱动查询对应的存储/scratch 上限。将引用容量与可容纳实例数分别规划，避免引用预算缩小时无条件削减实例覆盖。不要依赖 CPU 同步读回，也不要将加载逻辑放回 renderpass。

超预算时按稳定且可解释的优先级接受完整实例，其余保留有效回退；不能截断一个实例的引用列表后发布残缺 BLAS。主相机之外的阴影/反射几何仍需覆盖。帧内紧凑引用偏移变化不应单独使已缓存的物体空间 BLAS 失效，但 CLAS 地址、几何内容和相关构建语义变化必须失效。

验收首先是正确性：总需求满足所有上限时，不再因场景实例排列而出现零容量拒绝；超预算仅产生可解释的完整回退；取消、重上传、CLAS 退休、在途资源生命周期正常；无越界或失效地址。随后才比较同 cut 的 CPU/GPU 准备和构建耗时。

## 2. BLAS 缓存粒度应从全场景 cut 改为实例

当前 `GPUDrivenStreamAsset.slang:3203` 对 active group 的实例、页、选择 mask、cluster 数逐项精确比较；任意变化设置一个全局 dirty 位，后续重置所有实例并重新准备动态构建。全局 CLAS 发布 revision 变化也会使缓存失效。

参考的 `docs/blas_sharing.md`、`geometry_blas_sharing.comp.glsl` 和 `traversal_init_blas_reuse.comp.glsl` 提供按 geometry 的共享、缓存和兼容 LOD 范围复用，减少重复遍历和构建。

推荐分两步：

1. **B2：逐实例 dirty 与生命周期缓存。** 只更新 cut 或引用 CLAS 代际改变的实例，保留未改变的 BLAS；避免无关页发布触发全场景重建。
2. **B3：相同物体空间 cut 的跨实例共享。** 使用稳定几何身份、选择内容和驻留代际验证；处理 alpha/不透明标志等构建兼容条件。先支持精确相同 cut，再研究参考实现的兼容 LOD 范围共享。

需要测量改变实例数、实际构建数、缓存命中和失效原因。不能以悄悄降低远处/阴影几何精度换取共享命中率。B1 的输入紧凑分配必须与 B2 的持久 BLAS 输出寿命分开设计。

## 3. 显存：紧凑可见记录优先于继续编码实验

当前 Full 的 `maxActiveGroups=1,048,575`。`visibleClusterCapacity()` 为 group 数乘 32，`visibleRecordBuffer` 仍按该容量乘 16 B 分配，约 **512 MiB**。它以 `activeGroupIndex * 32 + localCluster` 寻址，没有随 8M candidate 缓冲一起收缩。

参考渲染器的 `renderClusterInfos` 是受 `maxRenderClusters` 限制的紧凑输出。可借鉴为 **V2：稠密 visible records + 稳定逻辑 ID 映射**，同时覆盖 HW/SW、两遍 HZB、Deferred 解码和调试可视化。

若经测量可采用 8M 条记录，记录本体将从约 512 MiB 降至 128 MiB，毛节省约 **384 MiB**；映射和回退元数据会抵消一部分。这是容量算术，不是实测收益。已有 candidate 峰值不能当作可见记录峰值；先采集各阶段发布量及跨阶段 ID 寿命，不能直接截短旧缓冲，也不能照搬参考的 1M 上限。

CLAS 分块分配已完成。历史报告只确认 backing 峰值减少 256 MiB，最终整卡 NVML 峰值未证明下降。暂不优先做活页搬移整理：先增加空洞、最大连续空闲区、退休占用、空块可释放率，确认碎片实际阻止增长或归还后再做。

另一项独立问题是根 cut 过重：现有审计根页约 2.956 GiB，而几何池为 3.5 GiB，细化空间仅约 0.544 GiB。继续降低池上限前，应审计 root-heavy primitive 的简化终止条件、材质/UV 边界和保真约束。先做局部 cook 探针，不以删除 terminal branch 或仅保留主相机可见 roots 降低内存。这项工作不依赖继续位置编码实验。

## 4. Streamer CPU：保留增量策略，减少全驻留扫描

`MeshletStreamResidency.cpp:997` 的 `Update resident demand` 仍逐页处理驻留集合。之前的优先级缓存、重复查询消除和 pacing 前维护均已存在，不能重复列作新方案。

参考 `stream_agefilter_groups.comp.glsl` / `streaming.glsl` 在 GPU 对驻留 group 维护 age，并输出到期回收候选。它仍扫描驻留集合，只是扫描位置和 CPU 反馈规模不同，不是零成本的增量算法。

推荐 CPU 端先围绕 unused 状态差异、需求 epoch 和已存在的到期队列更新，测量维护量与“状态改变页数”的相关性。若全量存活更新仍占显著时间，再比较 GPU age/到期反馈。丢失或截断反馈、源帧过期、页面重新驻留时保守保护，回收继续等待 CLAS/BLAS 在途使用完成。

## 5. 不用参考截图直接决定更换渲染路径

已有参考截图为 RT/path tracing、3 bounces、DLSS-RR Quality，渲染尺寸 2203×715，LOD 1 px；约 466K selected clusters，Frame 11.6 ms、Traversal 0.32–0.34 ms、BLAS 0.89–0.91 ms、Render 6.96–6.97 ms。

Metallic 最近内存路线为输出 1797×660、内部 1198×440、DLSS-SR Quality、LOD 1.5 px、混合可见性光栅与 RT 阴影，且相机、材质和选择集合并未全部对齐。参考截图是均值，不能与漫游 P95 相除得出性能差距。

最近 180 帧 CLAS 内存路线带工作量读回，属于诊断运行。其 SW 和 Deferred 仍值得关注，但不能用它的新长帧数字取代正常性能基准；更早的全 HW/1/2/4/8 px 对照还早于多个 shader 改进，也不能直接决定当前默认分流阈值。

完成 B1、统一几何覆盖后，重新分开比较固定 camera/cut/residency 与实际漫游。若 SW 仍是 GPU 主项，再针对当时的三角形 setup、bbox 访问、覆盖率和原子数选择工作量优化。RT 主可见性可以作为后续独立对照，不作为这轮直接替换方案。

## 推进顺序与测量协议

| 顺序 | 工作 | 主要判据 |
|---|---|---|
| 1 | B0 + B1：BLAS 诊断与实际 cut 紧凑引用分配 | 消除固定实例区间偏置，预算内完整覆盖，合法回退可解释 |
| 2 | B2，随后 B3：实例 dirty、精确 cut 共享 | 相同质量下构建量随实际变化实例减少 |
| 3 | V2：紧凑 visible records | 扣除映射后的物理分配减少，ID/两遍光栅/材质解码正确 |
| 4 | Streamer 存活维护增量化；局部 root cook 审计 | CPU 全驻留扫描减少；根表示保持保真并释放细化空间 |
| 5 | 按新 GPU 工作量证据选择 SW 或 RT 对照 | 正常漫游 P95 与 GPU 关键路径改善 |

每阶段保持用户已确定的编辑器视口尺寸、DLSS Quality、LOD 1.5 px、完整材质与阴影；记录相机绝对路线、输出/内部分辨率、cook/shader 缓存状态和显存预算。冻结对照用于定位，固定漫游用于验收，诊断读回与正常计时分开。至少交错重复三轮，检查后台负载及程序退出后的显存；帧 P95 目标为 33.33 ms，并单列长帧数量及原因，避免均值掩盖停顿。

同时报告动态 BLAS 覆盖、fallback、selected clusters、驻留/退休/物理分配，避免通过少构建几何获得表面提速。对齐参考时逐项匹配 camera、LOD 定义、渲染尺寸与材质/阴影设置；不同算法的 pass 只比较职责和工作量，不能将同名范围视为等价。

## 本地证据入口

- [CLAS 按需分配与历史验证](ZorahFullClasDemandAllocation20260929.md)
- [工作缓冲解耦](ZorahFullWorkBufferCapacity20260929.md)
- [pacing 前维护与正常计时边界](ZorahFullPrePacingMaintenance20260925.md)
- [参考截图设置与数据](ZorahFullReferenceProfileNextSteps.md)
- [显存与根 cut 审计](ZorahFullPeakVramPlan20260928.md)
- 原始历史采样：`build-release/clas-demand-final-roam/run1/{Capture.json,Frames.jsonl,Summary.md}`。
- Metallic 实现：`Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp`、`MeshletStreamResidency.cpp`，`Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang`。
- 参考实现：`E:/vk_lod_clusters/shaders/{blas_setup_insertion.comp.glsl,blas_clusters_insert.comp.glsl,geometry_blas_sharing.comp.glsl,traversal_init_blas_reuse.comp.glsl,stream_agefilter_groups.comp.glsl,streaming.glsl}`。
