# ZorahFull：逐实例 dirty 与 BLAS 复用

日期：2026-09-29。接续实际选中 cut 的 BLAS 引用分配；未修改 cook、LOD 质量、几何/CLAS 预算或 render pass 的流送职责。

## 结果

默认启用逐实例复用。Full 固定路线三次独立进程对照，BLAS build 的运行均值中位数 **8.807 → 0.413 ms（降低 95.3%）**；整帧均值中位数 **41.299 → 32.731 ms（降低 20.7%）**。这两种指标分别表示 pass 成本和实际帧时间，不相互替代。

每帧平均构建实例由约 10,862 降为 577，另有约 10,286 个实例直接复用。仍参与动态 BLAS 的几何引用平均为约 437.7 万和 438.0 万，差异约 0.07%；收益没有伴随动态覆盖数量下降。六次运行的 BLAS overflow、引用/构建槽/单 BLAS/存储拒绝、非法 group、页面 IO 失败及请求溢出均为 0。

**尚未达到持续 30 fps**：复用侧三次整帧 P95 为 53.756、49.837、54.327 ms，且仍有未就绪 CLAS 的完整实例 fallback。每次 180 帧路线发生两次有界存储重排；并非所有帧都只重建局部实例。

## 实现

### 失效范围限定到实例

在 GPU 上按实例保存上一帧 active group 区间，逐项比较 `{pageIndex, clusterSelectionMask, clusterCount, publicationGeneration}`。比较的是实例内部次序，其他实例插入/移除造成的紧凑数组偏移变化不会使当前实例失效。先完成全部旧 key 读取，再提交新 key，避免当前帧写入覆盖另一个实例仍需读取的历史区间。

未变的实例复用已有 BLAS；cut、所引用 CLAS 的发布代际、首次准入或缓存失效才要求重建。物体空间 BLAS 不因世界变换单独变化而失效，变换仍交给 TLAS。没有使用有碰撞风险的 cut 哈希。

CLAS 页表增加 32 位发布代际，legacy 和 compact pool 均写入；退休/重传/地址变化通过对应页面发布触发失效。全局 revision 不再使所有实例 dirty。全局 revision 的高 32 位改变时保守清空缓存，防止代际回绕误命中。命令取消继续通过 submission transaction 使缓存失效。

### 持久化存储与显式目标地址

原来的 implicit packed batch 会重新排列输出，不能只重建一部分并保留其他实例。本次切换为 explicit destination：

- 初始化时查询按 cluster 数分档的驱动 BLAS 尺寸，并按设备要求对齐；不猜测驱动内部结构大小。
- 每实例保留 `storageOffset/storageCapacity`，尺寸足够时原地重建或直接复用。需要增长时从同一有界存储池追加区间。
- GPU 前缀扫描计算本帧增长需求及全部存活实例紧凑需求。池内空间不足时，在原池内重新排布并重建所有本帧准入实例，确保被覆盖的旧 BLAS 不会继续作为缓存使用。
- 不额外分配第二个 BLAS arena，不同步读回 CPU，不增加运行时搬移调用。无法容纳的实例整体 fallback，禁止发布部分或越界 BLAS。
- 只为 dirty 实例写 cluster references 和 build records；TLAS 直接引用实例的持久化地址，不依赖临时 batch index。

这是有界追加/重排实现，尚不是逐实例 free-list 分配器。离开 active cut 的实例及增长后的旧区间在下次重排回收；因此长漫游仍可能触发全批重建尖峰。未做跨实例相同 cut 的 BLAS 共享，也未缩小现有 arena/scratch/reference 容量预算。

### 可观测性

新增 dirty/reused/admitted 实例、live/rebuilt cluster references、storage rejected、arena used/repack 和 publication invalidation 计数。`blasClusterReferences` 表示本帧重建引用，`blasLiveClusterReferences` 表示全部动态引用；二者不能混用。`blasPublicationInvalidated` 计数单位为 group，不是唯一实例。反馈异步到达，`blasFeedbackFrame` 才是相应 GPU 统计帧，不能把读回所在帧的 CPU 长帧直接归因于一次重排。

现有 scope 保留：`BLAS cut compare` 包含引用前缀阶段，`BLAS setup` 包含存储分配阶段，`BLAS insert` 包含新 key 提交；`BLAS build` 独立计时。JSON 中记录 `blasInstanceReuseEnabled`，runner manifest 记录开关。

可使用 `METALLIC_BLAS_INSTANCE_REUSE=0` 强制每帧重建以做对照；默认启用。这个开关保留新的存储布局和分配器，只关闭复用。

## 回归

Release sample 和 RHI tests 构建成功。开启 Vulkan validation，以下 8 项全部通过（78.3 秒），日志无 Validation Error、VUID 或 DeviceLost：

- `RhiRendering.stream_blas_selected_allocation`：生产 shader 入口，137 个实例、多扫描块/尾块、稀疏 mask、精确容量、地址进位、区间不重叠、容量拒绝。连续 GPU 帧验证静止复用、无关发布/变换、单页代际变化、其他实例导致的区间位移、单实例增长、存储重排/不足、取消重置和重新发布。
- `RhiRendering.stream_blas_cut_cache`：实际 runtime/CLAS，退休当前 cut 引用的非 root 页面后正确失效。原测试退休任意页面并期待全局失效的假设已移除。
- `RhiResource.clas_actual_sizes_and_move`。
- `RhiResource.clas_compact_lifecycle`：同时修正测试硬编码的 4 字节页表 stride。
- `RhiRendering.minizorah_clas_in_flight`。
- `RhiRendering.stream_clas_runtime_lifecycle`。
- `RhiRendering.stream_clas_eviction_reupload`。
- `RhiRendering.zorah_full_first_frame`：MiniZorah→Full、完整准备、后续渲染和材质覆盖检查。

已查看 Full settled/base-color 图，建筑、人物、植被及材质覆盖正常，保留既有单样本噪声。此图为 960×540 原生、DLSS 关闭的回归，不能代替编辑器 DLSS 漫游逐像素对照或长期稳定性验证。合成分配测试验证输入/地址和边界；真实 AS 构建由其他 runtime/场景测试覆盖。

证据：`build-scheduling-release/blas-reuse-final-validation.log` 和 `build-scheduling-release/blas-reuse-final-validation/`。构建日志位于两个 build tree 的 `blas-reuse-final-build.log`。

## Full 三次交错对照

RTX 5070 Ti；同一最终二进制和 shader 摘要；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、8M candidates。已有磁盘 cook/shader/纹理缓存，每次新进程，预热 3 秒，180 帧绝对相机路线覆盖逻辑 30 秒。窗口隐藏，保留编辑器渲染路径及默认 VSync/Reflex。`workloadEvery=0`，软件/分类工作量计数与验证层均关闭，六次均 `diagnosticRun=false`。

执行顺序 A1、B1、B2、A2、A3、B3；A 为强制重建，B 为默认复用。

| 运行 | BLAS build 均值 ms | 整帧均值 ms | 整帧 P95 ms | >33.33 ms 帧数 |
|---|---:|---:|---:|---:|
| A1 | 8.799 | 41.299 | 64.564 | 121/180 |
| A2 | 8.826 | 40.595 | 62.709 | 111/180 |
| A3 | 8.807 | 41.850 | 59.831 | 135/180 |
| B1 | 0.413 | 32.731 | 53.756 | 71/180 |
| B2 | 0.383 | 29.368 | 49.837 | 47/180 |
| B3 | 0.418 | 33.033 | 54.327 | 67/180 |

各次运行均值的中位数：

| 指标 | 强制重建 | 逐实例复用 |
|---|---:|---:|
| GPU envelope ms | 37.985 | 28.791 |
| BLAS build ms | 8.807 | 0.413 |
| BLAS 输入准备五个不重叠阶段合计 ms | 0.679 | 0.525 |
| TLAS build ms | 0.319 | 0.307 |
| Early software raster ms | 13.536 | 13.205 |
| Deferred ms | 6.229 | 6.296 |
| 每帧构建实例 | 10,861.8 | 576.9 |
| 每帧复用实例 | 0 | 10,285.9 |
| 每帧动态 live references | 4,376,571 | 4,379,606 |
| 每帧重建 references | 4,376,571 | 241,464 |
| 每次路线重排次数 | 2 | 2 |

两侧 BLAS 输入准备合计只加互不嵌套的 reset/count/compare/setup/insert，不加父级。所有其他 scope 为 inclusive，不累计父子时间。B 侧已使用 arena 高水位最大约 396 MiB，实际固定分配没有据此缩小。

这些是同配置、同绝对相机路线的流送运行，异步页面完成时序使 cut/驻留并非逐帧锁定；不宣称 byte-exact 同 cut A/B。约 0.07% 的 live references 差异和近似相同的准入量用于排除明显减载，不构成逐像素等价证明。

整卡监测仍有桌面/浏览器低量负载，程序退出后约 8–9% GPU、4.3–4.7 GiB 背景驻留；并非硬件独占。六次采样峰值约 13,861–14,170 MiB，不能据此宣称峰值 VRAM 优化。交错重复保留波动范围；没有把单次运行的 180 帧当作 180 个独立实验，也没有执行 WorkControl 单 shader 的 ExperimentRunner/M3 验收。

该对照隔离的是**新 explicit allocator 内的复用收益**，没有重新测旧 implicit allocator 二进制。因此不能将历史上一轮的 BLAS 6.517 ms 与这里强制重建约 8.8 ms 的差异解释为分配器性能变化。

原始证据：`build-release/blas-reuse-ab/{A1,A2,A3,B1,B2,B3}/run1/` 中的 Capture、Frames、Summary、GPU/进程监测与日志；汇总 `build-release/blas-reuse-ab/Comparison.json`。各 manifest 的二进制、shader、资产元数据、配置与相机一致性已核对。

复现单次 B（输出目录必须为新目录）：

```powershell
$env:METALLIC_BLAS_INSTANCE_REUSE='1'
.\Tools\RunZorahFullRoam.ps1 -OutputRoot build-release/blas-reuse-repeat-B `
    -DurationSeconds 30 -WarmupSeconds 3 -Runs 1 `
    -RouteConfig build-release/blas-selected-normal-route.json -TimeoutSeconds 480
```

路线覆盖不等于持续 30 秒实时时长，更不等于长时间驻留压力测试。后续优先处理依然占约 13.2 ms 的 early SW 光栅及其长尾；BLAS 的下一项针对池内空洞和重排尖峰，不能通过重新降低动态覆盖掩盖。跨实例共享另行验证收益与引用生命周期。
