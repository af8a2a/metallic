# CLAS 长寿页分布与块内碎片（2026-09-29）

## 改动

在 [start/grow/max 分段池](ZorahFullClasCapacityPolicy20260929.md) 上继续改进分配策略。原路径把所有页放入最先能容纳的块，块内也选第一个可用洞；一个长寿根页即可阻止同块中冷页腾出的空间整块归还。

1. Streamer 将已有 `lockedFallbackPages_` 作为持久页集合传给 CLAS 池；池在初始化时复制位图并验证页号，不按 LOD 层数猜测寿命。某页包含根 group 即按整页持久处理，覆盖在细 LOD 提前终止的 DAG 分支。
2. 根页和可淘汰页只进入各自生命周期的非空块。空块可改类复用，因此 start 预提交块和已回收的动态块不被永久绑定。两类共享原 max 和联合预算，没有新增各自独占的整池预分配。若 grow 大于混合生命周期池的一半，普通块的实际增长量限制为一半，防止小预算被第一类的一个块完全预占；这是单块粒度约束，不是类配额。大页仍可单独申请大于粒度的块。
3. 在同类可容纳的块中选择剩余空间最少的块，尽量集中活页；块内选择最小合适空洞，保留大连续区域给较大页面。几何及其他 `MeshletStreamStorage` 调用仍默认 first-fit。
4. 可通过 `persistentClasGrowBytes` 收紧根页新块粒度（0 继承公共 grow，非零值按设备对齐并限制在公共粒度内）。Full 使用根页 32 MiB / 动态页 64 MiB；MiniZorah 保持公共 64 MiB。start 预提交的空块仍可供任意类使用。
5. 新增根页/动态页 used 和 backing，以及 `fragmentedFreeBytes`：每块空闲总量减去该块最大空洞后求和。它表示分散空洞字节，不代表总空闲或一定无法使用的字节。

只在新页分配时打包；已发布地址不变，不新增活 CLAS 搬移或 BLAS 失效。继续沿用 GPU 在途租约、退休宽限期、空块保留和预算不足重试规则。无需 Shader 布局变化。

## 边界

隔离可能多保留一个生命周期类别的尾块；收益取决于可回收页是否足以腾空整块。动态类中长期可见的非根页不会自动晋升，已形成的活页空洞也不会被在线压实。本次避免把高频地址搬移和 BLAS 重建引入漫游关键路径。若仍需强制整理，应按块收益选择受害者，保留旧地址至所有缓存 BLAS 失效及在途使用结束，再做有界 MOVE；不能仅修改页地址表就释放旧位置。

## 验证

同一 MSVC Release 配置构建 sample / RHI 测试；不改变 cook、shader、质量阈值或联合预算。

生命周期 GPU 测试新增：两页本可共享一个大块，但根页与动态页必须隔离；退休动态页后实际归还其完整块，根页地址不变。分别验证公共 grow 大于小池预算时仍给两类留有分配机会、独立根页粒度。碎片测试构造 1,024 B / 512 B 两个洞，要求 257 B 请求（对齐后 512 B）使用小洞并保留 1,024 B 连续区域，释放后仍可完整合并。

中间版本在 16 MiB Bunny 实时阴影测试中复现动态 CLAS 无法准入：默认 64 MiB grow 占满了小池。已修复为混合类别下普通块不大于预算一半，修复后的八项回归通过；保留失败与修复证据在 `clas-lifetime-validation`、`clas-lifetime-realtime-recheck`、`clas-lifetime-final`。最终根页专用粒度版本也完成八项回归：`clas_actual_sizes_and_move`、`clas_compact_lifecycle`、`minizorah_clas_in_flight`（1,200 帧）、`stream_clas_runtime_lifecycle`、`stream_clas_eviction_reupload`、`zorah_full_first_frame`、`streamed_realtime_pipeline`、`streamer_meshlet_fragmented_storage`，均通过、无跳过，启用 Vulkan 验证，未见验证错误。证据为 `build-scheduling-release/clas-lifetime-policy-validation.log` 及同名目录的报告。

检查最终 Full settled 图像，未见新增缺块。与本轮改前 `clas-capacity-full` 的 960×540 原生测试输出相比，settled 仅 2 个像素不同、base-color 仅 1 个像素不同。这不等于 DLSS 长时间时序稳定性验收。

## 同路线观测

使用 `Tools/RunZorahFullRoam.ps1` 与 `blas-selected-normal-route.json`，每组一轮 180 帧，逻辑路线 30 秒、预热 3 秒，输出 1797×660、渲染 1198×440，DLSS Quality、LOD 1.5 px。新进程启动，已有磁盘缓存，无并行编译或本任务 GPU 测试。初始整卡驻留和后台负载有波动，不做整卡峰值/帧时间改善结论。

原 first-fit 64 MiB、分池 64 MiB、分池 32 MiB 对照均保留。64 MiB 分池中根页使用 1,430.018 MiB，backing 1,472 MiB；动态页 backing 320 MiB。全改 32 MiB 后根页 backing 降到 1,440 MiB，但动态页增长到 352 MiB，总峰值仍为 1,792 MiB，因此最终保留动态页的 64 MiB 粒度，只收紧根页。

| 策略 | backing MiB | 根页 backing MiB | 动态页 backing MiB | 块数 |
|---|---:|---:|---:|---:|
| 改前 first-fit，64 MiB | 1,792 | 未分池 | 未分池 | 28 |
| 分池 + best-fit，均为 64 MiB | 1,792 | 1,472 | 320 | 28 |
| 分池 + best-fit，均为 32 MiB | 1,760–1,792 | 1,440 | 320–352 | 55–56 |
| 最终根页 32 / 动态页 64 MiB | **1,760** | **1,440** | **320** | **50** |

最终采样范围内 backing 比改前少 **32 MiB（1.79%）**。四组 CLAS 子分配峰值均为 1,829,085,824 B（1,744.352 MiB），shader 摘要与 cook 文件元数据一致。异步驻留使逐帧 cut/子分配量略有差异；这不是强制冻结 cut 的性能试验。

最终 root used 为 1,430.018 MiB；动态页 used 为约 252.8–314.3 MiB。分散空洞最大约 61.75 MiB，说明动态类中仍有活页分隔的空洞。所有组采样期间整块释放计数均为 0；不能把少申请的 32 MiB 称为漫游中回收了 32 MiB。动态页退休后可独立释放整块由 GPU 生命周期测试证明。

四组请求溢出、I/O 失败和 BLAS overflow 都为 0。几何 allocationFailures 每帧 1，是原预算准入计数；BLAS 暂缺 CLAS 实例在加载期间非零，最终均回到 0，不宣称全程无临时 fallback。

最终一轮帧均值 27.428 ms、P95 34.412 ms，19/180 帧超过 33.33 ms，未满足整段至少 30 fps。改前均值 27.691 ms、P95 35.655 ms；单轮不足以归因提速。CLAS completion/expiry CPU scope 均值从 0.245 到 0.249 ms，CLAS build CPU scope 从 1.654 到 1.871 ms；后者包含构建/搬移/录制工作，非分配器独立耗时。块数增加的代价仍需多轮固定工作量验证，不据此宣称 CPU 零成本。

原始证据：

- `build-release/clas-lifetime-baseline/`：改前二进制。
- `build-release/clas-lifetime-final-roam/`：分池 64 MiB。
- `build-release/clas-lifetime-32m-roam/`：分池 32 MiB 对照，不作为最终策略。
- `build-release/clas-lifetime-policy-roam/`：最终根页 32 / 动态页 64 MiB。
- `build-release/clas-lifetime-comparison.json`：范围、CPU scope、shader/asset 身份；各运行另有 Manifest、Capture、Frames 和 NVML 采样。

本次已处理根页混入动态块的分布问题，以及新分配浪费大洞的模式；未实现在线跨块活页整理。若目标是远大于 32 MiB 的进一步下降，当前 1,430 MiB 固定根 CLAS 是主要下限，应继续根 cut/cook 结构审计，或为动态碎片实现带缓存 BLAS 失效和旧地址退休的有界搬移。
