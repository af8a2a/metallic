# VSM 已接受优化的生产路径

## 2026-11-05 Mark 共享缓存原子读取修复

候选 `6f0b3b7dcfe14fdd5dd2c498d9da4f0e94548b29` 的共享缓存预筛把普通 LDS 读取与其他 lane 的原子写入混用；首尾组屏障不能消除累积阶段的竞争。修复将 Mark 去重页、receiver key 和 mask word 定义为原址 `Atomic<uint>`，显式使用 relaxed load/store/exchange/compareExchange/or。保留 256 槽 receiver 缓存、先读预筛、矩形位并行计算和 row-base 外提，不增加资源绑定或 CPU ABI。

Relaxed 的依据是现有算法：初始化在首个组屏障前完成；receiver key 在累积阶段只从空槽变为一个 page，冲突继续直接写全局 mask；mask 仅单调 OR，预筛读到旧值最多导致重复 OR；最后一个组屏障完成全部贡献后才汇出。Mark 页命中复用插入 lane 最终执行的全局 OR，allocator 在 dispatch 发布后读取结果。原子对象始终直接访问 Workgroup 存储，不复制为局部值。

先尝试删除预筛、恢复无条件原子 RMW 的最小修复，正确性通过，但独立三组交错计时的 Mark 比优化前慢 5.06%（95% CI：4.64%–5.48%），因此未采用。该轮原始数据保留在 `build/vsm-mark-safe-v2-20261105/`，不与后续原子读取版本合并。

验证还修复了 compact 测试的旧归属断言：grid 合法省略的点光面保持白色 shadow / 品红 debug 清除值，不代表该面拥有 receiver。仅从归属计数中排除此组合，仍逐像素比较全部 shadow/debug 输出，并保留真实彩色面的唯一归属断言；未解析投影的品红 debug 配合黑色 shadow，不在排除范围内。修改前 baseline 与原候选也在同一断言失败，失败记录保留在 `build/vsm-mark-safe-20261105/`。

原子读取版本的正确性验证：

- 重建 Release `MetallicRHITests`；页面层级/矩形 mask 独立 oracle 72 案、receiver-mask 缓存 27 帧、pageSize 持久切换 8 帧在三份 shader 上通过，修复版额外通过 compact 24 状态（193×129、方向/聚光/点光、Hard/PCF/SMRT、点光接缝/角点），validation 为 0。该用例分别测试各类灯光，不代表同时混合灯光矩阵。
- Sponza 第 16/47/79 帧 PNG、HDR、depth、visibility 和 shadow 原始字节完全一致，已查看第 79 帧。第 16–79 帧 canonical 输入、stats32、Build60 和稳定工作量一致；仅保留既有 `markPageRequests`、`markReceiverMaskAtomicOrs` 两项调度计数例外。
- 实际普通/诊断 Mark SPIR-V 均通过 Vulkan 1.4 `spirv-val` 及访问/控制流审计：4KB Workgroup 布局，19 个静态缓存访问点全为原子指令，无普通缓存读写；强 CAS 参数、两道组屏障及阶段顺序均通过。修复版相对原候选仅 Mark 产物不同，其他六个检查入口逐字节相同。

计时为独立的三组 `A,U,G,G,U,A`（优化前、原候选、原子读取修复版）共 18 个进程；每进程 96 帧、预热 48 帧。Sponza 1920×1080、SMRT 2×16/history32、分层 2048 页、6 clipmaps、page128，第 16 帧冻结 streaming 并 cut、关闭 jitter；计时关闭 validation、图像捕获、读回、GPUWork、Build 子阶段和 CPUProfile。保留磁盘 shader/PSO 缓存，不锁频、不保证 GPU 独占。统计单位是每组进程中位数的 log 比值，三组采用 Student-t 95% CI（df=2），不把逐帧样本当独立实验。

| 修复版相对优化前 | 耗时下降点估计 | 95% CI |
| --- | --- | --- |
| Mark | 7.78% | 7.14%–8.41% |
| Shadows | 1.31% | −1.86%–4.39% |
| RenderGraph GPU | −1.99% | −4.17%–0.14% |

修复版相对原候选的 Mark 耗时下降 0.034%（95% CI：−0.012%–0.080%），在当前工作负载上保留了原候选的 Mark 收益。Shadows 总耗时的收益未确认，整图超过 2% 的回退未能排除，不能将本轮结果描述成整体性能提升。完整 24 个 GPU 进程（6 个正确性/诊断、18 个计时）按预定顺序完成，无追加样本；前轮失败和最小修复计时单独保留，不合并统计。

原子读取版本的证据位于 `build/vsm-mark-atomic-20261105/`；生成的图像、SPIR-V 和计时不提交源码库。当前结论限定于本机 Slang 2026.18.2 直接生成的 Vulkan SPIR-V 和 RTX 5060，不推断其他编译目标、GPU 或 FPS 收益。

## 2026-11-04 生产路径收敛

2026-11-04：继续将当前默认采用的实现固定为标准路径，删除生产设置、benchmark 切换及旧参考实现：

- 方向光 Projection 正交计算、uniform 资源访问与接收点输出归属按真实灯光类型选择。
- 方向光 per-face SMRT 使用[投影空间射线状态](VirtualShadowMapProjectedRay.md)；按用户决定固化到默认路径，不提供实现切换。
- 多面 one-pass Projection 按活动面数和统计模式自动选择；单面与不兼容模式保留各自必要内核。
- 方向光 SMRT Mark 固定使用 receiver footprint 覆盖，删除旧 ray-pages 开关和不可达 helper；Projection propagated mapping、持久 receiver mask 与聚合写入不再提供旧路径切换。
- 分层方向光缓存自动维护静态页 HZB 并用于动态 cluster 剔除；资源和缓存条件仍决定适用范围。
- 可选 `shadowLOD` 固定使用层级遍历和深度误差预算；可选 `shadowLODSmartBias` 固定采用资源允许的 refined receiver。移除 `shadowLODHierarchy`、`shadowLODDepthGuard`、`shadowLODSmartBiasRefine` 的参考旁路。

`shadowLOD`、`shadowLODSmartBias`、`cacheEnabled`、`cacheLayering`、receiver footprint、过滤质量、预算及诊断仍是实际功能配置；默认关闭的 compact Build、dirty-page hierarchy 等候选未被强制采用。类型、可选资源、分层缓存和 shader 兼容性条件不是 A/B 开关，继续保留。

实验工具相应退役新实时 A/B 的生成/执行入口，保留历史密封包的验证与报告。此清理不追加 GPU 性能结论。

页面缓冲固定包含 propagated mapping 尾部，因此不再需要运行时切换引起的布局失效判定。兼容的分层方向光快照在发布前已分配静态 HZB，保留命令和上一帧状态的资源寿命；删除仅为运行时启用 HZB 而设置的延迟初始化检查。灯类型、缓存层及真正资源条件仍参与兼容性判断。

## 2026-11-04 验证

- Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过；shader ownership audit 392 文件通过。
- 工具 CPU 回归共 84 项：77 项通过，7 项历史报告绘图用例因本机 Python 缺少 `matplotlib` 未完成。退役入口和属性拒绝检查通过；未为绘图安装依赖或追加全套测试。
- 五个针对性 GPU 用例通过，validation 均为 0：Room resident/stream PCF 4 状态、propagated mapping 独立 CPU oracle 19 案、单/多灯 LOD Pass 3 帧、receiver-mask 缓存 27 帧、静态页 HZB 17 帧。未重跑 86/122 状态大矩阵。
- 初轮 LOD fixture 把完整 resident cut 复用时的零 metadata 命中数误判为重建。完整 cut 命中直接返回保留几何，不查询层级 metadata；修正断言后复验通过。首帧选择、稳定 cluster/triangle 数、零误差可见性及后续缓存复用仍受检查，生产代码未因此修改。
- 默认 Sponza 1920×1080、SMRT 2×16、history32、分层缓存和 2048 页，80 帧、预热 48 帧、第 16 帧冻结 streaming 并 cut；图中不包含任何本轮退役属性。第 16/47/79 帧 PNG、HDR、depth、shadow 与清理前默认捕获精确一致，已查看第 79 帧输出。
- 第 16–79 帧 stats32、29 项稳定 Work、全部 Build55 和 canonical 输入精确一致；沿用 `markPageRequests`、`markReceiverMaskAtomicOrs` 两项调度计数例外并逐帧记录。Mark、Build、Projection、mesh、fragment 的实际 shader SHA 完全相同，validation 为 0。没有新增性能 A/B 或长期场景内存测试。

本轮证据位于 `build/vsm-standard-20261104/`：`CPU.log`、`Build.log`、`Focused/`、`FinalGPU/`、`Capture/` 和 `Verification.json`；生成证据不提交源码库。以下为 2026-10-31 的历史验收记录。

2026-10-31：将已接受的 Mark、Build 优化固定在生产实现中，删除对应 Settings 布尔值、Pass 属性解析及仅用于 A/B 的参考分支。诊断计数、分阶段 GPU 时间和仍在评估的候选保持可用。

| 固定策略 | 验收依据 |
| --- | --- |
| 共享 Mark 去重与 cluster 粗筛 | [组合验收](VirtualShadowMapMarkClusterOptimization.md)；共享 Mark 单项未确认净收益，组合已确认 |
| 局部灯 Mark 不变量复用 | [local-only 验收](VirtualShadowMapMarkSamples.md)；方向光仍使用已接受的独立路径 |
| 方向光 Mark / Build 专用编译 | [Mark](VirtualShadowMapDirectionalMark.md)、[Build](VirtualShadowMapDirectionalBuild.md) |
| cluster / triangle dirty-page bitset | [cluster](VirtualShadowMapsDirtyPageBitset.md)、[triangle](VirtualShadowMapTriangleBitset.md) |
| 方向光 Mark clipmap 数专用编译 | [clipmap 验收](VirtualShadowMapMarkClipmapSpecialization.md) |
| 方向光 Mark 页尺寸移位专用编译 | [page-size 验收](VirtualShadowMapMarkPageSize.md)；本轮由候选转为固定启用 |
| 方向光 Mark 有界 tile 请求集合覆盖 | [实现与验证](VirtualShadowMapMarkHashCoverage.md)；2026-11-01 按用户决定默认采用，无新增 A/B 开关 |

方向光/局部灯的编译排列仍按真实灯光类型选择；clipmap 数和 pageSize 仍进入程序缓存身份。局部灯继续使用通用页地址计算。可配置页尺寸的二次幂约束、采样与阴影质量设置不变。

2026-10-31 当次范围未包含 one-pass Projection、Mark ray-pages、方向光 Projection；这些实现现已按上文固定。compact build、Projection statistics，以及 LOD 启用、缓存分层、需求驱动预算等实际功能或资源策略继续独立配置。默认采用不等于每个候选都已确认 GPU 净收益。

对应旧实验的实时生成/执行入口退役，历史密封包的验证与报告能力保留。旧报告中的开关命令仅描述当时实验，不应继续用于当前二进制；重新进行已退役 A/B 需要检出其历史实现。

本次为生产路径收敛，不把诊断运行时间解释为新的性能收益。

## 验证结果

- Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过；shader ownership audit 388 文件通过；实验工具 CPU 测试 86 项通过。
- 12 项 GPU 回归通过，validation 为 0：生产矩阵 122 状态、两项持久参数切换各 8 帧、allocator、remap 15 案、indirect 参数 5 案、compact build 82 状态、方向光 Projection 116 状态、Mark ray-pages 106 状态、内容缓存 50 状态、计时诊断 20 状态及 Projection 86 状态。
- Sponza 清理前已接受路径与清理后默认路径的第 16/32/63 帧 PNG、HDR、depth、shadow 精确一致；第 16–63 帧 stats32、45 项 Build 计数、15 项稳定工作量及 streaming 输入一致。保留既有 `markPageRequests` 原子调度例外。
- Mark、Projection、Raster SPIR-V 完全相同。诊断 Build 删除了不可达的零计数调用；剔除该死块并规范化常量/ID 后，其余 6533 条指令一致，两份原始模块均通过 Vulkan 1.4 `spirv-val`。
- 验证中修复了旧 GPU probe 的 208/216 字节 push ABI 不一致、Remap 缺少失效统计资源绑定，以及计时/Projection 夹具把首次需求预算重分配误认作 Warm 的假设。固定夹具预算以保留严格零重绘断言，生产需求预算默认值未变。初次失败记录保留。

证据位于 `build/vsm-production-cleanup-20261031/FinalReport.json`，跨版本比较位于 `MigrationVerification.json`，历史 page-size 完整证据在 `HistoricalPageSizeReport/` 重新核验通过。后续修改仅涉及测试夹具，生产代码未变。按用户要求不追加 GPU 测试、不重复 Sponza 验证；本轮不新增性能 A/B 结论。
