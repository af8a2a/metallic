# VSM 已接受优化的生产路径

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
