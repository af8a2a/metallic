# VSM 已接受优化的生产路径

2026-10-31：将已接受的 Mark、Build 优化固定在生产实现中，删除对应 Settings 布尔值、Pass 属性解析及仅用于 A/B 的参考分支。诊断计数、分阶段 GPU 时间和仍在评估的候选保持可用。

| 固定策略 | 验收依据 |
| --- | --- |
| 共享 Mark 去重与 cluster 粗筛 | [组合验收](VirtualShadowMapMarkClusterOptimization.md)；共享 Mark 单项未确认净收益，组合已确认 |
| 局部灯 Mark 不变量复用 | [local-only 验收](VirtualShadowMapMarkSamples.md)；方向光仍使用已接受的独立路径 |
| 方向光 Mark / Build 专用编译 | [Mark](VirtualShadowMapDirectionalMark.md)、[Build](VirtualShadowMapDirectionalBuild.md) |
| cluster / triangle dirty-page bitset | [cluster](VirtualShadowMapsDirtyPageBitset.md)、[triangle](VirtualShadowMapTriangleBitset.md) |
| 方向光 Mark clipmap 数专用编译 | [clipmap 验收](VirtualShadowMapMarkClipmapSpecialization.md) |
| 方向光 Mark 页尺寸移位专用编译 | [page-size 验收](VirtualShadowMapMarkPageSize.md)；本轮由候选转为固定启用 |

方向光/局部灯的编译排列仍按真实灯光类型选择；clipmap 数和 pageSize 仍进入程序缓存身份。局部灯继续使用通用页地址计算。可配置页尺寸的二次幂约束、采样与阴影质量设置不变。

以下控制不属于此次固化范围：one-pass Projection、Mark ray-pages、方向光 Projection、compact build、Projection statistics，以及 LOD、缓存分层、需求驱动预算等实际功能或资源策略。默认采用不等于每个候选都已确认 GPU 净收益。

对应旧实验的实时生成/执行入口退役，历史密封包的验证与报告能力保留。旧报告中的开关命令仅描述当时实验，不应继续用于当前二进制；重新进行已退役 A/B 需要检出其历史实现。

本次为生产路径收敛，不把诊断运行时间解释为新的性能收益。

## 验证结果

- Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过；shader ownership audit 388 文件通过；实验工具 CPU 测试 86 项通过。
- 12 项 GPU 回归通过，validation 为 0：生产矩阵 122 状态、两项持久参数切换各 8 帧、allocator、remap 15 案、indirect 参数 5 案、compact build 82 状态、方向光 Projection 116 状态、Mark ray-pages 106 状态、内容缓存 50 状态、计时诊断 20 状态及 Projection 86 状态。
- Sponza 清理前已接受路径与清理后默认路径的第 16/32/63 帧 PNG、HDR、depth、shadow 精确一致；第 16–63 帧 stats32、45 项 Build 计数、15 项稳定工作量及 streaming 输入一致。保留既有 `markPageRequests` 原子调度例外。
- Mark、Projection、Raster SPIR-V 完全相同。诊断 Build 删除了不可达的零计数调用；剔除该死块并规范化常量/ID 后，其余 6533 条指令一致，两份原始模块均通过 Vulkan 1.4 `spirv-val`。
- 验证中修复了旧 GPU probe 的 208/216 字节 push ABI 不一致、Remap 缺少失效统计资源绑定，以及计时/Projection 夹具把首次需求预算重分配误认作 Warm 的假设。固定夹具预算以保留严格零重绘断言，生产需求预算默认值未变。初次失败记录保留。

证据位于 `build/vsm-production-cleanup-20261031/FinalReport.json`，跨版本比较位于 `MigrationVerification.json`，历史 page-size 完整证据在 `HistoricalPageSizeReport/` 重新核验通过。后续修改仅涉及测试夹具，生产代码未变。按用户要求不追加 GPU 测试、不重复 Sponza 验证；本轮不新增性能 A/B 结论。
