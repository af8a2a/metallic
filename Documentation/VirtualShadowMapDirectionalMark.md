# VSM 方向光 Mark 编译特化

`VirtualShadowMapSettings::directionalMark` 及两个 VSM Pass 默认启用方向光专用 Mark，已有图无需添加属性。`directionalMark: false` 可恢复通用 Mark；显式布尔 `benchmarkDirectionalMark` 优先。只有方向光生效；局部光保持原程序。此前已默认启用的局部光 `optimizeMarkSamples` 不变，方向光仍不使用那项循环不变量提取。

## 当前实现与契约

`VSM_DIRECTIONAL_MARK` 只传入 Mark 编译入口，有效状态进入程序缓存键和实际 shader 日志。最终候选仅特化四个辅助函数：

- `vsmProjectPosition` 直接使用正交投影的 `clip.xyz`。
- `vsmReceiverDomain` 移除方向光不适用的局部光域判断。
- `vsmContains` 保留原正交 inset 和深度边界。
- `vsmVirtualTexel` 保留原正交 texel 计算，去掉透视分支。

`makeProjections` 为全部方向光 fine/coarse projection 写入 `metric.w=0`；缓存复用不改变此字段，`compatibleLight` 禁止跨光类型复用。原采样表达式、`push.lightType` 数据流、资源布局、dispatch 和 barrier 均保留。Build、Projection 及局部光程序不启用该宏。

本项减少每个 receiver/sample 的不适用计算，不增加逻辑 receiver 剔除。`markReceiverDispatches`、`markSamplePositions`、`markLevelTests` 应完全一致；共享哈希导致的 `markPageRequests` 原子次数仍按既有调度相关规则处理。

## 正确性与实验约束

- Release `MetallicRHITests`、`Metallic` 构建；387 文件 shader module audit；99 项 VSM Python 测试。
- GPU 等价矩阵 82 状态、10 组默认路径与关闭 shared Mark 的额外控制，覆盖 resident/stream、cold/warm/dirty/overflow/recover、12 levels、32 steps、近面/屏外/点光接缝。单 pass 按接口契约只覆盖 Hard；多光 pass 覆盖 Hard/PCF/SMRT。比较图像、depth、stats32、45 build 和 15 稳定工作量计数。
- Sponza：1920×1080，SMRT 2×16，6 clipmaps，128×128 virtual pages，2048 physical pages，32 帧历史，无 jitter；第 16 帧 freeze streaming/camera cut，前 32 帧预热。
- 独立 diagnostic A/B 各 64 帧，16/32/63 帧检查 PNG、线性 HDR、depth、shadow；逐 canonical 帧验证输入、统计和工作量。正确性通过才进入普通计时。
- 普通计时关闭 validation、GPUWork、图像回读和 build 子阶段时间戳。固定 discovery 两个 ABBA 区组；仅在 promising 时追加预声明的独立 confirmation 三个区组，每进程 96 帧。主指标是 `markMs`，确认门槛为收益 95% 下界至少 3%、graph 收益下界不低于 −2%。不合并两个阶段，不追加采样追逐显著性。
- 保留磁盘缓存，不锁频，不停止其他应用；串行运行不等于系统 GPU 独占。原生编译资源与静态 SPIR-V 计数不替代 GPU 时间。

首轮测试夹具误加入 legacy 单 pass 不支持的 PCF/SMRT，按既有接口修正后重新验证，未修改运行时支持范围。

## 首版未通过的完整场景验证

首版还将 Mark 局部 `push.lightType` 固定为零，以便跨辅助函数消除更多分支。82 状态矩阵通过，但 Sponza 第 18 帧的 requested fine pages 从 3628 变为 3629，unadmitted fine pages 从 1581 变为 1582。三个截图检查点的 PNG/HDR/depth/shadow、全部 45 build 和 15 稳定工作量均一致，但不满足请求页精确等价，因此未进入普通性能采样。

这与采样页边界数值差异相符，但根因尚未证明。最终候选撤回局部 `push.lightType` 常量传播，恢复原采样数据流；没有放宽验证门槛，也没有用 `precise` 修改基线。首版证据保存在 `build/vsm-directional-mark/`，其原生资源数值不作为最终候选结论。

## 最终结果：2026-10-30，RTX 5070 Ti / 621.01

最终证据位于 `build/vsm-directional-mark-v2/`。82 状态与 10 组控制通过，validation 消息为零。完整场景 diagnostic 的全部 canonical 输入、stats32、45 build、15 稳定工作量及三个截图检查点完全一致，第 18 帧请求页差异不再出现。已查看场景图像及诊断/计时图表。

Discovery 的 Mark 收益点估计为 24.17%，95% 区间为 [−13.20%, 49.21%]，只作为进入确认的依据。独立 confirmation 三个 ABBA 区组、12 个进程得到 **accept / confirmed_gain**：

| GPU 范围 | A 通用版 ms | B 专用版 ms | 区组收益 | 95% CI |
|---|---:|---:|---:|---:|
| Mark | 1.998976 | 1.623024 | 18.22% | [4.05%, 30.30%] |
| Shadows | 4.110120 | 3.737648 | 9.08% | [8.54%, 9.61%] |
| RenderGraph | 7.125784 | 6.786384 | 5.13% | [1.87%, 8.28%] |
| Allocator | 0.088976 | 0.096648 | −8.52% | [−9.62%, −7.43%] |
| Build | 0.150200 | 0.151304 | −0.82% | [−1.09%, −0.56%] |
| Projection | 1.017848 | 1.016592 | 0.07% | [−0.12%, 0.26%] |

A/B 两列是各六个进程中位数的中位数；收益与区间使用独立区组 log ratio，不是这两列直接相除。确认阶段 B 的 Mark 进程间相对范围为 16.22%，保留全部样本。Allocator 约增加 0.0077 ms、Build 约增加 0.0011 ms，原因尚未单独归因；Shadows 与 graph 的整体收益已包含这些代价。结论限于当前 Sponza 配置和 GPU，不是全场景或端到端 FPS 承诺。该轮未追加性能轮次，测量时默认关闭；随后按用户要求启用默认路径。

编译证据通过实际日志 SHA → 缓存原始 SPIR-V → FNV 字节指纹 → 原生 pipeline 统计关联：

| Mark 编译指标 | A | B |
|---|---:|---:|
| 普通 SPIR-V 字节 | 70668 | 64340 |
| 静态指令数 | 4665 | 4206 |
| 条件分支 | 209 | 179 |
| 驱动 Register Count | 55 | 40 |
| 驱动 Shared Memory Size | 1024 | 2048 |
| 驱动 Binary Size | 76800 | 68096 |

两个程序在 SPIR-V 中都只声明 1024 字节共享数组，驱动额外共享内存的原因未证明；不从异常的原始 Local Memory Size 推断 spill。Build/Projection 的原始 SPIR-V 和原生资源保持一致。普通/诊断 A/B 四个模块均通过 `spirv-val`。

普通 Mark A 保持旧 SHA `75c7eab175e205c89180575cc773d9d9a0d7f7273a8baa866090631464297cac`；B 为 `f60bda9ffb62aaa4d96e3906876b51f06e30c4bed74316632b292d1600cb21df`。诊断 A 保持 `1e6ccfc9d9a0d5ddea5c7211859a084b2fd191fcb35aadf7395db0da74d07a3a`；B 为 `0594c208f1440185836d4c51a4ac4cdb12a94163cd081c7b7f4f8cc1ffa55091`。结束时复核 739 个源码与 20 个二进制身份，无变更。

证据入口：

- [确认报告](../build/vsm-directional-mark-v2/ConfirmationReport/report.md)及[进程中位数图](../build/vsm-directional-mark-v2/ConfirmationReport/ProcessMedians.png)。
- [精确等价诊断](../build/vsm-directional-mark-v2/DiagnosticReport/report.md)。
- [编译审计](../build/vsm-directional-mark-v2/CompilerAuditFinal/CompilerAudit.md)与[原生资源](../build/vsm-directional-mark-v2/ResourcesReport/report.md)。
- [最终身份检查](../build/vsm-directional-mark-v2/FinalIdentityCheck.json)。

现有报告目录下的 `Reproduce.py NEW_REPORT_DIRECTORY` 可离线重验并重建报告。新实验使用 `Tools/Perf/VirtualShadowMapBitset.py run --experiment directional-mark`；主指标为 `markMs`，旧 build 实验协议不变。

## 默认启用验证

用户要求默认启用后，仅将 settings 默认值改为 `true`，两个 Pass 继承该默认值，无需修改现有图。显式生产属性 `directionalMark: false` 和 benchmark 覆盖优先级保留。Shader 源码未改动，本轮验证默认选择与正确性，不重新声明性能收益。

证据位于 `build/vsm-directional-mark-default/`：

- Release `MetallicRHITests` 与 `Metallic` 重新构建通过。
- GPU 82 状态矩阵通过，validation 消息为零。10 组控制扩展为七个分支：Generic、Directional、Default、两个关闭 shared Mark 的分支、无 benchmark 覆盖的 ProductionOff/ProductionOn。Default 直接对照 Directional；覆盖两个 Pass、默认与显式开关、benchmark 优先级及局部光回退。
- Sponza 三个独立 diagnostic 进程（各 64 帧）：省略两个 Mark 属性、仅生产属性开启、仅生产属性关闭。16–63 帧输入、stats32、45 build、15 稳定工作量完全一致；16/32/63 帧 PNG/HDR/depth/shadow 完全一致。
- 省略属性和显式开启都执行诊断专用版 `0594c208…`；显式关闭恢复通用版 `1e6ccfc9…`。独立普通默认路径运行 96 帧，实际 Mark/Build/Projection SHA 三元组与此前确认接受的候选一致，Mark 为 `f60bda9f…`。点光和聚光的实际日志保持 `directionalMark=false`。
- 739 个源码与 20 个二进制身份复核无差异。未新增性能 A/B；普通路径进程只验证编译选择，不将单次计时解释为新的收益。

[默认路径验证报告](../build/vsm-directional-mark-default/report.md)包含实际编译身份及离线复核命令。

## 无效 receiver 的后续边界

当前 depth/background/dispatch-tail 和局部光 range/cone/near 拒绝已经发生在 normal 与采样循环之前。空 8×8 tile 仍支付共享哈希初始化和组同步；下一步应先测量空 tile 比例和跨光重复准备成本，再决定是否增加可复用的有效 receiver tile 列表，连同生成、同步和消费总成本做 A/B。

不能在组同步前让部分 wave 提前退出，也不能只保留点光 receiver 的 canonical face，因为滤波射线可能跨面。深度输入不足以证明背面、双面和透射材质的光照无效，不能直接添加 `N·L <= 0` 剔除。
