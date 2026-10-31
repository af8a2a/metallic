# VSM 多灯需求预算

`VirtualShadowMapLightsPass` 的运行时属性 `demandDrivenBudget` 默认 `true`，未写该字段的 graph 使用需求驱动预算，显式 `false` 保留均分策略。该策略使用已完成 GPU 帧的真实细页请求分配各面的物理容量。整灯粗页准入顺序、公开灯表布局、光源槽与稳定 ID 的关系保持原契约。

## 分配与反馈

先按 stable ID 保留完整粗页覆盖：方向/spot 一页，point 六页；预算不足以容纳整灯时仍发布 `budgetDropped`。全部需求已知、当前各面容量满足最近请求且配额总和仍在预算内时，保留现有容量。出现缺页、未知新面或预算收紧时，剩余细页使用 capped max-min 分配，先满足较小需求，再把余量分给仍缺页的面；余数按 stable ID / face 顺序分配，每面上限 4096 页。重新分配时总需求满足后允许额度闲置。

这是受压时借用空闲配额的策略，不主动缩减已满足需求的 atlas。缩容会改变光栅 viewport 和深度插值舍入；保持充足布局既避免无用的缓存重建，也保留宽裕预算场景的逐字节结果。它不承诺自动归还宽裕预算下的闲置显存。保留的是上次录制配额；取消的 resize 不会成为有效 GPU 缓存，后续仍由 `VirtualShadowMaps` 的提交事务和容量兼容性决定是否重绘。

每面 GPU `statistics[0]` 是未被物理配额截断的 requested fine pages，不含 coarse。Publish 阶段只拷贝这一项（4 B/面）到 manager 自有 HostReadback buffer。下次 execute 只读取 copy 自身 `SubmissionTransaction` 已接受且 frame completion 已提交、完成的样本；取消或未提交的样本不推进需求历史。没有新增 CPU wait；现有 Pass 的 `supportsFrameOverlap=false` 仍使 RenderGraph 在执行前等待前帧，本改动没有解除该限制。

未知面以 4095 个 fine target 启动，精确复现旧均分。非零需求加 12.5% 余量、向上量化至 8 页，再限制为 4095；零需求 target 为零。首次样本和增长立即更新，只有连续八个已完成样本均不超过当前 target 的一半时才收缩，收缩到该窗口内最大的量化 target，避免低需求交替导致立即重新扩容。

缓存和反馈按稳定 light ID 关联，灯表继续按 source slot 发布。删除、禁用、预算拒绝或 point/spot 类型转换使旧代反馈失效；scene/device、主视图身份、尺寸与需求设置变化清理不兼容历史。普通相机或光源移动由下一完成帧的需求跟踪；过渡帧仍使用既有 coarse/ancestor fallback，不保证与当帧精确需求分配等价。

反馈 buffer 只有在样本已消费、GPU 完成且命令不再持有 wrapper 时才复用。命令保留独立反馈资源，移除 Pass 或灯光不会释放在途 copy 的存储。不同 light generation 通过 weak ownership 隔离，取消样本不会变成零需求。

## 资源与统计范围

这一步仍使用每面独立 atlas，不是跨灯共享物理页池。`physicalPageCount` 变化会使该面缓存重建；量化和滞回减少稳定场景的反复调整，不能保证任意动态争用下无重绘。总逻辑配额不超过 `physicalPageBudget`，但 atlas 网格取整、静态深度分层、在途旧代资源及 native allocation padding 不属于这个逻辑页数硬上限。删除/缩容后的实际显存释放仍受完成与读者寿命约束。

新增 `budgetDiagnostics` 为 uint32 buffer，含 8-word header 和 `lightCapacity * 6` 个 8-word face block。face block 位置始终是 `sourceEntryIndex * 6 + face`，与 compact 输出层数无关。

| Header word | 含义 |
|---|---|
| 0–3 | 版本 1、策略启用、已准入面数、已获得完成反馈的面数 |
| 4–5 | 总页预算、实际分配页数（含 coarse） |
| 6 | 相对上次录制决策改变配额的面数，不是 GPU 已完成 resize 次数 |
| 7 | 本次新建反馈 buffer 数 |

| Face word | 含义 |
|---|---|
| 0–2 | 需求已知、最近原始 requested fine pages、滞回后的 fine target |
| 3–4 | 实际页配额（含 coarse）、同帧均分策略页配额 |
| 5–7 | 反馈帧龄、最近需求相对当前配额的未满足细页数、反馈 frameIndex 低 32 位 |

未知需求不是已知零需求；disabled/rejected 面保持零 block。`memory[2]` 包含 manager 持有的反馈 buffer bytes，详细 GPU scope `VSM budget feedback` 单独记录拷贝与屏障成本。原 atlas load 计数与本次 CPU 读回无关。`statistics[6]` 在局部光域筛选前计数，不能作为每灯 receiver 需求权重。

## 验证计划与复现

CPU `virtual_shadow_map_budget_policy` 覆盖确定性余数、上限、预算单调性、max-min 公平性和需求滞回。GPU `virtual_shadow_map_demand_budget` 使用真实 point、热点 spot、空 spot，比较 Equal / EqualRedraw / Demand / DemandRedraw / 充足预算 Reference，保留冷启动、预算与光源变化过程；同策略 cached/forceRedraw 精确比较，Hard 阴影另用 authored-triangle oracle。反馈取消由单独的真实 graph 生命周期测试覆盖。

普通计时预先固定 OpenPBR/P4、1920×1080、SMRT 2×16、cache 开启、temporal smoothing 关闭。Slack=2048 页，Pressure=256 页；两组各 3 个 ABBA 区组，共 24 独立进程，每进程 96 帧、32 帧预热，不追加样本。五个独立诊断进程各 64 帧，在 32/63 帧保存图像；Pressure 另有 2048 页参考。Reference 是充足预算 VSM 图像，不替代独立几何真值。

Slack 要求 shadow/HDR、页配额与 atlas bytes 精确一致；Pressure 要求有效细页准入增加、拒绝减少，并且逐灯相对充足预算参考的 MAE 不恶化。更多有效细页可能增加光栅工作，时间、画质和资源利用率分别报告，不把此变化自动解释为加速。普通计时关闭 validation 与诊断读回；候选正常的需求反馈拷贝保留并计入性能。

工具为 `Tools/Perf/VirtualShadowMapDemandBudget.py`，提供 `prepare`、`run --phase diagnostic|timing`、`verify`、`report`。证据输出在 `build/vsm-demand-budget/`；GPU 结果以最终报告为准，计划本身不代表验收通过。

## 2026-10-31 验证结果

Release `Metallic`、`MetallicRHITests` 构建通过。CPU 预算测试含 52,730 个枚举分配、边界/滞回和 18 个布局保留用例；Python 证据工具四项测试通过，包括图像、配额、atlas、需求、反馈和封存篡改负对照。最终 GPU 预算测试共 254 帧、27 组 authored-triangle oracle，另有两种反馈取消路径，均通过且 validation 为零。

反馈取消测试先提交 Equal 帧初始化 graph，再启用 Demand，分别取消整个未提交帧，以及提交前缀但取消反馈 copy 所在尾部。两者首个重试反馈仍未知，下一已完成帧才变为六面已知。它没有验证冷启动外部 graph 整帧取消后的状态回滚；外部 executor 在这种情况下仍需要既有的重编译恢复流程。

OpenPBR/P4 固定场景的五个正式诊断进程全部通过。9 个面各有独立 atlas，稳定请求共 399 个 fine pages：

| 总预算 | 策略 | 准入细页 | 拒绝细页 | 稳态 atlas MiB |
|---|---|---:|---:|---:|
| 2048 | Equal / Demand | 399 | 0 | 135.000 |
| 256 | Equal | 164 | 235 | 16.875 |
| 256 | Demand | 235 | 164 | 17.250 |

2048 页组 shadow/HDR、主视图深度和 visibility 逐字节一致。256 页组有效细页增加 43.3%，拒绝页减少 30.2%；两个方向光与充足预算参考一致，point/spot 相对参考 MAE 均下降。充足预算 VSM 是图像参考，不是独立几何真值。稳态候选配额与反馈缓冲分配无抖动、无 dirty/coarse 重绘。

矩形 atlas 取整使受压组稳态容量增加 0.375 MiB；预热/相机切换重配时，仍被持有的旧代 atlas 使候选记录到 34.125 MiB 峰值。逻辑页数约束不构成显存峰值硬上限。

RTX 5070 Ti / driver 621.01，24 个独立普通进程（每组 3 个 ABBA），每进程 96 帧 / 32 预热；原始缓存保留、未锁时钟、无系统级 GPU 独占。下表为完整区组 log ratio 的时间变化与 95% t 区间：

| 场景 | Shadows | RenderGraph（不含 present） | Projection |
|---|---|---|---|
| 2048 页 | +0.37% [-2.51%, +3.33%] | +0.81% [-4.42%, +6.32%] | +0.48% [-2.83%, +3.90%] |
| 256 页 | -1.03% [-2.51%, +0.47%] | -0.12% [-2.86%, +2.70%] | -3.09% [-3.84%, -2.34%] |

候选反馈 scope 的进程中位数汇总为 5.376 μs；完整 Shadows 和 RenderGraph 区间跨零，不宣称整体 GPU 加速。收益定位是受压预算的有效页利用和参考画质改善。该轮 A/B 时默认为 `false`；后续按用户要求改为默认启用，单独验证缺省属性路径，不将旧数据当作新性能测量。

旧 v1 主动缩小宽裕预算 atlas 的诊断因 6 个点光阴影像素、4 个 HDR 像素变化而失败，保留于 `Diagnostic/`。v2 通过源策略保留充足布局消除该组差异，未放宽字节一致门槛，也未修改 shader 的深度或阴影公式。

本机完整结果：[汇总](../build/vsm-demand-budget/Report.md)、[诊断](../build/vsm-demand-budget/Diagnostic2Report/Report.md)、[普通 GPU A/B](../build/vsm-demand-budget/TimingReport/Report.md)。原始包、图表和日志均为本地构建输出，不纳入源码。

## 默认启用与缺省属性验证

随后按用户要求，将注册的 runtime setting 和执行时的属性缺省值同时改为 `true`。仓库七个 VSM graph 均未显式填写该字段，无需迁移即可采用新默认；显式 `false` 仍选择均分。需求反馈需要绑定正在录制的 `RenderFrameContext`，即使 `cacheEnabled=false` 也需要；无跟踪帧的外部调用方可显式关闭需求预算。

新增第六个 `Default` 测试臂，从 authored/runtime properties 均省略该属性，检查注册默认值为 true，并在原有 254 帧序列中与显式 Demand 比较。阴影字节、完整配额/反馈/缓存统计、内存统计和灯表逐帧相同，27 组独立几何 oracle 保留。需求预算、反馈取消、compact 和固定配额局部灯四项 GPU 回归全部通过，validation 为零。固定配额局部灯测试显式关闭该功能，以保留“某灯改变不触发另一灯缓存重建”的原前提；它不作为默认路径证据。

另运行五个 OpenPBR/P4 诊断进程：2048/256 页各比较缺省属性与显式 true，256 页增加显式 false 对照。Default/On 在 32、63 帧的原生 shadow、HDR、depth、visibility 字节一致，后 32 帧完整分配/缓存统计一致；实际反馈 scope 和九面已知状态均验证。256 页 Default/On 准入 235 个细页，Off 准入 164 个。所有进程 validation 为零。本轮只验证默认行为，不作新的性能结论；详见[默认路径报告](../build/vsm-demand-budget-default/Report.md)。
