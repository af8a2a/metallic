# VSM 标页合并与 cluster 粗筛

默认启用 cluster 粗筛，保留原标页路径。共享页缓存标页已实现为可选实验，但独立耗时收益未获确认，默认关闭；最初 wave 合并候选已因明显回退撤除。所有路径保留原采样、分辨率、页面预算和粗图兜底。

## 标页

每个 8×8 线程组初始化 256 个共享页缓存槽（1 KiB），在所有 receiver/disabled/rounded-tail 提前退出之前完成一次组同步。`vsmMarkPage` 在原有面、containment 和 SMRT last-page cache 检查后，将完整页号通过 atomic exchange 写入乘法 hash 对应的槽。旧值等于当前页号时复用已有请求；否则执行原全局 atomic OR。Hard/PCF/SMRT 共用该 helper，投影、射线与采样计算不变。

完整页号含 level；不同页发生 hash 冲突只会驱逐缓存并增加重复写，不会复用错误页。每个命中都能追溯到一个实际插入线程，该线程在 dispatch 结束前必执行全局 OR；现有 mark→allocator buffer barrier 保证 allocator 晚于所有写入。有效页号小于 196608，`UINT_MAX` 空槽标记不与合法页冲突。共享页缓存不需要 subgroup feature。默认 `VirtualShadowMapSettings::optimizeMark=false`；显式设置 `benchmarkLegacyMark=false` 或 `settings.optimizeMark=true` 才启用实验路径。两个 Pass 从 settings 默认值推导缺省开关，Slang 无宏编译也默认禁用；`VSM_OPTIMIZE_MARK` 进入 shader permutation/cache identity。

## Cluster 粗筛

Fine build 的每个 64-thread 组先取得一个保守世界包围盒：

- Resident：CPU 按连续 64 个 triangle slots 合并 AABB。位移先按各基础三角的最大高度范围扩界，覆盖其所有 microtriangles。结果随原 caster cache 上传和复用。
- Stream：读取完整 terminal cut 对应的 128-triangle cluster header 的 sphere；每个 64-slot 组不跨 candidate，按 instance 仿射变换得到保守世界 AABB，并覆盖材质位移和浮点舍入。使用完整 caster cut，屏外物体仍参与阴影。

各 level 的预检由组内 lane 并行执行：投影包围盒，检查其页矩形是否存在已准入的 dirty fine page，生成共享 level mask。空 mask 免解码三角；其余三角沿用原精确裁剪和逐页判定，仅跳过确定无关的 level。组同步前只允许 uniform early exit，rounded dispatch 尾部 lane 参与同步。

透视包围盒接触或跨越近面时保留该 level；粗筛不依据 projected depth 拒绝。投影页矩形额外扩一页，world-space 浮点余量独立计算。Stream 变换的余量按绝对乘加项量级计算，覆盖大坐标与平移抵消，而非只看抵消后的中心大小。Reference cooker 先量化坐标再构建 cluster bounds，device compact 保留 XYZ float bits。

Coarse raster 保持完整 caster 遍历。`benchmarkDisableClusterCull=true` 或 `settings.clusterCulling=false` 可选原逐三角 build；宏为 `VSM_CLUSTER_CULL`。内部 geometry parameters 从 48 B 扩到 64 B，resident bounds 句柄在 offset 48；原 push layout、公开 32-word statistics 和 16×u64 workload 布局不变。新 bounds buffer 沿用 caster owner/leases 与已有场景、变换、材质、位移及纹理 publication 失效键。

## 计数及验证方法

`markPageRequests` 统计实际执行的 atomic OR。Build 的 active slots/decoded triangles/level tests 统计粗筛后真实工作；`buildPageTests` 只含原精确三角阶段的页检查，不含 cluster 预检。计数下降不能单独证明净收益，仍以普通 GPU timing 判断。

`virtual_shadow_map_optimization_equivalence` 同帧运行 Legacy、MarkOnly、ClusterOnly、Both 四个独立缓存，严格比较 shadow/debug/depth 与每面完整旧统计。测试覆盖 resident/stream、PCF/SMRT、Cold/Warm/Dirty/Overflow/Recover，以及屏外、近面、点光 room/seam。原 content 回归同时覆盖三种对照，保留位移、MASK、取消与缓存更新序列。

真实场景通过 `VirtualShadowMapOptimization.py` 调用既有 acceptance runner，使用同一 EXE 和两个开关构造四臂。诊断模式抓取同帧线性 HDR、深度、阴影及 PNG hash，不额外推进 temporal 相位；普通 timing 关闭验证、readback 和 work atomics。mark/build 使用送入 ShaderModule 并绑定 pipeline 的 SPIR-V SHA256 建立身份。

固定预算：四个独立诊断进程；discovery 两个 `Legacy Mark Cluster Both Both Cluster Mark Legacy` 区组；有希望时再跑三个独立 confirmation 区组。每个 arm 的 ABBA 子序列以区组 log 比值计算 95% Student-t 区间，使用各进程 warmup 后中位数；confirmation 不池化 discovery。目标为对应阶段至少 3% 收益且 graph 回退不超过 2%。这套 VSM 协议复用 `ExperimentRunner.block_gain` 数学，不冒充仅支持 WorkRaster 的通用实验入口。

本机案例与预算保存在 `build/vsm-optimization/cases/`，保持原始 Sponza native 1920×1080、LOD0、6 clipmaps、2048 physical pages、2×16 temporal SMRT。每个 timing 进程 96 帧、warmup 32；diagnostic 64 帧、warmup 32。GPU 采集串行、源文件冻结、每进程 timeout 600 秒；保留磁盘缓存，不锁时钟，也不保证系统 GPU 独占。

## 已完成的 GPU 回归

`build/vsm-optimization/Regression/Process.log` 与 `RegressionFinal/Process.log` 合计验证通过以下 7 种回归：

- `virtual_shadow_map_local_lights`
- `virtual_shadow_map_profiling`
- `virtual_shadow_map_optimization_equivalence`
- `virtual_shadow_map_filter_demand`
- `virtual_shadow_map_content`
- `virtual_shadow_map_scroll_precision`
- `virtual_shadow_map_demand_equivalence`

初轮 optimization equivalence 在 Recover 阶段重复赋予已恢复的原矩阵，setter 对未变化的值返回 false，被测试误认为失败；修正为仅在 Overflow 阶段恢复矩阵后完成重跑。该修正没有放宽图像或统计比较。最终证据为 `build/vsm-optimization/RegressionFinal/VirtualShadowMapOptimizationEquivalence/VirtualShadowMapOptimizationEquivalence.json`：32 个状态全部通过四臂 shadow/debug/depth 和每面完整 stats32 严格比较，Vulkan validation 消息数为 0。32 个状态观察到 mark 工作减少，1 个状态观察到 cluster 剔除减少三角解码；这些是覆盖与工作量证据，不是耗时收益。

上述回归验证了同一生产图输入下的冷启动、缓存复用、脏页、溢出恢复、resident/stream、多灯、过滤、屏外与近面情况；content 回归另外覆盖位移、MASK 和取消后的重试。真实 Sponza 的独立进程比较见后文。

## 初次独立进程诊断未通过

原始证据保留在 `build/vsm-optimization/Diagnostic/`。四个进程均成功完成，但 `DiagnosticReport/Failure.json` 的跨臂验证失败，不能把进程成功解释为正确性通过，也未据此开始性能计时。

Legacy 与 Cluster 的每帧旧统计及第 32、63 帧捕获完全一致；Mark 与 Both 也彼此一致，但两组之间存在差异。输入核查发现前 8 帧的几何驻留历史不同：

| 帧 | Legacy / Cluster resident pages | Mark / Both resident pages | 额外证据 |
| --- | ---: | ---: | --- |
| 0 | 112 | 112 | 两组 pending pages 均为 0 |
| 1 | 112 | 112 | pending pages 为 1 对 0；旧统计已出现差异 |
| 2 | 113 | 112 | geometry used bytes 为 16,991,232 对 16,850,432 |
| 3 | 177 | 176 | geometry used bytes 为 22,254,336 对 22,085,888；非背景 receiver 数为 2,070,496 对 2,070,493 |
| 8 | 497 | 496 | pending pages 为 6 对 7；此帧 stats32 已相同，仍不能据此认定输入相同 |
| 9 起 | 503 | 503 | pending pages 均为 0；已记录的驻留字段趋于一致 |

四臂相机、实际 1920×1080 extent、execution ID 序列、无 jitter、禁用 RayQuery/RTAS 的约束相同；纹理语义 SHA256 在全部 64 帧中一致且不变，纹理驻留均为 380,401,664 B。两组第 32、63 帧的深度 SHA256 均为 `60997064611af988ef98d5c8b52a50b8e80a1aafc879e3066887e2cf67be0575`，但阴影和线性 HDR SHA256 不同。实际加载记录与 SPIR-V 逐项核对还确认，四臂加载的 68 个其它 shader 的 SPIR-V 一致，没有发现这些 shader 的代码漂移。

前史不同能够保留到后续 temporal 输出：`VirtualShadowMapTemporal.slang` 将历史计数封顶在 32 后继续做加权累积，并将结果保存为 half；它不是在第 32 帧清空旧样本的滑动窗口。因此后续深度和 stats32 相同，并不意味着已经得到相同的 temporal 历史。现有证据证明了几何驻留前史不同，尚未证明首个不同页面的具体触发原因；不能直接把这次失败归因于 mark 算法，也不能略过失败宣称真实场景正确。

## 重新声明的固定快照比较

新案例独立保存在 `build/vsm-optimization/cases-primed/`，保留原始诊断包。分辨率、场景、相机、LOD、页面预算、2×16 SMRT 和 temporal 最大历史长度均不变，明确规定：

- 第 16 帧设置 `freezeStreamingFrame=16`，冻结已准备的几何驻留、可见 raster cut 与纹理流送。
- 同一帧触发既有 `events[].cameraCut`，同时以 `comparisonStartFrame=16` 声明跨臂比较边界。`cameraCut` 改变 `ViewConstants.frame[3]`，VSM temporal 的 `sameProjection` 检查因此拒绝上一帧历史；第 16 帧从当前 raw shadow 重新开始。采样使用的 `frame[0]` 不变。
- Diagnostic 捕获第 16、32、63 帧，64 帧总长、warmup 32 不变；timing 为 96 帧、warmup 32，关闭图片捕获和 work atomics。
- 第 16 帧起要求 resident pages、geometry used bytes、原拓扑 hash、primitive metadata hash、纹理语义 SHA256 等输入字段跨臂严格一致，pending pages 必须为 0 且没有流送错误；随后严格比较完整旧 stats32 和声明捕获帧的 PNG/HDR/depth/shadow。输入门禁失败即拒绝采集结论。

第 0–15 帧继续完整保留并标记为准备前史，不能删掉或当作已经通过的跨臂冷启动验证；冷启动等价性证据来自上面的 32 状态同帧回归。新的比较范围是固定相机下的冻结几何与纹理快照。冻结会停止日常 traversal/streaming 更新，因此后续整帧计时只能描述这个快照工作负载，不能推广为动态流送或交互相机场景的整帧收益。Diagnostic 的读回、验证和 work atomics 也不能作为普通 timing 样本。

## 首个候选的固定快照诊断结果

最初候选使用 subgroup BroadcastFirst/Elect 合并同页写入。下面的诊断证明该候选等价，但随后的普通计时否定了其性能，已被共享页缓存候选替换。这里保留完整历史证据，不将其计数冒充当前实现的计数。

`build/vsm-optimization/DiagnosticPrimedReport/report.md` 已通过当前工具离线验证。四臂第 16、32、63 帧的 PNG、线性 RGBA32F HDR、深度和阴影逐位一致，HDR nonfinite components 均为 0；第 16–63 帧的输入签名和完整旧 stats32 一致。已打开第 63 帧图像检查，并检查 `DiagnosticWork.png` 的数据、坐标和裁切。

诊断进程各 1 次，以下是第 32–63 帧计数中位数，不能用作耗时收益：

| 工作量 / 帧 | Legacy | Mark only | Cluster only | Both |
| --- | ---: | ---: | ---: | ---: |
| Mark 原子 OR | 10,980,622 | 5,403,601 | 10,980,622 | 5,403,601 |
| Build 解码三角 | 262,267 | 262,267 | 68,603 | 68,603 |
| Build 三角 level 测试 | 1,573,602 | 1,573,602 | 91,441 | 91,441 |
| Build 精确三角 page 测试 | 5,850,128 | 5,850,128 | 1,688,862 | 1,688,862 |

Mark 原子操作减少 50.8%，cluster 粗筛后的解码三角减少 73.8%。粗筛自身的 page 检查不在最后一行内，其代价由普通 build GPU timing 计入。

普通 timing 不开启 `benchmarkCaptureSemanticSource`，因此没有纹理语义 SHA；报告明确标记 unavailable，并检查可用驻留字段。完整纹理语义等价证据仅来自独立 diagnostic，不能把空 SHA 当成计时进程的语义证明。原始 shader 身份调查位于 `build/vsm-optimization/ShaderIdentityInvestigation/`：其它 68 个入口一致；Mark 的原 4,657 条去 debug SPIR-V 指令及浮点 SSA 操作数均可映射到候选，新增 subgroup 操作与控制流共得到 4,738 条指令。这不证明 driver SASS、寄存器分配或收缩行为一致。

## 首个候选的普通计时：拒绝 wave 合并

`DiscoveryPrimedReport/report.md` 包含两个完整平衡区组、16 个独立进程。Mark only 的 mark 中位数约 2.10→4.29 ms，区组估计为 -102.65% 收益，整图回退 15.94%；Both 的 Shadows 收益只有 1.80%，达不到 3% 筛选目标。不能用原子数量下降掩盖阶段耗时上升，因此不采用 wave 合并，也不对该候选追加确认样本。

同轮 Cluster only 的 build 约 7.96→5.23 ms，区组收益估计 34.70%，属于 discovery promising；此轮尚不构成独立确认。原 wave 源文件快照保存在 `RejectedWaveCandidate/`，诊断和计时包保持原样。

## 共享页缓存候选

第二个且本次最后一个标页候选使用上面的 256 槽共享缓存，避免逐请求 subgroup 通信，允许跨同组不同采样步骤复用请求。共享 atomic、初始化和 barrier 也有代价，因此仍需实际测量，未做缓存大小扫描。

新预算独立声明于 `build/vsm-optimization/cases-shared/Budget.json`：同样四进程诊断、两个 discovery 区组；有希望的路径再用三个独立 confirmation 区组确认。输入准备、质量、warmup 与上一轮一致，源文件身份更新；不将两种标页算法的样本池化。完成该预算后停止，只采用获得确认的路径。

`RegressionShared/` 重跑优化等价、过滤需求、content 三项全部通过（178.984 秒）。等价测试仍为 32 状态、257×193 非整除尺寸，validation 为 0；覆盖组边缘和空背景线程。`DiagnosticSharedReport/` 的四进程输入、完整 stats32 和三个捕获点逐位一致，HDR 无非有限值。工作量中位数为：

| 工作量 / 帧 | Legacy | Mark only | Cluster only | Both |
| --- | ---: | ---: | ---: | ---: |
| Mark 全局原子 OR | 10,980,622 | 375,733.5 | 10,980,622 | 375,698.5 |
| Build 解码三角 | 262,267 | 262,267 | 68,603 | 68,603 |

共享缓存将全局标页原子减少约 96.6%，cluster 将解码三角减少 73.8%。中位数可为半整数；共享槽的碰撞顺序影响重复 OR 数量，但不影响请求集合。上述结果仍仅为 diagnostic 计数。

## 独立确认结果与采用决定

环境为 RTX 5070 Ti、驱动 621.01；Sponza native 1920×1080、固定相机/LOD0、6 clipmaps、2048 physical pages、2×16 temporal SMRT。第 16 帧冻结几何/纹理并 camera cut，96 帧中前 32 帧为 warmup。Confirmation 使用预定三个平衡区组、24 个独立进程，每臂 6 个；未追加样本，未池化 discovery，未剔除 post-warmup 帧。源文件在采集期间冻结。

| 候选与主指标 | 区组收益估计 | 95% 区间 | 判定 |
| --- | ---: | ---: | --- |
| Mark only / mark | 3.81% | -5.16% – 12.02% | inconclusive，默认关闭 |
| Cluster only / build | 35.46% | 33.37% – 37.48% | performance_confirmed，默认开启 |
| Both / Shadows | 22.53% | 18.17% – 26.65% | 组合有收益，但不能替代 mark 单项确认；不选为默认 |

默认采用的 Cluster only 相对 Legacy：Shadows 区组收益为 **23.65% [20.83%, 26.38%]**，整图 GPU 为 **18.15% [14.84%, 21.33%]**。这些是冻结快照、串行提交 preview 的 GPU 时间，不能推广为动态流送、交互相机或编辑器端到端 FPS 收益。时钟未锁定，GPU 不保证系统独占；mark 的独立进程波动已保留在图和原始数据中。

下表为每进程去掉 warmup 后取中位数，再跨每臂 6 个进程取中位数，单位 ms；它是描述性汇总，上表置信区间以完整区组 log 比值计算：

| GPU 范围 | Legacy | 默认 Cluster only |
| --- | ---: | ---: |
| reset | 0.00828 | 0.00670 |
| mark | 2.41705 | 2.40828 |
| allocator | 0.08843 | 0.08867 |
| build | 8.03656 | 5.21006 |
| raster | 0.04866 | 0.04802 |
| projection | 1.05284 | 1.05374 |
| Shadows pass | 12.01315 | 9.19866 |
| graph | 15.48671 | 12.66628 |

最终修改仅选择默认开关后，重新 build，并执行一个省略两个实验开关的生产 LightsPass diagnostic。`DefaultControlReport/Control.json` 验证了最终默认确为旧 mark + cluster；实际 mark/build SPIR-V 与确认臂相同，第 16–63 帧的输入、完整 stats32，以及第 16/32/63 帧 PNG/HDR/depth/shadow 全部一致。该进程只验证控制面，不加入性能样本。Settings 直调和旧单灯 Pass 的缺省解析已静态审查并编译，未借这一个生产图声称覆盖它们的动态运行。

最终 Release 编辑器和 RHI tests 构建成功；shader module ownership audit、19 项 optimization runner 测试、15 项 GPU profile runner 测试通过。输出图像、diagnostic 计数图和 confirmation 的进程中位数/区组置信区间图均已打开检查。未提交改动。

## 证据与复现

- 实现、边界和采用决定：本文。
- 最终普通计时报告：`build/vsm-optimization/ConfirmationSharedReport/report.md`；关键图为 `BlockGains.png`、`ProcessMedians.png`，CSV/JSON 保留所有进程和帧。
- 当前算法质量证据：`build/vsm-optimization/DiagnosticSharedReport/report.md` 与 `RegressionShared/`。
- 最终默认控制面：`build/vsm-optimization/DefaultControlReport/report.md`；复核脚本为 `build/vsm-optimization/VerifyDefaultControl.py`。
- 被拒绝的 wave：`RejectedWaveCandidate/`、`DiscoveryPrimedReport/`。原前史不一致的诊断：`Diagnostic/`、`DiagnosticReport/Failure.json`，均未覆盖。

离线重算使用当前受信任工具，不运行归档包内未知脚本。以下只读原证据、写入新的报告目录，不重新采样 GPU：

```powershell
python -B Tools/Perf/VirtualShadowMapOptimization.py report build/vsm-optimization/ConfirmationShared build/vsm-optimization/ConfirmationRecheck --diagnostic build/vsm-optimization/DiagnosticShared --reference build/vsm-optimization/DiscoveryShared
python -B build/vsm-optimization/VerifyDefaultControl.py build/vsm-optimization/DefaultControlRecheck
```

`ConfirmationSharedReport/Reproduce.py` 提供同样的报告重算入口，`ReportIdentity.json` 记录 Python/matplotlib、工具和计划 hash；封存子包含二进制、源码、输入和日志身份。大资产仍受 acceptance 的 path-only 身份限制，diagnostic 另外验证已发布纹理语义 hash 与几何拓扑/驻留字段，不能等同于所有源资产的完整内容 hash。
