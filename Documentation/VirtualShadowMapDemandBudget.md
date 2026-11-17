# VSM 多灯共享物理页池与需求预算

`VirtualShadowMapLightsPass` 为所有准入灯面使用一个 `VirtualShadowMapPhysicalPool`。`demandDrivenBudget` 默认 `true`；GPU 在同一帧完成各面的 receiver marking 后，根据真实请求分配全局物理页。显式 `false` 保留按面均分的逻辑上限，也使用共享图集。运行路径不再读回需求后由 CPU 重建每面 atlas。

## 当帧分配与缓存

CPU 仍按 stable light ID 顺序执行整灯粗页准入：方向/spot 一页，point 六页；预算不足以容纳整灯时发布 `budgetDropped`。全部面完成请求计数后，GPU 先保留这些粗页，再分配 `physicalPageBudget - admittedFaces` 个细页。Demand 使用 max-min 公平分配，先满足小需求，再把剩余容量分给仍缺页的面；不可整除的余数按 stable ID / face 顺序分配。fine target 就是当帧 `statistics[0]`，不含 coarse，不使用历史余量、量化或收缩滞回，也没有独立 atlas 的每面 4095 个细页上限。单个热点面可借用全部可用细页。

Equal 策略为每面计算均分上限及确定性余数，实际 grant 是该上限与当帧请求的较小值。空面仅获得 coarse grant；其未使用的均分份额在 Equal 中不会转给其他面，Demand 则可以使用它。冷启动、重新启用及灯型变更都在首次成功执行的当帧得到真实请求和 grant，不存在未知需求的过渡帧。超出当帧 grant 的请求继续使用既有 ancestor/coarse fallback。

全局 allocator 先保护所有面本帧已准入的 resident 页，再从空闲槽和本帧未使用的 resident 页中安排新页。页表与 light ID/face 的缓存身份仍分离，grant 改变不重建图集，也不要求清空其他灯缓存。删除、禁用或整灯预算拒绝后的旧物理槽可由其他面复用；改变灯投影只使对应缓存失效。预算总量、page size 或静态分层模式改变会创建新的 pool storage generation，当前实现不迁移旧图集的深度内容，这类变化仍需要重绘。

共享 pool 及每面缓存跟随录制命令的 `SubmissionTransaction` 提交。取消的命令不发布新 cache/history；仅提交无关前缀不代表 VSM 尾部已执行。后续成功帧从已接受的 pool 状态继续。资源仍由完成点及命令/读者寿命保护；这不是任意 graph 阶段部分提交后自动回滚的承诺。`supportsFrameOverlap=false` 保持不变，RenderGraph 仍在执行前等待前帧；共享池没有解除该限制。

## 图集与统计范围

共享深度图集为 `Texture2DArray`，最多 262144 个逻辑槽。容量不超过 4096 时，单层网格为 `X=ceil(sqrt(capacity))`、`Y=ceil(capacity/X)`；更大时每层固定 64×64 页，层数向上取整。`pageSize=128` 时单层最大为 8192×8192。所有灯使用同一 viewport 与层网格；mapping 保存全局 physical slot。静态分层使用同样布局的第二张图集；static HZB 按最多 4096 页一个 buffer bank 寻址。

`physicalPageBudget` 限制当代全局逻辑槽数及所有当帧 grant 之和，不是显存字节上限。网格/层取整、静态图集、HZB、各面页表和工作 buffer、仍被引用的旧 generation 以及 native allocation padding 需要分别计量。`memory[0]`/`[5]` 统计持有的图集 bytes/页数，`memory[2]` 包含共享 pool 与各面 buffer；不再有 manager 需求反馈 readback buffer 或 `VSM budget feedback` scope。独立 `VirtualShadowMaps::record` 路径仍可使用每面 atlas，不能与 LightsPass 的共享池结果混用。

`budgetDiagnostics` 保持 8-word header 和 `lightCapacity * 6` 个 8-word face block 的 buffer 布局，语义版本升级为 **2**。face block 始终位于 `sourceEntryIndex * 6 + face`，与 compact 输出层数无关；GPU Publish 在所有 face Assign/Build 后写入诊断和灯表 `pageBudget`。

| Header word | 版本 2 含义 |
|---|---|
| 0–3 | 版本 2、Demand 策略启用、准入面数、已发布当帧需求的面数 |
| 4–5 | 全局逻辑页容量、当帧 grant 总和（含 coarse） |
| 6 | 本帧新分配的 fine 页总数，即各面 `statistics[20]` 之和 |
| 7 | 本帧重用的干净 fine 页总数，即各面 `statistics[17]` 之和 |

| Face word | 版本 2 含义 |
|---|---|
| 0–2 | 当帧有效标记、原始 requested fine pages、fine target（等于原始请求） |
| 3–4 | 当帧实际 grant（含 coarse）、同请求下 Equal 策略的 grant（含 coarse） |
| 5–7 | 0（无反馈帧龄）、当前拒绝细页数 `statistics[2]`、录制 `RenderFrameContext` 的 frameIndex 低 32 位 |

disabled/rejected 面保留全零 block。准入空面是有效零需求，不能与未准入混淆。`statistics[12]` 为执行后的 face grant，等于 `statistics[1]+1`；它不再表示独立 atlas 的预留容量。`statistics[6]` 在局部光域筛选前计数，仍不能作为每灯 receiver 需求权重。版本 1 的 quota-change/feedback-buffer 字段不能按版本 2 解读。

## 当前回归与证据要求

GPU `virtual_shadow_map_demand_budget` 比较 Equal / EqualRedraw / Demand / DemandRedraw / 充足预算 Reference / 省略默认属性六个分支，保留 Hard authored-triangle oracle、同策略 cached/forceRedraw 阴影字节相等、宽裕预算跨策略字节相等，以及预算变化、source reorder、禁用/删除、灯型变化、PCF/SMRT 场景。新增 CPU 独立 round-robin oracle 校验每个已执行帧的 max-min grant、稳定余数顺序和版本 2 统计；它不复刻 GPU 的水位二分实现。历史 CPU `virtual_shadow_map_budget_policy` 仍测试旧 host policy helper，不能替代共享 GPU allocator 的验证。

保留注册名 `virtual_shadow_map_demand_feedback_lifecycle` 以兼容现有测试调用，但当前覆盖的是共享 allocation/publication 的取消生命周期：初始化真实 graph 后，分别取消整个未提交帧和无关前缀已接受后的尾部；完成输出必须不变。首个成功重试立即发布六面当帧 grant，并检查录制帧号、独立几何 oracle、随后稳定帧的阴影字节及缓存复用。它不再要求一次额外成功帧才获得需求。

普通 A/B 应固定场景、预算、缓存预热与过滤参数，分别报告页准入、画质、共享图集内存和 pass/frame 时间。更多细页可能增加光栅工作，不自动代表加速。充足预算 VSM 图像仅是图像参考，不能替代独立几何真值。`Tools/Perf/VirtualShadowMapDemandBudget.py` 的输出须按对应诊断版本解释；旧证据目录不得作为共享池的新结果。

## 2026-11-17 共享池验证结果

复用 `build-release` 完成 `Metallic`、`MetallicRHITests`、`MetallicVulkanDeviceFeaturesTests` 构建。22 项 Vulkan 特性 CPU 测试、9 项预算证据工具 Python 测试、441 个 Slang 文件的模块审计及 `git diff --check` 均通过。

以 `--rhi-bindless --rhi-validation` 运行以下 13 项定向 GPU/RHI 测试，最终全部通过：

- `virtual_shadow_map_physical_pool`、`virtual_shadow_map_local_lights`、`virtual_shadow_map_demand_budget`、`virtual_shadow_map_demand_feedback_lifecycle`、`virtual_shadow_map_compact`。
- `virtual_shadow_map_cache_layers`、`virtual_shadow_map_physical_hzb`、`virtual_shadow_map_allocator`、`virtual_shadow_map_remap`、`virtual_shadow_map_propagated_mapping`、`virtual_shadow_map_projected_ray_state`。
- `registry_identity_capacity_and_views`、`resource_range_and_shader_input_contract`。

新共享池测试完成 36 组 cached/forceRedraw 比较，覆盖当帧借页、全局槽唯一性、增删/重排、跨灯实际槽转移、取消重试、预算缩放、点光六面、静态分层和第二图集层。5000 页预算下，热点面获得 4999 页（含一粗页），下一帧复用全部 4998 个细页且新分配为零；第二层具有实际深度写入，并通过独立 ground/blocker 几何 oracle。scalar、cluster 和 window mesh 光栅路径均通过跨层比较。Combined/Layered 阴影输出图像已检查。

现有需求预算测试完成 254 帧、27 组独立几何 oracle；局部灯测试完成 88 帧序列；两种取消路径均保持已完成输出不变，并在首次重试立即发布当前需求。上述图像/预算/生命周期报告的 Vulkan validation 计数均为零。首次局部灯运行被并行源码注释更新触发的 shader source epoch 检测中止；冻结源码后单独重跑通过，未放宽断言。

本机证据：[主回归日志](../build/vsm-shared-pool/regression.log)、[最终重跑日志](../build/vsm-shared-pool/final-gpu.log)、[共享池结果](../build/vsm-shared-pool/final-gpu/VirtualShadowMapPhysicalPool/Report.json)、[预算结果](../build/vsm-shared-pool/regression/VirtualShadowMapDemandBudget/VirtualShadowMapDemandBudget.json)。产物保留在忽略的构建目录中。

本轮未做性能 A/B，也未运行 262144 页峰值显存或超过 65535 个同时脏静态页的极限场景；后者的二维 HZB 参数生成、边界保护及相关入口已通过编译和 SPIR-V 校验，不将这些检查表述为极限容量运行验证。

## 历史：2026-10-31 独立 atlas 与 CPU 反馈验证结果

本节及下节均记录共享池实现之前的每面独立 atlas / 版本 1 CPU 需求反馈路径。其配额、反馈延迟、显存和计时结论只属于当时版本，不能用于证明当前共享物理池的正确性或性能。

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

## 历史：独立 atlas 路径的默认启用与缺省属性验证

随后按用户要求，将注册的 runtime setting 和执行时的属性缺省值同时改为 `true`。仓库七个 VSM graph 均未显式填写该字段，无需迁移即可采用新默认；显式 `false` 仍选择均分。需求反馈需要绑定正在录制的 `RenderFrameContext`，即使 `cacheEnabled=false` 也需要；无跟踪帧的外部调用方可显式关闭需求预算。

新增第六个 `Default` 测试臂，从 authored/runtime properties 均省略该属性，检查注册默认值为 true，并在原有 254 帧序列中与显式 Demand 比较。阴影字节、完整配额/反馈/缓存统计、内存统计和灯表逐帧相同，27 组独立几何 oracle 保留。需求预算、反馈取消、compact 和固定配额局部灯四项 GPU 回归全部通过，validation 为零。固定配额局部灯测试显式关闭该功能，以保留“某灯改变不触发另一灯缓存重建”的原前提；它不作为默认路径证据。

另运行五个 OpenPBR/P4 诊断进程：2048/256 页各比较缺省属性与显式 true，256 页增加显式 false 对照。Default/On 在 32、63 帧的原生 shadow、HDR、depth、visibility 字节一致，后 32 帧完整分配/缓存统计一致；实际反馈 scope 和九面已知状态均验证。256 页 Default/On 准入 235 个细页，Off 准入 164 个。所有进程 validation 为零。本轮只验证默认行为，不作新的性能结论；详见[默认路径报告](../build/vsm-demand-budget-default/Report.md)。
