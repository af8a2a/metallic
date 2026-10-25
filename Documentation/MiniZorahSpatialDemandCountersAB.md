# MiniZorah：Demand / Frontier 硬件计数器与任务内空间划分（2026-10-25）

> **收尾状态（2026-10-25）**：本页是历史实验记录。空间候选已从当前运行时撤回，`METALLIC_SPATIAL_DEMAND` 不再控制运行路径；两个空间 A/B 工具只保留历史验证，明确拒绝新采集。以下原始结果与封存证据不变。最终保留范围和恢复验证见 [收尾记录](MiniZorahSpatialDemandCloseout.md)。

**已取得修改前后的硬件证据，并完成同场景普通 A/B。新候选遍历总耗时变化 +6.88% [+6.46%, +7.29%]；完成 discovery 后触发预定回退停止条件，候选拒绝，未运行 confirmation。** 任务数已与 ordered 路径相等，但任务数不是有效工作量的充分指标。本轮没有锁定 Frontier 的具体 cache/stall 根因，也没有把诊断时长当作普通 GPU 性能收益。实验路径保留在 `METALLIC_SPATIAL_DEMAND=1` 后，默认仍为 0。

## 先采计数器，再修改

对上一轮 `separate-group-refs-v3` 先执行 4 个指标组的 pilot AB 和正式 ABBAAB：**8 次 pilot + 24 次正式进程，每组每侧 3 次独立运行**。之后才实现 `ordered-task-spatial-v4`，对同一个 `participation_load` 组执行 **2 次 pilot + 6 次正式进程**。诊断与普通计时使用独立构建、配置和进程；两版数据各自与本版 A 比较，不把跨版本时间差作为因果 A/B。

NvPerf SDK 2026.3，GB203 / RTX 5070 Ti，driver 621.01；未改变 GPU 时钟策略。原始 packing / memory / issue 请求组分别要求 6 / 2 / 9 passes，被明确拒绝。根据同一 GPU 的 CounterAvailability 重新声明并验证四个单 pass 组，没有自动改名、填零或暗中跨 pass 合并。原 SMSP thread-instruction 和 long-scoreboard / barrier / membar 组合未满足单 pass，未取得这些 stall 指标。

所有有效采集均完成一个 SDK pass、零 dropped 数据，且恰好包含真实生产 dispatch 的三个范围。记录 module、entry、绑定 SPIR-V hash、phase、workgroup/subgroup、thread count 和 dispatch XYZ。A/B 的实际 dispatch 身份相同：Demand 是 **64 个 CTA × 64 threads = 4096 workers**，Seed / Frontier 各 19,144 个 CTA；Frontier WG32。

- `Streaming/Seed` 包围整个 Clear cooperative dispatch，包含 sparse state/history 清理、root 筛选和任务压紧，不是纯 root-test 成本。
- Demand 不含 DemandReset；Frontier 不含独立 Mask / Prefix / Emit。三个范围均排除外围 barriers。
- 前后 14-buffer 语义与 depth / visibility readback 是独立稳定性快照，**不是计数目标帧的 readback**。它们在计数前后相同，不能据此把逐任务归一化改称目标帧直接计数。
- SDK 报告 Vulkan 1.4 `officiallySupported=false`。本机采集成功不等于官方支持认证；warning 与原始数据均保留。

## 修改前的根因证据

以下是 v3 正式 `participation_load` 三次中位数。L1 使用 global-load requested sectors；它表示访问量，不是命中率。

| Demand 指标 | A ordered | B v3 spatial | 观察 |
|---|---:|---:|---|
| warp instructions (realtime) | 7,898,488 | 5,837,748 | -26.09% |
| L1 load sectors | 2,301,152 | 1,749,920 | -23.95% |
| SM active / elapsed | 64.51% | 47.56% | 活跃比例下降，不等于 occupancy |
| pred-on lanes / realtime warp instruction | 13.8725 | 14.6625 | 参与代理反而提高，不能说 v3 lane 参与下降 |
| active warps / active cycle（独立 occupancy 组） | 1.9454 | 1.8483 | 单独一组采集，不与别组逐次拼接 |

因此，v3 Demand 回退不能简单归结为“执行指令或读请求更多”。实际任务从 25,900 增到 46,053，但访问量减少，SM 活跃率更低；少量常驻 CTA、串行遍历与长尾值得诊断，尚未获得足以确认具体等待原因的 stall 计数器。

v3 Seed 的 warp instructions 为 29,720,324 → 30,267,168，L1 请求为 3,811,856 → 4,946,560（+29.77%）。这是整个 Clear/Seed dispatch 的增量，不把它全部归到某个 root 比较。

v3 Frontier 的 warp instructions 中位数均为 34,513,076，L1 请求约 3,836,800，线程参与代理也相同；执行周期仍增加。L2 read sectors 的 A/B 极差/中位数达 31.67% / 16.53%，DRAM reads 达 101.43% / 55.25%，且 pilot 与正式组方向不同，**不能据此确认 cache miss 增加**。

使用原始 CounterDataImage 做纯 CPU SDK 重解码，先逐项验证所有原指标 double exact，再读取 SM elapsed/active cycles 和 range-average cycle rate。v3 与 v4 各 6 份图像 × 3 ranges 的阳性对照全部通过。SM sum/avg 为 70；频率差异不足以解释 Demand / Frontier 回退。GPC 相关请求返回 NaN，作为缺失保留，没有换指标或补零。

## 针对性实现与实际打包结果

v4 让 Seed 始终使用原 ordered roots8、实例任务表和 root IDs；每个原任务在独立空间 buffer 中映射到 `[begin,end)` 私有 forest，forest 内继续使用 leaf8。保留 coarse-to-fine LOD 次序、原 group/tile IDs、LOD metric/hysteresis、history、祖先传播、ordered cut、缺页 fallback、request/admission/priority、owner 和 cook 契约。workers 保持 4096，没有通过改变并行度混入另一候选。

原始 topology 和 primitive metadata 按字节保持不变。独立 block 的三 word header 保存 node count、range map offset、group-ref offset；原 root ID 查两 word range，group ref 仍还原原 group/tile。上传、barrier、submission lease、取消和 ready 生命周期继续使用现有机制。原任务预算检查已覆盖 v4，不再检查一个不存在的候选任务表。

原 roots8 子树最多 6 个原叶、384 groups；leaf8 forest 最多 48 叶、95 nodes，低于声明的 128-node 上限。空间分割不跨原任务边界，因此会牺牲部分全局空间聚合。

| 稳态诊断计数 | A ordered | v3 B（历史正式） | v4 B（本轮正式） |
|---|---:|---:|---:|
| 任务模板 | 145,039 | 225,406 | **145,039** |
| seeded / processed tasks | 25,900 | 46,053 | **25,900** |
| visited nodes | 100,645 | 52,223 | 80,877 |
| tested groups | 95,361 | 76,221 | 76,327 |
| group slot 利用率 | 77.34–77.97%（v4 A） | 72.84–74.69% | 67.36–68.86% |
| 独立空间 bytes | 0 | 28,167,040 | 27,820,288 |

v4 对 A 仍减少 19.96% group tests；相对 v3，任务减少 43.76%，节点测试却增加 54.87%，group tests 几乎不变。v4 的任务测试组直方图（0、1–8、9–16、17–32、33–64、65–128、129–256、257+）为 `[146, 24832, 312, 384, 143, 68, 15, 0]`；95.88% 的任务仍只测试 1–8 groups。最大实际任务 50 nodes / 248 tested groups。slot 利用率只描述 group-evaluation 打包，不等于硬件 occupancy 或整个遍历阶段 lane 利用率。

v4 每次领取任务多一次 range-map 间接查找；每 lane 仍串行找到命中叶后再通过 waveWork 分发。逐 group 的 instance/primitive 读取、两 word ref、history 和 ancestry atomics 主体没有新增。源码逻辑读取数量不能替代最终机器 load 计数。

## 修改后的硬件结果

下表是 v4 正式单 pass 三次中位数；全部逐次值和极差见计数器图表。它们是诊断指标，不是普通计时验收。

| 指标 | A | B v4 |
|---|---:|---:|
| Seed warp instructions | 28,383,672 | 28,383,660 |
| Seed L1 load sectors | 3,733,792 | 3,733,888 |
| Demand warp instructions | 7,860,028 | 8,542,496 |
| Demand pred-on thread instructions | 109,623,680 | 105,279,104 |
| Demand L1 load sectors | 2,297,504 | 2,184,224 |
| Demand SM active / elapsed | 63.96% | 51.19% |
| Demand pred-on lane proxy | 13.9427 | 12.3242 |
| Frontier warp instructions | 34,513,080 | 34,513,080 |
| Frontier L2 read sectors | 460,056 | 461,000 |
| Frontier DRAM read bytes | 688,128 | 1,395,456 |

Seed 的指令与 L1 访问基本恢复到 A 水平；Demand 则 warp instructions **+8.68%**，pred-on lane proxy **-11.61%**，L1 请求只减少 4.93%。这与更多节点测试、较低 group slot 利用率一致：**合并任务并未减少总执行工作**。

本轮 Frontier DRAM reads 增加在三次中重复出现（A/B spread 1.82% / 7.17%），L2 请求量近似不变。它支持继续检查访存状态，但仍不能直接确定 cache 层级、依赖等待或可回收时间。离线 SM cycle rate 的 A/B 中位数：Demand 2.92193 / 2.92233 GHz，Frontier 2.91415 / 2.91638 GHz；频率变化不能解释观察到的周期差异。没有新增 GPU 轮次来取得这些频率信息。

## 普通 MiniZorah A/B 与画面

相同 MiniZorah 场景、相机和灯光，1920×1080，geometry / CLAS 各 1 GiB，DLSS Off，关闭 temporal jitter、stochastic filtering、sigma denoise。A/B 使用同一普通 Release 二进制，通过进程局部环境变量 0/1 切换；普通构建 NvPerf OFF，未混入 validation、counter 或 trace。每进程来源收敛后冻结驻留、reset owner/history、强制实际 Demand，16 settle frames 后采集 64 个计时帧，诊断 readback 在计时之外。

预先固定 quality ABBA + discovery 4 ABBA blocks + 独立 confirmation 6 blocks。主指标 `Stream traversal` 的确认收益下界须 >0；Seed、Frontier、整图保底下界须 >-2%。本次完成 **20 个进程：4 quality + 16 discovery（4 独立 ABBA blocks）**，触发完成阶段的回退门槛后停止，没有追加 confirmation、重试普通进程或扩展预算。

`Stream traversal` 是 `recordSceneTraversal` 包围的完整 GPU scope，包含需求、Frontier 与后续流送/RT preparation 阶段；不是三个计数范围简单相加，也不是整帧或编辑器 FPS。普通 Seed scope 还包含 state reset / DemandReset 和相关 barriers，不能与仅 Clear dispatch 的 NvPerf Seed 绝对时长互换。

ms 是各侧进程中位数的均值；变化使用完整独立 ABBA block 的 log(B/A) Student-t 95% CI，正数代表 B 更慢。没有把 64 帧当作独立进程样本。

| GPU scope | A (ms) | B (ms) | B 耗时变化，95% CI |
|---|---:|---:|---:|
| **Stream traversal** | 2.249022 | 2.403662 | +6.88% [+6.46%, +7.29%] |
| LOD clear / demand seed | 0.170622 | 0.170126 | -0.29% [-0.57%, -0.01%] |
| LOD frontier | 0.227530 | 0.252054 | +10.78% [+10.63%, +10.93%] |
| Detail demand | 0.273976 | 0.405902 | +48.16% [+45.87%, +50.49%] |
| RenderGraph GPU envelope | 7.744816 | 7.897854 | +1.98% [+1.71%, +2.24%] |

`noiseWarning=false`。总遍历 A/B 的单侧极差/中位数为 0.55% / 0.45%；整图为 0.52% / 0.27%。持久 shader / driver cache 未清理，普通计时进程中的 source compile 次数合计 0。全部样本保留，没有离群过滤或阶段间混算。 各进程 PDH stderr 为空。 合作锁与进程遥测不证明系统 GPU 全程独占，也不能把进程采样直接归因到某一 GPU scope。

![Ordinary timing and block intervals](../build/spatial-demand-task-local-20261025/Report/timings.png)

质量 ABBA × seeds 0/1 共 8 份捕获，`Deferred.color`、`FinalBlit.color`、`ReflectionQuality.color`、`Shadows.shadow`、`VBuffer.depth`、`VBuffer.visibility` 全部有限、逐位相同，maxAbs / RMSE 都是 0。已打开同一 Reinhard + sRGB 预览检查；显示变换不参与原始数值门槛。语义检查包括 active/RT cut、实际地址验证后规范化的 TLAS owner/fallback、page/CLAS、纹理发布、requests、history。

原 topology 两侧均 **191,728,008 bytes / hash 72454115969087686**，primitive hash 均 **11858536166639557600**。空间 buffer 的 26.53 MiB 低于既定 64 MiB 实验验收上限，未改变运行时驻留预算。

![A/B quality preview](../build/spatial-demand-task-local-20261025/VisualReview/quality-preview.png)

## 验证、恢复与边界

- 普通 sample、NvPerf sample、RHI tests 三套已有配置构建通过；未更换 compiler、generator 或普通构建的 SDK 选项。
- 6 项 focused tests 全部通过，0 skipped：RT demand、Frontier32、spatial demand、原 spatial hierarchy、新 task-spatial hierarchy、initial loading。前三项 GPU oracle 共 **584 帧（104 + 160 + 320）**，覆盖 WG32/64、需求/祖先/history/cut/request、缺页和取消；spatial fixture 每帧验证与 A 任务数相等。候选 initial loading 额外验证 Device → Host → Device 上传生命周期。
- 32 项 CPU counter/schema/固定预算检查 + 28 项 v4/legacy 检查通过；旧 v3 普通与旧 v1 counter 完整证据经当前 verifier 通过，旧证据未重写。
- Shader module ownership audit 375 files / 263 modules / 112 implementing units 通过；diff whitespace 检查通过。
- 按用户授权，仅对 `build-spatial-nvperf/Source/MetallicGPUDrivenSample.exe` 做一次 UAC 完整性修复，从 Low 恢复 Medium，后续 relink 保持 Medium。原先写出失败、multipass 拒绝和对应日志均保留；临时诊断探针已移除，没有更改目录/系统安全策略。
- `evidenceValidity=valid`、`effectAssessment=confirmed_regression`、`nextAction=reject_candidate_keep_default_off`。A/B 的环境变量只用于子进程，未写入用户或系统环境；默认 ordered 路径保持。实验源码和全部证据保留，没有提交或发布。

本轮进一步限定了原因，却没有证明 Frontier 的具体 stall/cache 机制。下一步应先拆分 interior/leaf visits、root rejection、每 wave 有效源 lane 与遍历长尾，并设计能控制前序 Demand 访存状态的 Frontier 对照，再选候选；不继续以任务数最少作为目标，也不从现有数据承诺增大 worker 数一定有效。固定 MiniZorah、冻结驻留和 GPU fixture 不等于 Full Zorah 或完整动态漫游验收。

## 可复核证据

- [本轮普通报告](../build/spatial-demand-task-local-20261025/Report/report.md)、[Summary](../build/spatial-demand-task-local-20261025/Summary.json)、[Verify](../build/spatial-demand-task-local-20261025/Verify.json)、[预先计划](../build/spatial-demand-task-local-20261025/Plan.json)。
- [修改前硬件图表](../build/spatial-demand-counters-20261025/CounterReport-v2/Report.md)、[修改后硬件图表](../build/spatial-demand-task-local-20261025/CounterReport-v2/Report.md)、[硬件逐项对照](../build/spatial-demand-task-local-20261025/CounterReport-v2/V3V4Comparison.md)。
- [正式任务打包分析](../build/spatial-demand-task-local-20261025/TaskPackingAnalysis-Formal/Analysis.md)、[分析脚本](../build/spatial-demand-task-local-20261025/TaskPackingAnalysis-Formal/Analyze.py)、[原始计数](../build/spatial-demand-task-local-20261025/CounterFormal/Manifest.json)。
- [本轮源码快照身份](../build/spatial-demand-task-local-20261025/CandidateSourceIdentity.json)、[修改前源码归档](../build/spatial-demand-counters-20261025/PreOptimizationSources.zip)、[UAC 修复后标签](../build/spatial-demand-counters-20261025/ExecutableIntegrity-UAC-After.txt)。
- [最终恢复与身份检查](../build/spatial-demand-task-local-20261025/FinalState.json)、[普通构建配置核对](../build/spatial-demand-task-local-20261025/OrdinaryConfigurationCheck.json)。采集后只补充工具文档的单 pass 资格说明并修正其编码，运行源码与 Python 工具保持采集身份。
- [GPU/CPU 层级测试](../build/spatial-demand-task-local-20261025/RHI.log)、[上传生命周期测试](../build/spatial-demand-task-local-20261025/InitialLoading.log)、[工具检查](../build/spatial-demand-counters-20261025/CandidateDraft/Validation.log)。
- [普通绘图脚本](../Tools/Perf/PlotSpatialDemand.py)、[硬件绘图脚本](../build/spatial-demand-counters-20261025/Analysis/ReportCounters.py)、[画面预览脚本](../build/spatial-demand-task-local-20261025/PreviewQuality.py)、[计数采集规则](../Tools/Perf/MeasureSpatialDemandCounters.md)。

```powershell
python -B Tools/Perf/MeasureSpatialDemand.py verify build/spatial-demand-task-local-20261025/Experiment
python -B Tools/Perf/MeasureSpatialDemandCounters.py verify build/spatial-demand-task-local-20261025/CounterFormal
python -B Tools/Perf/PlotSpatialDemand.py build/spatial-demand-task-local-20261025/Experiment --output <new-report-directory>
python -B build/spatial-demand-counters-20261025/Analysis/ReportCounters.py build/spatial-demand-task-local-20261025/CounterFormal --offline-decode build/spatial-demand-task-local-20261025/OfflineDecode --output <new-counter-report-directory>
```
