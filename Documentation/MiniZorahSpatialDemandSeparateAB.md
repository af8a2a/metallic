# MiniZorah：独立空间元数据与任务合并 A/B（2026-10-25）

> **收尾状态（2026-10-25）**：本页是历史实验记录。空间候选已从当前运行时撤回，`METALLIC_SPATIAL_DEMAND` 不再控制运行路径；两个空间 A/B 工具只保留历史验证，明确拒绝新采集。以下原始结果与封存证据不变。最终保留范围和恢复验证见 [收尾记录](MiniZorahSpatialDemandCloseout.md)。

**原始元数据布局和画面保持一致，但本轮没有解决 Seed / Frontier 回退，也没有确认遍历总收益。** Frontier 慢 13.60%，触发预先声明的回退停止条件；Demand 也慢 13.38%。Verifier 返回 `evidenceValidity=valid`、`effectAssessment=confirmed_regression`，本轮处置为拒绝此候选，不进入独立确认、不追加采样。实验实现保留在 `METALLIC_SPATIAL_DEMAND=1` 后，默认仍为关闭；它不是已接受的性能优化。

## 实现与契约

- 空间节点、`{groupIndex, sourceTileIndex}` 引用、实例任务偏移和任务根列表移入独立 GPU buffer。原 topology 始终按 ordered roots8 构建，保留原四 word 实例表、任务列表及所有 primitive offsets。
- 独立 buffer 的 header 保存任务根偏移、两 word 实例任务表偏移和各 primitive 的 block 偏移。CPU / Slang 使用同一布局。params 复用既有槽为 `spatialDemandBufferIndexPlusOne`，没有扩大 ABI。
- 继续使用最多 8-group 的空间叶子；每个 LOD level 现在只有一个根，单叶直接作为根，不生成 unary parent。空间任务子树上限从上一轮的 32 提高到 128 nodes。
- Clear / Seed 选择对应的实例任务表和根列表；Demand 通过独立引用还原原始 group / tile ID。需求位、原 tile 祖先传播与 Frontier 仍使用原 topology。
- 原 group ID、LOD metric / hysteresis、ordered cut 顺序、缺页 fallback、page admission / priority、RT owner 和 cook 格式保持不变。空间任务超预算沿用 ordered-demand fallback。
- 独立 buffer 接入 Host / Device 创建、分批上传、资源 barrier、submission lease、取消和完成释放；总 metadata 计量包含它，原 topology bytes 单独记录。
- 新增原 topology 和 primitive metadata 的字节 hash，按十进制 uint64 字符串输出，避免 JSON 消费者丢失整数精度。

代码位于 [MeshletLOD.cpp](../Source/Runtime/Render/MeshletLOD.cpp)、[MeshletStreamRuntime.cpp](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp) 和 [GPUDrivenStreamAsset.slang](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang)。本轮证据使用 `metallic-spatial-demand-ab-v3`；[上一轮 leaf8 证据](MiniZorahSpatialDemandLeaf8AB.md)和 v1 证据不修改、不混合统计。

## 固定实验与结果

RTX 5070 Ti / driver 621.01，Release GPUDrivenSample，MiniZorah，1920×1080。沿用相同相机、灯光、资产 manifest 和配置；geometry / CLAS 各 1 GiB，DLSS Off，关闭 temporal jitter、stochastic texture filtering、sigma denoising。质量捕获开启反射，8 samples。

A/B 使用同一二进制，通过环境变量 0/1 切换。每进程先收敛并冻结 geometry / texture 来源、重置 owner / history、强制实际 Demand 计算，再等待 16 帧并采集 64 个普通计时帧。诊断 readback 在计时之后。持久 shader / driver cache 未清理；首个质量进程编译受影响 shader，16 个普通进程的日志中源码编译次数均为 0。

运行前固定 **一个候选，quality ABBA，discovery 四组 ABBA，独立 confirmation 六组 ABBA**。主指标为 `Stream traversal`，确认收益下界须 >0；Seed、Frontier、整图的收益下界各须 >-2%。任何完整阶段的主指标收益上界 <0，或保底指标收益上界 <-2%，即停止。本次完成 **20 个进程：4 quality + 16 discovery**；Frontier 触发停止，没有运行 confirmation，没有重试进程、移除样本或扩展预算。

下表为完整 discovery。ms 为各侧进程中位数的均值；变化及区间使用 **4 个独立 ABBA block 的 log(B/A) Student-t 95% CI**，因此百分比不直接等于表中均值的比值。正数表示 B 更慢。

| GPU scope | A（ms） | B（ms） | B 耗时变化，95% CI |
|---|---:|---:|---:|
| **Stream traversal（主指标）** | 2.306180 | 2.325776 | **+0.93% [-3.20%, +5.24%]** |
| Demand seed | 0.172630 | 0.175816 | +1.85% [1.65%, 2.04%] |
| Frontier | 0.226736 | 0.257568 | **+13.60% [12.95%, 14.25%]** |
| Detail demand | 0.274684 | 0.311430 | +13.38% [12.11%, 14.67%] |
| RenderGraph GPU envelope | 7.933456 | 7.864076 | -0.82% [-4.59%, 3.11%] |

Frontier 的整个区间都越过 2% 回退保底，因此候选被拒绝。Seed 的小幅回退仍存在，区间略跨 2% 边界；遍历总耗时和整图的区间跨零，均不能认定改善或整体回退。这是阶段内 GPU scope 的结果，不是编辑器 FPS 或端到端加速。

![A/B timing and block intervals](../build/spatial-demand-separate-20261025/Report/timings.png)

## 实际工作量与内存

以下诊断计数在 16 个普通进程内各侧一致：

| 指标 | A ordered | B 独立空间数据 |
|---|---:|---:|
| tested groups | 95,361 | 76,221（-20.07%） |
| visited nodes | 100,645 | 52,223（-48.11%） |
| seeded / processed tasks | 25,900 | 46,053（+77.81%） |
| 任务模板 | 145,039 | 225,406 |
| 空间节点 | 0 | 253,577 |
| 独立空间 buffer bytes | 0 | 28,167,040（26.86 MiB） |
| 原 topology bytes | 191,728,008 | 191,728,008 |
| 原 topology hash | 72454115969087686 | 72454115969087686 |
| primitive metadata hash | 11858536166639557600 | 11858536166639557600 |

新增 26.86 MiB 包含独立 header、节点、group refs、实例表和任务根列表，低于预先声明的 64 MiB 验收上限；这个上限没有改变运行时驻留策略。

与上一轮相同场景的确定性工作量相比，模板从 336,869 减到 225,406（-33.09%），**实际任务只从 47,014 减到 46,053（-2.04%）**。模板合并没有等比例减少实际入队工作；实际任务仍明显多于 ordered。这里只比较确定性计数，不把两轮不同二进制的计时拼成新的 A/B。

原始布局和 hash 已相同，Frontier 仍回退，说明“恢复原布局即可消除回退”不足以解释本场景。更少 group tests 也没有自动转化为更短 Demand 时间。下一步应先分析小根任务的打包效率及 Demand 之后的缓存/调度状态；本轮没有硬件 cache / stall 计数器，不能把这些方向写成已确认根因。

![Traversal work](../build/spatial-demand-separate-20261025/Report/traversal.png)

![Actual tasks and templates](../build/spatial-demand-separate-20261025/Report/tasks.png)

## 波动与遥测限制

`noiseWarning=false` 只表示没有超过工具的 10% 单侧极差告警线，不代表测量无波动。A 的总遍历极差/中位数为 **9.90%**，整图为 **9.79%**；B 分别为 0.82% 和 2.89%。图中的两个较高 A 点全部保留，总遍历区间因此较宽。没有选择有利区组或丢弃尖峰。

16 个普通进程中，`discovery-08-A` 的 GPU 进程计数器发生 **1 次 PDH 查询重试**，原始日志和空档保留，没有补零。全进程各 engine 采样峰值约为 dwm 5.92%、System 4.81%、msedge 3.68%；它们不能相加，也不能据此归因到某个 GPU scope。合作锁和遥测不证明 GPU 全程独占。Verifier 的 `evidenceValidity=valid` 表示当前身份、渲染计时、语义和图像门槛通过，不等于遥测连续性证明。

## 正确性与验证

- 复用现有 `build-release` 与 `build-pass-stages-nrd` Release 配置，两套 sample 构建及 RHI test 构建通过，未改变 SDK / 编译器 / generator。
- 最终聚焦 **5/5 RHI 测试通过，0 skipped**：RT demand、Frontier32、spatial demand、initial loading、CPU spatial hierarchy。GPU oracle 共 **480 帧**，覆盖 ordered、spatial32、spatial128 及 WG64 / WG32；精确比较需求位、原 tile 祖先闭包、cut、requests、history、缺页和取消。受控场景要求 group tests 小于 ordered，任务合并要求 spatial128 小于 spatial32。
- 原 topology 和 primitive metadata 在 GPU fixture 内比较完整字节；MiniZorah 全部进程也验证相同大小和完整 uint64 字节 hash。
- 候选额外运行 `stream_initial_loading`，**1/1 通过**：Device → Host → Device 三次重载均为 `separate-group-refs-v3`，独立 buffer 5,976 bytes，metadataReady 为真；保留提交、取消、就绪和预算检查。
- 首次 RHI 尝试保留在 `RHI.log`：一次 WG32 编译失败，随后同入口编译成功；补充编译诊断后最终重跑通过，未宣称已定位编译器问题。另一项失败来自“spatial128 必须少于 ordered8”的错误任务假设；保留原场景并增加 spatial32 对照，改为直接验证本次 32→128 合并效果，没有放宽语义 oracle。
- quality ABBA × seeds 0/1 共 **8 份捕获**。`Deferred.color`、`FinalBlit.color`、`ReflectionQuality.color`、`Shadows.shadow`、`VBuffer.depth`、`VBuffer.visibility` 全部有限、逐位相同，maxAbs / RMSE 均为 0。
- ordered / unordered active 与 RT cut、校验实际地址后规范化的 TLAS owners / fallback、page / CLAS、纹理发布和 requests 一致。计时后的语义捕获也全部通过。
- 26 项纯 CPU 工具验证通过；旧 v2 完整 verify 与两个旧 v1 raw run 兼容检查通过，旧证据没有重写。Shader module audit：375 files / 263 modules / 112 implementing units；diff whitespace 检查通过。
- Python 图表和同一显示变换下的 A/B 预览均已生成并打开检查，未见 A/B 画面差异；原始浮点数据才是逐位比较依据。

![Identical A/B preview](../build/spatial-demand-separate-20261025/Report/quality-preview.png)

没有重跑已知基线也失败的 900 帧旧 fixture，没有改变其 cold scheduling 统计断言。固定视角与 480 帧契约测试不等于完整动态 RT 漫游或 Full Zorah 验收。

## 证据与复现

根目录：`build/spatial-demand-separate-20261025`。`Plan.json` 在采样前记录预算；`Experiment/` 已封存，报告和验证输出位于其外。`SourceSnapshot/` 保留 14 个相关源/工具/测试文件，其中 10 个和测量 manifest 的身份直接核对一致。

- [汇总数据](../build/spatial-demand-separate-20261025/Summary.json)、[重新验证结果](../build/spatial-demand-separate-20261025/Verify.json)、[原始统计](../build/spatial-demand-separate-20261025/Experiment/discovery.json)、[计划](../build/spatial-demand-separate-20261025/Plan.json)。
- [图表报告](../build/spatial-demand-separate-20261025/Report/report.md)、[绘图脚本](../Tools/Perf/PlotSpatialDemand.py)、[任务计数绘图](../build/spatial-demand-separate-20261025/PlotTaskCounts.py)、[预览脚本](../build/spatial-demand-separate-20261025/PreviewQuality.py)、[工具规则](../Tools/Perf/MeasureSpatialDemand.md)。
- [最终 RHI 日志](../build/spatial-demand-separate-20261025/RHI-Final.log)、[候选上传日志](../build/spatial-demand-separate-20261025/RHI-SpatialUpload.log)、[首次失败日志](../build/spatial-demand-separate-20261025/RHI.log)、[CPU 工具验证](../build/spatial-demand-separate-20261025/ToolsValidation/Validation.log)。

```powershell
python -B Tools/Perf/MeasureSpatialDemand.py verify build/spatial-demand-separate-20261025/Experiment
python -B Tools/Perf/PlotSpatialDemand.py build/spatial-demand-separate-20261025/Experiment --output <new-report-directory>
python -B Tools/CheckShaderModules.py
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --gtest_filter=*meshlet_lod_stream_spatial*:*meshlet_lod_stream_rt_demand:*meshlet_lod_stream_frontier32:*stream_initial_loading --rhi-validation
```
