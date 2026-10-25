# MiniZorah 空间 Demand 层级实验（2026-10-25）

> **收尾状态（2026-10-25）**：本页是历史实验记录。空间候选已从当前运行时撤回，`METALLIC_SPATIAL_DEMAND` 不再控制运行路径；两个空间 A/B 工具只保留历史验证，明确拒绝新采集。以下原始结果与封存证据不变。最终保留范围和恢复验证见 [收尾记录](MiniZorahSpatialDemandCloseout.md)。

本页记录第一版完整 tile 重排候选，源代码已继续演进为 [8-group 空间叶子候选](MiniZorahSpatialDemandLeaf8AB.md)。本页原始结果与证据保持不变。

本次候选保留了现有 LOD 和流送语义，固定视角画面与 GPU 状态比较通过，但遍历性能回退。**不接受为默认优化**：`METALLIC_SPATIAL_DEMAND` 未设置或为 `0` 时仍使用原有层级，`1` 仅启用实验候选。四个 discovery ABBA block 已满足预先声明的回退停止条件，没有继续运行 acceptance confirmation，也没有延长采样寻找收益。

## 实现与契约

- `MeshletLOD.cpp::buildMeshletLodSpatialDemandHierarchy` 从原有 ordered tile 提取叶子，使用 `meshopt_spatialClusterPoints` 按空间中心自底向上构建四叉树，复用保守 bounds 聚合与 preorder/escape 布局。
- 叶子的连续 group 范围、误差、bounds 和 flags 保持原值；新增映射将空间叶子还原成原始 tile ID。
- `MeshletStreamRuntime` 在内存 topology 中追加独立节点和映射，仅调整 distributed Demand 的任务根。磁盘 cook、group ID、原有 parent 表和 tile bitset 布局保持不变。
- Demand 命中后写回原始叶子及原始祖先位；Frontier、safe cut、稀疏遍历顺序、LOD 误差与 hysteresis、page admission/priority、resident fallback 和 RT owner 规则保持原实现。
- 增加实际绑定 Demand shader 的 SPIR-V 身份和层级计数，便于严格核实 A/B。

A/B 使用同一修改后可执行文件，区别为环境开关 `0/1`。这是两种 topology 的对照，不是原提交二进制与修改后二进制的比较；没有测量新增选择分支对旧二进制的开销。

## 工作负载与统计

本机 NVIDIA GeForce RTX 5070 Ti，驱动 621.01；Release `MetallicGPUDrivenSample`，MiniZorah，1920×1080，DLSS Off，关闭 temporal jitter、stochastic texture filtering、sigma denoising，反射 8 samples，geometry/CLAS 各 1 GiB。使用同一相机、灯光与配置。资产 manifest 完整哈希了 5 个依赖文件，共 39,551,537,023 bytes；后续运行核对其身份。

每个独立进程先加载并收敛来源、冻结驻留、重置 owner/history，强制实际 Demand 计算，然后等待 16 帧、保留全部 64 个普通计时帧。诊断 readback 在计时之后，quality 在独立进程中完成。预先声明 quality ABBA、discovery 四个 ABBA block、独立 confirmation 四个 ABBA block；正确性失败或统计确认回退即停止。

有效实验目录为 `build/spatial-demand-20261025/Experiment-final`。完成 4 个 quality 和 16 个 discovery 进程。下面的 ms 为各进程中位数的均值；相对变化及 95% CI 来自四个独立 ABBA block 的 log(B/A)，不将帧数当作独立样本。

| GPU scope | A 原层级（ms） | B 空间层级（ms） | B 耗时变化，95% CI |
|---|---:|---:|---:|
| Detail demand | 0.27385 | 0.29781 | **+8.75% [7.41%, 10.11%]** |
| Stream traversal | 2.25551 | 2.30750 | **+2.31% [1.88%, 2.73%]** |
| RenderGraph GPU envelope | 7.89098 | 7.97763 | +1.10% [-1.44%, 3.71%] |
| Demand seed | 0.17170 | 0.16847 | -1.88% [-2.14%, -1.63%] |
| Frontier | 0.22753 | 0.25301 | +11.20% [10.82%, 11.58%] |

整图区间跨零，不能据此断言整帧变快或变慢。该测试也不等价于整个 Zorah、动态流送或不同相机路线的性能。

| Demand 诊断计数 | A | B |
|---|---:|---:|
| 任务模板 | 145,039 | 105,193 |
| 实际 seeded/processed 任务 | 25,900 | 26,406 |
| visited nodes | 100,645 | 128,947 |
| tested groups | 95,361 | 95,361 |
| 新增空间节点 | 0 | 72,737 |
| 新增 topology bytes | 0 | 4,971,420（4.74 MiB） |

固定叶子 payload 和保守父节点使最终 group tests 没有减少；worker 节点访问反而增加 **28.12%**，任务模板减少未转化为实际遍历收益。代码还显示小树新增父节点、最后不足四个节点时可能产生单子节点父层，且 <=8 节点任务切分限制了上层整体剪枝机会。这些是下一轮设计要处理的结构问题。没有采集硬件 stall/cache 指标，不能将 Frontier 的额外耗时归因于某个具体缓存机制。

普通计时没有 validation/NvPerf/Trace。保留 GPU/process telemetry；其他进程各自 engine 峰值包含 dwm 2.63%、System 3.27%、cloudmusic 1.09%、msedge 3.73%。合作锁和这些记录不证明 GPU 独占，不能把百分比直接求和成整卡竞争率。

## 正确性与验证边界

- CPU 空间层级测试：叶子覆盖、payload/原序不变、保守 bounds、任务范围、空/单叶/未知 bounds。
- GPU spatial-demand 回归：160 帧 WG64/WG32，跨 bit word 映射，原始叶子和祖先位精确比较，missing page、取消及 cut/history oracle；包含叶宽 1/8/31/32/33/63/64。
- 四项聚焦 RHI 测试全部通过，0 skipped；启用 Vulkan validation，未发现 validation error。
- quality ABBA × seeds 0/1 共 8 份捕获。六个图像资源 `Deferred.color`、`FinalBlit.color`、`ReflectionQuality.color`、`Shadows.shadow`、`VBuffer.depth`、`VBuffer.visibility` 均有限且 bitwise 相同，maxAbs/RMSE 为 0。
- 有序和无序 active/RT cut、有效 TLAS owner、page/CLAS 状态、texture publication、requests 等语义比较全部通过。TLAS 物理地址先对照实际 owner/fallback 分配验证，再规范化比较。
- 补充同一 900 帧移动路线：A/B 均完成 8 次 cut 检查，最终验证 13,782 个实例、21,982 个 active groups，visible-over-target 为 0；无 capacity fallback。**完整移动测试两组均失败**于既有 `Cold scheduling work counts do not partition visits` 断言，未放宽或删除检查。这只能作为已执行子检查的证据，不能标记完整 roaming 验收通过。
- 移动补验使用纯光栅模式。最初 CLAS-on 测试在基线失败，因为该光栅图没有 RT 消费者，运行时关闭 CLAS，与旧 fixture 期望冲突。动态 RT 正确性未验收；固定视角 RT 状态/反射比较已通过。
- shader module audit 通过：375 files / 263 modules / 112 implementing units。Release 构建和 `git diff --check` 通过。

## 复核与复现

从仓库根目录执行，使用已经配置的构建树和 x64 VS 开发环境：

```powershell
cmake --build build-release --target MetallicGPUDrivenSample -j 12
cmake --build build-pass-stages-nrd --target MetallicGPUDrivenSample MetallicRHITests -j 12
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --gtest_filter=*meshlet_lod_stream_spatial*:*meshlet_lod_stream_rt_demand:*meshlet_lod_stream_frontier32 --rhi-validation
python -B Tools/CheckShaderModules.py
python -B Tools/Perf/MeasureSpatialDemand.py verify build/spatial-demand-20261025/Experiment-final
python -B Tools/Perf/PlotSpatialDemand.py build/spatial-demand-20261025/Experiment-final --output <new-report-directory>
```

重跑方法见 [MeasureSpatialDemand.md](../Tools/Perf/MeasureSpatialDemand.md)，必须选择新输出目录。当前配置和资产清单在本地 evidence 根目录的 `Config.json`、`Assets.json`。这个编辑器图即使 DLSS Off 也需要 Streamline SDK，因此普通 A/B 使用 `build-release`；RHI 测试复用另一已有构建树。

原始记录、进程返回码、实际 shader 身份、资产/源码/二进制哈希均保存在本地输出目录，未加入源码控制：

- [复核结果](../build/spatial-demand-20261025/Verify.json)、[统计数据](../build/spatial-demand-20261025/Experiment-final/discovery.json)、[时序图](../build/spatial-demand-20261025/Report-final/timings.png)、[遍历计数图](../build/spatial-demand-20261025/Report-final/traversal.png)。
- [移动测试摘要](../build/spatial-demand-20261025/MotionSummary.json)、[A 完整报告](../build/spatial-demand-20261025/Motion-Raster-A/Baseline.json)、[B 完整报告](../build/spatial-demand-20261025/Motion-Raster-B/Baseline.json)。
- 更早的 `Experiment`（Streamline 配置不符）、`Experiment-release`（sandbox 下缺少 GPU counter）、`Experiment-measured`（第 11 次 discovery 的 PDH 采集失败）都保留且未并入有效统计。PDH 监控现保留错误并有界重试；有效最终实验无采集错误。

下一轮应先评估更小、独立的空间叶子与原 group ID 映射，以及小树扁平化和任务粒度，让空间剪枝实际减少 group tests，再单独声明新候选与采样预算。本轮没有修改 LOD/流送策略来制造性能差异。
