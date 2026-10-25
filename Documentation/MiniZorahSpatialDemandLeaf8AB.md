# MiniZorah：8-group 空间叶子 A/B（2026-10-25）

> **收尾状态（2026-10-25）**：本页是历史实验记录。空间候选已从当前运行时撤回，`METALLIC_SPATIAL_DEMAND` 不再控制运行路径；两个空间 A/B 工具只保留历史验证，明确拒绝新采集。以下原始结果与封存证据不变。最终保留范围和恢复验证见 [收尾记录](MiniZorahSpatialDemandCloseout.md)。

**本页保留 leaf8 / task32 / 交错存储候选的历史证据。** 当前 `METALLIC_SPATIAL_DEMAND=1` 已推进为[独立空间元数据 / task128 候选](MiniZorahSpatialDemandSeparateAB.md)，不能用本页计时代表当前源码；默认仍为关闭。

本轮历史结果：细粒度剪枝生效，Demand 局部收益得到独立确认，但尚未确认遍历总收益。group tests 减少 20.07%，Demand 耗时降低 6.27%；Seed 和 Frontier 的额外耗时抵消了部分收益。完整候选判定为 `inconclusive / retain_candidate`，没有追加采样。[上一轮](MiniZorahSpatialDemandAB.md)的证据保持不变。

## 本轮实现

- 空间叶子从原先完整的 <=64-group ordered tile 改为最多 **8 个 group**。用实际 group 的 metric sphere 和 refinement bounds 聚类，按 LOD level 分别构建 forest。
- 四个及以下叶子保持平铺；bottom-up 分组的单节点尾部直接提升，不生成 unary parent。
- 叶子索引 `{groupIndex, sourceTileIndex}` 引用表，每个原始 group 恰好出现一次。同一空间叶子可跨原 tile，GPU 因此逐 group 还原原始 tile ID 并写回其祖先位。
- 原始 group ID、LOD metric/hysteresis、ordered tile/parent 表、safe cut 顺序、page admission/priority、缺页 fallback、RT owner 和 cook 格式保持不变。
- 候选任务子树上限为 **32 nodes**，基线仍为 8。新增布局身份与 64 MiB 空间元数据验收上限；这个上限只用于实验判定，没有改变运行时驻留预算。

实现位于 `MeshletLOD.cpp`、`MeshletStreamRuntime.cpp` 和 `GPUDrivenStreamAsset.slang`。新 A/B 工具使用 `metallic-spatial-demand-ab-v2`，旧 v1 证据仍按旧规则验证。

## 同场景实验

RTX 5070 Ti / driver 621.01，Release GPUDrivenSample，MiniZorah，1920×1080。相机、灯光、配置和资产身份沿用上轮；DLSS Off，关闭 temporal jitter、stochastic texture filtering 和 sigma denoising，质量捕获反射 8 samples，geometry/CLAS 各 1 GiB。原有资产 manifest 的 size/mtime 与依赖集合重新核实，没有重复哈希未改变的大资产。

同一二进制通过环境变量 `0/1` 切换 A/B，因此比较的是两种 topology，不是旧提交二进制与新二进制。现有磁盘/驱动缓存保留。每进程先收敛并冻结来源、重置 owner/history、强制真实 Demand 计算，再等待 16 帧、保留 64 个普通计时帧；诊断 readback 位于计时之后。

预先固定一个候选、quality ABBA、discovery 四组 ABBA、独立 confirmation 四组 ABBA。本次完整完成 **36 个进程**：4 quality + 16 discovery + 16 confirmation，没有重试进程、删除样本或追加预算。两个性能阶段分别统计，不合并估计。

下表只使用独立 confirmation。ms 是每个进程中位数的均值；变化与区间使用四个独立 ABBA block 的 log(B/A) Student-t 95% CI，正数表示 B 更慢。

| GPU scope | A（ms） | B（ms） | B 耗时变化，95% CI |
|---|---:|---:|---:|
| Detail demand | 0.277638 | 0.260228 | **-6.27% [-7.56%, -4.95%]** |
| Stream traversal | 2.261662 | 2.270058 | +0.37% [-0.20%, 0.95%] |
| RenderGraph GPU envelope | 8.035100 | 7.876360 | -1.93% [-6.59%, 2.97%] |
| Demand seed | 0.171316 | 0.175760 | +2.59% [2.16%, 3.02%] |
| Frontier | 0.227408 | 0.250222 | +10.03% [9.71%, 10.36%] |

Discovery 的 Demand 降幅为 5.73% [4.84%, 6.61%]，与确认阶段方向一致。确认阶段的遍历总耗时区间跨零，不能认定总体加速。整图收益下界为 -2.97%，未达到预先要求的 >-2% 保底，因此完整候选没有被接受；这不否定已经观察到的 Demand 局部收益，也不是统计确认的整体回退。

| 诊断计数（32 个普通进程内各侧一致） | A | B |
|---|---:|---:|
| tested groups | 95,361 | 76,221（-20.07%） |
| visited nodes | 100,645 | 50,880（-49.45%） |
| seeded / processed tasks | 25,900 | 47,014 |
| 任务模板 | 145,039 | 336,869 |
| 新增空间节点 | 0 | 248,412 |
| 空间节点与引用表等新增 bytes | 0 | 26,781,696（25.54 MiB） |
| 总 topology bytes | 191,728,008 | 219,277,024 |

总 topology 增加 **26.27 MiB**，包含任务根列表的增长；25.54 MiB 是实验上限检查的空间结构本身，二者口径不同。group tests 真正减少，但更多任务以及额外元数据访问需要付出成本。Demand 约节省 17.41 µs，而 Seed 增加约 4.44 µs、Frontier 增加约 22.81 µs；这些是局部 scope，不应相加后冒充整帧收益。

两个阶段都出现 `noiseWarning=true`。确认阶段 A 整图单侧极差/中位数约 10.82%；全部样本保留。背景 GPU 进程遥测中，System/msedge/dwm 各 engine 全进程采样峰值约 10.54% / 10.29% / 5.67%，不能相加，也不能推定它们正好发生在某个 GPU scope 内。

原始日志复核发现，32 个普通计时进程中的 19 个发生过 1～2 次 PDH query 重试，均未达到五次退出上限；空档保留，没有补零。渲染计时、源/二进制/shader 身份、语义和图像门槛通过，当前 verifier 返回 `evidenceValidity=valid`，但该返回值不证明遥测连续或 GPU 独占。背景竞争归因仍有限，整图差异维持不确定判定。

## 正确性与验证

- 两套已有 Release 构建通过；未更改其 SDK、编译器或生成器选项。
- 四项聚焦 RHI 测试通过，0 skipped，Vulkan validation 未报告错误。CPU 检查 group/tile 唯一映射、跨 bitword、同层 forest、保守 bounds、tiny/empty/unknown、无 unary，以及真实 RT metric 下需求不漏且测试减少。
- GPU spatial-demand 回归共 **320 帧 ordered/spatial WG64/WG32**，比较精确 demand 位图、原 tile 祖先闭包、cut、requests、history、缺页、取消和边界宽度；受控场景的实际 GPU group tests 必须严格减少。
- quality ABBA × seeds 0/1 共 **8 份捕获**：`Deferred.color`、`FinalBlit.color`、`ReflectionQuality.color`、`Shadows.shadow`、`VBuffer.depth`、`VBuffer.visibility` 均有限且逐位相同，maxAbs/RMSE 均为 0。
- ordered/unordered active 与 RT cut、实际 TLAS owner/fallback 地址校验后规范化的 TLAS、page/CLAS 状态、纹理发布、requests 均一致。计时阶段之后的语义捕获也通过。
- 时序/计数图已运行 Python 生成并打开检查。A/B 渲染预览使用相同显示变换检查，原始浮点数据才是逐位比较依据。
- Shader module audit：375 files / 263 modules / 112 implementing units；`git diff --check` 通过。

本轮没有重跑已知在基线也失败的 900 帧旧 fixture，也未修改其 cold scheduling 统计断言。固定视角与 GPU 契约测试不等于完整动态 RT 漫游或 Full Zorah 验收，默认开关保持关闭。

## 证据与复现

本地证据根目录：`build/spatial-demand-leaf8-20261025`。`Plan.json` 记录运行前预算与候选；`Experiment/` 是已封存证据，报告和复核输出位于其外。

- [汇总数据](../build/spatial-demand-leaf8-20261025/Summary.json)、[复核结果](../build/spatial-demand-leaf8-20261025/Verify.json)、[确认阶段原始统计](../build/spatial-demand-leaf8-20261025/Experiment/confirmation.json)。
- [计时图](../build/spatial-demand-leaf8-20261025/Report/timings.png)、[遍历计数图](../build/spatial-demand-leaf8-20261025/Report/traversal.png)、[画面对照预览](../build/spatial-demand-leaf8-20261025/Report/quality-preview.png)。
- [绘图代码及数据说明](../Tools/Perf/PlotSpatialDemand.py)、[测量工具说明](../Tools/Perf/MeasureSpatialDemand.md)、[聚焦测试日志](../build/spatial-demand-leaf8-20261025/RHI.log)。

```powershell
# 使用对应的 x64 VS 开发环境和已经配置的构建树。
cmake --build build-release --target MetallicGPUDrivenSample -j 12
cmake --build build-pass-stages-nrd --target MetallicGPUDrivenSample MetallicRHITests -j 12
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --gtest_filter=*meshlet_lod_stream_spatial*:*meshlet_lod_stream_rt_demand:*meshlet_lod_stream_frontier32 --rhi-validation
python -B Tools/CheckShaderModules.py
python -B Tools/Perf/MeasureSpatialDemand.py verify build/spatial-demand-leaf8-20261025/Experiment
python -B Tools/Perf/PlotSpatialDemand.py build/spatial-demand-leaf8-20261025/Experiment --output <new-report-directory>
```

本轮结束时提出的待验证假设是：把空间数据放到原 ordered topology 之外，保留原始元数据布局，并减少实际 seeded task 数。此处被测的历史实现逐 primitive 插入空间结构，改变了后续原始 tile 的地址间距；这只是 Frontier 回退的待验证解释，没有硬件 cache/stall 证据，不能当作已确认根因。后续独立实现与新一轮结果见[独立空间元数据 A/B](MiniZorahSpatialDemandSeparateAB.md)，本页历史样本未合并或重写。
