---
id: "0045"
title: "Material: 材质性能基线只有单机单 GPU 数据、5 项不稳定且缺硬件计数器与程序规模扫描"
area: material
kind: perf
priority: P2
state: in-progress
milestone: M4-Performance
labels: ["area:material", "kind:perf", "priority:P2", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/MaterialPerformanceBaseline.md:5-9
  - Documentation/MaterialPerformanceBaseline.md:48-52
  - Documentation/MaterialPerformanceBaseline.md:74
  - Documentation/MaterialValueProgramsM2.md:69
  - Documentation/MaterialCoverageNativeSlang.md:136-149
---

## 背景 / Context

路线图 M0 要求为材质系统建立可重复的性能基线，M5 要求「只将具有可重复收益的策略
纳入默认选择」。当前唯一一份完整采集是 2026-10-06 的 56 项场景 × 3 进程基线，
之后只有针对 NaN 的三项定向复测。

## 现状 / Current state

- 采集条件：RTX 5070 Ti / 驱动 616.92 / Release，512×512，32 帧预热 + 64 帧计时，
  保留磁盘缓存、**不锁 GPU 时钟**、关闭 validation/capture、**后台程序保持运行**
  （`MaterialPerformanceBaseline.md:5`）。
- 56 项中 53 项输出有限且三次逐像素一致，48 项通过 10% 波动检查，
  **5 项性能不稳定**，3 项透射 PT 因 HDR NaN 被剔除
  （`MaterialPerformanceBaseline.md:3`）。
- 不稳定的 5 项集中在参数扫描组：M04_BrushedMetal-uniform-Deferred（12.4%）、
  M08_MaskedCard-textured-Deferred（12.7%）、S02_IOR_Roughness-uniform-PT（18.2%）、
  S03_Coat-uniform-PT（17.7%）、S04_Anisotropy-uniform-PT（21.7%）
  （`MaterialPerformanceBaseline.md:27`、`:43`、`:48`、`:50`、`:52`）。
- 文档自己声明：「5 项不稳定数据只能作为观察值，不用于判定小幅优化收益；本轮未查明
  尖峰来源，**也未采集硬件 stall/occupancy 计数器**」（`MaterialPerformanceBaseline.md:74`）。
- 统计范围明确剔除了一些必要项：M8 使用 fused 生产 ray loop、**未计入独立实验性 ray
  queue 分类**（`MaterialPerformanceBaseline.md:11`）。
- 路线图 M2 要求的程序规模压力配置（1/16/64/256 个程序、1/100/1000 个实例，扫描编译
  时间、SPIR-V 体积、寄存器与运行成本）在 M2 文档里仍然列为「尚未关闭」，并说明
  「当前前端预算测试接受 1/16/64、拒绝 256，不能代替这些性能实验」
  （`MaterialValueProgramsM2.md:69`）。
- Coverage native codegen 的整场景验收里，deferred 侧全部判为 Inconclusive，PT 收益
  只对特定 Coverage-active fixture 成立，明确「Deferred gains are not established by
  these noisy process measurements」（`MaterialCoverageNativeSlang.md:136-149`）。

## 问题或缺口 / Problem

1. **没有稳定的材质 pass 成本口径。** 5 项不稳定样本覆盖了 Fuzz、各向异性、涂层这几类
   最贵的材质，正是优化最该关注的区域；20% 级别的波动使任何 5–10% 的改进都无法被
   判定。
2. **缺硬件计数器。** 只有 GPU timestamp 与 pipeline executable properties 的寄存器数，
   没有 stall/occupancy，无法把成本归因到材质求值、纹理采样还是分支/寄存器压力。
3. **缺程序规模维度。** 「更多程序、更多实例」是材质系统的主要扩展方向，没有任何
   对应测量；M5 的 DGC / wavefront 决策因此无据可依。
4. **单一硬件。** 全部数据来自一张 RTX 5070 Ti；wave32 分箱这类优化在小 warp 或
   非 wave32 设备上的行为完全没有性能证据（正确性路径另有覆盖）。

## 期望结果 / Expected outcome

- 一份可复跑的材质性能矩阵，覆盖：材质类别 × PT/Deferred × 程序数量扫描，
  每项给出中位数、波动区间与稳定性判定。
- 每类高成本材质至少一份硬件计数器归因（Nsight/NvPerf stall 与 occupancy），
  能把成本分到材质求值 / 纹理 / 分流。
- 稳定项与不稳定项分开归档，不稳定项不作为优化结论依据。

## 验收标准 / Acceptance criteria

- [ ] 沿用 `Tools/Perf/MaterialBaseline.py` 与 `build/material-perf-20261005-config/Cases.json`
      原始资产与设置，重跑 M04-BrushedMetal、M08-MaskedCard 与 S02/S03/S04 四组，
      给出三进程稳定性判定；若仍 >10%，记录尖峰所在的帧区间。
- [ ] 新增程序规模扫描用例（1/16/64/256 Program × 1/100/1000 实例），输出编译时间、
      SPIR-V 字节数、寄存器数与材质 pass GPU 时间；256 程序超出预算时给出明确诊断。
- [ ] 至少一项采集附带硬件 stall/occupancy 计数器，并在报告中把材质 pass 时间分解说明。
- [ ] 不稳定项与冒烟 NaN 修复后的复测结果分表保存，不混入同一排名。
- [ ] `MaterialPerformanceBaseline.md` 增加「本轮仍未覆盖」章节，明确列出单硬件、
      无时钟锁定、无硬件计数器、无程序规模扫描四项。

## 相关文档 / References

- [MaterialPerformanceBaseline.md](../../MaterialPerformanceBaseline.md)
- [MaterialValueProgramsM2.md](../../MaterialValueProgramsM2.md)
- [MaterialCoverageNativeSlang.md](../../MaterialCoverageNativeSlang.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
