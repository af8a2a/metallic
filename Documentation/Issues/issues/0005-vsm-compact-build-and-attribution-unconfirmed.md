---
id: "0005"
title: "VSM: compact/indirect build 两次实验都未确认收益，需先拆 finalize 与逐页计数原子成本再决定去留"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapCompactBuild.md:38
  - Documentation/VirtualShadowMapCompactBuild.md:42-50
  - Documentation/VirtualShadowMapCompactBuild.md:63
  - Documentation/VirtualShadowMapCompactBuild.md:69
  - Documentation/VirtualShadowMapBuildAttribution.md:26-36
  - Documentation/VirtualShadowMapBuildAttribution.md:54
  - Documentation/VirtualShadowMapBuildAttribution.md:75-81
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:65
  - Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang:2002-2003
  - Tools/Perf/VirtualShadowMapRetiredExperiments.md:25-26
---

## 背景 / Context

`compactBuild` 把原 64-triangle group 的 cluster 预筛结果压成
`{firstTriangle, levelMask}` 候选，再让精确 `vsmBuildMain` 走 indirect dispatch，省掉重复
预筛和两次 group barrier。后续一轮实验只合并了 compact 输出的两次发布同步
（`VirtualShadowMapBuildAttribution.md:50-52`）。

## 现状 / Current state

- 第一轮：discovery 判定 `inconclusive / uncertain`，按协议停止，未跑 confirmation。
  Build 全阶段 0.149672 → 0.159208 ms（−6.40%，CI [−17.83%, +3.92%]）、
  Shadows −1.05%、RenderGraph −1.11%（`VirtualShadowMapCompactBuild.md:42-48`）。
  两个区组的 build 点估计分别慢 5.55% 和 7.26%，但区间仍含收益。
- 第二轮（合并发布同步）：Build 0.150840 → 0.158944 ms（−5.58%，CI [−16.95%, +4.68%]），
  判定 `valid / uncertain / retain_candidate`，同样未进 confirmation
  （`VirtualShadowMapBuildAttribution.md:75-79`）。
- 分段归因显示成本在阶段间重新分布：B 臂 `compact sync` 约 49–51 μs，同时
  `finalize + 发布屏障` 从 123.664 μs 降到 79.216 μs，净差约 11.5 μs
  （`VirtualShadowMapBuildAttribution.md:26-36`）。
- 工作量侧看不出可行性：Sponza 每帧 4098 个 source cluster，逻辑存活 1072 组（26.16%），
  压缩仍要预筛全部 4098 组，精确三角与页访问量没有减少
  （`VirtualShadowMapCompactBuild.md:50`）。
- 代码与实验入口都还在：`VirtualShadowMapSettings::compactBuild = false`
  （`VirtualShadowMaps.h:65`）、两个 Pass 的 `benchmarkCompactBuild`、
  `vsmCompactMain` 的 `[numthreads(64,1,1)]`（`VirtualShadowMaps.slang:2002-2003`）、
  工具侧 `compact-build` 仍列为 live 实验（`VirtualShadowMapRetiredExperiments.md:25-26`）。
- 文档已给出下一步方向：先拆开 finalize kernel 与后置发布范围，再检验逐页 ready-page
  计数原子是否为热点（`VirtualShadowMapCompactBuild.md:63`、
  `VirtualShadowMapBuildAttribution.md:81`）；同时提到 32-thread compact 与单计数器
  append 是代码检查发现的后续候选，未混入本轮（`VirtualShadowMapBuildAttribution.md:54`）。

## 问题或缺口 / Problem

1. 两轮实验都没有确认收益也没有确认回退，`Build` 区间下界一直为负，实验停在
   `retain_candidate`，没有一个可终止的条件。
2. 关键成本项仍不可分离：79–124 μs 的 finalize 区间同时包含 kernel 与后置发布屏障，
   文档自己写「不能全部归给 finalize 算法」（`VirtualShadowMapBuildAttribution.md:81`）。
3. 存活率 26% 意味着压缩阶段有一多半 lane 做的是「判定不存活」，而线程组仍是 64 线程；
   这条最直观的候选还没被试过，实验却已经在讨论是否收尾。
4. 该候选保留一套额外的 `uint2` 候选列表 scratch（Sponza 理论 +32800 B）与一条独立
   indirect 路径，长期挂在生产代码里没有收益证据。

## 期望结果 / Expected outcome

把成本拆到可归因的最小单位后再判定：要么找到 ≥3% 的 build 收益并采纳，要么以「已确认无
收益」归档并撤除实验路径（工具侧登记为 retired）。

## 验收标准 / Acceptance criteria

- [ ] `VirtualShadowMapBuildAttribution.py` 增加「finalize kernel 本身」与
      「后置发布屏障」两个可分离区间，同一批次内 A/B 报告这两项，不再合并成
      `finalize + 发布屏障`。
- [ ] 给出逐页 ready-page 计数原子是否为热点的证据（诊断计数器或 native 计数器），
      并在报告中明确该原子在 B 臂 build 中的占比或「无法测量」的边界。
- [ ] 至少尝试一项结构性候选（32-thread compact 或单计数器 append，见
      `VirtualShadowMapBuildAttribution.md:54`），与其 A/B 数据一并报告。
- [ ] 预声明预算（区组数、进程数、样本数）在首个 GPU 进程前写入 `Budget.json`；
      未达 confirmation 条件时按协议停止，不追加样本。
- [ ] 最终判定二选一：采纳则更新 `VirtualShadowMaps.h:65` 默认值与
      `VirtualShadowMapProductionOptimizations.md` 固化表；归档则撤除 `compactBuild`
      路径并把 `compact-build` 从 `VirtualShadowMapRetiredExperiments.md:25-26` 的
      live 清单移到历史清单。
- [ ] 期间 `virtual_shadow_map_compact_build_equivalence`（72 状态）与
      `virtual_shadow_map_compact_build_arguments`（10 探针）必须继续通过、VVL=0，
      且 `Build` 阶段结论只按 pass 区间报告，不写成整帧 FPS。

## 相关文档 / References

- [VirtualShadowMapCompactBuild.md](../../VirtualShadowMapCompactBuild.md)
- [VirtualShadowMapBuildAttribution.md](../../VirtualShadowMapBuildAttribution.md)
- [VirtualShadowMapsCompactUpdates.md](../../VirtualShadowMapsCompactUpdates.md)
- [Retired VSM experiments](../../../Tools/Perf/VirtualShadowMapRetiredExperiments.md)
