---
id: "0024"
title: "ZorahFull: 生产 1440p 漫游三轮均 sampledRouteMeets30Fps=false，长帧只能逐帧归因"
area: geometry
kind: perf
priority: P1
state: open
milestone: M4-Performance
labels: ["area:geometry", "kind:perf", "priority:P1"]
depends_on: []
evidence:
  - Documentation/ZorahFullPerformanceBaselineAfterNsightFix20260930.md:3
  - Documentation/ZorahFullPerformanceBaselineAfterNsightFix20260930.md:7-11
  - Documentation/ZorahFullPerformanceBaselineAfterNsightFix20260930.md:19-33
  - Documentation/ZorahFull30FpsPlan.md:45
  - Documentation/ZorahFullSwGroup32Default20260929.md:30
---

## 背景 / Context

漫游 30 fps 的验收口径是：固定路线、固定视口/DLSS Quality/LOD 1.5 display px/完整材质与阴影下，p99 ≤ 33.33 ms，并且最终路线上的采样帧无超预算。这条门槛在多轮验收里一直未闭合。

## 现状 / Current state

- 2026-09-30 最终生产基线（2560×1440 输出 / 1707×960 DLSS Quality、独立进程三轮、每轮预热 10 s + 采样 30 s）：
  - 帧均值中位数 16.779 ms（约 59.6 FPS）；P95 24.12–25.18 ms；P99 27.81–28.97 ms；最大 37.63–59.55 ms。
  - 每轮仍有 **2 / 4 / 2 帧超过 33.333 ms**（0.11% / 0.23% / 0.11%），最长连续 1 帧；三轮 `sampledRouteMeets30Fps` 均为 **false**。
- 该基线 GPU 子范围（三轮均值）：Graph 13.05 ms、VBuffer 9.80 ms（Traversal 3.45、SW early 2.08、SW late 0.13、HW early 1.26）、Deferred 1.06、DLSS 1.11。
- 前一轮同口径的默认 group32 漫游（1797×660 / 1198×440）反而是均值 23.56 ms、P95 33.46 ms、70 帧超预算、最长连续 2 帧，并记录了每帧 1 次、共 1274 次页面准入失败。
- 同一文档明确：不计算新旧精确加速比（代码与默认着色路径已变），也不把不同分辨率的数字混用。

## 问题或缺口 / Problem

1. 指标已经接近门槛（P99 27.8–29.0 ms），但**布尔门是逐帧判定**，2–4 个长帧就把整轮判负；当前没有把长帧与具体原因（首访缺页、纹理升级、CLAS/BLAS 重建、pacing 抖动、后台负载）对齐的分析。
2. 两条历史路线（1440p 生产 / 660p group32）结论不一致，缺一个统一口径的「≥3 轮长漫游 + 分段（首次进入/近墙/转身/返回）」报告。
3. 2026-09-30 之后进入 CLAS 细分模板、strand、VSM 等主线，没有再做过同口径漫游基线，当前状态未知。

## 期望结果 / Expected outcome

- 把「是否达到 30 fps」拆成两个可核对的事实：分段 p99 ≤ 33.33 ms，以及超预算帧的逐帧原因归因（每类原因的帧数与最长连续段）。
- 若长期只能做到 2–4 个长帧，需要一个经过讨论的判定口径（例如「超预算帧 ≤ N 且单帧 ≤ M ms」），而不是继续让布尔门恒假。

## 验收标准 / Acceptance criteria

- [ ] 用 `pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -RouteConfig Tools/Perf/ZorahFullBaseline.Live.json -NoVSync` 在当前 HEAD 复跑，记录 `sampledRouteMeets30Fps`、p50/p95/p99/max、超 33.33 ms 帧数与最长连续段。
- [ ] 每一个超预算帧按「首次访问 / 回访 / 纹理升级 / CLAS-BLAS 重建 / 后台负载」归因，并给出对应 scope 的当帧数值（不是平均值）。
- [ ] 分段报告：首次进入、近墙、转身、返回各自的 P95/P99 与超预算帧数。
- [ ] 同轮记录 geometry/CLAS/scratch/texture/BLAS 各池的有效与物理字节、整卡峰值，证明没有靠缩小工作集换取帧时。
- [ ] 若提出放宽判定口径，需在 issue 中给出理由与替代门（含回归风险），并由维护者确认后再改验收文本。

## 相关文档 / References

- [ZorahFullPerformanceBaselineAfterNsightFix20260930.md](../../ZorahFullPerformanceBaselineAfterNsightFix20260930.md)
- [ZorahFull30FpsPlan.md](../../ZorahFull30FpsPlan.md)
- [ZorahFullNextOptimizationPlan.md](../../ZorahFullNextOptimizationPlan.md)
