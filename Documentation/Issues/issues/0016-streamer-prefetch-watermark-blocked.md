---
id: "0016"
title: "Streamer: Full 场景预测预取恒被内存水位阻断（预取请求=0），预取与准入缺少拒绝原因分类"
area: geometry
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:geometry", "kind:perf", "priority:P2"]
depends_on: ["0017"]
evidence:
  - Documentation/StreamingMeshletLod.md:111
  - Documentation/StreamingMeshletLod.md:78
  - Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:1629-1635
  - Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:686-690
  - Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:5489-5490
  - Documentation/ZorahFullPerformanceBaselineAfterNsightFix20260930.md:43
---

## 背景 / Context

`predictivePrefetch` 默认 `true`，按相机位移/朝向外推预取相机，提前量取近期 demand→drawable P95 加一帧（限 25–250 ms），并受「请求名额 1/4 + 空闲上传名额」约束。设计意图是减少首次转向与进入新区的长帧。

## 现状 / Current state

- 2026-09-30 Full 固定路线（2560×1440 输出 / 1707×960 内部、360 个固定姿态、静态与预测各一轮）实测：**两组实际预取请求均为 0**。
- 原因是几何驻留约 3.30–3.76 GB，而预算为 3.76 GB（`maxResidentBytes = 3758096384`），常驻长期超过预取水位；新增的 `prefetchMemoryWatermarkBlocked` / `prefetchQueueBlocked` 计数把这一限制显式暴露出来。
- `canPrefetchPage()` 的判定是「可用容量地板 = `capacity - prefetchReserveBytes_`，未设置时取 `capacity * 3 / 4`」；在 Full 的 3.5 GiB 池被根页占满时，该条件几乎恒假。
- 需求侧同样被压满：2026-09-30 生产三轮中每帧分配拒绝 0–1 次（累计 1309–1329 次/1788 帧），属可解释的预算压力，但按帧 schema 只导出聚合计数。

## 问题或缺口 / Problem

1. 预测预取这条链路在目标场景里**从未真正执行**，无从证明收益；文献中被引用的「预取晋升」「提前量」统计在当前 Full 配置下恒为空。
2. 缺少「为什么没预取」的结构化归因：水位、上传名额、以及需求/预取冲突各自贡献多少，代码只有两个布尔计数，没有分类直方图。
3. 准入拒绝只按帧累加，未区分无可用空间、碎片、无可回收冷页、CLAS/联合预算等来源，也没有「资源变化 epoch 驱动重试」的等价机制来替代逐帧整批重试。

## 期望结果 / Expected outcome

- 在根页容量问题（见 0017）缓解后，让预测预取在 Full 路线上至少在一段可复现区间内真实触发，并按源帧去重统计命中的缺页追赶下降。
- 预取/准入的拒绝原因可分类观测，作为容量策略决策依据。

## 验收标准 / Acceptance criteria

- [ ] 固定路线两轮（`predictivePrefetch=false/true`）导出的 `prefetchRequests` 不再恒为 0，且 `prefetchMemoryWatermarkBlocked` / `prefetchQueueBlocked` 与 `prefetchRequests` 之和可解释每一帧的请求名额使用。
- [ ] 同路线「本页缺失阻塞 group-frame」「缺页追赶进入 cut 的 cluster 次数」按反馈源帧去重后，预测组相对静态组给出方向一致的下降，或明确记录为何无收益。
- [ ] 新增拒绝原因分类计数（至少：无连续空间、碎片、无冷页可回收、CLAS 预算、上传名额）并进入 `Frames.jsonl` / Profiler，合计与现有 `allocationFailures` 对齐。
- [ ] `MetallicRHITests --rhi-validation --gtest_filter='*streamer*:*stream_clas_cut_readiness:*meshlet_lod_stream_rt_demand'` 全绿。
- [ ] 结论不把「预取请求数增加」当成收益，必须同时报告帧 P95/P99 与内存峰值是否退化。

## 相关文档 / References

- [StreamingMeshletLod.md](../../StreamingMeshletLod.md)
- [ZorahFullNextOptimizationPlan.md](../../ZorahFullNextOptimizationPlan.md)
- [ZorahFullPerformanceBaselineAfterNsightFix20260930.md](../../ZorahFullPerformanceBaselineAfterNsightFix20260930.md)
