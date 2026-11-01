---
id: "0015"
title: "Streamer: Track merged demand 仍逐唯一请求查表，S1 延迟跟踪 ≤0.5 ms 目标未达"
area: geometry
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:geometry", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/ZorahFullStreamerS1.md:3
  - Documentation/ZorahFullStreamerS1.md:11
  - Documentation/ZorahFullStreamerS1.md:42
  - Documentation/ZorahFullStreamerS1.md:50
  - Documentation/ZorahFullStreamerNextSteps.md:50
  - Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:424
  - Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:999-1012
---

## 背景 / Context

S1 的目标是把 Streamer 延迟观测（`Reset / latency tracking` + `Track merged demand`）从 2.25–2.27 ms/帧压到 ≤0.5 ms/帧。该区间每帧要处理约 1.35 万个页面请求、6.2 万个驻留页，而实际执行准入只有 42–44 次，观测成本与真实工作量严重不成比例。

## 现状 / Current state

- 已落地：合并反馈批次只读一次时钟、pending 哈希表改为 1024 项按需索引块、256 桶时间轮替代 `beginFrame` 全表扫描、惰性续期。
- 实测（2026-09-24 最终版三轮）：两区间合计 **0.98–1.30 ms/帧**，P95 1.54–2.03 ms，相比基线 2.16–2.80 ms 下降约一半，但**仍未达到 0.5 ms 验收目标**。
- 同一轮 Stream Begin 5.39–6.97 ms，上传 974.6–978.7 页/秒（基线 981–990），loadFailures/requestOverflows 均为 0。
- 该采样不是隔离交错 A/B：采样期间另有两个 Edge 进程各占约 81% 单核等效 CPU，且 GPU 进程遥测为空，因此现有数字只能作为上界参考。

## 问题或缺口 / Problem

1. `Track merged demand` 仍对每个唯一请求执行一次 `latency_->request(...)`，并在其中查页表与读时钟；合并去重只消除了重复请求，没有消除「每帧一次的逐请求成本」。
2. S1 文档明确留下的下一步——复用合并需求已经取得的 `pages_` 驻留 / fallback 查询结果——**没有落地，也没有对应的实测**。在缓存失效边界（resident、fallback、取消、重试）未定义前，不能简单用「已有 pending」跳过驻留检查，否则会漏记需求或污染尾延迟统计。
3. 缺少 `demand→drawable` 直方图的逐帧导出，真实 Full 尾延迟分布的前后对照尚未完成。

## 期望结果 / Expected outcome

- 合并需求的驻留 / fallback 查询只做一次，并复用到延迟跟踪与准入过滤两处，失效边界写成文档化契约。
- 两区间合计稳定 ≤0.5 ms/帧、P95 ≤1.0 ms，且延迟统计语义（样本数、abandoned、pending、首末需求时间）与现状逐项等价。

## 验收标准 / Acceptance criteria

- [ ] 空闲 GPU、隔离负载下跑 `pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build/<new-dir> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -RouteConfig Tools/Perf/ZorahFullBaseline.Live.json`，三轮逐帧合计（`Reset / latency tracking` + `Track merged demand`）均值 ≤0.5 ms、P95 ≤1.0 ms。
- [ ] 同一批 `Frames.jsonl` 中 `demand→drawable` 样本数、abandoned、pending、P50/P95/P99 与改动前逐项对照，不得通过漏记达标。
- [ ] 上传吞吐 ≥970 页/秒，loadFailures = requestOverflows = 0。
- [ ] `MetallicRHITests --rhi-validation --gtest_filter='*streamer*:*stream_initial_loading:*streaming_task_queue*'` 全绿，覆盖取消、重试、预取升级与页 ID 复用。
- [ ] 结论中明确区分「CPU scope 下降」与「整帧改善」，不把节省的 CPU 毫秒换算成等额 FPS。

## 相关文档 / References

- [ZorahFullStreamerS1.md](../../ZorahFullStreamerS1.md)
- [ZorahFullStreamerNextSteps.md](../../ZorahFullStreamerNextSteps.md)
- [Streamer.md](../../Streamer.md)
