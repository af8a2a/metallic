---
id: "0058"
title: "RenderGraph: 文档仍称 self-submission 无 GPU 计时、跨队列边未支持，与实现不符"
area: rhi
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:rhi", "kind:task", "priority:P2"]
evidence:
  - Documentation/RenderFrameLifetime.md:156-157
  - Documentation/ProjectArchitecture.md:705-706
  - Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:1800-1824
  - Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:1856-1869
  - Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:3724
  - Source/Editor/EditorApplication.cpp:7172
---

## 背景 / Context

编辑器的实时预览走 `graphExecutor_->execute(RenderGraphSubmitDesc{...})`
（`EditorApplication.cpp:7172`），即 self-submitting / 多队列路径。性能判断、GPU
归因与「哪些 pass 真的有 GPU 时间」都依赖这条路径的计时与依赖能力。

## 现状 / Current state

代码已经比文档更靠前：

- `beginInterval()` / `endInterval()` 按 `timingQueueIndex()`（Graphics / Compute /
  Copy）选择各自的 timestamp query pool 写 bottom-of-pipe 时间戳；query 用尽时置
  `profilingOverflow` 并跳过。
- `resolveGpuTimings()` 逐队列读取 query 结果，把每帧、每节点、每 section 的
  `gpuMilliseconds` / `gpuTimingAvailable` 填回 `RenderGraphExecutionStats`，
  因此 compute/copy 队列上的 pass 也有 GPU 毫秒值。
- self-submission 入口本身开启计时：
  `const bool graphicsTimings = desc.graphicsQueue && impl_->gpuTimingSlots[0].queryPools[0];`
  并在首个 segment 调用 `beginGpuTiming(*commands)`。

但文档仍是旧结论：

- `RenderFrameLifetime.md:156-157`：「GPU timing collection for self-submission
  remains unavailable; existing external command-buffer timestamp collection is
  unchanged.」——该行内容最后一次修改于 2026-09-04，而上述按队列计时能力在 10 月
  才成形。
- `ProjectArchitecture.md:705-706` 的「当前架构约束」写着「跨队列资源边尚不支持」
  「尚无 transient aliasing」，两者都与后来实现的多队列 timeline 等待与 opt-in
  aliasing 冲突。

另外确实仍存在的限制（已由 0038 跟踪）：Tracy 发布要求 `slot.frameTimer.queue == 0`
且所有 node 都在 graphics 队列，因此**只要执行里出现 async compute/copy 节点，整帧
就不产生 Tracy GPU zone**，即使各节点的 GPU 毫秒数已算出。

## 问题或缺口 / Problem

1. 能力被系统性低估：按 `RenderFrameLifetime.md` 规划的人会以为实时路径没有 GPU 计时，于是另建 profiler。
2. 新实现缺显式验证：per-queue pool 选择、`profilingOverflow` 跳过、跨队列 section
   合并是 10 月新增的，没有回归；编辑器 Statistics 面板读到的是未经测试的路径。
3. 三处资料（`RenderFrameLifetime.md`、`ProjectArchitecture.md`、`TracyGpuProfiling.md`）
   对同一能力给出不同甚至相反的结论，读者无法判断权威来源。

## 期望结果 / Expected outcome

- 文档统一到实际实现：self-submission 路径会采集 Graphics/Compute/Copy 三个队列的
  pass 级 GPU 时间，并写明跳过条件（`timestampQueries` 不支持、
  `timestampValidBits == 0`、query 槽位耗尽、无有效 `RenderFrameContext`）。
- `ProjectArchitecture.md` 第 15 节按已实现能力重写（跨队列 timeline 等待已存在；
  transient aliasing 为 opt-in、默认关闭）。
- 为该路径补 GPU 计时回归，并把「Tracy 仅发布纯 graphics 执行」的现状写成显式限制
  （详细验收在 0038）。

## 验收标准 / Acceptance criteria

- [ ] 新增 RHI 用例：`execute(RenderGraphSubmitDesc)` 下产生 async compute（必要时
      copy）节点，断言这些节点的 `gpuTimingAvailable == true` 且
      `gpuMilliseconds > 0`，且统计里 section 数与该队列实际 `used` 区间一致：
      `build\tests\MetallicRHITests.exe --gtest_filter='*render_graph_multi_queue_submit*:*render_graph_execution_viewer*' --rhi-validation --rhi-async-compute`
- [ ] 槽位耗尽分支有可观察信号（`profilingOverflow` 进入执行统计或日志），并在用例中
      被覆盖一次。
- [ ] `Documentation/RenderFrameLifetime.md`、`ProjectArchitecture.md`、
      `TracyGpuProfiling.md` 三处描述一致，且明确指出 Tracy 的 graphics-only 限制。
- [ ] 编辑器 Statistics 面板在该构建下显示非零 GPU 列（用既有 `EditorProfiler`
      smoke 路径断言，不要求人工截图）。

## 相关文档 / References

- [RenderFrameLifetime.md](../../RenderFrameLifetime.md)
- [TracyGpuProfiling.md](../../TracyGpuProfiling.md)
- [ProjectArchitecture.md](../../ProjectArchitecture.md)
- [RenderGraphExecutionViewer.md](../../RenderGraphExecutionViewer.md)
- 相关 issue：0038（Tracy 异步提交 GPU 时间轴）
