---
id: "0031"
title: "Tooling: Nsight 注入期间禁用 Aftermath 自动 checkpoint，GPU 进度定位能力缺失"
area: tooling
kind: task
priority: P2
state: blocked
milestone: M2-Correctness
labels: ["area:tooling", "kind:task", "priority:P2", "state:blocked"]
evidence:
  - Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:1236-1262
  - Documentation/NsightKhrOpacityMicromapInvestigation.md:100-142
---

## 背景 / Context

Nsight Graphics 注入与 Aftermath 同时开启时，`VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV`
会触发 RTAS 相关的 GPU page fault。原始复现是 async Sponza 场景：103 个 BLAS、10 个 OMM、
34,908 个 alpha 三角形、异步上传 + compaction + TLAS，日志报 `DeviceLost` 与
`Error_DMA_PageFault`（dump `.cache/aftermath/Metallic_Engine_Editor-50480-1.json`）。

## 现状 / Current state

- 已做的 A/B（Nsight 2026.3.1 / driver 616.64）：
  - 注入开 + Aftermath 开 + 无诊断改动 → `DeviceLost` / `Error_DMA_PageFault`；
  - 只关 shader debug info → 同样 GPU fault；
  - 只关 resource tracking → 同样 GPU fault；
  - 只关 automatic checkpoints → 通过。
- 生产实现（`VulkanRHI.cpp:1236-1262` 的 `configureAftermathDiagnostics`）：当
  `vulkan::toolingHooks().captureInjected()` 为真时默认清掉 automatic checkpoint 位，
  保留 crash dump、resource tracking 与 shader debug info，并打印说明性 warn；
  `METALLIC_AFTERMATH_AUTOMATIC_CHECKPOINTS=1|0` 可强制覆盖。
- `VulkanRHI.cpp:1244-1248` 留有明确的 `TODO(Nsight Aftermath)` 指出恢复条件，并交叉引用
  `NsightKhrOpacityMicromapInvestigation.md`。

## 问题或缺口 / Problem

自动 checkpoint 是定位「注入期 DeviceLost 时 GPU 执行到哪一步」的主力手段。注入期间把它
默认关掉，等于在唯一同时需要两种工具的场景里放弃该能力：崩溃只能得到 dump 与资源追踪，
缺少自动 checkpoint 的进度标注。目前根因未确认（NVIDIA 实现内部），属于需要上游修复的
阻塞项，而不是可自行绕过的配置问题。

## 期望结果 / Expected outcome

更新 Nsight Graphics / 驱动后，在注入 + Aftermath 同时开启、automatic checkpoint 打开
的条件下完成复现用例与完整 realtime 回归；确认无 `Error_DMA_PageFault` 后删除
`configureAftermathDiagnostics` 里的注入期分支，使诊断标志不再依赖运行时注入状态。

## 验收标准 / Acceptance criteria

- [ ] `MetallicRHITests --rhi-aftermath --rhi-async-compute --filter scene_acceleration_structure_pass_standard`
      在 Nsight 注入进程（`METALLIC_NSIGHT_GRAPHICS_CAPTURE=1` 或外部注入）中开启 automatic
      checkpoints 后通过，且无 `Error_DMA_PageFault` / `DeviceLost`。
- [ ] 编辑器 async RTAS 路径（BLAS + OMM + compaction + TLAS）在同一注入进程中完成一整帧抓取
      与 `ngfx-replay` 回放，退出 0。
- [ ] 删除注入期分支后 `grep -rn "TODO(Nsight Aftermath)" Source` 为空，且
      `METALLIC_AFTERMATH_AUTOMATIC_CHECKPOINTS` 覆盖开关一并移除或改为纯诊断保留。
- [ ] 报告记录 Nsight 版本、驱动版本、Aftermath dump 与抓取的对应关系。

## 相关文档 / References

- [NsightKhrOpacityMicromapInvestigation.md](../../NsightKhrOpacityMicromapInvestigation.md)
- [DescriptorHeapGpuFault.md](../../DescriptorHeapGpuFault.md)
- [NsightCaptureReplayInvestigation.md](../../NsightCaptureReplayInvestigation.md)
