---
id: "0032"
title: "RHI: 为规避 Nsight 悬垂引用，命令缓冲在抓取期永不释放并全局复用"
area: rhi
kind: task
priority: P2
state: blocked
milestone: M3-Hardening
labels: ["area:rhi", "kind:task", "priority:P2", "state:blocked"]
evidence:
  - Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:4464-4485
  - Documentation/NsightCommandBufferCaptureInvestigation.md:32-67
  - Documentation/NsightCommandBufferCaptureInvestigation.md:128-171
---

## 背景 / Context

Nsight Graphics 2026.3.1 在抓取注入期间会保留已释放 `VkCommandBuffer` 包装对象的引用。
用户提供的 11:39 dump 在 `ngfx-capture-interception.dll + 0xec137` 崩溃，`rcx == 0`，
读取地址 `0x1a8`；同一地址在失败运行中已被编辑器属性数据（`resetSerial`、`camera.eye`）
复用。RTTI 确认该对象类型是 Nsight 私有 `NV::Pylon::Capture::Vulkan::CommandBuffer`，
即这是悬垂引用而非缺少可选特性。11:33 的 `reset-before-free` 方案不足以解决交互场景。

## 现状 / Current state

- 引擎侧落地的是「重置而不释放」：`CommandBuffer::~CommandBuffer()` 在
  `impl_->capturePool != nullptr` 时执行 `vkResetCommandBuffer` 并把原生 handle 推回
  `capturePool->availableBuffers`，而不是 `vkFreeCommandBuffers`（`VulkanRHI.cpp:4464-4485`）。
  退役的 pool 按设备/队列族缓存复用，池与缓冲只在设备销毁（GPU idle 与外部库关闭之后）时销毁。
- 复用范围只覆盖抓取注入会话；普通执行保持立即释放。
- 已通过的验证（本地证据 `.tmp/nsight-command-lifetime/`）：full-size SDK 抓取 + teardown
  exit 0；一个进程内三次抓取 + 两次 resize（1198x438 / 878x258 / 1198x438）exit 0；
  外部注入 frame-600 抓取 exit 0；两者的 `ngfx-replay --loop-count 3` 均报告 replay 成功。
- `VulkanRHI.cpp:4476-4477` 留有 `TODO: Remove after an updated Nsight passes the full-size
  capture and resize regression without reuse`。

## 问题或缺口 / Problem

这是一个常驻 RHI 生命周期分叉：抓取会话里命令缓冲与 command pool 的内存只增不减，
reset 会丢弃录制内容和 descriptor heap 预留，command allocator 内存保持在高水位直到设备
销毁；调用方必须自行保证退休前已完成 GPU 工作（没有额外 GPU wait）。它与「正常释放」
两条路径长期并存，是后续任何命令缓冲生命周期改动的额外约束。

## 期望结果 / Expected outcome

更新后的 Nsight 运行时在**不使用**该复用的情况下通过 full-size 抓取、resize、同进程多次
抓取与 teardown 回归；随后删除 `capturePool` 相关分支与池缓存，恢复单一释放路径。

## 验收标准 / Acceptance criteria

- [ ] 未启用 capture-only 复用时，`METALLIC_NSIGHT_GRAPHICS_CAPTURE=1` +
      `METALLIC_SMOKE_TEST_NSIGHT_CAPTURE=1` 的 `MetallicGPUDrivenSample --smoke-test`
      在 viewport 分辨率下完成三次抓取（中间 resize 至 1280x720 再回 1600x900）并 exit 0。
- [ ] 上述抓取产物逐个 `ngfx-replay --present-hidden --loop-count 3` 退出 0。
- [ ] 外部注入（`ngfx-capture.exe --capture-frame <N>`）与无注入普通运行均 exit 0。
- [ ] 删除复用分支后 `grep -rn "capturePool" Source/Runtime/Render/GAPI` 为空，
      且不再出现 capture-only 的高水位内存保留。
- [ ] 报告记录 Nsight 版本与驱动版本；交互式 Graphics Debugger 与 ZorahFull 的未覆盖面明确写出。

## 相关文档 / References

- [NsightCommandBufferCaptureInvestigation.md](../../NsightCommandBufferCaptureInvestigation.md)
- [NsightCaptureRuntimeIntegrity.md](../../NsightCaptureRuntimeIntegrity.md)
