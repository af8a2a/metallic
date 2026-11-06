---
id: "0039"
title: "RHI: Streamline 与 VK_EXT_descriptor_heap 状态互斥，靠绑定空 legacy set 换取 DLSS 输出"
area: rhi
kind: task
priority: P2
state: blocked
milestone: M3-Hardening
labels: ["area:rhi", "kind:task", "priority:P2", "state:blocked"]
evidence:
  - Source/Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp:421-529
  - Source/Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp:1240-1260
  - Documentation/DlssNr.md:130-151
---

## 背景 / Context

`VK_EXT_descriptor_heap` 的 heap 命令与 legacy descriptor-set 命令会互相使对方的状态失效。
Streamline 2.11.1 仍按 descriptor-set 方式录制 DLSS 工作，而上游 NVIDIA-RTX/Streamline#109
报告 heap 保持激活时输出为黑。引擎因此在 `evaluate` 之前绑定一个空 legacy set，并通知 RHI
在下一个引擎 dispatch 前重新绑定自己的 heap。

## 现状 / Current state

- 实现位置：`VulkanStreamline.cpp:421-529`。`initializeDescriptorHeapWorkaround` 在
  `device.descriptorHeapEnabled` 时创建空的 set layout / pipeline layout / pool / set，
  失败时整体回滚并返回 `Error::Failure`；`prepareDescriptorStateForStreamline` 在
  compute bind point 绑定该空 set；`destroyDescriptorHeapWorkaround` 负责销毁。
- 该兼容处理在设备初始化阶段即启用（`:1240-1260`），并在状态快照里作为
  `descriptorHeapWorkaround` 发布（`:923`），便于诊断。
- DLSS-NR 路径复用同一 workaround：evaluate 前绑定 Streamline 的空 legacy set，NGX dispatch
  之后让自身缓存的 descriptor 绑定状态失效以便后续 pass 重新绑定（`DlssNr.md:148-150`）。
- 代码注释明确标注依赖上游版本（Streamline 2.11.1）与上游 issue 编号。

## 问题或缺口 / Problem

这是一个「两套描述符状态模型互斥」的常驻旁路：它每次 evaluate 都要插入一次
`vkCmdBindDescriptorSets`，并依赖 RHI 侧「被外部改写了绑定状态」的隐式约定；
新增加的外部 SDK 调用点（如 DLSS-NR、后续 SR/RR 变体）都必须复制这套约定，否则出现黑帧或
错误资源绑定。上游修复后应整体删除，但目前没有任何回归会提示它可以移除。

## 期望结果 / Expected outcome

升级到不再要求 legacy descriptor-set 的 Streamline 版本（或上游 #109 关闭）后，删除空 legacy
set 的创建/绑定/销毁与 RHI 的重新绑定提示，让 DLSS evaluate 直接工作在 descriptor heap 状态下。

## 验收标准 / Acceptance criteria

- [ ] 记录 Streamline 版本与驱动版本；在原型分支上去掉 workaround 后 DLSS-SR / DLSS-RR
      输出非黑（保存对比图像），SSL/Reflex 冒烟通过。
- [ ] `MetallicRHITests --rhi-streamline --filter dlss_nr_runtime` 与
      `dlss_nr_runtime_slider` 的逐字节回归通过（参考序列在独立 feature 生命期内采集）。
- [ ] 删除后 `grep -rn "descriptorHeapWorkaround\|prepareDescriptorStateForStreamline" Source` 为空，
      且不再存在「外部 SDK 改写绑定状态后需手动失效 RHI 缓存」的隐式约定。
- [ ] 若上游仍未修复，文档化该限制并在每个新增 Streamline 调用点显式标注同样的要求。

## 相关文档 / References

- [DlssNr.md](../../DlssNr.md)
- [NvLowLatencyVk.md](../../NvLowLatencyVk.md)
- [ResourceAccessABI.md](../../ResourceAccessABI.md)
