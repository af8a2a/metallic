---
id: "0036"
title: "RHI: 跨 shaderObject 配置复用驱动内部 pipeline cache 触发 DeviceLost，根因未确认"
area: rhi
kind: defect
priority: P1
state: blocked
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P1", "state:blocked"]
evidence:
  - Documentation/DescriptorHeapGpuFault.md:39-114
  - Documentation/DescriptorHeapGpuFault.md:230-253
  - Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:10225-10245
  - tests/rhi/DescriptorHeapCodePatternTests.cpp:230-250
---

## 背景 / Context

2026-09-09 的对照表明：设备启用 `shaderObject` **不是**充分的故障条件，驱动内部 pipeline
cache 的复用参与了故障。同一测试二进制、同一原始生产 SPIR-V、相同设备特性与负载，只切换
`VkDevicePipelineBinaryInternalCacheControlKHR::disableInternalCache`，结果由通过变为
`DeviceLost`。应用 pipeline cache（`.pso`）是第二层：旧 `VisibilityBufferPass.pso` 在
内部缓存关闭时仍会失败。

## 现状 / Current state

- 首次编译顺序对照（同一组指令、2426 words，仅改 SPIR-V generator word 生成两个 pipeline key）：
  `SO=false` 首次编译 → `SO=true` 执行 = `DeviceLost`；同一缓存回到 `SO=false` 执行 = 通过；
  同 shader 在 `SO=true` 重新编译（禁用内部缓存）= 通过；`SO=true` 首次编译 → `SO=false` 仅编译
  → `SO=true` 执行 = 通过。所有行 global key 均为 `725085dc37461347`。
- 原始 shader 缓存 A/B（同一二进制 SHA-256）：内部缓存 `disabled` 通过（410 ms），
  `enabled` 时 `frame.wait()` 返回 `DeviceLost`（3616 ms）。
- 完整 VBuffer 路径：仅关内部缓存仍失败（Aftermath 显示 `Error_DMA_PageFault`，最早推进到
  `GPUSceneSubsystem::recordInstanceCull()` 的 reset dispatch）；再隔离旧应用 `.pso` 后
  完整测试约 12 秒通过。
- 诊断开关均为显式启用、默认关闭：`METALLIC_VK_INTERNAL_PIPELINE_CACHE`、
  `METALLIC_VK_LOG_PIPELINE_KEYS`、`METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST` 等
  （`VulkanRHI.cpp:10225-10245`、测试侧 `DescriptorHeapCodePatternTests.cpp:230-250`）。
  GPU 执行要求关闭内部缓存或显式允许 DeviceLost，否则 skip；compile-only 不受限。

## 问题或缺口 / Problem

现有证据只能说明「不兼容的编译结果被跨配置复用」，无法指认驱动内部缓存哪一层或哪段实现
存在缺陷；A/B 使用了不同 nonce，没有对同一冷 key 反转首次编译顺序，且该开关同时改变扩展与
feature 启用状态，本轮未单独区分。需要厂商结合缓存 binary 才能确认，属于上游阻塞项。
同时旧 `.pso` 与应用/驱动两层缓存的交互也未解释。

## 期望结果 / Expected outcome

在厂商定位前，把「故障配置不得进入日常自动回归」写成制度性约束（故障配置会造成短暂 GPU hang /
桌面卡顿），并把通过配置固定为诊断基线；厂商修复后按同一 A/B 复测再考虑放开默认值。

## 验收标准 / Acceptance criteria

- [ ] 提交缓存 binary 与 pipeline key 日志给 NVIDIA，并以「跨 `shaderObject` 配置的 pipeline key
      未区分编译配置」为假设记录结论（含 global key / pipeline key 值）。
- [ ] 修复后：`METALLIC_VK_INTERNAL_PIPELINE_CACHE=enabled` +
      `MetallicRHITests --rhi-validation --filter descriptor_heap_code_pattern` 通过且不再需要
      `METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST=1`。
- [ ] 同一冷 key 反转首次编译顺序的对照（无 nonce 混杂）完成，注明 `shaderObject` 扩展与 feature
      的独立贡献。
- [ ] 旧 `.pso` + 内部缓存关闭的组合复测，说明应用缓存是否需要独立失效策略。
- [ ] 不把 SPIR-V generator nonce 扰动写成修复方案；`METALLIC_GPU_PATTERN_SPIRV_GENERATOR`
      保持诊断性质。

## 相关文档 / References

- [DescriptorHeapGpuFault.md](../../DescriptorHeapGpuFault.md)
- [NativeDescriptorHeapStrideWorkaround.md](../../NativeDescriptorHeapStrideWorkaround.md)
- [DynamicResourceUpgradeStatus.md](../../DynamicResourceUpgradeStatus.md)
