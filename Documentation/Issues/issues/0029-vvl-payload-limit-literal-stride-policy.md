---
id: "0029"
title: "RHI: VVL 误报 task/mesh payload 上限，literal descriptor heap stride 策略被迫常驻"
area: rhi
kind: task
priority: P2
state: blocked
milestone: M2-Correctness
labels: ["area:rhi", "kind:task", "priority:P2", "state:blocked"]
evidence:
  - Documentation/NativeDescriptorHeapStrideWorkaround.md:1-32
  - Documentation/NativeDescriptorHeapStrideWorkaround.md:229-278
  - Source/Runtime/Render/GAPI/ShaderTarget.h:20-22
  - Source/Runtime/Render/GAPI/Vulkan/VulkanDescriptorHeapShaderStrides.cpp:118
  - Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp:10897
  - Source/Runtime/Render/GAPI/Vulkan/DescriptorHeapShaderABI.h:13
---

## 背景 / Context

Native descriptor heap（`spvDescriptorHeapEXT`）下 Slang 把统一 heap stride 表达为
`OpConstantSizeOfEXT -> OpSpecConstantOp UGreaterThan -> OpSpecConstantOp Select`。
Validation Layers 只要模块里出现过任何 `OpSpecConstantOp`，就把整个模块标记为含
spec constant，payload 构造器写入 `kSpecConstant`（4294967293），task/mesh payload
上限检查随后把这个标记当字节数，产生假 `08758` / `08755`。引擎因此改为：运行时查询
所选设备的 `SPIRVResourceHeapStride` / `SPIRVSamplerHeapStride`，以字面量
`OpDecorate ArrayStride` 编译，并移除原来的 `OpConstantSizeOfEXT` 尺寸改写。

## 现状 / Current state

- 策略仍在仓库里，并带显式移除标记 `TO-REMOVE(VVL payload-size)`：
  `ShaderTarget.h:20-22`（编译策略与文档指引）、`VulkanDescriptorHeapShaderStrides.cpp:118`、
  `VulkanRHI.cpp:10897`（literal stride 发射）、`DescriptorHeapShaderABI.h:13`（只读 ABI 检查）。
- 对照实验未给出解除条件：把原始 `OpConstantSizeOfEXT` 表达式放回去后，官方 VVL 1.4.350 /
  1.4.363 仍报 `08758` / `08755`；synthetic raw unified-stride pair 还报 `08760`。
- 检查未被削弱：字面量 128 B 控制零 VUID，真实 20,000 / 32,768 B 超限仍正确报错。
- 上游未修：审计的 VVL upstream main `10de01c3` 仍保留同一模块级 spec-constant payload
  构造器（仅源码核对）。
- 结论在文档中已明确写为「literal-stride policy remains required」，且这是一个持续
  维护成本很高的设备绑定策略。

## 问题或缺口 / Problem

字面 stride 让每个 native 模块与具体设备绑定：stride 不匹配或仍含 opaque-size 表达式的
native SPIR-V 会在 module 创建前被拒绝并要求重编译；离线编译必须显式提供 stride；
stride 进入 native 编译器 cache identity。删除该策略会立刻恢复假 payload VUID，
因此它不能作为「临时实验」被顺手清理，需要上游修复才能退场。

## 期望结果 / Expected outcome

上游 VVL 修正 payload 上限对模块级 spec-constant 标记的处理（或 Slang 不再以
`OpSpecConstantOp` 表达统一 stride）后，按
`NativeDescriptorHeapStrideWorkaround.md:255-278` 的清单一次性退回
`SPIRVUnifiedDescriptorHeapStride`，恢复设备无关输出并再次 bump compiler request
version，防止缓存里旧的字面 ABI 二进制被当成设备无关产物复用。

## 验收标准 / Acceptance criteria

- [ ] 未修改的 native task / masked mesh 编译输出（保留 `OpConstantSizeOfEXT` 与统一
      stride 表达式）在记录的官方 VVL 版本下创建 graphics pipeline，无
      `08758` / `08755` / `08760`；validation callback 结果与 `VkResult` 同时记录。
- [ ] 128 B 字面 payload 控制零 VUID；真实 20,000 / 32,768 B 超限仍报对应错误。
- [ ] `MetallicRHITests --rhi-validation --filter descriptor_heap_code_pattern` 与 native
      嵌套布局 / 最终 descriptor index / heap 切换 / 混合 32-64 位原子 / 采样回读用例全部通过。
- [ ] `grep -rn "TO-REMOVE(VVL payload-size)" Source tests` 为空，且 SPIR-V 尺寸改写 pass 未回归。
- [ ] 报告记录 VVL 版本 + DLL SHA-256、驱动版本、Slang 版本。

## 相关文档 / References

- [NativeDescriptorHeapStrideWorkaround.md](../../NativeDescriptorHeapStrideWorkaround.md)
- [DynamicResourceUpgradeStatus.md](../../DynamicResourceUpgradeStatus.md)
- [ResourceAccessABI.md](../../ResourceAccessABI.md)
