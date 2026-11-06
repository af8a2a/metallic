---
id: "0035"
title: "RHI: AS handle 无法统一为 heap index —— 驱动 heap 数值 load 错误与 Slang hook native 崩溃"
area: rhi
kind: defect
priority: P2
state: blocked
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P2", "state:blocked"]
evidence:
  - Documentation/AsHandleInvestigation.md:1-30
  - Documentation/AsHandleInvestigation.md:27-106
  - Source/Runtime/Render/GAPI/Vulkan/NativeDescriptorHeapSPIRV.h:130-140
  - Documentation/DynamicResourceUpgradeStatus.md:84-92
---

## 背景 / Context

普通 buffer/图像已经迁移到 dynamic resource（最终 `shaderIndex` + `ResourceHeapEXT`），
但 AS 仍是例外：正式路径传完整 64-bit AS device address 并经显式 resolver 转成 AS。
Slang 2026.18.2 的 native AS 分支从 `ResourceHeapEXT` 读 uint64 再
`OpConvertUToAccelerationStructureKHR`，其索引单位是 8 字节，而 Vulkan RHI 的 AS
allocator 复用 buffer slot，`shaderIndex` 单位是 `bufferDescriptorSize`（本机 16）。
同一个 heap 字节偏移在两者中应分别编码为 `B/8` 与 `B/16`。

## 现状 / Current state

- 独立 Vulkan + 手写 SPIR-V 复现（不经 Metallic 的 heap 管理、shader cache 与 SPIR-V 修复）：
  CPU 写入 heap byte 128 的 `0x1122334455667788` 正确；`ResourceHeapEXT` + `OpLoad uint64`
  读同一字节返回错误值（纯 buffer probe 返回 0，带 ray query 的 probe 返回其他值）；
  同一字节经 `PhysicalStorageBuffer` 读取正确；`OpBufferPointerEXT` 读 heap 中的
  storage-buffer descriptor 再读其数据也正确。四份 probe 均通过 `spirv-val --target-env vulkan1.3`，
  最终两次 GPU 日志无 validation 警告。错误值并非恒为零，不能靠零值特判修补。
- Slang 官方定制点 `export getDescriptorFromHandle<T>`：最小 shader（单个
  `DescriptorHandle<RWByteAddressBuffer>`，hook 只转发默认实现）在 **native** 下编译即
  `0xC0000005`，mapped 正常；独立 module 导出再 import 同样崩溃。因此当前不能换成官方 hook。
- 引擎侧已有保护：`NativeDescriptorHeapSPIRV.h:130-140` 对 AS heap-address load 直接给出错误
  并要求使用 `Metallic::resolveDescriptor`；`ResourceRegistry.cpp` 同步显式 AS 地址契约。
- 已验证可行方案（PoC，未集成）：AS slot 写 storage-buffer descriptor → `OpBufferPointerEXT`
  → 普通 buffer 数据中的 uint64 地址 → `OpConvertUToAccelerationStructureKHR`；用非零
  buffer descriptor index 3 的真实 BLAS/TLAS 得到 committed type 1、t = 1.0。尚未测性能。

## 问题或缺口 / Problem

两处上游阻塞叠加：本机驱动对 `ResourceHeapEXT` 的普通数值 load 返回错误值，Slang 的
descriptor hook 在 native 下崩溃。二者都不是仓库内可修的；结果是 AS 无法与 buffer 使用
同一 handle 表示，AS 的索引单位与 buffer 不同，`ResourceHandle<AS>` 的统一契约还不存在。

## 期望结果 / Expected outcome

短期采用已验证的 address-cell 方案并保持 AS 特化在资源后端/resolver 内；长期跟进驱动与 Slang，
恢复直接 opaque AS descriptor load。任何切换都必须通过真实 GPU 回归，不能只调 stride 或删掉
现有编译期保护。

## 验收标准 / Acceptance criteria

- [ ] 独立 probe 在新驱动上复测：`ResourceHeapEXT` + `OpLoad uint64` 与
      `PhysicalStorageBuffer` 对同一 heap 字节返回一致值；若仍不一致，记录驱动版本并保持 resolver。
- [ ] AS 地址单元生产集成：`ResourceHandle<AS>` 与 buffer 使用同一 index 单位，覆盖非零/多 AS
      索引、数组、跨帧替换、refit 与 compaction、partitioned AS 的 GPU 回读。
- [ ] Slang `export getDescriptorFromHandle` 恒等 hook 在 native 下不再崩溃（编译 + 管线创建），
      才允许用官方 hook 替换显式 resolver。
- [ ] AS address-cell 的地址生命周期与在途帧保持一致（rebuild/compaction 变更地址时安全更新）。
- [ ] 提供额外间接访问对 ray-query / path-trace 工作负载的测量，或明确写「未测量」。

## 相关文档 / References

- [AsHandleInvestigation.md](../../AsHandleInvestigation.md)
- [DynamicResourceUpgradeStatus.md](../../DynamicResourceUpgradeStatus.md)
- [AtomicBufferBdaAssessment.md](../../AtomicBufferBdaAssessment.md)
- [ResourceAccessABI.md](../../ResourceAccessABI.md)
