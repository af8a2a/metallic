# AS handle 路径调查（2026-09-25）

## 结论

AS 可以统一成面向 shader 的 heap index；当前阻断来自 Slang 的 AS ABI 特例、本机驱动的 heap 数值读取错误，以及 Slang 自定义 descriptor hook 的 native 编译崩溃。不能仅把现有 AS device address 改回 `shaderIndex`。

已用不依赖 Metallic 或 Slang 的 Vulkan + 手写 SPIR-V 程序复现 heap 读取错误，并验证一个可行的统一 index 方案：AS slot 存储指向地址单元的 storage-buffer descriptor，经正常 buffer load 取得 AS address，再转换为 AS。非零 index 3 的真实 BLAS/TLAS ray query 返回 triangle hit、距离 1.0。

这证明统一 handle 表示可行；当前 PoC 尚未成为生产实现，也未测量额外间接访问的性能。项目正式路径仍使用完整 64-bit AS address 和 `Metallic::resolveDescriptor`。

## 1. 三种数值不能混用

| 路径 | handle 表示 | shader 读取方式 |
| --- | --- | --- |
| Slang 默认 native buffer | buffer descriptor 数组索引 | `ResourceHeapEXT` → `OpBufferPointerEXT` → buffer 数据 |
| Slang 默认 native AS（2026.18.2） | 64-bit 地址单元数组索引，默认 stride 8 | `ResourceHeapEXT` → `OpLoad uint64` → `OpConvertUToAccelerationStructureKHR` |
| Metallic 当前 AS（mapped/native） | 完整 64-bit AS device address | 显式 resolver → `OpConvertUToAccelerationStructureKHR` |

Vulkan RHI 的 AS allocator 复用 buffer slots，`shaderIndex` 的单位是 `bufferDescriptorSize`。本机该值为 16；Slang 默认 native AS 的索引单位却是 8。因而同一个 heap 字节偏移 `B` 在两者中分别应编码为 `B/16` 与 `B/8`。这会导致非零索引错位，尤其容易被 index 0 的测试掩盖。

但修正索引单位只解决寻址；本机在正确地址的 heap 数值 load 仍返回错误值，见下节。Slang 的 `-spirv-unified-descriptor-heap-stride` 也不会覆盖这条 AS 特例；AS 分支读取的是显式 `-spirv-resource-heap-stride`，默认 8，最小 8。[固定版本源码](https://github.com/shader-slang/slang/blob/v2026.18.2/source/slang/slang-emit-spirv.cpp#L7616)

AS 的实际 descriptor 写入大小与数组 stride 也不同。本机 `vkGetPhysicalDeviceDescriptorSizeEXT(AS)` 返回 8，而 `OpConstantSizeOfEXT(OpTypeAccelerationStructureKHR)` 按 Vulkan 契约使用对齐后的 `bufferDescriptorSize`，即 16；不能从前者推导后者。

Vulkan 明确允许 `ResourceHeapEXT` 中的 AS descriptor 对应 `OpTypeAccelerationStructureKHR`，也允许 heap 指针指向普通数据类型。因此 uint64-address lowering 是当前 Slang 的实现选择，并非 Vulkan 要求 AS 永远绕过 heap。AS descriptor 的 `pAddressRange.size = 0` 合法，也不是此次故障原因。[Vulkan heap interface](https://github.com/KhronosGroup/Vulkan-Docs/blob/v1.4.350/chapters/interfaces.adoc#L1101)、[AS range 契约](https://github.com/KhronosGroup/Vulkan-Docs/blob/v1.4.350/chapters/descriptorheaps.adoc#L264)

## 2. 独立 GPU 复现定位到 heap 数值读取路径

环境：Windows、NVIDIA GeForce RTX 5070 Ti（GB203-A）、驱动 616.92、Vulkan SDK headers 1.4.350；SPIR-V 使用 `VK_EXT_descriptor_heap` / `VK_KHR_shader_untyped_pointers`。证据仅覆盖本机 GPU/驱动。

独立程序直接调用 Vulkan，使用手写 SPIR-V，不经过 Metallic 的 heap 管理、shader cache、SPIR-V 修复或 Slang 编译。测试 heap 采用 host-visible/coherent 内存，同一次 dispatch 同时读取相同字节：

| 对照 | 结果 |
| --- | --- |
| CPU 写入 heap byte 128 的 `0x1122334455667788` | 写入值正确 |
| `ResourceHeapEXT` + `OpLoad uint64` 读取 byte 128 | 错误：纯 buffer probe 返回 0；带 ray query 的 probe 返回其他值 |
| `PhysicalStorageBuffer` 指针读取同一 heap byte 128 | 正确：`0x1122334455667788` |
| `OpBufferPointerEXT` 读取 heap 中的 storage-buffer descriptor，再读取其数据 | 正确 |
| 相同 heap 中 descriptor 的原始字节通过 physical pointer 回读 | 与 CPU 字节一致 |

这排除了 Metallic slot 分配、Slang 输出生成、host flush 或 heap 内容未上传作为该最小复现的原因；证据强烈指向当前驱动对 `ResourceHeapEXT` 普通数值 load 的实现。错误值并非恒为零，不能用零值特判修补。

四份 probe SPIR-V 均通过 `spirv-val --target-env vulkan1.3`；最终两次 GPU 日志没有 Vulkan validation 警告/错误。这里的实际 GPU 数据对照才是正确性证据，静态验证通过本身不足以判断运行结果。

之前还尝试过直接 opaque AS descriptor load 和动态 stride，未恢复正确结果。本次没有再次提交已知可能挂起的变体；上述数值读取定位也不等于已经解释 opaque AS load 的驱动内部故障。

## 3. 上游为何仍有 AS 特例

- [Slang #10671](https://github.com/shader-slang/slang/issues/10671)：报告 Slang 2026.5.1、RTX 4090、驱动 595.97 的 native AS descriptor 路径 DeviceLost，提出读取 uint64 address 再转换。
- [PR #11209](https://github.com/shader-slang/slang/pull/11209)：2026-06-03 合入上述 lowering。
- [#11231](https://github.com/shader-slang/slang/issues/11231)：后续报告非零 index 的 heap stride/load-width 不一致。
- [PR #11494](https://github.com/shader-slang/slang/pull/11494)：2026-06-16 合入 uint64 数组和默认 stride 8 修复。

本机 2026.18.2 源码已经包含这些改动。它们修复了上游已知的 lowering/stride 问题，却不能证明当前驱动的数值 load 正确；PR 中的 SPIR-V/FileCheck 验证也不能替代 GPU 执行。

此外，不应把 opaque AS descriptor 的首 8 字节在本机恰好等于 AS address，当成跨驱动保证。若应用主动采用 raw-address ABI，应由应用明确写入由 Vulkan 查询得到的 AS address。

## 4. 官方自定义 hook 还有独立编译阻断

Slang 文档提供 `export getDescriptorFromHandle<T>` 作为统一解引用的定制点。最小 shader 只含一个 `DescriptorHandle<RWByteAddressBuffer>`，hook 甚至只转发默认实现：

```slang
export T getDescriptorFromHandle<T : IOpaqueDescriptor>(DescriptorHandle<T> handle)
{
    return defaultGetDescriptorFromHandle(handle);
}
```

| Slang 2026.18.2 CLI 对照 | 结果 |
| --- | --- |
| 无 hook，native | 编译成功 |
| 上述恒等 hook，mapped | 编译成功 |
| 上述恒等 hook，native | 进程崩溃 `0xC0000005` |
| 含 AS 特化分支的 hook，native | 同样崩溃 |
| 独立 module 导出 hook，再 import，native | 同样崩溃 |

这一故障不需要 AS shader、Metallic 模块或 GPU 就能复现。已确认最小触发条件，尚未对 Slang 进程做内部堆栈定位。因此当前显式 `resolveDescriptor` 有实际必要，不能假定换成官方 hook 就能恢复所有隐式解引用。[官方定制说明](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/03-convenience-features.html)

## 5. 可落地的统一方案

### 已验证：AS 地址单元经 buffer descriptor 访问

```text
ResourceHandle<AS>.index
  → ResourceHeapEXT 中的 storage-buffer descriptor
  → OpBufferPointerEXT
  → 普通 buffer 数据中的 uint64 AS address
  → OpConvertUToAccelerationStructureKHR
  → ray query
```

这是 native heap 访问，没有 Binding/DescriptorSet mappings。PoC 将 AS 地址写入普通 buffer 数据，以非零 buffer descriptor index 3 访问，实际构建一个 triangle BLAS 和一个实例 TLAS，回读 committed intersection type 为 1、t 为 1.0。它绕过的是错误的 heap 数值 load；buffer descriptor 仍由 native heap 读取。

如下一步要求统一 index，建议采用此方案，并保持所有 AS 特化在资源后端/resolver 内：

1. 在 RHI/ResourceRegistry 中管理 AS address cells；对应 heap slot 写 `STORAGE_BUFFER` descriptor。不要把原来的 opaque AS descriptor 直接当作 buffer descriptor 使用。
2. 统一外部 handle 的 index 契约；AS 使用与 buffer 相同的索引单位。调用方通过统一 resolver 或项目的 `ResourceHandle<T>` 包装访问，暂不依赖会崩溃的 Slang export hook。
3. address-cell 内容、descriptor 和 AS 本体共同遵守 GPU 在途生命周期；AS rebuild/compaction 引起地址变化时创建或安全更新单元，避免覆盖仍被旧帧引用的数据。普通 refit 是否需要更新由地址是否变化决定。
4. GPU 回归覆盖非零/多 AS 索引、数组、跨帧替换、refit/compaction 和真实场景；partitioned AS 单独验证。
5. 测量额外间接读取对实际 ray-query/path-trace workload 的影响。PoC 没有提供性能结论。

API 与 handle 表示可以统一，底层类型特化仍然存在；image、buffer 和 AS 本来就使用不同的 GPU 取资源指令。

### 直接 opaque AS heap 路径

长期可跟进驱动与 Slang 的 native AS descriptor 支持，恢复直接 opaque descriptor load。必须重新通过真实 GPU 回归，不能只调整 stride 或删除当前编译期保护。当前先保留 address resolver，或在完成上述实现/回归后切换 buffer-address-cell 方案。

## 本地证据与复现入口

所有实验保存在 `.cache/as-handle-investigation/`（本地输出，不属于正式运行路径）：

- `README.md`：构建、汇编、运行命令和回读字含义。
- `HeapRead.cpp`：独立 Vulkan runner，支持纯 buffer 和真实 BLAS/TLAS 两种模式。
- `HeapReadPhysical.spvasm` / `heap-read-physical-final.log`：同一 heap 字节的两种读取对照。
- `AsBufferBridgeIndex3.spvasm` / `as-buffer-bridge-index3.log`：非零 AS index 的 ray-query 成功证据。
- `Hook*.slang` / `hook-results.json`：native hook 编译崩溃与成功对照。
- `spirv-validation.json`：四份 SPIR-V 的独立验证退出码。
- `slang-emit-spirv.cpp`、`interfaces-350.adoc`、`descriptorheaps-350.adoc`、`slang-*.json`：固定版本源码/规范与上游 issue/PR 快照。

本次只补充调查、独立 probe 和注释，未把 PoC 切入正式渲染路径。已有 native/mapped 回归结果见 [升级状态](DynamicResourceUpgradeStatus.md)；它们不替代新方案的生产集成验证。
