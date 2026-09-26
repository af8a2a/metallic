# Native 混合位宽原子指针编译崩溃调查

日期：2026-09-25。环境：RTX 5070 Ti、NVIDIA 616.92、Slang 2026.18.2、Vulkan validation 1.4.350。此次只重跑和诊断，没有修改生产渲染代码。

## 结论

三项失败都来自 `VisibilityBufferPass::createPipelines` 创建 `streamClusterRasterMain` 的同一调用：`VulkanRhi.cpp` 内 `vkCreateComputePipelines` 进入 `nvoglv64.dll + 0x1202596`，抛出首次 CPU 访问异常 `0xc0000005`。

已用 8 行 Slang 将相同驱动异常缩小到：**同一个 native shader 中，32 位和 64 位整数原子操作同时通过 KHR untyped StorageBuffer pointers 访问**。原始 Slang typed-pointer 模块可创建管线；应用 `normalizeNativeDescriptorHeapSpirv` 后崩溃。最小复现访问地址为 `0x14`，与完整用例的模块偏移一致。

这是先前修复 native 嵌套结构布局所用 untyped-pointer 转换触发的驱动编译兼容问题。CPU 最终索引迁移前的源码也有同样结果。将失败笼统归为“复杂 shader 驱动问题”不够准确；现在已有小型混合位宽原子操作复现。

## 独立进程重跑

| 测试 | native | mapped |
| --- | --- | --- |
| `tessellation_recursive_render` | 失败，同一驱动首次异常 | 通过 |
| `tessellation_displacement_render` | 失败，同一驱动首次异常 | 通过 |
| `render_graph_gpu_driven_mixed_producer_render` | 失败，同一驱动首次异常 | 通过 |

每项单独启动进程。两个细分用例在 GoogleTest 捕获首次异常后，进程清理还会以 `0xc0000409` 结束；这不是首次故障。调试器均捕获到前面的 `0xc0000005`。先前位移细分的 streamasset 文件打开失败属于前序异常遗留，本次独立运行已确认其实际失败点。

## 对照证据

独立 `PipelineCompileProbe` 只创建设备、shader module 和 compute pipeline：不创建 descriptor heap、buffer、command buffer，也不提交 GPU 工作；传入 `VK_NULL_HANDLE` pipeline cache。

| 输入 | 验证层开启 | 验证层关闭 | spirv-val |
| --- | --- | --- | --- |
| CPU 索引迁移前，Slang 原始 typed pointers | 管线创建通过 | 通过 | 通过 |
| CPU 索引迁移前，规范化为 untyped pointers | CPU 访问异常 | 同样异常 | 通过 |
| 当前源码，Slang 原始 typed pointers | 管线创建通过 | 通过 | 通过 |
| 当前源码，规范化为 untyped pointers | CPU 访问异常 | 同样异常 | 通过 |

上述 shader 均从源码重新编译，未使用应用 shader cache。因此，本次故障发生于驱动管线编译阶段；不能归因于 resource index 上传、push 参数偏移、descriptor 生命周期、GPU 内存访问、测试顺序或应用缓存。验证层也不是触发条件。驱动内部具体缺陷仍需 NVIDIA 定位；我们确认的是可复现的输入条件与异常位置。

## 最小复现

```slang
struct Push { uint words; uint pixels; uint offset; };
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Push push) {
    RWStructuredBuffer<uint> words = DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.words, 0));
    RWStructuredBuffer<uint64_t> pixels = DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(push.pixels, 0));
    uint value; InterlockedAdd(words[push.offset], 1u, value); InterlockedMax(pixels[value], uint64_t(value) << 32);
}
```

以 `-target spirv -profile spirv_1_6 -entry main -capability spvDescriptorHeapEXT` 编译，然后应用当前 normalizer。

| 访问组合（规范化之后） | 管线创建 |
| --- | --- |
| 32 位 load + 64 位 atomic max | 通过 |
| 32 位 atomic add + 64 位 atomic max | 同一驱动异常 |
| 32 位 atomic max + 64 位 atomic max | 同一驱动异常 |
| 32 位 atomic add + 64 位 atomic add | 同一驱动异常 |
| 两次 32 位原子操作 | 通过 |
| 两次 64 位原子操作 | 通过 |
| 单次 64 位原子操作 | 通过 |
| 32 位 load + 64 位 store | 通过 |

此表全部模块通过 `spirv-val --target-env vulkan1.3`。它们仅验证管线编译，未声称实际 GPU 结果正确。

## 修复方向与验证边界

在完整 `streamClusterRasterMain` 中，仅保留 `RWStructuredBuffer<uint64_t>` 的 typed 指针、继续转换其余 buffer，管线创建与 spirv-val 都通过。仅跳过 GetDimensions 或普通 uint buffer 的转换仍失败。这给出了保留标量原子访问类型的修复方向，但还需要正式实现、GPU 回读及三个渲染用例的验证。

不能直接全局关闭 normalization：原始 typed 路径存在先前已验证的嵌套结构布局错误，会重新引入错误材质索引及 DeviceLost。

另做过在原子指令前插入 `OpBitcast` 到 typed pointer 的诊断实验：虽然管线创建成功，但 spirv-val 和验证层拒绝其 logical pointer operand，因此该变体不是合法修复，已排除。

## 本地复现与证据

所有证据在 `.cache/native-failure-rerun/`，属于本地输出：

- `summary.json`：用例结果、对照矩阵、输入/测试二进制 SHA256。
- `test-results.json` 及六份测试日志：三项用例各自的 native/mapped 运行。
- `stack-*.log`：三个实际用例的首次异常和完整原生栈。
- `PipelineCompileProbe.cpp`、`probe-results.json`：无资源/提交/缓存的八组编译对照。
- `mix-atomic32-atomic64.slang`、`minimal.spvasm`、`minimal-stack.log`：小型 shader、转换后字节码和相同驱动异常。
- `minimal-results.json`、`operator-results.json`：位宽/操作种类的隔离结果。
- `select-results.json`：完整 shader 的指针类型选择性转换实验。
- `typed-atomic-results.json` 和对应 validation log：已拒绝的非法 bitcast 实验。

```powershell
$env:METALLIC_SLANG_DESCRIPTOR_MODE = 'native'
& .\build\tests\MetallicRhiTests.exe --rhi-validation '--gtest_filter=*tessellation_displacement_render'
# 独立探针已构建时：normalize/on 可稳定复现，raw/on 可创建管线。
& .\.cache\native-failure-rerun\PipelineCompileProbe.exe .cache/native-failure-rerun/mix-atomic32-atomic64.spv normalize on
```


后续方案比较及按真实类型识别的六入口编译 PoC 见 [修复与重构方案](NativeAtomicPointerFixPlan.md)。
