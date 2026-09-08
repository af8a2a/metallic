# Descriptor heap GPU fault isolation

## 已确认的触发边界

2026-09-08 至 2026-09-09，在 GB203-A / NVIDIA 616.64、Slang 2026.1.2 上，
独立计算用例使用原始生产 shader、descriptor heap 和完整 FrameContext：

```cpp
DeviceDesc{
    .enableValidation = true,
    .enableBindlessDescriptorHeap = true,
    .enableShaderObject = false, // 对照通过；仅改成 true 后 DeviceLost
};
```

具体负载是 `ImportancePdfCompute::buildEnvironment`，随后执行
`EnvironmentLightingPrecompute` 的 integrate 和 finalize。
SH 使用 procedural 256×128 输入、256 个 partial groups、32 字节用户 push，
两套 descriptor table 的索引依次为 0、1；CPU 回读并核验 partials 和最终系数。
两组均关闭 Aftermath，启用 Vulkan validation。

| 设备配置 | 结果 |
| --- | --- |
| 最小 descriptor heap 设备 | 通过，回读校验成功 |
| 仅额外启用 `enableShaderObject` | `frame.wait()` 返回 `DeviceLost`，约 4 秒后驱动恢复 |
| 完整 RenderGraphPreviewRenderer 设备特性 | `DeviceLost` |

这里隔离的是**设备启用 shaderObject 与 descriptor heap compute pipeline 的组合**。
最小用例没有调用 `vkCreateShadersEXT` 或 `vkCmdBindShadersEXT`，也没有 mesh、
VBuffer、OpenPBR 求值、渲染图或窗口。失败用例没有开启 Aftermath，尚不能从该用例的
`DeviceLost` 单独断定 PDF、SH integrate、SH finalize 中哪一条 GPU 指令失败。

源码入口：

- `Source/Editor/EditorApplication.cpp:2349` 和
  `Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:2704` 无条件设置
  `enableShaderObject = true`。LookDev 的 OpenPBR/VBuffer 路径使用 `VkPipeline`，
  仍被这一全局设备配置带入已复现的组合。
- `Source/Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp` 将 `enableShaderObject` 转为
  `VkPhysicalDeviceShaderObjectFeaturesEXT::shaderObject` 和 `VK_EXT_shader_object`。
- `ComputeProgram::initialize()` / `dispatch()` 始终创建并绑定普通 compute pipeline。
  Vulkan 路径使用 `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`，
  不因 `shaderObjectEnabled` 改走 shader object API。

这已经给出可复现的配置边界，但还没有证明驱动内部的错误位置，也不能把关闭该特性
称为已验证的 LookDev 完整修复。本次没有修改生产 shader、编译器或 RHI 行为。
Vulkan 允许启用 Shader Object 后继续使用 pipeline，这个组合本身不构成 API 违规；见
[Shader object / pipeline interaction](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-objects-pipeline-interaction)。

## 为什么不能归因于结构体复制

原始完整 VBuffer 用例通过 Aftermath 的资源追踪和自动检查点，关闭 shader debug info，
捕获到 `Error_DMA_PageFault`：Graphics Processing Cluster 读取 GPU 虚拟地址 0。
最早正在执行的命令是环境 SH dispatch。

最初检查了以下 Slang / SPIR-V 模式：

```text
uniform Struct push 的成员访问，或 StructuredBuffer<Struct>[i] 后访问字段
    → OpLoad(struct) → OpCopyLogical → OpCompositeExtract
```

只删除结果完全由字段提取消费的 `OpCopyLogical`，保留原始结构体 load、SPIR-V ID、
头部及其他指令，能依次让 SH、ClusterLightGrid、GPUDriven reset、instance cull、
HZB 完成。所有变体通过 `spirv-val --target-env vulkan1.3`。

但两个阴性对照也让原始 SH 完成并把故障推到 ClusterLightGrid：

1. **只把 SPIR-V 头部 word 2 从 `0x00280000` 改为 0，全部指令字节不变。**
2. 保留原始头部和全部原始指令，仅插入一条合法 `OpNop`。

因此删除 `OpCopyLogical` 的效果不具有特异性，不能认定该指令或源代码结构体复制是根因。
成功执行的 task/mesh shader 仍含该指令；后续失败的 SliderDebug shader 则没有该指令。
SPIR-V 规范明确说明 generator word 不影响语义且允许为 0，见
[Physical Layout](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_physical_layout_of_a_spir_v_module_and_instruction)。

上述 metadata / 指令扰动会改变二进制哈希，支持继续检查驱动代码生成或隐式缓存；
当前实验尚不能区分两者。SH 的 `ComputeProgram` 没有传入应用 pipeline cache，
Vulkan 创建调用使用 `VK_NULL_HANDLE`；应用没有按 shader hash 复用旧的 `VkPipeline`。

## 已完成的其他对照

- 5 个小型字段/结构体访问模式，每模式 1024 groups，最小设备下全部通过。
- 原始 SH shader 独立 integrate：1×1 纹理和 procedural 256×128 均通过。
- SH integrate + finalize 的三种 table 配置全部通过：两套表 0→1、两套表 0→0、
  一套表 0→0。后两种配置由 FrameContext 分配不同 heap；一套表使用 constant mapping。
- 上述三种配置各加上生产 PDF 前驱，也全部通过。
- 最后只启用设备的 Shader Object 特性，独立用例失败。

表布局、连续 dispatch 和 PDF 前驱本身不足以触发该最小用例。没有更改资源保活逻辑、
环境纹理尺寸、normal/TBN 或 OpenPBR BSDF 来获得这些结果。

## 复现工具与证据

独立测试位于 `tests/rhi/DescriptorHeapCodePatternTests.cpp`，shader 位于
`tests/rhi/shaders/DescriptorHeapCodePattern.slang`。默认运行 `push_fields` 和最小设备，
不自动开启已知会引发驱动恢复的配置。

**故障配置会造成短暂 GPU hang / 桌面卡顿。用户反馈该影响后，已停止 GPU 测试；
下列命令仅供之后明确安排复现时使用，不应放入日常自动回归。**

```powershell
$env:METALLIC_GPU_PATTERN = 'environment_sh_procedural'
$env:METALLIC_GPU_PATTERN_TABLES = 'shared'
$env:METALLIC_GPU_PATTERN_PREFIX_PDF = '1'
$env:METALLIC_TEST_AFTERMATH = '0'

# 通过的最小设备对照
$env:METALLIC_GPU_PATTERN_SHADER_OBJECT = '0'
$env:METALLIC_GPU_PATTERN_PREVIEW_DEVICE = '0'
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter descriptor_heap_code_pattern --rhi-validation

# 已知会触发 DeviceLost / 驱动恢复的单变量配置
$env:METALLIC_GPU_PATTERN_SHADER_OBJECT = '1'
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter descriptor_heap_code_pattern --rhi-validation
```

原始 shader 缓存已逐一校验并恢复。现有证据保存在本地，未加入 Git：

| 证据 | 路径（仓库根目录起） |
| --- | --- |
| 15 份独立测试日志、哈希和 9 份缓存恢复校验 | `.cache/aftermath/code-pattern-20260909/manifest.json` |
| 仅启用 Shader Object 的失败日志 | `.cache/aftermath/code-pattern-20260909/pattern-sh-device-shader_object.log` |
| 相同负载的最小设备通过日志 | `.cache/aftermath/code-pattern-20260909/pattern-sh-pdf-1-tables-shared.log` |
| SH 仅修改 generator 的 Aftermath 对照 | `.cache/aftermath/captures/20260908-235708-ea05082a/` |
| SH 仅添加 OpNop 的 Aftermath 对照 | `.cache/aftermath/captures/20260908-235715-dd8ae485/` |
| 累计消融至 HZB 后，SliderDebug 正在执行 | `.cache/aftermath/captures/20260908-235445-c963201f/` |

下一步若继续验证内部缓存，应先查询 `VK_KHR_pipeline_binary` 的
`pipelineBinaryInternalCacheControl` 支持，再考虑设备级 `disableInternalCache` 对照。
该对照尚未实施；见 [Khronos proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html)。
