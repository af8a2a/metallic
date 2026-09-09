# Descriptor heap GPU fault isolation

## 运行时要求：统一启用 Shader Object

`DeviceDesc::enableShaderObject` 现在默认且必须为 `true`。显式传入 `false`
会在创建 Vulkan instance 之前返回 `InvalidArgument`；设备选择同时要求
`VK_EXT_shader_object` 扩展和 `shaderObject` feature，不再通过候选回退禁用该能力。
不支持的设备返回 `Unsupported`，RHI 测试沿用既有 skip 行为。
该要求也覆盖只创建 `VkPipeline` 的设备、测试主设备和各独立计算测试。

应用 `.pso` 的 Vulkan backend tag 从 `MTVK` 升级为 `MTV2`。旧缓存没有记录
Shader Object 特性状态，因此统一按 `Incompatible` 从空缓存重建；无需手动删文件。
这不会清除或绕过驱动内部缓存。历史 `false` 配置写入驱动缓存的机器仍可能需要
显式使用 `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled` 来隔离已有污染，不能把
强制开启功能等同于所有历史 GPU hang 已修复。

`descriptor_heap_code_pattern` 测试也始终启用该能力。旧环境变量
`METALLIC_GPU_PATTERN_SHADER_OBJECT=0` 会报错；`=1` 仅保留命令兼容性。
跨 `false/true` 的负向缓存实验保留在独立工程 `E:\vulkan-minimal-devicelost`，
不再经过 Metallic RHI。该独立工程已在 2026-09-09 复现故障及通过对照。

### 本次策略验证

Debug 构建 `MetallicRhiTests`、`LookDev`、`Metallic` 成功。
`shader_object_required`、`validate_device`、`optional_feature_soft_request` 和
`pipeline_cache_persistence_and_shader_invalidation` 四项通过；测试覆盖默认设备
能力、显式关闭拒绝、其他特性软请求和旧 backend tag 的缓存拒绝。

使用原始 SPIR-V，并显式设置 `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`，
`descriptor_heap_code_pattern`（procedural SH、PDF 前驱、integrate + finalize）和
`visibility_buffer_deferred_openpbr` 两项 GPU 回归通过，没有 `DeviceLost`。
首次沙箱运行的 VBuffer 临时缓存写回失败；退出沙箱复核后，旧缓存被判为
`Incompatible`，新 `MTV2` 缓存成功保存并在后续渲染重新载入，完整用例约 10 秒通过。
旧缓存副本和日志保留在
`.cache/validation/shader-object-required-20260909-214524/`，包含
`contract.stdout.log`、`gpu.json`、`cache-migration.json` 和三张 LookDev 对比 PNG。
这些 GPU 结果只覆盖关闭驱动内部缓存的配置；该诊断设置未设为运行时默认值。

## 当前结论：跨 shaderObject 配置复用不兼容缓存

2026-09-09 的进一步对照表明，**设备启用 shaderObject 不是充分的故障条件；
pipeline 缓存复用参与了当前设备上的故障**。同一测试二进制、原始生产 SPIR-V、
相同设备特性和负载下，仅切换驱动内部 pipeline cache 控制，结果由通过变为
`DeviceLost`。关闭该缓存并为 VBuffer 使用新建的应用 pipeline cache 后，完整
`visibility_buffer_deferred_openpbr` GPU 测试通过。

随后的首次编译顺序对照进一步隔离出：同一 shader 在 `shaderObject=false` 设备上
编译后，由 `shaderObject=true` 设备命中同一 pipeline key，会触发 `DeviceLost`；
原设备复用通过，新设备禁用缓存重新编译也通过。另一组先在 `true` 设备编译的对照通过。
证据支持当前 RTX 5070 Ti / NVIDIA 616.64 的跨特性缓存兼容性问题，
但没有直接观察驱动内部机器码，不能指定其内部缓存哪一层或哪段实现存在缺陷，
也没有证明所有驱动或所有场景都已修复。诊断开关默认关闭；下面记录策略实施前的
实验，当时旧 `.cache/pso/VisibilityBufferPass.pso` 已按哈希校验恢复。
当前的功能要求和应用缓存失效策略见上节。

## 首次编译顺序与 pipeline key

两组均使用原始 SH 的全部指令，仅在内存中将 SPIR-V generator word 改为此前未使用的
`0x0028a909`（A）和 `0x0028b909`（B）。它们保留同一组指令和 2426 words，
`vkGetPipelineKeyKHR` 返回不同的 pipeline key，因此可分别建立缓存。
准备步骤只创建 pipeline，不分配测试资源或提交命令；执行步骤仅做 SH integrate，
无 PDF 前驱、无 finalize、无 Aftermath，启用 validation 并核验 2304 个 partial float4。

| 组别与顺序 | 内部缓存 | 结果 |
| --- | --- | --- |
| A：SO=false 首次编译 → SO=true 执行 | enabled | `DeviceLost` |
| A：同一缓存回到 SO=false 执行 | enabled | 通过 |
| A：同一 shader 在 SO=true 重新编译并执行 | disabled | 通过 |
| B：SO=true 首次编译 → SO=false 仅编译 → SO=true 执行 | enabled | 通过 |

所有行的 global key 都是 `725085dc37461347`。
A 的 pipeline key 为 `6eb5461bf9ea5a0d`，B 为 `a8d246d1f48a556c`；
每组在 SO=false / true 下的 key 均相同。**相同 key 单独不足以证明错误**，
但与首次编译顺序、原设备通过、禁用缓存恢复这三组执行对照结合，
支持不兼容编译结果被跨配置复用的判断。
由于 A/B 使用不同 nonce，没有对同一冷 key 反转首次编译顺序，仍存在少量
nonce 对编译或缓存分桶影响的混杂；此外该开关同时改变扩展和 feature 启用状态，
本轮未单独区分二者。需要厂商结合缓存 binary 才能确认内部实现层的根因。

应用中的触发模式可表述为：

```text
Device A: descriptorHeap=true, shaderObject=false
    createComputePipeline(shader S)  // 产生缓存，未提交也足够
Device B: descriptorHeap=true, shaderObject=true
    createComputePipeline(shader S)  // 同 global key + pipeline key
    dispatch()                      // 命中上述缓存时 DeviceLost
```

该实验从 205126 至 205244 的八个独立进程使用同一个测试二进制；
证据在 `.cache/aftermath/code-pattern-20260909/` 下的 `*-key-*` 目录，
每个目录包含 `run.json`、stdout 和 stderr。
首次编译 nonce 是诊断隔离变量，不是生产修复：再次复用相同 nonce 时，
先前缓存已存在，不能继续称为一次新的首次编译实验。

## 原始 shader 的缓存 A/B 对照

两次独立进程均运行 procedural 256×128 SH、PDF 前驱、integrate + finalize、
shared descriptor tables，启用 validation 和 shaderObject，关闭 Aftermath。
二进制 SHA-256 均为
`f67ca8bb630b6f29bb8b1039a3f7e301c1166cec17563fdedf165b4d267bd980`；
SH 原始 generator 为 `0x00280000`，2426 words，原始字节 FNV-1a64 为
`0xe759c9934a137d48`，没有 generator 覆盖或 SPIR-V 指令改写。

| 驱动内部缓存模式 | 结果 |
| --- | --- |
| `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled` | 通过，SH partials 和最终系数回读校验成功，测试用时 410 ms |
| `METALLIC_VK_INTERNAL_PIPELINE_CACHE=enabled` | `frame.wait()` 返回 `DeviceLost`，测试用时 3616 ms |

两组都显式启用同一 `VK_KHR_pipeline_binary` 扩展和 feature chain，只有
`VkDevicePipelineBinaryInternalCacheControlKHR::disableInternalCache` 的值不同。
该设备报告 `pipelineBinaryInternalCache=1`、`pipelineBinaryInternalCacheControl=1`。
因此结果不能只归因于启用扩展或改变设备 feature chain。

## 完整 VBuffer 路径的两层缓存对照

仅关闭驱动内部缓存、保留旧应用 `.pso` 时，完整 GPU 测试仍然失败。
对应 Aftermath 捕获关闭 shader debug info：环境 SH 和 ClusterLightGrid 已完成，
最早正在执行的命令推进到 `GPUSceneSubsystem::recordInstanceCull()` 内的 reset dispatch；
错误仍为读取 GPU 虚拟地址 0 的 `Error_DMA_PageFault`。

继续保持内部缓存关闭，仅暂时隔离旧 `.cache/pso/VisibilityBufferPass.pso`，使应用
从空缓存创建 pipeline，随后完整测试在约 12 秒内通过，输出：

- `LookDevVBufferReference.png`
- `LookDevVBufferDeferred.png`
- `LookDevVBufferComparison.png`

这次通过启用 validation、关闭 Aftermath，使用原始 shader，并保留完整设备特性。
旧 `.pso` 的 SHA-256 是
`bce179da455824c16e4f9b47a9a4bac6fc1c2ccd9f564b95af17aa52dd48599c`，
已恢复到原路径。新建 `.pso` 的 SHA-256 是
`7ef473718d840731c65a5a2ac3035e3d3debebca2aa02d5b5ca28329e0122238`，
另存于通过实验的证据目录中，文件名为 `fresh-VisibilityBufferPass.pso`。
这里验证了一个完整通过的诊断配置，尚未将该配置设为产品默认值，也不能由此判断
旧缓存为何产生不正确代码。

## 历史定位：设备特性边界

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

这一步给出当时缓存状态下可复现的配置边界，但没有证明驱动内部的错误位置，
不能把关闭该特性称为已验证的 LookDev 完整修复。该阶段没有修改生产 shader、
编译器或 RHI 行为；后续仅增加显式启用的 RHI 缓存诊断控制，见上文。
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

上述 metadata / 指令扰动会改变二进制哈希，当时尚不能区分驱动代码生成与隐式缓存。
后续内部缓存 A/B 已直接验证缓存控制对结果的影响，但未解释内部原因。
SH 的 `ComputeProgram` 没有传入应用 pipeline cache，
Vulkan 创建调用使用 `VK_NULL_HANDLE`；应用没有按 shader hash 复用旧的 `VkPipeline`。

## 前期完成的其他对照

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
`tests/rhi/shaders/DescriptorHeapCodePattern.slang`。默认模式为 `push_fields`，设备
始终开启 Shader Object。GPU 执行要求关闭驱动内部缓存，或显式启用
`METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST=1`；否则跳过。compile-only 不受此限制。

**故障配置会造成短暂 GPU hang / 桌面卡顿。用户反馈该影响后曾停止 GPU 测试，
本轮在用户要求继续验证后执行了上述对照。下列故障配置不应放入日常自动回归。**

新增诊断选项均为显式启用，未设置时保留原有行为：

| 选项 | 行为 |
| --- | --- |
| `METALLIC_VK_INTERNAL_PIPELINE_CACHE=enabled` / `disabled` | 使用对称的扩展和 feature chain，分别允许/禁用驱动内部 pipeline cache；不影响应用 `.pso` 文件的读写 |
| `METALLIC_VK_LOG_PIPELINE_KEYS=1` | 请求记录 pipeline key；必须同时选择上述一个内部缓存诊断模式 |
| `METALLIC_GPU_PATTERN_INTEGRATE_ONLY=1` | 环境用例只执行 integrate，验证 partials，跳过 finalize 和系数读取 |
| `METALLIC_GPU_PATTERN_SPIRV_GENERATOR=<uint32>` | 接受十进制或 `0x` 十六进制，只覆盖当前编译结果内存中的 word 2，不修改磁盘 shader cache；日志记录原始/实际 generator、原始 SPIR-V 字节哈希和字数 |
| `METALLIC_GPU_PATTERN_COMPILE_ONLY=1` | 完成 shader 编译和 `ComputeProgram` pipeline 初始化后立即返回；不创建测试纹理/缓冲、不执行 PDF、不记录命令或提交 GPU 工作 |
| `METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST=1` | 显式允许诊断用例执行潜在不兼容的驱动缓存；关闭内部缓存或 compile-only 时不需要 |

内部缓存诊断会检查扩展、`pipelineBinaries` feature 和
`pipelineBinaryInternalCacheControl` property。不支持时 RHI 返回 `Unsupported`，
测试按 unsupported/skip 报告，不能将这种结果计为通过。非法选项返回 `InvalidArgument`。
`compile-only` 仍会创建设备和 pipeline，用于隔离编译/缓存写入与 GPU 执行。

下面是当前固定 Shader Object 开启的缓存 A/B 命令。其余配置保持一致：

```powershell
$env:METALLIC_GPU_PATTERN = 'environment_sh_procedural'
$env:METALLIC_GPU_PATTERN_TABLES = 'shared'
$env:METALLIC_GPU_PATTERN_PREFIX_PDF = '1'
$env:METALLIC_TEST_AFTERMATH = '0'

# 关闭驱动内部缓存的对照
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
$env:METALLIC_GPU_PATTERN_PREVIEW_DEVICE = '0'
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter descriptor_heap_code_pattern --rhi-validation

# 显式允许执行旧驱动缓存，可能触发 DeviceLost / 驱动恢复
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'enabled'
$env:METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST = '1'
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
| 原始 SH / PDF，内部缓存 disabled，通过 | `.cache/aftermath/code-pattern-20260909/204251-original-cache-disabled-e315ee/` |
| 相同二进制与原始 SH / PDF，内部缓存 enabled，失败 | `.cache/aftermath/code-pattern-20260909/204307-original-cache-enabled-5413ba/` |
| 内部缓存 disabled、旧应用 `.pso`，SH/Cluster 完成后 reset 失败 | `.cache/aftermath/code-pattern-20260909/captures-cache-disabled/20260909-204440-54db8c41/` |
| 内部缓存 disabled、新应用 `.pso`，完整测试通过，含 3 张 PNG 和新 `.pso` | `.cache/aftermath/code-pattern-20260909/204648-full-fresh-app-cache-87141c/` |

各新增实验目录的 `run.json` 保存命令、环境选项、二进制哈希和退出码；完整通过实验
另记录旧/新应用缓存哈希以及 `originalCacheRestored=true`。缓存控制 API 参考
[Khronos proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html)。

剩余待定位的是：缓存内容在何种设备配置/创建顺序下产生、之后是否被不兼容地复用，
以及内部缓存与应用缓存是否共享同一缺陷。现有通过/失败对照不足以证明具体的内部键
缺失，也不能把 SPIR-V generator 扰动当作修复方案。
