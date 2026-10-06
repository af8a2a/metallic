# Shader 模块

See [Color Pipeline](../Documentation/ColorPipeline.md) for ACEScg working-space and source/display contracts.

Renderer 默认使用 scene-linear ACEScg/AP1/D60 HDR；Rec.709/D65 保留为兼容模式。显示变换、SDR/scRGB/HDR10 输出与
LookDev 默认值见 [Display Output](../Documentation/DisplayOutput.md)。

Metallic 的可复用 shader 库使用 Slang module。子系统之间用 `import`，同一模块的实现
文件用 `__include`，第三方 HLSL 的宏配置与文本包含留在 Interop 或程序内部。

| 目录 | 职责 |
| --- | --- |
| `Modules/ShaderCore.slang`、`Modules/Core/` | ResourceHandle、BufferSpan、旧资源兼容接口、相机、顶点解码、SH、显示颜色；不声明 push constant |
| `Modules/Core.slang` | ComputeProgram 具名资源参数、常量和底层测试兼容入口，重新导出 ShaderCore |
| `Modules/Material.slang`、`Modules/Material/` | CPU/GPU 共用的材质与纹理数据布局 |
| `Modules/MaterialProgram.slang`、`Modules/MaterialProgram/` | 无模型/资源依赖的 SurfaceMaterialContext、MaterialInstanceRef 与 BSDFEval/BSDFSample；见 [Phase 2](../Documentation/MaterialSystemPhase2.md) |
| `Modules/DebugLambert.slang`、`Modules/DebugMirror.slang`、`Modules/SurfaceLighting.slang` | 三阶段 Surface Programs、泛型 `shadeSurface` 与生产共用的直接光循环；见 [Phase 3](../Documentation/MaterialSystemPhase3.md)、[Phase 5](../Documentation/MaterialSystemPhase5.md) |
| `Modules/SlabClosure.slang` | Single / Dual Slab canonical families、Mix / Layer 原型；见 [Phase 10](../Documentation/MaterialSystemPhase10.md) |
| `Modules/OpenPBR.slang`、`Modules/OpenPBR/` | Adobe OpenPBR 1.1 原生 Slang 移植、静态泛型 LUT/Feature provider；见 [模块说明](Modules/OpenPBR/README.md) |
| `Modules/FiberMaterial.slang`、`Modules/FiberLighting.slang`、`Modules/RTXCRFiber.slang` | 独立 Fiber 三阶段契约、projected lighting、RTXCR Chiang Program；见 [M6](../Documentation/MaterialSystemM6Fiber.md) |
| `Features/Strands/NativeStrands.slang` | 原生曲线有限多层 visibility、Fiber 着色、motion/identity 与显式 overflow；见 [M7](../Documentation/MaterialSystemM7Strands.md) |
| `Modules/RayMaterialExecution.slang`、`Features/PathTracing/RayMaterialQueue.slang` | 按命中点独立的 Surface/Fiber 求值与 prepare、有界 Program 分类；见 [M8](../Documentation/MaterialSystemM8RayExecution.md) |
| `Modules/RTXCRHair.slang`、`Modules/RTXCRHair/` | RTXCR Chiang / Separate Chiang / Far Field 原生 Slang vendored 实现；见 [来源与接口](Modules/RTXCRHair/README.md) |
| `MaterialClosureClassification`（CPU）、`tests/rhi/shaders/ClosureSchedulingProbe.slang` | Program → Closure Family 逻辑调度及 fused/split GPU A/B；生产保持 fused，见 [Phase 11](../Documentation/MaterialSystemPhase11.md) |
| `Features/PathTracing/OpenPBRSurface.slang`、`Modules/OpenPBRClosure.slang` | PT / Deferred 共用的 OpenPBR Material Program、Closure、PreparedClosure；见 [Phase 4](../Documentation/MaterialSystemPhase4.md) |
| `Modules/GPUDriven.slang`、`Modules/GPUDriven/` | GPU 场景、meshlet LOD、剔除、混合光栅化、可见性编码和材质分箱 |
| `Features/VisibilityBuffer/VisibilityMaterialBinning.slang` | 8×4 Wave32 稀疏 Program 分箱、prefix allocation 和 indirect dispatch；见 [Phase 6](../Documentation/MaterialSystemPhase6.md) |
| `Modules/Lighting.slang`、`Modules/Lighting/` | 物理光照、光源选择、光照网格、环境过滤和阴影参数 |
| `Modules/ShaderToHuman.slang`、`Modules/ShaderToHuman/` | 原生 Slang 调试文本、2D/3D 绘制和泛型 Scatter；`Metallic.ShaderDebug` 命名空间 |
| `Modules/ColorGrading.slang`、`Modules/ColorGrading/` | ACES 2.0 / UE Film、全局调色、custom LUT 与三维 LUT 编解码 |
| `Interop/NeuralTextures.slang` | NTC 的唯一模块适配入口，封装 Generic/CoopVec 和无 NTC 的回退 |
| `Interop/NRDEncoding.slang` | NRD 前端编码的唯一模块适配入口 |
| `Interop/Denoising/NRD/` | 已适配的 NRD pass、bindings、配置和算法快照；保留程序内 HLSL 宏 |
| `ThirdParty/RadianceCache/` | SHARC/NRC 头文件与许可证，保留原有 HLSL 包含方式 |
| `Features/` | Shader programs：入口、pass 资源和流程相关代码；路径保持兼容现有 C++ 和管线资产 |
| `Licenses/` | 第三方 shader 许可证 |

测试探针放在 `tests/rhi/shaders/`。RTXCR Geometry / Subsurface、RTXTF、NTC 等 SDK 继续使用
`External/` 下的源码。RTXCR Hair GPU 实现使用原生模块，上游快照保留为独立验收基线。
OpenPBR GPU 实现使用原生模块，`External/openpbr-bsdf` 保留为独立验收基线和 CPU LUT 数据来源。
ShaderToHuman 使用 `import ShaderToHuman;`，仅保留原生 Slang 移植；
来源记录、许可证和使用说明见 [ShaderToHuman](Modules/ShaderToHuman/README.md)。
所有 vendor 文件与许可证保持原有内容。

## 使用库

```slang
import Core;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    let resources = getResourceParameters<GPUProbeResourceParameters>();
    RWStructuredBuffer<uint> output = resolveBuffer<RWStructuredBuffer<uint>>(resources.output);
    output[id.x] = packVisibilityId(id.x, 0);
}
```

模块入口列出同一子系统的实现，例如 `GPUDriven.slang`：

```slang
#language slang 2026
module GPUDriven;
__include "GPUDriven/GPUDrivenSceneCommon.slang";
__include "GPUDriven/GPUDrivenCullingCommon.slang";
// 其余实现文件也由此入口列出。
```

实现文件声明 `implementing GPUDriven;`，公开 API 标为 `public`，辅助实现标为 `internal`。
命名空间使用 `Metallic`、`Metallic.GPUDriven`、`Metallic.Material`、`Metallic.Lighting`。
消费者直接导入自己使用的模块，不依赖其他模块的间接导入。
当前固定的 Slang 2026.18.2 下，模块公开结构体的成员仍显式标注 `public`。

新 compute pass 使用 `ComputeKernel` + `ParameterWriter`，以具名 typed 参数承载资源。
Shader 获取使用 [ShaderRegistry](../Documentation/ShaderRegistry.md) 的 `getComputeKernel` /
`getComputeProgram`，或 `getShader` 后由统一管线入口创建；默认自动复用持久 PSO，Pass 不管理缓存文件。
DR-first 的新接口使用 `ResourceHandle<T>` / `SamplerHandle`（32 位）和
`BufferSpan<T>` / `RWBufferSpan<T>`（descriptor、字节偏移、元素数）。通过
`resolveUniform` / `resolveNonUniform` 解析资源，通过 span 的 `load` / `store` 或
`loadNonUniform` / `storeNonUniform` 访问普通 buffer；真正物理地址用 `PhysicalPtr<T>` 明确表达。
CPU 使用 `sampledImageHandle`、`storageImageHandle`、`samplerHandle` 和 `bufferSpan<T>`。
全部 renderer image/buffer 入口与外部适配边界见 [ResourceAccessABI](../Documentation/ResourceAccessABI.md)。

`ParameterTransport::InlinePush` 将参数块直接推送到 byte 0，不上传 root、slot table 或标量图像句柄数组；
shader 导入 `ShaderCore` 并声明 `[[vk::push_constant]] ConstantBuffer<Params>`。
`ParameterTransport::DescriptorBuffer` 用于较大的参数块；shader 额外导入 `ParameterRoot`，
通过 `getParameters<Params>()` 读取 DR buffer；push 根是 12 字节的 index / byteOffset / wordCount。后端检查实际 push 数据容量，不会静默截断。
两种传输共用 registry、资源保留、帧代次检查和 prepared dispatch；ABI 包含传输方式，不能混用。
`ShaderSampledImage` / `ShaderStorageImage` / `ShaderBuffer` / `ShaderSampler` 同样是 32 位 engine handle 别名。
Slang 的 `DescriptorHandle` 仅保留在底层 resolver；AS 仍保留独立的完整地址语义。
旧 `DataSpan` / `ShaderDataSpan` 与 `writer.dataBuffer()` 已移除，使用 `bufferSpan()` / `dataSpan()`。

[PostProcessParameters.h](../Source/Runtime/Render/Core/PostProcessParameters.h) 共用 C++/Slang 字段声明与显式 padding：
FinalBlit、SliderDebug（包括 DLSS-NR overlay）和 AutoExposure 使用 inline push，ColorGradingLUT 使用 DR 参数块。
AutoExposure 的 Histogram、Reduce、Apply 复用同一份不可变参数；barrier 来自阶段读写声明，不能从 handle 推测访问。
新增参数 ABI 时应验证字段偏移、GPU 读回、mapped/native 路径和生命周期；共享声明不等于自动完成布局验证。

[LightingKernelParameters.h](../Source/Runtime/Render/Core/LightingKernelParameters.h) 提供 ClusterLightGrid、
LightGridDebug、PrepareLightsPdf、BuildReGIR 和 EnvironmentLightingPrecompute 的共享 inline 参数。
普通光照数据通过带范围的 `RWBufferSpan<T>` 访问；PDF 每次归约直接传入源、目标 mip 的 storage handle，
不再上传完整 mip 句柄数组或创建环境 PDF 的占位光源 buffer。无 frame 的录制同样通过参数包保留资源与 kernel。
Lighting 库只导入 `ShaderCore`，不隐式引入任何参数根布局。

[RTXDIPostProcessParameters.h](../Source/Runtime/Render/Core/RTXDIPostProcessParameters.h) 共用
RTXDI Confidence（104 字节）和 Composite（36 字节）的 inline 参数声明。
Confidence 的各滤波阶段分别编码不可变参数快照，复用已注册的具名图像句柄；
历史纹理和 ping-pong 梯度的访问与同步仍由 RenderGraph 阶段声明负责。

[PathTraceStageParameters.h](../Source/Runtime/Render/Core/PathTraceStageParameters.h) 提供 SHaRC clear/resolve（80 字节）
和 NRC 输出累积/tonemap（36 字节）的共享 inline 参数。SHaRC SDK 需要 StructuredBuffer 对象进行原子操作，
因此这三个缓存 buffer 使用具名 descriptor handle；维护阶段直接读取 settings，不再依赖 cacheParams 的公共前缀。
主追踪及其 OpenPBR、NTC、VisibilityBuffer 使用共享的 `SceneResourceParameters` 具名 DR 字段。

VisibilityBuffer Deferred 是纯光栅表面的实时 resolve，不包含 ray-query 积分器。
材质的语义 Feature、作者策略和编译选择由 CPU [Feature System](../Documentation/MaterialSystemPhase7.md)
统一分析；AlphaMode / doubleSided 使用独立签名，不能直接当作 lighting shader keyword。
[Coverage Program](../Documentation/MaterialSystemPhase8.md) 从 Value 源码提取独立的覆盖表达式，
在 VBuffer 写入深度/可见性之前及 RT/shadow 候选命中处求值，和 Surface 共用不可变材质参数快照。
[Value IR](../Documentation/MaterialSystemPhase9.md) 统一 Surface/Coverage 的验证、优化、切片与稳定身份；
TextureSample 显式选择 LOD、梯度或 RayCone，不能使用依赖 quad 的隐式导数。
`Features/PathTracing/SceneSurface.slang` 和 `OpenPBRSurface.slang` 提供共享表面求值，
`Features/VisibilityBuffer/VisibilityBufferLighting.slang` 负责 ClusterLightGrid、显式阴影输入和 IBL。
有界 IBL 权重默认使用 native FP16，几何与 HDR 累加保持 FP32；`halfPrecision: false`
提供相同算法的 FP32 对照。详见 [VisibilityBufferDeferred](../Documentation/VisibilityBufferDeferred.md)。

`Core` 的生产用法是 `getResourceParameters<Params>()` 加显式 resolver：buffer 使用
`resolveBuffer<StructuredBuffer<T>>(resources.indices)`，图像和 sampler 使用 `resolveUniform`，
纹理数组使用 `resolveNonUniform(ResourceHandle<Texture2D<float4>>(resources.materialTextures.load(index)))`。
[NamedResourceParameters.h](../Source/Runtime/Render/Core/NamedResourceParameters.h) 共用 C++/Slang 字段；
CPU 初始化 `ComputeProgramDesc.resourceParameters`，使用
[NamedResourceLayouts.h](../Source/Runtime/Render/Core/NamedResourceLayouts.h) 的对应布局（如 `kGPUProbeResourceLayout`）。
编码器检查字段范围、对齐、重叠、资源类型和数组形式，并将 CPU 输入 ID 写入具名字段。
生产 GPU 参数块没有逻辑 slot table；场景结构为 440 字节，buffer/image/sampler 字段为 4 字节，
数组和原始数据字段为 12 字节 span，AS 独立保留完整地址。常量仍通过 `getConstants<T>()` 读取。
`getResource<T>(slot)` / `getResourceArray<T>(slot, index)` / `getData<T>(slot)` 仅用于未迁移的底层测试适配器，
不要在 `Features`、`Interop` 或生成的材质代码中新增调用。
`Core` 导入 `ParameterRoot`，因此不能和另一份 inline push 声明混用。RHI 不在用户 push 数据前插入 heap header。

Lighting 的算法显式接收 `StructuredBuffer<GPUPunctualLight>` 或 `PunctualSamplingResources`；
库内不再固定光源、ReGIR、PDF 的槽位。顶点位置读取同样显式接收 buffer；
是否使用硬件 position fetch 由调用程序决定。着色法线、几何法线和 TBN 的求值顺序保持原有语义。

## SDK 兼容边界

- 固定功能经 adapter module 导入。`NRDEncoding` 固定前端编码配置，`NeuralTextures`
  由编译 session 的 `METALLIC_HAS_NTC` / `METALLIC_NTC_COOPERATIVE_VECTOR` 选择实现。
- `#define` 不跨 `import` 或 `__include` 传播。消费者局部定义不能改变已导入模块；
  SDK 全局排列使用 `SlangShaderDesc::macroDefines`，它们也参与缓存键。
- 同一程序中，每个 vendor header 有一个 canonical owner。其他模块导入 owner，
  不重复包含 vendor header；include guard 不能跨 module 去重。
- NRD pass、OpenPBR 的纹理回调和 feature 宏、SHARC/NRC 等仍允许
  程序内 `#define` + `#include`。`Features/` 中复用这些配置的路径追踪、引导图和实时着色
  文件仍属于程序组合层，不能被 `Modules/` 反向引用。
- 新的 Metallic 子系统使用 `import`，不要通过 `#include` 引入 `Modules/` 的实现文件。
  同一模块拆文件使用 `__include`；新功能特化优先使用泛型、接口或显式参数。

## 编译、缓存与验证

`SlangShaderDesc::moduleName` 仍是相对 program 搜索根的路径，例如
`Features/Environment/EnvironmentLightingPrecompute`，入口名保持不变。
编译器先搜索 program 根和显式 SDK 路径，再搜索根目录及项目 `Shaders/` 下的 `Modules/`
和 `Interop/`。测试根和独立 SDK 根因此也可以直接 `import Core;`。

独立使用 CLI 时需传入模块路径，例如：

```powershell
External/slang/bin/slangc.exe Shaders/Features/Environment/EnvironmentLightingPrecompute.slang `
    -I Shaders/Modules -I Shaders/Interop -target spirv -profile spirv_1_6 `
    -matrix-layout-row-major -entry environmentLightingPrecomputeMain -o build/Environment.spv
```

每次实际编译使用新的 Slang session，避免热重载或 SDK 宏排列复用旧 module IR。
session 内的重复导入由 Slang 去重；磁盘 SPIR-V 缓存仍按请求和完整传递依赖校验。
依赖来自 Slang API，覆盖主模块、`__include` 实现和 adapter 的 HLSL header。
本次搜索规则更新提升了缓存请求版本，使旧 include 布局的缓存不会误命中。

`slang_shader_modules_and_vendor_interop` 验证菱形导入、可见性、SDK 宏隔离和排列、
依赖去重、缓存命中、实现文件及 vendor header 编辑后的缓存失效和热重载。
原有 include 缓存及后台热重载测试继续覆盖兼容路径。RHI 的光源、SH、顶点、meshlet、
混合光栅、HZB、材质分箱与渲染测试验证实际 GPU 行为。

参考：[Slang 模块与访问控制](https://shader-slang.org/slang/user-guide/modules.html)、
[Slang 编译 API](https://shader-slang.org/docs/compilation-api/)。
