# Shader 模块

Renderer 固定使用 scene-linear Rec.709/D65 HDR；显示变换、SDR/scRGB/HDR10 输出与
LookDev 默认值见 [Display Output](../Documentation/DisplayOutput.md)。

Metallic 的可复用 shader 库使用 Slang module。子系统之间用 `import`，同一模块的实现
文件用 `__include`，第三方 HLSL 的宏配置与文本包含留在 Interop 或程序内部。

| 目录 | 职责 |
| --- | --- |
| `Modules/ShaderCore.slang`、`Modules/Core/` | DescriptorHandle、DataSpan、相机、顶点解码、SH、显示颜色；不声明 push constant |
| `Modules/Core.slang` | 旧 ComputeProgram 资源表兼容入口，重新导出 ShaderCore |
| `Modules/Material.slang`、`Modules/Material/` | CPU/GPU 共用的材质与纹理数据布局 |
| `Modules/GPUDriven.slang`、`Modules/GPUDriven/` | GPU 场景、meshlet LOD、剔除、混合光栅化、可见性编码和材质分箱 |
| `Modules/Lighting.slang`、`Modules/Lighting/` | 物理光照、光源选择、光照网格、环境过滤和阴影参数 |
| `Modules/ColorGrading.slang`、`Modules/ColorGrading/` | ACES 2.0 / UE Film、全局调色、custom LUT 与三维 LUT 编解码 |
| `Interop/NeuralTextures.slang` | NTC 的唯一模块适配入口，封装 Generic/CoopVec 和无 NTC 的回退 |
| `Interop/NRDEncoding.slang` | NRD 前端编码的唯一模块适配入口 |
| `Interop/Denoising/NRD/` | 已适配的 NRD pass、bindings、配置和算法快照；保留程序内 HLSL 宏 |
| `ThirdParty/RadianceCache/` | SHARC/NRC 头文件与许可证，保留原有 HLSL 包含方式 |
| `Features/` | Shader programs：入口、pass 资源和流程相关代码；路径保持兼容现有 C++ 和管线资产 |
| `Licenses/` | 第三方 shader 许可证 |

测试探针放在 `tests/rhi/shaders/`。OpenPBR、RTXCR、RTXTF、NTC 等 SDK 继续使用
`External/` 下的源码。ShaderToHuman 的固定版本头文件仍在 `Features/Debug/ShaderToHuman/`。
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
    RWStructuredBuffer<uint> output = getResource<RWStructuredBuffer<uint>>(0);
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
当前固定的 Slang 2026.1.2 对结构体成员仍需显式标注 `public`，不要依赖新版本的成员默认可见性。

新 compute pass 使用 `ComputeKernel` + `ParameterWriter`，以具名 typed 参数承载资源。
`ParameterTransport::InlinePush` 将参数块直接推送到 byte 0，不上传 root、slot table 或标量图像句柄数组；
shader 导入 `ShaderCore` 并声明 `[[vk::push_constant]] ConstantBuffer<Params>`。
`ParameterTransport::DeviceAddress` 用于较大的参数块；shader 额外导入 `ParameterRoot`，
通过 `getParameters<Params>()` 读取一个 64 位根地址。后端检查实际 push 数据容量，不会静默截断。
两种传输共用 registry、资源保留、帧代次检查和 prepared dispatch；ABI 包含传输方式，不能混用。
CPU 通过 writer 获取 `ShaderSampledImage` / `ShaderStorageImage` / `ShaderSampler`，
shader 使用对应 `DescriptorHandle<T>` 和 `resolveDescriptor()`；普通数据使用 `ShaderDataSpan` / `DataSpan<T>`。
不要截断 64 位 handle：AS 仍保留完整地址语义。

[PostProcessParameters.h](../Source/Runtime/Render/Core/PostProcessParameters.h) 共用 C++/Slang 字段声明与显式 padding：
FinalBlit、SliderDebug（包括 DLSS-NR overlay）和 AutoExposure 使用 inline push，ColorGradingLUT 使用 BDA 参数块。
AutoExposure 的 Histogram、Reduce、Apply 复用同一份不可变参数；barrier 来自阶段读写声明，不能从 handle 推测访问。
新增参数 ABI 时应验证字段偏移、GPU 读回、mapped/native 路径和生命周期；共享声明不等于自动完成布局验证。

[LightingKernelParameters.h](../Source/Runtime/Render/Core/LightingKernelParameters.h) 提供 ClusterLightGrid、
LightGridDebug、PrepareLightsPdf、BuildReGIR 和 EnvironmentLightingPrecompute 的共享 inline 参数。
普通光照数据通过带范围的 `DataSpan<T>` 访问；PDF 每次归约直接传入源、目标 mip 的 storage handle，
不再上传完整 mip 句柄数组或创建环境 PDF 的占位光源 buffer。无 frame 的录制同样通过参数包保留资源与 kernel。
Lighting 库只导入 `ShaderCore`，不隐式引入任何参数根布局。

[RTXDIPostProcessParameters.h](../Source/Runtime/Render/Core/RTXDIPostProcessParameters.h) 共用
RTXDI Confidence（160 字节）和 Composite（56 字节）的 inline 参数声明。
Confidence 的各滤波阶段分别编码不可变参数快照，复用已注册的具名图像句柄；
历史纹理和 ping-pong 梯度的访问与同步仍由 RenderGraph 阶段声明负责。

[PathTraceStageParameters.h](../Source/Runtime/Render/Core/PathTraceStageParameters.h) 提供 SHaRC clear/resolve（96 字节）
和 NRC 输出累积/tonemap（48 字节）的共享 inline 参数。SHaRC SDK 需要 StructuredBuffer 对象进行原子操作，
因此这三个缓存 buffer 使用具名 descriptor handle；维护阶段直接读取 settings，不再依赖 cacheParams 的公共前缀。
无 guides 的 Standard 主追踪与 VisibilityBuffer 的共享资源表仍使用下述兼容入口。

主追踪共享的 shading vertex、index、primitive、instance 和 fallback position 已使用带范围的 `DataSpan<T>`，
覆盖 Standard/OpenPBR、guides、VisibilityBuffer deferred 和 alpha shadow；不再为这五类普通数据注册 buffer descriptor。
`loadPathTraceTriangle` 在解引用前检查完整索引链，并用减法检查避免偏移溢出；fallback position 另查范围。
无 guides 的 Standard/realtime/deferred 仍通过 ComputeProgram 的 DataBuffer 兼容表传递 span。

`StreamSceneRayQuery` 的 pages、page table、instances 和 header 也使用有界 BDA `DataSpan`。
streaming deferred 外层保留 `ComputeProgram` 兼容入口（90–93 为 DataBuffer），这四项不再分配 buffer descriptor；
续射页容量直接来自 span，不再读取 slot 94 的参数块。共享 stream 解码通过静态泛型 reader 同时支持 descriptor 和 BDA；
属性解码必须在相同三角形的范围校验成功后调用。

[PathTraceParameters.h](../Source/Runtime/Render/Core/PathTraceParameters.h) 为
Standard ScenePathTraceGuides、OpenPBRRayQueryPathTrace 和 OpenPBRRayQueryPathTraceGuides 提供 296 字节具名资源根，通过 BDA root 提交。
CPU 使用 ParameterWriter 编码不可变 settings、几何 span、场景/历史/环境/LUT/光源/NTC 和七路 guide 句柄，
直接调用 ComputeKernel，不构造编号 binding 表；参数包保留资源直到 GPU 完成。
自定义材质继续支持事务式编译和热重载。NeuralTextures 只导入 ShaderCore，显式接收推理资源。
OpenPBRDirectLighting 显式选择兼容入口，维持 realtime/deferred 现有布局；不应让 typed 入口读取 Core 的资源根。

`Core` 保留 `getResource<T>(slot)`、`getResourceArray<T>(slot, index)`、`getConstants<T>()`
作为尚未迁移的 ComputeProgram / SDK 调用的兼容入口。它导入 `ParameterRoot`，通过根地址读取资源表和常量，
因此不能和另一份 inline push 声明混用。数组通过 slot 的 `payload` 地址读取 registry 的句柄，
再用 `nonuniform` 选择 descriptor，不要求连续分配。RHI 不在用户 push 数据前插入 heap header。

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
- NRD pass、OpenPBR 的纹理回调和 feature 宏、SHARC/NRC、ShaderToHuman 等仍允许
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
