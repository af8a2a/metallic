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
FinalBlit、SliderDebug（包括 DLSS-NR overlay）、AutoExposure 和 ColorGradingLUT 使用 inline push。
ColorGradingLUT 的 80 字节根携带九个 typed handle 和调色设置地址；144 字节 GradingPush 作为不可变 BDA 快照，
由同一 ParameterWriter 参数包保留至提交完成，运行时设置变化在下一次执行重新编码。
UpscalerGuideResolve 使用 40 字节 inline push，包含四个具名图像句柄与 jitter；通过 ComputeKernel 提交，
不再构造编号资源表。GPU 回归直接验证前景深度选择、UV motion、jitter 和非整工作组尺寸。
DLSS depth export / alpha resolve 共用 16 字节 `DLSSSupportParams`；图形入口绑定编码后的 inline 数据，
compute 入口使用 ComputeKernel。两者均通过共享 registry 保留图像资源，不再维护私有 heap 或手写 shader index。

[StreamDeferredParameters.h](../Source/Runtime/Render/Core/StreamDeferredParameters.h) 为 streaming deferred 提供
136 字节 inline 参数，由 ComputeKernel 执行。设置、可见记录、active groups、页表、header、页数据和颜色输出均使用有界 BDA span，
visibility 使用完整图像句柄。页数据通过静态泛型 reader 复用属性解码，读取范围同时受参数容量和 span 长度约束；
其他 streaming 入口继续通过 descriptor reader 复用同一字节读取函数。
旧 raster bindings 的间接资源查找和颜色 buffer descriptor 已移除；阶段同步沿用 RenderGraph 访问声明。

[StreamCompositeParameters.h](../Source/Runtime/Render/Core/StreamCompositeParameters.h) 为 streaming 全屏合成提供
24 字节 inline 参数（颜色 BDA span、宽、高），入口已拆到 `Features/GPUDriven/StreamComposite.slang`。
合成不再传入 MeshletStreamUserPush，也不经 raster bindings descriptor 间接读取颜色；编码参数保留颜色资源，
shader 检查范围与行偏移溢出。几何剔除和光栅入口仍使用原 ABI，留待后续迁移。

[MaterialVisualizationParameters.h](../Source/Runtime/Render/Core/MaterialVisualizationParameters.h) 为材质 ray-query 可视化提供
224 字节 inline 参数，共享 112 字节相机/模式设置和完整资源句柄。ComputeKernel/ParameterWriter 接管所有资源保留，
包括 fallback position、材质贴图数组及可选 NTC 资源；移除编号 binding 表和 getResource/getConstants 依赖。
GPU 回归覆盖 13 种材质模式、mapped/native 和奇数尺寸 resize 恢复；NTC SDK 未启用时不代表验证了 NTC 推理路径。

[MaterialRasterParameters.h](../Source/Runtime/Render/Core/MaterialRasterParameters.h) 提供材质 shader-object
共用的 160 字节 inline 参数：positions、material indices、materials、transforms 为有界 BDA，camera 与 batch offset 直接内联。
每个批次在 beginRendering 前编码不可变快照；默认/备用 shader 共用 ABI，移除私有 heap、五个 descriptor 与可变参数 buffer。
GPU 三材质回归验证 default/alternate/default 切换、非目标批次保持原色，以及奇数尺寸 resize 后恢复。

[BunnyWireframeParameters.h](../Source/Runtime/Render/Core/BunnyWireframeParameters.h) 提供 wireframe shader-object
共用的 160 字节 inline 参数：positions/transforms 为有界 BDA span，128 字节相机和线框设置直接内联。
移除私有 heap、三个 buffer descriptor 和可变参数 buffer；资源由参数包保留，顶点解引用检查 span 范围。
GPU 回归覆盖 mapped/native、vertex/fragment 布局、相机变化和奇数尺寸 resize 后的输出恢复。

[ImageSampleParameters.h](../Source/Runtime/Render/Core/ImageSampleParameters.h) 提供 ImageSample 的
8 字节 inline 图像句柄；图形入口使用共享 registry 和执行绑定，不再维护私有 heap 或截断 shader index。
每帧从 prepared scene 获取当前图像并编码资源保留；GPU 回归检查连续帧与奇数尺寸 resize 后的输出恢复。

[RenderGraphBufferParameters.h](../Source/Runtime/Render/Core/RenderGraphBufferParameters.h) 提供
Write/Copy 共用的 32 字节 inline 参数，普通 buffer 通过有界 BDA span 访问，无需分配 view/descriptor。
GPU aliasing 回归覆盖独立/复用分配、graphics/compute 队列、串行/并行录制、joined/pipelined 提交与跨帧读回。

[DebugVisualizationParameters.h](../Source/Runtime/Render/Core/DebugVisualizationParameters.h) 提供
普通场景和 streaming RTAS 可视化共用的 112 字节 inline 参数（AS、输出图像、camera/mode）。
两条 CPU 路径使用 ParameterWriter / ComputeKernel；AS 使用完整规范句柄，支持 mapped/native 描述符模式。

[MaterialSampleParameters.h](../Source/Runtime/Render/Core/MaterialSampleParameters.h) 提供 RTXCR 材质演示的
72 字节 inline 参数（输出句柄和 64 字节材质设置），直接通过 ComputeKernel 提交。
独立 pass 回归覆盖 overview、Chiang hair、far-field hair、subsurface、曝光更新和奇数尺寸 resize。

[VisibilityMaterialParameters.h](../Source/Runtime/Render/Core/VisibilityMaterialParameters.h) 提供
VisibilityBufferMaterial 的 152 字节 inline 参数，统一 resident/streaming 图像与 buffer 句柄及标量设置。
串行和并行准备均使用 ComputeKernel 的不可变 prepared dispatch；未使用的可选资源不注册占位 descriptor。
`StreamActiveGroup` 的 shader 布局统一位于 GPUDriven 模块，供材质解码和 ray-query 解码复用。

[MaterialErrorParameters.h](../Source/Runtime/Render/Core/MaterialErrorParameters.h) 提供错误材质的
16 字节 inline 参数。首次材质编译失败时，color 输出错误棋盘，其余 guide 清零；共享初始化/分派函数
使用 ComputeKernel 和 ParameterWriter，测试覆盖 mapped/native、多种标量/向量格式及奇数尺寸。

[DebugProbeParameters.h](../Source/Runtime/Render/Core/DebugProbeParameters.h) 提供 GPU probe 的 72 字节
inline 参数及共享归约记录布局。参数在 barrier 录制前完成编码；源/输出通过共享 registry 保留，
probe 后恢复原资源状态和执行绑定。GPU 回归覆盖大于单组的扫描、非有限值、位字段、watch 与绑定恢复。
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
Standard 主追踪与 VisibilityBuffer 主着色现已通过共享 inline 根提交。

主追踪共享的 shading vertex、index、primitive、instance 和 fallback position 已使用带范围的 `DataSpan<T>`，
覆盖 Standard/OpenPBR、guides、VisibilityBuffer deferred 和 alpha shadow；不再为这五类普通数据注册 buffer descriptor。
`loadPathTraceTriangle` 在解引用前检查完整索引链，并用减法检查避免偏移溢出；fallback position 另查范围。
Standard/realtime/deferred 均通过 ComputeKernel inline 根引用这些有界 span。

[ResidentLODParameters.h](../Source/Runtime/Render/Core/ResidentLODParameters.h) 为 resident LOD 的
reset/select/arguments/scatter 共用 184 字节 inline 参数，所有输入、输出及 scratch 使用有界 BDA span。
CPU 直接消费 GPUScene buffer views，编码四个 prepared dispatch，保留阶段访问计划及 indirect consumer 同步；
LOD 计算不注册 buffer descriptor，raster consumer 仍按自身接口保留 selection/arguments 句柄。

[InstanceCullParameters.h](../Source/Runtime/Render/Core/InstanceCullParameters.h) 定义 resident 实例剔除/reset 的
104 字节 inline 参数。settings 使用不可变 BDA 快照，instances、visibility、visible IDs 与 stream owner mask
使用带范围的 span，counter/HZB 使用 canonical typed handle。CPU 按 early/late 选择 HZB，先准备 reset/cull
参数包，再由 GPUScene 录制原有同步。
[StreamInstanceCullParameters.h](../Source/Runtime/Render/Core/StreamInstanceCullParameters.h) 提供 streaming
实例剔除/reset 的 112 字节 inline 参数；runtime 统一编码 settings 快照、实例/可见性 span、
counter/HZB handle 与标量剔除设置，独立 StreamAsset 和混合 VisibilityBuffer 共用。
GPUScene 实例剔除只接收 prepared dispatch，旧 pushData 字段及无调用方的 recordCull 接口已删除。

[HZBParameters.h](../Source/Runtime/Render/Core/HZBParameters.h) 定义 resident/streaming 共用 HZB 的 56 字节 inline 参数。
每个 mip 直接携带深度图像 handle、HZB 有界 span、源/目标尺寸与偏移及正反 Z 标志，
不再读取 streaming params/rasterBindings 描述符或整份 MeshletStreamUserPush。
CPU 先编码全部不可变 `PreparedComputeDispatch`，再由 GPUScene 按统一访问计划录制 mip 间 barrier；
参数包负责保留 HZB 与深度资源；两条路径共用 `Features/GPUDriven/HZB.hzbMain`。
[HZBSPDParameters.h](../Source/Runtime/Render/Core/HZBSPDParameters.h) 提供 SPD 的 40 字节 inline 参数，
深度、输出与计数器均使用 canonical typed handle，保留 mip 6 与最后工作组计数器的 release/acquire。
GPUScene HZB 录制只接收 prepared dispatch，旧 pipeline/heap/pushData 描述已移除。

`StreamSceneRayQuery` 的 pages、page table、instances 和 header 也使用有界 BDA `DataSpan`。
streaming deferred 外层使用 `ComputeKernel` inline 入口；四个 span 已统一为共享 `StreamSceneParameters`，
通过 settings 中的 BDA 根地址传递，不再使用 90–93 数字槽位或 buffer descriptor。
`ParameterWriter` 编码不可变快照，并将四个资源保留至提交完成；material bin dispatch 复用同一快照。
ScreenSpaceShadows 的 CLAS alpha-mask 查询复用同一声明，通过 `ShadowTraceParameters.streamScene` 传递根地址。
旧 90–94 绑定已移除。主追踪 settings 为 272 字节 BDA 数据，不是原生 push constant。
续射页容量直接来自 span，不再读取 slot 94 的参数块。共享 stream 解码通过静态泛型 reader 同时支持 descriptor 和 BDA；
CLAS surface 通过 `decodeStreamRaySurface` 统一执行三角形校验和属性解码，避免未校验的属性读取。
`StreamSceneDecode.slang` 显式接收 `StreamSceneParameters`，供 CLAS ray-query 与 GPU 边界测试共用；
实例读取检查地址、stride 和索引，三角形解码检查 header、cluster stride、页表及页数据范围。
CPU 的 deferred 与 alpha shadow 共用 `encodeRayQuerySnapshot`，保持 span 布局、编码及资源保留一致。
[ShadowTraceParameters.h](../Source/Runtime/Render/Core/ShadowTraceParameters.h) 定义阴影入口的 88 字节 inline ABI。
常规、NTC/CoopVec、stream TLAS 与 pending 五种变体统一使用 ComputeKernel；深度和五路输出为 typed handle，
320 字节设置为有界 BDA span，几何为共享 PathTraceParameters 的不可变快照。移除编号资源表、
16 字节 ShadowGeometryPush 和按贴图数量重建 pipeline 的逻辑；参数包保留贴图、几何及 BLAS 间接所有者。
原有 RenderGraph 阶段同步、参数池供 deferred 消费和 SIGMA 接入保持不变。
`StreamSceneRayQuery` 显式接收 `StreamSceneParameters`、加速结构、bitangentFlip 与静态泛型
`IStreamRayMaterials`；不读取 `gScene` / `gMaterials`，不依赖 `ScenePathTracePush` 或参数根。
`ScenePathTrace` 在调用边界适配 typed root 的材质和 alpha 采样，
`ScreenSpaceShadows` 共用同一适配器；空场景根地址在边界拒绝。CPU 的 64 字节场景布局与资源保留不变。
法线和 TBN 仍保持 authored/world-space 语义。`stream_data_decode_bounds` 的 GPU 探针覆盖
显式材质 provider、正反向射线、bitangent 翻转、无效材质/实例/三角形及 mask/blend alpha 阈值；
alpha 候选判定只接收场景与材质 provider，单次加载材质、插值 UV，不构建完整 TBN；
不透明候选也先检查几何范围，探针覆盖无效三角形、空 pages、截短 header 与空 page table。
这些辅助层检查不等同于 CLAS 遍历或完整场景的视觉验证。
共享 `ScenePathTrace.slang` 仅提供 typed 场景辅助实现；标准生产入口为 `ScenePathTraceInline.slang`。
旧 `PathTraceParameterRoot.hlsli` 和编号槽位分支已移除。stream surface 探针使用显式材质 provider，
不创建场景资源根；motion-vector 探针使用 16 字节 inline 有界输出 span，不再占用 slot 63。

[PathTraceParameters.h](../Source/Runtime/Render/Core/PathTraceParameters.h) 为
Standard ScenePathTraceGuides、OpenPBRRayQueryPathTrace 和 OpenPBRRayQueryPathTraceGuides 提供 296 字节具名场景资源快照，由共享 inline 根通过 BDA 地址引用。
CPU 使用 ParameterWriter 编码不可变 settings、几何 span、场景/历史/环境/LUT/光源/NTC 和七路 guide 句柄，
直接调用 ComputeKernel，不构造编号 binding 表；参数包保留资源直到 GPU 完成。
自定义材质继续支持事务式编译和热重载。NeuralTextures 只导入 ShaderCore，显式接收推理资源。
OpenPBRDirectLighting 的 realtime/deferred 分支均使用 inline 根，不读取 Core 的资源根。

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

Streaming 页表初始化/更新使用共享 `StreamPageTableParameters`（32 字节 inline push）：
`pages` 与 `patches` 均为有界 BDA span。每次更新由 `ParameterWriter` 保存独立 patch 快照，
替代按 frame slot 复用的可变 upload buffer；提交前多次录制不会覆盖前一次数据。
更新只修改 residency word，保留 `lastRequestFrame`；初始化清零两个 word。
旧 update descriptor/header 及 raster 根中的对应 reserved 字段均已移除。

Streaming traversal 使用共享 `StreamTraversalParameters`（96 字节 inline push），由 `ComputeKernel` 录制。
设置与 resident page 列表为每次录制保留的 BDA 快照，instances 为有界 BDA；
primitives/groups/nodes 及含原子操作的 page table/requests 通过 canonical typed handles
传给现有共享遍历 helper。旧 resident-page upload ring 和 descriptor 已移除；
该入口不再读取旧 raster 根；旧根及 reserved 字段现已移除。

Streaming active build、cooperative LOD 与 distributed demand 共用 168 字节
`StreamActiveBuildParameters` inline ABI。设置由 BDA 快照保留，资源统一为 canonical
typed handles；阶段与 demand/统计/栅格裁剪启用标志显式传入。旧设置里的四个资源索引
已变为 reserved，LOD topology/state/demand 的独立租约已移除，由参数包保留资源。
三个入口使用 `ComputeKernel`，保留原阶段顺序、group barrier 与跨阶段同步。

Streaming TLAS 输入使用 `StreamTLASParameters`（80 字节 inline push）：settings、
instances、BLAS records、fallback addresses 与 output 全部为有界 BDA span。
设置快照与输入/输出 allocation 由参数包保留；入口无需 buffer descriptor。
旧 fallback-address/TLAS-output descriptor 已移除，原加速结构构建同步保持不变。
Dynamic BLAS 地址表由 BLAS 输入阶段写入，不属于 TLAS 输入入口的读取资源。

Streaming BLAS 输入构建使用 `StreamBLASParameters`（112 字节 inline push）与
`ComputeKernel`。设置为 BDA 快照，所有缓冲使用 canonical typed handles，header
和 uint4 扫描区共享同一资源；各阶段保留独立参数、缓存重置状态和 CLAS 发布版本。
旧 BLAS/CLAS 专用租约与裸索引已移除，原生构建缓冲仍保留到提交完成。

Streaming raster 候选压缩的 setup/count/prefix/scatter 共用
`StreamCandidateParameters`（72 字节 inline push）。active header、groups 与间接参数
使用 BDA span，bin 和 visibility 使用 typed handles；不再读取 raster bindings 中的
visibility 裸索引。四个阶段通过 `ComputeKernel` 直接/间接录制，保留稳定顺序、
容量溢出时整体硬件回退及早/晚重试语义。后续 cull 同样使用独立的 inline ABI。

[StreamClassifyParameters.h](../Source/Runtime/Render/Core/StreamClassifyParameters.h) 定义 stream cluster
P0/P1 分类共用的 80 字节 inline ABI。settings 使用不可变 BDA 快照，active groups、page words
和 GPUScene instances 使用有界 span，输出 bin 使用 typed registry handle。
P0 保留材质覆盖及曲面细分的强制硬件判定，P1 消费 cull 已过滤的队列；两个入口通过
`ComputeKernel` 录制，参数包将输入资源保留到提交完成。

Streaming cluster cull 的 P0/P1 共用 `StreamClusterCullParameters`（128 字节 inline push）。
settings 与 raster settings 为不可变 BDA 快照，十二个资源使用 canonical typed handle；
资源索引解析只保留在旧 raster 适配层，共享加载器显式接收资源与 early/late phase。
Cull 与间接参数 finalize 各自编码参数包，经 `ComputeKernel` 录制并保留资源；
保留页请求、HZB 重试和稳定 bin 顺序，移除旧 candidate-arguments 专用 descriptor 租约。

混合光栅稳定分桶的 reset/histogram/arguments/scatter 共用 `HybridBinParameters`
（56 字节 inline push）。bin 使用 typed registry handle，间接参数输出使用有界 BDA span；
四阶段通过 `ComputeKernel` 录制并保留 allocation，沿用原有阶段同步、稳定顺序和溢出回退。
分桶私有 heap 的 bin/arguments descriptor 已删除。`producerPixelBuffer` 暂作为旧 raster
consumer 的 header 数据透传，分桶自身不解引用它；像素寻址仍需随 raster 入口继续迁移。

混合光栅 clear/arguments/triangle raster 使用 `HybridRasterParameters`（56 字节 inline push），
queue/pixels 为 typed handles，间接参数为有界 BDA span；三阶段均通过 `ComputeKernel` 录制。
`hybridRasterTriangle` 接收 typed pixel handle，未迁移的 raster 入口经裸索引适配重载调用同一算法。
私有 queue/arguments descriptor 已删除。

混合光栅 graphics resolve 使用 `HybridResolveParameters`（24 字节 inline push），以有界
BDA span 读取 64-bit depth/visibility 像素；资源在 draw 前编码并绑定，保留到提交完成。
`VisibilityHybridRasterizer` 已无私有 heap、旧 `HybridPush` 或裸 push 提交，host settings
与各入口 wire ABI 分离。外部 raster 仍透传 bin header 的 producer pixel index，待后续迁移。

Streaming 软件光栅的普通、参考、plane、cooperative、WorkBins、WorkControl 和 Group32/64/128 九个入口共用
[StreamRasterParameters.h](../Source/Runtime/Render/Core/StreamRasterParameters.h)（88 字节 inline push）。
settings 为不可变有界 BDA 快照；pages、groups、header、page table、instances、bins 和 pixels
使用 canonical typed handles，record base/capacity 显式传入。入口不再解析 rasterBindings buffer
或 bin header 的 pixel index，生产默认 Group32 同样通过 `ComputeKernel` 与参数包保留资源。
cooperative 与 group 入口共用显式资源的 wave 解码核心，保留原有 barrier、尾部和反射变换规则。
WorkControl 回放保留生产 `ComputeKernel`，为 scratch allocations 重新编码 canonical handles
和 settings BDA 地址，归档 v2 的 `Push.bin`、`ReplayPush.bin` 及两份 settings；每次恢复后仍
逐字节校验像素输出、只读输入和生产 guard。v1 快照只保留离线校验，不按 v2 执行。
shader trace 复用同一 88 字节布局，在生产参数绑定后替换诊断 shader。
GPU 测试对九个 inline 入口逐像素对照：plane 保持覆盖/ID 且深度误差不超过 1e-6，
其余 packed 输出相同。硬件 raster 的旧资源适配层仍待迁移。

诊断 workload 的 reset/coverage 阶段共用
[StreamWorkloadParameters.h](../Source/Runtime/Render/Core/StreamWorkloadParameters.h)（96 字节 inline push）。
它复用 88 字节 raster 输入，另加 typed uint64 counter handle；未使用的 raster pixel handle
保持 invalid，不注册或保留生产像素 buffer。两个阶段共用不可变参数包，保留原有阶段访问声明与
同步。私有 workload descriptor、旧 push 编码及软件解码 raw-push 适配器已删除。
计数表示覆盖和原子尝试次数，不表示深度竞争获胜次数；GPU 回归检查 early/late、空列表及
不同分类路径的计数守恒。预热列表中的 Composite 顶点/片元入口指向 `Features/GPUDriven/StreamComposite`。

Fragment 中的 BDA 范围校验失败必须显式 return；不能只 discard 后继续读取物理地址。
Resolve 的 indexed-mesh/奇数尺寸回归覆盖该路径，避免 helper invocation 越界读取。

[RTXDITraceParameters.h](../Source/Runtime/Render/Core/RTXDITraceParameters.h) 定义 SceneRTXDI 主追踪的 128 字节 inline 根。
十四路当前/历史/降噪输出使用 typed handle；设置和共享 PathTraceParameters 场景快照使用 BDA。
普通几何复用 PathTracePrimitive/Instance/Material 布局，通过有界 span 读取，候选和提交命中均检查完整索引链，
fallback position 也检查范围。NTC、环境与 ReGIR 资源显式编码；移除编号 binding 表及按贴图数量重建管线。
原有 ReSTIR 阶段访问、历史发布和 RELAX 后处理保持不变。布局测试覆盖 mapped/native；无 NTC 构建不验证 NTC 推理。

[PathTraceInlineParameters.h](../Source/Runtime/Render/Core/PathTraceInlineParameters.h) 为标准无缓存路径追踪提供
40 字节 inline 根，直接携带 settings 地址、输出和两路历史句柄；其余资源复用 PathTraceParameters 场景快照。
生产入口为 `ScenePathTraceInline.scenePathTraceMain`，CPU 复用 guides/OpenPBR 的 ParameterWriter 编码链，
无编号 binding 表，也不因贴图数量变化重建管线。NRC 的独立 inline 入口见下文；
标准回归覆盖 off/SHaRC/off 的 ABI 切换、逐帧 fallback 检查、有限 HDR 与奇数尺寸 resize。

[SharcTraceParameters.h](../Source/Runtime/Render/Core/SharcTraceParameters.h) 将 SHaRC update/query 主追踪统一到
72 字节 inline ABI，包含标准追踪的 40 字节根、不可变缓存设置地址及三个 canonical 缓存 handle。
两阶段共享同一参数包；场景编码与标准/OpenPBR/guides 共用，移除 SHaRC binding 20–23 及可变缓存设置 buffer。
clear/update/resolve/query 的资源访问和取消后的清理语义保持不变；维护阶段仍使用独立的 96 字节 inline 参数。
布局测试分别编译 update/query 的 mapped/native 变体，场景测试验证 off/SHaRC/off 切换、历史发布和取消。

[NRCTraceParameters.h](../Source/Runtime/Render/Core/NRCTraceParameters.h) 为 NRC update/query 提供 88 字节 inline ABI：
共享标准主追踪参数、不可变 cache settings BDA 和五个具名 buffer handle。
两个阶段共用 BeginFrame 之后编码的快照，移除 binding 20、24–28 和旧缓存参数池；
QueryRadianceParams 保持 ScalarDataLayout。SDK train/resolve 与提交后 EndFrame 顺序保持不变。
`nrc_trace_parameter_spirv_layout` 覆盖 update/query 的 mapped/native 字段偏移和共享头缓存依赖。
启用 `METALLIC_TEST_NRC_CACHE=1` 可运行真实阶段/取消回归及 `nrc_context_lifecycle`：
后者只执行 initialize/configure/destroy，不执行 shader，用来隔离 SDK 资源生命周期问题。
当前 NRC 0.15 环境在 context-only 测试的设备销毁阶段仍报告原生对象泄漏，
同一进程销毁设备后再次初始化 NRC 还观察到无效原生句柄及访问异常，诊断时应单独进程运行该测试。
因此 ABI 布局与 shader warmup 通过不代表 NRC 完整运行验证通过。

[PathTraceGuidesInlineParameters.h](../Source/Runtime/Render/Core/PathTraceGuidesInlineParameters.h) 定义标准/OpenPBR guides 共用的
96 字节 inline ABI：40 字节主追踪参数加七路 guides 的 typed handle。
不带 guides 的 OpenPBR 复用 40 字节 PathTraceInlineParameters；所有普通主追踪入口均由 ComputeKernel inline 提交。
`PathTraceInlineRoot.hlsli` 集中声明根参数和 shader 访问器，SHaRC/NRC 嵌套根也复用该访问器。
CPU 只编码一次不可变场景快照；已移到 inline 的 settings、输出、历史和 guides 在快照中清零，
避免保存两份有效资源表示。几何、材质、光源、LUT 和 NTC 资源仍由快照及 ParameterWriter 保留。
布局回归同时检查 inline 根与场景快照的 C++/SPIR-V 偏移；实际 guides 回归读取八路输出，
检查有限 HDR、法线长度、roughness/depth 范围、静态 motion 和奇数尺寸 resize。

[RealtimeLightingParameters.h](../Source/Runtime/Render/Core/RealtimeLightingParameters.h) 将 SceneRealtimeLighting 主入口迁到
48 字节 inline ABI：共享 PathTraceInlineParameters 加环境 irradiance SH 的具名 buffer handle。
材质、几何、LUT 和灯光沿用不可变场景快照；不再为该入口构造编号 binding 表或按贴图数量重建管线。
OpenPBRDirectLighting 的灯光读取统一经过 gPathTraceLights，deferred 分箱算法保持原样。
共享 CPU 编码器仅在实际需要时编码 ReGIR/PDF，避免普通 realtime 未分配这些资源时编码失败。
`realtime_lighting_parameter_spirv_layout` 验证 mapped/native 字段布局；光度回归覆盖 lux/EV 等价、删灯、
手动/HDR 输出与自动曝光，对照回归比较透视/正交下的 baseColor、shadingNormal 和最终着色。

[DeferredShadingParameters.h](../Source/Runtime/Render/Core/DeferredShadingParameters.h) 定义 VisibilityBufferDeferred 的
96 字节 inline 根和 304 字节不可变资源快照。直接与分箱入口共用 ABI，根中显式传递 binIndex、
visibility/depth/domain、motion/deviceDepth 输出及场景快照地址。
GPUScene 的八类 raster 视图编码为有界 BDA span，保留 offset/size/stride；分箱和 tile 数据也使用 span。
灯光网格、环境、阴影、纹理反馈、sampler 和 stream 解码资源使用具名 typed handle，不再构造数字槽位表。
每个材质类仅编码一份 inline 根，场景与 deferred 快照在批次内复用，借助 ComputeKernel.prepareIndirectBatch
保留间接参数切片和各分类 kernel；各类写入互斥像素，阶段同步及取消/提交资源保留保持原契约。
ScenePathTracePass 的普通、缓存与 deferred 入口均已使用 ComputeKernel，旧 ComputeProgram 数组与绑定表已移除。
`deferred_shading_parameter_spirv_layout` 覆盖 direct/binned 的 mapped/native 根和资源快照字段偏移。

Material binning 的 reset/classify/arguments 三阶段共用 `MaterialBinningParams.h` 的 80 字节 inline 根。
visibility 为 typed handle，bins/tiles/indirect 输出为有界 span；六组 resident/streaming 输入 span
通过 96 字节不可变资源快照传递。同一参数包保留全部资源，原有阶段访问计划负责同步。

`TextureResidencyProbe` 使用共享 `TextureResidencyProbeParameters.h` 的 32 字节 inline 参数，
反馈缓冲为有界 BDA span；不再通过 slot 0、getConstants 或 GetDimensions 取资源与范围。
streaming、分段提交/取消和 RenderGraph 多消费者测试共用 typed 参数编码，参数包保留反馈缓冲。

纹理采样探针共用 `TextureProbeParameters.h`：TextureResourceProbe 使用 32 字节 inline 参数，
TextureStreamingProbe 使用 64 字节。CPU 将所选纹理编码为单个 typed handle，不上传整张编号资源表；
输出与反馈为有界 BDA span，sampler 使用 typed handle，所有资源由帧参数包保留。
KTX2 资源、压缩上传和 streaming 稳定性测试覆盖实际采样与物理 mip tail 替换。

位置获取与压缩顶点 smoke-test 共用 `SceneProbeParameters.h` 的两个 32 字节 inline ABI。
位置探针引用 typed 场景资源快照，输出为有界 span；顶点探针的输入和输出均为有界 span。
位置测试继续覆盖扩展/回退、实例移动、authored tangent 与截断 CPU slice，移除旧 slot 63 和编号几何绑定。

`OpacityMicromapProbe` 同样使用 `SceneProbeParameters.h` 中的 32 字节 inline 根，
引用 typed 场景快照和有界 uint2 输出 span；OMM 开关、材质更新与 standard/partitioned TLAS 切换
继续通过逐射线 CPU alpha 参考验证。KHR OMM 的 validation 运行需要 1.4.357 或更新的验证层；
较旧验证层下只验证 shader alpha 回退，真实 OMM 测试需明确传入 `--rhi-no-validation`。

ShaderToHuman gather/scatter 示例使用 `ShaderToHumanParameters.h` 的 8 字节 inline 输出句柄。
调用方应通过 ParameterWriter.storageImage 编码目标图像，以 kShaderToHumanABI / InlinePush
创建 ComputeKernel 并提交参数包；scatter 仍需要在先前图像写入之后同步。fragment 示例只使用
SV_Position，不访问输出图像。现有测试覆盖 mapped/native 的三个入口编译及 vendor include 依赖。

NRD 各算法入口使用共享 `NRDParameters.h` 的 16 字节 inline 根，直接传入算法常量与资源表的
BDA 地址，移除 ParameterRoot 的额外根读取。两个不可变快照仍由同一 ParameterWriter 参数包保留；
算法资源槽表、dispatch plan、同步与历史恢复逻辑保持原有语义。

NRD 资源快照统一为 `NRDResourceHandles`（400 字节）：CPU 使用 ShaderSampledImage、
ShaderStorageImage 和 ShaderSampler，shader 保存完整 uint2 wire handle，由生成 binding 按算法类型
构造 DescriptorHandle<T>。不再截断为 uint32 索引或补零重建句柄；VendorNrd.py 同步生成此格式。

StreamSceneParameters 是普通 64 字节 BDA 数据快照，不再有独立 dispatch ABI。
shadow/deferred 的父 ParameterWriter 同时保留四个 span 与快照，最终 inline 参数包统一绑定并保留资源，
移除嵌套 EncodedParameters 和提前 bindResources。

stream 解码与 surface/alpha GPU 探针共用测试侧 `StreamProbeParameters.h` 的 80 字节 inline 根，
直接携带 StreamSceneParameters 和输出 span；40 个 GPU 检查保留 null、错误 stride、截断范围、
材质 provider、TBN 和 alpha 阈值覆盖，并确认不产生 descriptor 写入。

MaterialBinningProbe 的 PROBE_TYPED 分支使用共享 80 字节 inline ABI，reset 和间接消费共用
四个有界数据 span；旧 fixture/对照分支单独编译。回归保留错误 ABI、非法 indirect 偏移、
大于单维 group 上限的调度、逐像素唯一覆盖和跨帧复用检查。

registry 纹理数组寿命与非一致 image/sampler 索引测试共用 `TextureBindingProbeParameters.h`，
分别使用 24/32 字节 inline 根。保留 descriptor 输出以验证 registry 生命周期；图像数组仍为
不可变完整 handle 快照。两条路径均支持 mapped/native 模式，包含提前释放和提交完成后复用检查。

DataSliceProbe 的普通数据生产和间接消费使用共享 56 字节 inline ABI；兼容 getData adapter
仅在 DATA_PROBE_ADAPTER 分支编译。测试继续验证 transfer-only 拒绝、slice 偏移/范围、
间接参数与输出边界、数据值及提交期资源保留。

VisibilityBufferComposite 使用共享 `VisibilityCompositeParameters.h` 的 72 字节 inline 参数。
图像与 resident/stream buffer 均传递完整 typed handle，由 ParameterWriter 参数包保留。
stream-only 场景不要求 resident buffer；关闭 debug 显示时保留清屏行为而不编码未使用资源。

剔除 GPU 探针共用测试侧 `CullingProbeParameters.h`：法线锥为 16 字节 inline 有界输出，
两阶段遮挡为 32 字节 inline 输出 span 与两个 typed HZB handle，均通过 ComputeKernel 提交。
共享遮挡判定显式接收阶段编号、历史/当前 HZB，不再依赖整份旧 raster push；
raster 调用边界负责按 frameIndex 选择 ping-pong HZB。

Resident raster 的 mesh、细分、分箱与软件入口共用 `ResidentRasterParameters.h`：
48 字节 inline 根携带绘制设置与 168 字节 typed 资源快照地址。
LOD selections 由完整 typed handle 提供，tessellation data 使用有界 BDA span；
352 字节相机参数只保留启用标志，不再嵌入这两类资源索引。
混合队列/分箱使用独立具名字段；当前/历史 HZB 在 CPU 编码边界选择，
每个提交参数包保留快照及对应 lease，不再使用 132 字节混合索引根。

旧 `VisibilityBufferShading` 的未调度着色实现、静态 LUT 索引宏及重复的 resident push 声明
已移除。生产材质解析继续使用 `VisibilityBufferDeferred` 的共享 typed inline ABI。

Stream mesh/细分及独立 stream pass 共用 `StreamHardwareParameters.h` 的 104 字节 inline 根。
八个 typed handle 保留完整 wire 值，分箱读/写 view 分别声明；显式 flags 控制可选队列/分箱。
settings 与 raster settings 通过两个不可变 BDA 地址传递，不再读取设置 buffer descriptor；
快照由同一父参数包保留，根大小保持 104 字节，ABI 已升级。
`MeshletStreamRuntime.hardwareParameters` 将自身资源加入父 ParameterWriter；调用方同时保留
混合光栅资源并提交统一参数包。旧 136 字节 raster 根与遍历/BLAS reserved 槽位已删除。
嵌套 raster settings 使用共享 `StreamRasterResourceSettings.h`，资源字段全部为完整 typed handle。
材质贴图 remap 的元素统一为 8 字节 `ShaderSampledImage` / `DescriptorHandle<Texture2D<float4>>`，
CPU 上传完整 `shaderValue()`，resident/stream alpha-mask 直接读取 typed handle，
不再截断为 32 位索引后补零重建；原有材质纹理 leases 继续随阶段提交保留。

Stream raster settings 已移除无消费者的 depth/visibility/deferred-color 图像输出、
visible-instance ID 列表与计数器索引，相关数据继续通过
各自的 typed cull/deferred 参数传递。七个资源字段使用 typed handle，位移表使用有界 BDA span；
共享结构为 128 字节，含显式 padding 和可选资源 flags；C++/Slang 共用同一声明。
hardware、cluster cull、active build ABI 同步升级，CPU 不截断 shaderValue，shader 不补零重建 handle。

位移材质表同样保留完整 `ShaderSampledImage`：每材质 16 words 中 word 0/12 分别保存 handle 低/高位，
UV、位移标量和 pattern 起点保持原位置；resident/stream 共用 typed `TessMaterial.texture`，enabled 控制可选采样。

Hybrid bin header 的 word 11 保留为零，不再传递 pixel descriptor 索引；
resident 软件光栅通过资源快照的 typed pixels，stream 通过 StreamRasterParameters.pixels 访问。
HybridBinParameters 收缩为 56 字节，删除无调用的 StreamHardware raster 包装和 raw-index 三角形适配器。

Hybrid GPU 探针共用 HybridProbeParameters.h：mesh 入队与分桶为 32 字节 inline 根，
prepared raster 对照为 64 字节；输入顶点/候选走有界 BDA，输出通过 registry typed handle，
编码参数包保留至测试 fence 完成。已移除共享光栅模块中仅供旧探针使用的 raw-index 入队与 prepared 适配器。

resident/stream 位移表统一使用 TessellationData 的有界 BDA 读取，不再分配位移 buffer descriptor。
stream 快照构造验证 CPU slice 的设备、范围、stride 与 alignment；阶段提交包保留表的原生 allocation。

SphericalHarmonicsProbe 使用共享 16 字节 typed inline 根与 ComputeKernel；保留 StructuredBuffer
输入/输出以验证球谐库的 load/store API，移除 slot 0/1 资源表，提交参数包保留 buffer leases。

ReGIRVirtualLightProbe 使用 ReGIRProbeParameters.h 的 64 字节共享 inline 根，具名 output/lights/grid/pdf
字段替代 slot 0/50/52/53；ComputeKernel 参数包保留资源，采样位置、次数与 seed 直接随根传递。

MaterialValueProbe 使用 32 字节 inline 根中的实例/输出 BDA span，并检查调用范围；
生成的 MaterialValueDispatch 必须由入口提供 METALLIC_LOAD_MATERIAL_VALUE 适配器，
不再隐式读取 slot 97。生产入口继续使用 PathTraceInlineRoot 的 typed materialValues。

AutoExposureFixture 使用共享 24 字节 inline ABI：typed storage image 与尺寸、亮度、异常值模式；
ComputeKernel 参数包保留每帧输出 view，曝光统计与内部阶段回归不再依赖旧 ComputeProgram 适配层。
