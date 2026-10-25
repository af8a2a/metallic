# Shader 模块

See [Color Pipeline](../Documentation/ColorPipeline.md) for ACEScg working-space and source/display contracts.

Renderer 默认使用 scene-linear ACEScg/AP1/D60 HDR；Rec.709/D65 保留为兼容模式。显示变换、SDR/scRGB/HDR10 输出与
LookDev 默认值见 [Display Output](../Documentation/DisplayOutput.md)。

所有项目自有 `.slang` 文件（含 Features 入口、Interop 包装、生成源码和测试）必须声明
`module Name;` 或 `implementing Name;`。子系统之间用 `import`；同模块拆分使用
`__include` 与匹配的 `implementing`，不允许无归属代码片段或对 `.slang` 的裸 `#include`。
第三方 HLSL 和共享 C++/Slang 头文件可以在唯一的模块 owner 内保留文本包含；不能通过改名
`.hlsli` 绕过模块归属。在仓库根执行 `python -B Tools/CheckShaderModules.py`，递归检查 `Shaders/**/*.slang`
和 `tests/**/*.slang`（含 editor 与 RHI 探针），验证唯一归属、同 owner 的实现引用和裸 include 禁令。
外部 SDK 源码不在扫描范围；运行时生成的 MaterialValueDispatch 则单独检查生成器的归属声明。
生成的 `MaterialCoverageGenerated` 独立模块也检查其生成器归属；后端与缓存约定见
[Native Slang Coverage](../Documentation/MaterialCoverageNativeSlang.md)。
配置了 Python 的测试构建同时注册 `MetallicShaderModuleOwnershipAudit` 和检查器回归测试。

编译复杂度与源码实验记录见 [Shader compilation complexity](../Documentation/ShaderCompilationComplexity.md)。
比较实际请求和最终 SPIR-V 规模，并分别验证完整冷缓存 warmup、渲染输出及 GPU 运行时间；
源码行数、import 数或更小的产物本身不能证明性能改善。

| 目录 | 职责 |
| --- | --- |
| `Modules/ShaderCore.slang`、`Modules/Core/` | ResourceHandle、BufferSpan、旧资源兼容接口、相机、顶点解码、SH、显示颜色；不声明 push constant |
| `Modules/Core.slang` | ComputeProgram 具名资源参数、常量和底层测试兼容入口，重新导出 ShaderCore |
| `Modules/Material.slang`、`Modules/Material/` | CPU/GPU 共用的材质与纹理数据布局 |
| `Modules/MaterialProgram.slang`、`Modules/MaterialProgram/` | 无模型/资源依赖的 `Material<TProgram>` 泛型封装、SurfaceMaterialContext、MaterialInstanceRef 与 BSDFEval/BSDFSample；见 [Phase 2](../Documentation/MaterialSystemPhase2.md) |
| `Modules/OpenPBRMaterialParameters.slang`、`Modules/RTXCRMaterialParameters.slang` | 各模型独立参数；在场景资源边界从上传记录提取，求值与 Closure 不携带其他模型 |
| `Modules/DebugLambert.slang`、`Modules/DebugMirror.slang`、`Modules/SurfaceLighting.slang` | 三阶段 Surface Programs、泛型 `shadeSurface` 与生产共用的直接光循环；见 [Phase 3](../Documentation/MaterialSystemPhase3.md)、[Phase 5](../Documentation/MaterialSystemPhase5.md) |
| `Modules/SlabClosure.slang` | Single / Dual Slab canonical families、Mix / Layer 原型；见 [Phase 10](../Documentation/MaterialSystemPhase10.md) |
| `Modules/OpenPBR.slang`、`Modules/OpenPBR/` | Adobe OpenPBR 1.1 原生 Slang 移植、静态泛型 LUT/Feature provider；见 [模块说明](Modules/OpenPBR/README.md) |
| `Modules/FiberMaterial.slang`、`Modules/FiberLighting.slang`、`Modules/RTXCRFiber.slang` | 独立 Fiber 三阶段契约、projected lighting、RTXCR Chiang Program；见 [M6](../Documentation/MaterialSystemM6Fiber.md) |
| `Features/Strands/NativeStrands.slang` | 原生曲线有限多层 visibility、Fiber 着色、motion/identity 与显式 overflow；见 [M7](../Documentation/MaterialSystemM7Strands.md) |
| `Modules/RayMaterialExecution.slang`、`Features/PathTracing/RayMaterialQueue.slang` | 按命中点独立的 Surface/Fiber 求值与 prepare、有界 Program 分类；见 [M8](../Documentation/MaterialSystemM8RayExecution.md) |
| `Modules/RTXCRHair.slang`、`Modules/RTXCRHair/` | RTXCR Chiang / Separate Chiang / Far Field 原生 Slang vendored 实现；见 [来源与接口](Modules/RTXCRHair/README.md) |
| `MaterialClosureClassification`（CPU）、`tests/rhi/shaders/ClosureSchedulingProbe.slang` | Program → Closure Family 逻辑调度及 fused/split GPU A/B；生产保持 fused，见 [Phase 11](../Documentation/MaterialSystemPhase11.md) |
| [OpenPBRSurface](Modules/OpenPBRSurface.slang)、[OpenPBRClosure](Modules/OpenPBRClosure.slang) | PT / Deferred 共用的 OpenPBR Material Program、Closure、PreparedClosure；见 [Phase 4](../Documentation/MaterialSystemPhase4.md) |
| `Modules/SimpleBSDF.slang` | vkGltf BSDF 的独立 Material/Closure/PreparedClosure 泛型封装，保留随机 lobe 求值契约；见 [SimpleBSDF 实验](../Documentation/SimpleBSDF.md) |
| `Modules/GPUDriven.slang`、`Modules/GPUDriven/` | GPU 场景、meshlet LOD、剔除、混合光栅化、可见性编码、材质分箱与流式几何属性解码 |
| `Modules/WaveWorkDistribution.slang` | 无资源依赖的 wave 工作分配；`Metallic.GPUDriven` 命名空间 |
| `Modules/ScenePathTypes.slang`、`Modules/ScenePathMath.slang` | 场景路径追踪布局与纯数学：ray cone、RNG、相机和 motion；数学使用 `Metallic.ScenePath` 命名空间 |
| `Modules/StreamTextureSampling.slang` | 显式传入 metadata/feedback buffer 的纹理采样下限和 mip 请求，不隐式读取 pass 参数 |
| `Modules/MaterialValueTextureSampling.slang` | 显式纹理与 feedback 参数的 Value IR 双线性/三线性采样、UV 变换与 LOD；生成代码的薄回调留在 SceneSurface |
| [SceneSurface](Modules/SceneSurface.slang) | 场景材质、纹理、环境与惰性资源 getter；保留 Update/Query 的参数读取策略 |
| [ScenePathTracing](Modules/ScenePathTracing.slang)、[OpenPBRPathTracing](Modules/OpenPBRPathTracing.slang) | 标准/OpenPBR 积分器的 module owner；路径顶点、缓存、guides、continuation 以 `implementing` 归属 |
| [SceneStandardGuides](Modules/SceneStandardGuides.slang)、[SceneShadowQueries](Modules/SceneShadowQueries.slang) | 独立 guides 求值与生产阴影查询，入口和探针显式导入 |
| [VisibilityBufferLighting](Modules/VisibilityBufferLighting.slang)、[OpenPBRDirectLighting](Modules/OpenPBRDirectLighting.slang) | Deferred/实时着色共用的直接光、阴影输入和环境光求值 |
| `Modules/SceneSurfaceMaterial.slang` | 材质诊断摘要和场景材质程序接口；OpenPBR 摘要显式泛型化 context |
| `Modules/StreamRayGeometry.slang` | 流式 ray 命中重建、authored TBN；封装惰性资源读取，保留 SHARC Update/Query 分支 |
| `Features/VisibilityBuffer/VisibilityMaterialBinning.slang` | 8×4 Wave32 稀疏 Program 分箱、prefix allocation 和 indirect dispatch；见 [Phase 6](../Documentation/MaterialSystemPhase6.md) |
| `Features/VirtualShadowMaps/VirtualShadowMapPrototype.slang` | 固定虚拟页的 GPU 工作生成、D32 页裁剪光栅与接收点查询；见 [VSM P0](../Documentation/VirtualShadowMapPrototype.md) |
| `Features/VirtualShadowMaps/VirtualShadowMaps.slang`、`Modules/VirtualShadowMapGeometry.slang` | 接收点按需页、clipmap、完整场景粗级及 resident/StreamAsset caster 解码；[VSM P1](../Documentation/VirtualShadowMapsP1.md) 保留逐帧重画，[VSM P2](../Documentation/VirtualShadowMapsP2.md) 增加持久页复用、GPU tile clear、Sun/Moon 独立缓存和 32 项统计 |
| `Modules/Lighting.slang`、`Modules/Lighting/` | 物理光照、光源选择、光照网格、环境过滤和阴影参数 |
| `Modules/ShaderToHuman.slang`、`Modules/ShaderToHuman/` | 原生 Slang 调试文本、2D/3D 绘制和泛型 Scatter；`Metallic.ShaderDebug` 命名空间 |
| `Modules/ColorGrading.slang`、`Modules/ColorGrading/` | ACES 2.0 / UE Film、全局调色、custom LUT 与三维 LUT 编解码 |
| `Interop/NeuralTextures.slang` | 未来 VT 的独立预留模块，封装 Generic/CoopVec；普通 shader 不导入 |
| `Interop/NRDEncoding.slang` | NRD 前端编码的唯一模块适配入口 |
| `Interop/Denoising/NRD/` | 已适配的 NRD pass、bindings、配置和算法快照；保留程序内 HLSL 宏 |
| `Modules/SHARC.slang`、`Modules/SHARC/` | 原生 Slang HashGrid、SHARC update/query/resolve；见 [模块说明](Modules/SHARC/README.md) |
| `ThirdParty/RadianceCache/` | SHARC 历史 HLSL 适配快照与版权声明，仅作迁移对照，生产代码不再包含 |
| `Features/` | Shader programs：入口、pass 资源和流程相关代码；路径保持兼容现有 C++ 和管线资产 |
| `Licenses/` | 第三方 shader 许可证 |

测试探针放在 `tests/rhi/shaders/`。RTXCR Geometry / Subsurface、RTXTF、NTC 等 SDK 继续使用
`External/` 下的源码。RTXCR Hair GPU 实现使用原生模块，上游快照保留为独立验收基线。
OpenPBR GPU 实现使用原生模块，`External/openpbr-bsdf` 保留为独立验收基线和 CPU LUT 数据来源。
SHARC 使用 `import SHARC;` 和 `using Metallic.SHARC;`，固定当前 renderer 的 64-bit key/atomic 配置，
update/query 宏仅控制调用方 pass，不改变模块类型或函数语义。
ShaderToHuman 使用 `import ShaderToHuman;`，仅保留原生 Slang 移植；
来源记录、许可证和使用说明见 [ShaderToHuman](Modules/ShaderToHuman/README.md)。
外部 vendor 算法和许可证保持原有内容；项目维护的 NRD Slang 包装显式声明 module。

## 使用库

```slang
module ExampleGPUProbe;
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
[SceneSurface](Modules/SceneSurface.slang) 组合路径类型、数学与纹理反馈模块，并保留路径专用
`safeNormalize` 转发。跨模块资源访问使用 getter；SHARC Update/Query 继续使用全局编译排列。
OpenPBR context 由材质 source 显式提供，guide 写出由独立参数控制，tessellation 实现归属
`GPUDrivenStreamAsset` 或 [VisibilityTessellation](Modules/VisibilityTessellation.slang)。
这些边界均不再依赖调用方 `#define` 或 `.slang` 文本注入；第三方 HLSL 配置保持在其 owner 内。
当前固定的 Slang 2026.18.2 下，模块公开结构体的成员仍显式标注 `public`。

新 compute pass 使用 `ComputeKernel` + `ParameterWriter`，以具名 typed 参数承载资源。
只需要参数包的一部分时，可用 `getResourceParameterField<Packet, Field>(byteOffset)`
读取字段或连续小参数组，避免先构造整个 `Packet`。偏移必须采用 scalar wire layout，
并由共享 CPU 声明的 `offsetof` 断言及 mapped/native GPU readback 验证。
旧 root buffer 传输下，此接口仍要求完整的 root 和 `Packet` 长度；短包、非 4 字节对齐或越界字段返回零值，
合法包中的缺失资源 sentinel 原样保留。它不改变参数编码或 descriptor 解析方式。
普通、未拆分的 compute 路径追踪正式默认启用 `inlineComputeRoot`，使用独立的 24 字节 push ABI；
显式设为 `false` 可恢复旧传输用于 A/B。Realtime、visibility deferred、ray pipeline 和 primary-guide split
路径的属性缺省为 `false`，显式设为 `true` 会被拒绝。通用 `Core` 的宏缺省仍为 `0`。
旧名 `experimentalInlineComputeRoot` 仅作为已保存配置的兼容别名，新名的显式布尔值优先。
通过全局编译定义 `METALLIC_INLINE_COMPUTE_ROOT=1` 选择 `InlineComputeRoot`；CPU 同时选择
`ResourceComputeKernelDesc.rootTransport=InlinePush`。资源包长度和字段检查保留，
仅省去外层 root buffer 及其读取。该模式不能同时导入 `ParameterRoot`。
具体范围和验证方法见 [Inline compute root](../Documentation/InlineComputeRoot.md)。
路径追踪使用固定 Scene 类型的 `getSceneResourceParameters()` /
`getSceneResourceParameterField<TField>()`。支持的 inline compute 路径默认由 host
通过 `trustedSceneParameters` 保证完整包，并以编译期大小契约去掉这两个入口的包长度检查；
其他路径仍委托通用 checked getter。通用接口的短包返回零契约不变，字段合法性检查也保留。
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
的共享 inline 参数。SHARC 模块需要 StructuredBuffer 对象进行原子操作，
因此这三个缓存 buffer 使用具名 descriptor handle；维护阶段直接读取 settings，不再依赖 cacheParams 的公共前缀。
主追踪及其 OpenPBR、VisibilityBuffer 使用共享的 `SceneResourceParameters` 具名 DR 字段。

VisibilityBuffer Deferred 是纯光栅表面的实时 resolve，不包含 ray-query 积分器。
材质的语义 Feature、作者策略和编译选择由 CPU [Feature System](../Documentation/MaterialSystemPhase7.md)
统一分析；AlphaMode / doubleSided 使用独立签名，不能直接当作 lighting shader keyword。
[Coverage Program](../Documentation/MaterialSystemPhase8.md) 从 Value 源码提取独立的覆盖表达式，
在 VBuffer 写入深度/可见性之前及 RT/shadow 候选命中处求值，和 Surface 共用不可变材质参数快照。
[Value IR](../Documentation/MaterialSystemPhase9.md) 统一 Surface/Coverage 的验证、优化、切片与稳定身份；
TextureSample 显式选择 LOD、梯度或 RayCone，不能使用依赖 quad 的隐式导数。
[SceneSurface](Modules/SceneSurface.slang) 和 [OpenPBRSurface](Modules/OpenPBRSurface.slang)
提供共享表面求值，[VisibilityBufferLighting](Modules/VisibilityBufferLighting.slang)
负责 ClusterLightGrid、显式阴影输入和 IBL。
有界 IBL 权重默认选择半精度策略，几何与 HDR 累加保持 FP32；`halfPrecision: false`
提供相同算法的 FP32 对照。内置 ACEScg 的共享 Deferred 模块使用 float 与 pack/unpack
保留舍入行为，由 native Bool specialization 选择精度；静态回退仍使用原有精度宏。
这项精度选择重构的 GPU 与完整冷启动验收尚未完成。详见
[VisibilityBufferDeferred](../Documentation/VisibilityBufferDeferred.md)。

`Core` 的生产用法是 `getResourceParameters<Params>()` 加显式 resolver：buffer 使用
`resolveBuffer<StructuredBuffer<T>>(resources.indices)`，图像和 sampler 使用 `resolveUniform`，
纹理数组使用 `resolveNonUniform(ResourceHandle<Texture2D<float4>>(resources.materialTextures.load(index)))`。
[NamedResourceParameters.h](../Source/Runtime/Render/Core/NamedResourceParameters.h) 共用 C++/Slang 字段；
静态 pass 优先使用 `ParameterWriter` 直接构造具名参数。动态资源清单通过
`METALLIC_RESOURCE_MEMBER(Params, member)` / `METALLIC_DATA_MEMBER(Params, member)`
引用字段，并用 `resourceParameterSize = sizeof(Params)` 声明参数块大小。
[ResourceMember.h](../Source/Runtime/Render/Core/ResourceMember.h) 从 C++ 成员类型和 `offsetof`
推导资源类别、字段格式及偏移；不再维护数字 ID 到字段的独立布局表。
编码器仍检查字段范围、对齐、重叠、资源类型和数组形式，保留可选资源与批量 dispatch 的契约。
生产 GPU 参数块没有逻辑 slot table；场景结构为 384 字节，buffer/image/sampler 字段为 4 字节，
数组和原始数据字段为 12 字节 span，AS 独立保留完整地址。常量仍通过 `getConstants<T>()` 读取。
`ComputeResourceMember` 不接受整数 ID；测试与生产均使用成员引用，旧数字 ID 和显式字段表兼容路径已删除。
`Core` 导入 `ParameterRoot`，因此不能和另一份 inline push 声明混用。RHI 不在用户 push 数据前插入 heap header。

Lighting 的算法显式接收 `StructuredBuffer<GPUPunctualLight>` 或 `PunctualSamplingResources`；
库内不再固定光源、ReGIR、PDF 的槽位。顶点位置读取同样显式接收 buffer；
光追顶点属性统一从 CLAS 对应的流送页解码，不再提供 position-fetch/fallback 分支。着色法线、几何法线和 TBN 的求值顺序保持原有语义。

## 项目自有 Slang 实现

`AlphaCoverage`、`MaterialTextureDecode` 和 `RTXCRMaterialAdapter` 使用独立 Slang
module，通过 `import` 显式共享。`SceneSurfaceMaterial` 的 OpenPBR 摘要以 context 泛型
消除程序局部 closure alias 依赖；`StreamRayGeometry` 使用共享 `ScenePathTypes`，
其资源宏不再泄漏到调用方。三角形 TBN 计算由 `ScenePathMath` 统一提供。
生成的 `MaterialValueDispatch.slang` 声明 `implementing SceneSurface;`，由 SceneSurface
以 `__include` 纳入同模块；生成器的 owner 声明受静态检查约束。纹理 adapter 先处理缺失索引，
再调用 `MaterialValueTextureSampling`，保留手动过滤、LOD 和颜色空间策略。
路径循环、阴影、guides 和 continuation 必须由命名 module 统一拥有；同模块实现单元
使用 `implementing`，入口只导入明确依赖，不再依赖调用点注入代码或宏。
[OpenPBRPathVertex](Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang) 提供显式状态参数和
控制结果的顶点处理函数，通用路径循环与单样本
[continuation](Modules/OpenPBRPathTracing/OpenPBRPathContinuation.slang) 共同调用；两者均声明
`implementing OpenPBRPathTracing;`，不再作为无归属片段注入调用点。
`MeshletStreamBLAS.h` 是 C++/Slang 共享常量契约，使用普通头文件。
第三方 SDK 和 NRD 快照保留其原始 HLSL 适配边界及许可证。

材质 `TextureInfo` 为 48 字节，`SceneMaterialData` 为 720 字节。两个索引后只保留
8 字节对齐 padding，使 StructuredBuffer 的两个 float4 位于偏移 16/32；不保留 NTC 字段。
Slang `sizeof` 与 SPIR-V StructuredBuffer 布局不能混为一谈；GPU 字段/数组步长探针验证两端契约。

`SceneMaterialData` 仅定义现有场景上传布局。OpenPBR、SingleSlab、DualSlab 和 Fiber
使用独立泛型 Program 与具体 Closure；模型选择在命中点或 pass 边界进行，公共光照循环
通过静态类型调用。Slab 的 Value 表达式只读取 `MaterialValueParameters`，不构造 OpenPBR
输入或 Closure。诊断、guides 和近似环境光读取 `SceneSurfaceSummary` 的数值摘要。
标准与 OpenPBR 路径的 Fiber 求值和 RR guide 在模型分支后单独读取 Fiber 参数，
共用颜色、粗糙度和 Fresnel guide 近似，不读取 Surface 纹理或发光参数。
`LegacyMaterialProgram`、`LegacyCoverageEvaluator`、混合 `SceneSurfaceClosure` 与
空的 weighted-closure 标记接口已移除，不提供旧类型别名。

## SDK 兼容边界

- 固定功能经 adapter module 导入。`NRDEncoding` 固定前端编码配置。
  `NeuralTextures` 仅为未来 VT 预留，使用独立 `NeuralTextureResourceParameters`；
  普通 shader 不导入、不携带 NTC 编译排列。
- `#define` 不跨 `import` 或 `__include` 传播。消费者局部定义不能改变已导入模块；
  SDK 全局排列使用 `SlangShaderDesc::macroDefines`，它们也参与缓存键。
- 同一程序中，每个 vendor header 有一个 canonical owner。其他模块导入 owner，
  不重复包含 vendor header；include guard 不能跨 module 去重。
- NRD pass 等包装自身也是 module；SDK 的 `.hlsli` 配置可以在包装内使用
  `#define` + `#include`。OpenPBR context 和自有 feature 的跨文件特化使用显式参数、
  泛型或接口，不能再通过包含 `.slang` 注入调用方上下文。
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
