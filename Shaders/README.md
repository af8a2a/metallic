# Shader 模块

Metallic 的可复用 shader 库使用 Slang module。子系统之间用 `import`，同一模块的实现
文件用 `__include`，第三方 HLSL 的宏配置与文本包含留在 Interop 或程序内部。

| 目录 | 职责 |
| --- | --- |
| `Modules/Core.slang`、`Modules/Core/` | 统一 compute 资源 ABI、泛型资源访问、相机、顶点解码、SH、显示颜色 |
| `Modules/Material.slang`、`Modules/Material/` | CPU/GPU 共用的材质与纹理数据布局 |
| `Modules/GPUDriven.slang`、`Modules/GPUDriven/` | GPU 场景、meshlet LOD、剔除、混合光栅化、可见性编码和材质分箱 |
| `Modules/Lighting.slang`、`Modules/Lighting/` | 物理光照、光源选择、光照网格、环境过滤和阴影参数 |
| `Interop/NeuralTextures.slang` | NTC 的唯一模块适配入口，封装 Generic/CoopVec 和无 NTC 的回退 |
| `Interop/NrdEncoding.slang` | NRD 前端编码的唯一模块适配入口 |
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

Core 是 `gComputeResources` 的唯一声明者，多个导入路径共用同一份 push constant。
它维持现有 RHI 的两个 heap 索引、资源表指针和常量指针布局，不引入新的描述符绑定。
`getResource<T>(slot)`、`getResourceArray<T>(slot, index)`、`getConstants<T>()`
取代原来的 `METALLIC_RESOURCE`、`METALLIC_RESOURCE_ARRAY`、`METALLIC_CONSTANTS` 宏。
数组访问继续通过 `nonuniform` 选择 descriptor。pass 可用本地别名描述槽位，但库不依赖消费者的宏。

Lighting 的算法显式接收 `StructuredBuffer<GpuPunctualLight>` 或 `PunctualSamplingResources`；
库内不再固定光源、ReGIR、PDF 的槽位。顶点位置读取同样显式接收 buffer；
是否使用硬件 position fetch 由调用程序决定。着色法线、几何法线和 TBN 的求值顺序保持原有语义。

## SDK 兼容边界

- 固定功能经 adapter module 导入。`NrdEncoding` 固定前端编码配置，`NeuralTextures`
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
