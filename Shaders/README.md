# Shader 目录

Shader 按公共库与渲染功能组织。`Shaders/` 是运行时唯一的项目 Shader 搜索根目录；
第三方 SDK 的额外搜索路径仍由相应功能提供。

| 目录 | 内容 |
| --- | --- |
| `Libraries/Math/` | 球谐类型、投影、旋转与辐照度运算 |
| `Libraries/Lighting/` | 物理光照单位、光源采样、程序化环境、光照网格布局与查询 |
| `Libraries/GPUDriven/` | GPU 场景数据、光栅与剔除公共代码 |
| `Libraries/Texturing/` | 神经纹理采样 |
| `Libraries/Denoising/` | NRD Shader 接口与配套配置 |
| `Libraries/RadianceCache/` | SHARC、NRC 接口与实现头文件 |
| `Features/Environment/` | HDRI 与球谐环境光预计算 |
| `Features/Lighting/` | 实时 OpenPBR 光照、光照网格构建、光源 PDF 与 ReGIR 构建 |
| `Features/PathTracing/` | 路径追踪、OpenPBR 路径追踪、降噪引导数据、SHARC 维护 |
| `Features/ReSTIR/` | RTXDI 重采样、置信度与合成 |
| `Features/GPUDriven/` | GPU 剔除、meshlet 流式渲染 |
| `Features/VisibilityBuffer/` | Visibility Buffer 生成、延迟着色与合成 |
| `Features/PostProcess/` | 自动曝光、色调映射、最终输出与 DLSS 辅助处理 |
| `Features/Debug/` | 材质/射线/光照网格可视化、SliderDebug、GPU Probe、[ShaderToHuman](Features/Debug/ShaderToHuman/README.md) |
| `Features/Samples/` | 三角形、线框、材质 Shader Object、图像与 RTXCR 示例 |
| `Features/SmokeTests/` | 运行时/RHI 共用的 bindless 与 RenderGraph 冒烟 Shader |
| `Licenses/` | Shader 库的第三方许可证 |

测试专用探针继续放在 `tests/rhi/shaders/`，通过相对路径引用这里的库。
`External/` 下的 OpenPBR、RTXCR、NTC 等依赖保持各自的目录。
ShaderToHuman 的固定版本核心头文件随调试功能保存在 `Features/Debug/ShaderToHuman/`，
该目录保留上游文件名、LICENSE 和 NOTICE。

## 模块加载与依赖

`SlangShaderDesc::moduleName` 使用相对搜索根目录的路径，省略 `.slang`：

```cpp
render::SlangShaderDesc{
    .moduleName = "Features/Environment/EnvironmentLightingPrecompute",
    .entryPointName = "environmentLightingPrecomputeMain",
    .searchPath = PROJECT_SOURCE_DIR "/Shaders",
};
```

[Slang 编译 API](https://shader-slang.org/docs/compilation-api/) 在配置的搜索路径中加载模块。
不要为每个功能添加搜索目录或依赖递归搜索；完整模块路径可以区分不同目录中的同名文件。

`#include` 使用相对当前文件的显式路径。例如 `Features/Environment/` 中使用：

```hlsl
#include "../../Libraries/Math/SphericalHarmonics.slang"
#include "../../Libraries/Lighting/ProceduralEnvironment.slang"
```

- `Libraries/` 只放可复用的类型、算法与接口，不依赖 `Features/`。
- `Features/` 放入口和依赖特定渲染流程的辅助代码；仅无入口并不代表属于公共库。
  例如 `Features/Lighting/OpenPBRDirectLighting.slang` 复用路径追踪的材质求值和射线查询，
  因此保留为功能代码，由实时与 VBuffer 路径共享。
- 当前依赖使用 `#include` 传递功能宏。目录整理不改变为 `import`，避免改变宏的作用域。
- 新增文件使用 PascalCase；同一功能的入口与专用辅助文件放在一起。
- 移动模块时同步更新运行时与测试的模块路径、跨目录 include 和文档引用。

SPIR-V 缓存键包含模块路径，依赖快照包含完整文件路径。库文件发生变化时，已有的缓存失效
和热重载机制会跟踪该依赖。`slang_shader_disk_cache_and_source_invalidation` 测试覆盖
跨 `Features/` 与 `Libraries/` 的依赖、缓存命中、库文件编辑和热重载。
