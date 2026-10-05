# ShaderRegistry

运行时与预热通过 `ShaderRegistry::instance()` 获取 shader。Registry 负责未编译/失效源码的 Slang 编译、已有 SPIR-V 的依赖验证与读取，以及 compute/raster PSO 的持久缓存加载和保存。新增 Pass 不需要创建、传递或保存 `PipelineCache`。

```cpp
#include "Runtime/Render/Core/ShaderRegistry.h"

auto result = ShaderRegistry::instance().getComputeKernel(device,
    {.moduleName = "Features/PostProcess/AutoExposure",
        .entryPointName = "autoExposureHistogramMain",
        .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
    {.parameters = parameterAbi<AutoExposureParams>(kAutoExposureABI, ParameterTransport::InlinePush)},
    kernel, log);
```

`getComputeKernel` / `getComputeProgram` 把源码请求、PSO 获取与成功发布合成一次调用；源码或创建失败时保留调用者上一代有效程序。需要编译产物/依赖清单的材质及其他多阶段适配器使用 `getShader`，随后由 `getShaderModule`、`getComputePipeline`、`getGraphicsPipeline` 或 `ComputeKernel::initialize` 取得执行对象。所有路径最终使用同一缓存策略。已有 GPU packet 的保留与 Device 生存期契约不变。

Registry 是进程单例，但全局仅保存源码字节标识与缓存分组。创建 module 时关联 RHI descriptor-heap/opacity-micromap 转换前后的字节标识，保持实际 GPU shader 的来源分组。GPU cache 通过 `Device::sharedState` 按设备创建一次，在 native Device 销毁前释放；不同 Device 不共享 native handles。每个返回的 PSO 是调用者拥有的 RHI 对象，持久缓存向驱动提供预编译数据；这是 `.pso` 复用，不是按日志 hash 直接返回任意执行对象。源码每次经过现有 Slang 依赖检查，不能从失效请求拿到旧字节码。

缓存放在 `.cache/pso/`。PT、SHaRC/NRC、实时光照、Deferred、VisibilityBuffer 与 GPUDrivenStreamAsset 保留既有文件名；其他 module 自动使用 `ShaderRegistry-<module basename>-<full module hash>.pso`，无来源元数据的原始 SPIR-V 使用 `ShaderRegistry.pso`。同一 module 的入口和变体共享容器，shader 字节、入口、完整 RHI 管线状态及 backend/device 兼容性决定实际 PSO 身份。分组只决定存储文件，不决定执行对象是否相同。

成功创建的新 PSO 立即保存，失败候选不记录为成功；命中时底层 `save()` 不做重复文件写入。缓存保存失败会记录 warning，但有效的当前管线仍可执行。显式 `desc.pipelineCache` 仅供需要独立控制缓存的底层调用者/测试使用，沿用调用者的保存契约；默认 null 必须走 Registry 持久缓存。

Shader Objects 通过 `getGraphicsShaderObjectProgram` 使用统一入口，但当前 RHI 没有 Shader Object 二进制持久化输入，不能宣称它们命中 `.pso`；其 SPIR-V 仍经过统一缓存。第三方 SDK 内部 shader 不由 Registry 管理。

编辑器显示使用 ImGui 的独立 descriptor/vertex ABI，经 `getShaderModule` 与 `getExternalGraphicsPipeline` 接入相同缓存。native 适配器提供完整稳定状态标识，调用 Vulkan 后端的 `createCachedGraphicsPipeline`，由后端锁住创建过程并仅在成功后记录 PSO。ImGui 的固定布局/混合/光栅契约变化时必须更新适配器的 ABI 版本；原始 native handles 由适配器拥有并在 Device 销毁前释放。

`MetallicShaderWarmupCore` 只链接 Registry 的编译部分，不依赖 Vulkan 设备；手动预热 target 的可选属性不变。GPU 部分属于 `MetallicRuntimeRender`。

验证入口：

- `MetallicShaderRegistryUsageAudit`：禁止 `Source/` 与 `Tools/` 中的 compiler/RHI/native Vulkan 获取旁路，Slang/RHI 实现和 Registry 是明确例外；底层 RHI 测试仍可直接验证后端。
- `shader_registry_source_pso_and_device_lifetime`：源码冷/热缓存、shader 与管线状态变体、失败发布、并发创建、不同 Device 的持久复用。
- `hdr_editor_imgui_composite`：编辑器显示管线的 native 缓存复用与 scRGB/PQ 像素读回。
- `MetallicLookDevPathTracePipelineCacheSmoke`：关闭驱动内部缓存，用两个实际 LookDev 进程检查 OpenPBR PT、SHaRC 与 NRC 的 Registry 缓存覆盖。

`[ShaderRegistry] PSO cache group=... hits=... misses=...` 是容器 hash-table 统计，不是 Vulkan native creation feedback；性能结论仍需同负载实际计时。
