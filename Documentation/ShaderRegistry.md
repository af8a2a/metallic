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

`getComputeKernel` 把源码请求、PSO 获取与成功发布合成一次调用；源码或创建失败时保留调用者上一代有效程序。`getComputeProgram` 已移除。需要编译产物/依赖清单的材质及其他多阶段适配器使用 `getShader`，随后由 `getShaderModule`、`getComputePipeline`、`getGraphicsPipeline` 或 `ComputeKernel::initialize` 取得执行对象。动态资源清单在 Registry 外通过 `initializeResourceKernel` 成对发布 Kernel 和 `ComputeResourceEncoder`；编码器不持有 executable。所有路径最终使用同一缓存策略。已有 GPU packet 的保留与 Device 生存期契约不变。

Registry 是进程单例，但全局仅保存源码字节标识与缓存分组。创建 module 时关联 RHI descriptor-heap/opacity-micromap 转换前后的字节标识，保持实际 GPU shader 的来源分组。GPU cache 通过 `Device::sharedState` 按设备创建一次，在 native Device 销毁前释放；不同 Device 不共享 native handles。每个返回的 PSO 是调用者拥有的 RHI 对象，持久缓存向驱动提供预编译数据；这是 `.pso` 复用，不是按日志 hash 直接返回任意执行对象。源码每次经过现有 Slang 依赖检查，不能从失效请求拿到旧字节码。

缓存放在 `.cache/pso/`。PT、SHaRC/NRC、实时光照、Deferred、VisibilityBuffer 与 GPUDrivenStreamAsset 保留既有文件名；其他 module 自动使用 `ShaderRegistry-<module basename>-<full module hash>.pso`，无来源元数据的原始 SPIR-V 使用 `ShaderRegistry.pso`。同一 module 的入口和变体共享容器，shader 字节、入口、完整 RHI 管线状态及 backend/device 兼容性决定实际 PSO 身份。分组只决定存储文件，不决定执行对象是否相同。

成功创建的新 PSO 交给设备持有的后台 worker 保存；同一 group 的请求合并，在最后一次新状态请求后 750 ms 保存，持续新增请求最多等待 5 s（不含已有保存任务的执行时间）。命中不延长等待；失败候选不记录为成功。正常 Device 销毁先排空并 join worker，再销毁 native cache；需要立即确保持久化的工具/测试可调用 `flushPipelineCaches(device)`，普通管线获取无需调用。保存失败会记录 warning 并保留待保存状态，后台延迟重试，当前管线仍可执行；退出/显式 flush 的失败尝试有界，避免永久等待。显式 `desc.pipelineCache` 沿用调用者的保存契约，`PipelineCache::save()` 仍同步；默认 null 必须走 Registry 持久缓存。

后台提取与文件处理不持有 PSO 创建/统计共用的元数据锁，独立保存锁防止旧文件覆盖较新保存。提取前捕获 hash 与 revision，成功后只提交该快照，保存期间新增的 PSO 留待下一次保存。默认 Vulkan cache 未设置 externally-synchronized 标志，native cache 访问由驱动内部同步，仍可能存在驱动锁竞争，依据 [Vulkan pipeline cache synchronization](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-cache)。`Saved pipeline cache` 在操作完成后打印，并记录 `extractMs`（驱动提取及输出缓冲分配）、`writeMs`（容器排序、校验和、写盘与原子替换）、`totalMs`；这些不是 shader 编译、PSO 创建或启动总耗时。`PipelineCacheStats` 的 revision、保存次数和阶段时间可验证合并及持久化状态。

Shader Objects 通过 `getGraphicsShaderObjectProgram` 自动使用 `.cache/shader-objects/<group>/<shaderBinaryUUID>/<programHash>.shaderbin`。首次从 SPIR-V 创建成功后，使用 `vkGetShaderBinaryDataEXT` 导出 linked vertex/fragment 的完整二进制对并原子保存；后续进程用 `VK_SHADER_CODE_TYPE_BINARY_EXT` 成对创建。它们使用独立容器，不使用 `VkPipelineCache`。第三方 SDK 内部 shader 不由 Registry 管理。

缓存标识包含两阶段最终设备 SPIR-V、入口、stage/nextStage、link/descriptor-heap/indirect flags、user push ABI、实际 heap mapping 和 RHI 创建契约版本。动态 raster/color state 不属于 shader object 编译输入。文件验证格式、阶段长度、载荷校验和、programHash，以及 UUID 相同且当前 binaryVersion 不低于保存版本的兼容关系。损坏/不兼容文件或驱动拒绝 binary 时清理部分 handles、回退 SPIR-V 并重新导出；导出/保存失败只记录 warning，当前 shader 继续执行。导入和导出缓冲区均显式保证 16 字节对齐。规则依据 [Vulkan binary compatibility](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-binary-compatibility) 和 [vkGetShaderBinaryDataEXT](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderBinaryDataEXT.html)。

`GraphicsShaderObjectProgram::cacheStats().binaryCacheHit` 只在 Vulkan 成功完成 BINARY 创建后为 true；`creationTimeNanoseconds` 仅统计 `vkCreateShadersEXT`，不包含 Slang、磁盘读取/保存和 binary 导出。`persisted` 表示已成功保存或使用已有有效文件。默认 Registry 自动持久化；显式 `binaryCacheDirectory` 可隔离缓存，空字符串关闭持久化；直接 RHI 调用默认不写磁盘。

编辑器显示使用 ImGui 的独立 descriptor/vertex ABI，经 `getShaderModule` 与 `getExternalGraphicsPipeline` 接入相同缓存。native 适配器提供完整稳定状态标识，调用 Vulkan 后端的 `createCachedGraphicsPipeline`，由后端锁住创建过程并仅在成功后记录 PSO。ImGui 的固定布局/混合/光栅契约变化时必须更新适配器的 ABI 版本；原始 native handles 由适配器拥有并在 Device 销毁前释放。

`MetallicShaderWarmupCore` 只链接 Registry 的编译部分，不依赖 Vulkan 设备；手动预热 target 的可选属性不变。GPU 部分属于 `MetallicRuntimeRender`。

验证入口：

- `MetallicShaderRegistryUsageAudit`：禁止 `Source/` 与 `Tools/` 中的 compiler/RHI/native Vulkan 获取旁路，Slang/RHI 实现和 Registry 是明确例外；底层 RHI 测试仍可直接验证后端。
- `shader_registry_source_pso_and_device_lifetime`：源码冷/热缓存、shader 与管线状态变体、失败发布、并发创建、不同 Device 的持久复用。
- `MetallicDeferredShaderCacheWriterTests`：合并、后台阻塞隔离、保存期间的新请求、失败重试和退出排空。
- `pipeline_cache_deferred_save_and_snapshot`：实际 Vulkan 缓存保存的同步/后台计时、并发新增与全量重新加载。
- `MetallicShaderObjectCacheFileTests`：文件兼容性、损坏、尺寸边界、原子替换与并发保存。
- `shader_object_binary_persistence_readback`：SPIR-V 导出、BINARY 命中、片元变体、损坏回退和第二 Device 的像素一致性。
- `MetallicShaderObjectBinaryProcessSmoke`：关闭驱动内部缓存，独立进程检查 binary 加载与精确像素读回。
- `MetallicShaderObjectBinaryRenderingSmoke`：检查 Vulkan 校验日志、Material/Wireframe 的两个 Device 生命周期及逐帧像素一致性。
- `shader_object_material_readback`：通过 BINARY 创建的两套 bindless 材质 shader 实际绘制与读回。
- `hdr_editor_imgui_composite`：编辑器显示管线的 native 缓存复用与 scRGB/PQ 像素读回。
- `MetallicLookDevPathTracePipelineCacheSmoke`：关闭驱动内部缓存，用两个实际 LookDev 进程检查 OpenPBR PT、SHaRC 与 NRC 的 Registry 缓存覆盖。

`[ShaderRegistry] PSO cache group=... hits=... misses=...` 是容器 hash-table 统计，不是 Vulkan native creation feedback；性能结论仍需同负载实际计时。
