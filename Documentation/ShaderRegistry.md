# ShaderRegistry

运行时与预热通过 `ShaderRegistry::instance()` 获取 shader。Registry 负责未编译/失效源码的 Slang 编译、已有 SPIR-V 的依赖验证与读取、Device 内 compute/raster 原生执行对象复用，以及 PSO 的持久缓存加载和保存。新增 Pass 不需要创建、传递或保存 `PipelineCache`。

Native descriptor heap 的编译缓存包含 resource/sampler stride；运行时使用所选 GPU 的
heap ABI，预热通过物理设备属性查询取得同一配置。模块创建会拒绝旧的尺寸表达式或
与设备不匹配的 literal stride，并明确要求重新编译。`TO-REMOVE(VVL payload-size)`
标记这项临时编译策略，移除条件及相关代码见
[Native descriptor heap stride workaround](NativeDescriptorHeapStrideWorkaround.md)。

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

Registry 是进程单例，但全局仅保存源码字节标识与缓存分组。创建 module 时关联 RHI descriptor-heap/opacity-micromap 转换前后的字节标识，保持实际 GPU shader 的来源分组。GPU cache 通过 `Device::sharedState` 按设备创建一次，在 native Device 销毁前释放；不同 Device 不共享 native handles。每次获取仍返回调用者拥有的 `unique_ptr<ComputePipeline>` / `unique_ptr<GraphicsPipeline>` 包装层，相同请求的包装层与 `PreparedExecution` 通过 `shared_ptr` 共享底层 Pipeline、Layout 和执行状态。源码每次经过现有 Slang 依赖检查，不能从失效请求拿到旧字节码。

默认 `getComputePipeline` / `getGraphicsPipeline` 先应用与 RHI 创建相同的 descriptor 验证，再查询 Device 内的 Native Object Cache：已有对象直接复用；同一 key 正在创建时等待该次创建；新 key 由一个调用者进入 RHI。成功对象发布给全部等待者，失败或异常清除 in-flight 状态并唤醒等待者，后续请求可以重新尝试。对象缓存锁不跨 Driver 创建、future 等待或原生对象销毁，因此不同 key 可以并发创建。显式 `desc.pipelineCache` 和直接 `Device::create*Pipeline` 始终绕过对象缓存，保留底层测试、独立 Benchmark 与指定 `VkPipelineCache` 填充的创建行为；外部 Graphics 适配器与 Shader Objects 保持各自已有路径。

对象 key 复用 `PipelineStateHash.cpp`：包含最终设备 SPIR-V 的 `contentHash`、实际使用阶段与 Entry Point、完整 RHI Pipeline State、Pipeline 类型和隐式 RHI 创建契约版本。源码名、日志名与磁盘分组不决定对象身份。热更新产生不同设备字节码或状态时取得新对象；旧包装层和已准备的 GPU 执行快照继续持有上一代对象。即使字节码相同，另一个 Device 的 module、无效入口或状态也不能绕过 descriptor 验证。

Native Object Cache 的 ready 索引使用弱引用，并以 compute/graphics 共用的 Strong LRU 保留最多 128 个最近使用对象。`setNativePipelineCacheCapacity(device, capacity)` 可调整容量；容量 0 只关闭强保留，仍保留弱引用复用和 single-flight。LRU 淘汰只释放缓存自己的引用，已有包装层和 `PreparedExecution` 继续有效；miss 和容量调整会清理过期弱引用。Device 销毁时先排空后台保存 worker，再释放设备持有的对象和持久缓存，最后销毁 native Device。调用者仍需在 Device 销毁前释放自己的 RHI 对象与执行快照，并在创建期间保持 Device 和输入 module 存活。

`nativePipelineCacheStats(device)` 返回独立的 `NativePipelineCacheStats`：`lookupCount`、`readyHitCount`、`singleFlightWaitCount`、`nativeCreateCount`、`failedCreateCount`、`evictionCount` 是累计计数；`nativeCreateCount` 统计被选出的创建 factory 调用次数，包含失败，不计查询前的验证失败或显式缓存调用。`readyEntryCount`、`residentCount`、`inFlightCount` 与 `strongCapacity` 表示当前索引、强保留、待创建和容量状态；ready 索引数量可能包含尚未清理的过期弱引用。对象缓存位于 Driver Binary/PSO Cache 之前，只有实际创建者更新 `.pso` 缓存并请求保存；ready 命中和等待者不会增加 `PipelineCacheStats` 的命中/未命中、反馈计数或 revision。两层统计分别反映执行对象复用与原生创建/持久化。

缓存默认放在 `.cache/pso/`；诊断时可设置 `METALLIC_PSO_CACHE_DIRECTORY` 使用独立目录，隔离首次 native 创建和跨进程重用，同时保留现有 SPIR-V 缓存。PT、SHaRC、实时光照、Deferred、VisibilityBuffer 与 GPUDrivenStreamAsset 保留既有文件名；其他 module 自动使用 `ShaderRegistry-<module basename>-<full module hash>.pso`，无来源元数据的原始 SPIR-V 使用 `ShaderRegistry.pso`。同一 module 的入口和变体共享容器，shader 字节、入口、完整 RHI 管线状态及 backend/device 兼容性决定实际 PSO 身份。分组只决定存储文件，不决定执行对象是否相同。

成功创建的新 PSO 或有效 native feedback 报告未命中的已有 PSO，交给设备持有的后台 worker 保存。这样应用 hash 命中后驱动补充的 backend data 仍会持久化。反馈无效时，每个 PSO 在本进程首次成功创建后保存一次；反馈无效的重复创建不重复标记。有效 native 命中不延长等待；失败候选不记录为成功。同一 group 的请求合并，在最后一次待保存状态请求后 750 ms 保存，持续新增请求最多等待 5 s（不含已有保存任务的执行时间）。正常 Device 销毁先排空并 join worker，再销毁 native cache；需要立即确保持久化的工具/测试可调用 `flushPipelineCaches(device)`，普通管线获取无需调用。保存失败会记录 warning 并保留待保存状态，后台延迟重试，当前管线仍可执行；退出/显式 flush 的失败尝试有界，避免永久等待。显式 `desc.pipelineCache` 沿用调用者的保存契约，`PipelineCache::save()` 仍同步；默认 null 必须走 Registry 持久缓存。

元数据锁只保护 hash 集合、统计和 revision；Graphics、Compute 及外部 Graphics 创建均在锁外调用驱动。创建前的 hash 查询与创建成功后的登记各使用短临界区；同一 hash 并发创建时，元数据命中按成功登记顺序计数。默认 Vulkan cache 未设置 externally-synchronized 标志，native cache 访问由驱动内部同步，仍可能存在驱动锁竞争，依据 [Vulkan pipeline cache synchronization](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-cache)。

后台提取与文件处理也不持有元数据锁，独立保存锁防止旧文件覆盖较新保存。提取前捕获 hash 与 revision，成功后只提交该快照，保存期间新增的 PSO 留待下一次保存。`Saved pipeline cache` 在操作完成后打印，并记录 `extractMs`（驱动提取及输出缓冲分配）、`writeMs`（容器排序、校验和、写盘与原子替换）、`totalMs`；这些不是 shader 编译、PSO 创建或启动总耗时。`PipelineCacheStats` 的 revision、保存次数和阶段时间可验证合并及持久化状态。

`GraphicsPipeline::creationStats()` 和 `ComputePipeline::creationStats()` 返回该包装层本次获取的不可变反馈。实际创建者获得 native/driver 时间、metadata 锁等待及 driver 命中状态；ready 命中和 single-flight 等待者的 `nativeObjectReused` 为 true，native/driver/metadata-lock 时间为 0，`feedbackValid` 为 false，`singleFlightWaitTimeNanoseconds` 单独记录等待者等待已有创建的时间。复用不会复制首次创建的时长冒充本次 Driver 工作。`pipelineCacheHit()` 仍表示底层对象首次原生创建时的 PSO hash-table 结果，对象复用应检查 `nativeObjectReused` 或独立对象缓存统计。

并发测量必须使用逐调用数据，不能用共享 `PipelineCacheStats.last*` 推断某次调用。`cacheLockWaitMs` 是创建前查询和创建后登记两次持久缓存 metadata 锁获取的等待总和；不包括对象缓存 single-flight 等待或驱动内部同步。使用 [ShaderPipelineBench](ShaderPipelineBench.md) 的 `--pipeline-workers 1` 与多 worker 批次，在同一请求集及缓存条件下比较；Bench 直接进入 RHI，保留真实原生创建测量。

Shader Objects 通过 `getGraphicsShaderObjectProgram` 自动使用 `.cache/shader-objects/<group>/<shaderBinaryUUID>/<programHash>.shaderbin`。首次从 SPIR-V 创建成功后，使用 `vkGetShaderBinaryDataEXT` 导出 linked vertex/fragment 的完整二进制对并原子保存；后续进程用 `VK_SHADER_CODE_TYPE_BINARY_EXT` 成对创建。它们使用独立容器，不使用 `VkPipelineCache`。第三方 SDK 内部 shader 不由 Registry 管理。

缓存标识包含两阶段最终设备 SPIR-V、入口、stage/nextStage、link/descriptor-heap/indirect flags、user push ABI、实际 heap mapping 和 RHI 创建契约版本。动态 raster/color state 不属于 shader object 编译输入。文件验证格式、阶段长度、载荷校验和、programHash，以及 UUID 相同且当前 binaryVersion 不低于保存版本的兼容关系。损坏/不兼容文件或驱动拒绝 binary 时清理部分 handles、回退 SPIR-V 并重新导出；导出/保存失败只记录 warning，当前 shader 继续执行。导入和导出缓冲区均显式保证 16 字节对齐。规则依据 [Vulkan binary compatibility](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-binary-compatibility) 和 [vkGetShaderBinaryDataEXT](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetShaderBinaryDataEXT.html)。

`GraphicsShaderObjectProgram::cacheStats().binaryCacheHit` 只在 Vulkan 成功完成 BINARY 创建后为 true；`creationTimeNanoseconds` 仅统计 `vkCreateShadersEXT`，不包含 Slang、磁盘读取/保存和 binary 导出。`persisted` 表示已成功保存或使用已有有效文件。默认 Registry 自动持久化；显式 `binaryCacheDirectory` 可隔离缓存，空字符串关闭持久化；直接 RHI 调用默认不写磁盘。

编辑器显示使用 ImGui 的独立 descriptor/vertex ABI，经 `getShaderModule` 与 `getExternalGraphicsPipeline` 接入相同持久缓存。native 适配器提供完整稳定状态标识，调用 Vulkan 后端的 `createCachedGraphicsPipeline`，由后端短时锁住元数据查询/登记并仅在成功后记录 PSO。ImGui 的固定布局/混合/光栅契约变化时必须更新适配器的 ABI 版本；原始 native handles 由适配器拥有并在 Device 销毁前释放。

`MetallicShaderWarmupCore` 只链接 Registry 的编译部分，不依赖 Vulkan 设备；手动预热 target 的可选属性不变。GPU 部分属于 `MetallicRuntimeRender`。

验证入口：

- `MetallicShaderRegistryUsageAudit`：禁止 `Source/` 与 `Tools/` 中的 compiler/RHI/native Vulkan 获取旁路，Slang/RHI 实现和 Registry 是明确例外；底层 RHI 测试仍可直接验证后端。
- `shader_registry_source_pso_and_device_lifetime`：源码冷/热缓存、shader 与管线状态变体、失败发布、并发创建、不同 Device 的持久复用。
- `MetallicNativePipelineCacheTests`：不需要 GPU 的 single-flight 成功/错误传播与重试、不同 key 并发、factory 异常恢复、弱引用过期、LRU 复用/淘汰、容量调整与类型/设备隔离。
- `shader_registry_native_objects_single_flight_and_bypass`：实际 Vulkan compute/graphics handle 复用、并发同 key 仅一次 native 调用、失败等待者唤醒、热更新/状态变体、逐获取指标、warm hit 验证与显式缓存/RHI 旁路。
- `MetallicDeferredShaderCacheWriterTests`：合并、后台阻塞隔离、保存期间的新请求、失败重试和退出排空。
- `pipeline_cache_deferred_save_and_snapshot`：实际 Vulkan 缓存保存的同步/后台计时、并发新增与全量重新加载。
- `pipeline_cache_concurrent_create_save_and_stats`：同一 cache 的并发 compute 创建、逐调用反馈计数、同时保存/读取统计及快照重载。
- `pipeline_cache_native_feedback_and_refresh`：已知 hash 与空 native cache 的组合、compute/graphics/native 适配器反馈链、backend 更新持久化、第二 Device 重载和像素读回。
- `MetallicShaderObjectCacheFileTests`：文件兼容性、损坏、尺寸边界、原子替换与并发保存。
- `shader_object_binary_persistence_readback`：SPIR-V 导出、BINARY 命中、片元变体、损坏回退和第二 Device 的像素一致性。
- `MetallicShaderObjectBinaryProcessSmoke`：关闭驱动内部缓存，独立进程检查 binary 加载与精确像素读回。
- `MetallicShaderObjectBinaryRenderingSmoke`：检查 Vulkan 校验日志、Material/Wireframe 的两个 Device 生命周期及逐帧像素一致性。
- `shader_object_material_readback`：通过 BINARY 创建的两套 bindless 材质 shader 实际绘制与读回。
- `hdr_editor_imgui_composite`：编辑器显示管线的 native 缓存复用与 scRGB/PQ 像素读回。
- `MetallicLookDevPathTracePipelineCacheSmoke`：关闭驱动内部缓存，用两个实际 LookDev 进程检查 OpenPBR PT 与 SHaRC 的 Registry 缓存覆盖。

`[ShaderRegistry] PSO cache group=... hits=... misses=...` 是容器 hash-table 统计；新增 `nativeHits/nativeMisses/nativeUnknown` 分别计数有效驱动命中、有效反馈未命中及无效反馈。`[PipelineCreate] Begin/End` 在 native 调用前后打印，记录 `nativeMs`（仅 Vulkan 创建调用）、`cacheLockWaitMs`（应用缓存锁等待）和有效反馈的 `driverMs`；无效反馈的 driver duration 和 native hit 为 unknown。驱动未报告命中不等于发生了 shader 编译；首次 native 创建仍可耗时较长。默认 Vulkan 1.4 已包含 creation feedback，无需额外开启扩展。外部 graphics 适配器已有的 feedback chain 会保留。性能结论仍需同负载、同缓存条件实际计时。

2026-10-12 的对象缓存验证使用现有 Debug `cmake-build-debug` 与 Release `build-release` 构建，未更换编译器或 SDK 配置。11 个 CPU 用例及 Registry 架构审计通过；开启 Vulkan Validation 后，两个 Registry 用例、三个 PipelineCache/Recipe/feedback 用例，以及四个 prepared dispatch、纹理提交与 Kernel GPU 读回用例通过。Registry 的 bindless 分支使用 `--rhi-bindless` 执行；初始未开启 bindless 的旧 Registry 用例跳过后已补跑通过。未出现 Vulkan VUID；执行读回测试中的两条 invalid-dispatch 日志来自预期的负向检查。Release Benchmark 的 9 个 GPU 回归通过，确认显式缓存/直接 RHI 测量仍执行原生创建。

```powershell
cmake --build cmake-build-debug --target MetallicNativePipelineCacheTests MetallicRHITests
ctest --test-dir cmake-build-debug -R '^MetallicNativePipelineCacheTests$' --output-on-failure
.\cmake-build-debug\tests\MetallicRHITests.exe `
  --gtest_filter=RHIResource.shader_registry_native_objects_single_flight_and_bypass:RHIResource.shader_registry_source_pso_and_device_lifetime `
  --rhi-bindless --rhi-validation
cmake --build build-release --target MetallicShaderPipelineBench
python -B tests/perf/TestShaderPipelineBench.py --executable build-release/Source/MetallicShaderPipelineBench.exe --gpu
```

原生计数 hook 验证了同 key 的 8 个并发请求只进入一次 `vkCreateComputePipelines`，并验证不同 key 可以同时进入原生调用；这不等于 Driver 内部没有串行化。Strong LRU 的容量按对象数量限制，尚未测量完整场景切换后的 Driver 内存占用或启动/帧耗时收益。
