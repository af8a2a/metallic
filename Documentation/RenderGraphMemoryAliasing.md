# RenderGraph memory aliasing 可行性与实现方案

调研日期：2026-09-30。依据当前工作区源码（包含尚未提交的 RenderGraph/RHI 修改）、作者资料及 Vulkan/VMA 官方文档。

首次调研后的实现更新：2026-10-01。首阶段已实现独立 `VkImage` 共享同一个别名槽位，并增加原生分配容量统计及 GPU 开关对比。下文保留调研时的架构分析，当前接口、测量范围与结果见下面的实现说明。

## 首阶段实现

通过 `RenderGraphCompileOptions::enableTextureAliasing = true` 显式启用，默认关闭。编辑器和 `RenderGraphPreviewRenderer` 也支持环境变量 `METALLIC_RENDER_GRAPH_TEXTURE_ALIASING=1`。pass 的输出字段必须声明完整初始化契约，例如：

```cpp
reflection.addTextureOutput("color")
    .colorWrite()
    .transient(RenderGraphInitialization::Clear);
```

`Clear` 和 `FullOverwrite` 都是 pass 的承诺：每次执行、所有 texel、整个初始图级使用边界（包含内部 stages）都会初始化。不满足该条件的字段必须保持 `Persistent` 或 `Unknown`。当前已审计 `CopyColorPass.color` 和 `AutoExposurePass.color` 的 `FullOverwrite`。场景未就绪时可能跳过的 pass 不参与；`VisibilityBufferPass` 保持原分配。

- `RenderGraphTextureAliasPlan` 根据活动图的语义依赖计算严格 happens-before。它合并输入别名的全部使用，要求前一成员的所有使用都先于后一成员；独立分支、同一 pass 同时使用的资源不共享。
- graph-owned Device Texture2D 的 native requirements 决定槽位容量、对齐和共同 memory type。强制 dedicated allocation、depth、外部资源及其他不支持的类型保持原分配。每个槽位绑定 offset 0，没有 offset packing 或 buffer aliasing。
- RHI 的 `textureAliasAllocationRequirements()` 和 `createAliasedTextures()` 创建独立 image/view 与共享 backing owner。记账和释放各一次；command/view 的保留链继续保护 image 与 backing。
- 每张别名 image 在每帧激活时从 `Undefined` 丢弃布局开始，先编码物理内存同步，再编码自己的布局切换。语义依赖和末端使用者接入现有 GPU submission/join 前驱，跨帧继续等待上一图执行及输出消费者。
- `markOutput`、`extraOutputs`、presentation 字段及 debug observer 排除复用。`outputResource()` 仍提供元数据；只有 `isExportedOutput()` 为真才能在图外使用。`transitionOutput()` 拒绝已别名的 transient；预览切换会重新编译并将所选输出加入 `extraOutputs`。
- 调用方提供 command buffer 的执行需要有效的 `RenderFrameContext`；无完成跟踪返回 `Unsupported`。上一外部执行尚未提交或取消时，新执行返回 `InvalidArgument`，避免多个未提交录制占用同一槽位。取消录制后需要重新编译，避免保留只在录制期前进的 persistent 资源状态。提交失败保持现有图失效与完成跟踪机制。
- `ResourceMemoryInfo` 保留独立 image 的 `allocationId`，新增共享 `backingAllocationId` 和 `backingSizeBytes`。viewer 的 backing 总量按 owner 去重，资源矩阵仍保留每张 image。

相关实现位于 [alias planner](../Source/Runtime/Render/RenderGraph/RenderGraphTextureAliasPlan.cpp)、[executor](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp) 和 [Vulkan backend](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp)。CPU 规划及 GPU 功能覆盖见 `tests/rhi/RenderGraphTextureAliasPlanTests.cpp`、`TextureAliasingTests.cpp`、`RenderGraphTextureAliasingTests.cpp`。

### 实现验证（2026-10-01）

复用 `build-scheduling-release` 的 Release/MSVC 配置构建 `MetallicRHITests` 与 `Metallic`，构建通过。39 项相关回归全部通过，包含六项 planner CPU 测试、五项 alias GPU 测试，以及 access plan、内部 stages、parallel/cancellation、resource memory info、viewer、AutoExposure 回归。

五项 alias GPU 测试另以 `async` profile 在隔离进程运行 synchronization validation，全部执行并通过，无跳过，各 `validation.json` 消息数均为零：

```powershell
.\build-scheduling-release\tests\MetallicRHITests.exe --tb-run --tb-suite sync `
    '--tb-filter=*texture_aliasing*' --tb-validation sync --tb-require-all `
    --tb-layer-path C:\VulkanSDK\1.4.350.0\Bin `
    --output-dir build-scheduling-release/texture-aliasing-sync
```

其中图级 A/B 测试在可用的 graphics、copy、compute 队列模式逐像素对照开关两种配置，每种模式执行四帧；捕获验证独立 image、重叠 backing、激活 barrier、末端使用者依赖和跨队列 semaphore wait。native 测试还覆盖不同尺寸/格式、预算只计一次、view/command 保活及失败原子性。报告位于本地输出 `build-scheduling-release/texture-aliasing-sync/report.html`。

本机两条失效的系统 Vulkan layer manifest 会使已有 AutoExposure 测试将 loader 错误计入验证错误。最终回归使用当前进程的 `VK_LAYER_PATH=C:\VulkanSDK\1.4.350.0\Bin` 与指向空目录的 `VK_IMPLICIT_LAYER_PATH`，没有修改系统配置或过滤测试断言。隔离 testbench 的 `--tb-layer-path` 已提供相同隔离机制。

设置 `METALLIC_RENDER_GRAPH_TEXTURE_ALIASING=1` 的编辑器 `--smoke-test` 通过一帧 Vulkan 提交与呈现；这不证明真实场景长期稳定性、复杂 SDK 路径的别名覆盖或端到端性能收益。没有为 SDK 私有缓存或 history 启用复用。

### 显存统计与受控 GPU 测量

`executor.textureMemoryStats()` 在编译完成后即可读取；同一份编译代统计发布到 `executionStats().textureMemory` 与 `executionSnapshot()->textureMemory`。编辑器 **Statistics**、**Execution → Memory** 和捕获 JSON 显示总量、候选/别名纹理计数及每个共享槽位的成员。只在编译时查询 native requirements，逐帧读取不查询原生分配或等待 GPU。

- `logicalBytes`：不复用时的独立分配容量。实际别名成员使用原始 descriptor（没有 `VK_IMAGE_CREATE_ALIAS_BIT`）查询；其他纹理使用实际独立 backing 容量。
- `backingBytes`：当前图的实际 backing 容量，按 `backingAllocationId` 去重。
- `savedBytes = max(logicalBytes - backingBytes, 0)`；反方向差额单独报告 `overheadBytes`。只有比较完整时才报告节省/额外开销及百分比；未知项和不完整槽位明确标记。
- 统计包含驱动要求的尺寸/对齐，输入别名不会重复计算。范围是当前编译图拥有的纹理，排除 buffer、scene、私有 imports、history、SDK 及其他 in-flight generation；不是整个进程的 VRAM 驻留量。

2026-10-01 在 NVIDIA GeForce RTX 5070 Ti / NVIDIA 616.92 / Release MSVC 上，受控图 `A clear → ReadA → B clear → ReadB` 的两张 RGBA8Unorm 图像共享一个槽位，B 的宽、高各为 A 的一半。下面是原生分配实测，而不是像素数估算；MiB 为 1,048,576 字节。

| 受控图 A 分辨率 | 独立分配 | 共享 backing | 节省 | 比例 |
| --- | ---: | ---: | ---: | ---: |
| 1920 × 1080 | 10.78125 MiB | 8.43750 MiB | 2.34375 MiB（2,457,600 B） | 21.739% |
| 3840 × 2160 | 40.31250 MiB | 31.87500 MiB | 8.43750 MiB（8,847,360 B） | 20.930% |

两组均执行 GPU clear/readback，开启/关闭的逐像素结果一致；`FrameResources.allocationBytes` 与 `deviceLocalBytes` 的对照差额均等于 `savedBytes`，各隔离 synchronization validation 消息数为零。比例不同来自 native image requirements 的分配粒度，不能直接使用裸像素占用替代。

复现命令（1080p；4K 将两个环境变量改为 3840、2160）：

```powershell
$env:METALLIC_TEST_ALIAS_WIDTH = '1920'
$env:METALLIC_TEST_ALIAS_HEIGHT = '1080'
.\build-scheduling-release\tests\MetallicRHITests.exe --tb-run --tb-suite sync `
    '--tb-filter=*texture_aliasing_vram_statistics' --tb-validation sync --tb-require-all `
    --tb-layer-path C:\VulkanSDK\1.4.350.0\Bin `
    --output-dir build-scheduling-release/alias-vram/controlled-1080p
```

原始 JSON、native capabilities、validation、逐像素 readback 与实际捕获值的 Memory viewer 截图在本地 `build-scheduling-release/alias-vram/controlled-1080p/` 和 `controlled-4k/`。该受控图证明共享机制节约了 device-local 分配容量；真实管线的收益取决于已审计候选及它们的 GPU 生命周期，不能将这里的比例套用到场景。

统计接入后的构建与 40 项相关回归通过；包含新统计测试的六项隔离 alias GPU 测试全部通过 synchronization validation。Memory viewer 截图已检查，统计和原生重叠范围一致。

### 默认生产管线 MiniZorah 的实际收益

相同 Release 构建、现有 shader cache、`--rhi-validation --rhi-realtime --rhi-async-compute` 配置分别运行关闭和开启的完整 `minizorah_realtime_pipeline` 回归。两次均通过原有 180 帧、shaded pixel、motion/depth guide、viewport resize、streaming budget 和 session retirement 检查；原始输出没有 VUID 或运行错误。两次均采用相同的 OMM capability fallback（本机 validation layer 版本不足该可选路径要求），没有对该路径的验证作出声明。这里不比较启动或帧耗时。

两张生产结果 PNG 不是逐像素相同：512 × 320 图像有 10,747 个像素不同，RGBA 的平均绝对通道差为 0.02261/255、最大差为 9，仅一个像素的任意通道差超过 8。两次运行均没有实际共享槽位，现有生产正确性 oracle 均通过；没有单独定位这些跨运行像素差异的原因。逐像素一致性的结论只适用于上述受控图。

取 `phase=180-frames-completed` 的稳态 512 × 320 样本，而非报告尾部的 retirement fixture：

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| graph-owned 纹理数 / backing 数 | 14 / 14 | 14 / 14 |
| 声明 transient / 固定输出数 | 1 / 1 | 1 / 1 |
| native-qualified 候选数 | 0（关闭时不查询） | 1 |
| 实际别名纹理 / 共享槽位 | 0 / 0 | 0 / 0 |
| 独立 / 实际纹理 backing | 8,208,384 / 8,208,384 B | 8,208,384 / 8,208,384 B |
| 节省 | 0 B | **0 B（0%）** |
| FrameResources 总分配（含 buffer 等） | 68,775,360 B | 68,775,360 B |

因此当前默认生产图节省 **0 MiB**。该图只有 `AutoExposure.color` 一个已审计候选，不能形成至少两个成员的别名槽位；`CopyColorPass` 虽已具备契约，但当前 pipeline asset 未使用。Visibility 和 scene-dependent 输出、history 及 SDK 私有资源继续保持原生命周期。增加实际收益需要逐一审计更多临时输出的完整初始化、跨帧使用与 GPU happens-before，不能仅将字段改成 transient。

本次稳态 graph texture backing 为 7.828125 MiB；整个 device-local VMA allocation 为 2,122,330,928 B，远大于该纹理子集。新统计准确覆盖第一阶段可优化的范围，场景 geometry/CLAS/RT 等分配以及 SDK、驱动驻留不能并入它的节省百分比。即时 compile 样本也可能包含上一代延迟释放的资源，因此 device telemetry 与当前编译图容量分别记录。

复现（关闭的那次将 `METALLIC_TEST_TEXTURE_ALIASING` 改为 `0`，使用独立输出目录）：

```powershell
$env:METALLIC_TEST_MINIZORAH = '1'
$env:METALLIC_TEST_TEXTURE_ALIASING = '1'
$env:VK_LAYER_PATH = 'C:\VulkanSDK\1.4.350.0\Bin'
$env:VK_IMPLICIT_LAYER_PATH = (Resolve-Path build-scheduling-release/empty-vulkan-layers).Path
.\build-scheduling-release\tests\MetallicRHITests.exe `
    '--gtest_filter=*minizorah_realtime_pipeline' --rhi-validation --rhi-realtime --rhi-async-compute `
    --output-dir build-scheduling-release/alias-vram/minizorah-on
```

`empty-vulkan-layers` 是此前验证创建的空目录；只对当前测试进程设置 layer 路径。原始统计在 `build-scheduling-release/alias-vram/minizorah-off/MiniZorahTextureMemory-AliasingOff.json` 与 `minizorah-on/MiniZorahTextureMemory-AliasingOn.json`，均带有 `correctnessVerified=true` 和各阶段原生 telemetry；对应 `run.log` 保留完整运行输出。

## 结论

Metallic 可以在现有 RenderGraph 与 VMA 3.3.0 上增加真正的物理内存别名，不需要重写图系统或升级 VMA。推荐顺序是：

1. 先建立 transient 初始化/导出契约及只分析的 alias plan，测量可复用资源规模。
2. 实现 graph-owned Device Texture2D 的物理别名：每个 alias slot 一个共享 backing，各独立 VkImage 绑定该 backing 的 offset 0。
3. 验证稳定性及 GPU 开销后，再扩展 buffer、offset packing、私有 transient imports 和更多调度策略。

首版默认关闭，通过编译选项显式启用；仅已审计、显式声明 transient 的字段参与。首版采用整个 pass 的使用跨度，并保留已有异步分支的并行机会。最大风险在 GPU 执行先后关系、跨帧状态、资源逃逸及提交失败，而不在贪心分配算法。

## 1. 调研时的实现与已有支撑（2026-09-30）

| 当前事实 | 源码入口 | 实现影响 |
| --- | --- | --- |
| 每个活动 output 独立创建 texture/buffer，资源保留在已编译图中 | [allocateGraphResources](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp)，约 1229–1413 行 | 拆分描述符解析、alias 规划及物理创建；目前没有 transient allocator |
| `inputAliases` 将输入映射到生产者输出 | 同文件 `rebuildInputAliases`，约 660 行 | 生命周期分析必须合并这些名字，不能把输入重复计成资源 |
| field 已有 usage/access/尺寸，缺少 transient 和首次初始化承诺 | [RenderGraphTypes.h](../Source/Runtime/Render/RenderGraph/RenderGraphTypes.h)，`RenderGraphField` | 首次 `writes=true` 不足以证明可丢弃旧内容 |
| 实际 native queue identity 在每次 execute 时确定 | Executor 约 3216–3228 行 | 编译期 pass 序号不能直接作为跨队列 GPU 时间 |
| 访问计划已有前驱、读者 frontier、布局 frontier、共享 barrier 编码器 | [RenderGraphAccessPlan.cpp](../Source/Runtime/Render/RenderGraph/RenderGraphAccessPlan.cpp)，约 251–407 行 | 可增设物理槽位 handoff，并接入现有提交前驱 |
| 普通 Device 分配当前要求 dedicated allocation | [VulkanRHI.cpp](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp)，`allocationInfoForMemory`，约 1085 行 | alias backing 需要独立创建策略，不能直接共享当前 allocation |
| resource impl 当前各自拥有 VmaAllocation 并调用 vmaDestroyBuffer/Image | 同文件约 3111、3164、3508–3525 行 | 必须拆分 native object 与 backing 所有权，否则重复释放、重复记账 |
| command/frame 已保留资源 impl 至提交完成 | [RenderFrameContext.cpp](../Source/Runtime/Render/Core/RenderFrameContext.cpp)，约 104–111 行；VulkanRHI 约 4459 行 | 保留 resource impl → backing 的强引用链即可复用现有 completion 生命周期 |
| execution viewer 已显示 native block、offset、size | [RenderGraphExecutionSnapshot.h](../Source/Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h)、[EditorRenderGraphViewer.cpp](../Source/Editor/EditorRenderGraphViewer.cpp) | 可以展示共享范围，但内存汇总需要按 backing 去重 |

当前 bundled VMA 为 3.3.0，已有 `vmaAllocateMemory`、`vmaCreateAliasingImage2`、`vmaCreateAliasingBuffer2` 和 `VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT`，见 [vk_mem_alloc.h](../External/VulkanMemoryAllocator/include/vk_mem_alloc.h)。现有 NRD 局部 pool 会复用同一个 Texture 槽位；这可以提供经验，但没有解决图级不同 VkImage 的共享 backing。

## 2. 参考资料能提供什么

| 参考 | 已读取范围及可借鉴内容 |
| --- | --- |
| [Frostbite GDC 页面](https://www.gdcvault.com/play/1024612/FrameGraph-Extensible-Rendering-Architecture-in) / [EA-DICE 原始讲义](https://www.slideshare.net/slideshow/framegraph-extensible-rendering-architecture-in-frostbite/72795495) | GDC 页面只读到元数据；讲义正文及备注可读。完整访问声明是寿命分析前提，async compute 的寿命必须延伸到同步点；compute/graphics 流水重叠需要真实同步与初始化。不能把讲义中的历史节省比例套用到 Metallic。 |
| [Pavlo Muratov 的 GPU DAG 文章](https://levelup.gitconnected.com/organizing-gpu-work-with-directed-acyclic-graphs-f3fd5f2c2af3) / [同作者 GPU Memory Aliasing](https://levelup.gitconnected.com/gpu-memory-aliasing-45933681a15e) | 是 [PathFinder](https://github.com/man-in-black382/PathFinder) 实现作者资料。对象池与 placed-resource overlap 的区分、按尺寸及冲突范围做贪心 packing 可借鉴。DX12 的 queue state/split barrier 细节需重新映射到 Vulkan。dependency level 本身没有证明 GPU 同步，这是本调研的判断。 |
| [Maister 的 Vulkan Render Graph 原文](https://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/) | 实际方案复用相同尺寸/格式的 VkImage/VkImageView；handover 继承同步信息并使用 Undefined 布局，排除 history/feedback。它提供了寿命与 handover 设计经验，不能被当作不同 image 共享 VkDeviceMemory 的完整实现。 |
| [给定 YouTube](https://www.youtube.com/watch?v=pr8HaIaZfpk) | 页面读取失败，未可靠取得标题、视频或 transcript；没有使用其内容作为依据。 |
| [给定 Halcyon PDF](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/wihlidal-halcyonarchitecture.pdf) / [作者来源页](https://www.wihlidal.com/blog/graphics/2018-11-30-halcyon-architecture/) / [EA-DICE 讲义镜像](https://www.slideshare.net/slideshow/seed-halcyon-architecture/124574267) | 原 PDF 超过浏览工具大小限制，作者页及讲义镜像可读。另外读取了 [EA 官方 Halcyon + Vulkan 讲义](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/wihlidal-munich2018-halcyonvulkan.pdf) 第 41–42 页的文字/备注：作者选择更简单的内存模型，并指出 alias barrier/discard 有成本。其 2018 年数字不能预测当前 Metallic 的损耗。 |

底层实现约束以当前 [Vulkan Memory Aliasing 规范](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-aliasing)、[VMA Resource aliasing](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/resource_aliasing.html) 和 [VMA allocation flags](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/group__group__alloc.html) 为准。

## 3. 三种可选路线

| 路线 | 能力 | Metallic 中的代价与定位 |
| --- | --- | --- |
| 完全兼容描述符的对象复用 | 多个逻辑资源交替使用一个 Texture/VkImage | RHI 改动较少，格式/尺寸等限制较多；仍需内容版本与 handoff。可作实验，但不必作为物理 aliasing 的必经步骤 |
| 独立 VkImage + 共享 alias slot | 不同尺寸/格式的 texture 在不同 GPU 使用阶段占用同一 backing | 需要 requirements、共享 owner 和物理槽位同步；推荐作为真正 aliasing 的首版 |
| 大 heap 的 offset packing，包含 buffer/image | 更灵活地让一个大资源与多个小资源共享范围 | 增加碎片、granularity、区间覆盖、BDA、预算及诊断复杂度；后置 |

首版以最少的新状态覆盖真实物理 aliasing：每组成员都绑定 offset 0，按该组最大需求分配 backing。为不同格式设置不同 VkImage，各自保留 view/descriptor/layout identity；不为了复用把资源强行改成同一张大 texture 或合并格式。

## 4. 首版参与条件

建议给 field 增加保守默认的 lifetime 与 initialization 声明，例如：

```cpp
enum class RenderGraphResourceLifetime { Persistent, Transient };
enum class RenderGraphInitialization { Unknown, Clear, FullOverwrite };
```

`Clear` 表示 graph/pass 在首次使用前覆盖全部被使用范围；`FullOverwrite` 是经过 shader/命令审计的完整写入承诺。二者都应与实际首个 stage 匹配，而不是只依据 pass 最终 access 推断。

首版候选同时满足：

- 图拥有的 Device Texture2D，普通 optimal tiling；先限制颜色格式。
- 明确 opt-in transient，首次使用完整初始化；同 pass 同时使用的资源不能共享槽位。
- 所有 GPU 访问都在可审计的 graph/pass 范围内；使用跨度包括整个 pass、debug copy 及 fork/join。
- 没有 history、持久缓存、外部/native SDK 所有权或强制 dedicated 要求。
- 没有被 markOutput、presentationOutput、extraOutputs、当前 preview 或显式 export 固定。

排除 HostUpload/HostReadback、borrowed swapchain image、AS/CLAS、streaming geometry/material allocations、HistoryResourceManager、未审计的 SDK/private imports。storage read-write、atomic 累积、attachment LOAD、仅写部分像素或 mip/layer 的资源默认不合格；完整 write access 与完整初始化不是同一概念。

## 5. 生命周期必须来自 GPU 偏序

编译期先合并输入别名，收集每个 logical resource 的所有 pass uses。判断 A 能先于 B 共用槽位，应证明：

```text
所有 A 的末端使用完成  happens-before  所有 B 的起始使用
```

同一 pass 属于冲突；不相依的 async branches 属于冲突。不能仅比较 first/last pass index、DAG depth 或 CPU 录制结束时间。对于首版 whole-pass 模型，末端和起始使用由访问 DAG 的 maximal/minimal uses 决定。

编译时可采用保守的图依赖闭包，避免把 allocation plan 绑定到每次执行的实际 queue 选择。但还要把用于证明安全的边落实到 GPU：当前 ActiveGraph 拓扑只生成 executionList，实际提交依赖来自 `accessPlan.passes[].predecessors`；同 layout 的 read→read 图边未必产生资源 hazard 前驱。

建议生成 `aliasSafetyDependencies` 并与 access predecessors 合并/去重。首版仅物化已有语义先后关系，不主动在独立分支之间增加串行化。执行时根据实际队列编码：同队列 handoff 使用 barrier；跨队列 handoff 通过 producer 完成 segment 的 signal/wait，再激活新 image。image layout transition 的 scope 必须在目标 queue 上合法；同队列重叠绑定的内存依赖可经现有 `MemoryBarrierDesc` 编码，见 [VkMemoryBarrier2](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryBarrier2.html)。

`parallelCompute` 会扩展出 producer、compute/graphics branches 和 join，见 Executor 约 3545–3583 行。A 的末次使用若在该 pass 内，B 必须等待 join；当前 pass completion 取最后的 join（约 3842–3844 行），可直接沿用。首版不做 pass 内 stage aliasing。

后续可以增加 `PreserveOverlap` 与 `MinMemory` 两种策略。前者利用已有 GPU 偏序；后者允许用额外依赖交换显存节省，其 GPU 时间必须单独测量。如果利用同 native queue 的额外顺序进一步 packing，需要调度模式对应的计划或明确增加依赖，不能在动态 async 切换时沿用未经验证的线性计划。

## 6. Alias slot 与 handoff 状态

建议将 allocation plan 与 access plan 分开，但在执行前合成：

```text
Resolved descriptors + logical uses + pin set
    → compatible requirements + GPU ordering proof
    → AliasAllocationPlan { resource → slot, safety dependencies }
    → independent image objects + shared backing
    → existing access plan + AliasTransitions
    → queue segments / barriers / accepted submission completion
```

每个槽位记录独立的 backing identity、成员和 last occupant/access frontier；每个 image 继续有独立 logical/native identity、layout 及内容有效性。两个 images 共享显存不应让 `RenderGraphComputeStages` 把它们合并为同一资源。

切换 A → B 要表达两个操作：

1. 让 A 所有未完成 reader/writer/layout 操作与 B 激活形成依赖，包括 B 的初始化/布局转换。
2. 将 B 作为 discard 激活，使用 `Undefined → first layout`，然后执行 clear/full overwrite。

按物理槽位同步的 frontier 不能因为 B 的 layout 设置为 Undefined 而清空；丢弃内容不等于前面的 GPU 工作已经结束。切换后的内容失效与 image 布局要求见 [Vulkan Memory Aliasing](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-aliasing)。

## 7. RHI 与 VMA 的最小接入

建议能力边界如下，具体命名可随实现调整：

```cpp
struct ResourceAllocationRequirements;
class AliasAllocation;  // shared backing, opaque native allocation
class AliasMemoryLease; // owner + allocation-local offset + capacity

// Device:
queryTextureAllocationRequirements(...);
createAliasAllocation(...);
createAliasedTexture(..., const AliasMemoryLease&);
```

requirements 要包含 native size、alignment、memoryTypeBits、requires/prefersDedicatedAllocation，以及 device、最终 descriptor/binding mode 来源。最好返回带来源的 token，避免用户自行拼出不匹配的 requirements。query 与真正 create 必须使用相同 usage、queueAccess、flags；绑定前验证最终 native requirements。必要时采用先创建 unbound image、查询再规划/绑定的事务式实现。

offset 0 的 slot requirements 为 `max(size)`、`max(alignment)` 和 `AND(memoryTypeBits)`，类型交集为零时分组失败并采用独立分配。`requiresDedicatedAllocation` 首版排除，`prefersDedicatedAllocation` 保留作策略提示。上述需求组合来自 [VMA alias guide](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/resource_aliasing.html)。

`AliasAllocation` 统一持有 VmaAllocation、容量、memory type、domain 和 budget accounting。image 子对象只销毁自身 VkImage；backing 在最后一个强引用释放时 `vmaFreeMemory`。TextureView 与命令仍保留 resource impl，再由 impl 保留 backing，不能只保留 memory 而提前销毁 VkImage。Device 必须存活至这些 backing/resource/view/commands 释放及相关提交完成，沿用当前 RHI 契约。

raw `vmaAllocateMemory` 不能直接复用当前 `VMA_MEMORY_USAGE_AUTO` helper；bundled VMA 明确要求 AUTO 的调用收到 resource CreateInfo。应使用 UNKNOWN 加显式 required/preferred memory flags，或选择经过交集验证的 memory type。

`VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT` 用于避免插入 resource-tied `VkMemoryDedicatedAllocateInfo`。首版可用 `DEDICATED_MEMORY_BIT | CAN_ALIAS_BIT`，给每个槽位一个独立 native block，简化预算与 granularity；“独立 block”与“绑定到某一个特定 image 的 dedicated allocation”需区分。见 [VMA allocation flags](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/group__group__alloc.html)。

可以在 alias image create 路径明确设置 `VK_IMAGE_CREATE_ALIAS_BIT` 并让 query 使用相同参数，但不要把该 bit 当作所有 discard-only aliasing 的通用前提。它对相同参数/绑定的 images 主要提供内容一致解释；首版仍使用 discard，不继承 A 的像素。后续 buffer/optimal-image 混合 packing 必须处理 [bufferImageGranularity](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-bufferimagegranularity)。

## 8. 当前跨帧顺序与失败路径

当前两个 submission slots 允许 CPU 录制重叠；图的 GPU frame N+1 首批仍等待 N 的 aggregate `lastSubmittedCompletion`，见 Executor 约 3302–3306、3489–3491 行。completion waits 使用 AllCommands，见 RenderFrameContext 约 170–182 行。

Editor 的后续使用也有现成链路：`transitionOutput()` 添加 graph completion dependency，并登记 UI frame completion 到 `externalCompletions`（Executor 约 3961–3968 行）；下次图执行合并这些外部依赖。多平台窗口通过尾部提交纳入 frame completion，见 EditorApplication 约 6970–6987 行。

因此，首版不必为两个 CPU slots 强制复制两份 alias backing，保留这些 GPU waits 即可。槽位仍须跨帧记录上一 occupant → 下一帧首个 occupant 的 handoff；不能仅重置 resource.state。如果未来允许同一图真正跨帧 GPU 并行，再按 in-flight generation/slot 分配独立 backing，或证明更细粒度的区间完成后再复用。

外部录制入口 `execute(CommandBuffer&)` 仅在绑定有效 RenderFrameContext 时登记 external completion（Executor 约 2927–2934 行）。首版 aliasing 只支持 executor-managed submit 或带可跟踪 frame context 的外部录制；无法跟踪提交/取消的 command buffer 回退至独立分配。还要防止同一 executor 多次 recorded-but-unsubmitted execution 在缺少完成依赖时同时占用该 backing。

Pipelined submission 会在后续 pass 尚未录完时接受 prefix。首版沿用当前失败策略：

- 未接受任何工作时，回滚本次 activation 状态；完成 CPU 录制收尾后可释放临时 owner。首版仍沿用当前 abort 使 compiled generation 失效的策略，重编译后重试。
- 接受 prefix 后失败时，保留 backing 至 accepted aggregate completion，令 compiled generation 失效并要求重编译。
- 重编译/resize/切换图前等待旧 submitted work；旧 owner 的预算一直计算至真正释放。

现有入口为 Executor 约 3343–3374 行和 RenderFrameContext 约 325–337 行。所有权保留解决销毁安全，GPU dependencies 解决同一范围的使用安全，两者都需要。

## 9. 输出、预览与 debug 契约

当前 `outputResource()` 可以取得任意已分配 output（Executor 约 3978 行）。外部读回、benchmark、编辑器预览都在使用这个接口。开启 transient 后，图结束时中间输出可能已经被覆盖，不能保持“任何 output 随时可读”的旧假设。

建议增加显式 export/pin 查询：

- `markOutput`、presentation、`extraOutputs`、当前 preview 和外部 readback 请求进入 pin set，首版完全排除 alias。
- 获取 transient 的 native handle 本身不允许事后消费其内容；真正外部消费必须在执行前 export，并重建 alias plan。
- `transitionOutput()` 检查 export generation；未固定的 transient 不能事后只改 layout 就恢复原内容。
- 可以保留描述符/诊断查询，并与可外部消费的资源句柄接口区分。

预览切换必须触发 pin set 检查/重编译。当前 EditorApplication 约 6627–6633 行及 preview renderer 约 4261–4269 行只检查 outputResource 是否存在；选择输出也可直接绑定已有 view。需要检查 `isExportedOutput()` 或 pin generation，而不能继续仅凭资源存在复用。

`enablePreviewOutputAccess` 当前只增加 usage bits，不能充当寿命保证。如果为了兼容而把所有可预览输出永久 pin，会显著压缩收益；推荐仅 pin 当前选择和明确的外部消费者。

首版有 RenderDebugObserver 时禁用 aliasing，保留现有检查行为。只采集不可变 execution snapshot 不必禁用。后续审计 checkpoint 的原位 copy/restore、private imports 后，可将其 GPU 使用纳入 whole-pass 生命周期再启用。

## 10. 预算与 execution viewer

当前每个 resource 独立增减预算；aliasing 后应由 backing owner 记一次真实 allocation，归属 FrameResources。backing 创建仍按选择的 memory type 和真实容量执行 `admitMemoryLocked()`，并在策略启用时设置 `VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT`；成员不能重复执行预算 admission/track，失败路径也只释放/扣账一次。资源逻辑容量总和、backing allocation bytes 和 device heap/block bytes 应分开报告。

建议保留当前 distinct resource generation 的 ID 契约，增加 `backingAllocationId`、slot ID、binding offset/size 和 backing capacity。memoryInfo 中 resource range 表示真实绑定需求，不能给每个成员都填共享 backing 的完整容量。

Viewer 当前 `overlaps()` 忽略相同 allocationId（约 236–243 行），memory 总和按 resource ID 计算（约 629–630 行）。必须改为：不同 native resources 可以共享 backing ID；物理 bytes 按 backing 去重，范围显示独立保留。增加 planned alias transition 与实际 encoded/wait 信息，区分使用跨度、物理 slot 寿命和 GPU completion。

## 11. Metallic 的候选与收益边界

实时链路包含 `VBuffer → Shadows/Deferred → DLSSSR → AutoExposure → DLSSNR → FinalBlit`。经过初始化及访问审计后，早期 visibility/material guides 与晚期曝光/后处理输出可能存在复用窗口。例如 VBuffer.visibility 的最后 graph 消费在 Deferred，而 AutoExposure.color 的首次使用在更后面；是否合格还取决于 exports、SDK 实际消费和 requirements。

相邻 pass 的 input/output 同时被一个 pass 使用，不能共享。例如 Deferred.color 与 DLSSSR.color 在 DLSSSR 中同时活跃。`PathTrace → AutoExposure → FinalBlit` 的最简三段图中，相邻资源互相冲突，FinalBlit 又被 pin，texture-only 首版可能几乎没有收益；不能承诺所有图都省很多显存。

首版不会降低 geometry/streaming、BLAS/CLAS、material residency、history 或 SDK 内部分配。大场景总 VRAM 若主要由这些资源占用，图级 transient aliasing 仅解决其中一部分。

已有本地 MiniZorah capture：[MiniZorahExecution.json](../build-scheduling-release/viewer-minizorah/MiniZorahExecution.json)，文件时间为 2026-09-26。它记录 26 个可见资源、12,201,264 bytes（约 11.64 MiB），其中包括测试 readbacks 和 private imports。该文件是既有小分辨率测试证据，资源/使用 schema 和当前源码也有差异；没有用它推算当前生产图 savings，或按比例外推到 4K。捕获文件不应提交至源代码。

建议实现前运行规划诊断，输出：独立分配基线、候选 bytes、排除理由、每个 slot 成员/容量、实际 backing bytes、padding，以及由于 pin/偏序冲突放弃的复用量。保守下界可用所有实际并行 live 资源需求的合计，但实际结果还受 alignment/type compatibility/slot packing 限制。

## 12. 落地顺序与验收

| 阶段 | 交付 | 验收重点 |
| --- | --- | --- |
| 0：声明与规划诊断 | transient/初始化/export 契约、逻辑使用图、偏序证明、只分析 slot 计划 | 不改变分配；找出真实候选，验证 independent branches 不合并 |
| 1：RHI 共享 texture backing | requirements query、owner/lease、offset 0 alias textures、预算/诊断 | 两个独立 VkImage 的 native ranges 确实重叠；释放一次，view/command retention 正确 |
| 2：RenderGraph 接入 | safety predecessors、AliasTransitions、跨帧 frontier、pin/recompile、失败事务 | 单/多队列、fork/join、多帧和 UI 消费正确；默认保持独立分配路径 |
| 3：生产评估 | 同 workload/settings 下 alias on/off、viewer 证据、像素和长期序列检查 | 真实 backing bytes 降低；记录 barrier/wait/GPU 时间，检查是否损害并行 |
| 4：扩展 | buffer aliases、offset packing、私有 transient 导入、可选 MinMemory | BDA usage flags、granularity、区间同步和跨帧 GPU 并行方案单独验证 |

Buffer aliases 后置的具体原因：`addressCommandFlags()`（VulkanRHI 约 530–540 行）现在依赖“没有 overlapping live buffers”，仅凭当前 BufferDesc 决定 storage usage flag。引入混合 usage aliases 后，需要 backing-wide usage summary，或为这些 allocation 保守使用 UNKNOWN_STORAGE_BUFFER_USAGE；BDA 不能假定 alias buffers 地址不同。规则见 [Khronos Address Command Flags](https://docs.vulkan.org/refpages/latest/refpages/source/VkAddressCommandFlagBitsKHR.html)。

测试应覆盖实际行为：

- requirements alignment/type 交集、强制 dedicated 排除、描述符改变使 token 失效；创建失败的资源和预算回滚。
- A 写入/被多个 reader 读完后切 B；重新激活 A 并 clear/full-overwrite 后读取，验证不会继承旧像素；GPU pattern readback 和内容比较。
- independent async branches 不 alias；依赖分支 handoff；sameQueue wrapper 去重；parallelCompute 必须等待 join。
- 多帧 occupant 循环、graph → UI → next graph、preview 内部输出切换、resize、图切换及 shutdown。
- 未提交失败、accepted prefix 后失败、重编译后的旧 backing retention；两个 image IDs 不被合并。
- real-time/path tracing/HDR/SDK 图的对照像素、history 稳定性及长时间流送，包含图外持久资源。

复用现有 [RenderGraphAccessPlanTests](../tests/rhi/RenderGraphAccessPlanTests.cpp)、[RenderGraphComputeStageTests](../tests/rhi/RenderGraphComputeStageTests.cpp)、[FrameContextTests](../tests/rhi/FrameContextTests.cpp)、[ParallelRecordingTests](../tests/rhi/ParallelRecordingTests.cpp)、[RenderGraphViewerTests](../tests/rhi/RenderGraphViewerTests.cpp)、[ResourceMemoryInfoTests](../tests/rhi/ResourceMemoryInfoTests.cpp) 及 [MemoryBudgetTests](../tests/rhi/MemoryBudgetTests.cpp) 的设施；新增源码注册于 tests/CMakeLists.txt。

构建和运行示例是后续实现的验收入口，本次调研没有执行这些命令：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*render_graph_access_plan*:*render_graph_compute_stages*:*render_graph_stages*:*render_graph_execution_viewer*:*resource_memory_info*' --rhi-validation --rhi-async-compute
```

后续新增 memory-aliasing 专项应有单独筛选项；GPU 能力 skip 不能视作该路径验证。生产比较记录分辨率、图/SDK 开关、队列模式、frames-in-flight、cache/warmup、pin/debug 设置，分别报告 graph backing、全设备显存及 GPU frame/pass 时间；显存收益不应写成帧时间收益。

## 本次验证边界

已完成当前源码只读审计、外部资料核对、既有 capture 范围检查和文档链接/格式检查。只新增此设计文档，未修改渲染实现，也未进行新的编译、GPU 输出、时序或性能验证。参考链接中未读取的视频及超限 PDF 已在资料表中列出。
