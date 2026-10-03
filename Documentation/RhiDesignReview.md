# Metallic RHI 设计评审：抽象简化与冗余消除

日期：2026-10-03。基线：`7226aa64d`（"Refactor render graph execution and simplify pipeline APIs"）。
工作区除 `External/microprofile` 外无改动。

本文是从**阅读源码**得到的静态结论，用于决定「哪些抽象该压缩、哪些该保留」。**本轮未编译、未运行测试、未做 GPU 测量**，因此文中所有「可删行数」是静态计数差值，不是实测收益；所有性能性表述都已避免。

与既有文档的分工：

- [RhiSimplificationResearch.md](RhiSimplificationResearch.md) / [SharedResourceRegistry.md](SharedResourceRegistry.md) 讲的是**迁移方向**（registry 身份、typed 参数、地址化、同步拆分）。
- 本文讲的是**已实现出来的抽象之间还剩多少冗余**，以及**哪些中间层可以直接删掉**，不重复迁移路线。

---

## 0. 决定性的前提：只有一个后端

`Source/Runtime/Render/GAPI/` 下除 `Vulkan/` 之外，只有与后端无关的公共设施（`RHI.h`、`CommandSubmission.*`、`PipelineCacheFile.*`、`PipelineStateHash.*`、`TextureFormat.h`）。全仓没有第二个 RHI 后端实现。

这条事实决定了整个评审的判据：**抽象不能再靠「可移植性」来论证自己，只能靠它是否让同一份契约表达得更少、更难写错。** 下文的每一条建议都按这个标准排序，而不是按「是否更接近某个理想架构」。

`RHI.h` 有 43 个 includer，其中 36 个是 `Source/` 下的生产头文件。所以它的每一次膨胀都会传导到整个渲染器。

---

## 1. 结论速览

| # | 判定 | 一句话 | 主要证据 | 可压缩量 | 调用方风险 |
|---|---|---|---|---|---|
| A1 | **架构级** | 同一「访问语义」被 3 个枚举 + 1 张手写转换表建模，且两个枚举**逐行同构** | `RHI.h:144-156` / `657-666`、`VulkanSynchronization.cpp:5-21` 与 `23-39`、`Core/ResourceSynchronization.h:30-43` | ~35 行 | **高**（Source ~106 + tests ~140 处引用） |
| A2 | **架构级** | 资源→descriptor 身份有 3 套并存，同一 `ResourceLease` 与 `BindlessHandle` 是同一概念的两个类型 | `ResourceRegistry.cpp:28-34`、`RenderGraphExecutor.cpp:1644-1697`、`GPUScene.h:309-376` | 迁移中，见 §3.2 | **高** |
| A3 | **架构级** | 24 值图访问枚举 + 11 处手写 switch 同时建模「类型 / 操作 / 用途」三个正交维度 | `RenderGraphTypes.h:60-85`、`RenderGraphNode.cpp:57-305,350-393`、`RenderGraphAccessPlan.cpp:92-150` | ~200 行 switch | 中-高 |
| B1 | 代码量 | 能力/扩展/特性管线被**独立枚举 7 次**，共 1057 行 | `VulkanRHI.cpp:1488-1583/1585-1636/1637-1930/1931-2188/2189-2440/2442-2546` | **~600 行** | 中-高（pNext 顺序） |
| B2 | 代码量 | 21 个包装类的特殊成员样板 266 行；`= default` 在本仓库**不成立** | `RHI.h` 126 行 + `VulkanRHI.cpp` 140 行 | 100–260 行 | 中（ABI） |
| B3 | 代码量 | 同一参数结构体的绑定号表按 Program 重复 5–6 份（377 行字段里只有 122 个不同字段） | `Core/NamedResourceLayouts.h:9-434` | ~255 行 | 中（shader 共享 ABI） |
| C1 | 一致性 | 公开 API 有 **3 种失败约定**并存：`Result<>` 前置校验 / `Result<>` 无校验 / `void` 静默返回 | `copyBuffer` vs `copyTexture` vs `copyTextureToBuffer` | — | 中 |
| C2 | 一致性 | `VkResult→Error` 有第二份实现，且**已行为分叉**（DGC 路径设备丢失不上报 Aftermath） | `VulkanGeneratedCommands.cpp:10-21` vs `VulkanRHI.cpp:98-121` | 12 行 | 低 |
| C3 | 一致性 | 同一 SPIR-V 内容哈希两套实现，结果不同且同名 | `PipelineStateHash.cpp:48-55` vs `VulkanRHI.cpp:10151-10156` | ~8 行 | 中（回放证据比对） |
| C4 | 一致性 | 载入状态枚举两份同名值 + 手写 1:1 转换 | `RHI.h:1199-1204` vs `PipelineCacheFile.h:17-22` | ~20 行 | 无 |
| D1 | 隐患 | `validScope` 用魔法位数做枚举上界校验，新增枚举值会**静默失效** | `VulkanSynchronization.cpp:117` | — | 高（正确性） |
| D2 | 隐患 | 94 个 `friend` 声明；`deviceIdentity()` 被 48 处调用做设备配对校验 | `RHI.h`、`CommandSubmission.h:59-67` | — | 中 |
| E1 | 可删 | 13 个公开方法零外部调用点 | 见 §5 | ~120 行 | 无 |

---

## 2. 架构级问题

### 2.1 A1：三个枚举建模一件事，其中两个逐行同构

这是本次评审**证据最硬、结论最清楚**的一项。`VulkanSynchronization.cpp` 里有两个函数：

```cpp
  5: VkImageLayout imageLayout(ResourceState usage, bool unified)
  7:     VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  8:     switch (usage) {
  9:     case ResourceState::Undefined: layout = VK_IMAGE_LAYOUT_UNDEFINED; break;
 ...
 23: VkImageLayout imageLayout(TextureLayout usage, bool unified)   // 同一份 switch，换枚举名
```

两个函数体**逐行同构**，唯一的差别是限定名前缀。而 `TextureLayout`（`RHI.h:657-666`，8 项）是 `ResourceState`（`RHI.h:144-156`，11 项）的**真子集**：前 8 个枚举名与顺序完全相同，`ResourceState` 多出 `IndirectArgument`、`DecompressionSource`、`DecompressionDestination`。

于是同一张映射表在仓库里有 **3 份**：

1. `VulkanSynchronization.cpp:5-21`（`ResourceState` → `VkImageLayout`）
2. `VulkanSynchronization.cpp:23-39`（`TextureLayout` → `VkImageLayout`，与 1 同构）
3. `Core/ResourceSynchronization.h:30-43`（`ResourceState` → `TextureLayout`）

**这不是「两个概念恰好相似」，而是同一个概念写了两遍。** 消费侧因此要在每个 barrier 调用点做双向翻译：`textureLayoutForResourceState()` 有 10 个文件 15 处调用，而 `.oldLayout = / .newLayout =` 的手工配对赋值在全仓有 66 处。`RenderGraphAccessPlan.cpp:267-268` 更典型——调用者拿到的本来就是 `ResourceState`，却先转 `TextureLayout` 塞进 barrier，backend 再转回 `VkImageLayout`。

值得注意的是 `VulkanNative.h:81` 已经暴露了正确方向：

```cpp
VkImageLayout nativeImageLayout(TextureView& view, ResourceState usage);
```

**语义 state 是主语，layout 是 backend 的推导结果**——这正是 A1 想要的形态，只是 `TextureBarrierDesc` 还没跟上。

**建议**：`TextureBarrierDesc.oldLayout/newLayout` 改为 `ResourceState`，删除 `TextureLayout` 与那两张手写表。`before/after` scope 已经在同一个结构体里，layout 完全可以由 state 推导。

**必须注意的陷阱**：`VulkanRHI.cpp:6047` 现在依赖 `TextureLayout` 的**枚举值顺序**做上界校验（`barrier.oldLayout > TextureLayout::General`）。`ResourceState` 把 `IndirectArgument` 插在 `ShaderRead` 与 `TransferSource` 之间，**序号不同**，所以不能 `static_cast` 互转——这恰恰说明必须做类型合并而不是加个转换函数。

**风险**：`TextureLayout` 是公开头里的公共类型，Source ~106 处 + tests ~140 处引用；`tests/rhi/harness/DiagnosticCases.cpp:78-85` 直接以该枚举为键验证 `imageLayout` 的映射，需要改写。这是本清单里调用方影响最大的一项，应单独成批、独立验证。

### 2.2 A3：24 值枚举把三个正交维度压成一个

`RenderGraphResourceAccess`（`RenderGraphTypes.h:60-85`）有 24 个值，但它表达的信息其实是三个正交维度的笛卡尔积：

```
{Texture, Buffer, AccelerationStructure} × {Read, Write, ReadWrite} × {Sample, Storage, Transfer, Indirect, Constant, Build}
```

这个结构可以在 `scopeForGraphAccess`（`RenderGraphAccessPlan.cpp:92-150`）里直接看到——大量 case 坍缩到同一个结果：

```cpp
 112:     case RenderGraphResourceAccess::TextureSampleRead:
 113:     case RenderGraphResourceAccess::TextureSampleReadGeneral:
 114:     case RenderGraphResourceAccess::BufferShaderRead:
 115:         return {shaderStages, AccessBits::ShaderRead};
 ...
 140:     case RenderGraphResourceAccess::TextureTransferRead:
 141:     case RenderGraphResourceAccess::BufferTransferRead:
 142:         return {PipelineStageBits::Transfer, AccessBits::TransferRead};
 143:     case RenderGraphResourceAccess::TextureTransferWrite:
 144:     case RenderGraphResourceAccess::BufferTransferWrite:
```

`Texture*` 与 `Buffer*` 的 transfer 版本**映射到完全相同的 `SyncScope`**，只因为资源类型不同就成了两个枚举值。转换损失是双份的：调用者要选对 24 个名字之一，而实现要为 24 个值各写一遍。

更重的是围绕它存在的 **11 处手写 switch**（7 处以 access 为键、4 处以 state 为键）：

| 位置 | 作用 | 分支数 |
|---|---|---|
| `RenderGraphNode.cpp:57-88` | `accessWrites` | 24 |
| `RenderGraphNode.cpp:120-158` | `stateForAccess` → `ResourceState` | 24 |
| `RenderGraphNode.cpp:160-196` | `textureUsageForAccess` | 24 |
| `RenderGraphNode.cpp:198-235` | `bufferUsageForAccess` | 24 |
| `RenderGraphNode.cpp:237-270` | `bufferViewTypeForField` | — |
| `RenderGraphNode.cpp:272-305` | `accessMatchesResourceType` | — |
| `RenderGraphAccessPlan.cpp:92-150` | `scopeForGraphAccess` → `SyncScope` | 24 |
| `RenderGraphNode.cpp:350-393` | `explicitAccessForState`（**反向**） | 2 个 switch |
| `RenderGraphComputeStages.cpp:26-41` | `validTextureState` | — |
| `RenderGraphExecutor.cpp:2297-2310` | `transition()` 内 state → `SyncScope` | — |

其中 `accessWrites()` 尤其能说明问题：`RenderGraphAccessPlan.cpp:176-181` 的 `captureGraphDeclaredAccess()` 已经从 `scope.access` 的位测试**推导**出读写，而 `RenderGraphNode.cpp:57-88` 又用手写表算同一件事。同一事实两种算法，正是分叉的来源。

`transition()`（`RenderGraphExecutor.cpp:2297-2310`）的三个分支与 `ResourceSynchronization.h:19,23,24` 的 `resourceSyncScope()` 位完全一致——这是**同一语义的第二份实现**，而 `resourceSyncScope` 就在手边（16 文件 32 处调用）。

**建议**（按增量风险排列）：

1. 零风险：`transition()` 改为复用 `resourceSyncScope()`，删掉那份本地 switch。
2. 低风险：`RenderGraphField` 只保留 `access`，`state`/`usage`/`bufferUsage`（`RenderGraphTypes.h:145-148`）改为按需派生——`applyAccessDefaults`（17 处调用）已经在做这件事，去掉冗余存储即可。
3. 中风险：把 24 值枚举换成 `{ResourceType, Operation, UsageBits}` 结构。收益是 7 处以 access 为键的 24 分支 switch 变成对 3 个正交维度的分别处理；代价是触及 shader 声明侧与 12 个测试文件。

第 3 项不应与 §3.3 的 ComputeProgram 收缩同批进行。

### 2.3 A2：三套「资源 → descriptor 身份」并存（迁移中，不要现在删）

生产代码里同时存在三处独立映射：

1. 设备级 `ResourceRegistry`：`RegistryEntry{kind, handle, value, allocation}`（`ResourceRegistry.cpp:28-34`），`acquire()`（`89-122`）在 `BindlessHeap` 上分配。
2. `RenderGraphExecutor` **自建第二个 heap** 并自己维护 `RenderGraphResource::bindlessHandle`：`RenderGraphExecutor.cpp:1644-1647` 调 `graphDevice.createBindlessHeap(...)`，`1662-1679` 对 sampled image 分配+写入，`1690-1703` 对 buffer 同样处理。这两个字段在 `RenderGraphTypes.h:267-268` 声明，又在 `471-472` 和 `RenderGraphExecutor.cpp:2348-2353` 各复制一份。
3. `GPUScene` 自有的 13 项 kind→lease 表（`GPUScene.h:309-376`、`GPUSceneSubsystem.cpp:1714-1760`）。

非 backend 的 `createBindlessHeap()` 生产调用点有 7 处（`VisibilityHybridRasterizer.cpp:29`、`BunnyWireframePass.cpp:102`、`ImageSamplePass.cpp:43`、`ResourceRegistry.cpp:178`、`SceneMaterialShaderObjectPass.cpp:133`、`RenderGraphExecutor.cpp:1644`、`StreamlineDLSSRRPass.cpp:832`）。

同时 `BindlessHandle{kind,index,shaderIndex}`（`RHI.h:1332-1351`）与 `ResourceLease`（`ResourceRegistry.h:43-53`）是同一概念的两个类型，且**索引存了两份**——`RegistryEntry::value` 与 `handle.shaderIndex` 在 `ResourceRegistry.cpp:202,229,263` 被赋成同一个值。访问方式也分叉：`VisibilityBufferPass.cpp:2065` 走 `ResourceLease::shaderIndex()` 方法，`RenderGraphBufferPasses.cpp:68` 走 `BindlessHandle::shaderIndex` 公有字段。

**但这条要谨慎定调。** HEAD 提交信息就是 "Refactor render graph execution and simplify pipeline APIs"，图私有 heap 很可能是**迁移进行中的中间态**，而不是遗留设计债。在 `SharedResourceRegistry.md` 的迁移完成前删它，会把「正在收敛」误判成「冗余」。

**建议**：不作为本次简化目标。改为登记为收敛项——图的 capture/回放（`Profiling/WorkControlReplay.cpp:228` 按 `productionHeap.desc()` 重建 heap）与 alias plan 是这条路的验收前提。真正低风险且立刻可做的是 `RenderGraphTypes.h:267-268` 的两个 handle 字段：它们在 `1678-1679` 被赋成同值，可以只留一个。

---

## 3. 代码量问题

### 3.1 B1：能力管线被枚举 7 次（1057 行）

同一批约 20 项设备能力在 `VulkanRHI.cpp` 中被独立枚举了 7 次，新增一项能力要同步改 7 处：

| # | 位置 | 角色 | 行数 |
|---|---|---|---|
| 1 | `1488-1583` | `VulkanExtensionSet` 字段 + `query()` 逐项查扩展名 | 96 |
| 2 | `1585-1636` | `VulkanDeviceFeatureRequest` 字段 + `from(DeviceDesc)` | 52 |
| 3 | `1637-1930` | `VulkanDeviceFeatureProbe` 成员 + `appendPNext` 链 + `supportsXxx()` | 294 |
| 4 | `1931-2188` | `VulkanDeviceFeatureSelection` 字段 + `select()` 逐项三元与 | 258 |
| 5 | `2189-2440` | `VulkanEnabledFeatureChain` 成员 + **第二条** `appendPNext` 链 | 252 |
| 6 | `2442-2546` | `enabledDeviceExtensions()` 的 `if (selection.x) push_back(...)` | 105 |

实测：`appendPNext(` 出现 41 次，`#ifdef VK_NV` 34 次，`#ifdef VK_EXT` 15 次。`VulkanDeviceFeatureProbe`（`1637-1685`）与 `VulkanEnabledFeatureChain`（`2189-2237`）的前 16 个字段**逐字重复**（相同类型、相同 `sType` 初始化器），只有成员名与用途（查询 vs 启用）不同。

**建议**：用声明表（X-macro 或 constexpr 表）把「能力名 → 扩展名 → 特性结构体类型 → 特性成员 → 选择标志」压成一张表，7 个枚举点改为遍历。预计 1057 行 → 约 350-450 行。

**保守做法**（如果不想动宏）：先把两个 16 字段同构结构提为一个共享基类，并把两条 `appendPNext` 链合并为一个「按 selection 逐项置 pNext」的函数，可删约 60-90 行。

**风险**：全部在匿名命名空间（`57-3039`）与文件内 `detail`，**调用方零改动**。但这段代码决定 `VkDeviceCreateInfo` 的 pNext 链顺序与扩展启用顺序，而 `VK_KHR_DEFERRED_HOST_OPERATIONS` 与 `VK_KHR_ACCELERATION_STRUCTURE` 之间存在顺序依赖（`2467-2470`）。pNext 顺序错误在部分驱动上会被验证层捕获，另一些驱动会静默忽略——**必须逐驱动回归，不能只看能否启动**。

### 3.2 B2：包装类样板 266 行，且「改成 = default」不成立

21 个公开包装类（`Queue`、`Fence`、`Semaphore`、`SwapchainSemaphore`、`Buffer`、`BufferView`、`TimestampQueryPool`、`RayTracingAccelerationStructureCompactionQueryPool`、`Texture`、`TextureView`、`ShaderModule`、`PipelineCache`、`GraphicsPipeline`、`ComputePipeline`、`GraphicsShaderObjectProgram`、`BindlessHeap`、`CommandBuffer`、`CommandPool`、`Swapchain`、`Device`、`RayTracingAccelerationStructure`）遵循同一模板：

`RHI.h` 中 126 行特殊成员声明（22 处 `~X();`、21 处移动构造、21 处移动赋值、20 处 `= default` 构造、42 处 `= delete`），`VulkanRHI.cpp` 中 140 行纯模板定义（21 处 4 行构造块、20 + 18 + 18 处 `= default`）。

**一个常见但在这里不成立的设想**：不能靠在头文件里写 `= default` 消除。`RHI.h:1388-1411` 只前置声明了 `detail::*Impl`，是**不完整类型**；`std::unique_ptr<Incomplete>` 的析构/移动只能在完整类型可见处实例化，所以 out-of-line 定义是当前 pimpl 形态的**必然结果**。

可选路径：

| 方案 | 做法 | 预计可删 | 风险 |
|---|---|---|---|
| 1 | `detail` 内 CRTP 基类持有 `impl_`，各类只声明自己的接口 | 100–130 行 | 中（改变 `impl_` 从直接成员为基类成员 → 对象布局变化） |
| 2 | 把 `detail::*Impl` 移入内部头，样板可在头内 `= default` | 200–260 行 | 中-高（打开公开头的 pimpl 边界） |
| 3 | 只合并 3 对同构实现（见下） | ~35 行 | **低** |

方案 3 已经 `Compare-Object` 验证为**零差异**：

- `SemaphoreImpl::~SemaphoreImpl()`（`3572-3580`）与 `SwapchainSemaphoreImpl::~SwapchainSemaphoreImpl()`（`3582-3590`）逐行相同。而且 `SwapchainSemaphore`（`RHI.h:1497-1517`）**没有任何公开方法**——它是个带 5 个 friend 的空壳。两者应合并为一个类型。
- `GraphicsPipelineImpl::~GraphicsPipelineImpl()`（`5326-5338`）与 `ComputePipelineImpl::~ComputePipelineImpl()`（`5362-5374`）函数体 11 行完全一致。

另外 `GraphicsPipelineImpl`（`3260-3268`）与 `ComputePipelineImpl`（`3270-3279`）的 6 个数据字段（`device`/`layout`/`pipeline`/`usesBindlessHeap`/`psoHash`/`pipelineCacheHit`）逐字相同，可提到 `PipelineCommon`。

### 3.3 B3：绑定号表按 Program 重复 5–6 份

`Core/NamedResourceLayouts.h`（434 行）为参数结构体定义了 12 张 `ComputeResourceField[]`。实测：

- 含 `offsetof(` 的字段行：**377**
- 不同的 `(结构体, 字段)` 组合：**122**
- 即 **约 68% 是重复声明**

而且重复不是「两个 Program 恰好用了同名字段」，是**同一个 `SceneResourceParameters` 字段被声明 4–6 次**：

```
6x  SceneResourceParameters.scene / .output / .vertices / .indices / .primitives / .instances
6x  SceneResourceParameters.materials / .materialTextures / .materialSampler
6x  SceneResourceParameters.ntcLatents / .ntcConstants / .ntcWeights / .ntcInfo / .ntcSampler
6x  SceneResourceParameters.rayStreamPages / .rayStreamPageTable / .rayStreamInstances / .rayStreamHeader / .streamPageCount
5x  SceneResourceParameters.lights / .lightsPdf / .regirGrid / .environmentMap / .environmentPdf / .shadowParameters
4x  （另有 50 余个字段各 4 次）
```

根因是**绑定号被绑定在 Program 上，而不是参数结构体上**。12 张表是同一个 122 字段结构体的 12 个近似子集；结构体加字段，就要在 4–6 张表里各加一行。这 12 张表被 10 个生产文件引用，`ComputeProgram.cpp:116-154` 还会为每个 program 把 layout 与 bindings 交叉校验一遍。

**建议**：把 binding id 变成参数结构体自身的 ABI 属性（一份 canonical layout），Program 只声明「用到哪些字段」。可去掉约 255 行重复声明与逐 program 的交叉校验。

**风险**：绑定号是 CPU 与 Slang 共享的 ABI（`ShaderResourceABI.h`、`ParameterRoot`），改动必须与 shader 及 12 个测试文件同批。**不要与 §3.4 的 ComputeProgram 收缩同时做。**

### 3.4 其他已确认的成对实现

`Compare-Object` 验证为零差异或可枚举差异的重复：

| 项 | 位置 | 重复量 | 风险点 |
|---|---|---|---|
| `makeHeapMapping` lambda 三份逐字相同 | `10334-10348`、`10579-10593`、`10889-10903` | ~45 行（含紧随的默认映射表） | 生成的 descriptor mapping 是 shader ABI，必须逐字节保持 |
| `DescriptorHeap` 三份 `write*Descriptor` 包装 | `2837-2858`、`2860-2879`、`2881-2902` | ~42 行 | 低 |
| AS / PartitionedAS writer 双份 | `1408-1435`、`1437-1460` | ~20 行 | 低 |
| 脏区间跟踪三份（6 个 min/max 成员 + 57 行方法） | `2909-2941`、`2943-2955`、`3001-3011`、`3031-3036` | ~40 行 | `vmaFlushAllocation` 范围算错是**静默错误**（无 VUID） |
| 执行状态失效 11 条赋值 ×2（归一化前缀后 0 差异） | `5790-5800`、`12025-12035` | ~32 行 | 9 个外部 SDK 调用点 |
| pipeline cache 解析 / layout 守卫 / teardown 三对 | `10270-10282`、`10552-10564` 等 | ~20-25 行 | 提取时必须保持 `unique_lock` 作用域 |
| 别名分配路径 | `9596-9697`、`9843-9930` | ~40 行 | 有回滚语义，必须保留「先全部挂 backing 再逐个绑定」的顺序 |
| `GeneratedCommands::updatePipelines/updateShaders` | `VulkanGeneratedCommands.cpp:309-339` | ~11 行 | 低 |
| `notifyExternalDescriptorSetBinding` 纯别名 | `VulkanRHI.cpp:12043-12048`、`12096-12099` | ~10 行 | 9 个调用点 + 1 个测试 |

**明确未发现重复的区域**（避免重复劳动）：`TextureView` 的 `createInfo()`（`3674-3687`）是 `VkImageViewCreateInfo` 的唯一构造点，被 `materialize()` 与 bindless 描述符写入 `writeImages`（`5605`）共用，设计良好；`VulkanTrace`、`VulkanSurfaceFormat`、`queueFamiliesForAccess`、`validScope`/`scopeInfo`/`toVkPipelineStages`/`accessFlags` 均为单一定义。

---

## 4. 一致性与隐患

### 4.1 C1：公开 API 有 3 种失败约定

`CommandBuffer` 的方法在「怎么报告失败」上不统一：

| 约定 | 例子 | 后果 |
|---|---|---|
| `Result<>` + 前置全量校验 | `synchronize()`（`5999`）、`copyBuffer()`（`6103`）、`dispatchIndirect()` | 调用者能在录制前拿到明确错误 |
| `Result<>` + 几乎不校验 | `decompressBuffers` / `validateDecompressionBuffers` | 有返回值但语义薄 |
| `void` + 静默返回 | `copyTexture()`（`6185`）、`copyTextureToBuffer()`、`copyBufferToTexture()`、`clearColorAttachment()` | 失败不可观测 |

`copyTexture()` 在 `6187-6196` 对 7 个条件做检查然后直接 `return`——**调用者无法区分「成功」和「因为源格式不匹配而什么都没做」**（`6200-6202` 还有一处 aspect 不匹配的静默返回）。同一个类的 `copyBuffer()` 却返回 `Result<>`。

**建议**：把 `copyTexture*`/`clearColorAttachment` 也改为 `Result<>`。这是**低风险的行为改善**（新增返回值不破坏调用点），收益是消除一整类静默失败。不建议反向把 `Result<>` 降为 `void`。

### 4.2 C2：VkResult 映射的第二份实现已经分叉

`VulkanRHI.cpp:98-121` 有正确、集中、含 `VK_ERROR_DEVICE_LOST → handleNsightAftermathDeviceLost()` 副作用的 `resultFromVk`（76 处调用）。但 `VulkanGeneratedCommands.cpp:10-21` 有第二份 `convertResult`：

```cpp
 10: Result<> convertResult(VkResult result)
 12:     if (result == VK_SUCCESS) { return {}; }
 13:     if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
 16:     if (result == VK_ERROR_DEVICE_LOST) { return makeError(Error::DeviceLost); }
 17:     if (result == VK_ERROR_FEATURE_NOT_PRESENT || ...) {
```

它缺少 Aftermath 钩子、缺少 `OutOfDate` 分类、多了 `VK_ERROR_FEATURE_NOT_PRESENT`。`VK_ERROR_OUT_OF_HOST_MEMORY` 在全仓只出现在这两处。

**这已经不是「重复」而是「已分叉」**：Generated Commands 路径上的设备丢失不会上报 Aftermath。删掉 `convertResult` 只省 12 行，**价值在于消除分叉**。

### 4.3 C3：同一 SPIR-V 内容哈希两套实现，结果不同

`detail::shaderContentHash`（`PipelineStateHash.cpp:48-55`）先哈希 `desc.spirv.size_bytes()` 再加字节：

```cpp
 48: uint64_t shaderContentHash(const ShaderModuleDesc& desc)
 50:     uint64_t hash = kFnvOffset;
 51:     hash = hashValue(hash, desc.spirv.size_bytes());
 52:     return ... ? hashBytes(hash, desc.spirv.data(), ...) : hash;
```

`VulkanRHI.cpp:10151-10156` 的匿名 `fingerprint` lambda 用同一组 FNV-1a 常量但**不含长度前缀**，写入 `inputSpirvFnv1a64`。两者同名含义却值不同。`PipelineCacheFile.cpp:24-25,42-50` 还有第三份 `hashBytes`。

**建议**：`fingerprint` 改用 `detail::shaderContentHash`，并在 `shaderContentHash` 的注释里把「是否含长度前缀」写成契约。**注意**：`inputSpirvFnv1a64` 被 `WorkControlReplay` 的证据与 `VulkanRHI.cpp:10802` 的诊断输出使用，改成含长度会改变既有回放记录的比对结果，需要同批更新证据格式说明。

### 4.4 C4：载入状态枚举两份

`PipelineCacheLoadStatus`（`RHI.h:1199-1204`）与 `PipelineCacheFileLoadStatus`（`PipelineCacheFile.h:17-22`）四个值同名同序，`VulkanRHI.cpp:3814-3836` 用手写 switch 逐值转换。内部类型，零风险，直接复用其一。

（`PipelineCacheFileIdentity` 与 `PipelineCacheStats` 字段互补，**不是**重复，无需合并。）

### 4.5 D1：魔法位数做枚举上界校验

`VulkanSynchronization.cpp:117`：

```cpp
117: if ((stages & ~((1ull << 15) - 1)) || (access & ~((1ull << 19) - 1)) || (!stages && access)) { return false; }
```

`PipelineStageBits` 用到位 0..14（15 位，`Host = 1ull << 14`），`AccessBits` 用到 0..18（19 位）——**今天是对的**。但只要有人给这两个枚举加第 16 / 第 20 个值，这个校验就会把**所有合法 scope 判为非法**，而且是运行期静默拒绝（返回 `InvalidArgument`），不是编译错误。

同类的 `VulkanRHI.cpp:6047` 用 `> TextureLayout::General` 做上界，依赖枚举值顺序——A1 的合并必须处理它。

**建议**：在枚举末尾加 `Count`/`Max` 哨兵，断言改为 `stages < (1ull << bitCount(PipelineStageBits::Count))`，并加 `static_assert` 保证 `Count` 与实际位数同步。这是**低改动、高价值**的正确性护栏，建议优先于任何重构先做。

### 4.6 D2：friend 密度与 `deviceIdentity()` 的配对校验

`RHI.h` 有 **94 个 friend 声明**，其中 `Texture` 8 个、`Buffer` 7 个、`Semaphore`/`Queue`/`TextureView` 各 5 个。这些 friend 存在的原因是包装类之间要互读 `impl_`——`VulkanRHI.cpp` 里 `x.impl_->` 形式出现 194 次，涉及 `buffer`(26)、`texture`(17)、`destination`(16)、`queryPool`(15) 等 16 个不同对象。

`deviceIdentity()` 是这条边界的**公开出口**：7 个类型暴露它，全仓 **48 处**调用，几乎全是同一个模式：

```cpp
if (!impl_ || !recording_ || view.deviceIdentity() != deviceIdentity())  // VulkanRHI.cpp:6390
if (... && commands.deviceIdentity() != impl_->tlas->deviceIdentity())   // SceneAccelerationStructure.cpp:1261
if (bound->buffer.deviceIdentity() != commandBuffer().deviceIdentity())  // RenderGraphComputeStages.cpp:89
```

同一个设备配对校验在 30+ 处被重新写一遍。

**建议**（不是删 friend，而是让 friend 有单一出口）：把跨包装类的 `impl_` 互访收敛到 `detail` 命名空间内的访问器（如 `detail::impl(const Buffer&)`），每个类只需 friend 一个访问器结构体，可把 94 个 friend 压到约 25 个；同时给 `deviceIdentity()` 的比对加一个 `sameDevice(a, b)` 辅助，把 30+ 处配对校验变成单点。

**顺带一项**：`CommandSubmissionContext`（`CommandSubmission.h:59-67`）是**纯虚接口 + 唯一实现** `FrameSubmissionContext`（`RenderFrameContext.cpp:105`），而且 `RenderFrameContext.cpp:143` 必须用 `dynamic_cast` 才能把它转回具体类型找回 `frame`：

```cpp
143: const auto* context = dynamic_cast<const detail::FrameSubmissionContext*>(commands.submissionContext().get());
```

虚接口在这里的作用只是打破 `RenderFrameContext ↔ RHI` 的循环依赖。用**具体的回调结构体**（存 `std::function` 或函数指针 + 不透明 `void*`）能达到同样的解耦，同时省掉 vtable 与 `dynamic_cast`。这是「抽象只服务一个实现」的典型，值得单独评估。

---

## 5. E1：零外部调用点的公开 API

对 `RHI.h` 里 173 个公开方法按「真实调用表达式」（`.name(` / `->name(` / `::name(`，排除 `RHI.h` 与 `VulkanRHI.cpp` 自身）统计，**13 个方法零外部调用点**：

| 类型 | 方法 | 声明 | 判定 |
|---|---|---|---|
| `BindlessHeap` | `allocateAccelerationStructure` | `RHI.h:1946` | 全仓零调用 |
| `BindlessHeap` | `writeConstantBuffer` | `RHI.h:1954` | 全仓零调用（实现存在：`VulkanRHI.cpp:5649`） |
| `BindlessHeap` | `writeSamplers` | `RHI.h:1949` | 仅内部被 `writeSampler` 转发（`5496`）→ 应转 private |
| `BindlessHeap` | `writeImages` | `RHI.h:1952` | 仅内部被 `writeSampledImage`/`writeStorageImage` 转发（`5561`/`5571`）→ 应转 private |
| `BufferSlice` | `subslice` | `RHI.h:1555` | 全仓零调用 |
| `BufferSlice` | `validateData` | `RHI.h:1559` | 全仓零调用 |
| `CommandBuffer` | `drawMeshTasks` | `RHI.h:2048` | 全仓零调用（只用 `drawMeshTasksIndirect`，2 处） |
| `CommandBuffer` | `setDepthStencilState` | `RHI.h:2038` | 全仓零调用 |
| `PipelineCache` | `filePath` | `RHI.h:1811` | 全仓零调用（`stats()` 有用） |
| `RayTracingAccelerationStructure` | `supportsQueueAccess` | `RHI.h:1699` | 全仓零调用（注意不是 `Buffer` 的同名用法） |
| `Semaphore` | `signal` | `RHI.h:1482` | 全仓零调用（实现存在于 `VulkanRHI.cpp:4899`） |
| `TextureView` | `hasNativeView` | `RHI.h:1763` | 全仓零调用 |
| `TextureView` | `prepareNative` | `RHI.h:1762` | 全仓零调用 |

`TextureView` 这一对尤其值得注意：RhiSimplificationResearch.md 把「lazy native view」列为方向，`VulkanNative.h:82` 也暴露了 `nativeImageView(TextureView&)`，但**消费侧从不调用 `prepareNative()`/`hasNativeView()`**——物化由 backend 内部隐式完成（`3674-3696` 的 `createInfo()` 被 `materialize()` 与 `writeImages` 共用）。也就是说「语义视图 vs 原生视图」在公开 API 上**目前没有用户**，要么让它成为显式契约（文档中的方向），要么承认物化是隐式的、收掉这两个方法。

`BindlessHeap` 的 4 个方法里 2 个是内部转发、2 个零调用——这个类的公开面（`RHI.h:1942-1960`，9 个方法）与实际使用严重不匹配。

**建议**：这不是「删代码」而是「删契约」。批量 API（`writeImages`/`writeSamplers`）转 private；零调用的 `writeConstantBuffer`/`allocateAccelerationStructure`/`subslice`/`validateData`/`setDepthStencilState`/`supportsQueueAccess`/`filePath` 评估删除，`drawMeshTasks`/`prepareNative`/`hasNativeView` 先确认是否为**测试或未来路径刻意保留**（若是，在声明处加注释说明保留理由，避免下次评审重复排查）。

---

## 6. 建议的实施顺序

按「风险从低到高、每步可独立验证」排序。**不建议把这些并成一个大改动**——按仓库既有约定，共享 RHI 接口变更需要同时检查 runtime 调用者与测试。

| 阶段 | 内容 | 涉及 | 验证 |
|---|---|---|---|
| 0 | D1 魔法位数护栏 + C4 枚举合并 + C2 删 `convertResult` | 内部，零调用方 | 编译 + `MetallicRHITests` |
| 1 | C1 把 `copyTexture*`/`clearColorAttachment` 改为 `Result<>` | 调用点需处理返回值 | 图形冒烟 + `RenderingTests` |
| 2 | C3 hash 统一；§3.4 的低风险成对合并（`updatePipelines/updateShaders`、`notifyExternalDescriptorSetBinding`、DescriptorHeap 写入包装） | backend 内部 | descriptor heap 写入后立即使用的场景，不能只跑启动 |
| 3 | B2 方案 3（3 对同构析构）→ 再评估方案 1（CRTP） | `detail` 内部 / 公开头 | 全量重编 |
| 4 | E1 公开面收缩 | `RHI.h` | 全量重编 |
| 5 | A3 第 1、2 步（`transition()` 复用 `resourceSyncScope`；`RenderGraphField` 去冗余存储） | RenderGraph | 需先补「同一 state 两种推导结果一致」的断言 |
| 6 | B1 能力管线表驱动（先做 16 字段结构去重，再评估 X-macro） | 匿名命名空间 | **逐驱动**回归 pNext 链，不能只看能否启动 |
| 7 | B3 绑定表去重 | 需 shader 同批 | GPU 回读等价用例 |
| 8 | A1 删除 `TextureLayout` | 公开契约，~246 处引用 | 独立批次，图形冒烟必须包含输出图像检查 |
| — | A2 图私有 heap 收敛 | **等 `SharedResourceRegistry.md` 迁移完成** | capture/回放 + alias plan |

**A1 与 B1 是两条最值得投入的线**：A1 消除的是「同一概念三个名字」这一整套认知负担，B1 消除的是「加一项能力改 7 处」这一整套机械劳动。两者都不改变外部行为，都可以只靠编译与既有测试验证。

---

## 7. 本轮未做的事（局限声明）

- **未编译、未运行任何测试、未做 GPU 测量。** 文中行数是静态计数（`Get-Content`/`Compare-Object`/正则匹配），「可删行数」是计数差值，不是编译后实测。
- 未运行 `ctest`，未做图像验证或输出对比。因此本文**不含**任何性能结论、视觉正确性结论或帧时间影响判断。
- A2（图私有 heap）按 HEAD 提交信息判断为**迁移中间态**，本轮不建议改动；该判断基于 commit subject，未核对迁移任务的完整计划。
- `Semaphore::signal` 的零调用点已全仓复核（`.signal(` / `->signal(` / `::signal(` 只命中 `VulkanRHI.cpp:4899` 的定义本身）。
- DLSS-NR 的 Width/Height 多别名（`VulkanDLSSNR.cpp:421-432`，35 处 `p->Set(` 里 10 个是 Width/Height 变体）**未列入建议**：NGX 参数名由 SDK 决定，很可能是跨版本兼容而刻意保留，仓库内无判定依据。其余「疑似」项（legacy slot ABI 是否仍被 shader 读取、`PreparedBindings` 是否等价于 `EncodedParameters`、`ResourceRegistry::heap()` 借出往返是否对应 SDK 失效边界）需要 SDK 侧证据。
- 摸底材料：`.cache/rhi-review/backend-findings.md`（后端冗余证据清单）与 `.cache/rhi-review/consumer-findings.md`（消费侧重叠证据清单）。二者为过程证据，非交付文档。
