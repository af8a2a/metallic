# Metallic RHI 设计评审：抽象简化与冗余消除

日期：2026-10-03（第 2 版，修订至 `3e6f9d96a`）。第 1 版基于 `7226aa64d`，已随 `6c72f2cd4` 入库。

**修订说明**：第 1 版的结论已被实施。`7226aa64d` 之后的 6 个提交完成了其中三项主要建议，并各自附带测试与文档更新：

| 提交 | 时间 | 内容 | 对应第 1 版结论 |
|---|---|---|---|
| `6c72f2cd4` | 14:35 | Separate resource states from texture layouts | A1（state/layout 分离） |
| `6e7f9ab6a` | 15:25 | Remove numeric compute resource slots | B3 相邻 |
| `b89072fb5` | 15:48 | Move Vulkan options and capabilities out of RHI | B1 前置 |
| `9ca2cefc0` | 16:17 | Simplify shader warmup request handling | **B1**（能力目录，−1240 行） |
| `0dcb39e36` | 16:43 | Unify RHI wrapper lifecycle declarations with a handle macro | **B2**（包装类样板，−712 行） |
| `3e6f9d96a` | 16:59 | Cache Vulkan device properties at device creation | B1 收尾（−129 行） |

因此本文档不再重复已关闭的结论，只保留：**已验证关闭的项（含验证方式）**、**当前仍然成立的项**、以及一项对第 1 版判断的**更正**。

基线：`3e6f9d96a`。工作区除 `External/microprofile` 外无改动。

---

## 1. 已关闭并已验证

### 1.1 B1 能力管线七重枚举 —— 关闭

第 1 版指出同一批设备能力在 `VulkanRHI.cpp` 中被独立枚举 7 次共 1057 行。现由声明表取代：

- `Vulkan/DeviceFeatureCatalog.inl`（325 行）：一张 `MT_VK_FEATURE` / `MT_VK_SIMPLE` / `MT_VK_EXTENSION` / `MT_VK_NODE` / `MT_VK_DEPENDENCY` 宏表，行按依赖顺序排列，由 `VulkanDeviceFeatures.h`（326 行）多次展开。
- 实测：`appendPNext(` 在 `VulkanRHI.cpp` 中从 **41 → 0**；`#ifdef VK_NV` 从 34 → 2；该提交 `VulkanRHI.cpp` **−1082 行**（+79 / −1161），新增两个文件共 651 行，净减约 430 行但把 7 处枚举收敛成 1 张表。
- 验证：`tests/rhi/VulkanDeviceFeaturesTests.cpp`（364 行）覆盖目录展开；`tests/rhi/VulkanDevicePropertiesTests.cpp`（105 行）覆盖属性缓存。

**残留风险未变**：该表仍决定 `VkDeviceCreateInfo` 的 pNext 链顺序与扩展启用顺序（`VK_KHR_DEFERRED_HOST_OPERATIONS` 与 `VK_KHR_ACCELERATION_STRUCTURE` 的顺序依赖）。表驱动把「改 7 处」变成「改 1 处」，但没有消除**必须逐驱动回归**这一验证要求——pNext 顺序错误在部分驱动上被验证层捕获、在另一些驱动上静默忽略。

### 1.2 B2 包装类特殊成员样板 —— 关闭

21 个公开包装类原先各有「移动构造 + 移动赋值 + 析构 + 拷贝删除 + 私有 Impl 构造」五段样板（`RHI.h` 126 行 + `VulkanRHI.cpp` 140 行）。现由 `RHIHandle.h`（40 行）的两个宏统一：

- `METALLIC_RHI_HANDLE(Type, Storage, ...)`：声明段，含 `Storage` 参数以便 unique/shared 两种持有方式共用同一形态，末尾恢复 `public:` 供各类继续声明自己的资源 API。
- `METALLIC_RHI_HANDLE_DEFINITIONS(Type)` / `METALLIC_RHI_HANDLE_CONSTRUCTORS(Type)`：定义段，展开在 `Impl` 完整处。
- 实测：`RHI.h` 2227 → **1954** 行（−273）；`METALLIC_RHI_HANDLE(` 21 处；`RHI.h` 中已无 `explicit X(std::unique_ptr<detail::` 声明（0 处）与 `~X();` 声明（1 处，`MemoryBudgetReservation`）；`VulkanRHI.cpp` 中手工 `= default` 归零（20 处 `METALLIC_RHI_HANDLE_DEFINITIONS` + 1 处 `_CONSTRUCTORS`）。该提交本身 `RHI.h` −248 行、`VulkanRHI.cpp` −170 行，另加 `RHIHandle.h` +40 行。
- 验证：`tests/rhi/RHIHandleTests.cpp`（138 行）三个用例——`RHIEmptyHandle.SupportsOpaqueConstructionAndMoveOnlyLifetime`（编译期覆盖每类的空构造/移动/析构）、`RHIHandleOwnership.LiveUniqueMoveDestroysOldImplExactlyOnce`、`RHIHandleOwnership.SharedMovePreservesRetainedAllocationAfterWrapperDies`。后两个尤其关键：它们覆盖了宏最容易被写错的两处语义——移动赋值须释放目标旧资源、shared 持有须让已提交工作保住分配。

累计效果：`7226aa64d` → `3e6f9d96a` 期间 `VulkanRHI.cpp` 由 **12154 → 10766** 行（−1388），`RHI.h` 由 2227 → 1954 行（−273），同时新增 `VulkanDeviceFeatureCatalog.inl` / `VulkanDeviceFeatures.h` / `VulkanDeviceExtensions.h` / `VulkanDeviceProperties.*` / `RHIHandle.h` 共 6 个边界更清晰的文件与 3 个测试目标。

宏方案的优点是**保留每类独立的 friend 与 unique/shared 持有差异**，没有引入基类、虚函数或额外状态（`RHIHandle.h:6-10` 明确记录了这一取舍），因此第 1 版担心的对象布局变化与 `friend` 可见性回退都没有发生。

### 1.3 A1 state/layout 重复 —— 关闭，但第 1 版判断需更正

**更正**：第 1 版把 `TextureLayout` 判为 `ResourceState` 的「真子集 + 冗余镜像」，并建议删除 `TextureLayout` 统一到 `ResourceState`。`6c72f2cd4` 采纳了「消除重复」这一目标，但选择了**相反的方向**，并且这个方向更正确：

- `ResourceState` 移出 `RHI.h`，落到 `Core/ResourceState.h`（23 行），注释明确「Coarse renderer usage for graph/history tracking, **not an RHI image layout**」。
- `ResourceState` 保留 `IndirectArgument`、`DecompressionSource`、`DecompressionDestination`——这三个是 **buffer 专属**用途，在纹理布局里没有合法对应。第 1 版把这点当成「子集关系」的证据，实际上它恰恰证明两者**不是同一概念**：`ResourceState::General` 可能只读也可能可写，无法反推同步范围。
- `ResourceSynchronization.h:32` 现在显式写下这条不变量：`// Never derive access scopes from TextureLayout: General may be read-only or writable.`
- 后端只剩一个 `imageLayout(TextureLayout, bool unified)` 重载（`VulkanSynchronization.cpp:5`）；原先那个逐行同构的 `imageLayout(ResourceState, ...)` 已删除。`RHI.h` 中 4 个原先接收 `ResourceState` 的接口（`RenderingAttachmentDesc`、`BindlessImageWrite`、`writeSampledImage`、`clearColorTexture`）改用 `TextureLayout`。
- `Documentation/ProjectArchitecture.md:401-406` 把边界写成了契约：**RHI 不接受 `ResourceState`，也不从布局反推同步范围**。

所以第 1 版两处具体的「逐行同构」证据（两个 `imageLayout` 重载）确实被消除了，但「删掉一个枚举」的方向是错的——正确的收敛是**分层的单向转换**：Core 用 `resourceSyncScope()` 生成同步范围，在纹理边界用 `textureLayoutForResourceState()` 单向选择布局。本文档采纳该结论，不再建议合并这两个枚举。

---

## 2. 当前仍然成立的项

第 1 版按「可压缩行数」排序；实施完成后，剩余项的**性质变了**——主要是契约一致性与维护隐患，而非代码量。按风险重排如下。

### R1 【隐患】枚举上界校验依赖魔法位数，新增枚举值会静默失效

`VulkanSynchronization.cpp:99`：

```cpp
 99: if ((stages & ~((1ull << 15) - 1)) || (access & ~((1ull << 19) - 1)) || (!stages && access)) { return false; }
```

`PipelineStageBits` 用到位 0..14（`Host = 1ull << 14`），`AccessBits` 用到 0..18——**今天正确**。但给任一枚举加值就会让该校验把所有合法 scope 判为非法，且失败形式是**运行期静默返回 `InvalidArgument`**，不是编译错误。`6c72f2cd4` 等 6 个提交都没有触及这里。

这是剩余项里**唯一真正的正确性隐患**，也是改动量最小的一项：在枚举末尾加 `Count` 哨兵，断言改为 `static_cast<uint64_t>(stages) < (1ull << bitWidth<PipelineStageBits>())`，并加 `static_assert` 锁住位宽与枚举值数量同步。**建议优先于其余所有项先做。**

### R2 【一致性】`VkResult → Error` 第二份实现，且已行为分叉

`VulkanRHI.cpp` 的 `resultFromVk` 是集中实现（含 `VK_ERROR_DEVICE_LOST → handleNsightAftermathDeviceLost()`）。`VulkanGeneratedCommands.cpp:10-21` 仍有第二份 `convertResult`，**6 个提交都未处理**：

- 缺 Aftermath 钩子 → Generated Commands 路径的设备丢失不上报；
- 缺 `OutOfDate` 分类；
- 多一个 `VK_ERROR_FEATURE_NOT_PRESENT`。

`VK_ERROR_OUT_OF_HOST_MEMORY` 在全仓只出现在这两处。删 `convertResult` 只省约 12 行，**价值在消除分叉**。需要一个后端内部头暴露 `resultFromVk`。

### R3 【一致性】同一 SPIR-V 内容哈希两套实现，结果不同

- `detail::shaderContentHash`（`PipelineStateHash.cpp:48-55`）先哈希 `spirv.size_bytes()` 再加字节；
- `VulkanRHI.cpp:8874-8880` 的匿名 `fingerprint` lambda 用同一 FNV-1a 常量但**不含长度前缀**，写入 `inputSpirvFnv1a64`。

两者同名含义却值不同，且 `inputSpirvFnv1a64` 被 `VulkanRHI.cpp:9516` 的诊断输出与 `WorkControlReplay` 的证据比对使用。统一时需同批更新证据格式说明。

### R4 【一致性】`void` 与 `Result<>` 两套失败约定并存

`CommandBuffer` 内部自相矛盾（`RHI.h:1792-1802`）：

| 签名 | 方法 | 后果 |
|---|---|---|
| `[[nodiscard]] Result<>` | `copyBuffer`、`synchronize`、`bindExecution` | 调用者能拿到明确错误 |
| `Result<>`（无 nodiscard） | `decompressBuffers` | 有返回值，语义薄 |
| `void` | `copyTexture`、`copyTextureToBuffer`、`copyBufferToTexture`、`clearColorAttachment` | **失败不可观测** |

`copyTexture` 对 7 个条件做检查后直接 `return`，调用者无法区分「成功」与「因格式不匹配而什么都没做」。改 `void → Result<>` 是**新增返回值**，不破坏既有调用点，收益是消除一整类静默失败。不建议反向降级。

### R5 【一致性】载入状态枚举两份同名值

`PipelineCacheLoadStatus`（`RHI.h`）与 `PipelineCacheFileLoadStatus`（`PipelineCacheFile.h:17-22`）四个值同名同序，`VulkanRHI.cpp:3814-3836` 用手写 switch 逐值转换。内部类型，零风险。

（`PipelineCacheFileIdentity` 与 `PipelineCacheStats` 字段互补，**不是**重复。）

### R6 【可删契约】13 个公开方法零外部调用点

按真实调用表达式（`.name(` / `->name(` / `::name(`，排除 `RHI.h` 与 `VulkanRHI.cpp` 自身）统计，`RHI.h` 的 173 个公开方法中**仍有 13 个**零外部调用点（与第 1 版一致，仅行号变化）：

| 类型 | 方法 | RHI.h 行 | 判定 |
|---|---|---|---|
| `BindlessHeap` | `allocateAccelerationStructure` | 1725 | 全仓零调用 |
| `BindlessHeap` | `writeConstantBuffer` | 1733 | 全仓零调用 |
| `BindlessHeap` | `writeSamplers` | 1728 | 仅内部被 `writeSampler` 转发 → 应转 private |
| `BindlessHeap` | `writeImages` | 1731 | 仅内部被 `writeSampledImage`/`writeStorageImage` 转发 → 应转 private |
| `BufferSlice` | `subslice` | 1487 | 全仓零调用 |
| `BufferSlice` | `validateData` | 1491 | 全仓零调用 |
| `CommandBuffer` | `drawMeshTasks` | 1816 | 全仓零调用（只用 `drawMeshTasksIndirect`） |
| `CommandBuffer` | `setDepthStencilState` | 1806 | 全仓零调用 |
| `PipelineCache` | `filePath` | 1637 | 全仓零调用（`stats()` 有用） |
| `RayTracingAccelerationStructure` | `supportsQueueAccess` | 1577 | 全仓零调用 |
| `Semaphore` | `signal` | 1436 | 全仓零调用（实现存在于 `VulkanRHI.cpp`） |
| `TextureView` | `hasNativeView` | 1616 | 全仓零调用 |
| `TextureView` | `prepareNative` | 1615 | 全仓零调用 |

`TextureView` 这一对值得单独定调：`ProjectArchitecture.md` 与 `RhiSimplificationResearch.md` 都把「lazy native view」当作方向，`VulkanNative.h:82` 也暴露了 `nativeImageLayout(TextureView&, TextureLayout)`，但**消费侧从不调用 `prepareNative()`/`hasNativeView()`**——物化由 backend 内部隐式完成。要么让它成为显式契约，要么承认物化是隐式的、收掉这两个方法。

**建议**：批量 API（`writeImages`/`writeSamplers`）转 private；其余评估删除。若 `drawMeshTasks`/`prepareNative`/`hasNativeView` 是为测试或未来路径刻意保留，**在声明处加一行注释说明保留理由**，避免下次评审重复排查。

### R7 【架构，近期不宜动】24 值图访问枚举 + 7 处手写 switch

`RenderGraphResourceAccess`（`RenderGraphTypes.h:60-85`）24 个值，表达的是 `{资源类型} × {操作} × {用途}` 三个正交维度的笛卡尔积。围绕它仍有 7 处以 access 为键的手写 switch：

- `RenderGraphNode.cpp:58` `accessWrites`、`:121` `stateForAccess`、`:161` `textureUsageForAccess`、`:199` `bufferUsageForAccess`、`:273` `accessMatchesResourceType`
- `RenderGraphAccessPlan.cpp:92` `scopeForGraphAccess`
- `RenderGraphNode.cpp:351` `explicitAccessForState`（反向）

证据最强的两处：`scopeForGraphAccess` 里 `TextureTransferRead` 与 `BufferTransferRead` 映射到**完全相同**的 `SyncScope`，仅因资源类型不同就是两个枚举值；而 `accessWrites` 用手写表算读写，`RenderGraphAccessPlan.cpp` 的 `captureGraphDeclaredAccess` 又从 `scope.access` 位测试推导同一件事——同一事实两种算法。

**但第 1 版把它列为「架构级」现在看是过度定级。** `6e7f9ab6a`（Remove numeric compute resource slots）与 `6c72f2cd4` 已经动了这个区域，此时再改 24 值枚举会与进行中的迁移冲突，且触及 shader 声明侧与多个测试文件。

**建议降级为登记项**，只做零风险的一步：`RenderGraphExecutor.cpp` 的 `transition()` 内 state → `SyncScope` 分支（`IndirectArgument`/`DecompressionSource`/`DecompressionDestination` 三支）与 `ResourceSynchronization.h:20,24,25` 的 `resourceSyncScope()` 位完全一致，改为复用后者即可删掉那份本地 switch。

### R8 【架构，迁移中】三套「资源 → descriptor 身份」并存

`ResourceRegistry`（设备级）、`RenderGraphExecutor` 自建第二个 heap（`RenderGraphExecutor.cpp:1646` 仍调 `graphDevice.createBindlessHeap(...)`）、`GPUScene` 的 13 项 kind→lease 表（`GPUScene.h:309-376`）三者并存。

**不建议现在动**：`SharedResourceRegistry.md` 的迁移尚未完成，图私有 heap 是中间态，现在删会把「正在收敛」误判成「冗余」。真正低风险且立刻可做的是 `RenderGraphTypes.h` 中那两个 handle 字段——它们被赋成同值，可以只留一个。

### R9 【架构，语义存疑】唯一实现的纯虚接口 + `dynamic_cast` 回取

`CommandSubmissionContext`（`CommandSubmission.h:61-66`）有 5 个纯虚函数，唯一实现是 `FrameSubmissionContext`（`RenderFrameContext.cpp:105`）；且 `RenderFrameContext.cpp:143` 必须 `dynamic_cast` 才能把它转回具体类型找回 `frame`。虚接口在这里的作用只是打破 `RenderFrameContext ↔ RHI` 的循环依赖——用具体的回调结构体（函数指针 + 不透明 `void*`）能达到同样解耦，省掉 vtable 与 `dynamic_cast`。

**但注意**：RHI 确实需要在提交路径上回调 renderer 的 `reserveResources`/`acceptResources`，抽象本身有理由。建议是**把接口具体化**，不是删除解耦。

### R10 【代码量，未变】绑定号表按 Program 重复 5–6 份

`Core/NamedResourceLayouts.h`（434 行，`6e7f9ab6a` 未触及）实测：377 个 `offsetof` 字段行，只有 **122** 个不同的 `(结构体, 字段)` 组合，即 **约 68% 重复**。重复形式是同一个 `SceneResourceParameters` 字段被声明 4–6 次：

```
6x  scene / output / vertices / indices / primitives / instances / materials / materialTextures / materialSampler
6x  ntcLatents / ntcConstants / ntcWeights / ntcInfo / ntcSampler
6x  rayStreamPages / rayStreamPageTable / rayStreamInstances / rayStreamHeader / streamPageCount
5x  lights / lightsPdf / regirGrid / environmentMap / environmentPdf / shadowParameters
4x  （另有 50 余个字段各 4 次）
```

根因是**绑定号绑定在 Program 上而非参数结构体上**：12 张表是同一个 122 字段结构体的 12 个近似子集。可去掉约 255 行重复声明。

**风险**：绑定号是 CPU 与 Slang 共享的 ABI，改动需与 shader 及测试同批。`6e7f9ab6a` 已移除数字 slot 上传路径，这项现在是该 ABI 收敛的自然下一步，但应独立成批。

### R11 【代码量，未变】`friend` 密度与 `deviceIdentity()` 配对校验

`RHI.h` 仍有 **91** 个 friend 声明（`Texture` 8、`Buffer` 7、`Semaphore`/`Queue`/`TextureView` 各 5）。`deviceIdentity()` 全仓 **49** 处调用，几乎全是同一模式：

```cpp
if (... && commands.deviceIdentity() != impl_->tlas->deviceIdentity())
if (bound->buffer.deviceIdentity() != commandBuffer().deviceIdentity())
```

建议：跨包装类的 `impl_` 互访收敛到 `detail` 内的访问器（每类只 friend 一个），并把 30+ 处设备配对校验抽为 `sameDevice(a, b)` 单点。

---

## 3. 第 1 版已被否证的建议

为避免后来者重复提出，明确记录：

| 第 1 版建议 | 现状 | 否证依据 |
|---|---|---|
| 删除 `TextureLayout`，统一到 `ResourceState` | **已否证**，选择反向分离 | `ResourceState` 含 buffer 专属用途（`IndirectArgument`/`Decompression*`），且 `General` 无法反推同步范围；`ResourceSynchronization.h:32` 与 `ProjectArchitecture.md:401-406` 已写成契约 |
| 用 CRTP 基类消除包装类样板 | **未采用**，改用宏 | 宏保留每类独立 friend 与 unique/shared 差异，不改变对象布局（`RHIHandle.h:6-10`） |
| `= default` 无法就地消除样板 | **成立**，且被绕过而非否定 | 宏的定义段仍展开在 `Impl` 完整处，符合该约束 |

---

## 4. 建议顺序

| 阶段 | 内容 | 验证 |
|---|---|---|
| 1 | **R1** 枚举哨兵 + `static_assert` | 编译 + `MetallicRHITests` |
| 2 | **R2** 删 `convertResult` 统一映射 | Aftermath 场景回归（行为会改善：设备丢失开始上报） |
| 3 | **R5** 载入状态枚举合并 | 编译 |
| 4 | **R3** 统一 SPIR-V 哈希 | 同批更新 `WorkControlReplay` 证据格式 |
| 5 | **R4** `copyTexture*`/`clearColorAttachment` 改 `Result<>` | 图形冒烟 + `RenderingTests` |
| 6 | **R6** 公开面收缩（先转 private 批量 API，再逐个定调） | 全量重编 |
| 7 | **R7** 第一步：`transition()` 复用 `resourceSyncScope()` | 需先补「同一 state 两种推导一致」断言 |
| 8 | **R10** 绑定表去重 / **R9** 接口具体化 / **R11** friend 收敛 | 各自独立成批 |
| — | **R8** 图私有 heap | 等 `SharedResourceRegistry.md` 迁移完成 |

---

## 5. 局限声明

- 本文档是**静态代码审查**。**未编译、未运行测试、未做 GPU 测量。**文中行数为 `Get-Content` / `Compare-Object` / 正则计数结果，「可删行数」是计数差值而非实测收益。
- 第 1 版的「已关闭」判定依据是**代码与测试的存在**，不是运行结果。`RHIHandleTests` / `VulkanDeviceFeaturesTests` / `VulkanDevicePropertiesTests` 由实施提交附带，本轮**未执行**它们，因此「关闭」指结构上已消除，不代表行为已验证。建议在具备 GPU 环境下按 `AGENTS.md` 的验证边界补跑，尤其是：descriptor heap 写入后立即使用、逐驱动 pNext 链回归、以及 `--rhi-validation` 下的同步用例。
- **R7/R8/R9** 的定级基于「迁移进行中」这一判断，该判断来自提交序列与既有文档，未核对完整迁移计划。
- DLSS-NR 的 Width/Height 多别名（`VulkanDLSSNR.cpp`，35 处 `p->Set(` 中 10 个是 Width/Height 变体）**未列入建议**：NGX 参数名由 SDK 决定，很可能是跨版本兼容而刻意保留，仓库内无判定依据。
- 摸底材料：`.cache/rhi-review/backend-findings.md`、`.cache/rhi-review/consumer-findings.md`、`.cache/rhi-review/descriptor-mode-convergence.md`。三者是过程证据，非交付文档；其中前两者的行号基于 `7226aa64d`，已部分失效。
