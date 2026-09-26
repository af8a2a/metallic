# Shared registry、typed 参数与子系统资源身份

## 已接入的路径

`Device::resourceRegistry()` 提供设备级共享 registry。`ComputeKernel` 只保存 shader、pipeline 与参数 ABI；没有 Program 私有 heap、binding layout 或 resource table。

生产路径 `MaterialBinning` 的 reset/classify/arguments 三个 kernel 已迁移：使用 `MaterialBinningParams` 的命名字段，三个 dispatch 复用一个不可变参数包。`VisibilityMaterialBinning.slang` 直接解析字段里的 typed descriptor handle，删除了这条路径的数字 slot 配置。

新增 typed 间接消费测试让两个消费 permutation 复用生产分桶的 bins/tiles/arguments 身份，执行真实 GPU indirect dispatch 与逐像素回读；同时保留原有 legacy 消费测试，验证两套接口混用时 heap 切换正常。

第二批将 GPUScene、meshlet streaming、VisibilityBuffer raster 和 NRD 接入同一个设备 registry：

- GPUScene 上传后发布 `ResourceLease`，移除额外 `BufferView` 和 consumer 私有 descriptor 分配。`createBindings()` 只复制当前 generation/revision 对应的 leases。
- `MeshletStreamRuntime` 与两个 raster 消费者不再创建私有 heap。resident/stream 共用材质纹理索引、材质 remap buffer 和 tessellation buffer，删除第二套上传及重映射。
- `ComputeProgram` 的 Core 资源表路径成为兼容适配器，ScenePathTrace、deferred shading 等现有调用者复用 registry 身份。只有显式 `usesResourceTable=false` 的旧 SPIR-V mapping 诊断路径保留私有表。
- NRD 使用 `ComputeKernel` / `ParameterWriter`，取消 sampler/image slot 池与逐帧游标。SDK 调度计划、历史状态、barrier 和取消恢复逻辑仍由 NRD 适配层管理；常量与资源索引存入同一提交参数区，不再借用 Streamer 常量缓冲。

**迁移边界：** ComputeProgram 的命名 typed 参数尚未全面替换数字 slot；raster/stream 的既有 push struct 也继续使用原 ABI。共享的是资源身份、descriptor 分配和提交期所有权。BDA buffer API、同步模型、NRC/DLSS 原生 SDK 资源包装和 lazy native view 不在本批范围。

## 参数与资源契约

```cpp
std::shared_ptr<ResourceRegistry> registry;
auto result = device.resourceRegistry(registry);
if (!result) { return result; }

ParameterWriter writer(device, *commands.frameContext(), *registry);
MyParams params{
    .input = writer.buffer(inputBuffer),
    .output = writer.storageImage(outputView),
    .width = width,
};
EncodedParameters encoded;
result = writer.encode(params, kMyParamsAbi, encoded);
if (!result) { return result; }
return kernel.dispatch(commands, encoded, groupsX, groupsY);
```

- CPU 的 `ShaderBuffer` / `ShaderSampledImage` / `ShaderStorageImage` / `ShaderSampler` / `ShaderAccelerationStructure` 是不同类型，wire 大小均为 8 字节，不再暴露 CPU allocator handle。
- Slang 导入 `ParameterRoot`，通过 `getParameters<MyParams>()` 访问根地址；普通资源用 `DescriptorHandle<T>` 和 `Metallic::resolveDescriptor`。AS 保留现有完整设备地址 resolver 契约，不改变 Native 正规化规则。
- ABI 包含调用者维护的版本 ID、大小和对齐。C++ 参数必须是 standard-layout、trivially-copyable；跨 C++/Slang 的字段偏移仍需显式断言及 GPU 测试，未引入自动反射生成器。`MaterialBinningParams` 已检查大小及首个标量偏移。
- 每个 packet 的资源必须由同一个 writer 注册，或通过 `writer.use(lease)` 引入同 registry 的 lease。不要把裸 wire handle 从另一个 writer/registry 复制后直接 encode；POD 参数不是能自动追踪任意字段的反射系统。
- `sampledImages()` 上传不可变 handle 数组；不同元素不需要连续 descriptor slot。当前空数组会返回错误，应由调用者明确处理缺省场景。
- Packet 只可在创建它的同一次 frame recording 使用。跨帧、已结束的 command buffer、错误 device 和错误 ABI 被拒绝。

## 保留与回收

Registry 的 cache 使用 allocation identity 和规范化视图描述（format/mip/layer/swizzle/layout），只弱引用资源。ResourceLease 强引用原生分配；TextureView 自身也持有图像分配，避免依赖可移动的 Texture 包装对象。

参数包保留资源 leases、参数 chunk 及数组 chunk。成功录制 dispatch 时，frame 保留参数包和 kernel 实现；间接 dispatch 额外保留 arguments 分配。因此调用者可以在录制之后销毁或替换原包装对象、清空 kernel，已录制工作仍拥有旧资源。

回收遵循现有 `RenderFrameContext`：未提交录制取消后释放；已提交工作在 frame 重置/复用时、确认所有队列完成后释放；部分成功的提交不能整体当作取消。Arena chunk 只在对应 completion 完成后复用，增长会增加 chunk，不搬迁已有 GPU 地址。

TLAS lease 只持有 TLAS 分配，不会自动发现其引用的 BLAS/场景资源。使用 `writer.retain(sceneOwner)` 保留完整场景快照；partitioned AS 也使用分配 lease，但其间接引用仍由场景所有者负责。外部 SDK 资源不自动获得 registry 所有权。所有 registry/packet/lease/frame 仍要求 Device 最后销毁。借用的 swapchain 图像不能通过 registry 获得拥有型 lease，当前会拒绝注册。

资源状态、访问范围和队列依赖继续由原有 RenderGraph/barrier/submit 契约管理；lease 解决存活期，不会推断读写 hazard。Registry 注册受 mutex 保护，但同一 RenderFrameContext 的 CPU 录制仍需遵循现有串行约束。

## 容量与开销

默认 epoch 固定为 64 samplers、8192 sampled images、1024 storage images、8192 buffers。Live slot 不重写、不搬迁；耗尽时回收已失去全部 owners/leases 的条目，仍不足则明确返回 OutOfMemory。没有隐藏的 heap resize 或重新编号。

正常 cache hit 只查询相应 key；不会在每个资源注册时扫描整张 cache。显式 `collect()` 或容量不足时回收。相同 sampler 描述一直缓存到 registry 销毁，其容量也固定。

`stats()` 暴露 descriptor 写入次数、cache hits、live descriptors、参数累计上传字节和 arena 保留容量。第一批测试断言跨 kernel 引用分桶结果只允许为新 output 增加一个 descriptor；第三批 BDA 迁移后，该消费者不再增加 buffer descriptor。这是减少重复 descriptor 写入的机制证据，不是 CPU/GPU 帧时间提升测量。

## 回归入口

```powershell
build/tests/MetallicRhiTests.exe --rhi-validation '--gtest_filter=*registry_*:*material_binning*'
$env:METALLIC_SLANG_DESCRIPTOR_MODE = 'native'
build/tests/MetallicRhiTests.exe --rhi-validation '--gtest_filter=*registry_*:*material_binning*'
```

新用例覆盖容量耗尽与安全复用、相同 view 去重、错误 registry、跨 kernel 复用、GPU 参数不可变性、参数区增长、双帧重叠、wrapper/pipeline 提前销毁、图像 handle 数组、取消以及多队列部分提交。

扩大回归时 `frame_self_submit_two_slots` 在 `independent copy branch was blocked by the graphics branch` 处失败。以 HEAD `470982feae846325d065296a4b737f4363e30f52` 的原始代码构建同条件对照，复现相同失败；不是本批新增回归。原始日志为 `.cache/rhi-registry/two-slot-baseline.log`，原始代码构建日志为 `baseline-build.log`。对照结束后已逐字节恢复本批修改并重新构建。

2026-09-26 最终验证：`MetallicRhiTests` 与 `Metallic` Debug 完整构建通过；排除上述已确认基线失败后，Mapped 的 28 项定向/既有回归全部通过，Native 的 6 项 registry/材质分桶测试全部通过，无 Vulkan VUID 告警。日志在 `.cache/rhi-registry/final-mapped.log` 与 `final-native.log`。未做编辑器交互、DLSS 场景切换或帧时间 A/B 测量。

## 第二批兼容 ABI 与刷新契约

Core 的资源 slot 改为 16 字节 `{uint64 handle; uint64 arrayAddress;}`，根 push 仍为两个 64 位地址。标量直接读取 canonical handle；数组读取参数区里的显式句柄列表，支持重复纹理、非连续 slot 以及不同程序复用同一纹理。所有引用 Core 的 shader 随源依赖重新编译；手工读取 `gComputeResources.resources` 的代码必须使用 `.handle` 字段。

帧录制中的兼容参数数据复用 `ParameterWriter` arena。没有 `RenderFrameContext` 的旧调用使用不可变专用 upload allocation，由 command recording 保留；调用者必须继续保证 command/pool 在 GPU 完成前存活且不重置。`CommandBuffer::retainResource()` 不改变提交事务的失败时机，部分提交仍按原 frame completion 契约回收。

Raster 捕获实际使用的 leases，NRD 捕获 packet 和 kernel 实现。场景纹理发布后，VisibilityBuffer 依据不可变 texture snapshot 更新派生 remap，不能只比较 alpha texture slot 列表：材质编辑和 mip streaming 都可能在 slot 不变时替换原生分配。旧 snapshot 的 CPU views 由 snapshot owner 保留，录制后的 GPU 分配由 leases 保留。

第二批回归结果与限制见本节后续记录；第一批日志和基线信息仅作为历史记录，不代表当前工作目录仍保留这些生成文件。

### 第二批验证（2026-09-26）

构建配置：`build-full`，Debug，SOURCE dependencies，NRD/tests 开启，OpenUSD/NTC 关闭。`Metallic`、`MetallicRhiTests`、`MetallicNrdTests` 构建和 `git diff --check` 均通过。

| 验证 | 结果 | 本地日志 |
| --- | --- | --- |
| Mapped registry / GPUScene / frame 生命周期 | 30 通过 | `.cache/registry-stage2/core-final.log` |
| Mapped 材质、streaming、CLAS、场景着色 | 19 首轮通过，材质刷新修复后单独通过；实时管线随后开启运行条件通过 | `scene-mapped.log`、`material-refresh.log`、`realtime.log` |
| 最终 raster / LOD / 材质刷新复验 | 3 通过 | `raster-final.log` |
| Native registry / GPUScene / 数组 / 材质分桶 / streaming | 26 通过 | `native.log` |
| NRD Mapped / Native | 各 10 通过，包括 SDK runtime 在提交前清空后 GPU 结果仍正确 | `nrd-final.log`、`nrd-native.log` |
| 编辑器 Native smoke | 退出码 0，成功提交并呈现一帧 | `smoke.log` |

表中未写目录的日志均在 `.cache/registry-stage2/`。这些回归之间存在用例重叠，不能相加作为独立用例数量。所跑验证未报告 Vulkan VUID。

`frame_self_submit_two_slots` 根据第一批已有基线记录排除，本轮没有重新构建其基线。Zorah probes 因缺少 `METALLIC_ZORAH_Z4_PROBES` 指定的 cooked fixture 跳过。实时 streaming 测试使用 compact Bunny fixture，会关闭 DLSS 节点；不代表完整 DLSS/NRD/NRC 编辑器场景切换验证。未进行 CPU/GPU 帧时间 A/B 测量。

## 第三批：CPU slice 与普通数据 BDA

`BufferSlice` 只从 `Buffer::slice()` 或父 slice 的 `subslice()` 创建。它保存原生分配的强引用、分配内字节偏移及长度，设备 identity、usage 和地址均取自同一分配，不能用裸地址伪造来源。子范围只能缩小，`UINT64_MAX` 表示父范围余量；失败会清空输出，也支持原地缩小。零长度 slice 可用于 CPU 范围计算，但 GPU 数据/命令入口拒绝它。Buffer 包装对象移动、销毁或替换不会改变旧 slice 的来源。

```cpp
BufferSlice records;
auto result = buffer.slice(records, byteOffset, byteSize);
if (!result) { return result; }
ParameterWriter writer(device, frame, *registry);
MyParams params{.records = writer.dataBuffer<MyGpuRecord>(records)};
EncodedParameters packet;
result = writer.encode(params, kMyAbi, packet);
if (!result) { return result; }
return kernel.dispatch(commands, packet, groupCount);
```

`ShaderDataSpan` 是 16 字节的 `{uint64 address; uint32 count; uint32 stride;}`，对应 Slang `DataSpan<T>`。它不分配、不写入 buffer descriptor。`dataBuffer(slice, stride, alignment)` 校验来源设备、非空范围、绝对地址对齐、步长整除、元素数量上限及 Storage/ShaderDeviceAddress usage；显式 `ShaderDeviceAddress` 数据分配不要求 Storage usage。模板入口从 CPU GPU-layout 类型取得 `sizeof/alignof`，仍要求该布局与 shader 一致。不能把其他 writer 的裸 wire 值复制进 packet 并期待自动获得所有权；必须经过当前 writer 的 `dataBuffer()`。

Slang 的 `.data` 是普通 GPU 指针，`.length()` 在 stride 与 `sizeof(T)` 不一致时返回零；消费端先检查 `.contains(index)` 或完整的派生范围，再访问内存。CPU 范围校验与 shader 索引校验缺一不可；BDA 指针访问不自动继承 SSBO robust bounds。参见 [Vulkan BDA 示例](https://docs.vulkan.org/samples/latest/samples/extensions/buffer_device_address/README.html) 和 [BDA 对齐说明](https://docs.vulkan.org/guide/latest/buffer_device_address_alignment.html)。这不是强制检查所有指针解引用的语言包装，也不是自动推断 C++/Slang ABI 的反射系统。

`CommandBuffer::copyBuffer(sourceSlice, destinationSlice)` 返回 Result，要求等长非空范围、正确设备/transfer usage，拒绝同一分配内重叠复制。`dispatchIndirect(slice)` 要求 Indirect usage、4 字节对齐及至少 12 字节；只读取开头三个 dispatch 计数。`ComputeKernel` 同样提供 slice 间接入口。旧 Buffer+offset 入口转交给这些校验，保留原签名。复制和间接命令在录制时保留实际分配，typed packet 和 Core 数据槽也保留同一分配，延续 frame 完成/取消/部分提交回收契约。Device 仍须晚于所有 slice、packet、command 和 GPU 工作销毁。保留解决存活期，不负责 barrier、队列依赖或阻止 CPU 提前覆盖正在使用的字节。

原生 BDA 在分配创建时查询一次；当前 VMA 路径不重定位存活的 buffer。CPU slice 不承担长期资产身份。Streaming 的 ResourceId/PageId、generation、resident 映射仍保留，解析到当前分配后再形成 slice，避免把可迁移页永久表示为裸地址。slice 保留的是原生 allocation，页内范围的复用与 residency pinning 仍由 streaming 原有协议负责。

### 生产接入与兼容边界

- 材质分桶的 records、instances、materials、shadingMaterials、bins、tiles、arguments、streamRecords 全部使用 DataSpan；图像继续使用共享 descriptor。`MaterialBinningParams` ABI 升至版本 2，152 字节，标量起始偏移 136。
- 分类、分桶、间接参数生成及 deferred 的 bins/tiles 消费形成完整 BDA 路径。shader 对场景索引、tile 派生索引和参数范围进行显式检查。
- `ComputeProgramBindingDesc::DataBuffer` 要求明确的 `dataStride/dataAlignment`，dispatch 可传拥有型 `data` slice，或由 Buffer+offset+size 构造 slice；两种来源不能混用。Core slot 仍为 16 字节，第二个 64 位 payload 对 descriptor 数组表示数组地址，对 DataBuffer 表示 count/stride。`getData<T>(slot)` 与 descriptor accessor 共存，不改变 Native descriptor 正规化策略。
- 数字 slot、未迁移的 StructuredBuffer、纹理/采样器/AS 与 SDK 接口继续使用共享 registry。未强制改写全部 shader 或 streaming 页表；后续普通数据迁移可沿相同入口逐项进行。没有改变同步模型，也未宣称 CPU/GPU 帧时间收益。

第三批回归覆盖父范围收窄/原地切片、溢出和对齐拒绝、错误来源设备/usage、包装对象替换、取消保留，以及非零偏移的复制 → BDA compute → GPU 间接调度 → Core 数据槽消费。后者使用没有 Storage usage 的数据分配，检查范围外哨兵不变，并断言 descriptor 写入与 live descriptor 均为零。材质分桶的 typed consumer 同样不增加 descriptor 写入。

### 第三批验证（2026-09-26）

配置沿用第二批 `build-full`；最终 `Metallic`、`MetallicRhiTests`、`MetallicNrdTests` Debug 构建通过。结果日志均在 `.cache/registry-stage3/`：

| 验证 | 结果 | 日志 |
| --- | --- | --- |
| Mapped registry / GPUScene / frame / streaming 上传 | 34 通过 | `core-mapped.log` |
| Mapped deferred / 场景材质 / streaming / CLAS | 19 通过 | `scene-mapped.log` |
| Native slice / registry / GPUScene / 材质分桶 / deferred / streaming | 30 通过 | `native.log` |
| 最终版本 slice/BDA 数据链与材质分桶复验 | Mapped / Native 各 4 通过 | `final-mapped.log`、`final-native.log` |
| NRD | Mapped / Native 各 10 通过 | `nrd-mapped.log`、`nrd-native.log` |
| Native 实时 streaming 管线 | 1 通过，显式开启 `--rhi-realtime` | `realtime.log` |
| Native 编辑器 smoke | 退出码 0，提交并呈现一帧 | `smoke.log` |

用例有重叠，表中数量不能相加为独立用例总数；所跑验证未报告 Vulkan VUID。仍排除前述已记录基线失败的 `frame_self_submit_two_slots`，本轮未重建该基线。实时管线为 compact Bunny fixture，关闭 DLSS 节点；这不是完整 DLSS 编辑器场景切换验证。没有帧时间 A/B 结果。测试改写的 `meet_mat.glb.meshlets.bin` 已按 HEAD 匹配的原备份恢复。
