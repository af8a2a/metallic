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

`stats()` 暴露 descriptor 写入次数、cache hits、live descriptors、参数累计上传字节和 arena 保留容量。新测试断言跨 kernel 引用分桶结果只允许为新 output 增加一个 descriptor。这是减少重复 descriptor 写入的机制证据，不是 CPU/GPU 帧时间提升测量。

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
