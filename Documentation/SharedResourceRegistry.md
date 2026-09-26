# Shared registry 与 typed compute 参数：第一批实现

## 已接入的路径

`Device::resourceRegistry()` 提供设备级共享 registry。`ComputeKernel` 只保存 shader、pipeline 与参数 ABI；没有 Program 私有 heap、binding layout 或 resource table。

生产路径 `MaterialBinning` 的 reset/classify/arguments 三个 kernel 已迁移：使用 `MaterialBinningParams` 的命名字段，三个 dispatch 复用一个不可变参数包。`VisibilityMaterialBinning.slang` 直接解析字段里的 typed descriptor handle，删除了这条路径的数字 slot 配置。

新增 typed 间接消费测试让两个消费 permutation 复用生产分桶的 bins/tiles/arguments 身份，执行真实 GPU indirect dispatch 与逐像素回读；同时保留原有 legacy 消费测试，验证两套接口混用时 heap 切换正常。

**迁移边界：ScenePathTracePass 的生产着色消费端、其他 ComputeProgram 调用者尚未迁移。** 它们仍拥有原来的 heap/table。本批交付是共享所有权与 typed 参数基础设施、生产分桶迁移及可回归的 typed 间接消费样例；不能视为研究报告阶段 1 的所有生产调用者收敛已经完成。

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

TLAS lease 只持有 TLAS 分配，不会自动发现其引用的 BLAS/场景资源。使用 `writer.retain(sceneOwner)` 保留完整场景快照；partitioned AS 和外部 SDK 资源尚无新适配。所有 registry/packet/lease/frame 仍要求 Device 最后销毁。借用的 swapchain 图像不能通过 registry 获得拥有型 lease，当前会拒绝注册。

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

后续生产迁移应优先完成 ScenePathTrace 的命名参数、材质纹理 handle 数组和场景快照保留，然后删除其私有 heap/table；保持 DLSS/NRD/NRC 的资源状态边界及 scene-binding 一致性检查。BDA buffer API、同步模型和 lazy native view 不属于本批改动。
