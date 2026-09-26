# 原子 buffer 使用 BDA 的收益与影响

日期：2026-09-25。范围：Metallic 当前 Vulkan / Slang dynamic resource 路径。本文包含代码调查、隔离 shader 编译和真实 GPU 原子回读；没有修改生产 shader/RHI，也没有测量 GPU 性能或重跑完整渲染场景。

## 结论

建议把 BDA 作为**独立、私有 64 位原子数据的正式候选**，首选 HybridRaster 的 packed depth/visibility buffer，其次是 workload 计数 buffer。暂不按“包含原子操作”这一条件迁移全部 buffer。

已经验证的收益是绕过当前 NVIDIA 驱动对 mixed 32/64-bit untyped StorageBuffer 原子的管线编译故障，同时保留现有 normalizer 对其余 descriptor buffer 的布局修复。架构收益是这些私有 buffer 可跨 heap 使用同一地址，不再需要为每个 producer 注册、传递不同 descriptor index。帧率收益尚无测量依据。

如果目标仅为尽快修复三个 native 渲染测试，原有的[平坦 int64 buffer 保留 typed 指针链方案](NativeAtomicPointerFixPlan.md)仍然改动更小；如果希望同时收敛私有原子资源的传参和访问方式，选择性 BDA 更值得实施。两者无需作为同一次改动捆绑落地。

## 已完成的验证

环境：NVIDIA GeForce RTX 5070 Ti，驱动版本原始值 2585198592（616.92），Slang 2026.18.2，Vulkan SDK 1.4.350.0。

| 验证 | 结果 | 能证明什么 |
| --- | --- | --- |
| 32 位 descriptor 原子与 64 位 BDA 原子同一 dispatch | GPU 回读通过 | 这两种指针路径可以在当前设备共同正确运行 |
| 256 个线程执行 64 位 Add / Min / Max / CompareExchange | 计数、高 32 位、最终值均正确 | 保留真正的 64 位原子语义 |
| 原子返回值与 32 位 ticket | 排序后与预期序列完全一致 | 多线程竞争下没有丢失累加或重复分配 |
| BDA 使用 allocation address + 64 字节 | 前后及内部 padding 哨兵完整 | 本用例的子分配寻址和边界正确 |
| 当前 normalizer 处理后的 SPIR-V | spirv-val 通过；Vulkan validation 无 warning/error | 不依赖非法的逻辑指针 bitcast |
| 六个 stream raster 入口的 BDA 变体 | 全部 SPIR-V 校验、Vulkan 管线创建通过 | 包括原来会崩溃的入口，均绕过编译故障 |

GPU 用例中，descriptor 32 位计数器使用非零最终索引 3；64 位目标使用 PhysicalStorageBuffer 指针。256 个线程得到：sum = `0x10000000100`，max = `0x1000000f00d`，min = `0x10000f00d`，CAS = `0xfeed00000000beef`。普通 64 位写入的 SPIR-V 带 `Aligned 8`；32 位原子仍通过当前 normalizer 生成的 untyped StorageBuffer 指针。

六个入口为 `streamClusterRasterMain`、`streamClusterRasterLegacyMain`、`streamClusterRasterPlaneMain`、`streamClusterRasterCooperativeMain`、`streamClusterRasterWorkBinsMain`、`streamClusterRasterWorkControlMain`。编译 PoC 只在 `.cache` 的 shader 副本中给 stream push 添加地址，并替换共享 raster helper 的像素访问；没有完成 resident/reset/resolve 的生产迁移。

证据位于 `.cache/atomic-bda-research/`：`AtomicBda.slang`、`AtomicBdaReadback.cpp`、`AtomicBda.spvasm`、`readback.log`、`results.json`、`pipeline-results.json` 和各入口日志。GPU 回读使用独立 Vulkan runner、host-visible coherent buffer；六入口仅创建管线，没有提交完整 raster 工作负载。因此不能据此宣称三个渲染测试已修复、完整画面一致，或 BDA 更快。

## 为什么能够绕开当前故障

当前失败链为：descriptor heap 的 `OpBufferPointerEXT` → normalizer 将 StorageBuffer 的指针链改为 untyped → 同一 shader 包含 32 位和 64 位原子 → NVIDIA 驱动创建 compute pipeline 时 CPU 访问异常。详细隔离结果见[原子指针调查](NativeAtomicPointerInvestigation.md)。

BDA 的 64 位目标直接成为 `PhysicalStorageBuffer` 下的 typed pointer，现有 normalizer 不转换这一访问链；32 位 descriptor 原子继续使用原路径。GPU PoC 已实际验证这一组合。BDA 没有取消原子指令，也没有把一次 64 位更新拆成两次 32 位更新。

这只解决迁移目标的指针路径，不意味着可以删除 normalizer：其他 dynamic resource 的嵌套结构仍需要既有显式布局修复。BDA 也不是 AS descriptor 的统一解法；普通 buffer 地址与 acceleration structure 的访问契约仍应分别处理。

## 收益与代价

| 项目 | 收益 | 代价或边界 |
| --- | --- | --- |
| 驱动兼容性 | 64 位原子不再进入当前有问题的 untyped StorageBuffer 路径；已有 GPU 证据 | 仅验证了本机驱动；完整生产回归仍必需 |
| CPU 资源传递 | 同一 allocation 的地址不随绑定 heap 改变；可删除目标 buffer 在私有/producer heap 中的重复注册 | 地址变化时需更新所有 CPU/GPU 参数包；不能靠重写 descriptor 重定向旧地址 |
| GPU 寻址 | 源码层省去目标 buffer 的 heap descriptor 解析，可直接 address + offset | 编译器可能已提升或缓存 descriptor load；地址和寻址中间值可能增加寄存器占用 |
| 存储和带宽 | 少量 descriptor 槽位与更新消失 | 像素数据仍为每像素 8 字节；原子争用、像素写流量和覆盖率计算不变 |
| ABI | 私有原子数据可以使用明确的地址参数 | 单个 index 32 位变为 address 64 位，需重新核对布局、反射、padding 和 push 大小 |
| 范围与同步 | 可自然访问对齐的子分配 | 地址不携带长度；边界、所有权、barrier 和队列依赖仍由应用负责 |

本机 storage-buffer descriptor 为 16 字节，删除几个槽位的显存收益很小；像素 buffer 仍约为 1080p 15.82 MiB / 4K 63.28 MiB。只迁移像素和 workload 也不能删除整个 heap，因为 queue、cluster、indirect arguments 等仍通过 descriptor 使用。不能把“少一次 descriptor 解析”直接换算成可观 FPS 提升。

Vulkan BDA 需要启用 bufferDeviceAddress、buffer 的 shader-device-address usage 和分配的 device-address 能力。Metallic 已有 `Buffer::deviceAddress()`，Vulkan backend 对相关 storage/uniform/indirect/transfer buffer 自动添加 usage，VMA allocator 已启用 BDA。迁移不需要先重写分配器，但目标 buffer 宜显式声明地址用途；64 位原子仍需检查并启用对应能力。[Khronos BDA 指南](https://docs.vulkan.org/guide/latest/buffer_device_address.html)

## 迁移对象与修改范围

| 对象 | 当前情况 | 建议 |
| --- | --- | --- |
| HybridRaster packed depth/visibility | `VisibilityHybridRasterizer::buffers_[1]`，width × height × uint64；共享 helper 执行 InterlockedMax | 第一阶段迁移；保持 packed 比较与整体原子语义 |
| Workload 计数 | `workloadBuffer_`，16 个 uint64，共 128 字节；reset / Add / Max 与 CPU 回读 | 第二阶段迁移；独立 workload 地址字段 |
| Queue、cluster/bin、间接参数 | 含 32 位计数、结构数据、indirect command 和多消费者 | 先保留 descriptor；BDA 不是修复当前 64 位混合故障的必要条件 |
| 页表、traversal、request 等 | 原子嵌入流式资源结构和状态协议 | 暂不迁移；会扩大到共享 ABI、容量和生命周期 |
| SHaRC | 通过 ComputeResources 和 SDK interop 使用 uint64 buffer | 独立审计；本轮 PoC 不覆盖，不按类型全局替换 |

关键文件与责任：

- `Source/Runtime/Render/VisibilityHybridRasterizer.{h,cpp}`：像素/计数 allocation 所有权、地址获取与刷新、私有 push、reset/resolve 和显式 buffer barrier。
- `Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp`：resident/stream producer 参数，重复 descriptor 注册，以及 workload capture 所需的 Buffer 对象。
- `Source/Runtime/Render/Streamer/MeshletStreamRuntime.h` 与 `Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang`：CPU/Slang 对应的 stream push ABI。
- `Shaders/Modules/GPUDriven/HybridRasterTriangle.slang`：两个共享 raster helper 改为像素指针访问。
- `Shaders/Features/VisibilityBuffer/VisibilityHybridRaster.slang`：reset 写入、fragment resolve 读取和 cluster 初始化必须一致更新。
- resident、stream、cooperative、work-bin/work-control 调用点及 `tests/rhi` probe：都必须传入相同 allocation 的地址。

## ABI 设计和容易遗漏的地方

建议在 CPU 参数中使用明确命名的 `uint64_t atomicPixelAddress`、`uint64_t workloadAddress`；Slang 可以接收 `uint64_t*`，或在 helper 边界把 64 位地址转成该指针。两端以反射和 `sizeof` / `offsetof` 断言验证，不能把它当作普通 shaderIndex。

1. **不直接把 bins[11] 扩为 uint64。** 当前 16-word cluster header 中，`bins[11]` 是 producer pixel index，`bins[12]` 已是 input count，后续 word 也用于计数。把高 32 位塞入下一 word 会破坏协议。优先通过 producer push/raster 参数传独立像素地址，再删除 bins[11] 的索引依赖；不要为此顺手重排整个 header。
2. **拆清 hybridQueueBuffer 的含义。** 当前该字段分别被用作 triangle queue、candidate indirect arguments 和 workload counter 的 descriptor。它不是专用原子资源字段，不能直接整体扩宽并改成 BDA。
3. `MeshletStreamUserPush` 当前 136 字节；按末尾追加一个 8 字节地址的方案为 144 字节，追加第二个为 152 字节。这里是明确的追加方案，其他重排必须重新计算。还需核对 mapped/native 参数上传、shader reflection、设备 limits 和 shader warmup 使用相同 ABI。
4. owner 的 Hybrid push、resident push 与测试 push 也需同步更新。reset/raster/resolve 是同一 buffer 的不同访问阶段，不能只改 InterlockedMax writer 后就移除读写阶段仍在使用的 descriptor。
5. 选择明确无效值 `0` 并在可选路径检查；不能沿用 `UINT32_MAX` 的 index sentinel，不能截断成 uint32。地址不持久化到资产、meshlet cache 或跨设备缓存。

## 地址生命周期、边界与同步

- 地址从 buffer allocation 获取，按 allocation 生命周期缓存。当前 `Buffer::deviceAddress()` 每次调用都会执行 Vulkan 查询；不需要每个 dispatch 重复获取不变地址。
- 当前 `setRenderExtent()` 在既有像素 allocation 容量内只改变尺寸，地址可保持不变；增长、重新初始化或资源替换时需刷新所有 producer/consumer 参数。
- 旧地址可能已写入正在执行或等待提交的 command buffer、push、GPU 参数块；必须等对应 GPU 使用完成后再释放旧 allocation。保持现有 frame overlap 所有权约束，不能依赖更新当前 CPU 地址来修复已经记录的命令。
- 64 位像素/计数元素和子分配偏移保持至少 8 字节对齐。地址不附带 descriptor range，显式使用 width/height、容量和 counter count 做范围控制；`GetDimensions()` 不能直接套用到裸指针。BDA 范围检查需要应用处理，具体 load/store 的对齐承诺必须与地址相符。[Khronos BDA 示例](https://docs.vulkan.org/samples/latest/samples/extensions/buffer_device_address/README.html)、[对齐指南](https://docs.vulkan.org/guide/latest/buffer_device_address_alignment.html)
- 保留 reset → raster → resolve 的 buffer 状态转换、compute/graphics 队列依赖和 indirect command 同步。原子操作不自动同步整个 dispatch 或相邻 pass。
- 地址字段不能替代 render graph 的资源依赖，也不能替代 DebugControl/capture 持有的 Buffer 引用。回读、诊断名称、资源销毁仍以 Buffer 对象为准。

## 性能验证与落地顺序

1. 先迁移 pixel 的完整 reset / producer / raster / resolve 链，清理对应重复 descriptor 注册；随后迁移独立 workload 地址。暂不改通用资源句柄、32 位结构 buffer 和 AS。
2. 把本轮隔离 GPU 用例转成正常 RHI 测试，补充动态像素索引、非零子分配偏移、guard 区域和 resize/recreate/address refresh；校验原子操作与嵌套 descriptor 读取同时正确。
3. 重跑原先三个 native 失败用例，覆盖六种 raster 入口、resident/stream、tessellation、mapped/native、缓存冷/热路径；完整画面和 GPU 回读按既有容差与正确基线比较。另验证 frame overlap、资源重建、SHaRC 与实际 DLSS workload 没有受共享代码改动影响。
4. 性能 A/B 使用同一场景、设备、驱动和 raster 变体，以正确工作的 typed-descriptor 修复候选或 mapped 实现作为明示基线。不能拿崩溃的 native 路径当基线，也不能把模式切换带来的差异全算到 BDA。
5. 分开记录冷启动编译、CPU descriptor 更新和 GPU 时间；GPU 比较整个 pass 及 raster kernel，必要时检查寄存器、占用率和 descriptor 相关访存。不同覆盖率/过度绘制和分辨率下重复测量，报告波动。只有实测支持时才提出性能收益。

本轮结论：选择性 BDA 已有功能可行性证据，兼容性与私有资源传参简化是明确收益；全面原子 buffer 迁移和 GPU 加速主张目前均缺少依据。
