# VisibilityBuffer Hybrid Rasterizer

`VisibilityBufferPass` 默认启用软硬混合光栅，常驻 GPUScene 和该 pass 内的 StreamAsset producer 共用实现。延迟着色继续读取原来的 `R32Uint visibility`、`D32Sfloat depth` 和 `rasterInfo`，图资产及 ID 编码不变。

## 数据流

1. Instance 剔除之后，compute 以每 cluster 一个 128 线程组执行 producer ownership、视锥、normal-cone 和相应 early / late HZB 测试。Resident 与 StreamAsset 分别读取 GPUScene 或流式页中的原始顶点和索引。
2. 同组线程投影共享顶点，然后检查每个三角形的屏幕包围盒与裁剪范围。整个 cluster 的三角形都满足软件尺寸阈值时进入软件箱；只要有一个三角形需要硬件处理，整个 cluster 就进入其材质对应的硬件箱。Masked cluster 直接进入硬件箱。
3. 每个输入写入自己的分类槽，随后 GPU 分块统计、前缀求和、散射，生成四个硬件材质箱和一个软件箱。压缩保留原始候选顺序和 visibility record ID，避免无序原子追加导致硬件等深图元在帧间改变覆盖顺序。全部计数和 indirect arguments 在 GPU 上生成。
4. Resident 硬件箱由 AS 每组消费 32 个 cluster，再由 MS 绘制；Stream 硬件箱每 cluster 调度两个最多 64 三角形的 MS 工作组。软件箱直接 indirect dispatch compute，每组加载一个 cluster 的共享顶点，一线程光栅一个三角形；这些 cluster 不进入 Mesh Shader，也不写入投影三角形队列。
5. 软件光栅使用子像素定点边函数、top-left 规则和基于舍入后顶点的屏幕线性深度。像素高 32 位保存深度排序键，低 32 位保存原始 visibility ID，以 `InterlockedMax(uint64_t)` 一次更新两者。普通 Z 对浮点深度位取反，Reversed-Z 直接使用浮点位。
6. 开启异步时，分箱完成的 graphics 段通过 timeline semaphore 放行 compute 软件光栅；硬件光栅在 graphics 队列上仅依赖分箱，二者可以重叠。汇合 graphics 段等待软件和硬件分支完成后，再进行深度合并。
7. 全屏合并输出 `SV_Depth` 与 ID，将软件最近交点合入原有硬件附件。每个 resident / stream 绘制阶段完成合并后才生成 HZB；冻结剔除相机的独立附件也执行同样流程。

## 设置和回退

- `hybridRaster` / **Hybrid Software Rasterization**：默认 `true`，关闭可做纯硬件对照。
- `clusterPrebin` / **Cluster Prebinning**：默认 `true`。关闭后切回 Mesh Shader 按三角形追加的软件队列路径，便于比较；`hybridRaster=false` 切回纯硬件。
- `asyncSoftwareRaster` / **Async Software Rasterization**：默认 `true`；需要 cluster 预分箱、独立 compute 队列和 RenderGraph 自提交路径。关闭可对照串行 cluster 光栅。没有独立队列、仅录制外部命令缓冲或关闭预分箱时自动串行。
- `softwareRasterMaxPixels` / **Software Triangle Size (px)**：默认 8，范围 1–32；每个三角形屏幕包围盒的宽和高都不超过阈值才接受该 cluster。
- 需要已启用的 `shaderInt64` 与 `shaderBufferInt64Atomics`，以及 1–8 位 `subPixelPrecisionBits`；不满足时保留完整硬件路径。
- Alpha Mask 继续走原有硬件双线性 alpha test；BLEND 仍遵循既有可见性 pass 的支持范围。跨近/远裁剪面、无效坐标和超范围三角形交给硬件裁剪。
- 分箱容量 `C` 覆盖 resident record capacity 与 stream candidate capacity 的较大者，每个箱都能容纳全部候选。分箱及暂存空间为 `64 + 32*C + 20*ceil(C/128)` 字节，五组 indirect arguments 共 60 字节；输入超过容量会在记录 GPU 命令前返回错误，场景容量增长时重建绑定并延迟回收旧资源。
- 软件深度/ID 占每像素 8 字节。为支持运行时对照，仍分配最多 262,144 个三角形的旧队列（16 MiB + 32 字节）及其 12 字节参数；仅关闭预分箱时才向该队列追加，队列溢出三角形保留硬件图元。
- GPU 捕获包含 clear、stable cluster bins、software triangles、merge 区段。Debug checkpoint 提供 `AfterResidentEarlyBins`、`AfterResidentLateBins`、`AfterStreamEarlyBins`、`AfterStreamLateBins`；资源 `hybrid.<pass>.clusters` 与 `hybrid.<pass>.arguments` 可读回计数和列表。
- Cluster buffer 前 16 个 uint 为 header：0–4 为各箱计数，5 为容量，6–7 为尺寸，8 为阈值浮点位，9 为 Reversed-Z，10 为子像素精度，11 为软件像素描述符，12 为候选数，14 为溢出计数。箱 `b` 的有效 record ID 位于 `16 + b*C` 起的 `header[b]` 项。

该实现使用 cluster 级 compute 预分箱，并保留硬件深度附件与软件合并步骤。软件通过独立 compute 队列与硬件并行，二者仍使用各自的深度存储并在汇合后合并。实际重叠程度和性能收益取决于 GPU 调度、场景中的软硬工作量及额外提交成本；需要在目标场景测量。

## 异步提交与资源生命周期

```text
Graphics: cull + bins + clear ──┬── HW raster ──────────┬── merge ── HZB
                               │                      │
Compute:                       └── SW raster ──────────┘
```

- `DeviceDesc::enableAsyncCompute` 选择独立 compute family，若只有通用 family 则尝试其中的第二条队列；不满足时保留通用队列。编辑器和预览 renderer 默认请求此能力，现有底层调用默认不变。`Queue::sameQueue` 检查实际 Vulkan queue，避免把两个包装对象误认为并行队列。
- `RenderGraphExecutionContext::parallelCompute` 在录制期间创建分支及汇合段，**不在 pass 内提交或等待 GPU**。所有命令录制成功后，executor 才按依赖提交；两条分支没有相互等待，后续 graph 消费者依赖汇合段。
- 调用 `parallelCompute` 后必须重新取得 `context.commandBuffer()`；原命令缓冲已结束录制。自定义 pass 必须保证分支读写不冲突，所用资源声明 Graphics / Compute 共享。外部命令缓冲 API 保留串行执行契约。
- GPUScene 全局缓冲原已支持共享；本次为软件光栅读取的私有参数、stream pages/records 与分箱、原子像素缓冲补齐共享声明。预分箱后的 stream 光栅不再产生页请求或更新可见记录；这些写入由前置 graphics 分类阶段完成。
- 帧完成点覆盖所有贡献队列，资源与 semaphore 保留到整批完成；软件或硬件分支录制失败会取消整批未提交事务。现有跨帧 history、尺寸变化和延迟回收规则继续有效。
- 主编辑器和离屏预览使用 `execute(RenderGraphSubmitDesc)`，在显示或读回输出前添加 graph 完成依赖。`executionStats().asyncComputeBranches` 报告实际建立的独立 compute 分支数量；GPU pass 时间覆盖分叉到汇合的耗时，捕获标签分别标注 resident / stream 的软件和硬件光栅。

## 参考

参考 [Epic Nanite SIGGRAPH 2021 演讲](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf) 的小三角形线程映射、边函数、64 位 depth/visibility 原子写和大三角形硬件回退；同时核对本地 `E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush`、`NaniteRasterizer.usf` 的子像素三角形设置。实现为 Metallic 自身的 Slang/RHI 代码。Vulkan 覆盖和插值规则见 [Rasterization](https://docs.vulkan.org/spec/latest/chapters/primsrast.html)。

## 验证

```powershell
build/tests/MetallicRhiTests.exe --gtest_filter="*hybrid_*" --rhi-validation
```

`hybrid_raster_depth_coverage_and_overflow`：真实 Mesh Shader 分流及 indirect compute，逐像素比较 HW/SW ID 和 D32 深度，覆盖共享边、裁剪、透视深度、正反面、两种 Z、三种阈值和容量为 1 的强制溢出。单独读回软件原子像素，证明实际执行了软光栅。

`hybrid_raster_scene_equivalence`：真实 Bunny 场景，对照纯硬件三角形 ID，检查 cluster/triangle 运行时切换、透视/正交、两种 Z、三种阈值，串行/异步开关，共 108 个连续 HZB 对照帧；输出 `HybridBunny.png`。

额外运行现有混合 producer、Alpha Mask、相机冻结、HZB 与延迟着色回归。独立旧版 `GPUDrivenStreamAssetPass` 的 `render_graph_gpu_driven_streamasset_pass_smoke` 在隐藏后恢复检查失败，使用 HEAD 原版流式 shader 亦可复现；它与统一 `VisibilityBufferPass` 的混合 producer 测试分别记录。

2026-09-11 本机验证：主程序构建通过；10 项相关 RHI 回归通过，新增的两项回归另以同步验证层运行通过，日志未出现 VUID / SYNC 错误；实时 pipeline smoke 通过 16 帧 DLSS 相机移动、共享视图和显式 Reset。日志与图像保存在本地 `.tmp/HybridValidation/`、`.tmp/HybridSync.log`、`.tmp/HybridSmoke.log`。

2026-09-12 cluster 预分箱验证：`hybrid_cluster_stable_bins_and_indirect_limits` 覆盖五箱稳定压缩、原始 ID、空输入、全部剔除、部分尾块、容量拒绝、2,097,121 个候选，以及 resident HW / stream HW / SW 参数的二维调度边界。混合 producer 从真实 shader 读回 resident/stream 软件 cluster 数量 506/19，模式切换和冻结相机保持原始 visibility/depth 不变。11 项相关 RHI 回归通过，启用同步验证后未发现 VUID / SYNC 错误；日志见 `.tmp/ClusterValidation.log`、`.tmp/ClusterMixed.log`。最终任务 payload 顺序修正后，5 项相关回归复测通过（`.tmp/ClusterFinal.log`）；主程序构建及实时 pipeline 的 16 帧 DLSS 相机移动与 Reset 冒烟通过。

异步验证新增 `frame_parallel_compute_join_and_cancellation`：验证独立/别名队列的分支选择、汇合后 GPU 数据可见性、compute/HW 录制失败及逆序事务取消。混合 producer 验证 early/late 的四个异步分支，并在串行/异步切换、冻结相机和可视化切换后比较原始 visibility/depth。日志 `.tmp/AsyncValidation.log`。

2026-09-12 异步软硬并行验证：24 项 RHI 回归全部通过（`.tmp/AsyncRegressionFinal.log` / `.xml`），包括 108 帧 Bunny 对照、resident/stream 四个分支、相机冻结、HZB、场景/材质刷新和失败取消。GPU 使用 graphics family 0 与独立 compute family 2。主程序构建通过，实时 pipeline 冒烟每帧记录 2 个实际异步分支，16 帧 DLSS 移动相机、共享视图及 Reset 通过（`.tmp/AsyncSmokeFinal.log`）。同步验证日志无 VUID / SYNC 错误。尚未对目标大场景测量性能收益。

验证环境为本地 `.tmp/VulkanValidationLayers`，版本 1.4.357，基于 `f4874ee` 并修复其 descriptor heap 地址跟踪：重新绑定 heap 时先移除旧地址范围，再更新 command buffer 状态，避免遗留指针在命令缓冲回收后造成验证层崩溃。补丁仅位于本地测试工具（`.tmp/ValidationHeapMapFix.patch`），没有修改引擎的 `External/`。可用以下环境复现同步验证：

```powershell
$env:VK_LAYER_PATH = 'E:\metallic\.tmp\VulkanValidationLayers\build\layers'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
build/tests/MetallicRhiTests.exe --gtest_filter="*hybrid_*:*frame_parallel_compute_join_and_cancellation*:*render_graph_gpu_driven_mixed_producer_render*" --rhi-validation
```

编辑器接入同时统一由 executor 推进 history 帧；完成 GPU 等待后回收旧命令缓冲，再允许重编译释放/复用 descriptor heap。交换链呈现提交使用 AllCommands 等待 acquire semaphore，覆盖其布局转换；设备创建为 Streamline 注入的 `VK_NV_low_latency2` 补齐设备支持的 present-id 扩展依赖。离屏 graph 提交仍可在交换链等待之前执行。
