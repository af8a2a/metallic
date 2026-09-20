# ZorahFull 统一 GPU 预算（T0）

2026-09-20，基于 `ae60d3d76` 的工作区实现。GPUDrivenSample 默认启用；普通编辑器和未配置的 RHI 调用保持不启用 admission 策略。

## 已实现

- VMA 启用 `VK_EXT_memory_budget`，读取每个 heap 的预算、使用量、block/allocation 字节与数量。扩展不可用时显式标记 `driverBudget=false`，使用 VMA 估计值。
- 所有本后端 VMA buffer/image 分配入口共用预算锁，包括描述符堆和 OMM identity 索引。按资源真正选择的 memory type 检查对应 local heap，分配不足返回 `OutOfMemory`。同时启用 VMA `WITHIN_BUDGET`；不会继续无条件分配。
- 分域追踪 geometry、CLAS、CLAS scratch、RTAS、material textures、frame、upload、other。数值是存活资源的实际 VMA allocation 大小；被旧场景/在途快照保留的资源到真正析构才扣除。driver 内部与 SDK 分配体现在 heap usage，不伪装成可精确归属的 RHI 字节。
- 可移动的 RAII 预留令牌为未来分配留额度；编译失败、重建和析构释放额度。RenderGraph 在加载前保留帧资源及活跃 DLSS 功能额度，帧资源分配前归还其预留，各功能首次执行结束后释放其预留并刷新 heap usage。
- Full 的 geometry/CLAS 池在纹理准备之前创建，因此纹理规划能看到其实际占用。纹理在 header 探测完成后取最新共享额度，沿用现有统一 mip cap 降档算法，实际创建较小尾链 image。最小尾链无法满足额度时明确失败。
- 纯上传用的 `HostUpload + TransferSource` 在策略启用时使用 `AUTO_PREFER_HOST`，避免在有大 BAR 的独显上与场景争抢 device-local heap；UMA 仍使用其可用 heap。直接供 GPU 读取的 host buffer 保持原有选择。
- 修正 Buffer/Texture 默认移动赋值时被覆盖的内部对象未释放 Vulkan 资源的问题：资源释放和记账放到 Impl 析构，移动赋值与普通析构走同一路径。

## 预算口径与默认值

`可新增额度 = max(0, heapBudget − max(heapUsage, VMA blockBytes) − safety − 未兑现预留)`。

已存在的几何、CLAS、纹理、旧资源和 staging 已在 heap usage 中，不能再次减去它们的分域总数。RHI 逐次检查目标 heap；纹理规划使用最大的 device-local heap 作为规划依据，最终分配检查仍以实际 heap 为准。

| 策略 | 默认值 | 生命周期/用途 |
| --- | ---: | --- |
| safetyBytes | 256 MiB | 持续保留，缓冲估计误差和未预见增长 |
| graphReserveBytes | 256 MiB | 图加载前预留，图资源分配前释放 |
| externalFeatureReserveBytes | 每个启用功能 512 MiB | SR/RR/NR 各自首次执行后释放；禁用 NR 不预留 |
| materialImageOverheadBytes | 每张 64 KiB | 纹理规划额外扣除的 driver/OS 开销估计，独立于 image allocation |
| deviceLocalHeapLimitBytes | 0（不额外限制） | 可配置的 local heap 上限；压力回归用它构造可复现限制 |

纹理有效 allocation 额度为 `min(原配置纹理上限, 可新增额度 − image 数 × 每图开销估计)`。Full 原有 512 cap / 2048 MiB 上限未改成另一套 preset；允许额度不足时才降低 cap。

每图 64 KiB 是当前独立分配路径的保守规划参数，不是 Vulkan 对所有驱动的保证。压力实验实测：只按 allocation 求和时，image 已分配约 260 MiB，driver heap 使用已经约 407 MiB，导致原先的 358 MiB 尾链计划中途被实时检查拦截。引入开销估计后，同条件提前选择较小尾链并完成。以后材质 image 改为子分配时应重新测量这个参数。

heap 数据通常至多每 100 ms 强制刷新，VMA 也会随分配活动更新；阶段日志强制刷新，包括外部功能调用前后。该预算是驱动估计，不是跨进程硬配额；其他应用突然抢占或 SDK 内部分配仍可能失败。512 MiB 的 DLSS 预留也不是其所有分辨率/功能组合的峰值上限。

## 实测验证

Release `MetallicGPUDrivenSample`、`MetallicRhiTests` 编译成功。开启 Vulkan validation 的核心 18 项回归通过，覆盖预算拒绝与恢复、资源保留和移动析构、图编译/resize/失败重试、buffer bindless、跨队列上传、BC mip、KTX 取消/错误路径、完整 Full 纹理加载。日志见 [regression.log](../build-release/gpu-memory-budget/regression.log)。

同一 Full 元数据、4418 张 KTX2，使用真实 GPU 上传和末尾逻辑纹理描述符采样。以下只验证纹理路径，不加载全量几何，不代表 Full 实时图/DLSS 首帧验收。

| 条件 | 选中 cap | image allocation | local heap usage¹ | 预留和 safety 之外剩余额度 |
| --- | ---: | ---: | ---: | ---: |
| 原 2 GiB 纹理上限，统一 admission 关闭（对照） | 512 | 1401.73 MiB | 1629.18 MiB | — |
| 新增额度 768 MiB，另保留 128 MiB future + 64 MiB safety | 256 | **357.99 MiB** | 586.68 MiB | 76.62 MiB |
| 新增额度 512 MiB，另保留 128 MiB future + 64 MiB safety | 128 | **96.81 MiB** | 332.74 MiB | 74.56 MiB |

¹ 测试 device 的 local heap 快照，包含 driver、分配器和采样探针等，并非纯纹理 image 总和，也不是 NVML 整卡占用。压力上限按初始 usage + 新增额度 + safety 设置，再用令牌扣除 128 MiB future；因此两档 header 探测后的共享可用额度分别是 640、384 MiB。4419 张 image（含 fallback）的开销估计共 276.19 MiB，在规划时进一步扣除。

证据：[对照](../build-release/gpu-memory-budget/regression/zorah-textures.json)、[768 档](../build-release/gpu-memory-budget/pressure768/zorah-textures.json)、[512 档](../build-release/gpu-memory-budget/pressure512/zorah-textures.json)。两档均完成全部上传、保持 4419 个物理描述符并通过 GPU 采样；192 MiB staging 峰值仍受上传批次限制。

额外 OMM 回归 `opacity_micromap_ray_query` 的 `initial alpha mask incorrect` 断言失败；9 月 14 日保存的修改前程序在当前 shader/环境下也有相同失败。记录为未解决的既有兼容/测试问题，不能把这一项记作通过；并非同 HEAD 的干净构建对照。[当前单测](../build-release/gpu-memory-budget/omm-absolute.log)、[旧程序对照](../build-release/gpu-memory-budget/omm-before.log)。

复跑压力测试（PowerShell，从仓库根目录）：

```powershell
$env:METALLIC_ZORAH_Z3_FULL = '1'
$env:METALLIC_TEST_TEXTURE_SHARED_MIB = '512' # 或 768
& build-release/tests/MetallicRhiTests.exe --rhi-bindless --rhi-validation `
  '--gtest_filter=*zorah_texture_resources' `
  --output-dir E:/metallic/build-release/gpu-memory-budget/recheck
```

## 后续边界

T0 实现统一观测、预留、纹理加载前降档和逐分配拦截。尚未实现纹理可见性反馈、运行中高 mip 升降级、冷纹理回收、MASK 基色单独保真下限或材质 image 子分配。低预算下当前统一 cap 也会影响 MASK、法线和粗糙度，质量不能仅凭采样正确就验收。

固定 geometry/CLAS 池没有自动缩容；未来 CLAS/RTAS/scratch 增长会被同一分配检查限制，尚无完整工作集自适应规划/增长额度调度。预算发生外部突变时可能在加载中途拒绝，尚无整套纹理自动销毁降档重试。下一步按 T1 处理逐图尾链质量和小纹理子分配，再用完整 Full 图验收首帧、DLSS 和 Mini↔Full 切换。