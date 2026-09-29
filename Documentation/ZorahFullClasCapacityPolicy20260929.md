# CLAS start / grow / max 与空块回收（2026-09-29）

后续根页/动态页分池和根页独立增长粒度见 [生命周期打包](ZorahFullClasLifetimePacking20260929.md)。

## 范围与参考

主线已有实际尺寸 CLAS MOVE、按需分段 Buffer 分配和 GPU 完成后空块释放，见 [原实现记录](ZorahFullClasDemandAllocation20260929.md)。本次补齐独立容量策略、回收滞后及观测；不把原来从整池预分配改为按需分配的收益重复计为本次收益。

参考本地 `E:/vk_lod_clusters/src/scene_streaming_utils.cpp` 的 `StreamingResident::initClas` / `growClas`：start 决定初始提交量，max 决定上限，后续扩容通过 large buffer 的物理块增长。Metallic 沿用已验证的分段池，不增加 sparse RHI。每页落在一个 Buffer 内，地址表仍发布绝对设备地址，增长不搬移活页，不改变稳定 cluster ID。

## 配置

配置由 `StreamerSubsystem` 经 `SceneStreamingConfig` 传入运行时，两种场景入口均支持。RenderPass 无新增加载职责。以下属性仅影响 `compactClas=true` 的持久 CLAS backing，旧固定槽构建池和 scratch/页表/地址表不受影响。

| Graph 属性 | MiniZorah / Full 预设 | 含义 |
|---|---:|---|
| `startClasBytes` | 0 | 初始化物理提交量；非永久保留底线，可在空闲时归还至零 |
| `growClasBytes` | 67,108,864（64 MiB） | 需要新增块时的目标大小 |
| `maxClasBytes` | 保持原值 | 单池物理 backing 上限，同时受设备联合预算约束 |
| `clasEmptyChunkRetentionFrames` | 30 | 块完全空闲且所有 GPU 使用结束后，额外保留的流送维护帧数 |

通用描述符默认仍为 start=0、grow=64 MiB、retention=0，旧调用方保持立即回收行为。start > max、grow=0 拒绝初始化。max 向下对齐设备要求；start/grow 向上对齐并限制在有效 max 内。非零 start 按 grow 分段申请；初始化失败释放部分已提交 backing。

分配优先利用已有块。页面大于 grow 时申请足以容纳该页的块；末块受 max 剩余额度限制。初始块释放后，后续按 grow 重新增长，不重复恢复 start。

## 安全回收

- 页面仍沿用退休宽限期与 GPU 完成条件；只有子分配归零的块才候选回收。
- 在途命令、未提交命令的资源租约仍阻止 Buffer 释放，取消后才允许回收。滞后倒计时从确认安全空闲后开始。
- 新分配复用空块时清除空闲时间。到期释放 Buffer，并释放池中的持有引用。
- 当保留的小空块占用了 max 额度、无法容纳较大页面时，提前释放安全空块后再申请。此时只跳过滞后时间，绝不跳过 GPU 完成条件。
- 全局预算拒绝增长的重试抑制继续有效；释放空块后可以提前重试。

本次没有跨块搬移存活 CLAS；部分使用的块不会归还。若根 cut 或长寿页散布于多数块，仍需后续按生命周期分池或整理活页，才能回收这些碎片。

Profiler / 漫游 JSON 新增 `clasStartBytes`、`clasGrowBytes`、`clasEmptyBytes`、`clasGrowthCount`、`clasReleasedBytes`。增长次数与释放字节是从池初始化开始累计，初始申请也计为增长。empty 表示无子分配的整块字节，可能还在等 GPU 完成或滞后到期，不代表立即可释放。`clasAllocatedBytes` 是 Buffer backing，`clasBytes` 是子分配，`clasCapacity` 是预算；均不等于整卡 NVML 用量或 VMA 物理堆释放量。

## 验证

MSVC Release 构建 `MetallicGPUDrivenSample`、`MetallicRhiTests` 通过，沿用既有构建树及 SDK。

7 项 GPU 测试通过，无跳过，启用 Vulkan 验证：

- `RhiResource.clas_actual_sizes_and_move`
- `RhiResource.clas_compact_lifecycle`（最终二进制重跑）
- `RhiRendering.minizorah_clas_in_flight`（1,200 帧）
- `RhiRendering.stream_clas_runtime_lifecycle`
- `RhiRendering.stream_clas_eviction_reupload`
- `RhiRendering.zorah_full_first_frame`（一次 MiniZorah→Full 切换）
- `RhiRendering.streamed_realtime_pipeline`

新增生命周期断言覆盖独立 start/grow/max、未到期保留、到期释放初始块、非法参数、容量压力下提前释放小空块以容纳实际 CLAS 页、增长/释放统计。既有断言继续覆盖实际尺寸 MOVE、跨块增长、活页地址稳定、未提交帧阻止释放、取消/复活/重上传及预算不足。

Full 输出为 960×540 原生分辨率、关闭 DLSS 的测试路径。检查 settled 图像无新增缺块；与 `incremental-residency-epoch-full` 基准相比，settled 图像 2 个像素不同，base-color 1 个像素不同。这不代表长时间 DLSS 漫游的逐像素稳定性验收。

证据目录：

- `build-scheduling-release/clas-capacity-validation/` 与同名 `.log`
- `build-scheduling-release/clas-capacity-full/` 与同名 `.log`
- 各目录 `reports/` 保存测试报告；Full 目录另有 `ZorahFullFirstFrame.json` 及 PNG。

## Full 固定漫游观测

使用 `Tools/RunZorahFullRoam.ps1`，现有 `blas-selected-normal-route.json`，固定 180 帧路线、逻辑时长 30 秒、预热 3 秒、1 轮；输出 1797×660、渲染 1198×440、DLSS Quality、LOD 1.5 px。新进程启动、已有磁盘 cook/shader/纹理缓存；未修改相机、cut 选择或质量参数，不是冷磁盘测试。

| 指标 | 本次观测 |
|---|---:|
| start / grow / max | 0 / 64 / 2,048 MiB |
| 采样持久 backing | 1,792 MiB，28 块 |
| 累计增长次数 | 28 |
| 采样空块 / 累计归还 | 0 / 0 |
| 请求溢出 / I/O 失败 / BLAS overflow | 0 / 0 / 0 |
| 几何 allocationFailures | 每帧 1（预算准入拒绝计数） |
| 帧时间均值 / P95 | 23.215 / 31.040 ms |
| 超过 33.33 ms | 5 / 180 帧 |
| NVML 整卡峰值 | 12,940 MiB |

这段路线没有形成完整空块，所以本次 Full 运行没有发生物理归还；归还正确性由实际 GPU 生命周期测试证明。不可把 2,048−1,792 MiB 当作本次相对主线新增节省，也不能据单轮帧时间/NVML 对旧报告宣称性能或峰值改善。若继续降低该路线驻留，需要先处理散布于块中的长寿页和根 cut，而非仅调小 grow。

证据：`build-release/clas-capacity-roam/{Manifest.json,Config.json,run1/Frames.jsonl,run1/Capture.json,run1/Gpu.csv,run1/Summary.md}`。
