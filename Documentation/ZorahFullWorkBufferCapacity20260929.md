# ZorahFull 分类工作缓冲容量解耦（2026-09-29）

## 实现范围

将分类候选、稳定分桶、分类描述和重试掩码使用的工作缓冲，与完整 cut 的可见记录地址空间解耦。新增 `maxRasterCandidates`，默认 **8,388,608**；Full 图中显式设置相同值，`0` 恢复原尺寸以便对照。

解析后的容量为 `min(visibleRecordCapacity, max(activeGroupCapacity, maxRasterCandidates))`。组容量下限用于保障组掩码、前缀计数和两遍 HZB 重试存储；非流式居民几何仍按自己的记录需求分配。视图重绑定同时处理增容和缩容。

以下容量未缩减：

- `maxActiveGroups` 与遍历/完整 cut。
- 按 `activeGroupIndex * maxActiveGroupClusters + clusterIndex` 寻址的可见记录（Full 约 512 MiB）。紧缩该地址空间需要另外修改索引映射，不能直接截断。
- BLAS 引用预算；其已有独立 `maxBlasClusterReferences`，不借助本次光栅队列预算限制它。
- 几何、CLAS、纹理驻留预算及 LOD 质量。

容量与配置由 Streamer 运行时维护，VBuffer 只读取已解析容量来建立光栅工作资源，没有新增 RenderPass 资源加载逻辑。

## 超额处理

GPU 展开前先计数。超过工作容量时，不写入部分候选，不执行该阶段的剔除/分类/SW 分桶队列；保留该阶段的完整组选择掩码，生成覆盖原始记录空间的 HW 间接绘制。HW 按掩码筛选，执行正常可见性检查和记录发布；早阶段仍发布 HZB 重试掩码，晚阶段继续消费它。普通 mesh 与 tessellation task 路径均接入此回退。

稳定 ID 不随工作队列容量重编号；无需 CPU 读取候选数再等待重分配。超额回退保证完整性，但可能明显增加 HW 工作量，因此它是保护路径，不是正常调度目标。

诊断新增 `candidateOverflow`、`requestedCandidates`、`hardwareFallback`、`hardwareCountIsRecordSlots` 与实际 `bufferBytes`。回退时 HW 计数代表扫描的记录槽位，不能当作实际可见簇数。Streamer 快照也导出组、可见记录、候选及 BLAS 引用容量。

## 容量选择与实测

先试的 4M 队列在 Full 固定路线中出现 4,328,033 个候选，12 个阶段采样中有 2 个超额回退，因此没有采用为默认值。原尺寸复测候选峰值达到 4,520,999；最终保留 8M 余量。

最终对照使用同一 EXE、同一 shader 摘要、相同绝对相机关键帧；RTX 5070 Ti，编辑器输出 **1797×660**、内部渲染 **1198×440**、DLSS Quality。每组 180 个固定路线帧，6 m 往返与转向，ready 后 warmup 3 秒；文件/着色器缓存未清空。每 30 帧抽样早/晚阶段 header，不执行 SW 工作量重放。Full 漫游未开启 Vulkan validation；独立 GPU 回归开启 validation。

| 指标 | 原尺寸（0） | 8M |
|---|---:|---:|
| 实际候选容量 | 33,554,400 | 8,388,608 |
| 实际工作 Buffer 字节 | 1,213,201,344 | 303,300,672 |
| 实际工作 Buffer MiB | 1,156.999 | 289.250 |
| 整卡峰值 MiB（含桌面/其他进程） | 14,341 | 13,482 |
| 监控首个样本 MiB | 3,799 | 3,759 |
| 候选抽样峰值 | 4,520,999 | 4,344,910 |
| 超额回退 / 阶段样本 | 0 / 12 | 0 / 12 |
| 完成漫游帧数 | 180 | 180 |
| 帧时间均值 ms | 33.817 | 33.240 |
| 帧时间 P95 ms | 69.350 | 47.420 |

工作 Buffer 明确减少 **909,900,672 字节，即 867.749 MiB**。这次整卡峰值下降 **859 MiB**，与缓冲差值接近，但包含其他分配、桌面及采样误差，不能视为仅该 Buffer 的精确贡献。

这是每组单轮的内存专项验证，cut/驻留持续变化而非冻结。不能把 P95 变化直接归因于本次优化，也没有达到全程 30 fps。12 个阶段样本不是每一帧的溢出证明；极端视角仍受完整 HW 回退保护。

## 验证与证据

构建：沿用 MSVC Release 配置，没有更改 SDK、编译器或构建选项。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample -j 6
cmake --build build-scheduling-release --target MetallicRhiTests -j 6
.\build-scheduling-release\tests\MetallicRhiTests.exe --gtest_filter=RhiRendering.stream_cluster_candidates_stable_parallel:RhiRendering.stream_indexed_mesh_raster_equivalence:RhiRendering.stream_cluster_cull_classify_equivalence:RhiRendering.hybrid_cluster_stable_bins_and_indirect_limits:RhiRendering.tessellation_displacement_render --rhi-validation --output-dir build-scheduling-release/work-capacity-final
```

5 项 GPU 回归通过，无跳过。包括 17 个候选展开用例（恰好满、超出一项、空/缩短列表、早晚阶段、超过 128 个块、尾部 poison 保护）、完整掩码恢复与 HW 间接参数；32 组 HW/超额回退/旧队列的深度和 32-bit 可见 ID 逐位对照；分类等价性、非流式稳定分桶、位移细分渲染。检查了输出的流式位移图像。普通 mesh 超额路径有逐像素回归；位移测试验证常规 stream/resident 路径，没有单独强制 tessellation 超额场景。

本地证据（构建目录，不纳入源码）：

- `build-scheduling-release/work-capacity-final/results.json` 与同目录报告。
- `build-release/work-capacity-comparison.json`：两组 Buffer 实测大小、峰值、哈希与相机路线核对。
- `build-release/work-capacity-full-legacy/run1/{Capture.json,Frames.jsonl,Gpu.csv,Summary.md}`。
- `build-release/work-capacity-full-8m/run1/{Capture.json,Frames.jsonl,Gpu.csv,Summary.md}`。
- `build-release/work-capacity-full-bounded-run/run1/Capture.json`：4M 容量不足的诊断证据。

复现用 `tools/RunZorahFullRoam.ps1`（PowerShell 7），RouteConfig 可覆写 `maxRasterCandidates`。固定参数 `routeFrames=180, workloadEvery=30, softwareWorkloadCounters=false, classifyCounters=true`，分别设置容量为 `0` 和 `8388608`。不要并行跑两组 GPU 工作负载。
