# ZorahFull：HZB 历史失效判定修复

2026-09-20。修复普通相机运动被当作 camera cut 的问题。构建、定向回归和 Full 对照已通过；30 秒漫游均值改善至 49.68 ms，仍未达到至少 30 fps。

## 修改与边界

`VisibilityBufferPass` 改为观察 `HistoryResourceManager::reprojectionInvalidationRevision()`，而不是通用 `invalidationRevision()`。普通 `CameraMotion` 仍清空渐进累积，但不会丢弃可用上一帧相机重投影的 HZB。相同相机重复设置也保留 HZB。

保留既有保护：显式 camera cut、投影类型/正反深度/近远裁剪面变化、帧序中断、渲染或输出尺寸变化、jitter 模式切换、场景 discontinuity，以及 GPUScene 的场景历史 epoch 和冻结剔除相机切换。移动后被遮挡历史误拒绝的几何仍通过 late pass 重新测试和补绘。没有修改剔除阈值、光栅算法、LOD 误差、流送预算或 GPU 同步策略。

涉及代码：

- [VisibilityBufferPass.cpp](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)：采用重投影失效版本，重命名对应缓存字段。
- [RenderViewTests.cpp](../tests/rhi/RenderViewTests.cpp)：补充重复相机、平移、帧间断、渲染/输出 resize、投影/深度变化，以及渐进累积与重投影历史的区别。
- [EditorRasterComparison.cpp](../Source/Editor/EditorRasterComparison.cpp)、[AnalyzeZorahFullRasterComparison.py](../Tools/AnalyzeZorahFullRasterComparison.py)：导出 `historyInvalidationPolicy=reprojection-v1`。新捕获逐帧验证 HZB 有效、同相机 bins/工作量/图像相同；旧因果捕获继续按原预期分析。

## 冻结同状态对照

RTX 5070 Ti，输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px，8 px 分流。每次运行内保持同 camera、cut、页映射和几何/CLAS/纹理驻留；关闭 jitter，HW/SW 串行，两轮换序，每模式 32 个计时帧。诊断计数帧排除在耗时外。本次开启 Vulkan validation。

| 指标 | 修复前：保持相机 | 修复前：重复设置 | 修复后：保持相机 | 修复后：重复设置 |
|---|---:|---:|---:|---:|
| 参数 HZB valid | 1 | 0 | 1 | 1 |
| Early SW clusters | 1,041,657 | 4,598,812 | 1,037,662 | 1,037,662 |
| Early SW 三角形 | 42,565,256 | 188,582,571 | 42,244,207 | 42,244,207 |
| bbox 访问 | 3,203,542 | 7,245,368 | 3,077,995 | 3,077,995 |
| 覆盖样本/原子尝试 | 638,149 | 1,420,953 | 612,525 | 612,525 |
| SW GPU，early+late ms | 9.906 | 43.015 | 9.714 | 9.750 |
| RenderGraph GPU ms | 17.881 | 51.439 | 17.479 | 17.408 |

修复后两种 SW 模式的全部计数、bins、visibility 与 depth 逐位一致，两轮一致；96/96 个计时帧（含 Full HW）HZB 参数有效。实际 shader 仍为 `GPUDrivenStreamWorkRaster.streamClusterRasterWorkControlMain`，mode 5，SPIR-V FNV1a64 `14699073418322314354`。

本次 cut=`7230532556492933287`，page mappings=`6928581180108922903`，221402 active groups。前后两次运行的预热驻留略有差异，因此最严格证据是各自运行内部的 A/B：相机重复设置造成的约 4.34 倍工作放大已消失，而不是将跨运行的每个数值都当作完全同状态比较。

## 不带计数的正常编辑器漫游

前后 `config`、绝对相机关键帧、显示/内部尺寸、完整 graph、VSync、隐藏编辑器运行方式和 validation 设置均一致：10 秒预热、30 秒路线，前进 6 个场景单位并左右转向。无诊断重放，无缺失 GPU 帧。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 采样帧数 | 454 | 604 |
| HZB 有效帧 | 0/454 | **604/604** |
| Early SW GPU ms | 28.540 | 9.201 |
| Late SW GPU ms | 0.005 | 0.685 |
| SW 总 GPU ms | 28.545 | **9.886** |
| RenderGraph GPU ms | 42.176 | **25.616** |
| CPU Prior frame drain ms | 33.768 | 16.739 |
| 整帧均值 ms | 66.132 | **49.685** |
| 整帧 P95 ms | 75.905 | 61.036 |
| 整帧最大 ms | 105.415 | 71.892 |
| 超过 33.33 ms 帧数 | 454 | 604 |

SW 总耗时降低约 65.4%，整帧均值降低约 24.9%（约 15.1 → 20.1 fps）。这是相同路线配置下不同运行的观测值，非每帧相同 cut/驻留或固定 GPU 时钟实验。late pass 有实际补绘成本，不能只报 early SW。

当前仍未达到持续 30 fps。604 帧 `drainReasonMask` 均为 4（external completion），修复减少了被等待的 GPU 工作，但没有消除逐帧 CPU drain。下一步应处理编辑器输出完成依赖与 CPU 等待，再测剩余 CPU 录制成本；不能直接删除资源生命周期保护。

## 验证与数据

- `MetallicGPUDrivenSample`、`MetallicRhiTests` Release 构建成功。
- `render_view_shared_constants_history --rhi-validation`：通过。
- `gpu_driven_temporal_occlusion_equivalence --rhi-validation`：30 个移动/jittered 视图与关闭遮蔽参考逐像素一致，包含切镜头、resize、正交和正反深度。
- `gpu_scene_cpu_core --rhi-validation`：通过，包含场景、View、resize、freeze/cut 历史隔离。
- `stream_cluster_cull_classify_equivalence --rhi-validation`：通过。
- Full 冻结验证：无 VUID/DeviceLost；图像、同状态和计数断言全部通过。旧捕获兼容性分析通过。
- Full 30 秒漫游：完整结束，无 DeviceLost；不代表已完成长期稳定性验收。

原始文件：

- 修复前因果对照：`build-release/full-sw-history-causal/run1/`。
- 修复后因果对照：`build-release/full-hzb-history-fixed/run1/`。
- 修复前普通漫游：`build-release/full-sw-workload-roam/run1/`。
- 修复后普通漫游：`build-release/full-hzb-history-roam/run1/`。
- 各目录保留 Capture、Summary、GPU 占用/竞争监控和日志；冻结目录含逐像素读回及 WorkloadSummary。
- [精简机器可读结果](ZorahFullHzbHistoryResult.json)。
