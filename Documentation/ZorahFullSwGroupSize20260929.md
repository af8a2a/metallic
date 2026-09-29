# ZorahFull：NVIDIA wave32 下的 32 / 64 / 128 线程组实验

2026-09-29。**32 线程组在本轮 Full 固定视角的三个独立进程中均最快。** 相对生产 WorkControl 128 线程，SW 均值下降 28.3–32.1%，约 2.01–2.26 ms；RenderGraph GPU 均值下降 12.9–17.9%。这是冻结状态的结果，不是持续漫游 FPS 验收。生产默认保持原有 128 线程，实验入口仅在显式启用时编译和使用。

## 实验条件

- RTX 5070 Ti；Vulkan 设备查询固定 subgroup min=max=32，驱动实际 pipeline 统计也为 subgroup 32。
- Full 完整图，输出 1797×660，DLSS Quality 内部 1198×440；LOD 1.5 px、SW 阈值 8 px；关闭 jitter，graphics queue 串行 HW/SW。
- 使用用户截图相机 Eye=(6.456737,4.134918,-7.741189)，Center=(-2.162123,2.872261,-5.524320)，FOV 60，reversed Z。
- 保留磁盘和 shader 缓存。每个进程待场景 ready 后预热 15 秒，冻结 geometry/CLAS/cut/TLAS 和纹理发布；每模式恢复 32 帧后采样 128 帧，三轮换序，每进程 1536 帧，三个独立进程共 **4608 帧**。
- 顺序：生产128→32→64→循环128；循环128→64→32→生产128；64→生产128→循环128→32。统计单位为三个独立进程的模式均值及对应收益，不把 4608 帧当作独立实验。
- 同一进程内检查 camera、内部尺寸、cut、page mappings、每帧驻留、软件列表、indirect args 和实际 module/entry/SPIR-V。三个进程恰好得到相同 cut `2103523889419739130`、page mappings `9544736054938858958`，207244 active groups；分析器不依赖跨进程 cut 相同才成立。
- 正式采样关闭 validation、pipeline statistics、NvPerf、GPU Trace、Graphics Capture 和 shader trace。短程 validation/资源统计 pilot 与正式时间分开。

## 结果

SW 为 early+late 软件 scope 之和。本视角 late 软件列表为空，少量空 dispatch 时间包含在内。

| 路径 | 进程1 SW ms | 进程2 SW ms | 进程3 SW ms | Graph GPU 均值范围 ms |
|---|---:|---:|---:|---:|
| 生产 WorkControl 128 | 7.063 | 7.092 | 7.111 | 12.698–12.785 |
| 新跨步循环 32 | **4.799** | **4.937** | **5.099** | **10.470–11.139** |
| 新跨步循环 64 | 5.263 | 5.494 | 5.493 | 10.922–11.502 |
| 新跨步循环 128，控制组 | 7.201 | 7.417 | 7.044 | 12.620–13.203 |

32 线程相对生产 128 的三进程 SW 收益为 32.05%、30.39%、28.29%，平均 **30.24%**。以进程配对收益计算、df=2 的双侧 Student-t 95% 区间为 **25.55–34.93%**。Graph 平均收益为 15.68%，区间 9.33–22.03%。样本数只有三个，这不是跨场景、跨驱动或漫游收益的置信区间。

64 线程相对生产 128 的 SW 平均收益为 23.59%。在相同新循环实现内部，32 对循环128的 SW 平均收益为 31.46%，所以不能把改善全部归因于替换原入口后的代码布局变化。32 在三个进程里也都快于 64。

32 的 SW P95 为 5.019 / 5.420 / 5.798 ms；生产 128 为 7.333 / 7.740 / 8.054 ms。完整每模式均值、p50/p95/p99、逐轮结果与原始帧均保留。

## 实现与正确性

[实验 shader](../Shaders/Features/GPUDriven/GPUDrivenStreamGroupRaster.slang) 保持一组对应一个 cluster，首 wave 装载、两次组同步、共享屏幕顶点、整数边函数、原深度表达式和 64-bit depth/ID atomic 均保留。顶点与 triangle 使用组大小为步长的循环；visibility ID 使用原始 triangle 索引，不使用重排后的 lane。

额外保留循环128控制组，区分线程组大小和代码组织的影响。原生产 [WorkControl shader](../Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang) 未修改。新三条入口由 `METALLIC_SW_GROUP_EXPERIMENT=1` 和隐藏的 benchmark 属性启用，普通启动不会编译新增 pipeline。硬件不满足固定 wave32 时拒绝实验。

- validation pilot：四条路径各 8 个计时帧，仅作诊断，不纳入性能结论。depth/visibility 原始字节、软件列表、dispatch、cut 与驻留一致，无 VUID / DeviceLost。
- 正式三进程：四条路径、全部前后检查点的 depth/visibility **逐位一致**；跨轮图像稳定，工作列表/间接参数相同，驻留不变，实际绑定四个不同 SPIR-V。所有进程正常导出并完成 shutdown，没有强制 teardown。
- 生产默认未替换；本轮没有验证持续漫游、非零 late 工作负载，也没有新增覆盖全部 0/31/32/33/63/64/65/127/128 边界的生产 streaming GPU 夹具。这些属于推广前的后续验收，不能从单视角等价自动推导。

正式 Full 对照首次短测发现旧分析器假定恰好两个 Stream 父 scope，实际已有同名 stage/helper 嵌套。现在选取最外层 early/late 区间，避免重复计算；增加单层、嵌套和缺失 GPU 结果三个回归。原 pilot 的采样成功、首次分析失败日志保留，修正分析器后重新离线分析，没有伪造一次新的 GPU 运行。

## 驱动资源与工作量

资源数据来自独立诊断 pilot，按实际绑定 SPIR-V 对齐正式采样。以下不是 occupancy 或运行时 stall 计数。

| 路径 | 寄存器 / 线程 | Shared B / 组 | 机器码 B |
|---|---:|---:|---:|
| 生产128 | 37 | 3656 | 22656 |
| 循环32 | 45 | 2240 | 22528 |
| 循环64 | 40 | 2240 | 23296 |
| 循环128 | 40 | 2240 | 23296 |

Local Memory Size 返回 68719476736 / 68719476752 一类异常大值，原样存档，不据此声明零 spill 或 16 B spill。

当前相机的 baseline coverage replay 记录：1,072,251 SW clusters，44,232,225 triangles，107,281,579 cluster 内唯一顶点处理，3,952,664 bbox visits，749,323 覆盖/原子尝试。约 42,394,695 triangles 在进入 bbox 扫描前被拒绝，包含背面、退化与空 bbox，尚未拆分原因。

**更小线程组有利于当前工作量，这是本轮实测结论；具体硬件原因仍是待验证假设。** 32 线程并非因为寄存器更少而获胜。它让描述装载时整组都在首个 warp，并减少低 triangle 填充时的空线程；但同时增加顶点/triangle 批次数，实际等待和 occupancy 尚未采集。

诊断 replay 固定使用 128 线程；其 `launchedTriangleLanes`、`sumWaveMaxRows` 等只描述诊断 kernel，不能拿来计算新 32/64 线程 kernel 的真实 lane 利用率。覆盖/triangle 总量可以用于同工作量检查。

测量期间有浏览器 video-decode 和桌面等背景活动：各进程检测到的视频解码峰值约 4.5% / 18.2% / 11.3%，copy 引擎也有活动。三个进程结果方向一致，但没有声称整卡实验室独占；逐进程、逐引擎原始监控保留在证据中。

## 验证、证据与复现

- Release `MetallicGPUDrivenSample`、`MetallicRhiTests` 构建通过：[构建日志](../build/sw-group-build.log)。
- 分析器三项 CPU 回归通过；Vulkan validation 的 `hybrid_raster_depth_coverage_and_overflow` 通过：[RHI 日志](../build/sw-group-rhi-regression.log)。这项是已有光栅回归，不冒充新增组大小边界覆盖。
- `stream_cluster_cull_classify_equivalence` validation/bindless 回归通过：[分类日志](../build/sw-group-classify-regression.log)。
- [正式完整证据](../build-scheduling-release/sw-group32-64-128-formal-01/Review.json)、[原始 Manifest](../build-scheduling-release/sw-group32-64-128-formal-01/Manifest.json)、[诊断 pilot](../build-scheduling-release/sw-group32-64-128-pilot-01/run1/Capture.json)、[实际资源日志](../build-scheduling-release/sw-group32-64-128-pilot-01/run1/stdout.log)。
- 正式目录封存原始图像、帧数据、监控、日志、可执行文件与测量后的源码快照，共 352 个文件；已用当前 verifier 复核文件哈希和原始图像字节：[复核结果](../build/sw-group-final-verification.json)。源码快照不是测量前完整 inventory；原 runner 验证了整批前后的 EXE 与 shader 摘要不变。资产只记录路径、大小/mtime，没有重新哈希或复制 205 GB stream，也不是可携带的完整场景快照。

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-scheduling-release/sw-group-new `
  -Executable build-scheduling-release/Source/MetallicGPUDrivenSample.exe `
  -SwGroupComparison -Runs 3 -Rounds 3 -SampleFrames 128 -SettleFrames 32 `
  -WarmupSeconds 15 -Width 1797 -Height 660 -RouteConfig build/SwGroupCamera.json

python -B Tools/Perf/AnalyzeSwGroupComparison.py `
  build-scheduling-release/sw-group32-64-128-formal-01 --verify
```

下一步优先推广 **32 线程候选的正确性覆盖与非零 late / 漫游对照**，通过后再考虑默认启用。暂不叠加去同步、wave shuffle、scanline 或浮点深度改动，避免失去本轮收益归因。
