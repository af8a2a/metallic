# MiniZorah：Stream 遍历与光栅队列成本

2026-09-16，基线 `b34040a`，RTX 5060 / 8 GiB / 驱动 610.47。

## Scope 的含义

截图里的 `Stream early/late` 包含候选生成、cluster 剔除、软硬分类、实际光栅和合并，并非页面流送管理。它们应与 vk_lod_clusters 的 `Render` 中对应工作比较，不能对比其 `Stream Begin/End`。截图中的参考程序还使用 ray tracing renderer，完整 pass 也不是相同工作量。

原来的 stream 两轮都创建独立 compute 分支，父 scope 的时间戳横跨 graphics / compute 的提交与汇合，因此包含调度和等待时间。子 scope 可重叠，不能简单相加当作父 scope。本机后台 Unity 的持续负载会显著放大这些时间；它不能解释所有遍历开销，但足以使同一构建的 VBuffer 耗时波动数倍。

`Stream traversal` 则有真实计算瓶颈：在固定路线的 `return_hold`，LOD frontier 约 1.13 ms，prefix 约 0.08 ms，emit 约 0.09 ms。最终访问 129,332 个 tile 层级节点，测试 80,813 个 group。frontier 仍为每实例一个 64 线程组；有序拓扑使同层 tile 顺序执行，大 primitive 的工作量不能在多个线程组间分摊。本地 `E:/vk_lod_clusters/shaders/traversal_run.comp.glsl` 的任务队列、subgroup 子任务分配是后续需要补齐的组织方式。本次没有实现该任务队列，不能声称已达到截图中的 0.084 ms。

## 改动

- 两个 MiniZorah sample 图默认 `asyncSoftwareRaster=false`，软光栅 compute 与硬件光栅使用 graphics 队列。保留 cluster 分箱、混合软硬光栅、两轮保守剔除和相同画质参数。
- 新增 `asyncLateRaster`，默认 false。开启 `asyncSoftwareRaster` 时，stream early 允许异步；late 需单独开启，适合对比少量补绘是否值得跨队列提交。纯流送默认异步分支由 2 个变为 0 个；early-only 为 1 个，全开为 2 个。resident 生产者保持原有队列控制方式。
- cooperative frontier 将设备内存屏障由每个叶子 tile 一次改为相邻 LOD 层之间一次，mask 计算前仍完整同步。同级 tile 不存在父子依赖，父级 active 状态只会被更细层读取。groupshared prefix 的同步仍然存在，首帧初始化与上一帧 active 状态清除也保留。
- 回放脚本增加 `-RasterQueues Default|Off|Early|All`，记录实际策略并验证对应 Software raster scope 的队列；CPU 录制与 GPU 执行的帧间重叠仍单独验证。

32 线程 / wave-prefix 和共享 active 位图方案均做过实验，未保留：前者 frontier 变慢，后者没有稳定收益且 emit 变慢。这些中间构建不用于本次最终收益或正确性结论。

## 回放方法

使用仓库 `Tools/RunMetallicCfgReplay.ps1` 与 `.cache/gpudriven-four/Replay.json`，每轮完整 3,000 帧、10 个阶段，按帧步进、无 present。实时管线输出 1920×1080，DLSS Quality 内部分辨率 1280×720，LOD 1.5 render px，几何/CLAS/动态 BLAS 预算 1024/512/256 MiB。CLAS、阴影、延迟照明、DLSS-SR、曝光等沿用同一配置。

Replay SHA-256：`feb79872154850af32db25a54ba3d22b48b9a04a10f7f2e8dadaf19f98f2f2d2`。可执行文件、shader、harness 和路线哈希、完整分布与后台竞争统计见 [StreamRasterCostResults.json](StreamRasterCostResults.json)。原始帧与 GPU 监控位于 `.cache/stream-cost/`。

所有表格均取 `return_hold` 的全部 300 帧，单位 ms，没有剔除异常帧。后台进程未暂停，测得的差异不能视为隔离环境下的固定提速比。

## 计时结果

| 构建 / 队列策略 / 轮次 | VBuffer | Traversal | Frontier | Early | Late | Host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 / 全异步 / m1 | 4.023 | 1.783 | 1.132 | 1.754 | 0.221 | 8.061 |
| 基线 / 全异步 / m2 | 4.026 | 1.789 | 1.132 | 1.753 | 0.223 | 8.048 |
| 仅按层同步 / 全异步 / m1 | 4.036 | 1.770 | 1.111 | 1.763 | 0.225 | 8.225 |
| 最终 / 默认串行 / 第一组 m1 | 3.936 | 1.772 | 1.112 | 1.762 | 0.148 | 8.081 |
| 最终 / 默认串行 / 第一组 m2 | 8.447 | 1.795 | 1.116 | 3.617 | 0.150 | 19.742 |
| 最终 / 全异步 / m1 | 4.068 | 1.766 | 1.110 | 1.756 | 0.223 | 8.206 |
| 最终 / 全异步 / m2 | 4.003 | 1.761 | 1.111 | 1.755 | 0.223 | 8.076 |
| 最终 / 默认串行 / 第二组 m1 | 8.402 | 1.793 | 1.116 | 3.599 | 0.150 | 20.045 |
| 最终 / 默认串行 / 第二组 m2 | 3.927 | 1.713 | 1.109 | 1.759 | 0.147 | 8.069 |

同一最终构建的全异步两轮 late 中位数为 0.222–0.223 ms，默认串行四轮为 0.147–0.150 ms。early 在较低竞争的阶段约 1.76 ms，两种队列方式基本相当；同队列 Software raster 子 scope 约 0.370 ms，全异步约 1.05 ms，但它们的并行关系不同，不能将子 scope 差值当作整个 early 的节省。

仅按 LOD 层同步时，frontier 由约 1.132 ms 变为 1.111 ms，观察到约 2% 的小幅改善。最终各轮 frontier 约 1.109–1.116 ms。大量跨实例与大 primitive 遍历工作仍然存在，本次没有消除主要算法差距。

后台负载在这些回放中反复变化：同一默认策略 VBuffer 中位数分别为 3.936、8.447、8.402、3.927 ms；本轮不报告总体提速百分比。所有重复轮次都保留在表格与 JSON 中，包含慢轮次。`GpuProcesses.csv` 记录的是非空进程/引擎采样，不能解释为精确的整卡占有率，也不能用全程采样中位数替代对应阶段的竞争强度。

## 正确性与验证

- Release `MetallicRhiTests` 与 `MetallicGPUDrivenSample` 构建通过。
- `meshlet_lod_stream_gpu_matches_reference` 开启 Vulkan validation 通过：共享父级、511 groups、新增 8,191 groups 共 545 组 GPU/CPU 对比，包含线性/BVH/cooperative、视图需求、预取、连续驻留变化、稀疏状态清理与容量回退。大拓扑覆盖同一 LOD 层的多个叶子 tile。
- `meshlet_lod_stream_scene_runtime_cut` 开启 validation 通过：真实 Bunny 的硬件、混合串行、仅 early 异步、两轮异步四种队列配置；对比完整 cut、有效 ID、深度、投影、两种 Z 和冻结相机，检查精确异步分支数量。
- `meshlet_lod_stream_per_instance_budget`、`hybrid_raster_scene_equivalence`、`render_graph_gpu_driven_mixed_producer_render` 开启 validation 通过。混合生产者默认执行 resident early/late 与 stream early，共三个异步分支。
- 最终默认策略另跑独立 3,000 帧质量回放：第 29 帧仍有未到达的细节，第 59 帧及之后所有检查点的可见超目标细化数量为 0，最终最大可见细化误差 1.499928 px。

最终 cut 与基线全部字段一致：19,394 active groups、162,989 selected clusters，early 26,806 hardware / 19,172 software，late 200 hardware / 361 software，容量回退实例数为 0。最终几何驻留 150,612,224 字节、9,290 页。队列优化没有通过少绘制几何获得较低时间。

专项测试日志：`.cache/stream-cost/final-tests.log`（四项通过，新夹具参数未修正时 oracle 失败）；修正后的完整 oracle 日志为 `oracle-final.log`，最终通过。Bunny 输出在 `final-tests/StreamMeshletLodSceneReport.json` 和同目录 PNG。新夹具曾沿用 511-group 的重置尺寸与粗 LOD 阈值，现已按叶子数量构造并缩放粗阈值，未放宽 CPU/GPU 一致性断言。

完整实时回放关闭 Vulkan validation，专项测试开启。Streamline 在报告与测试结束标记写完后的进程退出可能停滞，脚本仅回收自身启动的进程，记录于 `Process.json`；退出清理不计入帧时间。

```powershell
# 默认 graphics 队列混合光栅；quality 使用单独回放
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/stream-cost/repeat-default -Realtime -QualityWithoutValidation
# 同一构建开启两轮异步，对照队列影响
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/stream-cost/repeat-all -Realtime -RasterQueues All -Cases m1,m2
```
