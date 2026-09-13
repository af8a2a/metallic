# MiniZorah Profiler 与流送图表

2026-09-13。参考本机 `E:/vk_lod_clusters/src/lodclusters_ui.cpp` 的分层计时表和 Geometry/CLAS 堆叠图，已接入 Metallic 的真实运行时数据。

## 使用

重新启动本次构建的 `build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe` 或 `Metallic.exe`，打开 Profiler。

- **Table**：GPU/CPU 平均毫秒数和实际队列；展开 GPUDriven 查看遍历、候选展开、剔除、软硬分类、分桶、软光栅、硬光栅、合并与两轮 HZB。**Detailed** 展示双方的末次、最小和最大值。
- **BarChart / LineChart**：选择 CPU/GPU 和 Scope。可单独观察 Stream early 等内部阶段，悬停折线查看具体帧。条形图按独立耗时显示；并行和嵌套区间不累加为帧耗时。
- **Streaming**：当前 pass 和 cooked asset；几何页占用、容量，驻留/总页面数、等待页面、I/O 队列、上传管线和反馈年龄。显示显存面积图、请求/上传/卸载曲线、上传 MiB/frame 与积压曲线。
- 两类历史最多保留 **500 帧**。GPU 查询按 execution ID 回填；未完成或不支持的数据排除出均值，合法的 0 ms 保留。重编译时清理计时历史；stream generation 改变或离开流送场景时清理对应流送历史。
- 流送面积图默认按实际占用缩放；勾选 **Include capacity in memory chart** 可看到预算/容量。CLAS 启用时堆叠显示其池占用；当前 MiniZorah VBuffer 路径显示 **CLAS disabled**。

## 计时与数据范围

1. 修复“包含 Compute/Copy pass 就关闭整组计时”的问题，各队列独立查询池。Graphics 前置 reset 与结束 join 包围图执行；软硬并行分支通过各自命令缓冲写时间戳，父级区间跨 join 结束。
2. 专用 Transfer 队列不能调用 `vkCmdResetQueryPool`，所以所有池在 Graphics 前置提交中 reset，随后通过已有队列依赖再写入；RHI 对时间戳写入的队列族校验保留。依据：[Vulkan reset 规范](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResetQueryPool.html)、[timestamp 规范](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp2.html)。
3. 查询环按真实提交完成状态回收。取消 fork/join 后不再向已结束的 producer 写时间戳；超出动态 scope 预算后标记不可用，后续帧可恢复。读取查询结果不增加 GPU 等待。
4. GPU 总行是 **RenderGraph GPU envelope**，不含编辑器 ImGui 绘制和呈现。父子及异步阶段存在重叠，不能把所有行相加。CPU Frame 在离屏测试中还包含初次编译和预览等待，不能把截图的 CPU Frame 均值当作编辑器稳态性能。
5. 流送样本直接取自 residency / CLAS 池的 CPU 计数器，不启用调试观察器、不额外回读 GPU、不扫描全场景页面或空闲块。Geometry 表示页面池已分配字节，含等待上传的分配；不是整张显卡进程显存。
6. 请求来自最近完成的 GPU 反馈；Completed uploads 是本帧 CPU 已确认完成的页，Upload MiB/frame 是本帧提交上传的字节。Upload pipeline 包含 I/O 阶段，不能与 I/O queued/active 相加。
7. 本次漫游未触发内存预算淘汰，卸载曲线为真实的 0；没有人为生成峰值。CLAS 启用路径的计数入口已接入，本次 MiniZorah 实测为关闭状态。

## 实测

RTX 5070 Ti，RelWithDebInfo，1920×1080，1.5 px，180 帧：前 60 帧固定视角，后 120 帧往返转视角。启用 Vulkan validation 和独立 Compute 队列，关闭调试观察器。GPU 与后台应用共享，数值用于证明计时/图表可观测性，不是与 vk_lod_clusters 或 Nanite 的性能对比。

- 180 帧都收到完整 pass 与内部阶段 GPU 数据，无 scope overflow；异步 Software raster 标记为 Compute。
- 总页数 1,356,959；结束时驻留 14,732 页，占用 363.54 / 1024 MiB。
- 请求峰值 2,653 页/帧，累计观测上传 361.84 MiB。末帧反馈年龄为 1 帧。
- 含启动阶段的 RenderGraph GPU 平均 4.478 ms。以下为转视角区间 60–179 帧各 scope 的平均值，不做相加：

| 区间 | GPU 平均 ms |
|---|---:|
| Stream traversal | 1.526 |
| LOD frontier | 0.990 |
| Prefetch | 0.389 |
| Stream early / Candidates | 0.087 |
| Stream early / Cluster cull | 0.042 |
| Stream early / Soft/hard classification | 0.311 |
| Stream early / Software raster | 0.850 |
| Stream early / Hardware raster | 0.607 |
| Early HZB | 0.057 |
| Late HZB | 0.052 |

[结构化摘要](E:/metallic/Documentation/MiniZorahProfilerStreamingResults.json) · [原始 180 帧数据](E:/metallic/build-relwithdebinfo/profiler-streaming/final/MiniZorahProfiler.json)

## 验证

构建目标：Metallic、MetallicGPUDrivenSample、MetallicRhiTests。

- 最终计时/UI 组 **8/8 通过**：timestamp_query、render_graph_gpu_profiling、cancelled_gpu_profiling、gpu_profiling_scope_budget、editor_profiler_history、frame_parallel_compute_join_and_cancellation、frame_cross_queue_graph_dependencies、minizorah_profiler_streaming。
- 渲染回归组 **6/6 通过**：MiniZorah profiler、MiniZorah VBuffer、MiniZorah 转视角稳定性、小场景可视化稳定性、VBuffer OpenPBR 材质、HZB 图像等价与计时。其中 profiler 与上组重复，合计 13 个不同测试。
- 独立流送回归组 **2/2 通过**：streamasset_only_first_frame、stream_metadata_vbuffer。总计 **15 个不同测试通过**。
- 日志无 VUID、SYNC 或 Vulkan validation 消息。渲染回归存在既有的 OMM 提示：本机验证层版本较旧，沿用 shader alpha traversal；不是本次计时错误。
- 四个 Profiler 页签已通过真实 ImGui draw data 离屏渲染并人工检查。未操作用户主桌面。

日志：[最终计时组](E:/metallic/build-relwithdebinfo/profiler-streaming/final.log)、[渲染回归](E:/metallic/build-relwithdebinfo/profiler-streaming/regression.log)、[独立流送](E:/metallic/build-relwithdebinfo/profiler-streaming/standalone.log)。

## 界面

### GPU/CPU 表格（展开内部阶段）

![GPU/CPU 层级计时](E:/metallic/build-relwithdebinfo/profiler-streaming/final/Profiler-Table.png)

### 当前场景流送

![流送内存、页面流量、上传与积压](E:/metallic/build-relwithdebinfo/profiler-streaming/final/Profiler-Streaming.png)

[GPU 历史曲线](E:/metallic/build-relwithdebinfo/profiler-streaming/final/Profiler-LineChart.png) · [GPU 条形图](E:/metallic/build-relwithdebinfo/profiler-streaming/final/Profiler-BarChart.png)
