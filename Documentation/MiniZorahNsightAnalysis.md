# MiniZorah Nsight 捕获与 Nanite 优化方向

2026-09-13；分析时 Metallic HEAD `cde4c2762`，本机 Unreal Engine 5.7.4。此次只做分析，没有修改渲染实现。

**下一轮优先拆分 cluster 剔除与软硬分类、并行展开候选。** 有效捕获回放中，这两部分占用约 4.19 ms，约为 8.20 ms GPU 帧的 51%。瓶颈更符合低线程利用率、组内等待和任务并行度不足，而非显存带宽饱和。后续再做硬件 mesh 顶点复用和 LOD 遍历重组。

## 测量范围和有效性

- 原始捕获：[MetallicGPUDrivenSample_2026_09_13_16_02_38.ngfx-capture](E:/metallic/Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_13_16_02_38.ngfx-capture)，第 2247 帧，Vulkan、RTX 5070 Ti / GB203、驱动 616.64。
- Nsight Graphics 2026.3.1 隐藏回放；交换链截图为 2400×1350，包含编辑器 UI。场景视口只是其中的子区域，导出元数据未给出其精确 extent，不能当作 2400×1350 的纯场景渲染测试。
- 普通回放画面与内嵌截图的 MAE 为 0。其 `iteration_times.csv` 的 10.521 ms 是 CPU 提交 3.479 ms 加等待 7.042 ms，`msGpuTime=-1`，不作为 GPU 帧时间。
- 对相同 capture 再采集 GPU Trace：`Top-Level Triage`、GPU clocks `unaltered`、开始/结束由 replay pass 界定，排除 replay reset，禁用额外呈现 blit。**有效样本为 `gpu-trace-clean`，GPU frame time 8.20451 ms**，41 个事件/区间指标行，未报告硬件事件缓冲溢出。
- 首轮 GPU Trace 缺省/数字指标集配置失败；随后一次采样报告硬件事件缓冲溢出且残留失败回放进程，均排除。清理后得到上述有效样本。第二次独立复测报 `GPU Performance Counters unavailable`，没有可用数据，因此这里不是多次均值、P95 或稳健性能保证。
- 捕获有 `VK_NV_low_latency` 版本不兼容标志（使用 2，工具支持 1），回放另有同 build 的 2026.3.1/2026.3 版本字符串告警及 NvAPI profile 注册失败日志。画面一致支持继续分析几何 workload；原程序 Reflex、CPU 帧时间与呈现节奏不由此次回放验证。

这是捕获帧的 GPU workload profile，不能诊断持续漫游的页请求尾延迟、启动 PSO 创建、原程序 CPU 记录长帧，也没有对同场景 UE Nanite 做等质量计时。[NVIDIA 回放说明](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)明确说明回放有独立的资源 reset、重放和等待行为。

## 时间分布

重复 marker 按 GPUDriven 执行顺序对应 early/late。时长均来自有效 trace 的 `D3DPERF_EVENTS.xls`。

| 工作 | early（ms） | late（ms） | 判断 |
| --- | ---: | ---: | --- |
| compact stream candidates | 0.430 | 0.508 | 共 0.938 ms；单线程组展开值得优先改 |
| classify stream clusters | 2.468 | 0.780 | 共 3.249 ms；最大可明确归因成本 |
| stream stable bins | 0.028 | 0.020 | 共 0.049 ms；当前前缀/散射本身很小 |
| stream hardware clusters | 0.911 | <0.001 | early 与 SW 异步重叠 |
| stream software clusters | 1.020 | <0.001 | early 与 HW 异步重叠 |
| MaterialResolve | 0.054 | — | 当前非主要矛盾 |
| Editor ImGui | 0.071 | — | CPU 编辑器成本不能由此推断 |
| FinalBlit | 0.012 | — | 优先级低 |

第一个 `GPUDriven` 段为 1.271 ms，包含 stream 更新、内部 LOD 等前置工作；capture 没有将 frontier/prefix/emit 分成独立 marker，**不能将整个段直接称为 LOD traversal 时间**。

late 光栅几乎为空，仍支付约 1.29 ms 的候选与分类，说明这帧大量后处理工作最终没有产生可观光栅任务。它不证明 late 可直接删除：上一帧 HZB 遮挡、本帧解除遮挡时仍必须恢复。

父子 marker 和异步分支不能直接相加为帧时间。特别是 0.911 + 1.020 ms 不是光栅关键路径 1.931 ms；硬件 marker 区间的全 GPU 指标也会包含同时执行的软件光栅。

捕获 API 流中有 18 次 `vkQueueSubmit2`、174 次 `vkCmdPipelineBarrier2`、51 次直接 dispatch、12 次 indirect dispatch、10 次 indirect mesh draw、11 次 `vkWaitSemaphores`。这些只是调用数量；不能据此认定每个 barrier/submit 都造成气泡。

## 计数器与源码归因

| 区间 | SM throughput（% peak） | DRAM（% peak） | 活跃线程/warp | barrier 采样占比 |
| --- | ---: | ---: | ---: | ---: |
| early candidates | 0.240 | 2.142 | 10.41 | 不用来归因 |
| late candidates | 0.180 | 2.225 | 7.39 | 不用来归因 |
| early classify | 20.164 | 5.498 | 9.57 | 65.6% |
| late classify | 20.212 | 2.918 | 4.11 | 71.8% |

候选区间 sync compute active 为 98.46% / 99.38%，但 SM throughput 极低。当前 [streamClusterPrepareMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3382) 只 `dispatch(1)`，128 个 lane 各自串行扫描一段 active groups，两次访问 mask 并逐 bit 写候选。单个工作组占住一条长链，无法分布到多数 SM；增加显存带宽无法解决这种并行度问题。

分类的 [streamClusterBinMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3443) 每 cluster 启动 128 线程。lane 0 完成 active group、页头、cluster、相机/实例数据加载和可见性判断，其他 lane 等待；存活后全组投影顶点、读取三角索引、检查软件尺寸，再做多次全组 barrier。大量被剔除 cluster 仍启动完整工作组。低活跃 lane 和高 barrier 采样与这种组织方式相符；尚无逐指令 source attribution，不能把全部等待精确归到某一个源代码 barrier。

活跃线程取 `Top_Level_Triage.sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime.ratio`。barrier 占比是同区间 `warps_issue_stalled_barrier.avg.per_cycle_elapsed` 除以全部 `warps_issue_stalled_*.avg.per_cycle_elapsed` 之和（包括 selected/not_selected），**不是 GPU 帧时间百分比，也不等于删除 barrier 后的可回收时长**。

全帧 SM throughput 15.29%、L2 6.91%、DRAM 5.60%，回放 VRAM committed 2675 MiB、demoted 0 MiB。这份样本不支持“显存吞吐饱和/显存超预算”为首要瓶颈；仍可能存在不连续加载带来的延迟和局部缓存问题。

## 对照 UE 5.7.4 Nanite 的具体改进

### P0：让剔除和软硬分类不再先解码全部几何

Nanite 在 [SmallEnoughToDraw](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:310) 中使用投影尺度、cluster `EdgeLength` 和实例变换决定 HW/SW；[cluster cull](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:815) 随后做 HZB，并将需要 clipping 的 cluster 强制走 HW。这里无需先对所有三角形执行完整投影分类。

建议分两步：先以 wave/批次处理多个 cluster 的元数据、frustum/cone/HZB 剔除，形成紧凑的存活列表；再使用 cook 保存的保守最大边长/投影范围快速决定 HW/SW。边界不确定或近裁面相交时保留当前精确分类/HW fallback。将无效候选尽早退出，避免每个被剔除 cluster 占用 128 线程并让其他 lane 等 lane 0。

新分类必须证明软件尺寸/裁剪约束，不能直接拿 Nanite 的边长阈值替代当前 `hybridTriangleFits` 的屏幕包围盒宽高阈值。阈值改变还会影响 HW/SW 分配和总光栅成本，验收必须同时看分类及 raster join。元数据变化需要 cook 格式/缓存失效规则；1.5 px LOD 目标独立保持。

**实验目标**：当前分类 3.25 ms，先争取减半；这是验收方向，尚未实测实现收益。保留可切换旧分类，以相同相机、相同驻留页、相同 cut 做 A/B。

### P1：并行稳定展开候选，并缩短空 late 路径

将单组扫描替换为多组 block count → prefix → scatter，分块读取紧凑的 `instanceIndex + selection/retry mask`，避免两次拉取完整 active group。沿用现有 record ID 与稳定顺序，保持等深覆盖结果。early 剔除同时压缩 late 重试工作；late 只处理重试项与重新可见实例，不再完整扫描 active groups。

Nanite 使用 wave 聚合计数写出可见 cluster，并为 main-pass 被遮挡项形成 post-pass 队列，见 [EmitVisibleCluster](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:709) 和该文件的 occluded cluster 输出。当前代码已具有 early/late retry mask，下一步是让生成/消费它的成本随实际重试数量缩放。

**实验目标**：两次候选展开总和从 0.94 ms 压到 0.2 ms 左右。不能为了省这段时间改为不稳定全局原子 append；应覆盖 0 候选、超容量完整 cut fallback、等深三角形和相机切换。

### P2：硬件 mesh 顶点复用，然后联合调整软硬比例

当前 [gpuDrivenStreamAssetMeshMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3312) 虽已并行，却仍每三角形输出 3 个独立顶点，每 cluster 切成两个 64-triangle chunk，重复加载/变换共享顶点。Nanite 的 [HWRasterizeMS](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.usf:2357) 输出唯一顶点、索引三角形，并将 visibility 信息放在 primitive attributes。

可改为 cluster 唯一顶点输出和 primitive ID 传递，先核对 Slang/Vulkan mesh 的每 primitive 接口与设备输出上限，保留 visibility record/triangle ID、绕序、双面和等深语义。当前 HW/SW 分支各约 1 ms，应同时测重叠时序；只优化 HW 可能被 SW 汇合等待掩盖。P0 之后再测 SW/HW 阈值，不必先重写已经存在的 64-bit visibility/depth 原子合并路径。

### P3：对前置 1.27 ms 细分后，再决定 persistent traversal

Nanite 提供 [PersistentNodeAndClusterCull](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:985) 等调度路径，将节点/cluster 工作放入 GPU 队列。当前 [buildActiveTable](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:3195) 是 reset/frontier/prefix/emit/finalize，以及按需 prefetch，多阶段协作遍历已实现。

下一步补齐这些内部阶段的 marker，并统计每 instance 的访问节点数、各组工作量分布和 prefetch 成本。只有确认实例粒度的尾部不均衡仍显著，再引入共享工作队列或分片调度。不要把本次未拆分段的 1.27 ms 全部作为 persistent traversal 的可优化预算。

### P4：压缩用于驻留与漫游，另立验收

Nanite 的 position quantization 和 cluster 编码可参考本地 `NaniteEncode.cpp`、`NaniteDataDecode.ush`。我们仍以 float4 stride 读取 position，进一步压缩可提高同预算驻留量，减少页面装卸，但本帧 DRAM 利用率很低，不能预期压缩直接解决 3.25 ms 分类瓶颈。压缩的验收应是固定 1 GiB / 1.5 px 下的 resident pages、request→drawable P95/P99、上传字节、淘汰重载次数及量化误差，而非只比较静态 GPU 帧。

## 推进顺序与验收

建议实现顺序为 **P1 并行候选（改动边界较小）→ P0 剔除/分类重组（最大收益预算）→ P2 顶点复用 → P3 遍历调度 → P4 压缩**。按问题收益排序则 P0 优先。

第一轮目标是固定 capture 对应视角/cut 的候选+分类总和下降 40% 以上，再检查端到端 GPU 帧是否同步改善；不能将局部加速直接换算成 FPS。维持 1.5 px 可见误差、VBuffer/参考 cut 一致、无丢面/近裁面漏洞、无等深闪烁。稳定 raster bins 当前不足 0.05 ms、MaterialResolve 约 0.05 ms，均先保留。CPU 启动、持久化 PSO cache、页面请求尾延迟沿用各自测试，不从本 capture 推断收益。

没有相同 MiniZorah 输入、视角、画幅、误差标准、材质及硬件下的 UE profile，因此目前只能比较实现与工作组织差距，不能声称“比 Nanite 慢多少倍”。[Epic Nanite 设计说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine?application_version=5.7)可作为总体机制参考，以上具体判断主要对应本地 5.7.4 源码。

## 数据与复现

- [结构化分析](E:/metallic/Documentation/MiniZorahNsightAnalysisResult.json)
- [有效事件时间](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean/BASE_UNLOCKED/D3DPERF_EVENTS.xls)、[区间硬件指标](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean/BASE_UNLOCKED/GPUTRACE_REGIMES.xls)、[采集配置](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean/BASE_UNLOCKED/REPRO_INFO.xls)
- [有效 GPU Trace](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean/ngfx-replay_2026_09_13_16_14_44.ngfx-gputrace)、[采集日志](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean.log)
- [隐藏采集脚本](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/RunTrace.ps1)、[解析脚本](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/Analyze.py)。CLI 用指标集名称 `Top-Level Triage`；确认 GPU performance counters 可用后运行，避免并发 profiling。输出为 TSV 文本，尽管后缀是 `.xls`。
- [捕获内嵌截图](E:/metallic/build-relwithdebinfo/minizorah-nsight-20260913/metadata_screenshot.png)
