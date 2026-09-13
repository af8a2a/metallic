# MiniZorah：分离批量剔除与软硬分类

2026-09-13。在并行候选展开之后，将 StreamAsset 的元数据剔除改为每线程一个候选，并只对存活项执行几何分类。没有修改 cook 格式、1.5 px LOD 判定、软件三角形尺寸限制或可见性 ID 编码。

## 实现

- [streamClusterCullMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3494)：每个 128 线程组并行校验、剔除 128 个候选，沿用页面范围检查、视锥、normal-cone、early/late HZB、页面请求和 visible record 发布逻辑。
- 每个 wave 只预留一次输出区间。一个存活任务占 16 B：原候选槽、顶点起始 word、打包后的顶点/三角形数量、三角形索引起始 byte。任务暂存于尚未使用的最终箱列表，header[0] 临时记录任务数；不增加容量级显存。
- [streamClusterBinMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3537) 的 indirect dispatch 改由存活项数量驱动，直接读取缓存的几何位置，避免再次执行完整页面校验。线程并行投影原始顶点并逐三角形调用原有 `hybridTriangleFits`；每个 wave 汇总硬件需求后才更新组标记。
- 分类任务可以无序执行，但分类结果写回原始候选槽。后续稳定分桶仍按原候选顺序输出，避免原子追加的执行顺序改变硬件等深图元覆盖。
- [cullStreamClusters](E:/metallic/Source/Runtime/Render/VisibilityHybridRasterizer.cpp:234) 负责剔除、参数生成及 buffer barrier。参数缓冲保持 36 B，第一组三元组改写为存活项数量，第二组仍按原候选数驱动 stable bins。空列表生成零 X 派发，超过 65,535 项使用二维派发。
- 新增一个 compute PSO，沿用 `VisibilityBufferPass.pso`。热缓存日志为 30 hits / 0 misses。新增 early/late `ClusterCull` 检查点，和旧版计时比较时应将它与 `Classify` 合计。

该路径保留精确的三角形分类，没有通过放宽阈值或把不确定的小三角形交给软件光栅换取性能。Resident producer 沿用其现有剔除和分类实现。

## 输出一致性

RTX 5070 Ti，1920×1080，1 GiB 页面预算，1.5 px，异步 HW/SW，RelWithDebInfo，开启 Vulkan validation。

| 固定视角 | active groups | early 候选 → 几何分类组 | late 候选 → 几何分类组 | HW / SW |
| --- | ---: | ---: | ---: | ---: |
| 入口 0 秒 | 22,891 | 226,893 → 67,233 | 80,778 → 0 | 42,879 / 24,354 |
| 近景 15 秒 | 20,265 | 187,000 → 64,690 | 57,137 → 0 | 40,334 / 24,356 |

两视角最终 camera、cut、分桶数完全一致，可见超标 refinement 为 0。最终 PNG 字节完全一致：

- 入口：`9824bc17b155a4f4466603fd488502c53c4e742cae5f8dca8b29669ace09aaf4`
- 近景：`b181ca6e019e5d95a28f7857054de6fdc80b00aa066d0fb871302122f3344f75`

## 性能测量范围

首次入口固定视角在 2.5–5 秒的六个收敛检查点中，early 剔除+分类中位数为 2.952 → 0.344 ms，late 为 1.173 → 0.041 ms。晚期没有存活项时，原实现仍为 80,778 个候选分别启动一个 128 线程组；新实现只执行批量剔除和少量间接参数准备。

首次测量期间有多个后台 Unity 进程，Metallic 退出后 GPU 仍达到 100% 利用率。近景旧分类检查点一度升至 11–19 ms，整轮帧计时也升至 20–50 ms，因此首次六轮的整帧百分位仅保留在原始 JSON，不用于计算加速比。

GPU 一度回落到约 2% 后启动了配对复测，但后台负载随后恢复；新路径近景测完、Metallic 已退出时 GPU 仍为 78%。复测也不能作为独占 GPU 的整帧性能结论。为避免修改产品中的运行时设置，复测旧分类使用临时 shader：旧版本主体加一个兼容新的 host 调用的空剔除入口。它多执行两次空派发和相应 barrier，开销由 `ClusterCull` 检查点单独记录；基准结束后在 `finally` 中恢复当前 shader。该兼容入口在约 5 秒的固定视角检查点中开销为 0.008–0.020 ms/阶段；这些近似旧路径对照的输出仍与新路径一致。所有原始结果保存在 `build-relwithdebinfo/minizorah-classification/`。

完整数字和原始报告路径见 [MiniZorahClusterClassificationResult.json](E:/metallic/Documentation/MiniZorahClusterClassificationResult.json)。GPU 调度工作量减少和输出等价已有确定证据；准确的整帧收益仍需在持续空闲的 GPU 上重新采样。

## 验证

- `Metallic`、`MetallicGPUDrivenSample`、`MetallicRhiTests` 构建通过。
- [stream_cluster_cull_classify_equivalence](E:/metallic/tests/rhi/StreamClusterClassificationTests.cpp:18)：8 个 fixture × early/late，新旧 GPU 调度输出逐项相等。比较原始候选标签、最终稳定箱列表、完整 visible records、retry masks，并验证存活项 indirect 参数。覆盖 73,760 个存活项、空列表复用、部分块、HZB 遮挡后恢复、正交/透视、普通/反向 Z、1/8/32 px、独立 render camera 和 jitter、非均匀缩放、双面实例、空三角形、无效页面/顶点数/索引范围及 malformed-index fallback。
- 候选展开的 14 个 GPU/CPU 用例通过。
- 18 项相关回归全部通过，107 秒：MiniZorah 首帧/VBuffer/质量审计、StreamAsset、LOD cut、混合光栅、异步队列、两帧槽、resize/reload、持久 LOD PSO cache。日志无 VUID 或 validation error。
- 两个固定视角及 60 秒漫游通过全部检查点。漫游包含冷启动和转向，期间允许暂时未收敛；不将最终固定视角的零超标推广到全部动态帧。

复现：

```powershell
& E:/metallic/build-relwithdebinfo/tests/MetallicRhiTests.exe `
  '--gtest_filter=RhiRendering.stream_cluster_cull_classify_equivalence:RhiRendering.stream_cluster_candidates_stable_parallel' `
  --rhi-validation --output-dir E:/metallic/build-relwithdebinfo/minizorah-classification/recheck
& E:/metallic/build-relwithdebinfo/minizorah-classification/RunCase.ps1 `
  -Name recheck-fixed-0 -Seconds 10 -FixedView 0 -LatencyOnly 0 -Transitions 0
```

后续 profiling 应优先重新量化 traversal、候选展开和 HW/SW raster；本次已明显压低分类调度成本，下一步不应继续按旧 capture 的成本比例分配优化精力。
