# MiniZorah M4：持续漫游

本阶段打通固定预算下的持续运行和可复现验收。**覆盖完整性、质量收敛和交互帧时分别统计；长测通过不能视为已收敛到 1.5 px。**

后续已实现工作组协作 frontier/emit、按 GPU 需求保留驻留缓存，以及碎片分配检查缓存；60 秒路线 GPU P95 从 59.90 ms 降到 24.06 ms，同步帧 P95 从 67.21 ms 降到 31.66 ms。当前实现与最新验收见 [遍历与驻留优化](MiniZorahRoamingOptimization.md)。下文保留 M4 初始基线，便于比较。

## 实现

编辑器输入链路的后续修复见 [视口相机修复](MiniZorahViewportCameraFix.md)：旧自动路线直接修改 pass camera，未覆盖编辑器共享 RenderView；现在 MiniZorah 跟随共享视口，并在漫游检查点读回验证实际渲染相机。

- 流式 active group 的 mask 稳定展开为紧凑候选，GPU 生成分类与 histogram/scatter 的间接派发参数。沿用原始 `groupIndex * maxClusters + clusterIndex` 记录号，不改变 VBuffer 材质解码或等深度绘制次序。工作量随实际 cut 增长；记录和分箱缓冲的预留容量仍计入显存。
- 当前视锥在 LOD frontier 前筛选实例，并清理上一帧的稀疏状态。手动 LOD 与无 raster bindings 的独立遍历保留原契约。HZB 仍执行 early/late 复测，历史遮挡不会永久阻止实例返回。
- 实例 prefix 改为 64 线程分段扫描，同时统计细 cut 和 terminal cut。容量不足时仍整批退回完整根 cut，不截断部分实例。
- 预算不足时每帧仅构建一次淘汰候选表；候选按年龄排序并在使用时复查。主动淘汰上限为每帧 256 页，多个 unload 合并到同一延迟释放任务，避免三槽任务环一次只能退三页。
- 异步 I/O 的已接纳队列最多为 `2 * maxPageLoadsInFlight`；锁定根页独立接纳。尚未发起且过期的请求可回收，已经发起的 I/O/上传仍保留完成生命周期。上传排序使用根页、最近需求、等待年龄和 payload 代价。
- 新增扫描、退避、接纳延迟、取消、上传字节等计数。`RenderGraphPreviewRenderer::render(..., false)` 可完成离屏帧而不读回输出，避免把图像读回计入性能样本。

屏幕误差收益尚未进入页请求排序；实例内部的协作 BVH 遍历和更细的可见性需求裁剪仍待实现。当前保留既有 ancestor payload 策略，不改写共享子组的父依赖与安装/撤销协议。

## 验收方法

`RhiRendering.minizorah_roaming` 使用完整 cook 与 metadata GPUScene，1920×1080、自动 LOD 1.5 px、异步 HW/SW、法线锥关闭。每个 60 秒周期包含局部移动、左右转向，以及第 20/25、50/55 秒的远近瞬移。

每五秒读取同一帧的 active cut、实例可见性和分箱计数，独立从 terminal clusters 重建 DAG 覆盖：未选的粗 cluster 必须有完整替代，共享子组的各父组必须一致，不允许重复或不可达的 selected group。可见/待复测实例必须拥有 cut；同时检查所有根页仍驻留、页池不超预算、无加载失败/非法请求，且紧凑候选数等于 mask 的 popcount。

质量计数 `overTargetRefinements` 使用与 GPU 相同的保守投影公式，统计 emitted cut 中仍超过阈值的 refinement；包括遮挡部分和受共享父关系限制的需求，**不是屏幕上误差超标的像素数，也不能全部解释为缺页数**。近裁面相交的无界投影单独计数。

计时帧关闭 debug observer 和输出读回。离屏图保留 `GPUDriven + MaterialResolve`，省略无 swapchain 时的 FinalBlit；GPU 计时覆盖 HW/SW 分支及 join，不累加重叠时间。检查帧和图片保存不计入计时样本。测试仍同步等待每帧完成，并启用 Vulkan 验证，不能直接当作编辑器呈现 FPS。

默认测试累计 660 秒计时帧，覆盖一条 60 秒路线加十分钟巡航。显存使用 Windows DXGI 的进程 local/non-local 统计，比较 120–180 秒与最后一圈的 local 峰值，允许 64 MiB 暂存分配波动。

```powershell
$env:METALLIC_TEST_MINIZORAH = '1'
$env:METALLIC_MINIZORAH_ROAM_SECONDS = '660'
$env:METALLIC_MINIZORAH_ROAM_MIB = '1024'
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_roaming --rhi-validation --rhi-async-compute --output-dir E:/metallic/build-relwithdebinfo/minizorah-m4/cruise
```

压力测试使用同一路线，将时长改为 60 秒、页池改为 64 MiB。降低预算会保留更粗的覆盖，较低帧时必须连同质量计数一起比较。

## 结果

2026-09-13，RTX 5070 Ti，RelWithDebInfo。**持续漫游、预算和 cut 完整性基线通过；33.3 ms 交互帧时及 1.5 px 质量收敛尚未达标，M4 性能验收保持开放。** 结构化摘要见 [MiniZorahRoamingResult.json](MiniZorahRoamingResult.json)，原始记录位于 `build-relwithdebinfo/minizorah-m4/`。

| 运行 | 计时帧 / cut 检查点 | CPU 记录 P95 | GPU P50 / P95 / P99 | 同步帧 P95 |
| --- | ---: | ---: | ---: | ---: |
| 1 GiB，660.03 s 巡航 | 34,431 / 132 | 本轮字段无效，见下文 | 2.52 / 61.36 / 68.41 ms | 68.99 ms |
| 1 GiB，60.05 s 最终阶段诊断 | 2,891 / 12 | 9.32 ms | 2.73 / 62.16 / 71.12 ms | 71.61 ms |
| 64 MiB，60.03 s 压力路线 | 3,750 / 12 | 6.02 ms | 17.13 / 23.31 / 24.45 ms | 28.90 ms |

三个运行均通过 DAG cut 重建、根驻留、候选计数和预算检查，没有 device loss、Vulkan 验证错误、非法页请求或加载失败。660 秒运行的页池采样峰值为 1,028,153,344 B（980.52 MiB），实际候选峰值 415,386，预留容量 4,194,304。64 MiB 档页池峰值 61.81 MiB，实际候选峰值仅 73,960；它保留更粗细节，不能用其较低帧时证明 1 GiB 档已经达标。不同 cut 的 refinement 数量也不同，误差超标计数不能直接作为跨预算的质量排名。

660 秒运行的 DXGI 进程 local 显存，120–180 秒与最后一圈的峰值均为 2,397,757,440 B（2.23 GiB），没有超过 64 MiB 的允许增长。全部检查点的 local 峰值为 2.48 GiB；CPU 提交内存峰值为 3.73 GiB、工作集为 3.92 GiB，包含测试额外打开的 CPU cut 验证元数据。采样峰值不等于逐帧精确峰值，页池也不等于总显存。

质量仍未收敛：长测单个检查点最多有 12,253 个保守投影超标 refinement。最后一个检查点（655 s）累计完成 1,848,804 次页上传、1,845,629 次 unload，上传 77,479,478,304 B（72.16 GiB）；其中预算主动淘汰 30,862 次，取消尚未发起的过期请求 22,224 次。这些数据说明远近切换与现有需求撤销仍造成大量反复装卸，队列上限和 CPU 退避只解决了失控工作量，尚未解决驻留收益与细节收敛。

长测的旧 CPU 采样从 GPU timing slot 读取了尚未更新的 CPU 字段，原始 JSON 中的全零数据无效。测试已改为直接读取当帧 `executionStats().cpuMilliseconds`；表中的两项 60 秒运行使用修正后的采样。长测之后还补了独立 shader fixture 的可选 raster binding 哨兵检查；MiniZorah 始终绑定该资源，实际漫游分支不变。最终 31 项回归和阶段诊断使用修正后的代码。

最终回归 **31 项全部通过**（159.96 s）：涵盖 streamer 年龄保护/批量卸载、BVH 与暴力遍历对照、共享父组/容量回退、稳定混合分箱、GPUScene、混合 producer、scene binding、异步场景切换、完整 MiniZorah M3 和 64 MiB 漫游。预算年龄保护测试同帧重复 10,000 次请求，只构建一次淘汰候选表；批量卸载测试验证八页共用一个延迟任务。`MetallicRhiTests`、`MetallicGPUDrivenSample`、`Metallic` 均构建成功。

### GPU 阶段诊断

最终 60 秒运行在检查帧加入六个 graphics timestamp，记录 HW/SW join 所在的粗阶段。第 15.03 秒的近景检查帧如下：

| 区间 | GPU 时间 |
| --- | ---: |
| GPUDriven pass 起点 → AfterTraversal（页更新、遍历、active cut 准备等） | 39.78 ms |
| AfterTraversal → AfterEarlyCull | 0.43 ms |
| AfterEarlyCull → AfterStreamEarlyBins | 5.46 ms |
| AfterStreamEarlyBins → AfterLateCull（含 early 光栅、合并、HZB 和 late cull） | 8.73 ms |
| AfterLateCull → AfterStreamLateBins | 4.99 ms |
| AfterStreamLateBins → AfterPass | 0.59 ms |
| GPUDriven 全 pass / MaterialResolve | 59.98 / 0.09 ms |

第 40.01 秒同类近景也有 39.55 ms 位于 AfterTraversal 之前，远景该区间约 1 ms。区间包含检查帧的拷贝/屏障，第一项由 pass 时间减去后续区间推导，不能解释成单个 frontier shader 的精确耗时。证据支持优先细分并优化遍历与 cut 准备，同时继续关注两次分箱约 10 ms 的成本。Nsight Graphics 2026.3.1 GPU Trace 本次捕获停滞，CLI 报告没有有效导出；没有使用旧 trace 或据此给出内核级结论。

首圈近景图：[roam-10.png](../build-relwithdebinfo/minizorah-m4/cruise/roam-10.png)。M3 的 48 帧固定视角仅代表短期采样；持续移动会加载更多细节，不能把短期页池占用外推为稳定工作集。

## 接续工作

1. 细分 AfterTraversal 前的页 patch、frontier、prefix、emit 计时，针对最大楼梯和高复用地板实现协作遍历或分阶段工作队列；共享父组与整批容量回退继续由现有 cut 验证器把关。
2. 将层级可见性、屏幕误差收益接入页需求，保护近期实际使用的页面；量化 request→drawable 延迟与重复上传，优先降低长测中的反复装卸。
3. 在相同预算、分辨率和质量指标下重跑 60 秒路线与 660 秒巡航，单独关闭 P95 ≤ 33.3 ms 和质量收敛条件，再决定驻留编码压缩与 HW/SW 阈值调优的顺序。
