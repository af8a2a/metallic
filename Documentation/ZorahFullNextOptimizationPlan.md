# ZorahFull：局部分桶之后的优化方向

日期：2026-09-20。此次完成代码核对和一轮短漫游诊断，没有修改渲染实现。

建议保留共享屏幕顶点与精确整数边步进，保持局部分桶默认关闭。下一阶段重点是减少 SW 实际扫描量、抑制无法准入页面的重复处理，并补齐 CPU 等待原因。继续增加排序层级或直接开启异步 SW 尚缺收益证据。

## 当前证据

本轮固定路线：RTX 5070 Ti，编辑器输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px，完整材质和阴影。预热 10 秒，采样 30 秒，共 431 帧，GPU 数据缺失 0 帧。

| 指标 | 本轮均值 |
|---|---:|
| 整帧 | 69.755 ms，约 14.3 fps |
| 整帧 P95 / P99 | 80.461 / 86.620 ms |
| RenderGraph GPU envelope | 47.359 ms |
| Early SW | 33.069 ms |
| Deferred GPU | 5.344 ms |
| Stream traversal GPU | 3.085 ms |
| Early classification / HW GPU | 1.561 / 1.229 ms |
| CPU Stream Begin | 9.774 ms |
| CPU Consume requests | 7.551 ms |
| CPU Deduplicate loads | 3.040 ms |

430/431 帧超过 33.33 ms。父子 scope 是包含关系；CPU 与 GPU 也可能重叠，不能把表中各项相加。此次有其他进程 GPU 引擎活动，频率、驻留未与旧测试配平，因此仅用于定位，不能作为新旧版本收益的正式 A/B。

此前冻结 camera、cut、驻留的换序 A/B 中，复用不分桶 SW 为 7.317 ms，局部分桶为 8.215 ms，即分桶增加 12.27%。冻结路径关闭流送发布、遍历、TLAS 和 jitter；它与普通漫游不能直接比较。两次测试起始相机相同，仍须核对实际 shader 分支、工作量、驻留和频率，不能把差异简单归因于视角移动。详见 [上轮报告](ZorahFullLocalWorkBins.md)。

## 1. 先建立漫游与冻结测试之间的对应关系

把漫游静止、近墙、转身三个阶段各取一个状态，在该状态冻结 camera、cut、页映射，然后运行匹配的 A/B。导出实际 rasterMode、shader/pipeline hash、分流阈值、early/late SW 三角形数、唯一顶点数、bbox 宽高/面积分布，以及 GPU 频率。不能只记录 JSON 中显式设置的属性，因为缺省值也影响最终入口。

为 SW 增加可关闭的诊断计数：bbox 访问像素数、通过覆盖测试的像素数、原子尝试次数、wave 最大行数及有效 lane 数。采用组内汇总或抽样，计数开启的运行不参与正式耗时排名。先区分空 bbox 扫描、lane 长尾与像素竞争各占多少，再决定重排规模。

同轮补上 RenderGraph 的 priorFrameDrain、slotWait、poolReset、frameSetup CPU scopes，以及 drain 原因和阻止 overlap 的 pass 名称。当前外层 Record RenderGraph 为 64.966 ms，内部 CPU envelope 为 22.086 ms；约 42.88 ms 的差值尚未细分。执行器在等待与 setup 之后才开始内部 CPU 计时，因此该差值值得调查，但不能全部视作等待。当前构建 METALLIC_HAS_NRD=0，不能归因于 NRD。

## 2. SW 首选实验：保守逐行区间裁剪

当前 bbox 嵌套循环会访问三角形之外的像素；分桶改变 lane 分配，没有减少总扫描量。下一次实验保留小 bbox 的矩形扫描，对较宽或细长三角形采用逐行有效 X 区间，减少空访问，不引入全局三角形队列或额外组同步。

本地 Nanite 的 NaniteRasterizer.ush 中 RasterizeTri_Adaptive 在矩形扫描和 RasterizeTri_Scanline 之间选择；这是可借鉴的算法方向，其阈值和浮点实现不应直接照搬。本项目需要从 snapped 整数边函数推导保守区间，向外取整，最后仍执行原覆盖测试，保留原 depth 表达式、top-left 规则及 packed depth/ID 原子规则。过小 bbox 上求交开销可能超过省下的扫描，必须实测交叉点。

验收：负坐标、屏幕裁剪、退化和细长三角形、标准/reversed Z、单双面、subpixel 模式、early/late 路径；原始 recordIndex/triangleIndex 不变；Full 多状态 depth/visibility 精确比较。上轮曾出现轻微重排后深度/ID 不一致，本轮不以放宽容差换性能。

若计数表明空扫描占比很低，则停止 scanline 实验，转而用当前 shader 的 Nsight source counters 判断 lane 长尾或原子竞争。旧内核的 lane/barrier 数据不能代替当前 37 寄存器版本的测量。[Nsight Shader Profiler 文档](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)提供源代码级执行与停顿分析。

## 3. CPU 请求侧：按资源变化重试失败准入

本轮每帧平均 13697.6 次页面请求、13501.3 次 allocationFailures，而实际上传约 63.8 页、回收约 61.9 页。几何池平均使用约 3.49 GiB，容量 3.5 GiB。两项计数之比约 98.57%，表示大量准入尝试失败；它不是唯一页面失败率，也不是 Vulkan OOM。

MeshletStreamResidency 当前仍逐帧合并请求、更新 page 状态，并对 ready 请求排序和尝试分配。优化顺序：

1. 区分失败原因：无可用空间、碎片、无可回收冷页、CLAS/联合预算等；记录同页重复尝试次数。
2. 为受阻页面保留需求与优先级，用资源变化 epoch 驱动重试：回收完成、预算变化、页状态变化时重新准入；保留有界超时重试与公平性，避免饥饿。
3. 将需求刷新与实际准入分开；按可用容量和页大小选择有限候选，不对所有受阻请求逐帧做完整 stable_sort 和分配。
4. 再评估 dense page-ID epoch marks 替代临时 hash 去重，以及 pending latency 的到期队列；先核算额外常驻内存。已有 unused 集合优化不重复重做。

工程目标是 CPU Stream Begin 压到 3–4 ms 范围，属于待验证目标。必须同时检查页面等待 P95/P99、返回视角重载、1.5 px 收敛和内存峰值，避免通过少处理请求制造表面提速。

## 4. 后续调优与质量约束

新扫描内核稳定后，重新做 1/2/4/8 px 分流对照，并覆盖多个冻结漫游状态。比较 classification + HW + SW + 整图，不只比较 SW。旧阈值结论来自旧内核；既有全 HW 冻结结果也不支持直接改成全 HW 默认。

当前 GPU envelope 47.36 ms 已超过 33.33 ms，因此仅缩短 CPU 等待不能完成 30 fps。把 GPU envelope 减去 Early SW 的余量约为 14.29 ms；在其他成本不变的粗略预算下，SW 需低于约 19 ms，实践应争取 15–18 ms 留出波动空间。这是预算估算，不是线性提速承诺。

帧间重叠优化依赖第 1 步的等待归因。若确认是资源发布造成全图 drain，优先考虑按帧保留 descriptor/资源 generation、GPU timeline 依赖和延迟回收；不能直接移除 fence 或把所有 pass 改为支持 overlap。异步 SW 与 HW 也可能竞争执行资源和可见性原子，暂不作为默认方案。

blasOverflowCount 均值约 8714，需要拆分每实例容量、全局引用容量及异常状态等原因，并检查受影响阴影/几何；当前聚合计数不足以确定单一根因。它是质量验收风险，但现有 BLAS/TLAS 耗时不是首要性能瓶颈。纹理独立上传 GPU 均值约 0.039 ms（仅有上传的 122 个样本），也不宜优先继续压缩。

跨 cluster 全局工作队列、原子合并、进一步分桶，以及 Deferred 材质优化，均排在上述证据和局部实验之后。全局队列若最终必要，必须有容量上限、溢出回退，并以原始 record/triangle/rowRange 保存身份。

## 推进建议与验收

下一批建议先完成“漫游工作量与等待原因计数”，随后做两个可独立 A/B 的改动：SW 保守 scanline、失败准入按资源变化重试。不要把二者一起打开后才测收益。

正式验收保持编辑器视口、DLSS Quality、1.5 px、完整材质与阴影；空闲 GPU 下交替次序测量，再跑至少 3 轮长漫游，分别报告首次进入、近墙、转身、返回阶段的整帧 P95/P99、超 33.33 ms 连续区间，以及质量与驻留指标。30 秒诊断不替代持续 30 fps 验收。

证据：

- [本轮整理数据](ZorahFullNextOptimizationData.json)
- [本轮原始汇总](../build-release/full-post-work-bins-analysis/run1/Summary.json)
- [本轮 Capture](../build-release/full-post-work-bins-analysis/run1/Capture.json)
- [RenderGraph 等待路径](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp)
- [页面请求消费实现](../Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp)
- 本地参考：E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush，第 230、292 行。
