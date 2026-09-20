# ZorahFull：软件光栅与分类工作量优化分析

2026-09-20；当前 Metallic HEAD `736b277eea640e6b7c4f3b3806f7760b6ad8ccd9`，本机 Nanite 源码 UE 5.7.4。此次只复核已有数据和源码，没有运行新的 GPU 采样或修改渲染策略。

**建议先完成工作量计数及全硬件/阈值对照，再推进元数据快速分类和软件光栅预计算、增量扫描。** Full 当前两项约 73 ms，是最明确的 GPU 优化对象；旧 MiniZorah capture 中的 barrier 比例不能解释当前 Full 的具体瓶颈。

## 1. 哪些结论已有实测支持

### 当前 Full：P0 时间戳证据

三轮隐藏编辑器固定路线，每轮 180 秒，4657 帧；RTX 5070 Ti，输出 1797×660、DLSS Quality 内部 1198×440，LOD 1.5 render px、完整材质/阴影，普通纹理细化上限 512。磁盘/shader 缓存保留。启动日志确认 `shaderDebugMode=disabled`，不是 shader debug 的未优化计时。

| 项目 | 三轮均值范围 | 优化含义 |
| --- | ---: | --- |
| Early 软件光栅 | 50.6–51.1 ms | 单项超过 33.33 ms |
| Early 软硬分类 | 21.8–21.9 ms | 独立的全几何预处理成本 |
| Early 硬件光栅 | 约 1.16–1.17 ms | 当前分桶下较小，不代表全部转硬件后仍然小 |
| Early cluster cull | 约 0.55 ms | 已拆分后的批量元数据剔除 |
| Early stable bins | 约 0.25 ms | 不应首先重写排序/前缀逻辑 |
| RenderGraph GPU envelope | 约 85–86 ms | 两大项约占 85%；不含编辑器合成与 Present |
| 编辑器帧 p50 | 113–120 ms | 还存在 CPU、提交及等待成本 |

Full 预设 `asyncSoftwareRaster=false`，当前主光栅分支按 SW→HW 执行，两段主项可以作为串行成本观察，不能再与父级 envelope 相加。仅在现有约 1.2 ms HW 分支上做异步重叠，直接收益量级很有限；重分流以后应重新评估。

对 Frames.jsonl 按同一 execution 的 early scopes 配对：三轮 Pearson r 分别 **0.9782 / 0.9752 / 0.9785**，SW/classification 比值中位数 **2.320 / 2.323 / 2.331**。第一轮起点静止时两者均值为 57.21 / 24.68 ms，返回末尾静止为 57.63 / 24.92 ms。由此可排除“只有移动时才昂贵”的解释；共同工作量、频率或资源等待均可能产生相关性，不能据此断言唯一根因。

这些数据没有实际可见 cluster/triangle、包围盒采样点、有效覆盖像素、原子写入数的逐帧统计。因此，**目前无法判定是几何数量异常，还是单位 cluster 成本异常，也不能宣称已证明 atomic contention 或带宽瓶颈。**

原始数据：[P0 报告](E:/metallic/Documentation/ZorahFullP0Benchmark.md)、[run1 Summary](E:/metallic/build-release/full-roam-p0-baseline/run1/Summary.json)、[Frames](E:/metallic/build-release/full-roam-p0-baseline/run1/Frames.jsonl)。本次提取结果：[分析数据](E:/metallic/Documentation/ZorahFullRasterWorkloadAnalysisResult.json)。

### 实际 Nsight Capture：MiniZorah 的历史证据

搜索了 Captures、build-release、build-relwithdebinfo 中的 capture/trace 导出。找到的 capture 均为 9 月 12–13 日 MiniZorah 阶段，未找到当前 Full 的有效 Nsight Trace。最新 capture 文件为 16:43:13，但没有对应有效 GPU Trace 导出；可核验的有效导出仍来自 **9 月 13 日 16:02:38 capture**，第 2247 帧。没有把新文件名等同于已完成性能分析。

此次用 Nsight wrapper 复查并直接读取 `gpu-trace-clean/BASE_UNLOCKED` 中 TSV 格式的 D3DPERF_EVENTS.xls、GPUTRACE_REGIMES.xls，逐行验证 marker 对齐，共 41 行。

| marker / TSV 行号（含表头） | 时间 ms | SM peak % | DRAM sectors peak % | active threads / warp |
| --- | ---: | ---: | ---: | ---: |
| early classify / 13 | 2.46848 | 20.1642 | 5.4980 | 9.5705 |
| late classify / 28 | 0.780447 | 20.2124 | 2.9181 | 4.1090 |
| early SW / 39 | 1.01968 | 30.7579 | 9.1541 | 15.1499 |
| early HW / 15 | 0.911103 | 30.3701 | 8.4551 | 15.3115 |

分类中的 barrier 采样占比约 65.6% / 71.8%，支持当时的等待/低线程利用率问题；**不是帧耗时占比或预期可回收比例**。SW 区间的 long-scoreboard 和 wait 样本值得关注，但该帧 HW/SW 异步重叠，区间计数并非单 kernel 隔离指标，不能归因为 SW 原子操作。低 DRAM 吞吐也不能排除访存延迟。

原始指标名分别是 `GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed`、`dram__sectors.avg.pct_of_peak_sustained_elapsed`、`Top_Level_Triage.sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime.ratio`。wrapper 的 DRAM 别名为 null、L2 别名对应 syslts，本文不把 null 当 0，也不把不同 L2 指标混用。

该帧 GPU 8.20451 ms；旧 capture replay 图像曾与参考一致，但仅一次有效性能样本，存在 low-latency 扩展版本告警。无硬件事件缓冲溢出的 clean 导出才被采用；其他失败/溢出/计数器不可用的尝试不纳入。完整有效性边界见 [旧分析](E:/metallic/Documentation/MiniZorahNsightAnalysis.md)。

**候选 count/prefix/scatter、每线程一个候选的剔除、存活任务压缩、硬件唯一顶点复用均已实现。** 不能继续按旧 capture 建议重新实施，也不能用旧分类占比估算 Full 收益。

## 2. 当前实现与 Nanite 的具体差距

### 分类：仍需遍历全部存活几何

[streamClusterBinMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3978) 为每个存活 cluster 派发 128 线程：读任务及实例状态，初始化 shared cluster，投影唯一顶点，逐三角形调用 hybridTriangleFits，再跨 wave 汇总 HW 标志。普通路径有三次组同步。一个不满足条件的三角形使整个 cluster 走 HW。

当前已经先剔除、后分类；剩余差距是**分类仍执行完整顶点/索引解码与投影，光栅又重复一次**。masked/blended coverage 标志或 tessellation 强制 HW，但判断发生在已派发的分类组内。

Nanite [SmallEnoughToDraw](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:310) 在 cluster culling 使用包围球投影比例、EdgeLength 和实例/变形缩放决定 HW/SW，不必先投影全部三角形。它还保留 clipping 等回退规则。两者阈值定义不同：Metallic 的 8 px 是三角形屏幕 bbox 的最大边，不是 Nanite 的投影边长阈值；不能直接照抄数值。

建议两级判定：

1. 将已知 coverage/tessellation 强制 HW 的判定前移至批量 cull，写分类标签，不再占用几何分类组。
2. 对可证明安全的 cluster，使用保守投影 bounds 或最大边长上界直接判为 SW；不确定项保留现有精确分类。测试额外偏向 HW 的宽松策略时，必须同时衡量 HW 负载，不能只看分类变快。
3. 先使用现有包围球做无需重 cook 的保守 SW 快路径，记录命中率。它包围整个 cluster，可能很保守。要扩大命中率，再增加实际 payload 的 max-edge 元数据；当前 [payload cluster](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.h:197) 没有这一语义字段。
4. max-edge 可在 cook 或页面首次解码时生成并缓存；不能每帧重新计算，也不应为实验立刻重 cook 全部 Full。旧缓存必须显式标记“元数据不可用”并回退，不能把 reserved=0 解释为零边长。

保守投影必须处理非均匀缩放/剪切、近远裁剪、w 的下界、屏幕外 ±32 边界、数值非有限值和亚像素取整。渲染相机与 culling camera 分离、jitter 和反射实例都要沿用现有语义。**不要用 LOD error 代替最大三角形边长**：它衡量简化误差，并不保证 SW 的安全工作范围。

### 软件光栅：重复顶点 setup + bbox 全扫描

[streamClusterRasterMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4030) 重新校验页面/cluster，投影唯一顶点到 shared clip，再由每个三角形线程执行 [hybridRasterTriangle](E:/metallic/Shaders/Modules/GPUDriven/HybridRasterTriangle.slang:19)。后者对每个三角形的三个 clip 顶点重复透视除法、viewport 转换及 subpixel snap；每个 bbox 样本重新计算三条 edge equation，覆盖后计算深度，写 64-bit InterlockedMax。

Nanite 的普通路径同样是 cluster 内并行顶点、每线程处理三角形，**不是所有三角形都交给像素并行队列**。实际可借鉴的差异：

- [ClusterRasterize](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.usf:680) 将变换后的顶点提前转成 subpixel 数据，写 shared 缓存，三角形直接引用。
- [SetupTriangle](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush:29) 预计算边方程/深度相关系数；[矩形路径](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush:133) 通过增量步进更新边值。
- [Adaptive 路径](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush:292) 在 programmable 或 wave 中存在较宽 bbox 时选 scanline，否则矩形扫描。scanline 避免遍历大量明显位于三角形外的 bbox 样本。

优先实现共享屏幕顶点预计算、增量整数边函数、三角形深度平面，再根据实际尺寸/长宽比直方图增加 scanline。深度算术重排可能影响等深比较，不能声称自动位精确；需要检查顶层覆盖规则、深度误差、共享边裂缝和 HW/SW merge。

不要预先写出每 cluster 128×float4 的全局投影缓存：每百万 cluster 仅一份就约 1.91 GiB，且增加读写流量。尽量消除分类侧几何访问，或在小范围内复用 setup；如融合分类与 SW raster，需要重新验证 barrier、shared 生命周期、稳定可见性 ID 和并发 merge。

### 原子竞争和并行度：作为计数驱动的后续实验

Metallic 每个覆盖样本都会发出 64-bit atomic。Nanite 的普通 VBuffer 写入也使用 64-bit atomic，因此“去掉 atomic”并非有效对标方案。Nanite 的 EarlyDepthTest 在所检查版本中受 programmable raster 等条件约束，不能描述成全部普通 SW 三角形都有提前深度剔除。

如果 Full 新数据证明 overdraw/原子竞争高，再尝试按保守 HZB 剔除、近到远调度，或安全的深度预检；不建议先引入无同步读写竞态。深度/ID 比较和等深胜出规则必须保持。像素队列、wave 内工作重分配、小 cluster 多任务合组，仅在长短任务分歧或低占用有证据时引入；额外队列、prefix、shared 和寄存器成本可能抵消收益。

## 3. 推进顺序与决策门槛

| 阶段 | 实施内容 | 必须回答的问题 / 验收 |
| --- | --- | --- |
| R0 工作量基线 | early/late 分开统计候选、cull 后 cluster、HW/SW cluster/triangle、强制 HW 原因；采样 bbox 尺寸/面积、背面/退化/空覆盖、覆盖样本和 atomic 尝试；附带工作量来源 execution ID | 同视角是工作过多还是每单位成本过高？记录每百万 triangle 的分类/光栅时间和每输出像素的 triangle/覆盖次数 |
| R1 路径分流 A/B | 同一固定 camera、cut 和驻留：当前 8 px，对照真正全 HW（绕过几何分类但保留 cull/稳定 record），再测试 1/2/4/8 px | 比较 cull+classify+bins+raster+merge 总成本及完整帧；缩小阈值是否只把成本转给 HW？不能通过丢弃页面/降低 LOD 取胜 |
| R2 元数据快速分类 | 强制 HW 前移，bounds 安全 SW 快路径，精确慢路径；再按命中率决定 max-edge 元数据 | 不增加错误 SW 分流；记录 fast/slow 命中率和 HW/SW 负载，保持 stable bins/ID；分类减少不能被 raster 增长抵消 |
| R3 SW setup 与扫描 | 顶点屏幕化/snap 复用、边函数增量、深度系数，随后小矩形/scanline 双路径 | 减少 bbox 无效迭代和指令；输出深度/覆盖通过，记录寄存器、shared、occupancy、active lanes；编译器可能已提取部分除法，需看生成代码/指标 |
| R4 可见工作量治理 | 审计 LOD cut 父子互斥、重复实例/cluster、退化小簇、属性约束造成的 cluster 碎片、HZB 保守度；有依据再调整 cone cull | 在保真与材质边界约束内降低 triangle/pixel；不能通过放宽 1.5 px 或未加载细节宣称同质量提速 |
| R5 再评估异步 | 在新分流比例下恢复 HW/SW 队列对照 | 比较实际 raster join、资源竞争和完整帧；不将两条并发范围直接相加 |

R0 的计数应采用 wave/group 汇总并异步读回已有完成帧。细粒度像素计数单独抽样，先测 instrumentation 开关的开销，不把每像素全局统计原子加入正式性能基线。记录最终有可见像素的 cluster 数可用独立诊断 resolve，避免把“通过粗剔除”称为最终可见。

Full 新 Nsight 采样至少选起点静止、近墙、转向三个固定状态；使用优化 shader + capture line info，不使用 ShaderDebug。采集 dispatch 大小、寄存器/shared/occupancy、active threads、barrier/scoreboard、缓存和 atomic 相关可用计数；检查 replay 画面、工具兼容、计数器有效性和硬件事件溢出。已有失败导出不能作为新测量。当前 P0 导出还缺上述工作量数，因此本报告没有给出伪精确加速倍数。

内核 A/B 应尽量复用同一驻留与 cut；随后再跑原 180 秒路线三轮，保留动态异步流送造成的差异，比较逐阶段 p50/p95/p99 和超 33.33 ms 连续段。维持 DLSS Quality、render extent、材质、阴影和 1.5 px；覆盖 masked/double-sided、反射/非均匀变换、裁剪、jitter、普通/反向 Z、early/late 遮挡恢复以及等深 ID。

## 4. 对 30 fps 目标的影响

只降低分类仍留下约 51 ms 的 SW，因此必须同时处理软件光栅。相反，当前约 73 ms 两项之外仍有约 13 ms 的 graph GPU 工作；若其余工作不变，要让 graph 本身进入 33.33 ms，两项合计需压到约 20 ms，约为当前的 28%。这是粗略预算约束，**不是承诺 3.6 倍可实现提速，也不等于完整编辑器已达 30 fps**；CPU Stream Begin 和显示循环还要独立收敛。

几何池持续 >99.2%、页面准入失败尝试和 BLAS fallback/overflow 说明容量问题仍需并行分析，但不能由“池满”直接推出 SW 变慢：粗 LOD 回退可能减少三角形、增大屏幕三角形，实际方向必须由工作量计数判定。增加显存预算也不能代替本轮优化。

Epic 公开文档强调 Nanite 依赖 LOD 与遮挡使工作量随像素规模变化，同时明确多层贴近表面和聚合几何会破坏这种缩放；因此最终应看可见工作量/像素以及 overdraw，而不只看 dispatch 时间。[Epic 内容与性能说明](https://dev.epicgames.com/documentation/unreal-engine/working-with-naniteenabled-content)。技术来源还包括 [SIGGRAPH 2021 Nanite 原始讲座入口](https://advances.realtimerendering.com/s2021/index.html)；这里具体分类/扫描算法以已核对的本机 UE 5.7.4 源码为准，未做同场景 Nanite FPS 比较。
