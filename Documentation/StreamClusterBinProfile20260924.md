# streamClusterBinMain：基于 Shader Profiler 的改进方案

输入：[streamClusterBinMain.csv](../Captures/NsightGraphics/streamClusterBinMain.csv) 与用户提供的 kernel stall 截图。本轮仅分析，没有改动 shader、RHI 或用户正在修改的编译选项。

**优先精简并协作加载分类数据，然后去掉 cull→classify 的重复判断。** 数据最直接指向初始化后其他 warp 的等待，以及分类前的依赖加载；尚不支持把主要成本归于像素原子竞争、共享内存带宽饱和或算术吞吐饱和。

## 数据口径

CSV 是 Shader Profiler 的源码/IL 导出，包含多个 shader 模块，不是 GPU Trace 的 FRAME/D3DPERF_EVENTS 表。使用 `cli-anything-nsight-graphics gpu-trace summarize` 会缺少必需表，不能套用该解析器。此次按 CSV 自身的两套表头逐段解析，只累加目标 SPIR-V 模块内带数字 IL 行号的 `Samples`，排除嵌入源码镜像和高层 `Total Samples` 的重复归因。

- 目标模块：`comp.10000.spv (ef24b8b4bc22971f)`，`OpEntryPoint GLCompute %streamClusterBinMain`，LocalSize 128×1×1。
- 目标 IL 自身样本合计 **10,310**；另有 cull 模块 28 个样本，已排除。kernel 主体 4101–4150 行与当前源码逐行比对一致（忽略缩进）。
- 截图显示 **9.59K / 87.60%**，与 CSV 合计不同。其选择范围/函数包含关系未完整提供，因此截图用于定性对照，不与 CSV 混用分母。87.60% 不能解释为 GPU 帧耗时占比。
- 截图 Barrier 34.46%、Long Scoreboard 14.07%、Wait 11.21%、Short Scoreboard 7.52% 都是采样状态占比。Not Selected 14.71% 表示 warp 已具备发射条件但调度器选择了其他 warp；Selected 6.60% 则确实发射了指令，两者不能简单当作可消除的等待。
- CSV 每个指令只列前三个 stall，因此累计原因计数是下界，未列出的原因不是零。此次没有 kernel 的 GPU 毫秒时间、分派次数、实测 occupancy 或 cache throughput，不能给出端到端收益百分比。

Nsight 将等待记录在 warp 无法继续前进的 PC，可能是依赖项的消费者，也可能是同步后的第一条指令。因此 `Barrier + LSU Shared Memory Load` 不等于 shared load 自身耗费了这些周期；见 [NVIDIA Shader Profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)。

## 定位到源码的证据

| 位置 | 样本 | 占目标 IL 样本 | 解释 |
|---|---:|---:|---|
| 首次同步后读取 vertexCount 的 IL 9834 | 2,795，其中 Barrier 2,783 | Barrier 占 27.0% | 对应源码 4138；读取之前是 lane 0 初始化共享结构 |
| 顶点投影后的同步，读取 triangleCount，IL 10424 | 538，其中 Barrier 523 | Barrier 占 5.1% | 对应源码 4141；等待各 warp 完成顶点工作 |
| 最终分类结果同步后的分支，IL 10918 | 364，其中 Barrier 356 | Barrier 占 3.5% | 对应源码 4149；等待 wave 归约/共享结果 |
| `needsCoverage`，源码 4116 | 1,108 | 10.7% | record→group→instance→flags 依赖链，还有动态 uint 除法 |
| candidate→record，源码 4110 | 441 | 4.3% | 额外依赖加载 |
| 整体置零/params/group 拷贝，源码 4127–4129 | 235 / 415 / 193 | 合计 8.2% | lane 0 串行初始化；与首个 barrier 的等待相联系 |

三个明确的 Barrier 热点共 3,662 个样本，其中 **76.0% 位于第一次同步之后**。这给出了改造先后顺序：先缩短第一次同步前的生产者路径，再研究后续两个同步。

CSV 高层源码的 4138 行共有 3,069 个样本，包含同一源代码行关联的多条 IL；不能把这 3,069 再加到表中的 IL 9834 上。其他调用点的 Total Samples、Dependency-Attributed Samples 同样不是可直接与 self samples 相加的独立时间。

### 不应从数据得出的结论

- 目标 IL 的最大 `Live Registers` 为 **36**。高层 helper 行出现 200+ 的值来自跨模块源码视图，不能拿它证明该 kernel 用了 205 个寄存器。即使 36，也不是驱动分配的 `# Reg`，不能由此计算 occupancy 或断言有无 spill。
- Barrier 高不能证明 shared bank conflict；Long Scoreboard 高不能证明 DRAM 带宽饱和。
- 这个 kernel 是软硬分类，源码只有用于组内结果合并的 `InterlockedOr`；软件光栅逐像素的 64-bit `InterlockedMax` 不在这里。不要用这张图决定先做 tile depth/atomic 优化。
- 导出的编译字符串包含 `-g2`，未显示 `-O0`；仅有符号不等于未优化编译。不能先把热点归咎于 Debug 模式。

## 建议实施顺序

### P0：分类专用小结构 + 首 wave 协作装载

[当前初始化](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4122) 由 lane 0 将整个 `StreamRasterCluster` 置零、复制通用 Params、复制整个 ActiveGroup，再写几个地址/count。其他三个 warp 等待；后续 helper 又按值使用这个大结构。

先保留 128 threads 和原有三个组同步，只改变数据组织：

1. 新建分类专用 shared 数据，只保留实际相机字段、world0..3、position word/stride、vertex/triangle count、triangle byte。移除分类不用的 page/request/LOD/material/bounds 等状态；每个需要的字段显式写入后，不再整体清零。
2. 首个完整 wave 的多个 lane 负责互不依赖的字段，例如 lane 0..3 读 world0..3，其他 lane 读相机向量、计数和几何地址；优先相邻字段合并读取，不让每个 lane 各自加载整个 struct。
3. 使用分类专用的顶点变换 helper，避免传递/加载整个通用结构。保持当前浮点表达式、render/cull camera 选择、jitter 和 near/far 判定。
4. 首次 `GroupMemoryBarrierWithGroupSync` 仍然保留。没有证明跨 warp 数据已就绪前，不能用 wave barrier 或直接删除同步。

这个阶段不需要改变 cull/classify buffer ABI，改动范围最小，正好检验最强的假设。验收看首次 barrier 的绝对样本/每工作量样本与 kernel 时间一起下降；仅看 Barrier 百分比下降可能是其他成本上升。

### P1：把确定的 HW 判断收敛到 cull，明确分类输入契约

[cull](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4071) 已经拥有 `instanceFlags`、tessellation 状态和 `forceHardware`。但当快速分类被禁用或保守回退 flag 生效时，确定的 HW 也可能进入精确分类列表；因此不能直接删除 [classifier 中的 coverage 检查](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4111)。

应先建立契约：进入精确分类列表的条目已确定是可精确判断的 opaque、非 tessellation cluster。将必须走 HW 的情况在 cull 中直接写原 candidate 的 tag；使球体 fast-SW 开关与这种语义确定的 HW 分流解耦，再去掉重复的 raster bindings、instance flags 和 groupIndex 查询。保留旧完整路径作参考，并覆盖 metadata 开关关闭及非法包围球时的行为。

进一步可在紧凑条目里携带 record/groupIndex 等已知数据，消除 candidate→record 加载和 `record / bins[13]`，但这应单独 A/B：条目目前为 4 words，改变布局必须同步 producer/consumer、scratch 区间与容量计算，额外写读可能抵消收益。不要把同一片 bin scratch 当作无限可扩展缓存。

### P2：初始化优化后，再试单 wave 或更小工作组

若仍有显著组间等待，再对照 32/64/128 threads。32 threads/cluster 可以用 `for (vertex=lane; vertex<vertexCount; vertex+=32)` 和对应 triangle loop 完整处理 128 个顶点/三角形；最终 HW 判断可直接 wave any，无需四个 wave 的共享 OR 归约。

但顶点输出供其他 lane 索引读取，仍需要符合 Vulkan/Slang 内存模型的同步，不能依靠“一个 warp 自动同步”假设。还要处理设备 subgroup size、尾 lane、波内循环退出与所有 lane 参与 collective。更少 threads 会增加每 lane 工作，可能降低访存并行度或提升寄存器需求，所以先作为可切换实验，不直接改默认。

### P3：再削减投影与三角索引工作

- `projectPosition()` 对每个顶点重建 camera basis、normalize、tan。可将不变的相机量预计算一次并复用；若每 cluster 放到 lane 0 计算，会重新加长首个 barrier，优先考虑按 view 准备或协作计算。保持表达式/精度和原分类边界一致。
- `hybridTriangleFits()` 为每个三角形重复执行三个顶点的透视除法、viewport 变换、有限值/clip 检查。可在唯一顶点阶段缓存 screen XY 和有效标志，三角阶段只做包围盒极值/阈值判断。需要沿用未 snap 的分类坐标、原始 float 判定及不确定边界回退，避免偷偷改变 HW/SW 分桶。
- CSV 在 `readPageByte()` 后的移位操作处出现大量 Long Scoreboard，例如目标 IL 10481 有 301 个此类样本，提示先前 global load 的依赖。可为连续三个 uint8 索引做分类专用解码，评估 1–2 次 uint word load 替代三个独立 readPageByte；跨 word/页末尾边界必须正确，不可无条件读第二个 word 越界。编译器是否已合并读取仍需最终 SASS 核对。
- 分类的 `hybridTriangleFits(a,b,c)` 对顶点排列对称，通常不需要 `streamClusterTriangle()` 的反射绕序修正；可使用不改 winding 的纯索引解码，保留非法索引 fallback。不过对应 reflection 行样本很少，这不是第一优先项。

## 验证标准

建议拆为三个可归因的实验：A=thin shared/cooperative load；B=A+producer 输入契约；C=B+32/64-thread 或投影缓存。每一步与当前 kernel 在同一 camera、cut、驻留、render extent、LOD、软件阈值及 HZB 状态下换序对照。

1. 记录精确 classifier 实际 workgroup 数，以及 fast SW/HW、fallback 数。比较 `cull + classify + stable bins` 合计，避免把成本转移到 cull 却宣称整体优化。
2. 比较完整 candidate tags、stable bins、record IDs、early/late retry、最终 visibility/depth；复用 [StreamClusterClassificationTests.cpp](../tests/rhi/StreamClusterClassificationTests.cpp) 的参考路径。覆盖 MASK/BLEND、tessellation、非法页面/球体、float3/float4、jitter、反射/shear、裁面、空列表和超 65535 的调度。
3. 重新采集同范围 Shader Profiler 与 GPU timestamp，分别看首个/第二个/最终 barrier、Long/Short Scoreboard、kernel 总时间，以及真实 `# Reg`、shared allocation 和 occupancy。固定窗口下的样本总量也受采样配置、工作次数、时钟影响，不能只比较百分比。

当前已有足够证据决定先做 P0/P1，但不足以承诺减少多少毫秒。截图中的 34.46% Barrier 不是删除同步后必然可回收的 34.46% kernel 时间。

证据：[去重统计 JSON](StreamClusterBinProfile20260924.json)、[可复现解析脚本](../build/nsight-visibility-20260924/AnalyzeBinProfile.py)。
