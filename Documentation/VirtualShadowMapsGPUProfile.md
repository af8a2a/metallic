# VSM GPU 阶段计时与工作量

`VirtualShadowMapPass` 和 `VirtualShadowMapLightsPass` 提供两个默认关闭、相互独立的 graph properties：

```json
"benchmarkGPUProfile": true,
"benchmarkGPUWork": false
```

`benchmarkGPUProfile` 通过现有 RenderGraph GPU timestamp/profile scope 发布阶段，编辑器 Profiler 和 acceptance JSON 都可读取。`benchmarkGPUWork` 编译诊断 shader permutation，并增加 `workload` 和独立 `buildDiagnostics` buffer 输出；修改该开关需要重新编译 graph。普通计时必须关闭 work，计数诊断中的时间包含原子操作开销。CPU attribution (`benchmarkCPUProfile`) 与详细 GPU profiling 分开运行，避免耗尽共享 section/query 预算；已有 `profilingOverflow` 仍是无效计时，不能将缺失阶段当零。

## 时间范围

每个已录制 face 各有以下六个 section。多灯报告逐阶段相加，不再累加父 scope。现有 query 能力、队列和完成信号规则不变，没有新建同步 readback 或额外 GPU wait。

| Section | 包含工作 |
| --- | --- |
| `VSM reset` | HostWrite/复用资源可见性、旧页表依赖、reset/remap 与其输出 barrier |
| `VSM mark` | receiver 需求标记及其输出 barrier |
| `VSM allocator` | count、prefix、admit、physical、assign 与内部 barrier |
| `VSM build` | 三角/页工作生成、finalize/indirect 参数生成和输出 barrier |
| `VSM raster` | atlas attachment 转换、冷 clear 或热页 clear、coarse/fine mesh raster、转为采样状态 |
| `VSM projection` | 输出资源同步、receiver projection/filter 及输出 barrier |

Reset 单列，避免把分配器之前的 remap 误计入 mark。时间戳沿用 BottomOfPipe，表示命令区间而非纯 shader 算术时间。父 pass 还包括 caster preparation、publication、可选 temporal 等工作，六项之和不等于整个 Shadows pass。空工作仍可产生一个合法的零/近零时间区间；不可用时间必须保持缺失。

## 工作量接口

原 `statistics` 仍为每 face 32 个 uint32，索引语义不变。诊断模式下私有 statistics buffer 尾部增加 128 B，公开为独立 `workload`：每 face 16 个 uint64，按 low/high 两个 uint32 编码，GPU 仅需要 32-bit atomics。单灯输出 128 B，多灯输出 `lightCapacity × 6 × 128 B`，与 compact 屏幕输出层数无关。未启用、未准入及未使用 face 清零。

独立 build 诊断在上述两个 ABI 之后再增加 200 B，即 byte 256 起的 25 个 uint64；work 开启时每 face 私有 statistics 总长为 456 B。公开 `buildDiagnostics` 是每 face 200 B，单灯 200 B、多灯 `lightCapacity × 6 × 200 B`；旧 32-word statistics 和 16-counter workload 的长度、偏移与含义不变。

定义见 [VirtualShadowMapProfiling.h](../Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h)。计数均在实际 GPU 路径产生，不以 triangle capacity 代替成功解码量。

| 阶段 | Counter | 含义 |
| --- | --- | --- |
| mark | `markReceiverDispatches` | 域内有深度的 receiver 线程次数；点光不同面可重复处理同一像素 |
| mark | `markSamplePositions` | markPosition 调用次数，包含随后被点光面检查拒绝的采样 |
| mark | `markLevelTests` | 通过面检查后的实际 level 投影测试 |
| mark | `markPageRequests` | 去重后实际 page-request atomic OR 次数，不是 unique pages |
| build | `buildActiveTriangleSlots` / `buildDecodedTriangles` | 通过范围/enabled/dirty 早退及可选 cluster 粗筛后的槽数 / 成功解码三角数 |
| build | `buildLevelTests` / `buildPageTests` | 通过可选 cluster level mask 后的实际三角×level 次数 / 精确三角阶段在 dirty/admission 检查前枚举的包围矩形页数；均不包含 cluster 粗筛本身 |
| raster | `rasterCoarseGroups` / `rasterFineGroups` | 实际执行的 coarse/fine mesh workgroups |
| raster | `rasterDecodedTriangles` / `rasterOutputPrimitives` | 成功解码的组数 / 裁剪后输出的三角数；不表示 fragments |
| raster | `rasterClearedPages` | 热缓存 mesh clear 的实际页数，不包含冷启动 attachment clear |
| projection | `projectionReceivers` | canonical face、enabled、domain 检查后进入查询的 receiver |
| projection | `projectionRays` / `projectionAtlasLoads` | 实际 SMRT ray 迭代 / hard 或 filtered atlas.Load 次数 |

Allocator 沿用旧统计中的 requested/admitted/rejected fine pages、dirty/cached pages、invalidated/remapped/newly assigned/valid resident pages。报告把这些页状态与 shader 执行事件分开。原 `statistics[6]` 在 point canonical-face 检查前计数，不能作为 shaded pixels；`statistics[30]` 是尝试的 filtered lookup，不能作为成功 atlas loads。

诊断 shader macro `VSM_WORKLOAD_DIAGNOSTICS` 属于 ShaderRegistry 缓存身份。关闭时编译掉新增计数代码；208 B push layout 和现有直接 shader tests 的 128 B statistics 合同保持不变。缓存深度有效性不依赖诊断开关。新增 mesh 写入通过 compute→mesh、mesh→compute/transfer 和完整 buffer import 同步。

## 独立 build 诊断

`shadow.gpuBuildDiagnostics` 使用独立 version 1 JSON，字段为 `enabled`、`faces`、`totals`。每个 face 沿用 `lightIndex`、`face`、`counters`；不是 compact 屏幕输出层。它与 `shadow.gpuWork` 共用 `benchmarkGPUWork` 开关，但两个 JSON 对象和原 16 个 counter 的 ABI 相互独立。关闭时只发布 `version/enabled:false`，普通 shader 不执行这些新增计数原子。

| Counter | 实际计数位置与含义 |
| --- | --- |
| `clusterGroups` | 通过 enabled、范围和非零 dirty 检查的 build workgroups；每组对应 64 个 triangle slots，不是 64 个成功解码三角 |
| `clusterBoundsValidGroups` | 成功读取保守 cluster bounds 的组数 |
| `clusterSurvivingGroups` | 粗筛后 level mask 非零的组数 |
| `clusterLevelTests` | valid bounds 的每组对每个 fine level 执行的检查次数 |
| `clusterPageTests` | cluster 粗筛实际读取的 page-entry 数，包含首次命中的那一次；不包含后续精确 triangle 阶段 |
| `clusterPageHits` | 在首次 eligible page 处结束的 level 检查次数 |
| `clusterNearPlaneKeeps` | perspective near-plane 保守保留的 level 次数，尚未读取 page |
| `clusterOutsideLevels` | guarded projected bounds 外拒绝的 level 次数，尚未读取 page |
| `clusterEmptyLevels` | 扫完候选 pages 仍无 eligible page 的 level 次数 |

另外 16 项 histogram 为 `clusterPageScan0`、`clusterPageScan1`、`clusterPageScan2To3`、`clusterPageScan4To7`、依此类推至 `clusterPageScan8192To16383`、`clusterPageScan16384Plus`。每个 level 检查恰好贡献一个 observation。长度是**遇到 early hit 或返回前的实际读页次数**，不是投影矩形面积；near/outside 对应长度 0。当前最大 axis 为 128，单次最多读取 16384 页，因此最后一档当前恰好代表 16384 次。

报告单独验证每个 frame/face 的 uint64 值及 face totals，拒绝累计溢出或不完整 enabled payload。利用原 statistics 的 clipmap 数 `stats[11]` 验证：

- `survivingGroups <= validGroups <= groups`。
- `levelTests = validGroups × stats[11]`；hit、near、outside、empty 四种结果之和等于 levelTests。
- histogram 总数等于 levelTests，零档等于 near + outside。
- pageTests 落在各 histogram 档加权下界与上界之间，单次上限为 16384。
- inactive、triangle capacity 为零或 `stats[16] == 0` 的 face 必须全零。

关闭 cluster 粗筛但开启诊断时，全零 build sidecar 是合法的已测数据；工具不会强制 groups 非零。旧证据缺少整个新字段时标为 `gpuBuildDiagnosticsAvailable:false` / `missing`，不会用零补齐；有新版字段却缺少部分 frame readback 或 enabled 对象的数据则失败。`work_frame` 的原返回结构不变，由独立 `build_frame` 解析新对象。Timing 模式禁止 enabled build sidecar。

## 采集和报告

使用既有 [acceptance runner](../Tools/Perf/VirtualShadowMapAcceptance.py)，在各自的新目录串行采集：普通 `timing` 只开 GPUProfile；`diagnostic` 可同时开 GPUProfile/GPUWork。后者每帧额外导出 `shadow.gpuWork` 和独立 `shadow.gpuBuildDiagnostics`，各自包含 version、enabled、逐 face 与 totals。Timing 模式拒绝 GPUWork。

```powershell
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases <timing-cases.json> --exe build-release/tests/MetallicRHITests.exe --output <new-timing-package> --runs 3
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases <diagnostic-cases.json> --exe build-release/tests/MetallicRHITests.exe --output <new-diagnostic-package> --runs 1
python -B Tools/Perf/VirtualShadowMapGPUProfile.py <timing-package> <new-report-directory> --diagnostic <diagnostic-package>
```

报告共用 acceptance 的包完整性校验，保留模式、源码/二进制/输入身份与 warmup；输出各进程中位数、逐帧/逐 face CSV、JSON、PNG 和复现脚本。统计单位为独立进程，单帧分布不是独立样本。Diagnostic 时间与普通 timing 分开，不从本次 profiling 推导优化收益。

Build sidecar 独立导出 `BuildDiagnosticFrames.csv`、`BuildDiagnosticFrames.json` 和存在有效数据时的 `BuildDiagnosticFaces.csv`。这些独立文件只包含 diagnostic 模式行，保留 package 标签；普通 timing 的 unavailable 状态仍可在各 package JSON 中查看。Histogram 图按每进程的 post-warmup 帧中位数汇总，再取进程中位数；原始每帧、每 face 数据保持完整。新 sidecar 不进入既有四臂优化工具的性能判定，也不与旧 workload 或普通计时混合。

Focused GPU 回归 `virtual_shadow_map_profiling` 比较 Off/Timing/Work/Both 四组同帧结果，覆盖 resident/stream、单灯、多灯 legacy/compact、cold/warm/dirty/overflow。既有 allocator、remap、content 和 filter-demand 回归继续覆盖原统计与缓存协议。报告工具的 CTest 名为 `MetallicVirtualShadowMapGPUProfileTests`。

## 2026-10-29 初始 profiling 验证（mark/cluster 优化前）

Release `Metallic` 与 `MetallicRHITests` 构建通过。GPU 上执行 profiling、compact、allocator、remap、demand_equivalence 五项回归全部通过；profiling 的 16 个状态各比较四种开关，图像、深度、旧 32-word statistics 一致，Timing 不改变 work counters。Core/synchronization validation 无错误。最初测试夹具遗漏太阳 illuminance，非零工作断言失败；已明确设为 3 并保留该次失败记录。15 项报告工具测试与 Slang module audit 均通过。

采集保留 `Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json` 的 native 1920×1080、LOD0、6 clipmaps、128×128 virtual pages、2048 physical pages、2 rays×16 samples、32-phase temporal SMRT 配置，仅添加诊断开关。设备为 RTX 5070 Ti，驱动 621.01。Timing 为 3 个独立进程，每个 96 帧、前 32 帧 warmup；diagnostic 为独立 64 帧、前 32 帧 warmup，包含两张场景 PNG，validation messages 为 0。均无 RT/CLAS 分配，输入、源码、二进制在各轮采集期间未变化。

| 阶段 | 三个进程中位数的中位数（ms） | 进程中位数范围（ms） |
| --- | ---: | ---: |
| reset | 0.0081 | 0.0080–0.0083 |
| mark | 2.6027 | 2.5223–2.6240 |
| allocator | 0.0889 | 0.0889–0.0896 |
| build | 8.0920 | 8.0250–8.2447 |
| raster | 0.0489 | 0.0475–0.0489 |
| projection | 1.0550 | 1.0546–1.0555 |

这组 workload 的主要耗时在 build。诊断后半段每帧中位数只有 23 个 dirty/newly allocated fine pages、约 5822 个 fine raster groups，但 build 仍解码 262267 个三角、执行 1573602 次 level 和 5850128 次候选页检查。Mark 每帧约 7040 万个采样位置、2.364 亿次 level 测试、1098 万次 page-request OR；projection 约 414 万条 SMRT rays、1452 万次 atlas loads。Allocator 请求 3629 个 fine pages、准入 2047 个，未准入 1582 个。后续可据此评估减少 build 全场景扫描及 mark 投影重复的收益；这里没有实施这些算法优化。

本机证据在 `build/vsm-gpu-profile/`：`RegressionFixed/`、`Diagnostic/`、`Timing/` 和 [ReportFinal/report.md](../build/vsm-gpu-profile/ReportFinal/report.md)。`ReportFinal/` 保存完整 CSV/JSON、图表与 `Reproduce.py`。检查了 Sponza 最终输出和计时/工作量图表；图像等价的自动断言来自前述回归夹具，未声称对 Sponza 做了 instrumentation on/off HDR A/B。保留磁盘 shader/PSO caches，串行运行、未锁 GPU clocks，也未保证系统 GPU 独占。这里的阶段时间不代表优化收益或整帧 FPS。

## 2026-10-29 build 诊断补充验证

Release 构建通过；报告工具 27 项测试、四臂工具 19 项测试及 Slang module audit 通过。`virtual_shadow_map_profiling` 的 20 个状态比较 Off/Timing/Work/Both，四臂优化等价回归的 32 个状态也通过，validation 均为 0。检查涵盖 resident/stream、单灯/多灯、hard/PCF/SMRT、cold/warm/dirty/overflow，以及禁用 cluster 粗筛时的零 sidecar。新增 TinySingleHard 使用 VPA=2、4 physical pages，覆盖小 dispatch 和 warm 清零；短扫描不会写高位 histogram 桶，因此该夹具不单独证明高位非零内容的复用清零。CPU reset 已明确覆盖全部 82 个诊断 uint，shader 清零索引为 `statistics[32 + id.x]`。

首次 Tiny 配置 VPA=1 因细级投影没有覆盖夹具 receiver 而未产生 fine work，保留在 `build/vsm-build-diagnostics/RegressionGPU/`；VPA=2 修正后的 20 状态成功记录在 `ProfilingFixed/`。前一目录中的四臂测试已成功，修正仅涉及 profiling 夹具，未更改渲染算法。

固定 Sponza 继续使用已默认开启的 shared mark 与 cluster culling，原 native 1920×1080、6 clipmaps、128×128 virtual pages、2048 physical pages、SMRT 2×16 配置不变。独立诊断进程采集 64 帧，前 32 帧 warmup，第 16 帧冻结 streaming 并 camera cut。与 `build/vsm-optimization/DefaultBothControl/` 的 sealed 基线核对，第 16–63 帧 canonical 输入及完整 stats32 相同，16/32/63 三个检查点的 PNG、线性 HDR、深度和阴影 hash 完全相同。新增 sidecar 的所有 64 帧通过逐 face 校验，场景及 histogram 图片已检查。

| 指标 | 后 32 帧中位数 |
| --- | ---: |
| cluster groups / valid bounds | 4098 / 4098 |
| surviving groups | 1072 |
| cluster level tests | 24588 |
| cluster-level 执行页扫描的次数（hit + empty） | 11544 |
| cluster 实际页读取 | 2934974 |
| hit / near / outside / empty | 1429 / 0 / 13044 / 10115 |
| fine triangle 页检查（原 workload 定义） | 1688862 |

Histogram 有 16 桶：`0`、`1`、`2–3` 至 `8192–16383`，以及 `16384+`（当前最大恰为 16384）。后 32 帧合计 369408 次非零扫描、94023618 次页读取；其中 1504 次完整 16384 页扫描占扫描次数的 0.407%，贡献 26.21% 的粗筛页读取。这个长尾可用于评估下一步 dirty-page bitset；读取数量不等于耗时占比。

完整逐帧/逐 face CSV、JSON、扫描长度图和复现入口见 [build 诊断报告](../build/vsm-build-diagnostics/Report/report.md)，一致性证据见 [Correctness.json](../build/vsm-build-diagnostics/Verification/Correctness.json)。本轮没有新增普通计时实验，诊断耗时不用于推导性能收益。
