# VSM GPU 阶段计时与工作量

`VirtualShadowMapPass` 和 `VirtualShadowMapLightsPass` 提供两个默认关闭、相互独立的 graph properties：

```json
"benchmarkGPUProfile": true,
"benchmarkGPUWork": false
```

`benchmarkGPUProfile` 通过现有 RenderGraph GPU timestamp/profile scope 发布阶段，编辑器 Profiler 和 acceptance JSON 都可读取。`benchmarkGPUWork` 编译诊断 shader permutation，并增加 `workload` buffer 输出；修改该开关需要重新编译 graph。普通计时必须关闭 work，计数诊断中的时间包含原子操作开销。CPU attribution (`benchmarkCPUProfile`) 与详细 GPU profiling 分开运行，避免耗尽共享 section/query 预算；已有 `profilingOverflow` 仍是无效计时，不能将缺失阶段当零。

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

## 采集和报告

使用既有 [acceptance runner](../Tools/Perf/VirtualShadowMapAcceptance.py)，在各自的新目录串行采集：普通 `timing` 只开 GPUProfile；`diagnostic` 可同时开 GPUProfile/GPUWork。后者每帧额外导出 `shadow.gpuWork`，包含 version、enabled、逐 face 与 totals。Timing 模式拒绝 GPUWork。

```powershell
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases <timing-cases.json> --exe build-release/tests/MetallicRHITests.exe --output <new-timing-package> --runs 3
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases <diagnostic-cases.json> --exe build-release/tests/MetallicRHITests.exe --output <new-diagnostic-package> --runs 1
python -B Tools/Perf/VirtualShadowMapGPUProfile.py <timing-package> <new-report-directory> --diagnostic <diagnostic-package>
```

报告共用 acceptance 的包完整性校验，保留模式、源码/二进制/输入身份与 warmup；输出各进程中位数、逐帧/逐 face CSV、JSON、PNG 和复现脚本。统计单位为独立进程，单帧分布不是独立样本。Diagnostic 时间与普通 timing 分开，不从本次 profiling 推导优化收益。

Focused GPU 回归 `virtual_shadow_map_profiling` 比较 Off/Timing/Work/Both 四组同帧结果，覆盖 resident/stream、单灯、多灯 legacy/compact、cold/warm/dirty/overflow。既有 allocator、remap、content 和 filter-demand 回归继续覆盖原统计与缓存协议。报告工具的 CTest 名为 `MetallicVirtualShadowMapGPUProfileTests`。

## 2026-10-29 本机验证

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
