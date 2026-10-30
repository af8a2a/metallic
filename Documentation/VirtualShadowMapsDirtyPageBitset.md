# VSM dirty-page bitset

`VirtualShadowMapSettings::clusterPageBitset` 及两个 VSM Pass 的生产配置现在默认开启，无需修改已有 RenderGraph。图属性 `clusterPageBitset: false` 可恢复逐页扫描。显式布尔属性 `benchmarkClusterPageBitset` 优先于生产属性，保留历史 A/B 图的 false/true 覆盖能力；关闭 cluster culling 时，bitset 的有效值始终为 false。

本次仅切换生产默认与配置读取，不改变 bitset 算法、页选择、精确三角裁剪或光照。Runtime 按有效值同时配置 owners scratch、shader 宏和程序缓存排列。独立 shader 入口的缺省宏仍为 0，兼容自行分配旧 owners 布局的 allocator/remap 测试；正常生产录制始终显式传入宏值。

`Tools/Perf/VirtualShadowMapBitset.py` 现在要求 A/B 两臂显式填写布尔型 benchmark 开关。缺省 A 不再被当成关闭，因为这与新的生产默认冲突。

## 实现

每个 face 在 allocator 的 owners scratch 末尾追加 `ceil(totalPages / 32)` 个 uint。Reset 清零；Admit 为已有 resident、admitted、dirty 页置位；Assign 为新分配的 admitted 页置位。Physical 不会驱逐已准入的旧 owner，因此 bit 与 build 原先的三条件页判定一致。已有 owners buffer 的 compute barriers 覆盖追加区域，无新增 dispatch。

Cluster 将原投影矩形逐行转换为 bit 区间，以首尾 mask 约束每行，遇到非零 word 即保留该 level。缓存未掩码的最后一个 word，支持非 32 对齐页轴和相邻行共享 word。近面保守保留、投影 guard、精确 build/raster 路径保持原有行为。

## 诊断

`benchmarkGPUWork` 的 build sidecar 升为 version 2：43 个 uint64，共 344 B，仍从 private statistics byte 256 开始。旧 25 项定义保持不变；追加 `clusterBitsetLevelTests`、`clusterBitsetWordTests` 和 16 桶 `clusterBitsetScan*`。

Legacy 的 histogram 记录真实 uint4 页读取；bitset histogram 记录真实 uint 字读取。两者分别累计，禁止将逻辑覆盖页数冒充实际读取。共享 level tests/outcomes 仍描述同一筛选结果。报告工具同时支持 v1/v2，旧证据缺少 bitset 指标时保持缺失。

## 预先固定的验证预算

独立工具为 `Tools/Perf/VirtualShadowMapBitset.py`，复用 acceptance 证据封装及 ABBA 区组统计。Sponza 固定 native 1920×1080、6 clipmaps、128×128 virtual pages、2048 physical pages、SMRT 2×16；第 16 帧 freeze streaming 与 camera cut，warmup 32 帧。

- Diagnostic：A/B 各 1 个进程，64 帧；检查 canonical 输入、完整 stats32 和 HDR/depth/shadow/PNG 检查点。
- Discovery：2 个 ABBA 区组，共 8 个进程，每进程 96 帧；普通计时关闭 GPUWork 和 validation。
- Confirmation：若 discovery 值得继续，独立 3 个 ABBA 区组，共 12 个进程；不与 discovery 合并。
- 主指标 build，目标收益 3%；graph 回退不得超过 2%。统计单位为完整独立 ABBA 区组，按区组 log 比值求区间。

GPU 实验串行、源码和二进制冻结；shader 日志记录实际 SPIR-V SHA256 及 `clusterPageBitset`。时钟未锁定，合作串行不保证系统 GPU 独占。诊断时间不参与性能判定。

## 2026-10-30 候选阶段验证结果

Release `Metallic` / `MetallicRHITests` 构建、Slang module audit、66 项相关 Python 测试通过。GPU 三套回归共 114 状态：bitset 等价 62、原四臂等价 32、profiling 20，validation 均为 0。3120 个 CPU 整数模型用例覆盖 VPA 1–128 的关键边界及 12 个 level；实际 GPU 矩阵另覆盖 VPA 31/33 的非对齐行。

Sponza 诊断 A/B 在第 16–63 帧 canonical 输入、完整 stats32、15 项确定性 workload 相同；16/32/63 检查点的 PNG、线性 HDR、depth、shadow 完全一致。每帧中位数都是 4098 个 cluster 组、1072 个 surviving 组，精确 triangle page tests 都是 1688862。粗筛由 2934974 次 uint4 页读取变为 166530.5 次 uint bitset 读取；两者的元素大小不同，不据此推导 DRAM 流量。无新增 dispatch，Sponza 每 face 追加 12292 B scratch。

Discovery 的两个独立 ABBA 区组取得 build 47.58% 的收益估计，随后按预算完成三个独立 confirmation 区组（12 个进程，A/B 各六个）。下表时间为各臂进程中位数的中位数，收益及区间来自 ABBA log 比值：

| GPU 范围 | A (ms) | B (ms) | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| build | 5.3293 | 2.7744 | 47.30% | 44.41%–50.04% |
| Shadows pass | 9.3924 | 6.8607 | 26.59% | 23.62%–29.44% |
| 完整 RenderGraph | 12.8528 | 10.3096 | 19.28% | 15.38%–22.99% |

判定为 `evidenceValidity=valid / confirmed_gain / accept`，达到预设 build/graph 门槛。Allocator 变化的区间跨零；projection 有约 0.36% 的小回退，全部阶段及不确定性均保留在报告。该结论针对上述固定 Sponza workload，不是端到端 FPS 测量。该轮验证时默认开关为 false，使用 `benchmarkClusterPageBitset: true` 启用候选；下面的生产验证独立采集，不复用该轮二进制或收益。

- [正确性及计数报告](../build/vsm-bitset/DiagnosticReportVerified/report.md)
- [Discovery 报告](../build/vsm-bitset/DiscoveryReport/report.md)
- [独立确认报告、逐进程和区组图](../build/vsm-bitset/ConfirmationReport/report.md)
- [复现绘图脚本](../build/vsm-bitset/ConfirmationReport/Reproduce.py)
- 原始封装：`build/vsm-bitset/{Diagnostic,Discovery,Confirmation}/`；GPU 回归：`build/vsm-bitset/Regression/`。

首次 diagnostic 汇总因 raw graph 未登记 Acceptance 自动添加的两个输出而被拒绝。修复严格限定为 case 显式声明的 display/comparison 输出，增加回归测试后对原封装离线复核通过；原始采集、Plan、失败记录未修改，也未重采挑选样本。图表及 candidate 最终场景输出已人工查看。

## 2026-10-30 生产默认验证

基于 `0227e2cc` 的屏幕输出/点光表共享实现独立复验。本次保留原算法，只改运行时默认、生产属性读取和性能工具的显式开关要求。现有样例无需追加属性即可启用；`clusterPageBitset: false` 是生产关闭路径，显式 benchmark 覆盖优先。

Release 编辑器、VSM Sample、RHI 构建及 Slang 审计通过（387 files / 275 modules / 112 implementation units），68 项性能工具测试通过。7 项 GPU 回归全部通过、无跳过：cache、profiling、optimization equivalence、cluster-page-bitset equivalence、projection equivalence、temporal lifecycle、content。

bitset 矩阵保持 62 状态，默认臂省略两个开关，基线显式设置生产 true + benchmark false；其中 10 个状态追加生产关闭和禁用 cluster culling 两臂。图像、32 项统计及确定性 work 门槛保持严格。实际 bitset level tests 为 680（单灯 Pass 96、多灯 Pass 584），证明测试执行了生产位图路径。原 32 状态优化矩阵保持 bitset 显式关闭；20 帧 profiling 验证生产默认，78 状态多灯投影矩阵也通过。独立 shader 的旧资源布局与运行时位图布局同时保留。

两种完整场景均为 RTX 5060、原生 1920×1080、SMRT 2×16、32 帧历史、2048 页、关闭 jitter/debug 输出，并使用当前共享点光表路径。A/B 都显式设置生产属性 true，只由 benchmark false/true 控制实际选择。每个 diagnostic 64 帧、warmup 32，第 16 帧冻结流送并 camera cut；16/32/63 的 PNG、线性 HDR、depth、shadow 精确一致，第 16–63 帧完整 stats32、canonical 输入及15项稳定 work 一致。并发 mark 原子请求次数按既有协议保留原值、不要求精确相同。两种最终截图均已目视检查。

Sponza 粗筛仍保留每帧中位数 1072 个 cluster 组；2934974 次 uint4 页读取改为 166530.5 次 uint word 读取，精确三角形页测试不变。owners scratch 的实测逻辑 buffer 增量为 12292 B，无新增 dispatch；不同元素类型的读取次数不直接换算为 DRAM 流量。

Sponza 完成 discovery 2 个 ABBA 区组后，独立完成 confirmation 3 个区组（12 个进程，每进程96帧、前32帧预热），结论为 `valid / confirmed_gain / accept`。下表 A/B 是进程中位数再取中位数；收益和95%区间按独立 ABBA log 比值计算，不能直接由两列相除得出。

| GPU 范围 | A：逐页扫描（ms） | B：bitset（ms） | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| build | 5.3268 | 2.4458 | 54.10% | 53.54%–54.64% |
| Shadows pass | 13.8284 | 11.0266 | 20.60% | 19.75%–21.44% |
| RenderGraph | 19.5166 | 16.7777 | 14.27% | 13.00%–15.52% |

普通计时关闭 validation 和诊断回读，保留磁盘缓存，不锁频、不停止用户应用。上述结果是 GPU pass/graph 耗时，不是端到端呈现 FPS。Mark、allocator 和 projection 的收益区间跨零，完整逐阶段数据保留，没有宣称这些阶段也得到加速。

Mixed 是 Sun/Moon/Point/Spot 的4灯9面静态场景，预热后没有 dirty build 工作。2 个 discovery ABBA 区组的 build 收益仅 0.15%，95% CI [-1.30%, +1.59%]，未达到3%目标，工具判定 `below_target`，未追加 confirmation。graph 收益为 -1.26%，CI [-1.49%, -1.03%]，即存在约1.3%的耗时回退，但没有越过预设2%回退限额。该场景不作为收益证据；生产化依据 Sponza 的独立确认，同时明确保留此代价和关闭路径。

所有采集与计时的源码、二进制、输入、实际 mark/build SPIR-V 及封装 hash 验证通过。没有修改性能门槛、合并 discovery/confirmation 或追加轮次追逐显著性。当前本地证据位于 `.cache/VSMBitsetProduction/`：`Regression`、`Sponza/{Diagnostic,Discovery,Confirmation}`、`Mixed/{Diagnostic,Discovery}` 及各自 `*Report`；`Cases`、`RunEvidence.py`、`Decisions.json` 保留完整输入和执行顺序。生成数据未加入版本控制。

复查命令（在原 x64 MSVC 环境中）：

```powershell
cmake --build build-release --target MetallicRHITests MetallicVSMSample Metallic --parallel 8
python -B -m unittest discover -s tests/perf -p 'TestVirtualShadowMap*.py'
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_cluster_page_bitset_equivalence:*virtual_shadow_map_optimization_equivalence:*virtual_shadow_map_profiling:*virtual_shadow_map_cache:*virtual_shadow_map_content:*virtual_shadow_map_temporal_lifecycle:*virtual_shadow_map_projection_equivalence' --rhi-bindless --rhi-validation --output-dir .cache/vsm-bitset-production-check
python -B Tools/Perf/VirtualShadowMapBitset.py run --phase diagnostic --cases-dir .cache/VSMBitsetProduction/Cases/Sponza --exe build-release/tests/MetallicRHITests.exe --output .cache/vsm-bitset-new-diagnostic
```
