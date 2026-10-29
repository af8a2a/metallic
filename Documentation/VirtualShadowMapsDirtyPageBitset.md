# VSM dirty-page bitset

`benchmarkClusterPageBitset: true` 在两个 VSM Pass 中启用候选；对应 `VirtualShadowMapSettings::clusterPageBitset`，默认 false，要求 cluster culling 已开启。A/B 的 shared mark、cluster bounds、精确三角裁剪及光照设置保持相同。

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

## 2026-10-30 验证结果

Release `Metallic` / `MetallicRHITests` 构建、Slang module audit、66 项相关 Python 测试通过。GPU 三套回归共 114 状态：bitset 等价 62、原四臂等价 32、profiling 20，validation 均为 0。3120 个 CPU 整数模型用例覆盖 VPA 1–128 的关键边界及 12 个 level；实际 GPU 矩阵另覆盖 VPA 31/33 的非对齐行。

Sponza 诊断 A/B 在第 16–63 帧 canonical 输入、完整 stats32、15 项确定性 workload 相同；16/32/63 检查点的 PNG、线性 HDR、depth、shadow 完全一致。每帧中位数都是 4098 个 cluster 组、1072 个 surviving 组，精确 triangle page tests 都是 1688862。粗筛由 2934974 次 uint4 页读取变为 166530.5 次 uint bitset 读取；两者的元素大小不同，不据此推导 DRAM 流量。无新增 dispatch，Sponza 每 face 追加 12292 B scratch。

Discovery 的两个独立 ABBA 区组取得 build 47.58% 的收益估计，随后按预算完成三个独立 confirmation 区组（12 个进程，A/B 各六个）。下表时间为各臂进程中位数的中位数，收益及区间来自 ABBA log 比值：

| GPU 范围 | A (ms) | B (ms) | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| build | 5.3293 | 2.7744 | 47.30% | 44.41%–50.04% |
| Shadows pass | 9.3924 | 6.8607 | 26.59% | 23.62%–29.44% |
| 完整 RenderGraph | 12.8528 | 10.3096 | 19.28% | 15.38%–22.99% |

判定为 `evidenceValidity=valid / confirmed_gain / accept`，达到预设 build/graph 门槛。Allocator 变化的区间跨零；projection 有约 0.36% 的小回退，全部阶段及不确定性均保留在报告。该结论针对上述固定 Sponza workload，不是端到端 FPS 测量。默认开关仍为 false，使用 `benchmarkClusterPageBitset: true` 启用已验证候选。

- [正确性及计数报告](../build/vsm-bitset/DiagnosticReportVerified/report.md)
- [Discovery 报告](../build/vsm-bitset/DiscoveryReport/report.md)
- [独立确认报告、逐进程和区组图](../build/vsm-bitset/ConfirmationReport/report.md)
- [复现绘图脚本](../build/vsm-bitset/ConfirmationReport/Reproduce.py)
- 原始封装：`build/vsm-bitset/{Diagnostic,Discovery,Confirmation}/`；GPU 回归：`build/vsm-bitset/Regression/`。

首次 diagnostic 汇总因 raw graph 未登记 Acceptance 自动添加的两个输出而被拒绝。修复严格限定为 case 显式声明的 display/comparison 输出，增加回归测试后对原封装离线复核通过；原始采集、Plan、失败记录未修改，也未重采挑选样本。图表及 candidate 最终场景输出已人工查看。
