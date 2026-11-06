# VSM cluster 候选压缩与 indirect build

实验开关为 `VirtualShadowMapSettings::compactBuild` / 两个 VSM Pass 的 `benchmarkCompactBuild`，默认 false。只有 cluster culling 开启时生效。现有 cluster/triangle dirty-page bitset 和方向光专用 build 保持各自设置。

## 工作与资源契约

`vsmCompactMain` 对原 64-triangle source group 执行同一 cluster bounds / level predicate，将存活组写成 `{firstTriangle, levelMask}`。精确 `vsmBuildMain` 通过 indirect dispatch 读取候选，不重复执行 cluster 预筛和两次 group barrier。候选仍保留原 triangle 索引及 level mask；尾组的越界 lane 保持必要的检查。

每个 source group 至多产生一个候选，分配 `ceil(triangleCount/64)` 个 uint2 槽即可覆盖最坏情况。owners scratch 在已有 allocator scratch 和 dirty-page bitset 后追加一个 uint count 和完整容量的候选列表；原 arguments buffer 在 draw 参数之后追加 dispatch uint3，每批最多 65535 个候选。

Reset 清除 count 和所有 indirect X，并令 Y/Z 为 1。压缩通过 count 原子分配 slot，再增加 `slot/65535` 对应的 indirect X；不增加单独的参数 finalize dispatch。零候选保持 X=0，上一帧参数不能沿用。CPU 只按容量登记有限个 indirect 命令，不回读候选数量。

压缩写入经 compute-write 到 compute-read / indirect-read 同步后供 build 使用；参数尾部与原 finalize 写入的 draw 前缀不重叠。资源沿用既有 pool、submission retention 和 completion 生命周期。未增加 push constants 或公开统计 ABI。

原 45 项 build diagnostics 保留语义：cluster 计数在压缩阶段产生，精确 triangle 计数在 indirect build 中产生。完整场景用同样的图像、stats32、15 项确定性 workload 和全部 build diagnostics 做等价门槛；共享 Mark 的 `markPageRequests` 依旧按原协议处理调度相关差异。

## 预声明验证预算

证据目录为 `build/vsm-compact-build/`。普通 build 主指标包含压缩、同步、indirect build 和原 finalize，不能把精确阶段的缩短单独当成整体收益。

- GPU 场景矩阵覆盖 resident/stream、方向/点/聚光、近面/屏外、缓存 cold/warm/dirty、overflow/recover、非 32 对齐页轴与不同 level 数；附带关闭 bitset、关闭 cluster culling、关闭方向光特化的控制组。
- 独立 production-kernel 探针读取候选 count、记录和 indirect 参数，覆盖零/稀疏/全部候选、跨 65535 边界以及 reset 清除旧参数。完整场景本身不额外读回 indirect 参数。
- Sponza 固定 native 1920×1080、6 clipmaps、128×128 virtual pages、2048 physical pages、SMRT 2×16、32 帧历史；第 16 帧 freeze streaming + camera cut，前 32 帧预热。
- Diagnostic A/B 各 1 个进程、64 帧，16/32/63 帧检查 PNG/线性 HDR/depth/shadow。
- Discovery 两个 ABBA 区组、8 进程，每进程 96 帧；有希望时独立 confirmation 三个区组、12 进程。不混合两阶段，不追加样本追逐显著性。
- 主指标 build 至少 3% 收益，RenderGraph 回退最多 2%；95% Student-t 区间以完整独立 ABBA 区组 log ratio 为统计单位。

普通计时关闭 validation、GPUWork 与图像回读；保留磁盘缓存、不锁频、不停止用户应用。串行实验不保证系统 GPU 独占，pass/graph 时间不等同呈现 FPS。

实验工具使用独立协议 `virtual-shadow-map-compact-build-v1`：

```powershell
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment compact-build --phase diagnostic --cases-dir build/vsm-compact-build/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-compact-build-new-diagnostic
```

## 2026-10-30 验证结果

RTX 5070 Ti / driver 621.01，Release。正确性通过，性能未达到准入条件，`compactBuild` 保持默认关闭。完成预声明的 2 个 diagnostic 进程和 2 个 discovery ABBA 区组（8 个普通进程）；discovery 判定 `inconclusive / uncertain`，按协议停止，没有运行 confirmation，也没有追加样本。

普通计时如下。A/B 数值为各进程预热后中位数的中位数；收益与区间由完整 ABBA 区组的 log ratio 计算，因此不等于直接用表中两个中位数相除。正收益表示更快。

| 测量范围 | A 直派（ms） | B 压缩 + indirect（ms） | 收益估计 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Build 全阶段 | 0.149672 | 0.159208 | -6.40% | [-17.83%, +3.92%] |
| VSM shadows | 4.106920 | 4.153240 | -1.05% | [-12.12%, +8.93%] |
| RenderGraph | 7.094400 | 7.195184 | -1.11% | [-1.15%, -1.07%] |

两个区组的 build 点估计分别慢 5.55% 和 7.26%，但只有两个独立区组，区间仍包含收益，不能宣称已经确认 build 回退。Graph 的小幅回退在 2% 容忍范围内，但这不能补足主指标的收益门槛。Mark / raster 也观察到变慢；Mark SPIR-V 未变化，不能归因为 Mark 算法修改。这里没有锁频或原生 ISA/寄存器证据，也没有分别测量压缩、同步和 indirect kernel，尚不能确定成本来源。

Sponza 预热后每帧的 source cluster 为 4098 组，逻辑存活候选为 1072 组（26.16%）。压缩仍须预筛全部 4098 组，原直派路径也已在预筛后排除其余组，因此精确三角形及页访问工作量并未减少。全部 45 项 build diagnostics 和 15 项确定性 workload 均逐帧、逐 face 相等；PNG、线性 HDR、depth、shadow 和 stats32 也完全一致。已检查第 63 帧画面与诊断/计时图表。

候选数据与 indirect 尾部理论增加 `4 + 4098*8 + 12 = 32800` 字节。Diagnostic 第 63 帧观测到 `Other` allocation bytes 和 live allocation bytes 同样增加 32800，allocation count 不变；这不是驱动总驻留显存差值的保证。

验证包括：

- `Metallic` 和 `MetallicRHITests` Release 构建通过；Slang 模块审计、diff 检查、79 项相关 Python 测试通过。
- `virtual_shadow_map_compact_build_equivalence`：72 状态（10 个状态附加开关控制），两类 Pass、多种光源、驻留/流送及缓存/溢出场景通过，无 Vulkan validation 消息。
- `virtual_shadow_map_compact_build_arguments`：10 个生产 kernel 探针通过，覆盖候选列表、level mask、计数、实际 indirect build、尾 lane、零候选和 reset。65535 边界使用人工预置 65534 项前缀，再实际追加两组，验证两个参数为 65535/1；不执行人工前缀，也不宣称完整大场景验证。
- Profiling 与方向光等价回归通过。直派预筛保留原内联实现，helper 只用于候选，默认路径的 Build、Reset、Mark、Projection SPIR-V 与此前生产版本逐字节一致。
- 普通/诊断三个实际运行程序均通过 `spirv-val`。普通 A 为 5226 条静态函数指令，B 为压缩 937 + indirect build 4383；这不代表动态工作量、原生指令或调度成本。
- 封存包及最终源码/二进制核验通过：739 个 source 文件、20 个 binary 文件无差异。

后续若继续优化，先分别测量 compact 与 indirect build 的成本及寄存器/占用情况，再针对实际瓶颈调整 append/dispatch 设计，并选择不同 cluster 存活率的真实工作负载重新声明实验。此次数据不足以支持默认启用。

本机证据（`build/` 不入库）：[性能报告](../build/vsm-compact-build/DiscoveryReport/report.md)、[诊断报告](../build/vsm-compact-build/DiagnosticReport/report.md)、[普通编译分析](../build/vsm-compact-build/CompilerEvidence/Ordinary/report.md)、[生产基线连续性](../build/vsm-compact-build/CompilerEvidence/BaselineRestored/report.md)、[最终文件身份](../build/vsm-compact-build/FinalIdentityCheck.json)。报告内附进程散点图、区组区间图和独立复现脚本。

最终 GPU 记录：[72 状态场景矩阵](../build/vsm-compact-build/RegressionSealed/VirtualShadowMapCompactBuildEquivalence/VirtualShadowMapCompactBuildEquivalence.json)、[10 项参数探针](../build/vsm-compact-build/ResetRestored/VirtualShadowMapCompactBuildArguments/VirtualShadowMapCompactBuildArguments.json)。

后续拆分 compact/indirect 时间及合并发布屏障的独立实验见 [Build 分段归因](VirtualShadowMapBuildAttribution.md)。该实验仍未确认收益，默认值保持关闭；本节历史证据未被覆盖或合并。
