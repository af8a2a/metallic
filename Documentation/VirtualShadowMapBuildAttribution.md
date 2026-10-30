# VSM compact / indirect build 分段计时

`benchmarkBuildSubstages=true` 配合 `benchmarkGPUProfile=true` 在现有 `VSM build` 父区间内增加子区间。默认关闭；不改变 stats32、work16 或 build diagnostics v3 的 ABI。父区间仍覆盖完整 build，分析时不能把父子区间重复相加。

| 子区间 | 范围 |
| --- | --- |
| `VSM build direct` | 原直派 build，包括多批之间的依赖 |
| `VSM build compact` | 候选压缩，包括 source 多批之间的依赖 |
| `VSM build compact sync` | compact 输出向 compute 与 indirect 参数消费者发布 |
| `VSM build indirect` | indirect build，包括多批之间的依赖 |
| `VSM build finalize sync` | build 输出到 finalize 的依赖 |
| `VSM build finalize` | finalize dispatch 及其后供 raster 使用的发布屏障 |

这些时间戳使用现有 RenderGraph 的 BottomOfPipe 机制，测量命令/依赖区间，不是隔离 kernel 或 barrier 的固有延迟。GPU 等待可能随边界移动。新增 `buildSubstageDiagnostics` 元数据明确测量性质；普通 compact A/B 工具拒绝开启这个开关的计时输入。

`VirtualShadowMapBuildAttribution.py` 验证独立的 attribution 协议：固定两个 ABBA 区组、8 个进程，每个进程 96 帧、前 32 帧预热，16 帧 freeze streaming 与 camera cut。关闭 validation、工作量原子计数和图像回读，使用普通 shader。报告检查封存包、实际 SPIR-V、源码/二进制、输入和每个 face 的父子范围；不存在的分支为 unavailable。

```powershell
python -B Tools/Perf/VirtualShadowMapBuildAttribution.py ATTRIBUTION_ROOT NEW_REPORT
```

## 优化前归因

2026-10-30，Sponza native 1920×1080、RTX 5070 Ti / driver 621.01。下表是每个 arm 四个独立进程中位数的中位数；分段中位数不能严格相加。A 为生产直派，B 为上一版 compact + indirect。

| 区间（μs） | A 直派 | B compact |
| --- | ---: | ---: |
| 完整 build | 149.056 | 160.552 |
| direct | 23.656 | — |
| compact | — | 15.920 |
| compact sync | — | 50.960 |
| indirect | — | 12.808 |
| finalize sync | 0.320 | 0.304 |
| finalize + 发布屏障 | 123.664 | 79.216 |

同步区间在四个 B 进程中为 49.584–51.680 μs，但不能把这 51 μs 都解释为额外开销。Finalize 区间同时缩短约 44 μs，诊断 build 的净差约 11.5 μs，表明部分等待在阶段之间重新分布。普通计时仍负责最终性能判定。

独立 native compiler resource 诊断将实际日志 SHA256 对应的原始 SPIR-V，以其 FNV64 内容指纹连接到输入/设备 shader 和 pipeline statistics，未仅依赖 cache key：

| 程序 | Register Count | Shared Memory Size（B） | Binary Size（B） |
| --- | ---: | ---: | ---: |
| A fused build | 80 | 2336 | 111872 |
| B compact | 28 | 32 | 12544 |
| B indirect build | 72 | 3328 | 103424 |

这些是编译资源，不证明运行时 occupancy 或 stall。驱动返回的 Local Memory Size 为异常大的原始值，报告保留原值，未据此判断 spill。

## 本轮候选与依赖保证

本轮只合并 compact 发布的两次同步。原来连续调用两次 `synchronize`，来源都是 `ComputeShader / ShaderWrite`，目标分别为 compute 读写和 indirect 参数读取。当前 Vulkan RHI 将普通 buffer barrier 降为全局 `VkMemoryBarrier2`；两个调用之间没有生产者或消费者命令，因此可把目标 stage/access 求并集，合为一次全局依赖。

合并后的目标为 `ComputeShader | DrawIndirect` 与 `ShaderRead | ShaderWrite | IndirectRead`。候选记录、统计及参数的可见性均保留；暖缓存 X=0 的参数写入也仍受覆盖。跨 source 批次、跨 indirect 批次和 finalize/raster 的依赖保持原样。Shader、线程组大小、原子 append 算法和资源布局均相同。

32-thread compact 和单计数器 append 是代码检查发现的后续候选，本轮没有将它们混入实验。预算在采样前记录于 `build/vsm-compact-build-v2/Budget.json`：优化前后各两个归因 ABBA 区组，普通 diagnostic 2 进程、discovery 2 区组，仅当 promising 时独立 confirmation 3 区组。

## 候选结果

合并后的独立归因批次如下。前后批次源码和二进制不同，分别验证、分别汇总，不混合为同一组统计证据。这些描述性中位数不能证明合并操作的因果收益。

| B 子区间（μs） | 两次发布同步 | 合并发布同步 |
| --- | ---: | ---: |
| compact | 15.920 | 16.288 |
| compact sync | 50.960 | 49.360 |
| indirect | 12.808 | 11.880 |
| finalize sync | 0.304 | 0.320 |
| finalize + 发布屏障 | 79.216 | 79.416 |
| 完整 build | 160.552 | 160.800 |

同步区间略低，但完整 build 没有相应下降。当前数据不支持把两次 API 同步调用视作主要可消除成本；必要的 producer 完成与 indirect 消费依赖仍然存在。三个实际 shader 的 SPIR-V 以及 native Register/Shared/Binary 统计前后完全相同。

关闭子区间时间戳的普通 A/B，以生产直派为 A、合并发布后的 compact 为 B，完成 2 个独立 ABBA 区组（8 进程）：

| 范围 | A（ms） | B（ms） | 收益估计 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Build 全阶段 | 0.150840 | 0.158944 | -5.58% | [-16.95%, +4.68%] |
| VSM shadows | 4.274320 | 4.192368 | +1.52% | [-17.10%, +17.17%] |
| RenderGraph | 7.596912 | 7.294640 | +3.49% | [-18.97%, +21.70%] |

A/B 是进程预热后中位数的中位数；收益与区间按完整 ABBA 区组 log ratio 计算，不是直接用表中 A/B 相除。Build 两个区组的收益为 -6.44% 和 -4.74%。判定为 `valid / uncertain / retain_candidate`，没有确认收益，也没有确认 build 回退。Graph 的正点估计同样不确定，不能称为整帧加速。按预声明规则未进入 confirmation，未追加样本。

保留合并后的实验实现，`compactBuild=false` 与 `profileBuildSubstages=false` 默认值不变。下一轮若继续定位，应先拆开 finalize kernel 与后置发布范围，再检验逐页 ready-page 计数原子操作是否为热点；当前 79–124 μs 范围不能全部归给 finalize 算法。

## 验证与证据

- Release `Metallic` / `MetallicRHITests` 构建通过；93 项 VSM Python 测试通过。初次 Python 测试受沙箱 TEMP 清理权限影响，改用工作区 TEMP/TMP 后通过，失败日志保留。
- 新 profiling 父子区间检查、72 状态 compact 等价矩阵、10 项 production-kernel 参数探针全部通过，无 validation 消息。
- Sponza diagnostic A/B 启用 synchronization validation；PNG、线性 HDR、depth、shadow、stats32、45 项 build diagnostics 和 15 项确定性 workload 完全一致。
- 实际检查了最终第 63 帧图像、前后分段图和普通计时散点/区间图。
- 封存证据及最终 739 个 source / 20 个 binary 文件核验一致。普通计时不锁频，不保证系统级 GPU 独占；范围是 pass/RenderGraph，不是呈现 FPS。

本机证据（`build/` 不入库）：

- [优化前分段报告](../build/vsm-compact-build-v2/AttributionBeforeReport/report.md)、[优化后分段报告](../build/vsm-compact-build-v2/AttributionAfterReport/report.md)、[前后比较图表](../build/vsm-compact-build-v2/BeforeAfterReport/report.md)。
- [普通性能 A/B](../build/vsm-compact-build-v2/DiscoveryReport/report.md)、[正确性诊断 A/B](../build/vsm-compact-build-v2/DiagnosticReport/report.md)。
- [原生资源统计](../build/vsm-compact-build-v2/ResourcesAfterReport/report.md)、[原始 shader 与资源前后身份](../build/vsm-compact-build-v2/ResourcesComparison.json)、[依赖保证](../build/vsm-compact-build-v2/AnalysisTools/MergedDependencyProof.md)。
- [GPU 场景矩阵](../build/vsm-compact-build-v2/RegressionCandidate/VirtualShadowMapCompactBuildEquivalence/VirtualShadowMapCompactBuildEquivalence.json)、[最终文件身份](../build/vsm-compact-build-v2/FinalIdentityCheck.json)。

分段报告附 `Reproduce.py`，比较图表附 `plot_report.py`；使用当前仓库 verifier 复核原始包，图表保留逐进程结果，不将帧作为独立样本。
