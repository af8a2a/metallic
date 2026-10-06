# 材质 GPU 性能基线（2026-10-06）

本次完成 56 项场景 × 3 个独立进程的实际 GPU 采集。53 项输出有限且三次 HDR 逐像素一致；48 项通过 10% 波动检查，5 项性能不稳定，3 项透射 PT 因 NaN 不进入性能排名。

NVIDIA GeForce RTX 5070 Ti，驱动 616.92，Release；内部输出 512×512，每项 32 帧预热、64 帧计时，最后另取一帧 HDR。保留磁盘缓存，不锁定 GPU 时钟，关闭 validation/capture；后台程序保持运行。

Surface 使用 White Studio HDRI 和资产自带相机。Deferred 开启 Program Binning、FP16 权重，高级 IBL 64 samples；PT 每帧 4 spp、深度 12。Fiber PT 使用 Claire groom、自己的 HDRI、深度 4；Native Strands 使用 576 segments、8 layers 的小 groom。不同几何、覆盖率和估计器之间的差值不能解释为纯 BSDF 成本。

统计值为三个进程各自 64 帧中位数的中位数，单位 ms。波动为进程中位数的 (max−min)/median；Graph 和材质 pass 任一超过 10% 即标为不稳定。未删除尖峰，也不将连续帧当作独立实验。

Graph 是 graphics queue 的 timestamp 时间跨度。Deferred 的 VBuffer 有两个 compute/graphics 分支，汇合后执行 Deferred；Graph 包含等待，不叠加并发区间。材质 pass 为 Reference/Deferred/PathTrace/Fiber 节点范围。M8 使用当前 fused 生产 ray loop，未计入独立实验性 ray queue 分类。

| 场景 | Graph ms | 材质 pass ms | Graph 波动 | Pass 波动 | 状态 |
|---|---:|---:|---:|---:|---|
| M01_NeutralDielectric-uniform-Deferred | 0.5165 | 0.0726 | 1.0% | 1.0% | 稳定 |
| M01_NeutralDielectric-uniform-PT | 0.7684 | 0.7533 | 0.6% | 1.1% | 稳定 |
| M01_NeutralDielectric-textured-Deferred | 0.5005 | 0.0814 | 1.6% | 1.0% | 稳定 |
| M01_NeutralDielectric-textured-PT | 0.7670 | 0.7491 | 0.5% | 0.9% | 稳定 |
| M02_SaturatedMetal-uniform-Deferred | 0.5105 | 0.0778 | 5.9% | 3.0% | 稳定 |
| M02_SaturatedMetal-uniform-PT | 1.2490 | 1.2263 | 3.0% | 2.4% | 稳定 |
| M02_SaturatedMetal-textured-Deferred | 0.5093 | 0.0808 | 4.6% | 0.1% | 稳定 |
| M02_SaturatedMetal-textured-PT | 1.2671 | 1.2463 | 2.4% | 3.3% | 稳定 |
| M03_CoatedPaint-uniform-Deferred | 2.3676 | 1.9401 | 0.7% | 0.7% | 稳定 |
| M03_CoatedPaint-uniform-PT | 1.1127 | 1.0856 | 1.6% | 0.7% | 稳定 |
| M03_CoatedPaint-textured-Deferred | 2.5402 | 2.0786 | 6.5% | 6.2% | 稳定 |
| M03_CoatedPaint-textured-PT | 1.2143 | 1.1980 | 2.4% | 1.8% | 稳定 |
| M04_BrushedMetal-uniform-Deferred | 3.8737 | 3.4318 | 12.4% | 8.9% | 不稳定 |
| M04_BrushedMetal-uniform-PT | 1.0593 | 1.0376 | 1.2% | 1.2% | 稳定 |
| M04_BrushedMetal-textured-Deferred | 3.8760 | 3.4399 | 0.2% | 0.0% | 稳定 |
| M04_BrushedMetal-textured-PT | 1.1530 | 1.1351 | 0.7% | 0.7% | 稳定 |
| M05_Fuzz-uniform-Deferred | 2.4658 | 2.0291 | 0.3% | 0.3% | 稳定 |
| M05_Fuzz-uniform-PT | 1.0771 | 1.0584 | 1.1% | 1.4% | 稳定 |
| M05_Fuzz-textured-Deferred | 4.0056 | 3.5657 | 0.8% | 0.6% | 稳定 |
| M05_Fuzz-textured-PT | 1.1740 | 1.1510 | 0.2% | 0.3% | 稳定 |
| M06_SolidGlass-uniform-Deferred | 0.5031 | 0.0784 | 3.1% | 0.8% | 稳定 |
| M06_SolidGlass-textured-Deferred | 0.5090 | 0.0796 | 3.8% | 0.8% | 稳定 |
| M07_Emission-uniform-Deferred | 0.5101 | 0.0795 | 2.2% | 2.6% | 稳定 |
| M07_Emission-uniform-PT | 0.8120 | 0.7963 | 0.2% | 0.7% | 稳定 |
| M07_Emission-textured-Deferred | 0.5036 | 0.0806 | 4.0% | 0.3% | 稳定 |
| M07_Emission-textured-PT | 0.8284 | 0.8097 | 0.1% | 0.7% | 稳定 |
| M08_MaskedCard-uniform-Deferred | 0.4958 | 0.0914 | 4.6% | 0.5% | 稳定 |
| M08_MaskedCard-uniform-PT | 0.8389 | 0.8218 | 0.2% | 0.2% | 稳定 |
| M08_MaskedCard-textured-Deferred | 0.4899 | 0.0829 | 12.7% | 0.2% | 不稳定 |
| M08_MaskedCard-textured-PT | 0.7122 | 0.6940 | 0.6% | 0.3% | 稳定 |
| S01_Metal_Roughness-uniform-Deferred | 0.5387 | 0.1156 | 2.9% | 0.2% | 稳定 |
| S01_Metal_Roughness-uniform-PT | 2.8978 | 2.8825 | 0.4% | 0.4% | 稳定 |
| S02_IOR_Roughness-uniform-Deferred | 0.5474 | 0.1155 | 0.2% | 1.1% | 稳定 |
| S02_IOR_Roughness-uniform-PT | 3.3659 | 3.3449 | 18.2% | 18.2% | 不稳定 |
| S03_Coat-uniform-Deferred | 5.6943 | 5.2488 | 7.0% | 7.5% | 稳定 |
| S03_Coat-uniform-PT | 3.2360 | 3.2188 | 17.7% | 18.3% | 不稳定 |
| S04_Anisotropy-uniform-Deferred | 5.4865 | 5.0426 | 0.5% | 0.3% | 稳定 |
| S04_Anisotropy-uniform-PT | 2.6521 | 2.6304 | 21.7% | 22.0% | 不稳定 |
| S05_Transmission-uniform-Deferred | 0.5395 | 0.1088 | 6.1% | 1.2% | 稳定 |
| S06_Fuzz-uniform-Deferred | 6.0812 | 5.6247 | 3.1% | 3.0% | 稳定 |
| S06_Fuzz-uniform-PT | 3.2239 | 3.2089 | 9.8% | 7.6% | 稳定 |
| H01_HDREmission-uniform-Deferred | 0.5027 | 0.0793 | 0.5% | 0.3% | 稳定 |
| H01_HDREmission-uniform-PT | 0.8099 | 0.7880 | 1.3% | 1.4% | 稳定 |
| H01_HDREmission-textured-Deferred | 0.5052 | 0.0805 | 3.2% | 1.3% | 稳定 |
| H01_HDREmission-textured-PT | 0.8264 | 0.8054 | 0.6% | 0.6% | 稳定 |
| SlabSingle-Deferred | 1.8312 | 1.4068 | 0.0% | 0.3% | 稳定 |
| SlabSingle-PT | 0.5587 | 0.5398 | 0.7% | 1.2% | 稳定 |
| SlabMix-Deferred | 1.8303 | 1.4092 | 1.0% | 0.4% | 稳定 |
| SlabMix-PT | 0.5654 | 0.5487 | 0.7% | 0.7% | 稳定 |
| SlabLayer-Deferred | 1.8335 | 1.4095 | 0.7% | 0.4% | 稳定 |
| SlabLayer-PT | 0.5716 | 0.5521 | 0.8% | 0.9% | 稳定 |
| RTXCRChiang-PT | 0.8624 | 0.8416 | 0.3% | 0.6% | 稳定 |
| NativeStrands | 3.7540 | 1.6367 | 0.5% | 0.7% | 稳定 |
| M06_SolidGlass-uniform-PT | — | — | — | — | HDR NaN，剔除 |
| M06_SolidGlass-textured-PT | — | — | — | — | HDR NaN，剔除 |
| S05_Transmission-uniform-PT | — | — | — | — | HDR NaN，剔除 |

M06 uniform/textured PT 在三轮均有 3 个 NaN 分量，位置为 (x=375,y=246) 的 RGB；S05 透射扫描 PT 也在三轮各出现 3 个 NaN 分量。原始图像、准确坐标和时间数据全部保留。M06 的独立 validation 复现同样出现 NaN，没有 Vulkan validation 错误。有限值与 A/A 检查不替代物理/Painter 参考正确性验收。

当前可优先分析 Fuzz/各向异性/涂层的高级 IBL 开销。5 项不稳定数据只能作为观察值，不用于判定小幅优化收益；本轮未查明尖峰来源，也未采集硬件 stall/occupancy 计数器。

[完整图表报告](../build/material-perf-20261006-report/report.md) · [逐项 CSV](../build/material-perf-20261006-report/Summary.csv) · [原始证据清单](../build/material-perf-20261006/Manifest.json) · [无效 HDR 明细](../build/material-perf-20261006-report/InvalidHDR.json)

[复测方法与边界](../Tools/Perf/MaterialBaseline.md) · [配置生成器](../Tools/Perf/PrepareMaterialBaseline.py) · [采集/校验器](../Tools/Perf/MaterialBaseline.py) · [绘图脚本](../Tools/Perf/PlotMaterialCatalog.py)

结果包位于本机 build/，不加入源代码。输入资产以哈希标识，没有复制成可移植快照。后续对照使用原配置 `build/material-perf-20261005-config/Cases.json`，原资产、相机、队列与质量设置，另建输出目录。

采集时 HEAD：`4edf756ad982f4d335670d0bec5cc940c8e1c46d`。生产 renderer/shader 未改动，本次仅扩展测量工具与 RHI harness；采集包保留精确源码和二进制哈希。
证据 Manifest SHA256：`9c4fe4b01d2e8c772f8a7661dce5e5feb1b8325b6bf1b469011e45869f9aee62`。

三次 RHI 采集进程均 exit 0；捕获完成不等于每项质量通过。当前 verifier 已核对完整帧窗口、后端队列约定、所有文件哈希、HDR 有限值计数与 A/A。工具单元测试 9/9 通过，所有报告图表与场景预览已人工查看。

外层 runner 首次收尾校验曾因错误要求 `asyncComputeBranches == 0` 返回 1；三次 GPU 采集本身均 exit 0。核对实际 VBuffer fork/join 与 graphics timestamp 边界后，已修正后端约定检查，当前 verifier 对原始封存包复核通过；没有修改或重封存原始证据。见 [检查说明](../build/material-perf-20261006-report/ReviewNotes.md)。
