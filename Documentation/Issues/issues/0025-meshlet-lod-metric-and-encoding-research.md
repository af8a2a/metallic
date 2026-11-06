---
id: "0025"
title: "Meshlet LOD: 误差度量与 Nanite/参考不可直接比较，更紧的保守投影界从未量化"
area: geometry
kind: research
priority: P3
state: proposed
milestone: M5-Research
labels: ["area:geometry", "kind:research", "priority:P3"]
depends_on: []
evidence:
  - Documentation/MiniZorahImplementationGapAnalysis.md:65-67
  - Documentation/MiniZorahImplementationGapAnalysis.md:32-45
  - Documentation/ZorahFullNextStepsAfterCompression20260929.md:5
  - Shaders/Modules/GPUDriven/MeshletLODMetric.slang:10-29
  - Documentation/StreamingMeshletLod.md:5
---

## 背景 / Context

Metallic 的流式 LOD 用 `meshletLodProjectedError()` 计算投影误差：对仿射矩阵取最大奇异值上界、扩张球半径包含 `worldBoundsPadding`，当扩张球碰到近裁面时返回 `3.402823466e+38`（强制最细）；透视分支再乘 `sqrt(1 + (radial/minZ)^2)` 的径向斜率项。RT 侧另有 `meshletLodRayProjectedError()`，按变换后误差扩张球的最近点算距离，因此纯相机旋转不改变指标。

## 现状 / Current state

- 像素预算以最终显示视口高度为基准（默认 1.5 display px，bias -4..4，`error * exp2(bias)`），CPU 用 `renderHeight/displayHeight` 换算后传给 GPU，需求/预取/最终 cut 共用同一阈值。
- 参考实现主要用 `error / max(near, distance - radius)`，固定 eye 转视角时距离项基本不变；Nanite 的 `GetProjectedEdgeScales` 也依赖视角。因此参考的 1 px 与 Metallic 的 1.5 px **不代表相同 cut 或相同质量**。
- 编码压缩这条线不在本条范围：`ZorahFullNextStepsAfterCompression20260929.md` 已明确「暂不继续位置/N/T 编码」，未合并分支（自适应位置编码、关节切线帧、BLAS 工作区内存）由 **0067** 跟踪，本条只做度量判断。
- 已有历史数字（同一模型全 LOD，Float32x4 位置 16 B vs 参考 12 B）：payload 56.340 GiB vs 44.106 GiB，位置占 payload 的 78.16%，理论上仅去掉第四个分量可减约 19.5% payload——但这是容量算术，不是显存或帧时收益，且 CLAS 输入、HW/SW 光栅、MaterialResolve 与校验代码都写了当前格式契约。

## 问题或缺口 / Problem

1. 「更紧的保守投影界」这个候选从未量化：屏幕边缘、近裁面、非均匀缩放/剪切下的误差分布没有对照，也缺少同一几何组的 Nanite/参考误差分布对比。
2. 现有关闭/切换证据不支持换公式：单纯换成更宽松的公式会减少 cluster 数，但属于降质而非优化，不能作为效率提升上报。
3. 位置量化（Nanite 式按精度量化 + 按位打包 + 安装期 GPU 转码）需要处理共享边界一致性，否则每个 cluster 独立舍入会产生裂缝。该分支层面的工作在 `0067` 里跟踪（停在未合并分支），本条只负责「度量是否值得换」的判断，不重复实现编码。

## 期望结果 / Expected outcome

- 给出「误差度量差异是否值得跟」的定量结论：在同一批几何组上对比投影界紧致度、屏幕边缘/近裁面误差与 cluster 数，判定更紧的界是否能在**不降低可见质量**的前提下减少 cut。
- 或者给出「不做」的技术结论与依据，并把它写进 LOD 文档，避免每轮优化重复提出。

## 验收标准 / Acceptance criteria

- [ ] 归档同一几何组的误差分布对照：Metallic 当前度量 vs 参考度量，按屏幕中心/边缘、近裁面距离、非均匀缩放与剪切分组统计 cluster 数与最大投影误差。
- [ ] 若提出更紧的保守界：在同一相机与 1.5 display px 下，像素对照（HDR/depth/visibility）无可见差异且 cut cluster 数下降；近裁面、jitter、非均匀缩放的覆盖测试全部保留。
- [ ] `meshlet_lod_ray_distance_and_hysteresis`、`*meshlet_lod*` 与 `stream_clas_cut_readiness` 回归全绿。
- [ ] 不在本条实现或验收位置/属性编码：相关契约迁移与边界一致性验收归 0067，本条结论只引用其状态。
- [ ] 结论不以「cluster 数减少」等同「性能提升」，必须附整帧或整图时间。

## 相关文档 / References

- [MiniZorahImplementationGapAnalysis.md](../../MiniZorahImplementationGapAnalysis.md)
- [ZorahFullNextStepsAfterCompression20260929.md](../../ZorahFullNextStepsAfterCompression20260929.md)
- [StreamingMeshletLod.md](../../StreamingMeshletLod.md)
- [MeshletLodDesign.md](../../MeshletLodDesign.md)
