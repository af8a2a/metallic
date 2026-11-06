---
id: "0019"
title: "SW raster: 仍无 scanline / 小 bbox 快路径，96.61% 三角形在 setup 阶段被汇总拒绝"
area: geometry
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:geometry", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/ZorahFullSoftwareRasterNaniteGap20260929.md:28-31
  - Documentation/ZorahFullSoftwareRasterNaniteGap20260929.md:45-58
  - Documentation/ZorahFullSoftwareRasterNaniteGap20260929.md:66-70
  - Shaders/Modules/GPUDriven/HybridRasterTriangle.slang:104-126
  - Shaders/Features/GPUDriven/GPUDrivenStreamGroupRaster.slang:42
  - Documentation/ZorahFullPerformanceBaselineAfterNsightFix20260930.md:26-27
---

## 背景 / Context

流式软件光栅是可见性缓冲的主成本之一。2026-09-29 已把默认入口提升为 `streamClusterRasterGroup32Main`（wave32 / 32 线程），并通过 313 个边界 fixture 与四入口逐字节输出对照；2026-09-30 的 1440p 生产基线中 SW early 2.079 ms、SW late 0.134 ms，均为默认 group32。

## 现状 / Current state

- 已补齐：首 wave 协作装载、共享唯一顶点投影与 snap、整数边函数按行/列增量步进、cull 内 coverage/tessellation 强制 HW + 球体保守 fast-SW。
- 仍未补齐（同一文档明确列出）：
  - **bbox 仍是整个包围矩形的双循环**，没有 Nanite 式小 bbox 矩形路径 + 宽 bbox scanline 的自适应选择；代码里只有 `lower/upper` 矩形扫描，没有逐行有效 X 区间。
  - 微三角形 setup 没有单样本快路径；历史样本中平均每 cluster 提交 40.86 个三角形、占 128 槽的 31.92%，而背面/退化/空 bbox 等汇总拒绝占 **96.61%**（1,441,086 个三角形进入 bbox）。
  - cluster 容量仍是 128 顶点 / 128 三角形（Nanite 为 256 / 128），完整属性拆点可能先撞顶点上限，导致三角形填充不足；这一点只有历史证据，缺当前视角的 cook 直方图。
- 已知负面结果：局部分桶在相同顶点复用条件下比不分桶慢 12.27%（8.215 ms vs 7.317 ms），不应重开。

## 问题或缺口 / Problem

SWR2/SWR3/SWR4 三个候选（微三角形 setup 缩减、矩形/scanline 双路径、cluster 密度）都还停在「建议先测」阶段，且现有计数不能回答 2.08 ms 的归因：`triangleId`/`vertexId` 分布、各 setup 拒绝原因、非空但零覆盖比例、bbox 宽高分布都缺。

## 期望结果 / Expected outcome

- 用当前视角的计数证明成本在装载/顶点变换/setup 还是像素写入，再按证据实现单一候选，而不是把 scanline、深度算术、分桶一起改。
- 保留原始 recordIndex/triangleIndex 与深度/可见性输出逐字节一致。

## 验收标准 / Acceptance criteria

- [ ] 刷新当前 Full 视角基线：保存截图相机、内部渲染尺寸、DLSS/LOD 设置，固定 cut/驻留/jitter/HZB priming，导出实际 module/entry 与 SPIR-V 哈希，并补 triangleCount/vertexCount 分布与 bbox 宽高分布。
- [ ] 每个候选单独 A/B，至少三轮换序，报告 early/late SW、raster 合计与 Graph，注明对照组。
- [ ] 覆盖 0/1/31/32/33/63/64/65/127/128 元素边界；组大小改变时不得按 lane 重新编号。
- [ ] 原始深度与可见性输出逐字节一致（沿用 `stream_group_raster_boundaries` 与 `ZorahFull` 四入口对照口径），不因数 ULP 深度差放宽容差。
- [ ] 若确认 128 顶点上限导致三角形填充不足，先出 cook 直方图证据，再评估 256 顶点方案，且不得放宽 1.5 px 或牺牲法线/UV 保真。

## 相关文档 / References

- [ZorahFullSoftwareRasterNaniteGap20260929.md](../../ZorahFullSoftwareRasterNaniteGap20260929.md)
- [ZorahFullSwGroup32Default20260929.md](../../ZorahFullSwGroup32Default20260929.md)
- [ZorahFullLocalWorkBins.md](../../ZorahFullLocalWorkBins.md)
