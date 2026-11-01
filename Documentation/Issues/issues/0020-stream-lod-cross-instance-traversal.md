---
id: "0020"
title: "Streamer: 安全 frontier 仍每实例一个工作组，跨实例持久遍历队列未实现（空间 demand 候选已全部撤回）"
area: geometry
kind: feature
priority: P2
state: proposed
milestone: M5-Research
labels: ["area:geometry", "kind:feature", "priority:P2"]
depends_on: []
evidence:
  - Documentation/StreamingMeshletLod.md:115
  - Documentation/StreamingMeshletLod.md:117
  - Documentation/StreamLODFrontier.md:9-19
  - Documentation/MiniZorahSpatialDemandCloseout.md:3-5
  - Documentation/MiniZorahImplementationGapAnalysis.md:55-61
---

## 背景 / Context

流式 LOD 的 GPU 路径是 reset → sparse clear / demand seed → distributed demand → ordered frontier → mask → prefix → sparse emit → indirect finalize。需求阶段已经跨实例并行（全局有界子树队列 + wave 内可变长度分配），但**安全 frontier 仍按实例调度**。

## 现状 / Current state

- `streamCooperativeLod32Main`（固定 subgroup 32 设备）与 64 线程回退入口都保持「每个实例一个工作组」；Wave32 用 wave prefix/sum 替代跨 wave LDS scan，并保留降序稀疏顺序、跨 LOD 屏障与最终发布。
- 2026-10-25 的空间 demand 实验收尾：leaf8 / task32 曾独立确认 Demand 局部下降 6.27%，但伴随 Seed / Frontier 回退，未确认遍历总收益；独立存储 v3 与任务内 forest v4 未通过验收。**所有未达标的空间候选已撤回**，代码恢复为原 ordered Demand 路径。
- 文档明确写着两处未做：「父子可绘制状态尚未改为全局持久化队列」「进一步释放完全被替代的祖先 payload、合并请求和优先级调度仍是后续优化」。
- frontier 临时状态为每实例 32 字节 + 每 group 12 字节，按实例 full group 数预留；BVH4 按 primitive 共享、每节点 40 字节。

## 问题或缺口 / Problem

大实例的 tile/group 链由一个工作组承担，其余实例完成后无法接手，形成组执行长尾；每实例的 active/mask/sparse ID 三份 uint32 也按全部 group 预留而非实际 frontier 大小。缺的正是能跨实例分摊实际工作的调度结构，而 Nanite 与参考实现都提供了节点/cluster 队列变体作为对照。

## 期望结果 / Expected outcome

- 先量化长尾是否真实主导（每实例 tested tiles、有效 nodes、组执行长尾、frontier/emit 时间、状态内存），再决定是否引入跨实例 node/tile 队列。
- 若引入：保留完整父级依赖、容量不足时完整回退、稳定 emit 顺序，不得用无序原子追加破坏等深覆盖契约。

## 验收标准 / Acceptance criteria

- [ ] 导出并归档：每实例 tested tiles / 有效 nodes 分布、frontier 与 emit 的 GPU 时间、active 状态内存实际占用，以及 group 执行长尾直方图。
- [ ] 给出「长尾是否由少数实例主导」的判定与阈值；若否，明确关闭该方向并记录原因。
- [ ] 若实现跨实例队列：`meshlet_lod_stream_frontier32`、`meshlet_lod_stream_frontier_lds`（288 帧 × 31/32/33/4095/4096/4097 groups）、`meshlet_lod_stream_terminal_mask`（48 帧）全部通过，且 complete cut 与参考实现逐项一致。
- [ ] 相同场景/相机/质量的图像与原始 depth/visibility 逐字节一致，输出顺序稳定（与 `MiniZorahSpatialDemandCloseout.md` 的 restoration 判定同口径）。
- [ ] 报告遍历总收益，而不是 Demand 局部 scope 下降；Seed/Frontier 回退必须同时列出。

## 相关文档 / References

- [StreamingMeshletLod.md](../../StreamingMeshletLod.md)
- [StreamLODFrontier.md](../../StreamLODFrontier.md)
- [MiniZorahSpatialDemandCloseout.md](../../MiniZorahSpatialDemandCloseout.md)
- [MiniZorahImplementationGapAnalysis.md](../../MiniZorahImplementationGapAnalysis.md)
