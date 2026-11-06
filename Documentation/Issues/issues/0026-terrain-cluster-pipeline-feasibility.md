---
id: "0026"
title: "地形: GPU-Driven cluster 化 M0–M2 未开工，且可行性文档的技术前提已被统一流式 VBuffer 取代"
area: geometry
kind: research
priority: P3
state: proposed
milestone: M5-Research
labels: ["area:geometry", "kind:research", "priority:P3"]
depends_on: []
evidence:
  - Documentation/TerrainMeshShaderFeasibility.md:11-27
  - Documentation/TerrainMeshShaderFeasibility.md:15
  - Documentation/UnrealLandscapeResearch.md
  - Documentation/Streamer.md:15
  - Documentation/StreamingMeshletLod.md:30
---

## 背景 / Context

地形方案的前提是「不建独立 tile 渲染路径」，而是把地形以 cluster 为单位并入现有 GPU-Driven 管线，与普通场景共用剔除、光栅、延迟着色、流式和 RT。可行性文档给出 M0–M4 五级里程碑，并认为 M1/M2 就能满足「地形与普通场景整合剔除 + 渲染」。

## 现状 / Current state

- M0（把 DAG 遍历 + 页流式移植进 Deferred 路径，补 HZB 遮挡与 per-cluster 材质）、M1（离线把地形烘焙为 `.meshstream.bin`，builder 支持高度场 + 裙边）、M2（cluster GPU 程序化生成）都没有开工，仓库里没有对应的 cooker 入口、测试或配置。
- 该文档的现状判断是按更早的代码写的：它认为 `GPUDrivenStreamAsset` 光栅侧「只输出调试颜色、无 visibility buffer/属性/材质/HZB」，而已经存在一条把 DAG 遍历、有序 frontier、页流式、属性解码与 deferred 着色贯通的生产路径（`reset → sparse clear → distributed demand → ordered frontier → mask → prefix → sparse emit → indirect finalize`），主视角可见性记录直接供 Deferred 消费。
- 因此 M0 的工作量估计与「两条管线交集」的论证都需要按当前代码重新基线化，不能用原文档的结论直接排期。

## 问题或缺口 / Problem

1. 缺一个当前的统一管线能力清单：流式几何是否已经具备地形所需的全部输入（高度场来源、裙边、per-cluster 材质、跨 cluster 接缝一致性、LOD 与高度位移的耦合）。
2. 地形与现有 meshlet LOD 的误差度量、页格式和 cook 契约没有对接方案；M1 的「高度场 → meshstream」语义（误差、终止条件、裙边如何参与简化）尚未定义。
3. 缺少与 Unreal Landscape 的可比边界说明：哪些能力属于渲染管线（可移植），哪些属于编辑器工具链（不在本次范围）。

## 期望结果 / Expected outcome

- 一份按当前 HEAD 重新基线化的地形可行性结论：明确统一管线还缺什么、M0 是否仍然必要、M1/M2 的最小可验证切片是什么。
- 若判定暂不做，也把结论与前提写回文档，避免下一轮重复评估。

## 验收标准 / Acceptance criteria

- [ ] 重新核对并列表当前流式管线对地形的支撑项与缺口（逐条给 `file:line` 证据），并指出原文档中已被取代的判断。
- [ ] 给出最小探针方案：一个高度场切片 → `.meshstream.bin` → 与普通场景同图渲染，并定义接缝、裙边、误差度量与终止条件的验收。
- [ ] 若排期 M1：产出可复核的 cook 报告（页数、根集合、误差分布）与首帧图像对照（无洞、无 LOD 跳变、无 TBN/法线异常）。
- [ ] 若判定不做：在 `TerrainMeshShaderFeasibility.md` 与 `UnrealLandscapeResearch.md` 顶部写入更新后的结论与日期。
- [ ] 全过程不改动现有 Full/Mini 场景的 cook 格式、预算或默认管线。

## 相关文档 / References

- [TerrainMeshShaderFeasibility.md](../../TerrainMeshShaderFeasibility.md)
- [UnrealLandscapeResearch.md](../../UnrealLandscapeResearch.md)
- [StreamingMeshletLod.md](../../StreamingMeshletLod.md)
- [Streamer.md](../../Streamer.md)
