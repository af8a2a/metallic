---
id: "0017"
title: "Streamer: 根页仍占 2.956 GiB / 3.5 GiB 几何池，root-reduction cook 后未执行 Full 全量重建"
area: geometry
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:geometry", "kind:task", "priority:P2"]
depends_on: []
evidence:
  - Documentation/ZorahFullRootReduction.md:3
  - Documentation/ZorahFullRootReduction.md:24-31
  - Documentation/ZorahFullRootReduction.md:49
  - Documentation/ZorahFullPeakVramPlan20260928.md:24
  - Documentation/ZorahFullInitialLoading.md:114
  - Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json:29
---

## 背景 / Context

ZorahFull 的几何池按预算一次建立 3.5 GiB（`maxResidentBytes = 3758096384`），而保底根 cut（全部实例的 terminal 分支）必须完整常驻才能保证首帧覆盖。2026-09-29 的审计给出根页设备字节 **3,173,527,552 B = 2.956 GiB**，只剩约 0.544 GiB 用于细化。

## 现状 / Current state

- 新的 LOD cook 策略已实现并通过三个真实大型 primitive 的离线和 GPU 验证：只对逐位相同的属性 tuple 合并顶点，法线按归一化方向 ≤1e-3 判定接缝；探针合计根 payload 341.64 → 143.62 MiB（-57.96%）。
- 该策略已写入离线 cook 默认路径，`kGeometryCookRevision` 从 2 升至 3、resident meshlet cache 从 3 升至 4，运行时只接受当前 revision。
- **但正式 Full 缓存没有用新策略重建**：文档明确「已有 Full 正式缓存不会因此自动变小」；2026-09-29 的首帧验收仍读到 `geometryUsedBytes = 3,173,527,552`。
- 三份探针收益差异极大（Dome -91.77%、Window -58.72%、Cylinder 仅 -12.51%），且新策略让三项磁盘缓存从 653.78 MiB 增至 987.61 MiB（+51.06%），所以探针比例不能外推到整场景。

## 问题或缺口 / Problem

1. 生产根 cut 仍是旧策略产物，2.956 GiB 的根占用没有变化；细化空间、预取水位、准入成功率都被这一项挤压（见 0016）。
2. 缺少「按 primitive / material 排序的根字节与终止原因」的整场景审计：目前只有 3 个探针和一份总量数字，无法判断哪些 primitive 值得继续改 coarse representation。
3. 全量重建需要预留完整输出 + 临时 decode/checkpoint 空间（历史缓存 54.9 GB 量级），没有记录过一次完整重建的磁盘/时间预算与验收结论。

## 期望结果 / Expected outcome

- 用 revision 3 cook 重建 Full 缓存，产出可复核的新根字节数、根页数、终止原因分布，并在同一 cfg 相机下通过保真验收。
- 几何池余量足以让细节细化稳定收敛，且不依赖扩大 3.5 GiB 预算。

## 验收标准 / Acceptance criteria

- [ ] 记录重建前的 `MetallicMeshletCook.exe --inspect` 与新缓存的根页字节、根 cluster 数、terminal 分支数，与 3,173,527,552 B 直接可比。
- [ ] 重建产物首帧 `geometryUsedBytes` 显著低于 2.956 GiB，且 `clasUsedBytes` / fallback BLAS 覆盖完整（ready 步数 = 根页数 × (1 + CLAS)）。
- [ ] 保真回归通过：`METALLIC_TEST_ZORAH_FULL=1` 的 `RHIRendering.zorah_full_first_frame`（两轮 Mini→Full）、`ZorahZ4Probes.json` 的 mappedNormal / baseColor / normalTexture / final 四类像素对照、MASK 与法线/TBN 检查。
- [ ] 三相 CPU 回归（重复顶点 soup、短法线大角度接缝、UV/切线保护、缓存兼容与 checkpoint 重启）全绿。
- [ ] 固定路线漫游记录 page load failure = 0，且细化不再因预算反复拒绝到零容量。
- [ ] 不通过删除根页、跳过 readiness 或放宽 1.5 px 来达成。

## 相关文档 / References

- [ZorahFullRootReduction.md](../../ZorahFullRootReduction.md)
- [ZorahFullPeakVramPlan20260928.md](../../ZorahFullPeakVramPlan20260928.md)
- [ZorahFullInteractiveLoadPlan.md](../../ZorahFullInteractiveLoadPlan.md)
