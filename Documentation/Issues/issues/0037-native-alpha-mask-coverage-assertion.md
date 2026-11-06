---
id: "0037"
title: "RHI: native alpha-mask 覆盖断言持续失败，前表面覆盖数远超上限且原因未证实"
area: rhi
kind: defect
priority: P2
state: open
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P2"]
evidence:
  - Documentation/NativeDescriptorHeapStrideWorkaround.md:71-80
  - Documentation/NativeDescriptorHeapStrideWorkaround.md:196-228
  - tests/rhi/RenderGraphTests.cpp:9919-9930
---

## 背景 / Context

`render_graph_gpu_driven_alpha_mask_render` 用 MASK/BLEND 材质的前后表面测试 GPU-driven
alpha 分类：前表面 coverage 像素必须落在 `1024..7500`，unexpected-color 必须 `<= 64`
（`RenderGraphTests.cpp:9919-9930`）。这个用例在 native 与 mapped 两种描述符模式下都失败，
是当前 alpha 语义在流式光栅路径上唯一持续的像素级红灯。

## 现状 / Current state

- 2026-10-07（driver 616.92 / 617.42）：coverage 12100，dark 4284，unexpected-color 0。
  保留的 2026-10-07 16:07:32 旧 RHI 可执行文件在同 Slang DLL / GPU / VVL 下复现完全相同的计数。
- 2026-10-17 统一 SDK 路径重测：coverage 11668，dark 4716，unexpected-color 0，仍无 VUID。
  用例的 scene helper 与断言自早期对照以来未改动，但共享渲染行为已变化。
- 一个静态候选解释：该场景后表面是无贴图、alpha=1 的 BLEND 面，现在被分类为 opaque，
  从而填补了前表面 MASK 的孔洞——计数与该几何吻合。**没有任何 winner / material-ID 抓取或
  隔离对照证实该解释。**
- 因此这些失败既不能证明「native 独有回归」，也不能证明「MASK 普遍失效」；独立的 material
  coverage 检查（30,752 个 MASK 像素的 raster / ray / shadow 三路结果）在 native 与 mapped
  及其 Slab 变体下全部通过。

## 问题或缺口 / Problem

断言阈值（7500）与真实几何预期之间的对应关系没有被验证，失败信息只给三个计数，无法区分
「分类错误」与「阈值假设过期」。这使 native 迁移的渲染回归结论长期带着一个未解释的红项，
也让任何 alpha/coverage 相关的后续改动难以判断是否引入了新问题。

## 期望结果 / Expected outcome

先用 material-ID / winner 抓取证明后表面被分类为 opaque（或否证），再决定是修分类还是按真实
几何修订断言上限，并把判据写进测试注释与文档；两种模式下都要有可解释的通过标准。

## 验收标准 / Acceptance criteria

- [ ] 提供 winner 或 material-ID 抓取证据，明确 11668 / 12100 coverage 里后表面 BLEND 面贡献的
      像素数，并据此判定分类是否正确。
- [ ] `MetallicRHITests --gtest_filter=*render_graph_gpu_driven_alpha_mask_render*` 在 mapped 与 native
      两种模式下都通过；若修的是阈值，需在测试注释里记录几何依据与新的上限来源。
- [ ] `--gtest_filter=*material_coverage*` 的 30,752 MASK 像素 raster/ray/shadow 三路检查保持通过，
      且四个 BLEND 用例的 ray probe 无回归。
- [ ] 报告写明所用驱动、VVL 版本，以及这些像素矩阵是否在关闭 VVL 下采集。

## 相关文档 / References

- [NativeDescriptorHeapStrideWorkaround.md](../../NativeDescriptorHeapStrideWorkaround.md)
- [NativeResidentImageInvestigation.md](../../NativeResidentImageInvestigation.md)
- [OpacityMicromaps.md](../../OpacityMicromaps.md)
