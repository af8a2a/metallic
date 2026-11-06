---
id: "0010"
title: "VSM: ARC 已按设备能力默认开启，但只有 2-ray 普通计时；8-ray 与 mapping 计数组合未测"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: ["0001"]
evidence:
  - Documentation/VirtualShadowMapAdaptiveRayCount.md:1-7
  - Documentation/VirtualShadowMapAdaptiveRayCount.md:21-31
  - Documentation/VirtualShadowMapAdaptiveRayCount.md:33-39
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:584
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:81-82
  - Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json:52-53
  - Documentation/VSMSponzaSample.md:39
---

## 背景 / Context

SMRT 自适应射线数（ARC）用子组整数归约判断「所有参与 receiver 都未命中」或「所有参与
receiver 累计 miss 为零」，在仍有剩余射线时提前退出，并按实际追踪条数归一化可见度。
`benchmarkAdaptiveRayCount` 属性与其设置开关已删除，启用条件固定为设备能力
`computeSubgroupBallotArithmetic`（`VirtualShadowMaps.cpp:584`，文档 `:1-7`）。

## 现状 / Current state

- 默认开启：`adaptiveRayCount` 直接取设备能力，没有运行时开关可关。不支持该能力的设备
  保留固定射线数（`VirtualShadowMapAdaptiveRayCount.md:1-7`）。
- 性能证据只有 2-ray：2-ray ABBA 区组的 Projection 改善约 1.9%（0.019 ms），
  且「localized nonzero image differences」；用户接受了图像差异并授权默认采用
  （`:33-39`）。文档自己写明「this is not a new statistical performance confirmation」。
- 8-ray 只验证了「substantial ray savings 与真实 umbra 退出」，**没有测普通 8-ray 性能**
  （`:35-39`）。
- 参考 workload 用的是 2-ray：Sponza 图 `smrtRayCount: 2, smrtSamplesPerRay: 16`
  （`Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json:52-53`、
  `VSMSponzaSample.md:39`），而设置默认是 8 rays × 8 samples
  （`VirtualShadowMaps.h:81-82`）。也就是说默认值路径本身没有参考 workload 的计时。
- 计数器口径是「receiver/light evaluation，不是 waves」，且 2-ray 配置下 umbra 计数器
  恒为 0（`:21-31`），所以 2-ray 与 8-ray 的计数不能直接比较。
- 工作树里新增了 4 个映射计数器（`projectionPageTableLoads`、`projectionMappingLoads`、
  `projectionMappedAncestors`、`projectionMappingSlowPaths`），与 ARC 三个计数器同属
  「每次 receiver/light 评估」的量级，但两套计数从未在同一实验里同时采集。

## 问题或缺口 / Problem

1. 一个默认开启的提前退出优化，只在「2 条射线」这一种配置上做过普通计时，而产品默认是
   8 条射线：ARC 的收益随配置变化很大（umbra 退出只在射线数 ≥2 且未命中累积为零时发生），
   8-ray 的实际加速比未知。
2. 图像差异被接受，但差异清单只覆盖 2-ray；8-ray 与 PCF/Hard 的边界没有记录。
3. ARC 与投影映射（若 0001 采纳）都改变「单次 receiver 查询实际读了什么/做了几次」，
   两套计数器叠加后无法再区分加速来源；没有联合采集口径，后续任何投影优化都会被
   ARC 的计数漂移污染。
4. 默认值与参考 workload 不一致（8 vs 2），文档没有说明哪个是「生产推荐」，读者无法
   判断该以哪一组数字为准。

## 期望结果 / Expected outcome

补齐 8-ray（设置默认）的普通计时与图像差异清单，并把 ARC 计数器与映射计数器的联合口径
写清楚；若 8-ray 无收益或图像差异超范围，就把 ARC 从「设备能力强制」改成显式设置项。

## 验收标准 / Acceptance criteria

- [ ] 8-ray（`smrtRayCount=8`，保持 `smrtSamplesPerRay` 与实际默认）至少 3 个独立 ABBA
      区组的普通计时，报告 Projection / Shadows / RenderGraph 区间与 CI。
- [ ] 8-ray 的图像差异清单：与关闭 ARC 的同一 fixture 比较 shadow/HDR/depth，
      给出像素差异数、最大差值与差异位置（沿用 2-ray 的 `build/vsm-arc-*` 报告格式）。
- [ ] 在 `VirtualShadowMapAdaptiveRayCount.md` 增加一节，说明
      `projectionAdaptive*` 与 `projectionPageTable*/Mapping*` 两类计数器的单位
      （各为 per receiver/light evaluation 还是 per lookup）与不可相加的原因。
- [ ] 一次同时开启 ARC 与投影映射的诊断采集，给出两套计数器的同帧数值，作为后续投影
      优化的基线（依赖 0001 的默认值决策）。
- [ ] 若 8-ray 收益 < 1% 或图像差异超出已接受范围：把 ARC 改为显式设置项
      （`VirtualShadowMaps.h` 增加字段 + 两个 Pass 属性），并在文档记录默认值。
- [ ] `VSMSponzaSample.md:39` 说明参考 workload 的 2-ray 选择理由，与设置默认 8-ray
      的差异在文档中显式对齐。

## 相关文档 / References

- [VirtualShadowMapAdaptiveRayCount.md](../../VirtualShadowMapAdaptiveRayCount.md)
- [VSMSponzaSample.md](../../VSMSponzaSample.md)
- [VirtualShadowMapPropagatedMapping.md](../../VirtualShadowMapPropagatedMapping.md)
- [VirtualShadowMapsGPUProfile.md](../../VirtualShadowMapsGPUProfile.md)
