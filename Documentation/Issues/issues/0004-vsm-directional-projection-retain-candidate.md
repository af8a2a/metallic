---
id: "0004"
title: "VSM: directionalProjection 停在 retain_candidate，RenderGraph 下界 −2.19% 未过保护门槛，默认值待决策"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapDirectionalProjection.md:3
  - Documentation/VirtualShadowMapDirectionalProjection.md:34-44
  - Documentation/VirtualShadowMapDirectionalProjection.md:9
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:62
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:391-392
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp:42-43
  - Documentation/VirtualShadowMapProductionOptimizations.md:15-17
---

## 背景 / Context

`directionalProjection` 只在 per-face `vsmProjectMain` 编译 `VSM_DIRECTIONAL_PROJECTION=1`，
裁掉四处透视计算：`vsmProjectPosition` 的透视除法、`vsmComparisonDepth` 的透视
receiver-plane 深度、`vsmRayDepth` 的径向深度换算、Hard Projection 的透视深度比较。
正交表达式、采样位置与顺序、页选择、资源访问均不变。

## 现状 / Current state

- 状态行写「已实现，`directionalProjection` 默认关闭；GPU 等价与 Sponza 图像检查通过；
  独立确认 Projection 收益 13.80%、Shadows 收益 4.33%」，但「RenderGraph 收益区间下界为
  −2.19%，未达到预定 −2% 保护门槛，判定为 `inconclusive / uncertain /
  retain_candidate`」（`VirtualShadowMapDirectionalProjection.md:3`）。
- 确认数据（三个独立 ABBA 区组 / 12 进程，`VirtualShadowMapDirectionalProjection.md:34-44`）：
  Projection 1.0163 → 0.8767 ms（13.80%，CI [12.56, 15.03]）、
  Shadows 3.4547 → 3.2907 ms（4.33%，CI [3.35, 5.31]）、
  RenderGraph 7.0871 → 6.9370 ms（1.76%，CI [−2.19, 5.56]）、
  Build −18.81%（CI [−393.79, 71.41]，区间宽到不可用）。
- 开关已在两个 Pass 上作为生产属性存在（`VirtualShadowMapLightsPass.cpp:391-392`、
  `VirtualShadowMapPass.cpp:42-43`），设置默认 `false`（`VirtualShadowMaps.h:62`），
  并进入 `Programs` 缓存键，可以逐帧切换。
- 收益只发生在 per-face 方向光程序上：混合多灯 `vsmProjectOnePassMain` 不传该宏，继续走
  通用程序（doc:9）。
- 生产固化文档把 one-pass Projection、方向光 Projection 等「实际功能或资源策略」明确
  排除在本次固化之外（`VirtualShadowMapProductionOptimizations.md:15-17`）。

## 问题或缺口 / Problem

1. 候选停在 `retain_candidate`：Projection 阶段收益有 CI 支撑，但整图保护门槛没通过，
   默认值因此悬空，没有人推进最后一次确认。
2. 唯一的整图区间下界是负的，而阶段收益 +4.33%（Shadows）说明增益集中在
   Projection，可能被其它未改程序或同步边界的漂移吃掉；文档自己写明「Build 离散较大，
   不能从中确认回退」（doc:44）。
3. 如果按当前证据默认开启，等于用一个下界为 −2.19% 的区间替换生产默认；如果直接放弃，
   则浪费已经确认的 13.80% Projection 收益。两种都需要一次决策，而不是继续留在
   「已实现但默认关闭」。
4. 混灯 one-pass 场景完全拿不到这项收益，但文档没有说明这是永久边界还是待补：
   one-pass 程序里同样存在 `metric.w` 分支。

## 期望结果 / Expected outcome

用一次预声明预算的独立确认把 RenderGraph 下界做到 ≥ −2%，然后采纳或归档；
若采纳，明确 one-pass 多灯是否另行实现（或在文档写死「仅 per-face」）。

## 验收标准 / Acceptance criteria

- [ ] 至少 3 个独立 confirmation ABBA 区组（12 进程），在 Sponza 与 Mixed 两个固定场景上
      复测，报告 Projection/Shadows/RenderGraph 的区组 log ratio 与 95% CI；
      RenderGraph 下界 ≥ −2%，且 Build 区间需收窄到可解释范围或明确标为不可用。
- [ ] 决策后更新 `VirtualShadowMapDirectionalProjection.md:3` 状态行与
      `VirtualShadowMapProductionOptimizations.md` 的固化/排除表。
- [ ] 若采纳：`VirtualShadowMapSceneTests.cpp:3361-3363` 的 `productionDefault`
      断言随默认值更新，等价矩阵保留
      `Baseline/Candidate/Default/ProductionOff/ProductionOn/Toggle` 六臂并全部通过。
- [ ] 若采纳：单独给出 one-pass 多灯（`smrt-onepass` 状态）是否同样启用该宏的结论；
      启用需图像逐字节一致 + 阶段区间，不启用需在 doc:9 写明「仅 per-face，永久边界」。
- [ ] 不锁频、不声明系统级 GPU 独占，报告范围写成 pass/RenderGraph，
      不写成呈现 FPS（沿用 doc:44 的口径）。

## 相关文档 / References

- [VirtualShadowMapDirectionalProjection.md](../../VirtualShadowMapDirectionalProjection.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [VirtualShadowMapOnePassProjection.md](../../VirtualShadowMapOnePassProjection.md)
- [Directional projection tool protocol](../../../Tools/Perf/VirtualShadowMapDirectionalProjection.md)
