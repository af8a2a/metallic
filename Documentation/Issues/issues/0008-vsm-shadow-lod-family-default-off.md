---
id: "0008"
title: "VSM: Shadow LOD 全家仍默认关闭，Refined receiver 暖缓存 +471%，运动与大场景质量未验证"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapShadowLOD.md:223
  - Documentation/VirtualShadowMapShadowLOD.md:243-256
  - Documentation/VirtualShadowMapShadowLOD.md:187
  - Documentation/VirtualShadowMapShadowLOD.md:191
  - Documentation/VirtualShadowMapShadowLOD.md:239
  - Documentation/VirtualShadowMapShadowLOD.md:264
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:66-71
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:199-205
---

## 背景 / Context

阴影专用几何 LOD 用共享 caster cut 替代完整几何，把 Sponza/Bunny 场景的 shadow triangles
从完整几何的 69,453 降到 17,348。围绕它长出了七个设置：`shadowLOD`、
`shadowLODHierarchy`、`shadowLODErrorTexels`、`shadowLODSmartBias`、
`shadowLODSmartBiasRefine`、`shadowLODScreenRayLength`，加上 Pass 侧的
`shadowLODDepthGuard`。这些设置全部默认关闭（`VirtualShadowMaps.h:66-71`），
Inspector 里成组出现（`VirtualShadowMapLightsPass.cpp:199-205`）。

## 现状 / Current state

- LOD 本身的收益是确认过的：Refined/Off 重绘 Shadows −19.61%（CI [−19.93, −19.30]）、
  RenderGraph −15.07%，但相对旧 Smart 的**重绘** Shadows 是 +130.43%
  （`VirtualShadowMapShadowLOD.md:245-256`）。
- 暖缓存是决定性回退：Refined 每帧多出约 0.522 ms Projection，使暖缓存
  Shadows +471.67%（CI [464.39, 479.03]）、RenderGraph +85.89%
  （`VirtualShadowMapShadowLOD.md:256`）。
- 结论已明确：「当前结果支持保留部分重绘收益，**不支持暖缓存性能收益或默认启用
  SmartBias**」（`:223`），文档同时把下一步写成「减少这项修正查询覆盖的像素和遍历成本，
  同时保留固定漏影集合与独立 oracle 门槛」（`:264`）。
- 正确性仍有已知缺口：Legacy 路径保留 669 处真实漏影（`:239`），
  Smart 初版修复 1,359 个 acne 同时新增 63 个漏影（`:213`），Refined 相对 Legacy
  「没有新增真实漏影」但没有消除 Legacy 已有错误。
- 验证范围明确未覆盖：大型场景长时间驻留压力、连续运动的 LOD 跳变、软阴影质量
  （`:187`）；多灯共用「所有投影中要求最细的 cut」，没有逐灯或逐页独立几何选择（`:191`）。
- 修正只作用于**有效 Hard 模式的方向光**，PCF、SMRT、spot、point 不应用该修正（`:199`）。

## 问题或缺口 / Problem

1. 七个设置构成一个实际只有「Hard + 方向光 + 重绘」可用的组合，其余组合要么无效要么
   回退，但设置没有任何运行时约束或 UI 提示，容易被当成通用开关打开。
2. Refined 的代价集中在 Projection（+0.522 ms/帧），暖缓存场景下把 LOD 的几何收益完全
   吃掉；没有暖缓存可接受的中间形态。
3. 质量侧只有固定 Bunny fixture 与三条视角的 authored-triangle oracle，连续运动引起的
   cut 切换、远距离掠射、软阴影（PCF/SMRT 不修正）都没有门槛。
4. 大型场景的 residency 与 cut 稳定性没有数据，而 LOD cut 与 stream 预算耦合
   （预算不足会持续用更粗回退，`:191`）。

## 期望结果 / Expected outcome

二选一收口：要么把这项能力收敛成「Hard 方向光 + 重绘」的显式子功能并补齐运动/大场景
门槛；要么把它降级为历史实验（撤除 SmartBias/Refine 的运行时开关，保留归档包）。
无论哪种，都要把「暖缓存不可用」写成硬约束，而不是留在文档中段。

## 验收标准 / Acceptance criteria

- [ ] 决策写入 `VirtualShadowMapShadowLOD.md:223` 与 `VirtualShadowMaps.h:66-71` 注释，
      并在 `VirtualShadowMapProductionOptimizations.md` 的排除/已否决表中登记。
- [ ] 若保留：把 Refined 的暖缓存 Shadows 区间压到 ≤ +30%（≥3 个 ABCDDCBA 区组，
      95% CI 以区组为样本），并给出 Projection 子区间拆分，说明 0.522 ms 花在哪。
- [ ] 若保留：在 Pass 侧对无效组合给出显式诊断（`shadowLOD=true` 且
      `filter != Hard` 或含局部光时打印一条日志/Inspector 警告），而不是静默不生效。
- [ ] 连续运动 + resize 的 LOD 跳变诊断：≥64 帧相机运动，记录每帧 cut 变化数
      （`cacheDiagnostics` 字段 29，见 `VirtualShadowMapCacheLayers.md:15`），
      无单帧闪斑或历史撕裂。
- [ ] 一个 ≥5×10^5 三角的大型场景长时驻留记录：cut 稳定性、显存峰值、
      stream 预算不足时的退化为粗 cut 的比例。
- [ ] 固定漏影集合与 authored-triangle oracle 门槛继续通过（不降低 2e-5 阈值、
      不使用 erosion 排除边界），并明确 PCF/SMRT/局部光是否需要 receiver 修正的结论。

## 相关文档 / References

- [VirtualShadowMapShadowLOD.md](../../VirtualShadowMapShadowLOD.md)
- [VirtualShadowMapCacheLayers.md](../../VirtualShadowMapCacheLayers.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
