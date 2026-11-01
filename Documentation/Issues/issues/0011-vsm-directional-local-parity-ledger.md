---
id: "0011"
title: "VSM: 已固化的特化全部只覆盖方向光，局部光/混灯没有对等台账，两项对等候选停在「保留后续」"
area: shadow
kind: research
priority: P3
state: open
milestone: M5-Research
labels: ["area:shadow", "kind:research", "priority:P3"]
depends_on: []
evidence:
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:587
  - Documentation/VirtualShadowMapProductionOptimizations.md:15
  - Documentation/VirtualShadowMapDirectionalProjection.md:9
  - Documentation/VirtualShadowMapPropagatedMapping.md:111-124
  - Documentation/VirtualShadowMapMarkPageSize.md:17-22
  - Documentation/VirtualShadowMapsScreenProjection.md:90
---

## 背景 / Context

VSM 近年固化的优化几乎都带一个隐含前提「方向光」：Mark 页地址移位只在
`directional` 时编译（`VirtualShadowMaps.cpp:587`，局部光 `markPageShift = -1`），
方向光 Mark clipmap 数专用编译、方向光 Mark/Build 专用编译、方向光 Projection 正交特化
同样如此（`VirtualShadowMapProductionOptimizations.md:9-15`）。文档一直写「局部灯继续使用
通用页地址计算」「局部光不启用该宏」（`VirtualShadowMapDirectionalProjection.md:9`）。

## 现状 / Current state

- 已固化路径的方向光/局部光差异在文档里是分散的一行注释，没有一处集中说明「局部光当前
  用哪条路径、缺了哪些特化、为什么」。
- 投影映射候选也只对局部光提供 mode 3（dyadic mip 祖先），方向光的 mode 2（整数父页）
  不适用（`VirtualShadowMapPropagatedMapping.md:111-124`），两条路径的正确性论证彼此
  独立（模运算 vs 相对坐标刻度）。
- 混灯场景走 `vsmProjectOnePassMain`，不传 `VSM_DIRECTIONAL_PROJECTION`
  （`VirtualShadowMapDirectionalProjection.md:9`），因此多灯图拿不到已确认的 Projection
  收益，这也解释了两条路径的收益无法合并比较。
- 两项已经被识别、但明确「保留后续」的对等候选躺在对照表里没有 owner：
  逐采样 UVZ 起点/步长递进、通过页表 `LODOffset` 做整数 clipmap fallback
  （`VirtualShadowMapMarkPageSize.md:21-22`）。
- 另一条已知未合并项：点光六面的 mark 轨迹生成尚未合并，也未引入 tile light list
  （`VirtualShadowMapsScreenProjection.md:90`）。

## 问题或缺口 / Problem

1. 收益全部在方向光上取得，但没有任何一处文档能回答「为什么局部光不做对等特化」，
   读者无法判断这是刻意边界还是一次次被推迟的顺手省略。
2. 两条路径的正确性论证分家（Mark 用整数移位 vs 通用除法；投影跨页取父页 vs 通用层级
   搜索），一旦出现方向光/局部光不一致的报告，没有共同的等价矩阵可以定位。
3. 两项对等候选没有验收门槛也没有归属，容易在后续重构中重复讨论。
4. 「点光六面共享 mark 轨迹」这类结构级候选更难排优先级，因为缺少局部光各阶段的成本
   占比数据。

## 期望结果 / Expected outcome

建立一张方向光/局部光能力矩阵，并用一次成本测量决定是否值得做对等特化；两项 deferred
候选各自给出「采纳前置条件」或「不做」的结论。

## 验收标准 / Acceptance criteria

- [ ] 在 `VirtualShadowMapProductionOptimizations.md` 增加方向光/局部光能力矩阵，逐项标注
      Mark 页地址移位、clipmap 特化、ray-page Mark、Projection 正交特化、映射 mode 2/3、
      ARC 的适用范围（已固化 / 仅方向光 / 未实现），每行给出对应源码位置。
- [ ] 同一固定场景把 Sun 换成 spot（其余设置不变）后，报告 Mark / Build / Projection
      各阶段占比，作为局部光对等特化的收益上界估计。
- [ ] 对 `VirtualShadowMapMarkPageSize.md:21-22` 的两项候选分别给出结论：采纳需列出
      前置条件（页表/失效生命周期变更范围）与验收门槛，或不做的理由。
- [ ] 说明混灯 one-pass 是否可能获得 Projection 正交特化（需要 `metric.w` 逐 job 判定），
      给出「不可能/可行但需 X」的结论，并同步 `VirtualShadowMapDirectionalProjection.md:9`
      的边界描述。
- [ ] 不新增默认开启路径；所有结论写入文档，不为本条目引入新的运行时开关。

## 相关文档 / References

- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [VirtualShadowMapMarkPageSize.md](../../VirtualShadowMapMarkPageSize.md)
- [VirtualShadowMapDirectionalProjection.md](../../VirtualShadowMapDirectionalProjection.md)
- [VirtualShadowMapPropagatedMapping.md](../../VirtualShadowMapPropagatedMapping.md)
- [VirtualShadowMapsScreenProjection.md](../../VirtualShadowMapsScreenProjection.md)
