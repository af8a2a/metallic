---
id: "0007"
title: "VSM: Mark 页尺寸特化已固定进生产，但同一轮确认里 Projection 范围 +35% 未归因"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: ["0001"]
evidence:
  - Documentation/VirtualShadowMapMarkPageSize.md:5
  - Documentation/VirtualShadowMapMarkPageSize.md:53-63
  - Documentation/VirtualShadowMapMarkPageSize.md:3
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:587
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:1144
  - Documentation/VirtualShadowMapProductionOptimizations.md:12
  - Documentation/VirtualShadowMapProfiling.md:16-25
---

## 背景 / Context

方向光 Mark 的页地址专用编译把 `vsmVirtualPage` 里的 `texel / pageSize` 换成
`texel >> VSM_MARK_PAGE_SHIFT`（`VirtualShadowMaps.cpp:1144` 注入宏），在等价矩阵 98 状态
通过后被固定为生产路径：现在只要方向光就无条件计算
`markPageShift = std::countr_zero(settings.pageSize)`（`VirtualShadowMaps.cpp:587`），
运行时开关与参考分支已退役（`VirtualShadowMapProductionOptimizations.md:12`）。

## 现状 / Current state

- 确认结果（12 进程 / 三个独立 ABBA 区组，`VirtualShadowMapMarkPageSize.md:53-61`）：
  Mark 1.3186 → 1.0712 ms（18.93%，CI [18.30, 19.56]）、
  Shadows 3.4293 → 3.1969 ms（7.08%）、RenderGraph 6.9019 → 6.6379 ms（5.10%）、
  Allocator 18.90%、Raster 4.46%，Build 21.83%（CI [−27.74, 52.16]，不可用）。
- 同一张表里 **Projection 1.0213 → 1.4169 ms，−35.30%（CI [−49.81, −22.19]）**。
  文档自己定性为实测回退：「不能因为 shader 未改就忽略。其范围包含同步，当前尚未确立
  native 指令层面的因果……后续应先定位 Projection 范围增长」
  （`VirtualShadowMapMarkPageSize.md:63`）。
- 那轮的两臂 Projection SPIR-V 相同，Mark/Build/raster mesh/depth fragment 的 SPIR-V
  也相同（`:41`），所以差异只能来自同步边界、调度或与该文未控制的其它变量。
- 该特化现在对方向光是**无条件生效**的生产路径（`VirtualShadowMaps.cpp:587`），
  所以这条未归因的 Projection 增长也随默认值一起进了今天的主线。
- 本轮之后默认值又变化了几次（需求预算、grid projection、ARC、以及工作树里的
  propagated mapping 新增 `VSM propagate mappings` scope 嵌在 build/finalize 内，
  `VirtualShadowMapPropagatedMapping.md:143-145`；`VSM build` 父区间语义见
  `VirtualShadowMapProfiling.md:16-25`），这些都会改变 Projection 前后相邻的同步结构。

## 问题或缺口 / Problem

1. 一个「只改 Mark 一行整数运算」的优化，带着 35% 的 Projection 阶段回退进了生产，
   且没有后续定位记录。
2. 35% 的 Projection 回退被 Mark 18.93% 的收益在 Shadows/RenderGraph 上掩盖：如果某个
   场景的 Mark 收益小（例如 Mark 采样数少、分辨率低、或非方向光占主导），净效果可能是
   负的，而当前没有数据能排除这一点。
3. 现在无法直接复测：运行时开关已退役，要重做 A/B 必须检出历史实现
   （`VirtualShadowMapProductionOptimizations.md:19` 已写明这一点），所以更需要一次
   「当前默认路径的 Projection 基线」记录，而不是继续引用旧确认表。
4. 文档 `:63` 的「后续应先定位」没有对应的 issue/任务编号，也没有验收门槛，容易永久搁置。

## 期望结果 / Expected outcome

在当前生产默认（含 demand budget、grid projection、ARC、hash coverage，以及 0001 决策后的
propagated mapping 状态）上重新测量 Projection 及其子区间，确认这条回退是否仍然存在；
存在则给出整图净收益区间，不存在则撤回旧声明并说明测量差异来源。

## 验收标准 / Acceptance criteria

- [ ] 用 `Tools/Perf/VirtualShadowMapAcceptance.py` 的 `mode=timing` 在固定 Sponza 相机上
      采集 ≥3 个独立 ABBA 区组，报告 `VSM projection` 及其
      `prepare / dispatch / publish` 三个子区间（`VirtualShadowMapProfiling.md:22`），
      而不是只报父区间。
- [ ] 明确写出本次测量的默认开关集合（demand budget/grid projection/ARC/hash coverage/
      propagated mapping 各为 on/off），并核对实际 SPIR-V SHA256（`[VSMShader]` 日志）。
- [ ] 若回退仍存在：补一组「低 Mark 收益」场景（例如降低 SMRT 采样或缩小视口），
      给出 Projection/Shadows/RenderGraph 的区间，证明净效果；否则把它登记为
      「Mark 专项加速，整图收益仅在该固定 workload 成立」。
- [ ] 若确认是测量伪影或同步边界重排：在 `VirtualShadowMapMarkPageSize.md:63` 追加
      更正段落（含新数据与差异解释），并撤回「Projection 实测回退」的表述。
- [ ] 结论写回 `VirtualShadowMapMarkPageSize.md:63` 与
      `VirtualShadowMapProductionOptimizations.md:12`，不再保留没有后续编号的
      「后续应先定位」句子。

## 相关文档 / References

- [VirtualShadowMapMarkPageSize.md](../../VirtualShadowMapMarkPageSize.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [VirtualShadowMapProfiling.md](../../VirtualShadowMapProfiling.md)
- [Mark page-size tool protocol](../../../Tools/Perf/VirtualShadowMapMarkPageSize.md)
