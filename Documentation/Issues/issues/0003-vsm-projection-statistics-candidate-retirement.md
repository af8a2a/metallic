---
id: "0003"
title: "VSM: Projection 组内统计候选已被判定确认回退，实验开关仍留在两个 Pass 的生产代码里"
area: shadow
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:shadow", "kind:task", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapProjectionStatistics.md:1-3
  - Documentation/VirtualShadowMapProjectionStatistics.md:45-62
  - Documentation/VirtualShadowMapProjectionStatistics.md:27-29
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:59
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp:41
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:379
  - Documentation/VirtualShadowMapReceiverTile.md:1-7
  - Documentation/VirtualShadowMapProductionOptimizations.md:15-17
  - Tools/Perf/VirtualShadowMapRetiredExperiments.md:25-45
---

## 背景 / Context

`optimizeProjectionStatistics` 把 PCF/SMRT 投影里逐像素累加的公共统计（有效 receiver、
coarse fallback、background、coarse outside、更粗 fine level、局部近面拒绝）合并到已有的
线程组汇总里，把三个 shared 计数器扩到八个（12 B → 32 B），以减少源码层面的全局 atomic
调用。

## 现状 / Current state

- 两个场景的 discovery 都判定为 `reject / confirmed_regression`，没有进入 confirmation：
  Sponza Projection −5.42%（CI −8.27%–−2.64%）、Mixed Projection −8.37%
  （CI −13.06%–−3.87%），Shadows/RenderGraph 区间跨零
  （`VirtualShadowMapProjectionStatistics.md:51-60`）。
- 文档结论是「保留原生产路径，不推广该候选；实验代码仅由显式开关保留用于复现」
  （`:1-3`、`:62`）。
- 代码完整保留：设置项 `VirtualShadowMapSettings::optimizeProjectionStatistics = false`
  （`VirtualShadowMaps.h:59`）、两个 Pass 的 `benchmarkProjectionStatistics` 解析
  （`VirtualShadowMapPass.cpp:41`、`VirtualShadowMapLightsPass.cpp:379`）、
  `VSM_OPTIMIZE_PROJECTION_STATISTICS` 宏分支与 `[VSMShader]` 日志字段。
- 同一子系统里最近的否决实验处理方式不同：空 receiver tile 实验「按用户要求撤掉实验实现、
  开关和专用测试」，只保留 `build/` 下的归档包（`VirtualShadowMapReceiverTile.md:1-7`）。
- 生产固化文档把「已接受优化」逐项登记并说明退役的旧开关不再适用于当前二进制
  （`VirtualShadowMapProductionOptimizations.md:15-19`），但没有登记被否决的候选。
- 工具侧已有一份退役登记 `Tools/Perf/VirtualShadowMapRetiredExperiments.md:25-45`，明确写
  「compact-build、mark-ray-pages、directional-projection、one-pass-projection 保持可用」，
  并写「其余候选控制，包括 … Projection statistics 与 ray-page Mark，本次未退役」。
  也就是说保留本候选是**有意为之**，但登记的是「工具入口是否可跑实验」，没有登记
  「渲染器还要为它维护多久」的判断依据。

## 问题或缺口 / Problem

1. 一个已被两次独立判定为回退的候选，仍以生产代码路径存在（多一个宏分支、多一个程序缓存
   身份字段、多 20 B/线程组 shared 配额），只是默认关闭。
2. 现有登记只回答「能不能重跑实验」，不回答「这条死分支的退役条件是什么」。仓库对否决候选
   有两种先例（receiver tile 彻底撤除、本候选保留），但没有一条规则说明何时从后者转向前者，
   后续读者无法判断哪些开关值得维护。
3. 该宏参与 `Programs` 缓存键与日志身份，任何未来的 Projection shader 改动都要维护这条
   死分支，且 `virtual_shadow_map_projection_equivalence` 必须继续覆盖它。
4. 文档里 `:27-29` 已写明可能原因是 shared atomic 竞争/额外清零与存活开销，但没有寄存器
   或硬件计数器证据，这个结论本身也需要固化，否则它会被反复重试。

## 期望结果 / Expected outcome

给出一次明确判定：撤除（与 receiver tile 一致），或保留但在代码注释、Inspector 与工具退役
登记中标注「已知回退候选、仅用于证据复现」，并把退役条件（例如「若下一轮仍无硬件计数器
证据则撤除」）写下来。

## 验收标准 / Acceptance criteria

- [ ] 决策写入 `VirtualShadowMapProjectionStatistics.md` 顶部状态行，并在
      `VirtualShadowMapProductionOptimizations.md:15-19` 之后新增「已否决/已撤除候选」
      小节，与 `Tools/Perf/VirtualShadowMapRetiredExperiments.md:25-45` 的 live 清单对齐。
- [ ] 若撤除：删除 `VirtualShadowMaps.h:59` 设置项、两处属性解析、shader 宏与分支、
      `[VSMShader]` 中的 `projectionStatistics` 字段；Sponza 与 Mixed 的第 16/32/63 帧
      PNG、线性 HDR、depth、shadow 与撤除前逐字节一致（默认关闭路径本身不应变化）。
- [ ] 若撤除：`virtual_shadow_map_projection_equivalence` 与
      `virtual_shadow_map_profiling` 全部状态通过、Vulkan validation 为 0，
      且原有「非本候选」控制臂数量在测试输出中可见。
- [ ] 若保留：两个 Pass 的 Inspector 标注已知回退，并在文档写明复现入口
      （`.cache/VSMProjectionStatistics/`）不被默认构建或样例依赖。
- [ ] 无论哪种结果，文档保留 `:27-29` 的「无 SASS/硬件计数器，不能确证单一原因」边界，
      不把回退归因写成已证明结论。

## 相关文档 / References

- [VirtualShadowMapProjectionStatistics.md](../../VirtualShadowMapProjectionStatistics.md)
- [VirtualShadowMapReceiverTile.md](../../VirtualShadowMapReceiverTile.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [Retired VSM experiments](../../../Tools/Perf/VirtualShadowMapRetiredExperiments.md)
