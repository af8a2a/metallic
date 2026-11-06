---
id: "0012"
title: "VSM: 方向光 Mark 的空 tile 与跨光重复准备成本未测量，唯一同类候选已被否决"
area: shadow
kind: research
priority: P3
state: open
milestone: M5-Research
labels: ["area:shadow", "kind:research", "priority:P3"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapDirectionalMark.md:92-94
  - Documentation/VirtualShadowMapReceiverTile.md:1-7
  - Documentation/VirtualShadowMapProductionOptimizations.md:7-13
  - Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h:13-16
  - Documentation/VirtualShadowMapsGPUProfile.md:37-40
---

## 背景 / Context

方向光 Mark 已完成多轮固化（共享页缓存去重 + cluster 粗筛、局部灯采样不变量复用、
方向光专用编译、clipmap 数与页尺寸专用编译、有界 tile 请求集合覆盖），
现在 Mark 的剩余开销集中在「没有有效 receiver 的 8×8 tile 仍要付共享哈希初始化与组同步」
以及多灯场景下跨光的重复准备（`VirtualShadowMapDirectionalMark.md:92-94`）。

## 现状 / Current state

- 文档已经把下一步写清楚，但只是建议而非任务：「空 8×8 tile 仍支付共享哈希初始化和
  组同步；下一步应先测量空 tile 比例和跨光重复准备成本，再决定是否增加可复用的有效
  receiver tile 列表，连同生成、同步和消费总成本做 A/B」（
  `VirtualShadowMapDirectionalMark.md:94`）。
- 唯一做过的同类实验已被否决并整体撤除：空 receiver tile 候选「因 Dense 的 level 测试
  计数差异及 SkyExterior 的显著性能回退被拒绝，没有进入默认路径」，实现、开关与专用测试
  全部撤掉，只留归档包（`VirtualShadowMapReceiverTile.md:1-7`）。那是 Projection 侧，
  Mark 侧从未做过同类尝试。
- 现有计数器能间接看出 8×8 tile 的浪费，但没有空 tile 比例：`markReceiverDispatches`
  只数「域内有深度的 receiver 线程」，`markSamplePositions` 数所有 markPosition 调用，
  `markLevelTests` 数通过面检查后的 level 测试（`VirtualShadowMapProfiling.h:13-16`、
  `VirtualShadowMapsGPUProfile.md:37-40`）。tile 总数可见、空 tile 数不可见。
- 已固化清单里明确写着「共享 Mark 单项未确认净收益，组合已确认」
  （`VirtualShadowMapProductionOptimizations.md:7`），说明 Mark 的单点收益本身很容易被
  同步/初始化成本抵消。

## 问题或缺口 / Problem

1. 一个明确的剩余开销没有任何量化：既没有空 tile 比例，也没有每 tile 的准备成本，
   因此无法判断是否值得做 tile 列表。
2. 唯一的同族实验（Projection 的 receiver tile）已经被否决，原因正是「计数差异 +
   显著回退」，但那条负面证据没有被回灌到 Mark 的议程，文档仍在建议「先测量再决定」，
   看起来像从零开始。
3. 多灯场景的跨光重复准备成本完全未知；这与已固化的「局部灯 Mark 不变量复用」是不同
   切面（那是单灯内的不变量，这是灯与灯之间的重复）。
4. 现有计数器无法支撑这个判断，若直接改 shader 做实验，很可能重复 receiver tile 的
   失败方式（先实现后测量）。

## 期望结果 / Expected outcome

先用现有计数器+一次受控诊断把空 tile 比例与跨光重复准备成本量化出来，再决定是否进入
实现；若不值得，就把 `VirtualShadowMapDirectionalMark.md:94` 的建议改写为已评估结论，
避免后续重复讨论。

## 验收标准 / Acceptance criteria

- [ ] 用现有 `markReceiverDispatches` / `markSamplePositions` / `markLevelTests` 推导
      空 tile 比例（必要时新增一个 `markEmptyTiles` 计数器，走诊断 shader permutation，
      不影响普通编译），给出 Sponza 与 Mixed 两场景的后 32 帧中位数。
- [ ] 给出多灯（≥1 方向光 + ≥2 局部光）场景下跨光重复准备的实测值，例如同 tile 被多个
      灯重复初始化的次数占比。
- [ ] 预声明门槛：只有「空 tile 准备成本占 Mark 区间 ≥ X%」或「tile 列表可回收 ≥ Y%」
      时才进入实现；X/Y 必须在采样前写入实验目录，不做事后调整。
- [ ] 若未达门槛：把 `VirtualShadowMapDirectionalMark.md:94` 改为「已测量、不采纳」并
      附数据；若达标：给出带生成、同步、消费总成本的 A/B 协议，并在协议中明确
      引用 `VirtualShadowMapReceiverTile.md` 的否决记录作为失败模式约束。
- [ ] 本条目不修改生产默认值；新增计数器（若需要）必须计入 workload 版本与文档
      ABI（见 0002）。

## 相关文档 / References

- [VirtualShadowMapDirectionalMark.md](../../VirtualShadowMapDirectionalMark.md)
- [VirtualShadowMapReceiverTile.md](../../VirtualShadowMapReceiverTile.md)
- [VirtualShadowMapMarkClusterOptimization.md](../../VirtualShadowMapMarkClusterOptimization.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [VirtualShadowMapsGPUProfile.md](../../VirtualShadowMapsGPUProfile.md)
