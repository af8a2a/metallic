---
id: "0050"
title: "PathTracing: 五个实验后端（RT pipeline / SER / vkGltf / SimpleBSDF / splitPrimaryGuides）全部默认关闭且未过严格数值门槛"
area: material
kind: research
priority: P2
state: in-progress
milestone: M1-Active
labels: ["area:material", "kind:research", "priority:P2", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/PathTraceTraversal.md:96-106
  - Documentation/PathTraceTraversal.md:210-223
  - Documentation/SimpleBSDF.md:145-151
  - Documentation/SimpleBSDF.md:334-337
  - Documentation/OpenPBRPrimaryGuidesSplit.md:118-124
  - Documentation/PathTracingMaterialBackendExperiment.md:654-667
---

## 背景 / Context

路径追踪在架构上有多个可选后端与优化实验，全部以 pass 属性或环境变量 opt-in，
默认路径始终是 compute + inline RayQuery + fused 材质求值。这些实验的价值在于为
「是否更换执行后端」提供证据，而不是作为功能本身交付。

## 现状 / Current state

| 实验 | 入口 | 默认 | 当前结论 |
|---|---|---|---|
| RT pipeline（非 RayQuery） | `traversalBackend: "rayPipeline"` | `rayQuery` | 全部 14 组配置未通过严格 HDR/guide 等价门槛；GPU 上更慢 |
| EXT SER（vkGltf / 主路径） | `shaderExecutionReorder: true` | 关闭 | SER 在三个 workload 上都是 **regression**；严格正确性未合格 |
| vkGltf BSDF | `bsdf: "vkGltf"` | OpenPBR | 比 OpenPBR 快，但是不同材质/积分器模型，不构成等价加速 |
| SimpleBSDF material generics | `bsdf: "simple"` | 直接调用 | wrapper 同 SER 内对比无法确立稳定收益；SER 开关本身违反等价门槛 |
| splitPrimaryGuides | `splitPrimaryGuides: true` | 关闭 | 最长创建 −14%，但每 recording 100 B/像素暂存与额外 dispatch，Sky +35.9% |
| Material backend Type | overlay（未进生产源码） | 生产源码不变 | No-Fiber 静态收益 16.31%，但 Mixed runtime +13.70%，不得作为通用默认 |

- RT pipeline 后端：28 个 correctness 进程完成，但 14 组配置全部失败于严格
  HDR/guide 等价门槛；隔离探针显示差异与 TraceRay/payload 无关，「precise driver-codegen
  source has not been established」（`PathTraceTraversal.md:210-216`）。
- vkGltf + SER：226/226 SER HDR 读回通过、39 组同臂对逐字节一致，但 22 个
  hardware-depth guide 比较失败（最大绝对误差 1.19e-7），另有 6 个 HDR 分量在
  性能读回中失败；「Strict SER correctness remains **unqualified**」
  （`PathTraceTraversal.md:96-106`）。
- SimpleBSDF：SER 切换本身在三例上都违反等价门槛（direct 路径 SER pass 时间
  +191.46% / +57.06% / +172.69%），文档明确这些是 diagnostic 而非 qualified 优化；
  且公开 Slang SER API 迁移后**未重跑**编译/PSO/GPU A-B 测量
  （`SimpleBSDF.md:145-151`、`:334-337`）。
- splitPrimaryGuides：记录 100 B/像素、512² 25 MiB、1920×1080 约 197.8 MiB、
  3840×2160 约 791.0 MiB per outstanding recording；「default remains fused」
  （`OpenPBRPrimaryGuidesSplit.md:41-46`、`:118-124`）。
- material backend Type：Mixed scene dispatch +13.70%，结论是「This candidate is not
  qualified as a general runtime improvement……Prefer explicit scene capabilities and
  coarse, independently selected material implementations for the next experiment」
  （`PathTracingMaterialBackendExperiment.md:9-17`、`:654-667`）。

## 问题或缺口 / Problem

1. **没有一条实验被正式关闭。** 五个实验各自留下「未合格 / 更慢 / 未重跑」，但没有
   一条写下「本路线到此为止」或「还需要哪一个具体实验才能翻案」。久而久之这些
   opt-in 分支会变成无法回收的技术债。
2. **数值等价门槛本身暴露了问题。** RT pipeline 与 SER 的失败都指向同一类现象：
   相同源码换执行模型后产生 1 ULP 级差异（尤其 hardware-depth 与 camera 方向），
   而验收要求 guide 逐字节一致。当前没有机制表达「哪些差异可接受、哪些是缺陷」。
3. **SER 的 regression 已有指向但未收敛。** SimpleBSDF 的 shadow-SER policy control
   把 Mixed4 的 SER 变化从 +60.76% 反转为 −28.15%，「strongly implicating shadow
   reordering and its associated codegen」，但这是跨周期测量，不能直接采纳
   （`SimpleBSDF.md:219-244`）。
4. **公开 API 迁移后证据过期。** 当前实验源码已改用 `HitObject::TraceRay` /
   `ReorderThread` / `Invoke`，但早先的性能数字属于旧手写融合实现，文档自己也说
   「must not be relabeled as results for the current split public-API implementation」
   （`SimpleBSDF.md:334-337`）。

## 期望结果 / Expected outcome

- 每个实验有一个明确状态：**采纳 / 保留为可选诊断 / 关闭并删除**，并给出翻案条件。
- 「执行模型差异导致的 1 ULP 偏移」有正式的容差与解释文档，区分可接受差异与真实缺陷。
- SER 的 shadow policy 假设用同一周期内的等负载 A/B 收敛。

## 验收标准 / Acceptance criteria

- [ ] 对五个实验各自新增状态行（采纳 / 可选 / 关闭 + 翻案条件），写入对应文档顶部。
- [ ] RT pipeline 后端：要么把 14 组等价失败定位到具体 codegen 源并修复，要么明确
      关闭该后端并说明保留的替代路径。
- [ ] SER：在同一测量周期内完成 shadow-only 与 full policy 的配对 A/B（至少 3 个独立
      进程、ABBA 顺序），给出是否取消 shadow SER 的结论。
- [ ] 公开 API（`HitObject`）实现重跑一次四阶段编译 + native PSO + GPU pass 测量，
      替换掉已过期的旧数字，或删除旧表格并标注「已失效」。
- [ ] 等价门槛文档化：guide 逐字节一致是否必须、1 ULP 差异的判定流程、以及
      `2e-5 + 2e-5·|ref|` HDR 容差的适用边界。

## 相关文档 / References

- [PathTraceTraversal.md](../../PathTraceTraversal.md)
- [SimpleBSDF.md](../../SimpleBSDF.md)
- [OpenPBRPrimaryGuidesSplit.md](../../OpenPBRPrimaryGuidesSplit.md)
- [PathTracingMaterialBackendExperiment.md](../../PathTracingMaterialBackendExperiment.md)
- [PathTracingLinkTimeSpecialization.md](../../PathTracingLinkTimeSpecialization.md)
