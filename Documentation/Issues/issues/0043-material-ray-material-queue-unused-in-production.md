---
id: "0043"
title: "Material: M8 ray-hit 分类队列没有生产消费者，路径追踪仍只有 fused program loop"
area: material
kind: task
priority: P2
state: in-progress
milestone: M1-Active
labels: ["area:material", "kind:task", "priority:P2", "state:in-progress"]
depends_on: []
evidence:
  - Source/Runtime/Render/Material/RayMaterialQueue.h:36-38
  - Documentation/MaterialSystemM8RayExecution.md:22-38
  - tests/rhi/RayMaterialTests.cpp:27
  - Tools/ShaderWarmup.cpp:105
  - Shaders/Features/PathTracing/RayMaterialQueue.slang
---

## 背景 / Context

M8 的目标是把「二次命中的求值与准备」显式化，并交付经过 GPU 验证的有界 Program
分类队列，使 ray-hit 侧可以按完整 MaterialProgramKey 归组消费，而不是在 fused path
loop 内就地求值。这是 M5 的 DGC / wavefront 路线的前置基础设施。

## 现状 / Current state

- 执行契约与队列本体都已落地并且有 GPU 证据：
  `Modules/RayMaterialExecution.slang` 的 `RayMaterialWork` / `evaluateRaySurface` /
  `prepareRaySurface` 契约，加上 `Shaders/Features/PathTracing/RayMaterialQueue.slang`
  的 reset → count → prefix → scatter 四个 ComputeKernel。
- `RHIRendering.ray_material_execution_queue` 用真实 Slab 与 RTXCR Fiber Program 比较
  串行 hit 顺序与分类后消费的 eval/sample/weight/RNG 输出，覆盖 5 个 Program、20 组
  容量/尾组/bounce 用例、574 次有效求值，分组前后逐位一致
  （`MaterialSystemM8RayExecution.md:40-46`）。
- 但**没有任何生产代码创建或提交 `RayMaterialQueue`**：全仓库对
  `RayMaterialQueue::classify` 的唯一调用方是 `tests/rhi/RayMaterialTests.cpp`
  （测试名注册于 `tests/rhi/RayMaterialTests.cpp:27`）。
- 头文件自己写明了这个边界：`Bounded classifier primitive; no default renderer
  scheduling policy is changed.`（`Source/Runtime/Render/Material/RayMaterialQueue.h:36-38`）
- 文档同样确认：「这是可复用的分类基础设施，默认完整场景仍使用 fused path loop。
  尚未将所有场景积分器切换成 wavefront，也未声明吞吐或寄存器收益」
  （`MaterialSystemM8RayExecution.md:38`）。
- 该队列模块还被显式排除在 shader warmup 请求过滤之外
  （`Tools/ShaderWarmup.cpp:105`），说明它目前既不在默认执行路径，也不在预热清单。

## 问题或缺口 / Problem

1. **基础设施与生产之间缺一层调度器。** 队列需要调用者提供 hit buffer、material
   program table、bins/indices/cursors/status 与 generation 保留；目前没有 pass
   承担这个职责，`ScenePathTracePass` 仍在 fused 循环里逐命中求值。
2. **容量与失败语义只在测试里被验证过。** 生产接入必须决定：溢出（overflow）、
   过期 generation（stale）、无效材质（invalid）在一帧内如何处理。文档要求
   「任何 invalid / stale / overflow 都要求调用者报告失败或完整重试，不得把部分
   结果当作完整帧」，但当前没有调用者实现这条策略。
3. **预热与缓存身份未接。** 队列 shader 被 warmup 过滤掉，一旦接入生产路径，
   冷启动会新增一次未预热的 kernel 编译。

## 期望结果 / Expected outcome

- 至少一条生产路径（建议：OpenPBR/Slab 的 secondary hit 消费）在显式开关下使用
  `RayMaterialQueue` 完成分类后消费，默认仍为 fused；开关两侧输出逐位一致。
- 溢出 / stale / invalid 有明确的帧级处理策略（失败并诊断，或整帧重试），并在日志
  中可观察。
- 队列 kernel 进入 shader warmup 目录与缓存身份计算。

## 验收标准 / Acceptance criteria

- [ ] 新增一个 GPU 场景回归：同一场景在 fused 与新队列路径（建议新增 pass 属性
      `rayMaterialQueue`，默认 `false`）两种设置下，PT 的原始线性 HDR 在声明的容差内
      一致（若只做重排则应逐位一致）。
- [ ] 该回归在 capacity 不足（人为缩小容量）时失败并给出 overflow 诊断，而不是
      静默输出部分帧。
- [ ] `RHIRendering.ray_material_execution_queue` 保持通过，且新增用例覆盖
      generation 在帧内过期后的重试路径。
- [ ] `Tools/ShaderWarmup.cpp` 不再过滤 `Features/PathTracing/RayMaterialQueue`；
      一次完整 warmup 请求列表包含该模块且 0 失败。
- [ ] 文档 `MaterialSystemM8RayExecution.md` 更新「生产消费者」一节，写明接入位置
      与默认开关状态。

## 相关文档 / References

- [MaterialSystemM8RayExecution.md](../../MaterialSystemM8RayExecution.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
- [PathTracingShaderEncapsulation.md](../../PathTracingShaderEncapsulation.md)
