---
id: "0057"
title: "RenderGraph: 精确 subresource 范围与 exclusive queue-family ownership 尚未纳入同步计划"
area: rhi
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:rhi", "kind:feature", "priority:P2"]
evidence:
  - Documentation/RenderGraphAutomaticSynchronization.md:249
  - Documentation/RenderGraphAutomaticSynchronization.md:120
  - Documentation/ProjectArchitecture.md:260
  - Documentation/RenderGraphMemoryAliasing.md:302
---

## 背景 / Context

自动同步计划目前覆盖「图拥有资源、跨 pass、整资源访问」。第一阶段有意按整资源
处理，先保证正确性，再收窄范围。

## 现状 / Current state

- 文档列出尚未纳入的四项：**buffer slice 精确区间、texture mip/layer/aspect、
  动态 bindless/BDA 可达集合、exclusive ownership transfer**；opaque 子系统内部
  同步继续承担这些责任。
- `acquireFromQueue` 只适用于已共享到目标 family 的资源，后端使用
  `VK_QUEUE_FAMILY_IGNORED`，文档明确「并不是 exclusive ownership transfer」，
  将来支持 exclusive 跨 family 导入时必须成对生成 release/acquire。
- 架构文档同样记着多队列入口「没有实现 exclusive queue-family ownership
  transfer」。
- aliasing 侧也按整资源排除不合格者（storage read-write、atomic 累积、attachment
  LOAD、仅写部分像素或 mip/layer 的资源默认不合格），因为缺乏 subresource 级证明。

## 问题或缺口 / Problem

工程后果是可见的：

1. **barrier 范围偏宽**：buffer 最终使用 global barrier，原生同步范围可能比逻辑
   资源范围更宽；HZB 多 mip、streaming page table 的内部 barrier 无法在同模型内
   消除，跨队列 copy/写回也无法只声明一段区间。
2. **别名收益被范围粒度限制**：需要「只写部分像素/mip/layer」的资源一律退回独立
   分配，`Render Graph Editor → Execution → Memory` 里的 saved capacity 因此偏保守。
3. **exclusive ownership 路径整个不在计划里**：只能对已 concurrent-shared 的资源
   做跨 family 使用；想把某个资源切到 exclusive 布局就离开自动同步体系。

## 期望结果 / Expected outcome

- 先支持整资源、再在同一模型内启用精确范围（buffer byte range、texture
  mip/layer/aspect），并让未迁移 pass 显式保持 legacy/opaque 契约。
- 引入成对 release/acquire 的 exclusive 跨 family 导入，且由 debug validator 拒绝
  漏报访问或混用手动状态。

## 验收标准 / Acceptance criteria

- [ ] 规划器接受 `BufferSlice` 区间与 texture subresource 范围，新增 CPU 用例覆盖
      分离/部分重叠 slice、mip/layer/aspect、两个 view 指向同一 allocation、
      未初始化读取、未排序多 writer 拒绝：
      `ctest --test-dir build-ci -R '^MetallicRHITests$' --output-on-failure` 中的
      `*render_graph_access_plan*` 用例通过。
- [ ] GPU 用例：跨队列 exclusive 所有权转移（release/acquire 成对）后像素一致，
      且在 `--rhi-validation` 下无 VUID。
- [ ] `MetallicRHITests` 的 `*render_graph_execution_viewer*` 中 memory 汇总在
      subresource 别名生效后仍报出可核对的总量与槽位明细。
- [ ] 文档「当前边界」一节更新为已实现范围，并明确仍由 opaque 子系统负责的部分。

## 相关文档 / References

- [RenderGraphAutomaticSynchronization.md](../../RenderGraphAutomaticSynchronization.md)
- [RenderGraphMemoryAliasing.md](../../RenderGraphMemoryAliasing.md)
- [RenderFrameLifetime.md](../../RenderFrameLifetime.md)
