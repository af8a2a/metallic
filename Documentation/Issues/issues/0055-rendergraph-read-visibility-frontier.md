---
id: "0055"
title: "RenderGraph: 同状态 read/read 快路径可能丢失 writer 可见性，尚未取得 GPU 级验证"
area: rhi
kind: defect
priority: P2
state: open
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P2"]
evidence:
  - Documentation/RenderGraphAutomaticSynchronization.md:116
  - Documentation/RenderGraphAutomaticSynchronization.md:109
  - Documentation/RenderGraphAutomaticSynchronization.md:222
---

## 背景 / Context

RenderGraph 的自动同步用 `GraphAccessPlan` 推导跨 pass 的 RAW/WAR/WAW、layout 与
跨队列前置依赖，再由 Vulkan 同步编码器落地。规划器的一条快路径是：同状态读取时
不新增 reader 之间的边，只合并 reader stage。

## 现状 / Current state

文档已经把这处列为「源码静态审计发现、需要专项验证」的情形：

- 访问组合表把 `Read → Read` 定义为「在 layout、owner 兼容，且所有读者均已被原
  writer 的可见性覆盖时」才不增加边。
- 但实现说明承认现状："当前 `transition()` 在同状态读取时只累积 lastScope，不再
  保留 writer 的可见性覆盖"。对 `compute 写 → compute 读 → fragment 读`，若第一道
  窄 barrier 只对 compute 读可见，后续仅合并 reader stage **不能**自动证明 fragment
  读也可见。
- 同一文档同时指出，验证层对动态 shader 访问（bindless/BDA 可达集合）有覆盖限制，
  「没有 validation 错误」不能证明使用集完整，因此这个问题大概率不会被现有
  `--rhi-validation` 回归自动暴露。
- 文档明确：本轮未用 GPU 复现，也不据此认定现有生产场景已经出错。

## 问题或缺口 / Problem

这是一个已知的、未被证伪的正确性漏洞面：

1. 没有人能回答「当前生产图里是否存在 compute→compute→fragment 的三段式读取」，
   因为缺一个能捕捉该模式的最小用例。
2. 若确实存在，症状是偶发、依赖驱动与阶段的可见性丢失，表现为个别像素/页数据
   陈旧，靠现有浮点容差与单帧冒烟测试几乎不可能稳定复现，事后定位成本极高。
3. `synchronization_scopes_batch_and_validation` 只验证编码器 scope 校验/合并，
   不覆盖该语义，回归网存在缺口。

## 期望结果 / Expected outcome

- 用 writer-frontier/visibility 模型补齐 reader 可见性覆盖（或证明在现有
  `transition()` 编码下等价），并给出可重复的判定用例。
- 在 equal-layout 读分叉上同时保证：后续 write 等待全部 reader；layout 变化仍等待
  之前的访问；layout 转换自身的写入不被 read/read 快路径吞掉。

## 验收标准 / Acceptance criteria

- [ ] 新增最小 GPU 用例：compute 写 → compute 窄 barrier 读 → fragment 读，
      读回像素必须等于 writer 结果；用例在
      `build\tests\MetallicRHITests.exe --gtest_filter='*render_graph_*' --rhi-validation`
      下通过（当前先允许记录为已知失败并保存原始日志）。
- [ ] 打开同步校验跑完整聚焦集，且不新增 VUID：
      `$env:VK_KHRONOS_VALIDATION_VALIDATE_SYNC='true'` 后运行
      `--gtest_filter=RHICommand.frame_*:RHIResource.frame_*:RHIRendering.frame_* --rhi-validation`。
- [ ] 对当前生产图（MiniZorah / 编辑器 realtime）执行 `rg graph` 静态导出，
      列出所有 `compute 写 → 读 → 其他 stage 读` 的三段式链路，逐条给出结论
      （覆盖或需要修复）。
- [ ] `Documentation/RenderGraphAutomaticSynchronization.md` 更新该段的验证结论，
      不再停留在「需要专项验证」。

## 相关文档 / References

- [RenderGraphAutomaticSynchronization.md](../../RenderGraphAutomaticSynchronization.md)
- [ParallelRecordingAndSubmission.md](../../ParallelRecordingAndSubmission.md)
