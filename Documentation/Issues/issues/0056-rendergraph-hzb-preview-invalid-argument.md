---
id: "0056"
title: "RenderGraph: Visibility 第二帧 HZB preview 稳定返回 InvalidArgument，回归未修复"
area: rhi
kind: defect
priority: P1
state: open
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P1"]
evidence:
  - Documentation/RenderGraphMemoryAliasing.md:143
  - Documentation/RenderGraphMemoryAliasing.md:105
---

## 背景 / Context

`RenderGraph` 的 transient/aliasing 扩展在 2026-09/10 做了一轮逐 pass 契约审查，
为图拥有的资源补充 `Clear` / `FullOverwrite` 初始化声明。审查过程中出现一个与
aliasing 无关、但可稳定复现的既有失败。

## 现状 / Current state

文档记录的原话是：扩展回归共 38 项、36 通过、RTXDI/RELAX preview 因
`METALLIC_ENABLE_NRD=OFF` 跳过；**Visibility 第二帧 HZB preview 返回
`InvalidArgument`**。并且已经确认：

- 在隔离进程、aliasing 开 / 关两种配置下都复现；
- 临时恢复修改前的 `VisibilityBufferPass.cpp`、重新构建后复现相同错误，因此
  未归因于当轮的 transient 标记改动；
- 原始日志保留，但「本次未修改这个已有 HZB 行为」，也没有另立跟踪项。

相关背景：`VisibilityBufferPass` 的 transient 输出只在 tessellation 开启时才参与
附件 clear，关闭时 dummy 未初始化并保持 `Persistent/Unknown`，说明这条 pass 的
初始化契约本身就比较绕，第二帧失败落在这个范围内并不意外。

## 问题或缺口 / Problem

1. 这是一个**已知、可复现、已定位到 pass 级别但无人负责**的正确性回归。任何
   第二帧 HZB preview 的调用方都会拿到 `InvalidArgument`，而错误种类是
   「参数非法」而不是「不支持」，容易把根因指向调用方，误导排查方向。
2. 因为它不是当轮改动的责任，逐 pass 审查的结论文档变成了事实上的「已知问题
   存放处」，没有进入可关闭的跟踪状态。
3. 后续做 aliasing 默认开启、HZB 历史复用或 streaming 页表复用时，这个失败会
   与新区改混在一起，增加归因成本。

## 期望结果 / Expected outcome

- 明确第二帧 HZB preview 的第二帧路径上哪个资源状态/描述非法（是
  `historyResources` 的 generation、HZB 视图尺寸，还是 transient 输出的初始化
  契约），给出根因与修复，或给出「第二帧 preview 不受支持」的显式 `Unsupported`
  语义与文档说明。
- 该场景进入自动化回归，不再依赖人工逐一执行。

## 验收标准 / Acceptance criteria

- [ ] 最小复现用例进入 `MetallicRHITests`（第二帧 HZB preview），修复后返回成功并
      校验像素，或在明确降级时返回 `Unsupported` 且断言原因：
      `build\tests\MetallicRHITests.exe --gtest_filter='*visibility*hzb*' --rhi-validation --rhi-bindless`
- [ ] alias 开 / 关两种配置下结果一致，原始日志作为证据附在 PR 中。
- [ ] `Documentation/RenderGraphMemoryAliasing.md` 的验证记录从「未修改这个已有
      HZB 行为」改为结论链接（issue + 修复提交）。
- [ ] `ctest --test-dir build-ci -R '^MetallicRHITests$' --output-on-failure` 无新增 skip。

## 相关文档 / References

- [RenderGraphMemoryAliasing.md](../../RenderGraphMemoryAliasing.md)
- [RenderFrameLifetime.md](../../RenderFrameLifetime.md)
