---
id: "0053"
title: "DebugControlPlane: 只有两个 pass 注册 checkpoint，其余 pass 无法 capture 或 probe"
area: tooling
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:tooling", "kind:feature", "priority:P2"]
evidence:
  - Documentation/DebugControlPlane.md:171-174
  - Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:212
  - Source/Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetPass.cpp:299-300
---

## 背景 / Context

调试控制面（`METALLIC_DEBUG_CONTROL` / `metallicctl`）的 GPU 证据全部以
`{pass, checkpoint}` 定位：`capture.batch`、`gpu.probe`、`watch` 都必须命中一个由
pass 自己注册的 checkpoint，否则返回 `Unsupported`（"Pass/checkpoint is not
registered in the active graph"）。

## 现状 / Current state

- 文档明确只列出两个发布者：`VisibilityBufferPass` 与 `GPUDrivenStreamAssetPass`；
  RTAS-only 可视化只发布 `Traversal/AfterPass`。
- 代码核对一致：`VisibilityBufferPass.cpp:212` 提供
  `AfterEarlyCull / AfterLateCull / AfterPass`（流式开启时追加
  `AfterStream*`、`AfterTraversal`）；`GPUDrivenStreamAssetPass.cpp:299-300`
  提供 `AfterTraversal / AfterEarlyCull / AfterLateCull / AfterPass`。
- `grep AfterEarlyCull|AfterLateCull|AfterTraversal` 在 `Source/Runtime/` 只命中
  上述两个 pass 与 `StreamerSubsystem.cpp:458` 的内部回调，没有第三个 pass 注册点。
- 结论：VSM（`VirtualShadowMapPass`）、路径追踪（`ScenePathTracePass`）、
  `SceneMaterialVisualizationPass`、`ScreenSpaceShadowPass`、`ColorGradingLUTPass`、
  `FinalBlitPass` 等 pass 在任何边界上都无法被 capture 或 probe。

## 问题或缺口 / Problem

需要采集非 GPUDriven 路径中间状态时，唯一的办法是改代码加临时读回或改用外部
profiler，这使控制面在「回调式问题定位」之外几乎不可用：

1. VSM 页表/页标记、投影统计等中间表无法在指定 pass 边界抓取。
2. 材质解析/可视化输出与后处理链只能看最终呈现结果，无法定位是哪个 pass 写坏。
3. `metallicctl rg graph` 能列出全部 pass，但只有两处有可捕获 checkpoint，用户在
   选择 `--pass` 时得不到可行动的能力提示（错误只在 capture 时出现）。

## 期望结果 / Expected outcome

- 为剩余内置 pass（至少 VSM、`SceneMaterialVisualizationPass`、`ScenePathTracePass`、
  `FinalBlitPass`）声明显式 layout/state 契约的 checkpoint，并保证拷贝发生在
  dynamic rendering scope 之外。
- `rg graph` / `hello` 输出中直接列出每个 pass 可用的 checkpoint 与可捕获资源，
  让调用方在选择前就知道哪些边界受支持。

## 验收标准 / Acceptance criteria

- [ ] `metallicctl --pid <PID> --json rg graph` 对新增 pass 列出 checkpoint 名称，
      且对仍未支持的 pass 明确标注 unsupported（不再只在实际 capture 时报错）。
- [ ] 新增 `MetallicRHITests` 用例，在两个以上 checkpoint 写入不同值并分别
      capture/probe，校验在线与离线导出结果独立：
      `build\tests\MetallicRHITests.exe --gtest_filter='*DebugControl*' --rhi-validation`
- [ ] `ctest --test-dir build-ci -L debug --output-on-failure` 通过；
      `pwsh -File tests\debug\DebugControlE2E.ps1 -EnginePid <PID> -CaptureDirectory .tmp\checkpoint-e2e`
      覆盖至少一个新 pass 的 checkpoint。
- [ ] `Documentation/DebugControlPlane.md` 的 checkpoint 发布者清单与代码一致。

## 相关文档 / References

- [DebugControlPlane.md](../../DebugControlPlane.md)
- [RenderGraphResourceInspector.md](../../RenderGraphResourceInspector.md)
