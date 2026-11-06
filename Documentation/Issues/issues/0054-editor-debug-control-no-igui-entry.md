---
id: "0054"
title: "Editor: 调试控制面没有编辑器内入口，生产 watch 仅限一次性专用进程"
area: editor
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:editor", "kind:feature", "priority:P2"]
evidence:
  - Documentation/DebugControlPlane.md:6-12
  - Documentation/AgenticShaderPrintfDebugger.md:337
  - Source/Editor/EditorApplication.cpp:2086
  - Source/Editor/EditorApplication.cpp:3418-3438
---

## 背景 / Context

调试控制面（capture / `gpu.probe` / `watch` / 离线导出）已经在 `DebugCore` 与
`metallicctl` 里成型，能力边界、预算与错误码都有文档。但它是为「外部 agent 通过
命名管道查询运行中的引擎」设计的控制面，自身没有编辑器界面。

## 现状 / Current state

- 文档自陈设计边界：「It is opt-in and independent of ImGui」；并说明生产采集
  目前是「one watch in a dedicated workload process, closed after instance
  destruction」。
- 编辑器只在启动路径上按 `--debug-control` / `METALLIC_DEBUG_CONTROL` /
  `METALLIC_SHADER_TRACE` 决定是否启动服务，之后没有面板、菜单项或状态提示。
- `Window` 菜单当前列出 Material Graph、Render Graph Editor、Scene Browser、
  Runtime Settings、Inspector、Statistics、Viewport、Assets、Console、Profiler、
  NVML Monitor、Streamline Debug，**没有任何调试控制面条目**。
- Shader Printf 侧同样承认「尚未推广到任意常驻编辑器连续 watch」。

## 问题或缺口 / Problem

用户可感知的缺口：

1. 想抓一次 VBuffer checkpoint、看一个 probe 结果或起一个 watch，必须先在终端
   设环境变量重启编辑器，再另开终端敲 `metallicctl`，编辑器里看不到 job 状态、
   错误或证据目录。
2. 长驻编辑器上的连续观测（例如「每 60 帧检查一次 request 溢出」）不受支持，
   只能开专用 workload 进程，这与「边编辑边排查」的实际使用方式冲突。
3. 出错时（`StaleHandle`、`Unsupported`、`BudgetExceeded`）错误只回到 CLI stdout，
   编辑器内的 `renderGraphStatus_` 等状态区不显示，用户容易误以为「没问题」。

## 期望结果 / Expected outcome

- 编辑器内新增只读的 Debug Control 面板：服务状态、provider/capability 列表、
  job 列表与状态机、probe/watch 结果、错误码与证据目录，支持从面板发起
  capture/probe/watch（复用现有协议，不新增能力）。
- 常驻编辑器支持有界连续 watch（受既有 `everyExecutions` / `maxSamples` /
  `timeoutMs` 与预算约束），不要求专用进程。

## 验收标准 / Acceptance criteria

- [ ] `Window` 菜单出现 `Debug Control` 项，面板可显示服务未启用时的手动启用提示。
- [ ] 编辑器 smoke 测试覆盖面板创建与 job 列表渲染，
      例如 `Source/Editor/EditorApplicationSmokeTests.cpp` 新增用例并通过
      `ctest --test-dir build-ci -R '^MetallicLookDevVBufferSmoke$' --output-on-failure`。
- [ ] 在常驻编辑器里创建 `everyExecutions=60` 的 watch，能在不改用专用进程的
      情况下拿到 triggered 证据；`metallicctl watch list` 与面板显示一致。
- [ ] 面板与 `metallicctl` 对同一 job 的 eval/export 结果逐字段一致（在线与离线）。
- [ ] `Documentation/DebugControlPlane.md` 更新使用入口与「不再要求专用进程」的
      状态描述。

## 相关文档 / References

- [DebugControlPlane.md](../../DebugControlPlane.md)
- [AgenticShaderPrintfDebugger.md](../../AgenticShaderPrintfDebugger.md)
