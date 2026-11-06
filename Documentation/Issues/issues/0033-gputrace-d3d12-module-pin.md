---
id: "0033"
title: "Tooling: GPU Trace 退出期依赖 D3D12.dll，靠 PIN 住模块维持注入进程存活"
area: tooling
kind: task
priority: P2
state: blocked
milestone: M3-Hardening
labels: ["area:tooling", "kind:task", "priority:P2", "state:blocked"]
evidence:
  - Documentation/NsightGPUTraceStartupInvestigation.md:7-58
  - Documentation/NsightGPUTraceStartupInvestigation.md:91-118
  - Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp:248-266
---

## 背景 / Context

2026-10-17 的 GPU Trace 集成暴露两个独立问题。其一是启动顺序：原先 host 在
`createDevice` 之前启动，一次冷启动在 native pipeline warmup 里花了 583.9 s（110 条管线），
host 的 target-ready 超时先到，握手返回 `GPU Trace host exited`。其二是退出期崩溃：
`CRT exit -> WarpViz.Injection.dll -> WarpVizTarget.dll -> D3D12Core.dll` 在
`D3D12Core + 0x53b3e` 访问 `[rax + 0x10]`；LLDB 的 unload breakpoint 显示
`nvoglv64.dll` 在 `vkDestroyDevice` 内调用 `FreeLibrary(D3D12.dll)`，把 Nsight 进程退出
回调仍要使用的虚表所在模块卸掉了。

## 现状 / Current state

- 启动顺序已改为在 `prepareBeforeSubmission`（设备创建与 native PSO 预热之后、SDK 激活之前）
  才启动 host，保留 60 s 激活看门狗、抓取 deadline 与 shader warmup；脚本侧支持
  `STARTUP_TIMEOUT_SECONDS` 作为冷缓存整进程预算。
- 退出期用 `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"D3D12.dll")` 在编辑器
  图形关闭前钉住已加载的 `D3D12.dll`，**仅在 GPU Trace 会话**，模块不存在时不加载
  （`NsightGraphicsCapture.cpp:248-266`，注释指向该调查文档）。
- 已验证：`export` 用例产出三份完整 GPU Trace（含 viewport resize 与恢复）且 exit 0；
  `invalid-metrics` 给出明确 host 诊断并 exit 1 不挂起；`disabled` 用例完整预热 + 正常渲染
  exit 0 且不触发握手与 PIN。
- 后续 2026-10-18 的 live 会话又暴露 host `Operation timeout`，已改为 host 使用
  `--no-timeout`、保留 `--trace-timeout 120`、把 host 超时消息上抛到编辑器错误文本，
  并完成三份 fresh trace（1797x694 / 1957x784 / 1978x694 恢复）。

## 问题或缺口 / Problem

PIN 让进程在整个生命周期内保留一个运行时依赖，而且只覆盖「设备创建成功之后的编辑器关闭」
路径：如果设备创建失败并因此销毁一个内部部分设备，再回到编辑器控制流，这条兼容处理没有
证明安全（文档已明确写出该边界）。这类 app-visible 依赖应随 Nsight/驱动升级复评并最终删除，
但目前没有任何自动化回归会提示它可以删除。

## 期望结果 / Expected outcome

在新的 Nsight Graphics / 驱动上验证「同一抓取 + 正常退出」不再需要 PIN，然后删除
`prepareForGraphicsShutdown` 的模块钉住逻辑，并在升级检查清单里登记复评条件。

## 验收标准 / Acceptance criteria

- [ ] 移除 PIN 后 RelWithDebInfo `LookDev.exe --nsight-gputrace` 的 `export` 用例仍产出三份
      完整 trace（含 resize 与恢复）且应用 exit 0。
- [ ] `invalid-metrics` 用例仍给出具体 host 诊断、exit 1 且不挂起；`disabled` 用例不加载
      也不钉住 `D3D12.dll`，完整预热后渲染并 exit 0。
- [ ] 覆盖「设备创建失败」路径：构造一次 createDevice 失败后关闭，记录是否仍出现
      访问已卸载虚表的崩溃。
- [ ] host 启动不早于设备创建与 native PSO 预热（顺序回归仍通过）。
- [ ] 删除后 `grep -rn "GET_MODULE_HANDLE_EX_FLAG_PIN" Source` 为空，文档同步更新复评条件
      与已验证的 Nsight/驱动版本。

## 相关文档 / References

- [NsightGPUTraceStartupInvestigation.md](../../NsightGPUTraceStartupInvestigation.md)
- [NsightCaptureRuntimeIntegrity.md](../../NsightCaptureRuntimeIntegrity.md)
- [TracyGpuProfiling.md](../../TracyGpuProfiling.md)
