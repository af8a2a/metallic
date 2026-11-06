---
id: "0030"
title: "Tooling: Nsight 抓取子进程处于 Low 完整性级别，报告与场景缓存写入失败"
area: tooling
kind: task
priority: P1
state: open
milestone: M3-Hardening
labels: ["area:tooling", "kind:task", "priority:P1"]
evidence:
  - Documentation/NsightCaptureRuntimeIntegrity.md:1-49
  - Tools/Perf/VirtualShadowMapGPUProfile.py:186-190
---

## 背景 / Context

Windows 把一个新进程创建在「用户完整性级别」与「可执行文件完整性标签」中较低的那个。
2026-11-01 的 VSM 抓取修复中发现，Release 产物继承了 `Low Mandatory Level:(NW)` 标签：
`ngfx.exe` 是 Medium，但真正被启动的 `MetallicRHITests.exe` 子进程是 Low，于是它既写不出
HTML 报告，也写不出 `StreamerSubsystem` 的场景流式 source-identity 文件。场景缓存根在
`PROJECT_SOURCE_DIR/.cache/scene-streams`，因此改工作目录或 `TEMP` 都无法规避。

## 现状 / Current state

- 复现路径清楚：直接用真实的 native zero-test GTest 启动即可复现报告写入失败，不需要
  跑 GPU 用例；`icacls` 看启动器或 shell 会漏掉真正子进程的 `TokenIntegrityLevel`，
  Python 写探针也会漏（解释器可执行文件没有 Low 标签）。
- 修复方式是把 EXE 与同目录 DLL 的字节复制到一个「Medium 执行上下文新建的目录」，校验
  源/暂存 SHA-256 后，用该目录运行抓取；不能复制旧文件的安全描述符，否则问题重现。
- 该修复目前只以本地产物存在（`build/vsm-nsight-repair-20261101/` 的
  `StageRuntime.py` / `NativeWriteProof.py` / `CaptureAB.py` 与
  `Captures/NsightGraphics/VSMRelativeMappingMedium-20261101/` 的有效 A/B trace），
  仓库里没有任何探测或拒绝逻辑。
- 抓取流程的仓库内工具只做**证据完整性**检查（`Tools/Perf/VirtualShadowMapGPUProfile.py:186`
  读的是采集产物里的 `Integrity.json` / `Processes.json`，与 Windows 完整性级别无关），
  既不校验子进程的 `TokenIntegrityLevel`，也不校验暂存哈希。

## 问题或缺口 / Problem

这是一处「验证基建缺陷」，不是渲染缺陷：Low 子进程产出的 trace 缺失场景编译结果，
导出的 UI-only trace 是无效证据，但它与成功抓取在文件名/退出码上几乎不可区分。该失败
直接导致一次 VSM 寄存器/占用率归因无法完成，并可能让其他依赖 Nsight 的验收（VSM 命中、
采集对比）得到不可信结论。

## 期望结果 / Expected outcome

把「暂存 + 校验」写进仓库内可重复执行的抓取前置步骤：检测目标 EXE 的完整性标签，必要时
用字节复制方式在 Medium 上下文重建运行目录，校验全部文件哈希，并把记录（源/暂存哈希、
host 与 target token 完整性级别、Nsight 与驱动版本）写进抓取 manifest；不满足条件时以
明确错误失败，而不是产出一个半成品 trace。

## 验收标准 / Acceptance criteria

- [ ] 抓取前置检查能读出 **实际被 Nsight 启动的子进程** 的 `TokenIntegrityLevel`；为 Low
      时以明确错误终止，不生成看似成功的 trace。
- [ ] 暂存运行目录中每个 EXE/DLL 的 SHA-256 与当前构建一致，并写入抓取 manifest。
- [ ] 一次完整抓取在 Medium 子进程中完成：能写出 HTML 报告与 `.cache/scene-streams` 的
      场景 source-identity 文件，且捕获包含请求的渲染负载（非 UI-only）。
- [ ] 用同一 manifest 里的路径执行 `ngfx-replay --present-hidden --loop-count 3` 退出 0。
- [ ] 文档记录该失败模式（子进程而非启动器的完整性级别）与恢复步骤，避免复现时误判为
      「场景编译失败」。

## 相关文档 / References

- [NsightCaptureRuntimeIntegrity.md](../../NsightCaptureRuntimeIntegrity.md)
- [NsightCommandBufferCaptureInvestigation.md](../../NsightCommandBufferCaptureInvestigation.md)
- [NsightGPUTraceStartupInvestigation.md](../../NsightGPUTraceStartupInvestigation.md)
