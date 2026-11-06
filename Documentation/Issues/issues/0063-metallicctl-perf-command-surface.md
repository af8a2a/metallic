---
id: "0063"
title: "Tooling: metallicctl 缺少 perf 命令面与能力状态，实验链固定在 Tools/Perf 脚本"
area: tooling
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:tooling", "kind:feature", "priority:P2"]
evidence:
  - Documentation/AgenticShaderOptimizationRoadmap.md:193-205
  - Source/Tools/MetallicCtl.cpp:271-284
  - Source/Runtime/Debug/DebugCore.cpp:395
---

## 背景 / Context

Agentic Shader 优化路线把「引擎控制面」定位为有限的、有类型的 perf 操作入口：
外部 runner 负责耗时编排与构建，引擎只接收有界且可校验的 case/采集请求。为此设计
了一组 `metallicctl perf ...` 命令与四态能力标记。

## 现状 / Current state

- 路线图第 7 节明确写着「以下均为拟新增 Metallic 命令，**不是现有 CLI**」，
  包括 `perf capabilities / import / case create / collect / analyze / experiment`。
- 实际 CLI 帮助里没有任何 `perf` 子命令：`metallicctl` 只支持
  `list | hello | schema | frame latest | rg graph | rg trace`、
  `capture batch|export`、`probe`、`watch ...`、`shader capabilities|sites|watch`、
  `inspect buffer|texture`、离线 `eval|stats|shader verify`。
- `DebugCore` 对外方法表同样没有 perf 相关方法，能力状态只有
  `shader.capabilities`（VVL Printf 侧），完整的
  `verified / supported-unverified / unsupported / blocked` 语义没有对应出口。
- 实验与分析能力实际分散在 `Tools/Perf/` 的 100+ 个 Python 脚本里
  （`WorkloadCase.py`、`ExperimentRunner.py`、`NsightSource.py`、`NsightShaderCsv.py`、
  `NvPerf.py`、`DeepProfile.py` 等），通过文件约定而非协议互相衔接。

## 问题或缺口 / Problem

1. **没有能力发现入口**：一个 agent 无法问「这台机器上 NvPerf 可用吗、Source/IL
   导出走到哪一步、哪个 case 已注册」。只能读一堆 Markdown 报告推断，这与路线图
   自己强调的「任何一项成功都不能自动推导其他项成功」相冲突。
2. **证据链靠脚本约定**：case 身份、证据包 schema、判决规则都在 Python 侧，
   引擎无法校验请求是否越界，也无法拒绝不完整请求；`metallicctl` 已有的有界协议
   （预算、错误码、stale 语义）没有被复用。
3. **不可审计**：同一实验用不同脚本组合产生不同文件布局，复核者难以判断某份
   证据是否按声明的门槛生成。

## 期望结果 / Expected outcome

- 在 `metallicctl` 增加最小 `perf` 子命令面：`perf capabilities --json`、
  `perf case create/list`、`perf collect`、`perf analyze`，复用现有传输、预算与
  错误码；构建、进程编排与 Nsight 调度继续留在引擎外。
- `perf capabilities` 输出每项能力的四态、工具版本、验证 artifact 与原因，
  且明确区分 capture / metrics export / shader summary / source hotspots /
  dependencies / 反汇编。

## 验收标准 / Acceptance criteria

- [ ] `metallicctl --help` 列出 `perf` 子命令，
      `metallicctl --pid <PID> --json perf capabilities` 返回逐项能力与状态。
- [ ] `perf case create --workload GPUDriven.Raster.Early.Software --state frozen`
      能在客户端侧留下带 case id 的 manifest，且引擎拒绝越界/未注册请求时返回既有
      错误码（`Unsupported` / `BudgetExceeded` / `StaleHandle`）。
- [ ] `MetallicDebugTests` 新增协议用例覆盖 perf 请求的边界与错误路径；
      `ctest --test-dir build-ci -R '^MetallicDebugTests$' --output-on-failure` 通过。
- [ ] 至少一条既有 Python 证据链（例如 `Tools/Perf/WorkloadCase.py` +
      `ExperimentRunner.py`）改为经 `metallicctl perf` 调用并保持输出 schema 等价，
      由 `ctest -R '^Metallic(WorkloadCase|ExperimentRunner)Tests$'` 验证。
- [ ] 路线图第 7 节从「拟新增」更新为实现状态与剩余缺口。

## 相关文档 / References

- [AgenticShaderOptimizationRoadmap.md](../../AgenticShaderOptimizationRoadmap.md)
- [DebugControlPlane.md](../../DebugControlPlane.md)
