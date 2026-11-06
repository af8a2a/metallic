---
id: "0064"
title: "Nsight: Source/IL 无头导出仍无验证后端，源码分析依赖交互 UI"
area: tooling
kind: research
priority: P2
state: blocked
milestone: M5-Research
labels: ["area:tooling", "kind:research", "priority:P2", "state:blocked"]
evidence:
  - Documentation/NsightHeadlessExportResearch.md:5
  - Documentation/NsightHeadlessExportResearch.md:15-16
  - Documentation/NsightHeadlessExportResearch.md:75-82
  - Documentation/AgenticShaderOptimizationRoadmap.md:7
---

## 背景 / Context

Agentic Shader 优化的首个可交付里程碑是「agent 能选中生产 shader、通过 Nsight 取得
可复核的源码热点并说明数据范围」。仓库已经完成了 CSV importer、结构化热点查询与
最小目标选择，瓶颈被明确收敛到「CSV 的生产端」。

## 现状 / Current state

- 调研结论："已找到公开的非交互采集和 metrics 自动导出能力；**未找到本机版本公开、
  受支持的 Source/IL CSV 无头导出接口**"，因此「不能承诺把 `profiledata.csv` 的
  生产环节改成纯 CLI」。
- 能力表把链路拆成三段：CLI/SDK 采集（有公开入口）、metrics 自动导出
  （`--auto-export` 官方支持）、**按 shader/时间范围导出 Source/IL（未找到公开命令
  或 SDK API）**；仓库 importer 这一段「已不需要 Nsight UI；瓶颈在 CSV 生产端」。
- 路线图确认 M1 现状为 partial：「可复用 adapter、重新打开/选择及回放异常恢复仍为
  partial」，UI 导出是当前唯一可用的生产通路（三次实际 UI 导出 3/3 通过）。
- 阻塞条件可持续验证：安装包 SDK 为 NsightGraphicsSDK 0.9.2，公开头文件只有
  `collectShaderInfo` / `collectExternalShaderDebugInfo` 等采集开关，没有读取分析
  结果的接口；`ngfx-rpc.exe` 的存在不能推出可维护的公开分析 API，逆向也不作为
  里程碑前提。

## 问题或缺口 / Problem

1. **无法自动化**：源码热点分析的每一次生产都要人工点 UI 导出，agent 闭环
   （路线图的 M3「候选自动判定」）因此在第一步就依赖人。
2. **验收口径容易滑坡**：`--auto-export` 只导出 metrics，把它当成源码分析能力会
   制造「看起来全自动」的假结论。文档已划清三种性质——不需要人工点击 / 不启动
   Nsight GUI 进程 / 不依赖交互桌面会话——但还没有一次端到端运行把三者分开记录。
3. 调研列出的 4 项最小后续实验（有界 GPU Trace + `--auto-export` +
   `--real-time-shader-profiler`、归档与判定、检查导出内容是否含 Source/IL、
   真离线三连导出）**尚未执行**，因此严格 headless 的能力状态仍是未知而非否定。

## 期望结果 / Expected outcome

- 执行那 4 项最小实验并把结论按「采集 / 导出 / 查询」三段分别记录下来；若确认只有
  指标表，则维持「严格无 UI 的 Source/IL 自动生产缺少已验证后端」，并把 UI adapter
  明确标注为仍依赖交互桌面。
- 对外暴露能力状态（`verified / supported-unverified / unsupported / blocked`），
  使调用方无法把 UI 导出路径误认为 headless。

## 验收标准 / Acceptance criteria

- [ ] 归档一次有界 GPU Trace 实验：独立 trace 文件、完整参数与日志、全部导出文件
      及其 hash，全部落在证据目录（禁止放进源码树）；记录新建进程与窗口清单。
- [ ] 对该导出做内容判定并留结论：是否包含模块身份、Source/IL 行、self samples、
      stall 类别与 scope；仅含指标表时记录版本与配置级缺口。
- [ ] 若发现离线入口，完成「关闭 Nsight UI，从同一归档 trace 重新导出三次」并核对
      shader/range 身份、结构化结果与错误退出；否则以厂商问题清单收尾。
- [ ] 结论写入 `Documentation/NsightHeadlessExportResearch.md` 的能力表，并在
      `metallicctl` 能力状态（见 0063）中体现为 `blocked` 及原因。
- [ ] `MetallicNsightSourceTests` / `MetallicNativeNsightSourceTests` 在无 UI 环境下
      的通过/失败含义在文档中写明：`ctest -R '^Metallic(Native)?NsightSourceTests$' --output-on-failure`。

## 相关文档 / References

- [NsightHeadlessExportResearch.md](../../NsightHeadlessExportResearch.md)
- [AgenticShaderOptimizationRoadmap.md](../../AgenticShaderOptimizationRoadmap.md)
- [NsightUIExportValidation.md](../../NsightUIExportValidation.md)
- [AgenticShaderOptimizationM1.md](../../AgenticShaderOptimizationM1.md)
