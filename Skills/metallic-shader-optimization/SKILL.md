---
name: metallic-shader-optimization
description: "Run Metallic's evidence-driven shader optimization workflow on Windows: qualify production workloads, test declared shader candidates with A/B measurements, and diagnose WorkControl dispatches with headless NvPerf or pipeline resource statistics. Use for Metallic GPU optimization, NvPerf capture, experiment verification, and interrupted experiment recovery; not generic Nsight UI/source export or CPU profiling."
---

# Metallic Shader Optimization

把优化假设变成可复核的 GPU 实验。使用 Metallic 仓库内维护的 runner，不复制实现，
不把这份 skill 的历史路径、指标或验收结果当成当前实测。

## 选择入口

先定位当前 Metallic checkout；本机已知路径为 `E:/metallic`。读取其 `AGENTS.md`，
检查工作区状态，只读取任务所需的仓库文档。缺少仓库时先定位或询问路径，
不要自动 clone 另一个项目。命令、支持范围以当前仓库代码和 `--help` 为准。

| 当前请求 | 入口及必读仓库文档 |
|---|---|
| 验证已有证据、解释结果 | 对应 runner 的 `verify`；不自动启动新 GPU 实验 |
| 新场景/配置的可信度、A/A | `Tools/Perf/WorkloadCase.py`、`WorkloadCase.md` |
| 测试一个具体 shader 候选 | `Tools/Perf/ExperimentRunner.py`、`ExperimentRunner.md` |
| WorkControl early/late 硬件指标、无 UI 采集 | `Tools/Perf/NvPerf.py`、`NvPerf.md` |
| 实际寄存器/共享内存等编译资源 | `Tools/Perf/DeepProfile.py run --mode resources`、`DeepProfile.md` |
| 需要 Nsight 阶段级计数器 | `DeepProfile.py run --mode triage`、`DeepProfile.md`；先核实本机 ngfx 能力 |
| 中断后 shader 恢复 | 有事务 journal 时用 `ExperimentRunner.py recover` |

读取 [运行与恢复](references/workflows.md) 中所选模式的段落。只执行请求需要的模式，
不要因为存在 M0–M4 就全部重跑。一般优化请求但尚无候选时，先检查已有证据，
提出可检验的单一假设，再选择必要诊断或实验。不要自动复测历史上已拒绝的示例候选。
本 skill 不要求 `cua-child`，上述 runner 不使用分析器 UI。源码/IL/SASS 导出需求
另查仓库 `Documentation/` 中 Nsight source 工作流；NvPerf 不是该导出的替代品。

## 实验的不变量

- 保持注册 case 的相机、历史 priming、内部渲染尺寸、质量、队列和驻留规则。
  生产 shader 的 module/entry、实际绑定 SPIR-V、软件列表、间接参数与 readback
  共同建立身份；宽 marker 或缓存键不能代替 shader 身份。
- 普通计时与 diagnostic 分开。NvPerf、GPU Trace、Graphics Capture、RenderDoc、
  validation、pipeline statistics 不可混入普通计时验收；NvPerf 不与其他诊断叠加。
- 使用新输出目录，GPU 实验串行且有超时。保留原始文件、日志、进程状态、配置、
  工具和二进制哈希。不在实验期间修改源文件/资产或运行另一个实验。
  合作锁不证明系统级 GPU 独占；只清理本次拥有的进程，不关闭用户其他程序。
- 优先复用已验证的资产 manifest 并检查对应 case 与 metadata。发生资产变动或
  迁移才建立新内容身份；不要因文件存在就当它有效，也不要每次默认重哈希数百 GB。
- A/B 用 runner 的进程级统计、正确性和确认阶段，不把帧数当独立实验数。
  不降低门槛、不反复重跑直到显著。观察、解释和候选决策分别报告。
- 候选由事务临时安装；accept/reject/inconclusive 后都核对 baseline 恢复。
  `accept` 不授权永久应用、提交或发布。未知并发修改保留现场，不能覆盖用户编辑。

## NvPerf 证据边界

当前 backend 是 Windows x64、Vulkan graphics queue、primed WorkControl 目标帧的
两个 command range，单 pass。它需要真实 GPU 与可用图形会话；隐藏窗口、无 Nsight UI
不等于无桌面服务或 isolated replay。使用同一套 SDK 的头文件和库。

采集成功需完整 pass、零丢失、恰好 early/late 两个范围、所请求指标及 SDK 单位齐全，
三次独立运行身份与 depth/visibility 一致。逐项报告重复性；不稳定数据保留为不稳定。
未知/不可用指标、缺失范围、多 pass 或 overflow 不得用零值或旧文件补齐。
多 pass 在恢复所有相关可写状态的契约建立前明确拒绝。

SM active 不等于 occupancy，编译资源不等于运行时 stall，stall 占比不等于可回收时间。
NvPerf 诊断时长不能直接作为候选加速。`verify` 当前重核哈希、readback 与 JSON 统计，
不重新解码 CounterDataImage；保留 `counterImageReevaluated=false` 的说明。
保留 SDK/Vulkan 兼容性标志和警告，不把本机成功改述为官方认证。

## 交付与停止

运行后用当前 verifier 复核，禁止执行证据包中归档的脚本。链接结果、原始证据和日志，
报告 case/范围、指标及单位、各次值/波动、判定、恢复状态和实际限制。
说明没有跑过的层级：MiniZorah 不等于 Full Zorah，readback 等价不等于参考渲染器正确性，
in-frame 不等于 isolated，诊断成功不等于接受优化。

完成所需验收后停止。失败先保存现场并定位原因；有具体修复才在新目录重跑。
未改变的失败或支持边界应明确报告，不自动扩大到驱动重装、系统时钟修改或漫长全套跑测。
