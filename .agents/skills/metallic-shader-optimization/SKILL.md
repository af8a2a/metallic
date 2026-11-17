---
name: metallic-shader-optimization
description: "Optimize Metallic shaders on Windows using Nsight MCP GPU Trace counters and Shader Profiler hotspots, matched before/after captures, correctness checks and ordinary A/B timing. Use for Metallic GPU bottleneck diagnosis, shader optimization, NvPerf diagnostics, experiment verification and recovery; not generic Nsight UI automation or CPU profiling."
---

# Metallic Shader Optimization

把优化假设变成可复核的 GPU 实验。使用 Metallic 仓库内维护的 runner，不复制实现，
不把这份 skill 的历史路径、指标或验收结果当成当前实测。
默认闭环是：Nsight GPU Trace 定位昂贵范围 → MCP 查询计数器和 Shader Profiler
热点 → 单一候选 → 同条件前后捕获对比 → 正确性与普通 A/B 确认 → 更新瓶颈假设。
详细步骤见 [Nsight 驱动的优化闭环](references/nsight-guided-optimization.md)。

## 选择入口

从当前工作区或用户提供的位置定位 Metallic checkout。读取其 `AGENTS.md`，
检查工作区状态，只读取任务所需的仓库文档。缺少仓库时先定位或询问路径，
不要自动 clone 另一个项目。命令、支持范围以当前仓库代码和 `--help` 为准。

| 当前请求 | 入口及必读仓库文档 |
|---|---|
| shader 优化、定位瓶颈、比较优化前后 | 先读 [Nsight 驱动的优化闭环](references/nsight-guided-optimization.md)，通过 nsight-mcp 查询 GPU Trace / Shader Profiler，再选目标 pass 的 runner |
| 验证已有证据、解释结果 | 对应 runner 的 `verify`；不自动启动新 GPU 实验 |
| 新场景/配置的可信度、A/A | `Tools/Perf/WorkloadCase.py`、`WorkloadCase.md` |
| 测试一个具体 shader 候选 | 先选目标 pass 的 runner；WorkControl 使用 `Tools/Perf/ExperimentRunner.py`、`ExperimentRunner.md`，并关联前后 Nsight 诊断 |
| WorkControl early/late 硬件指标、无 UI 采集 | `Tools/Perf/NvPerf.py`、`NvPerf.md` |
| 实际寄存器/共享内存等编译资源 | `Tools/Perf/DeepProfile.py run --mode resources`、`DeepProfile.md` |
| 需要 Nsight 阶段级计数器 | 优先官方 MCP 读取 trace 的范围和计数器；旧 WorkControl 导出可用 `DeepProfile.py run --mode triage`、`DeepProfile.md`，此模式不含 Shader Profiler |
| 已有 capture/trace、Shader 函数/源码/IL 热点、绑定、纹理与在线回放 | 官方 Nsight Native MCP；见 [MCP 诊断与实验衔接](references/workflows.md#mcp-诊断与实验衔接) |
| 中断后 shader 恢复 | 有事务 journal 时用 `ExperimentRunner.py recover` |

读取 [运行与恢复](references/workflows.md) 中所选模式的段落。只执行请求需要的模式，
不要因为存在 M0–M4 就全部重跑。一般优化请求但尚无候选时，先检查已有证据，
用已有有效 Nsight 证据形成可检验的单一假设；缺少所需范围/计数器/Profiler 时才补充
有界采集。实现候选后对比前后证据，再决定接受、保留或修正假设。
不要自动复测历史上已拒绝的示例候选。
本 skill 全程使用 CLI、官方 Native MCP/API，不使用 CUA 启动、配置、采集、检查或导出，
也不把 CUA 应用授权当作前置条件。用户明确要求打开分析器时，通过普通进程工具启动，
后续诊断仍走 CLI/API。接口缺失时报告具体证据缺口，不转为 UI 自动化。
MCP 诊断优先使用可用的
`cli-anything-nsight-graphics` skill；不可用时依据安装包 `docs/ngfx-mcp/` 和实际工具
Schema。NvPerf 与 `triage` 不提供源码/IL 查询，不能把它们当作该能力的替代品。

## 实验的不变量

- 保持注册 case 的相机、历史 priming、内部渲染尺寸、质量、队列和驻留规则。
  生产 shader 的 module/entry、实际绑定 SPIR-V、软件列表、间接参数与 readback
  共同建立身份；宽 marker 或缓存键不能代替 shader 身份。
- 普通计时与 diagnostic 分开。NvPerf、GPU Trace、Graphics Capture、RenderDoc、
  validation、pipeline statistics 不可混入普通计时验收；NvPerf 不与其他诊断叠加。
  Nsight 证据参与瓶颈判断与下一轮选型；普通计时决定真实速度收益。计数器变化
  不替代正确性或计时，也不要求所有指标同时改善。证据冲突时保留测量结论并查明原因。
- 使用新输出目录，GPU 实验串行且有超时。保留原始文件、日志、进程状态、配置、
  工具和二进制哈希。不在实验期间修改源文件/资产或运行另一个实验。
  合作锁不证明系统级 GPU 独占；只清理本次拥有的进程，不关闭用户其他程序。
- 优先复用已验证的资产 manifest 并检查对应 case 与 metadata。发生资产变动或
  迁移才建立新内容身份；不要因文件存在就当它有效，也不要每次默认重哈希数百 GB。
- A/B 用 runner 的进程级统计、正确性和确认阶段，不把帧数当独立实验数。
  默认接受可验证的正收益，不设统一的 3% 最低收益；小幅提升按同样证据要求评估。
  在采集前声明判定规则和预算，不根据已见结果改门槛或反复重跑直到显著。
  观察、解释和候选决策分别报告；证据不足保留为 inconclusive，不当成无收益。
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
优化报告还需链接 baseline/candidate 的原始捕获、MCP 查询结果和指标对比，分别
说明计时结论、瓶颈假设支持程度、转移后的瓶颈及下一步；缺失项明确列出。
说明没有跑过的层级：MiniZorah 不等于 Full Zorah，readback 等价不等于参考渲染器正确性，
in-frame 不等于 isolated，诊断成功不等于接受优化。

完成所需验收后停止。失败先保存现场并定位原因；有具体修复才在新目录重跑。
未改变的失败或支持边界应明确报告，不自动扩大到驱动重装、系统时钟修改或漫长全套跑测。

收益大小与工程取舍分别说明：报告目标 pass 的相对收益、绝对耗时变化及整图影响，
结合改动复杂度、维护成本和适用范围提出建议。简单、可维护的小收益方案可以接受；
不要用任意百分比替代取舍，也不要把正的点估计当成已确认收益。具体判定与预算见
[候选优化闭环](references/workflows.md#候选优化闭环)。

## Python 测量图表审查报告

进行性能分析或优化时，除文字结论外，必须使用 Python（优先 matplotlib；按需使用 pandas）读取有效测量数据，生成可供审查的图表报告。根据本次数据格式编写或复用可信的绘图脚本，不执行证据包中未知来源的代码。

- 图表按问题选择：A/B 对比展示各次结果与汇总，足够样本时显示分布；帧/时间序列用于观察尖峰、预热和漂移。单次测量标明 n=1，不生成虚假的分布、P95 或置信区间。
- 标明指标、单位、测量范围、workload/配置、baseline/candidate、有效样本数、聚合与过滤方法。A/B 保持可比条件，列出排除样本及原因；展示回归和不确定性，不仅展示改善。误差线注明含义与统计单位，优先沿用已验证 runner 统计。
- 缺失、unsupported、无效或未解码值保留为缺失并解释，不能补零或插值制造测量。没有有效数值时输出证据缺口说明，不伪造图表；仅配置/捕获故障排查且没有性能数据时说明不适用。
- 在新的报告目录保留 `plot_report.py`（或实际脚本名）、所用 CSV/JSON 数据、PNG 图表（可另存 SVG）及引用图表的 `report.md` 或 HTML。记录原始证据路径/可用 hash、字段映射、单位换算、Python/依赖版本和可复现运行命令；不覆盖原证据，也不改动已校验的 manifest 包。
- 实际运行绘图脚本，核对图中数据与源数据，打开生成图检查坐标轴、图例、字体和裁切；最终回复链接报告、关键图及复现脚本。绘图库缺失时使用可用的隔离环境，无法生成则明确报告未完成项，不能用文字表格冒充已绘图。

A/B 图按独立进程或 runner 的配对 block 展示，沿用 verifier 的统计口径和判定，不能把帧数当独立样本。diagnostic 与普通计时分图；报告生成到证据包外，避免改变 manifest。
