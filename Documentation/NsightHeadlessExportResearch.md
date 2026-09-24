# Nsight Source/IL CSV 无头导出调研

2026-09-24；目标环境：Windows、Nsight Graphics 2026.3.1（build 38722833）、Vulkan / RTX 5070 Ti。问题是能否替代当前 Shader Profiler Source/IL 视图的 UI Export，而不只是将点击操作包进一个 CLI。

**结论：已找到公开的非交互采集和 metrics 自动导出能力；未找到本机版本公开、受支持的 Source/IL CSV 无头导出接口。** 当前不能承诺把 `profiledata.csv` 的生产环节改成纯 CLI。这个结论来自本机帮助、SDK 头文件、现有封装与官方文档核对，不是对所有内部接口的“不可能性证明”。本轮没有运行新的 GPU 采集，也没有完成 Source/IL 无头导出的运行时验证。

## 能力边界

| 目标 | 调研结果 | 对 Metallic 的意义 |
| --- | --- | --- |
| CLI 启动、Graphics Capture、回放 | 有公开入口 | 已有编排可以继续复用 |
| GPU Trace 非交互采集 | `ngfx.exe` 支持 GPU Trace 活动；SDK 提供采集控制 | 可独立推进采集自动化 |
| GPU Trace metrics 自动导出 | `--auto-export` 有官方文档与本机帮助 | 是指标/事件表路径；不能据此推导 Source/IL 导出 |
| 源码相关采样和符号收集 | `--real-time-shader-profiler` 及 shader/debug-info 收集配置存在 | 能收集相关数据不等于能离线取出全部分析结果 |
| 已保存 trace → 指定 shader、时间范围 → Source/IL CSV | 核对范围内未找到公开命令或 SDK API | 严格 headless 的源码诊断链仍有缺口 |
| 已有 Source/IL CSV → 结构化热点/源码查询 | 仓库 importer 已验证 | 这一段已不需要 Nsight UI；瓶颈在 CSV 生产端 |

验收时分别记录三种性质：**不需要人工点击**、**不启动 Nsight GUI 进程**、**不依赖交互桌面会话**。第一项通过不能代替后两项通过。隐藏窗口也不能证明后两项；目标应用本身是否需要窗口是另一个约束。

## 证据

### 本机 CLI

原始输出保存在 `build/nsight-headless-research-20260924/`，执行记录见 `probes.json`；调用均为帮助查询，没有采集副作用：

| 调用 | 实际返回的相关能力 | 未发现的目标入口 |
| --- | --- | --- |
| `ngfx.exe --help-all` | GPU Trace、`--auto-export`、`--real-time-shader-profiler`、采集/符号配置 | 对既有报告按 shader/range 导出 Source/IL 的选项 |
| `ngfx-capture.exe --help` | Graphics Capture 采集 | Shader Profiler 报告导出 |
| `ngfx-replay.exe --help` | 回放、metadata/functions/objects、截图、回放性能报告 | PC sample 与源码/IL 关联结果导出 |
| `ngfx-rpc.exe --help` | Replayer UI Server；transport、pipe、port 等连接配置 | 公开的源码分析/导出方法或协议 schema |

`ngfx.exe --help-all` 返回码为 1，但输出了完整帮助文本；不能把这个返回码记成 GPU 采集失败。其余三个帮助调用返回 0。`ngfx-replay --present-hidden` 是窗口呈现选项，`--metadata-objects` 是对象元数据，均不等价于 Source/IL CSV。

`ngfx-rpc.exe` 的存在意味着安装包有 UI 服务组件，不能直接推出存在可维护的公开分析 API。当前没有把内部 RPC 逆向作为可用方案，也没有向其发送请求。

### 本机 SDK 与 CLI 封装

核对的是安装包中的 **NsightGraphicsSDK 0.9.2**，不是只看旧版 Injection SDK。路径：

`C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/SDKs/NsightGraphicsSDK/0.9.2/include/`

- `NGFX_GPUTrace_Common.h` 提供 `NGFX_GPUTrace_GetTraceFileCount`、`GetTraceFilePath`、`WaitForTraceFilePath`；Vulkan 接口提供注入、初始化、激活和开始/停止采集。
- `NGFX_GPUTrace_Common_Types.h` 包含 `traceShaderBindings`、`collectShaderInfo`、`collectExternalShaderDebugInfo`。这些控制采集内容；获取 trace 路径不等于读取分析结果。
- 所检查的公开头文件未找到按 shader/时间范围读取源码采样表或导出 Source/IL CSV 的接口。[官方 SDK 指南](https://docs.nvidia.com/nsight-graphics/UserGuide/sdk.html)同样以程序化采集为主；在线指南与本机 SDK 小版本不同，实际接入以安装头文件为准。

本机 `cli-anything-nsight-graphics` 0.2.0 的 `core/gpu_trace.py` 处理 `FRAME.xls`、`GPUTRACE_FRAME.xls`、`D3DPERF_EVENTS.xls`，另有可选 `GPUTRACE_REGIMES.xls`。它封装厂商采集命令并解析 metrics 表，没有新增 Source/IL 导出后端。给现有命令加包装不能消除缺失的厂商接口。

历史目录 `build-relwithdebinfo/minizorah-nsight-20260913/gpu-trace-clean/BASE_UNLOCKED` 中确有上述表及 `REPRO_INFO.xls`。这只是历史 metrics 自动导出的旁证；当时配置与本轮源码采集不同，不能用这些目录中没有 Source CSV 来证明所有配置都不支持。

### 官方文档

- [GPU Trace CLI](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)：`--auto-export` 的承诺是 metrics 导出；源码采样开关说明数据可用于 profiler 视图，没有给出 Source/IL 离线导出命令。
- [通用 CLI 参数](https://docs.nvidia.com/nsight-graphics/UserGuide/launch-application-overview.html)：提供启动、活动选择、输出目录等选项，未列出该报告转换入口。
- [Shader Profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)：描述 shader 表、源码/低层视图及 UI 操作。CSV 的 Source/IL 实际格式已有本地真实导出证明，但文档没有提供等价 headless API。
- [Graphics Capture CLI](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)：采集、回放和元数据能力不能直接替代源码采样分析。
- [2026.3 发布说明](https://developer.nvidia.com/nsight-graphics/get-started)：集成在 Graphics Debugger 中的 GPU Trace 分析直接展示结果，不生成独立 GPU Trace 文件。这与本轮三次 UI 导出的来源一致；现有 `.ngfx-capture` 不能当作已归档的 `.ngfx-gputrace` 分析报告。

## 路线调整

**保留源码分析接口，分开验收采集、导出与查询。** M0 的真实 CSV 与三次 UI 保存结论不变；它们不证明 headless 能力。M1 仍需补可复用 adapter，并显式暴露导出后端与桌面依赖。若要求严格无 UI，应把 Source/IL 自动生产标为缺少已验证后端，而不是把 UI adapter 包成 CLI 后宣布完成。

| 路径 | 适用目标 | 决策 |
| --- | --- | --- |
| CLI/SDK 采集独立 GPU Trace + 官方 metrics 自动导出 | 非交互采集、可归档报告、常规指标 | 优先验证；与 Source/IL 导出分开验收 |
| Source/IL UI adapter + 现有 importer | 当前可获得的源码热点、stall 与源码关联 | 暂时保留；明确仍依赖 UI |
| NVIDIA 提供离线导出/API | 严格无 UI 的 Source/IL 分析 | 需要厂商确认或新增公开接口；当前不承诺实现日期 |
| Nsight Perf SDK / NvPerf | 应用内 Vulkan 计数器、range 指标与自动实验 | 可选后端；不是现有 Source/IL CSV 的直接替代 |
| 逆向 trace 或内部 RPC | 探索内部实现 | 本轮未做；未形成可维护方案，不作为里程碑前提 |

[Nsight Perf SDK](https://developer.nvidia.com/nsight-perf-sdk)公开支持应用内 Vulkan 指标采集、自定义触发/输出和持续集成场景。它可减少常规测量对桌面工具的依赖，但所核对的公开资料没有证明能提供与当前 Nsight Source/IL CSV 等价的逐源码行采样与 stall 关联。不能为绕过 UI 而悄悄降低数据语义。

## 最小后续实验与退出条件

以下是待执行实验，不是本轮已完成结果：

1. 用有界 workload，经 `ngfx` 的 GPU Trace 活动启用 `--auto-export` 与 `--real-time-shader-profiler`；保留 shader pipelines、bindings 和调试信息收集。使用本机支持的 architecture/metric-set，设置超时和短采集上限，不沿用禁用 shader pipelines 的旧模板。
2. 归档独立 trace、完整参数/日志、全部导出文件与 hash；记录新建进程和窗口，分别判定“无点击”与“无 GUI 进程”。无交互桌面环境要单独验证，不能从隐藏窗口推断。
3. 检查自动导出是否包含模块身份、Source/IL 行、self samples、stall 类别及 scope。如果只有指标表，则记录该版本与该配置的运行时缺口；若发现额外源码文件，交给现有 native importer 核验后再提升能力状态。
4. 只有发现真正的离线导出入口后，才做关闭 Nsight UI、从同一归档 trace 重新导出三次的验证，并核对 shader/range 身份、结构化结果和错误退出。独立重新采集的样本值允许变化，不要求其 CSV 字节一致。

向 NVIDIA 核实的具体问题可直接采用：**Nsight Graphics 2026.3.1 Windows/Vulkan 是否有受支持的 CLI 或 SDK，能在不启动 GUI、无需交互桌面的情况下，从已保存的 GPU Trace 按 shader module/entry、队列与时间范围，导出 Source/IL 行级 self samples、stall 和符号关联？是否支持集成 Graphics Capture GPU Trace 的结果？** 本轮没有向外部发送此问题。

严格 headless 的退出条件是“厂商接口存在 + 当前机器端到端实测通过”，而不是帮助中出现了 `--auto-export`。
