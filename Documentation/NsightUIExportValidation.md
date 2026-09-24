# Nsight 三次自动 UI 导出验证

2026-09-24：**3/3 通过**。使用 `@oai/sky` 实际操作 Nsight 的 Profiler Shader Source → Export → Save CSV file，分别生成三个新文件。三份文件原生格式解析、包完整性检查、SHA256 和结构化结果比较均通过。

验证范围是**同一已加载报告、相同选择状态下的三次独立导出操作**。本轮没有重新打开报告三次，没有验证三次 GPU 采样稳定性，也没有完成可复用无人值守 adapter。

## 输入、选择和采集

- Capture：`Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_24_21_22_39.ngfx-capture`，3,135,729,096 bytes；本轮重新计算 SHA256 为 `a04d1365bac634bfb5686e79d6fd7abbec8fedbcee9ad925205bb8fbf667d219`。
- Nsight UI：2026.3.1.0 / build 38722833；RTX 5070 Ti，driver 616.92。
- Graphics Debugger 集成 GPU Trace，报告会话时间 22:07:42；Top-Level Triage，实时 Shader Profiler 与 Trace Shader Bindings 开启，Max Duration 1000 ms，Lock to Base，Warp State Samples Per PM Interval 10。
- Profiler Overview：Entire Trace，显示范围 0.00–11.44 ms。这是 UI 舍入值，不是假定的精确时间戳。
- Source view：选择 `GPUDrivenStreamWorkRaster.slang`，Region=`Aggregate`，Languages=`Slang and SPIRV`；右侧模块 `comp.10000.spv (4d0378657da7b7d1)`。Shader Pipelines 按该 hash 筛选，入口 `main`；CSV 中 `OpEntryPoint` 确认 `%streamClusterRasterWorkControlMain`。
- Device=`Vulkan - 181789 Samples`。未应用队列专属筛选；可见 `Vulkan Graphics Q:0` 轨道。验收范围为 `shader-in-window`，不声明 dispatch 隔离。

[范围截图](../build/nsight-ui-exports-20260924-01/trace-range.png)、[采集设置](../build/nsight-ui-exports-20260924-01/gpu-trace-settings.png)、[trace 信息树](../build/nsight-ui-exports-20260924-01/trace-information-tree.txt)已保存。集成报告未归档成独立 `.ngfx-gputrace` 文件；以 capture hash、会话和导出物关联本次验证。

## 三次实际保存

| 文件 | 本地写入时间（UTC+8） | 大小 | 选择 / 保存证据 |
| --- | --- | --- | --- |
| [export-01.csv](../build/nsight-ui-exports-20260924-01/export-01.csv) | 22:11:27 | 1,786,525 bytes | [选择](../build/nsight-ui-exports-20260924-01/selection-01.png) / [保存](../build/nsight-ui-exports-20260924-01/save-01.png) |
| [export-02.csv](../build/nsight-ui-exports-20260924-01/export-02.csv) | 22:12:22 | 1,786,525 bytes | [选择](../build/nsight-ui-exports-20260924-01/selection-02.png) / [保存](../build/nsight-ui-exports-20260924-01/save-02.png) |
| [export-03.csv](../build/nsight-ui-exports-20260924-01/export-03.csv) | 22:12:57 | 1,786,525 bytes | [选择](../build/nsight-ui-exports-20260924-01/selection-03.png) / [保存](../build/nsight-ui-exports-20260924-01/save-03.png) |

每次均打开新的 Save CSV file 对话框，输入独立路径并保存；没有复制 CSV 充当导出。三份 SHA256 均为：

`f09f29110217236cdb60027572ca27c133cfb8d4465aa204972f65dbe3d2b1f2`

每份 CSV 有 26,518 条记录、65 条源码数值行、26,413 条 IL 数值行，未映射非空记录为 0。源码与 IL 明确报告的 Samples 各为 4,290，两层不相加。最热 IL 文本行 24316 报告 3,437 Samples，其中 Barrier 3,431；这些数值仅作为导出内容的核对指纹，不推导性能收益。

三份独立 context 使用不同 export ID，固定相同 capture hash、module/entry、会话与选择。`native-import` 和 `verify` 全部通过；[compare-repeats](../build/nsight-ui-exports-20260924-01/comparison.json) 返回 `structured_results_equal=true`。该纯文件比较器仍返回 `automation_verified=false`，因为文件相等本身不证明 UI 操作。UI 操作验收由本轮工具执行与截图证据单独记录在 [Validation.json](../build/nsight-ui-exports-20260924-01/Validation.json)。

## 限制与后续

初次 Live Replay 发生 `Fatal Exception: EXCEPTION_ACCESS_VIOLATION`，随后离线重连。后续 Live Replay 的 GPU Trace 显示采集成功并加载报告，但加载后又出现 `Replay connection lost (code: 0, status: 0)`；再次离线重连后完成三次导出。[首次异常](../build/nsight-ui-exports-20260924-01/replay-access-violation.png)与[报告加载后的断连](../build/nsight-ui-exports-20260924-01/trace-loaded-replay-lost.png)均保留。

Capture/UI 的 2026.3.1/2026.3 版本字符串提示及 VK_NV_low_latency 扩展版本提示仍存在。未验证实际重放画面正确性；本报告只接受导出与解析一致性，不将该次采集作为合格的性能优化基线。

用户原 `profiledata.csv` 的 614 Samples 属于另一份报告。本次 4,290 是新采集结果，两者不能用于直接比较性能。原 CSV 的历史选区没有被本轮追溯证明。

M0 能力与证据基线的真实格式和三次导出检查已收尾。M1 仍需实现可复用 UI adapter、重新打开/重新选择的重复验证、选择状态失配检测，以及回放异常的诊断和恢复。现有依赖视图、完整低层反汇编等未验证能力继续保留原状态。
