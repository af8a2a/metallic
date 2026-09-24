# M1：Nsight 源码分析进度

2026-09-24 最新进展：真实 Source/IL parser 与夹具已验证；38 项测试通过。**同一固定报告的三次自动 UI 导出 3/3 通过，字节与结构化结果一致。** Capture、目标 module/entry 与 Aggregate/Entire Trace 范围已核对，见 [导出验收](NsightUIExportValidation.md)。M1 仍为 partial：可复用 UI adapter、重新打开/重新选择验证及回放异常恢复尚未完成。

以下为新 CSV 到达前的首次 M1 历史记录；其中原生格式、运行时阻塞及 UI 尚未操作的描述已由上述验收更新，不代表当前状态。

按用户最新要求，cua-child 不再是前置条件。后续可直接通过 computer-use 的窗口级接口操作 Nsight；当前没有控制过 Nsight UI，也没有产生新的 GPU 数据。

## 已落地

[离线工具](../Tools/Perf/NsightSource.py)提供 `inspect / import / verify / shaders / hotspots / source / compare-repeats`，[使用与数据契约](../Tools/Perf/NsightSource.md)给出了命令和格式约束。

- 原始 CSV、布局、上下文、解析结果与解析器均有 SHA256；查询前核验原始证据并重新解析。输出目录不可覆盖。
- 显式映射多段表头及重复列名，保留引号内换行、CSV 记录号和物理行范围。布局绑定原始文件 hash，禁止猜测列名或单位。
- module/entry 唯一选择；IL 与源码表示分离，self/inclusive/dependency samples 分离。相同逻辑行的副本保留多份定位但不重复计数，冲突则失败。
- 空单元格保留 unknown，不改为零；Live Registers 不解释为分配寄存器数；top-k stall 原因保留下界标志。
- 源码上下文只读导出物中实际出现的源码，不使用当前工作树补齐历史源文件。声明的 requested/achieved scope 不一致则失败。
- 三份导出物可以比较结构化结果，但相同 export ID 被拒绝，比较通过也不证明实际 UI 导出独立完成。
- 基线收集默认采用 desktop 后端声明，child 改为可选；任何后端就绪都不等于源码导出已验证。

当前是**显式布局导入框架**。合成测试不代表 Nsight 原生格式；尚不能声称直接支持远端历史 CSV。第一版要求数据行中存在 module/entry 列；如果真实导出用段落标记表达身份，需拿到原件后实现有证据约束的格式适配。依赖视图和 IL→source 映射也未实现。

## 本轮真实阻塞

1. **原始源码 CSV 在远端**。用户已确认本地没有原件。历史派生报告保留的 SHA256 是
   `13222ffc71e5172a028a0b33e2cce19562d7bf185d500fcf642a914bf6cedf02`。
   尚不能复现目标模块 10,310、另一模块 28 个 IL self samples；没有从派生 JSON 合成原始证据。
2. **原生桌面运行时启动失败**。本轮发现 `mcp__node_repl__js`，尝试初始化 `@oai/sky` 并执行 `sky.list_windows()`；第一次进程退出，重试与 reset 后重试均在 Node kernel 启动阶段报：
   `windows sandbox failed: helper_unknown_error: apply deny-read ACLs`。
   没有获得窗口列表，也未执行 UI 输入。此故障与 cua-child 无关，不是 Nsight 导出不支持的证据，也不是自动审批拒绝。

依照已读取的 [computer-use 技能](C:/Users/11252/.codex/plugins/cache/openai-bundled/computer-use/26.917.71314/skills/computer-use/SKILL.md)及其 [guidance](C:/Users/11252/.codex/plugins/cache/openai-bundled/computer-use/26.917.71314/docs/guidance.md) 的恢复步骤（“reset the JavaScript session … retry once, then stop and report”）停止重复 UI 尝试。技能对超时规定了有限恢复；本轮启动崩溃采用同样的有限恢复策略，没有绕过运行时调用隐藏 helper。

## 验证与验收边界

`tests/perf/TestNsightSource.py` 的 17 项测试全部使用明确标记的合成契约数据，覆盖多模块、两个表示层、嵌入换行、重复列、重复/冲突行、空值、范围错配、未知入口、源码隔离、原始 hash、文件篡改、CLI 错误和三次离线比较。M0 的 10 项测试继续通过。两组测试通过独立 CTest 接入，不要求渲染器构建。

| M1 退出条件 | 当前结果 |
| --- | --- |
| 专用、可追溯的离线查询接口 | 已实现映射框架；原生 Nsight dialect 未验证 |
| 固定 artifact → workload/queue/range → 唯一 shader | UI 未实现/未验证；离线唯一 module/entry 检查通过 |
| summary / IL / 源码真实导出及符号关联 | 缺原始文件与可用 UI 运行时 |
| 自动完成三次导出并取得一致结构化结果 | 未执行；只有离线比较能力 |
| 混模块、缺身份、范围错配、陈旧文件失败可诊断 | 离线契约测试通过；无法检测虚假的 UI 范围声明 |

## 继续推进顺序

1. 同步远端原始 CSV 和它对应的 capture/trace、版本、选择范围与 shader 身份。先核对历史 hash；新导出则建立独立预期。
2. 用真实原件补齐原生 dialect、module/entry 段落映射和去重规则，归档经许可的小型真实夹具。历史原件应复现 10,310 / 28；新证据不套用旧值。
3. 在原生桌面运行时可用后，用已观察到的 Nsight 窗口验证打开 artifact、选择范围和 shader、summary/source 导出及符号。每步保存实际选择状态，失焦/空表/错误 shader 不复用旧结果。
4. 对同一固定 artifact 重新打开并导出三次。记录每次实际导出日志与范围，调用 `compare-repeats`；只有 UI 行为与原始证据均通过，才将 M1 标为完成。

官方能力依据：[Shader Profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)说明表格 CSV 与源码视图操作；[GPU Trace CLI](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)的 metrics 自动导出不能证明源码/依赖视图有无头接口。本轮没有逆向私有 capture/trace 格式。
