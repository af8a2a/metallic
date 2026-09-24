# Agentic Shader 优化 M0：能力与证据基线

2026-09-24 最新收尾：**新 CSV 与真实 Source/IL 格式验证通过，三次实际自动 UI 导出 3/3 通过，M0 收尾完成。** 三次独立保存来自同一固定报告，字节和结构化结果一致。回放访问冲突与断连已记录，性能有效性和 M1 无人值守恢复不在本次通过范围。详见 [三次导出验收](NsightUIExportValidation.md)、[收尾报告](AgenticShaderOptimizationM0Closeout.md)与[最新机器摘要](AgenticShaderOptimizationM0Closeout.json)。

以下保留首次 M0 的历史记录；其中“缺失 CSV / child worker”描述的是首次收集时的状态，不代表最新进展。

**交付物**

- [基线工具](../Tools/Perf/Baseline.py)和 [使用说明](../Tools/Perf/README.md)：自动发现工具、固定 SHA-256、保存原始命令输出、归档小型证据；拒绝复用已有输出目录。
- [Baseline schema](../Tools/Perf/Baseline.schema.json)：区分能力的 verified / supported-unverified / unsupported / blocked，文件的 missing / empty / present / hash-mismatch / unreadable。
- [本轮机器可读摘要](AgenticShaderOptimizationM0.json)和 [完整基线](../build/perf-m0-20260924-01/baseline-final/Baseline.json)：39 项文件，包含工具、当前源码快照、历史 Trace/capture/表格、子会话探测日志与 case。
- [最小历史 case](../build/perf-m0-20260924-01/baseline-final/case.json)：legacy SW，归因范围包含相邻 HW/SW；queue/dispatch/source hash 仍未知，禁止用于当前性能验收。
- [真实夹具与来源](../tests/perf/fixtures/Provenance.json)：完整 D3DPERF_EVENTS 文本及 GPUTRACE_REGIMES 的表头/第 34 行，合计约 175 KiB；保留字节、重复列名、原始 SHA-256 和行号。
- [九项完整性测试](../tests/perf/TestBaseline.py)：已接入 CTest，可以独立于引擎配置运行。

**本机结果**

| 项目 | 状态 | 验证范围 |
| --- | --- | --- |
| GPU / driver | RTX 5070 Ti / 616.92 | 本机查询结果，不与远端 RTX 5060 数据混用 |
| Nsight | 2026.3.1，unified+split | doctor info/versions 成功；四个 Nsight 程序的 SHA-256 已记录 |
| wrapper | 0.2.0 | 记录入口、包信息和 Python 模块摘要 |
| CLI discovery | verified | 仅安装与选项发现 |
| Graphics Capture / GPU Trace metrics export | supported-unverified | 本轮未执行新采集，历史成功不能替代本机当前验证 |
| Shader Summary / Hot Spots / dependencies | supported-unverified | 支持依据已登记，本机导出与关联尚未验证 |
| 完整低层反汇编 | supported-unverified | 发行版相关能力未确认 |
| 源码 UI 自动导出 | blocked | cua-child 初始化失败 |

本机 ngfx 的 --help-all 输出完整帮助但返回 1，原始返回值未改写，也未作为采集成功证据。rg 查找指定原件/临时解析器返回 1，表示没有匹配。schema 和完整性通过只证明格式与已有证据哈希满足检查。

**原始证据与缺口**

用户确认 streamClusterBinMain.csv 在远端。历史报告记录的预期 SHA-256 为：

```text
13222ffc71e5172a028a0b33e2cce19562d7bf185d500fcf642a914bf6cedf02
```

历史目标模块是 comp.10000.spv (ef24b8b4bc22971f)，目标 IL self samples 为 10,310，另一模块为 28。这些是原报告的预期，**本轮未从原始 CSV 重新计算**。历史临时 AnalyzeBinProfile.py 也未在本机找到。不能从分析 JSON 反向制造一个“原始 CSV”。

已归档 9 月 20 日 Full legacy SW Trace 的五张表、ProfileReady、配置、run manifest 和日志。其驱动为 616.64，Git 为 f734cac1e41c0f02a4e600ded63743cd6f6856cd；该范围包含相邻 HW/SW。当前代码、驱动和生产 WorkControl shader 不能视作同一实验。

真实小样本是 GPU Trace TSV，**不是 Shader Profiler 源码/IL CSV**。它们用于证据归档和后续范围表解析回归，不能替代源码 importer 的 golden fixture。

两个原有文件为零字节，已标为 empty，未删除：

- MetallicGPUDrivenSample_2026_09_21_23_29_42.ngfx-capture
- MetallicGPUDrivenSample_2026_09_21_23_30_25.ngfx-capture

**Child Session 探测**

在 C:/Users/11252/Documents/Codex/tools/cua-child-mcp/cua-child.exe 找到程序。status 返回 childSessionId=null、workerReady=false。通过用户指定的 cua-child.exe mcp --timeout 25 做了一次初始化探测，退出码 1，报告：

> Child worker did not become ready.

[原始错误](../build/perf-m0-20260924-01/cua-child-initialize.stderr.txt)已归档。工具提示首次启用后的 Windows 登录状态可能相关，但本轮没有确认根因。没有自动登录、修改策略、注销 Windows 或操作主桌面。恢复 worker 是 M1 UI 自动导出的前置条件；文件 importer 可在远端 CSV 同步后独立推进。

**验证与复现**

九项测试覆盖：缺失/空文件/hash 不符，原件移除后归档仍可校验，归档与日志篡改可检出，失败命令不能支持 verified，verified 必须有 scope 和已知证据，目录不能复用，超时保留失败，空清单不能通过，真实夹具的哈希/重复列/marker。部分测试包含多项断言。

独立 CTest 通过；实际 Baseline.json 通过 JSON Schema 和完整性校验。只修改工具、测试入口、小样本及文档，已有 Streamer 改动未修改。

```powershell
python -B Tools/Perf/Baseline.py --verify build/perf-m0-20260924-01/baseline-final/Baseline.json
cmake -S tests/perf -B build/perf-m0-tests
ctest --test-dir build/perf-m0-tests -C Debug --output-on-failure
```

采集新基线使用新目录；原件到位后可用 --shader-csv 指定，检查上述历史 hash：

```powershell
python -B Tools/Perf/Baseline.py --output build/perf-m0-next --source-location remote-only --child 'C:/Users/11252/Documents/Codex/tools/cua-child-mcp/cua-child.exe'
python -B Tools/Perf/Baseline.py --output build/perf-m0-with-source --shader-csv '<local-original-csv>' --child 'C:/Users/11252/Documents/Codex/tools/cua-child-mcp/cua-child.exe'
```

不同的新导出可以用 --new-export 归档，但会保留 provenance 待验证项，需要关联其 capture、shader 身份、选区和环境，不能继承旧样本预期。

**退出条件**

| 条件 | 状态 |
| --- | --- |
| 工具/GPU/driver 固定与原始发现结果归档 | 完成 |
| 区分支持声明、本机验证和阻塞 | 完成 |
| 现有原始 Trace/表格可定位且哈希正确 | 完成，仅代表历史证据完整性 |
| 最小 case 与真实小样本 | 完成，GPU Trace 样本 |
| Shader Profiler 原始 CSV 与源码样本 | 待远端原件同步或独立重导出 |
| 子会话自动导出路径 | worker 待恢复，M1 前置依赖 |

M0 不标为完成。下一步是同步真实 CSV 后完成格式夹具及 importer，再在可用的 Child Session 验证自动导出。
