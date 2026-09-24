# M0 收尾：真实 CSV 验证与 UI 导出验收

2026-09-24。**CSV 检查与真实格式解析通过；重跑三次实际自动 UI 导出 3/3 通过，M0 收尾完成。** [最新验收](NsightUIExportValidation.md)记录同一固定报告的三个独立保存、选择截图、文件哈希和结构化比较。重新打开/重新选择与性能有效性不在本次通过范围。

初次输入为用户提供的 `Captures/NsightGraphics/profiledata.csv`，以下 614 Samples 等结果仍描述该原件。重跑从同目录 capture `MetallicGPUDrivenSample_2026_09_24_21_22_39.ngfx-capture` 采集新报告，核对了 capture → shader → Aggregate/Entire Trace 范围，三份新导出各为 4,290 Samples。原件的历史时间范围没有被追溯证明。

## 新数据检查结果

| 项目 | 结果 |
| --- | --- |
| CSV 大小 / SHA256 | 1,785,487 bytes / `c14f9367f825029169ecabc115cdf8356a0ceee0d505278df01c2014056869bc` |
| 原始记录 | 26,518；源码数值行 65，IL 数值行 26,413 |
| 模块 | `comp.10000.spv (4d0378657da7b7d1)` |
| SPIR-V 入口 | `OpEntryPoint GLCompute %streamClusterRasterWorkControlMain "main"`，原 CSV 记录 87 |
| IL 明确报告的 Samples | 614，26 个非空 Samples 行 |
| 源码明确报告的 Samples | 614；与 IL 不相加 |
| 嵌入源码镜像 / 引用 | 35 / 1；保留定位，不重复计入 IL |
| 最热 IL 行 | 24316，原 CSV 记录 24398，502 Samples，Barrier 502 |
| Barrier 下界 | 517；所有 top-three 原因合计下界 604 |
| 最大 Live Registers | 34；不等于分配寄存器数或 occupancy |

源码第 63 行的 `Total Samples=5165`、`Samples=6`，第 19 行为 380 / 8。解析器分别保存它们，不将 Total Samples 自动重命名为 inclusive，也不把它们加入 IL Samples。空白单元格保持未知；614 是明确报告值的和，不表示每个空白行均被测得零。

该文件是 WorkControl shader 的新证据，不能套用历史分类 shader 的 10,310 / 28 预期。旧 `streamClusterBinMain.csv` 未恢复，不影响本轮使用独立新基线。

## 原生格式实现与使用

新增 [NsightShaderCsv.py](../Tools/Perf/NsightShaderCsv.py)，通过 [NsightSource.py](../Tools/Perf/NsightSource.py) 的 `native-import` 接入已有查询接口。

支持范围限定为本次真实导出的两种精确表头、短文件/模块标记、数值行、SPIR-V entry 声明与嵌入源码注释。未知表头、损坏列数、冲突重复行和不完整 stall 对会失败。源码关联要求单个导出模块和唯一匹配的导出内源码引用；多模块/多入口歧义不取第一个。Vulkan 入口 `main` 与函数符号分别记录。

```powershell
python -B Tools/Perf/NsightSource.py native-import --raw Captures/NsightGraphics/profiledata.csv --output build/nsight-source-new
python -B Tools/Perf/NsightSource.py hotspots build/nsight-source-new --module "comp.10000.spv (4d0378657da7b7d1)" --entry main
python -B Tools/Perf/NsightSource.py source build/nsight-source-new --module "comp.10000.spv (4d0378657da7b7d1)" --entry main --record 19
```

输出目录须为新目录。CSV 本身没有 capture hash、Nsight 版本或实际 timeline 范围；未提供的上下文保持未知，`compare-repeats` 拒绝用这些不完整上下文验收三次导出。

## Capture 元数据

文件大小 3,135,729,096 bytes，SHA256：
`a04d1365bac634bfb5686e79d6fd7abbec8fedbcee9ad925205bb8fbf667d219`。

官方 metadata 命令读取成功，记录 capture 版本 2026.3.1/build 38722833、RTX 5070 Ti、driver 616.92、frame 37。元数据中保留了 capture 2026.3.1 与 replayer 2026.3 的版本兼容警告，以及 VK_NV_low_latency 扩展版本提示。没有运行性能重放；元数据成功不证明重放、捕获画面正确性或当前 CSV 的实际选区。

## 初次 UI 阻断记录（已恢复）

本轮 `@oai/sky` 初始化/`sky.list_windows()` 在 Node kernel 启动阶段失败。显式 reset 后重试仍报：

```text
windows sandbox failed: helper_unknown_error: apply deny-read ACLs
```

本轮两个失败进程为 21996 / 27400，均未取得窗口列表，UI 输入次数为零，自动导出完成次数为零。失败转录位于 [ui-probe.json](../build/perf-m0-closeout-20260924/ui-probe.json)，明确标注为工具结果转录，不冒充原始进程日志。cua-child 不是前置条件，此故障也不是自动审批拒绝。

[computer-use 技能](C:/Users/11252/.codex/plugins/cache/openai-bundled/computer-use/26.917.71314/skills/computer-use/SKILL.md)要求使用其运行时；其 [恢复指引](C:/Users/11252/.codex/plugins/cache/openai-bundled/computer-use/26.917.71314/docs/guidance.md)包含“retry once, then stop and report”。本轮在重置后仍启动失败，采用该有限恢复原则停止重复探测，没有绕过运行时或修改 Windows 安全设置。

后续已修复 sandbox ACL 状态文件损坏，原生运行时恢复。重跑实际操作三次 Export/Save，生成 `export-01.csv`、`export-02.csv`、`export-03.csv`，全部为 1,786,525 bytes，SHA256 均为 `f09f29110217236cdb60027572ca27c133cfb8d4465aa204972f65dbe3d2b1f2`。原生解析、完整性和结构化比较均通过，见 [Validation.json](../build/nsight-ui-exports-20260924-01/Validation.json)。初次错误证据保留，不再作为当前阻塞。

此次 Nsight Live Replay 出现访问冲突和断连；已加载的新报告可离线导出。该异常限制性能基线与无人值守可靠性结论，不影响三次保存同一报告的字节一致性验收，详见 [范围与限制](NsightUIExportValidation.md)。

## 验证与证据

- [真实字节夹具](../tests/perf/fixtures/NsightSourceReal.csv)：11,392 bytes / 143 条记录，包含完整源码表、所有有 Samples 的 IL 行、入口和注释；[来源映射](../tests/perf/fixtures/NsightSourceReal.provenance.json)保留原始 SHA256、记录和物理行，未修改单元格。
- [11 项真实格式测试](../tests/perf/TestNativeNsightSource.py) + 17 项映射契约测试 + 10 项基线测试，CTest 3/3 通过；[日志](../build/perf-m0-closeout-20260924/CTest.log)。
- [最终解析包](../build/perf-m0-closeout-20260924/source-final/manifest.json)、[热点](../build/perf-m0-closeout-20260924/hotspots.json)、[源码上下文](../build/perf-m0-closeout-20260924/source-context.json)。
- [收尾基线](../build/perf-m0-closeout-20260924/baseline/Baseline.Closeout.json)完整性通过；初始收集结果保留在同目录 `Baseline.json`。
- [机器可读收尾摘要](AgenticShaderOptimizationM0Closeout.json)。

官方 [Shader Profiler 文档](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)说明采样位置可能在等待依赖的消费者处，且 Selected 并非 stall。因此本报告仅完成证据/格式检查，不据这些样本直接提出“可加速百分比”。
