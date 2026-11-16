# 运行与恢复

所有命令从已确认的 Metallic checkout 执行。下面是当前实现的命令模板，先检查
所选 runner 的 `--help` 和对应 `.md`。替换示例路径；输出目录必须是新的。
仅做离线复核时无需构建、SDK 或重新采集。

## 环境发现与前置检查

从当前项目配置、可用工具和已有证据解析环境，不假设盘符、用户名、安装位置或
历史构建目录。下面命令中的变量须先解析为当前任务的实际值，不能原样照抄运行：

- `$casePath`、`$assetsManifest`：目标 workload 与已核验的资产身份清单。
- `$buildDir`、`$sampleExe`：兼容的 Release 构建目录及其实际产物。
- `$candidatePath`：本轮声明的候选；`$outputDir`：本轮新的输出目录。
- `$evidenceDir`：需离线复核或恢复的既有证据包。
- `$nvperfSdkRoot`：仅在使用 NvPerf 时发现并核对的配套 SDK 根目录。

优先复用匹配的 case、构建和证据；工具路径从当前安装发现，MSVC 环境使用项目
规定的开发者 shell。历史报告用于理解限制，不充当当前基线或默认复测任务。

检查 CMake cache 对应当前 source root、Release 和目标 EXE。变更 C++ 后构建，
只查历史结果时不构建。使用已初始化的 VS x64 shell；若从 PowerShell 启动 `.cmd`
初始化脚本，初始化与 CMake 必须在同一个 cmd 进程，环境不会回传父 PowerShell。
必要时在 `build/` 生成短构建脚本并检查退出码，不改全局编译器配置。
Python live runner 需要 `psutil`。检查依赖后再安装确实缺少的依赖，不重装 Nsight。
GPU 命令权限不足时对有界命令走环境提供的审批机制，不以修改 ACL、禁用安全设置
或切换不明桌面会话规避限制。

## 可信工作负载与 A/A

```powershell
python -B Tools/Perf/WorkloadCase.py run --case "$casePath" --assets "$assetsManifest" --exe "$sampleExe" --output "$outputDir" --runs 3 --timeout 240
python -B Tools/Perf/WorkloadCase.py verify "$outputDir"
```

先查可复用的合格 case 和资产 manifest。需要重新生成 manifest 时先读
`Tools/Perf/WorkloadAssets.py --help` 与 `WorkloadCase.md`，评估对应资产范围；原始流程
可能顺序读约 360 GB。不要为一次已有包的 verify 生成新 manifest。
`AA.json` 的 stable 仅表示 A/A 门槛通过，不接受任何候选。

## 候选优化闭环

优化前先按 [Nsight 驱动的优化闭环](nsight-guided-optimization.md) 建立瓶颈假设，
候选后关联同范围 GPU Trace、计数器和 Profiler 对比。以下是 WorkControl 的普通
计时入口；其他 pass 选其对应 runner，不将本入口视为通用 shader 捕获器。

先读 `ExperimentRunner.md` 和候选 JSON schema。此 runner 当前仅允许
`Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang` 的明确文本替换。
候选记录假设、当前 baseline SHA-256、旧/新文本及预期匹配次数。不能盲改旧 hash
使过时的候选通过。多文件/C++/质量设置候选需要扩展契约，不能伪装成受支持候选。

```powershell
python -B Tools/Perf/ExperimentRunner.py run --case "$casePath" --candidate "$candidatePath" --assets "$assetsManifest" --exe "$sampleExe" --build-dir "$buildDir" --output "$outputDir"
python -B Tools/Perf/ExperimentRunner.py verify "$outputDir"
```

默认 3 个 discovery ABBA block，有希望的候选再运行 5 个独立 confirmation block，
每进程限时 300 秒；确认当前默认值，在采集前固定两阶段预算并说明实验规模。
当前规则不设 3% 最低收益：discovery 的软件总时间收益点估计严格大于 0，且没有
已确认的目标退化或超预算 graph 退化，即进入 confirmation；只有独立 confirmation
的收益 95% 区间下界严格大于 0、graph 收益下界至少 -2% 才接受。
例如确认区间为 [0.2%, 0.8%] 的小幅收益可以接受；[-0.2%, 0.8%] 仍为
inconclusive；零收益不能接受。正确性、身份、恢复和证据完整性要求仍须通过。
exit 0 可以是 accept 或 reject，exit 2 是 inconclusive；读取 `Decision.json`、
确认阶段和事务恢复状态，不能只看 exit code。

样本预算不足以分辨小收益时保留候选和全部结果，并说明测量精度限制。后续实验应
先声明预算、可检验的改动或噪声来源处置；不自动追加样本或只保留显著批次。
旧证据按归档时的政策复核，不用新规则改写历史 verdict。其他 pass 使用其对应
runner 和证据契约；有明确非劣效预算的 overhead case 不等同于正收益优化。
历史 `Candidate.CompactWorkVertices.json` 已拒绝，不作为默认优化配方。

## NvPerf：生产 dispatch 计数

确认 SDK 完整且为配套发行版本。当前 CMake 检查 `NvPerf/include`、
`redist/NvPerfUtility/include` 与 `NvPerf/bin/x64/nvperf_grfx_host.dll`。
必要时在 VS shell 中执行：

```powershell
cmake -S . -B "$buildDir" -DMETALLIC_ENABLE_NVPERF=ON "-DMETALLIC_NVPERF_SDK_ROOT=$nvperfSdkRoot"
cmake --build "$buildDir" --target MetallicGPUDrivenSample --config Release
python -B Tools/Perf/NvPerf.py run --case "$casePath" --assets "$assetsManifest" --exe "$sampleExe" --build-dir "$buildDir" --output "$outputDir"
python -B Tools/Perf/NvPerf.py verify "$outputDir"
```

runner 自行清理继承的 `METALLIC_*` 环境并设置所需开关，串行启动三次独立进程，
默认每进程 240 秒。不要在外面再注入 ngfx。默认测 `gpu__time_duration.sum` 和
`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`。有明确诊断问题时用
`--metrics <JSON数组文件>` 指定 1–16 个精确名称；availability 与 pass 计划决定
支持性，不猜 metric alias，不替换成语义不同的相近指标。

每次查看 `app/nvperf/NvPerf.json`、`Capture.json`、`Process.json`、stderr，以及
`CounterAvailability.bin`、`ConfigImage.bin`、`CounterDataPrefix.bin`、`CounterDataImage.bin`。
整批查看 `Summary.json` 与 `Manifest.json`。将 verify 输出放在包外，避免改变
manifest 的精确文件清单。SDK 2025.1 本机 Vulkan 1.4 的正式支持标志曾为 false；
升级后重新验证，不静默忽略，也不把旧警告当作新版本必然失败。

## 编译资源与 Nsight 阶段诊断

生产寄存器、共享内存和 binary size 使用独立 resources 批次，事先构建 Release：

```powershell
python -B Tools/Perf/DeepProfile.py run --mode resources --case "$casePath" --assets "$assetsManifest" --exe "$sampleExe" --build-dir "$buildDir" --output "$outputDir"
python -B Tools/Perf/DeepProfile.py verify "$outputDir"
```

明确需要 A/B 编译资源比较时加入已声明的 `--candidate`。核对
`PipelineStatisticsBinding` 的 input/device SPIR-V、cacheKey 与 shader/entry 映射，
不把旧日志的 spirv 缓存键当原始 SPIR-V hash，也不把 Local Memory sentinel 当 spill。

只有问题需要更宽的阶段活动时选择 `--mode triage`。从本机 ngfx 能力取得路径、
architecture 和 metric-set，再按 `DeepProfile.md` 运行。默认保持 clocks unaltered，
不把不同 clocks 策略批次合并。选择完整 Stream early/late 层级；原生表可能为 TSV
但扩展名为 `.xls`。两个同名列含义不明时保留列号与原值，不任选或相加。
通用 software triangles 不等于 WorkControl；此模式也不完成 Source/IL 导出。

## MCP 诊断与实验衔接

实际优化按 [Nsight 驱动的优化闭环](nsight-guided-optimization.md) 完成前后对比与
联合判定。下面仅列接口选择与证据限制；不要以能力发现代替真正读取计数器和热点。

先发现本机官方 Native 的实际 Schema；提供对应工具时，已有 `.ngfx-capture` 用
`graphics_debugging_replay.open_capture` 离线检查元数据、事件和 Shader，已有
`.ngfx-gputrace` 用 `gpu_trace.open_trace` 直接分析，无需先导出 `.xls`。
只有旧 CLI 时使用其已支持的查询/导出；所需接口缺失则报告证据缺口，不转为 CUA。
仅需已有文件的查询时，不重新采集或启动目标。

- Native 使用渐进发现：先读 family/group/tool details 的实际 Schema，再调用对应
  工具。2026.4 的初始五个入口不表示其余工具缺失，也不证明旧安装具备这些接口；不要将 Native 与 Workflow
  的工具名、ID 和会话语义混用。可用工具与字段以当前安装包实际返回为准。
- Graphics Capture 的纹理、Buffer、几何读取及回放 GPU 计时等需要在线回放时，
  先完成 `go_online` 及其返回的确认流程，取得实际输出后再判断可回放性。
  内嵌的 capture 截图不证明在线输出正确。GPU Trace 已采集的时间线/计数器使用
  trace 查询，无需为此启动 Graphics Capture 在线回放。
- 由 MCP 负责外部注入 Metallic 时，不再传入应用的 `--nsight-capture` 自注入开关。
  只有需要源码关联/Shader 性能归因时才按当前编译入口启用
  `METALLIC_SHADER_CAPTURE_SYMBOLS=1`；保留优化，不用 unoptimized shader-debug 计时验收。
- `get_shader_profile(query="function_scope_tree")` 可读 Top-down/Bottom-up 函数及内联
  作用域的 Self/Total PC samples 和 stall；`line_performance` 支持 `source` / `intermediate`
  逐行归因。编译内联和调试关联会影响归因，样本不等于调用次数或独立函数耗时。
- `get_shader_analysis(query="instruction_mix")` 是 Shader 级类别统计，不能直接按函数
  筛选，也不能把其 count 当动态执行次数。本机 Public Schema 未开放 SASS 查询；
  `sass_counters.available=false` 时把相关数值保留为缺失，类别名称为空时保留原始 ID，
  不猜测 ALU/TEX 等标签。不要从工具名或 UI 能力推断 MCP 也能导出完整 SASS。
- 将诊断发现映射到当前生产 Shader、entrypoint、绑定 SPIR-V、dispatch 和具体范围，
  再按已声明候选进入 `WorkloadCase` / `ExperimentRunner` 的正确性与 A/B 验收。
  MCP 的回放 Shader 编译/替换不能绕过候选、事务和恢复流程；一次回放或 profiler
  诊断不能证明普通运行收益，也不能代替这些 runner 的接受判定。

## 中断、失败与恢复

保留失败包。先检查 `Process.json`、stdout/stderr、SDK 状态与 artifact freshness；
定位超时、未知 metric、overflow、缺失范围、身份漂移或资产不匹配，不通过放宽
阈值来修复数据。修正已知原因后用新目录重跑。

`build/shader-experiment.lock` 存在时检查 owner PID/输出目录与真实进程，活跃 owner
不得并行启动新实验。对 M3/resources 的 `Transaction.json`，owner 已停止后使用：

```powershell
python -B Tools/Perf/ExperimentRunner.py recover "$evidenceDir"
```

未知当前 shader 字节必须保留并人工对齐，不能直接复制 baseline 覆盖。NvPerf-only
不修改 shader，也没有恢复事务，不能盲套 recover。其陈旧锁需确认 owner 已退出、
输出确属该次 NvPerf 运行且没有 shader 事务后，才清理该锁，并保留失败记录。

## 修改工具后

只在修改 runner/backend 时执行相关测试，而不是每次读报告都测全仓库：

```powershell
cmake -S tests/perf -B build/perf-tests
ctest --test-dir build/perf-tests -C Debug --output-on-failure
```

编译测试与合成测试不能代替真实 GPU 验收。变更 SDK 集成时分别验证 ON/OFF 构建、
无 SDK 请求的明确失败、错误指标拒绝，以及真实三进程采集，测试后报告配置状态。
