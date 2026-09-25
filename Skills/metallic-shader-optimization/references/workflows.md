# 运行与恢复

所有命令从已确认的 Metallic checkout 执行。下面是当前实现的命令模板，先检查
所选 runner 的 `--help` 和对应 `.md`。替换示例路径；输出目录必须是新的。
仅做离线复核时无需构建、SDK 或重新采集。

## 本机线索与前置检查

2026-09-25 验证时的本机线索，使用前重新检查存在性和实际版本：

- checkout：`E:/metallic`。
- Release：`build-release/Source/MetallicGPUDrivenSample.exe`，单配置 CMake/Ninja。
- case：`Tools/Perf/WorkloadCase.MiniZorahHistory.json`，非零 early/late WorkControl。
- 资产清单：`build/m2-assets-20260925/mini.json`，不含完整资产副本。
- SDK 压缩包：`D:/NVIDIA_Nsight_Perf_SDK_2025.1_Public_Windows.zip`；解压目录：
  `build/nvperf-sdk-2025.1`。不把 SDK 复制进 skill 或提交仓库。
- MSVC 初始化：`E:/VisualStudio/VC/Auxiliary/Build/vcvars64.bat`。
- 已验证 NvPerf 包：`build/nvperf-20260925-02`；报告：
  `Documentation/AgenticShaderOptimizationNvPerf.md`。这是历史证据，不是当前性能基线。

检查 CMake cache 对应当前 source root、Release 和目标 EXE。变更 C++ 后构建，
只查历史结果时不构建。使用已初始化的 VS x64 shell；若从 PowerShell 启动 `.cmd`
初始化脚本，初始化与 CMake 必须在同一个 cmd 进程，环境不会回传父 PowerShell。
必要时在 `build/` 生成短构建脚本并检查退出码，不改全局编译器配置。
Python live runner 需要 `psutil`。检查依赖后再安装确实缺少的依赖，不重装 Nsight。
GPU 命令权限不足时对有界命令走环境提供的审批机制，不以修改 ACL、禁用安全设置
或切换不明桌面会话规避限制。

## 可信工作负载与 A/A

```powershell
python -B Tools/Perf/WorkloadCase.py run --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --assets build/m2-assets-20260925/mini.json --exe build-release/Source/MetallicGPUDrivenSample.exe --output build/workload-new --runs 3 --timeout 240
python -B Tools/Perf/WorkloadCase.py verify build/workload-new
```

先查可复用的合格 case 和资产 manifest。需要重新生成 manifest 时先读
`Tools/Perf/WorkloadAssets.py --help` 与 `WorkloadCase.md`，评估对应资产范围；原始流程
可能顺序读约 360 GB。不要为一次已有包的 verify 生成新 manifest。
`AA.json` 的 stable 仅表示 A/A 门槛通过，不接受任何候选。

## 候选优化闭环

先读 `ExperimentRunner.md` 和候选 JSON schema。当前仅允许
`Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang` 的明确文本替换。
候选记录假设、当前 baseline SHA-256、旧/新文本及预期匹配次数。不能盲改旧 hash
使过时的候选通过。多文件/C++/质量设置候选需要扩展契约，不能伪装成受支持候选。

```powershell
python -B Tools/Perf/ExperimentRunner.py run --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --candidate build/candidate-new.json --assets build/m2-assets-20260925/mini.json --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release --output build/experiment-new
python -B Tools/Perf/ExperimentRunner.py verify build/experiment-new
```

默认 3 个 discovery ABBA block，通过后再运行 2 个 confirmation block，每进程限时
300 秒；确认当前默认值，提前说明长实验的规模。保持原门槛：软件总时间收益区间
下界至少 3%，graph 收益下界至少 -2%。exit 0 可以是 accept 或 reject，exit 2 是
inconclusive；读取 `Decision.json`、确认阶段和事务恢复状态，不能只看 exit code。
历史 `Candidate.CompactWorkVertices.json` 已拒绝，不作为默认优化配方。

## NvPerf：生产 dispatch 计数

确认 SDK 完整且为配套发行版本。当前 CMake 检查 `NvPerf/include`、
`redist/NvPerfUtility/include` 与 `NvPerf/bin/x64/nvperf_grfx_host.dll`。
必要时在 VS shell 中执行：

```powershell
cmake -S . -B build-release -DMETALLIC_ENABLE_NVPERF=ON -DMETALLIC_NVPERF_SDK_ROOT=E:/metallic/build/nvperf-sdk-2025.1
cmake --build build-release --target MetallicGPUDrivenSample --config Release
python -B Tools/Perf/NvPerf.py run --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --assets build/m2-assets-20260925/mini.json --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release --output build/nvperf-new
python -B Tools/Perf/NvPerf.py verify build/nvperf-new
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
python -B Tools/Perf/DeepProfile.py run --mode resources --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --assets build/m2-assets-20260925/mini.json --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release --output build/resources-new
python -B Tools/Perf/DeepProfile.py verify build/resources-new
```

明确需要 A/B 编译资源比较时加入已声明的 `--candidate`。核对
`PipelineStatisticsBinding` 的 input/device SPIR-V、cacheKey 与 shader/entry 映射，
不把旧日志的 spirv 缓存键当原始 SPIR-V hash，也不把 Local Memory sentinel 当 spill。

只有问题需要更宽的阶段活动时选择 `--mode triage`。从本机 ngfx 能力取得路径、
architecture 和 metric-set，再按 `DeepProfile.md` 运行。默认保持 clocks unaltered，
不把不同 clocks 策略批次合并。选择完整 Stream early/late 层级；原生表可能为 TSV
但扩展名为 `.xls`。两个同名列含义不明时保留列号与原值，不任选或相加。
通用 software triangles 不等于 WorkControl；此模式也不完成 Source/IL 导出。

## 中断、失败与恢复

保留失败包。先检查 `Process.json`、stdout/stderr、SDK 状态与 artifact freshness；
定位超时、未知 metric、overflow、缺失范围、身份漂移或资产不匹配，不通过放宽
阈值来修复数据。修正已知原因后用新目录重跑。

`build/shader-experiment.lock` 存在时检查 owner PID/输出目录与真实进程，活跃 owner
不得并行启动新实验。对 M3/resources 的 `Transaction.json`，owner 已停止后使用：

```powershell
python -B Tools/Perf/ExperimentRunner.py recover build/interrupted-experiment
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
