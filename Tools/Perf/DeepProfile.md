# DeepProfile：按需深化诊断

`DeepProfile.py` 在已经合格的 MiniZorah history case 上，分开采集生产 compute
pipeline 的编译资源和 Nsight GPU Trace 的区间硬件计数器。结果没有 accept/reject
优化评分；时间收益仍由 `ExperimentRunner.py` 的普通计时与正确性门槛决定。

## 运行

Python 标准库负责解析/复核；实机启动另需 `psutil`，用于仅清理本次拥有的子进程。
先使用 VS x64 开发环境构建 `MetallicGPUDrivenSample` 的 Release 版本。
工具检查 Release cache 与可执行文件位置，记录实际 EXE/DLL hash，但不会自动构建。
资产清单先用 `WorkloadAssets.py` 生成；示例沿用本机 M2 清单，迁移后必须重新建立。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample --config Release

# 三组交错 A/B，统计生产绑定的 WorkControl。省略 candidate 则三次 A。
python -B Tools/Perf/DeepProfile.py run --mode resources `
  --case Tools/Perf/WorkloadCase.MiniZorahHistory.json `
  --assets build/m2-assets-20260925/mini.json `
  --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release `
  --candidate Tools/Perf/Candidate.CompactWorkVertices.json `
  --output build/my-resource-diagnostics

# 三次串行 SDK capture + 官方 CLI 自动导出，无 UI 操作。
# 路径、架构和 metric-set 支持先以本机 ngfx --help-all 核对。
python -B Tools/Perf/DeepProfile.py run --mode triage `
  --case Tools/Perf/WorkloadCase.MiniZorahHistory.json `
  --assets build/m2-assets-20260925/mini.json `
  --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release `
  --ngfx 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/host/windows-desktop-nomad-x64/ngfx.exe' `
  --architecture 'Blackwell GB20x' --clocks base --output build/my-triage-diagnostics

python -B Tools/Perf/DeepProfile.py verify build/my-resource-diagnostics
python -B Tools/Perf/DeepProfile.py verify build/my-triage-diagnostics
```

输出目录必须不存在；每个进程默认上限 240 秒，最多允许配置到 900 秒。
保持 case 的相机、历史预热、extent、质量和驻留策略，诊断只使用 1×8 帧；
triage 通过 SDK 界定一个目标帧，硬件采集和资源查询不会同时开启。
`--clocks` 默认为 `unaltered`；`base` 请求 Nsight 在采集期间锁定 base clocks。
复核会检查 REPRO_INFO 中实际报告的时钟策略，自动发现 BASE/BASE_UNLOCKED 等
唯一导出集合；不会把不同策略的批次合并比较。
普通 METALLIC 环境变量全部清理后显式设定，避免继承未知实验开关。
只应在 GPU 空闲时启动；合作锁防止本工具与 M3 runner 并行修改 shader，
不构成系统级独占 GPU，也不能排除其他程序的 GPU 活动。

沿用 M3 候选限制、原文/hash 检查和事务恢复。无论诊断结论如何均恢复原文件，
不会自动晋升候选。强制终止留下锁时，确认拥有者进程退出后使用：

```powershell
python -B Tools/Perf/ExperimentRunner.py recover build/my-resource-diagnostics
```

遇到未知并发修改会保留原文与事务文件并拒绝覆盖；采集失败不重新利用旧表。
失败批次保留为 failed，不能离线改成完成状态。

## 可验证的关联

`PipelineStatisticsBinding` 将四类身份明确分开：

- 调用端 Slang 输出的 `inputSpirvFnv1a64`，与 workload 的生产绑定记录对应。
- 实际提交给 Vulkan 的 `deviceSpirvFnv1a64`，包含后端可能进行的 OMM 改写。
- `cacheKey`，包括 metadata 的内部缓存 hash；旧日志里的 `spirv` 实际是这个键。
- 源 module/entry 和 Vulkan `entry`；Slang 生成的 Vulkan entry 可以是 `main`。

只有 opt-in statistics 开启才计算新增指纹并记录映射。查询的对象就是创建后
返回给调用者的 compute VkPipeline，没有构建额外 graphics 诊断 pipeline。
parser 要求资源记录对应唯一生产映射，缺失/冲突/驱动不支持均失败。
保留原始行号、驱动资源名称和值、描述，不把异常 Local Memory Size 解释为 spill。
此接口是[编译结果统计](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html)，
不是运行时硬件计数器，也不直接证明 occupancy 或某种 stall 的变化。

triage 严格读取真实 TSV 格式的 `GPUTRACE_REGIMES.xls`，同时核对
`D3DPERF_EVENTS.xls` 的逐行顺序，还原其每级八空格的原生层级。
选择包含 WorkControl 的完整 **Stream early / Stream late 阶段**，不用末段名称模糊匹配。
`Hybrid raster: software triangles` 属于通用 raster，不能当作 WorkControl；
后者的 `stream software clusters` 在本机原生表中未独立导出，因此不开放 dispatch 级查询。
旧窄区间试跑仅保留离线复核兼容性，不作为 WorkControl 指标。
同名计数器的两列都保留原始一基列号；导出没有标明两列含义，不能擅自选一列或相加。
缺列、重复目标 marker、非有限值、负 sentinel、空表和 schema 改变均拒绝。
空的 `FRAME.xls` 不作为硬件指标成功的证据。

当前固定的最小集合为 SM/L2/L1TEX/DRAM 的四项原名利用率指标，以及 shared load/store
bank conflict 原始计数。没有用 substring alias 代替不同的计数器。
指标范围是 **整个 early/late 阶段时间窗口内的设备活动**，包括 cull、分类、软硬光栅及合并，
不是 WorkControl dispatch 专属计数。
不同层级区间、early/late 或同时执行的工作不可简单相加。
triage 模式关闭 shader pipeline collection，不能用它宣称完成 Source/IL 导出。

## 重复性与归档

每侧恰好三个独立进程；先验证进程串行、生产绑定侧内一致、输入相同、depth/visibility
完全相同，再报告指标。资源统计要求逐项一致。硬件指标分别保留三个值、中位数和
`(max-min)/median`，预置上限 10%；中位数为零而存在非零值时不评为稳定。
全零值即使重复也不作为“非零指标稳定复现”的依据。不自动重试到稳定。

`Summary.json` 的 `repeatable=false` 表示所选集合没有全部通过，不妨碍检查其余
有效数据；每个指标都有 `stable`，不能只挑成功项后宣布整套指标稳定。
它也不能证明 counter 活动只来自 Metallic。

原始 trace、表、日志、case、配置、shader 原文、源码清单、资产清单、调用环境、
工具代码、runtime hash 与 Analysis/Summary 全部由 Manifest 引用并校验 hash。
`verify` 使用当前解析器重读原始数据，而非执行归档脚本；报告当前 verifier hash。
资产仍是 M2 的声明依赖清单与运行前后 metadata 检查，不是完整可移植场景包。
性能接受、孤立状态恢复、SASS 依赖分析均不在这个入口的验收范围内。
