# Material System Phase 11 — Material Program → Closure Family

生产 Deferred 已显式区分 Material Program 与 Closure Family，继续使用 fused dispatch。
独立 split 原型完成 GPU A/B；本次数据不支持把全屏 Closure Buffer 升为生产默认路径，
因此没有进入 Phase 12 的生产 Packed Closure Buffer 实现。

## 生产逻辑分类

[MaterialClosureClassification](../Source/Runtime/Material/MaterialClosureClassification.h) 接收按现有
active executable slot 排列的 Program→Family 映射，生成：

- `programFamilyBins`：原始 Program slot 对应的 dense Family slot。
- `families`：Family slot 对应 canonical Closure Family。
- `familyProgramOffsets`、`programOrder`：按 Family 稳定分组的 Program 执行顺序。
- `programCount`、`closureFamilyCount`：不含背景的数量，可直接统计比例。

Program / Family slot 0 都保留给背景。空场景返回合法的 background-only schedule；
缺失、未知 Family 或把 Surface Family 填到背景 slot 会被拒绝。
SingleSlab、DualSlab、OpenPBRComposite 三类均能组成逻辑表。

[ScenePathTracePass](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) 从已有编译
ProgramKey 去重表建立映射。当前生产 Surface 特化均输出 OpenPBRCompositeClosure，
同族 Program 连续执行，但每个 Program 仍执行自己的 material evaluation + closure lighting。
间接参数、tile mask 与 push 中的 bin ID 始终使用原 slot；分组顺序不会改变像素归属。
原有 fixed-five-class fallback 保留。没有改变 MaterialBinning 的 GPU ABI、shader 绑定或管线资产。

映射随 material generation 更新重建，shader reload 时清理。日志形式为：

```text
[MaterialClosureClassification] generation=... programs=... families=... backgroundPrograms=1 mode=fused
```

这里统计当前 generation 的 executable table，不是可见像素数或非空 GPU bin 数。
现有生产回归场景记录的是 `1:1`；独立 Slab 基准覆盖 `2:1`、`8:1`。
保留 Program 特化有利于独立优化材质求值；Family 只定义兼容的 Closure 布局与光照实现，
不能用 Family ID 替代 Program 的求值代码。

## Fused / split 原型

[MaterialClosureSchedulingTests.cpp](../tests/rhi/MaterialClosureSchedulingTests.cpp) 通过 Phase 10 IR
lowering 创建 8 个程序定义，混合 Mix / Layer 及不同的程序内 procedural 求值。
八份 fused SPIR-V 的身份各不相同，最终都输出 `DualSlabClosure`。

[ClosureSchedulingProbe.slang](../tests/rhi/shaders/ClosureSchedulingProbe.slang) 对相同 fixture 比较：

```text
共同：visibility program IDs → program tile queues → indirect arguments

Fused：每个 Program 间接调度 → evaluate + 16-light shading

Split：每个 Program 间接调度 → evaluate → 96-byte DualSlab payload + Family ID
       → GPU Family classification → family tile queue + indirect arguments
       → 一个共享的 DualSlab Lighting Kernel
```

prototype 使用 32-lane 线性 tile、最多 8 个 Program，每个 Program 预留 tile task 区间。
它不替换 Phase 6 的生产稀疏分配器，也不支持任意 Family 的通用 split 后端。
GPU Family 分类读取 **求值阶段写出的 Family ID**，合并不同 Program 的有效 lane；
共享 Lighting Kernel 只读取 Closure，不重跑 source/provider。当前只物化 DualSlab，
不存在覆盖所有模型的巨大 union。SingleSlab 与 OpenPBR 仍走各自已有执行路径。

输入中包含背景、空 bin、部分 wave，覆盖 1×1、17×9、257×129、1024×512；
大尺寸同时测试连续条带和逐像素混合分布，Program 数分别为 2、8。
两个路径都包含相同的逐像素执行次数检查；Closure、输出和原子计数均分配在 device-local memory。
读回在 timestamp 范围之后进行。不是对 HostReadback/PCIe buffer 原子写入做性能比较。

每个 active pixel 必须恰好求值一次、照明一次，输出正确的编译期 Program ID；
任务数/间接参数与 CPU 计算一致。稀疏选择的像素另与独立 CPU 光照公式比较。
所有帧与 fused 参考图比较，容差 2e-5，本次实测最大差为 **0**。
已生成并检查 Coherent / Mixed 诊断图；它们是材质调度 fixture，不是最终场景画质展示。

## 测量协议与结果

[ClosureScheduling.py](../Tools/Perf/ClosureScheduling.py) 复用现有性能工具的进程、锁、hash 和
pipeline statistics 解析支持。现有 WorkControl runner 不支持本负载，因此提供独立入口，
不会把材质调度试验伪装成 WorkControl 优化或自动晋升候选。

- validation、普通 timing、compiler resources 三批分别执行，每批 3 个独立、串行进程。
- 每个 fixture 排除前 2 对预热帧，保留 6 对测量帧；每对交替执行 AB / BA。
- 表格先取每进程每侧中位数，再取 3 个进程中位数；图中保留三个进程点。
- 普通计时关闭 validation、capture symbols 和 pipeline statistics；统计查询来自独立批次。
- 主比较范围包括 material evaluation、可选的 Closure 写入/Family 分类和 Lighting；
  共同的 Program classification 单独记录，不包含在主表。reset、编译、CPU recording、readback 也不包含。
- 输出包含每帧原始 GPU 时间、进程起止、EXE/DLL hash、source snapshot、实际 SPIR-V 身份、正确性和 XML。
  verifier 检查文件完整性、进程串行、固定样本矩阵、shader 身份、失败/skip 和诊断混用。

2026-10-04，NVIDIA GeForce RTX 5070 Ti，driver 616.92，MSVC Release，1024×512、16 个光源：

| ProgramCount : ClosureFamilyCount | 分布 | Fused ms | Split ms |
| --- | --- | ---: | ---: |
| 2:1 | 连续条带 | 0.029328 | 0.492576 |
| 2:1 | 逐像素混合 | 0.053568 | 0.470064 |
| 8:1 | 连续条带 | 0.052768 | 0.555600 |
| 8:1 | 逐像素混合 | 0.216928 | 0.425136 |

8:1 条带的 split 三个进程中位数为 0.555600 / 0.516160 / 0.563152 ms，保留这项波动；
没有删除慢/快样本，也没有把帧数当成独立实验数或给出虚假的置信区间。
所有四种大尺寸负载中 split 都更慢。仅 payload 占 48 MiB，加上 Family IDs 与 task queue
共新增 50.125 MiB（不含两个路径共有的 Program queues、输出和测试 instrumentation）。

独立 `VK_KHR_pipeline_executable_properties` 诊断来自实际提交的 compute pipeline，
按 input/device SPIR-V 指纹与 debug label 对齐，三个进程逐项一致：

| kernel | 驱动 Register Count |
| --- | ---: |
| Fused，P0–P7 | 63 |
| Material evaluation，P0–P7 | 36 |
| 共享 DualSlab lighting | 28 |

寄存器数下降没有转化为此原型的整体收益。Register Count 不是 runtime occupancy。
驱动同时返回 `Local Memory Size=68719476736` 的异常值，原样保留，不解释为 spill。

Program switching 使用另一个控制实验：分别重复绑定相同 pipeline 和交替绑定不同 pipeline，
同样数量的单 workgroup dispatch，每次写不同地址，排除 dispatch 间数据依赖。
128 / 512 dispatch batch 分别对应 2 / 8 个 Program。三个进程的净差折算 ns/dispatch：

- 2 Program：-0.125 / 0.625 / 0.125。
- 8 Program：-0.09375 / -0.15625 / 3.21875。

差值符号和大小不稳定，没有测得可靠的额外切换成本；不能据此声称生产切换免费，
也不能把微基准差值作为 OpenPBR 的 per-switch 成本。时钟未锁定，未证明系统 GPU 独占。

**决策：生产保留 fused，只落地逻辑两级分类和统计。** 这份原型证据不支持启用全屏 Closure Buffer。
它也不排除其他 payload 压缩方案、更昂贵的材质图、更高光源数或其他 GPU 上 split 可能受益；
这些都需要新的等负载测量，不能由当前数据外推。没有改变渲染质量或宣称端到端加速。

## 复现与证据

```powershell
# x64 MSVC developer shell
cmake --build build-scheduling-release --target MetallicRHITests MetallicSceneTests Metallic -j 8
python -B Tools/Perf/ClosureScheduling.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/closure-validation-new --mode validation
python -B Tools/Perf/ClosureScheduling.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/closure-timing-new --mode timing
python -B Tools/Perf/ClosureScheduling.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/closure-resources-new --mode resources
python -B Tools/Perf/ClosureScheduling.py verify build/closure-timing-new
python -B Tools/Perf/ClosureScheduling.py plot build/closure-timing-new --resources build/closure-resources-new --report build/closure-report-new
```

输出目录必须是新目录。普通计时不运行 Nsight/NvPerf；pipeline 资源查询通过已有 Vulkan 后端 opt-in，
没有增加 mandatory SDK/dependency。完整 ShaderWarmup、全 RHI suite、长时间时域稳定性和大场景内存测试未运行。

本地忽略目录 `build/` 下的证据：

- `material-phase11-validation-final/`：3 个 validation 进程，各 2 个新测试通过，Vulkan validation errors 为 0。
- `material-phase11-timing/`、`material-phase11-resources/`：各 3 个独立进程，原始样本、日志、manifest 与源码快照。
- `material-phase11-report/report.md`、`FusedSplit.png`、`Switching.png`、`Registers.png`：Python 生成并检查的图表报告。
- `material-phase11-regression/Tests.xml`：9 项既有分箱、Deferred、材质编辑、Closure/Lighting GPU 回归全部通过。
- `material-phase11-baseline/`、`material-phase11-report/comparison.json`：独立的 Phase 0 场景回归通过；
  OpenPBR PT、OpenPBR Deferred、RTXCR Chiang 三组 RGBA32F 均逐位一致，RMSE / max error 为 0。
- Scene CTest 通过；ClosureScheduling / DeepProfile / ExperimentRunner / WorkloadCase 四组性能工具 CTest 通过。
  新增证据检查器的 6 个测试覆盖重复/缺失样本、诊断混入、无效时间、输出差异、skip 和文件篡改。

Vulkan loader 仍有既有 EOSOverlay / `E:\Validation.json` 缺失警告；不是本次 workload 的 validation error。
Phase 10 已确认的 Native Heap 契约测试失败不在本阶段选择的测试集中，未声称其已修复。
