# M4：按需深化分析的首个验收切片

2026-09-25。本轮完成 **实际生产 pipeline 资源统计 + 无 UI 的阶段级硬件指标采集**，
并用它解释 M3 候选的资源变化。MiniZorah 非零 late case 的 early 六项指标集合通过
三次独立采集的重复性门槛。完整 early+late 集合仍有 late DRAM 不稳定项；
没有把它标成全部通过，也没有宣称完成生产内核隔离重放。后续 NvPerf backend
的独立验收见本文末尾及专门记录，不与此处阶段级 CLI 证据混合。

入口为 [DeepProfile.py](../Tools/Perf/DeepProfile.py)，复现、失败处理及证据契约见
[DeepProfile.md](../Tools/Perf/DeepProfile.md)。[机器结果与原始包 hash](AgenticShaderOptimizationM4.json)
随代码保存；大型原始数据留在 build 中。

## M3 回退的资源证据

RTX 5070 Ti / driver 616.92，Release，普通优化 shader，关闭 capture symbols。
单独开启 `VK_KHR_pipeline_executable_properties`，三组 AB 交错、六个独立进程。
每次保持 M2 的相机历史、资源/工作列表与质量设置，执行 1×8 个目标帧；
本批次的 GPU timestamp 不参与优化接受判定。

| 驱动统计 | 基线 A，三次完全一致 | uint3 独立数组候选 B，三次完全一致 |
| --- | ---: | ---: |
| Register Count | 37 | 41 |
| Shared Memory Size | 3512 B | 5560 B |
| Binary Size | 24064 B | 24064 B |
| subgroup | 32 | 32 |

候选最终资源占用增加 **4 个寄存器、2048 B 共享内存**。depth/visibility 在全部
前后检查点一致；已记录输入不变量一致，侧内 SPIR-V 稳定。两侧输入及 device
SPIR-V FNV 分别为 `14699073418322314354`、`5624891983249778816`，与 M3 的实际
生产绑定一致。本次没有 OMM 改写导致的两种字节指纹差异。

源码上，候选另建 `g_WorkControlVertices`，而入口的低 subgroup fallback 仍调用
使用 `g_StreamRasterWords` 的 `rasterStreamCluster`。这与实测资源增长相符，
但尚未通过驱动内部布局证明每一字节增量的来源。不能从 uint3 的逻辑大小推断
最终分配，更不能据此宣称 occupancy 改善。

M3 的正常计时已经判定该候选软件耗时增加约 11.48%，继续保持 reject。本轮资源
结果推翻了“缩小数组声明会降低最终资源占用”的假设，但 **没有证明这四个寄存器
或共享内存变化就是全部回退原因**。occupancy、bank conflict 和 stall 的因果归属
需要进一步实验。Local Memory Size 返回 `68719476736`，保留原值，不用它推断 spill。

[原始六进程包](../build/m4-resources-20260925-02/Manifest.json)及
[当前离线复核](../build/m4-resources-20260925-02-audit.json)均可定位。
原始 shader 已恢复为 SHA-256
`153788a450366a61c63ae1f1623ac66a5175f87bbba288c1917b523752b2bbac`。

## Nsight 真实格式与目标范围

Nsight Graphics 2026.3.1，官方 `ngfx.exe` + SDK start/stop，一个目标帧，
`Top-Level Triage`，`Blackwell GB20x`。该模板关闭 shader pipeline collection，
不收集 source symbols，不声称 headless Source/IL 导出。

源码核对发现必须区分两个 label：

- `VisibilityHybridRasterizer.cpp` 的 `Hybrid raster: software triangles` 属于通用 raster。
- `VisibilityBufferPass.cpp` 的 `Hybrid raster: stream software clusters` 包含 WorkControl。
  后者没有作为独立行出现在本次原生表中。

因此正式选择包含生产调度的完整 `.../Visibility raster/Stream early` 和
`.../Visibility raster/Stream late`。这里还包括 cull、分类、硬件光栅、合并等工作。
计数器只能描述这些区间内的设备活动，不能称为 WorkControl 独占计数，也不能用
导出 marker 的时长替代 M3 的 software timestamp。

真实 TSV 有两个兼容性特点：事件表使用八空格缩进，指标表使用完整路径；指标名
成对重复。工具还原层级并逐行核对，保留两列各自的原始列号和值，禁止 DictReader
静默覆盖。此次重复两列数值相同，下面仅展示一次，不把它们计作两个独立观测。

## 三次阶段级硬件验证

[正式批次](../build/m4-triage-20260925-04/Manifest.json)采用采集期 base clocks，
三次 REPRO_INFO 都报告 `Locked to Base`；工具默认仍为 unaltered，复现时必须
显式使用 `--clocks base`。时钟策略不同的批次不合并。全部 CLI exit 0，SDK complete，
trace 和导出表非空；SDK snapshot 与帧验证的输入/绑定/输出一致。

固定集合为 SM、L2、L1TEX、DRAM 四项原名利用率，加 shared load/store bank-conflict
计数。重复性预置为 `(max-min)/median <= 10%`，每项分别判定。

| 指标，中位数 | early | early 波动 | late | late 波动 |
| --- | ---: | ---: | ---: | ---: |
| SM % peak sustained elapsed | 38.8717% | 1.15% | 22.2883% | 6.79% |
| L2 % peak sustained elapsed | 23.0051% | 0.93% | 13.8683% | 6.41% |
| L1TEX % peak sustained elapsed | 26.7745% | 1.14% | 15.0610% | 5.72% |
| DRAM sectors % peak sustained elapsed | 8.68033% | 7.38% | 3.04881% | **47.34%，未通过** |
| shared load bank conflicts，原始导出计数 | 188161000 | 0.12% | 31137200 | 0.24% |
| shared store bank conflicts，原始导出计数 | 6228220 | 0.90% | 1096960 | 1.91% |

early 的六项集合稳定，满足当前切片“至少一个实际硬件指标集合稳定复现”的目标。
late 为 5/6，通过项不能掩盖 DRAM 不稳定；整体 `repeatable=false`。
`stableNonzeroMetrics=22` 表示 11 个“阶段×指标”各有两列，不是 22 个独立计数器。
没有从这些数据推出 bandwidth saturation、bank conflict 的具体 shader 归因或可回收时间。
利用率的定义与硬件单元见 [NVIDIA GPU Trace 架构说明](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html)。

[当前离线复核](../build/m4-triage-20260925-04-audit.json)重新读取原始 trace/table hash、
原始 readback、SDK 完成状态、逐轮 shader 身份与进程生命周期。保存的分析与重新
计算结果一致，不执行归档脚本。

## 失败批次也保留

- `m4-resources-20260925-01`：旧统计日志的 `spirv` 实为包含 metadata 的缓存键，
  不能直接 join workload 原始字节指纹。补充 input/device 指纹与 shader 名称映射后
  才运行正式六进程批次；没有按日志邻近关系猜测对应 pipeline。
- `m4-triage-20260925-01`：采集/导出成功，但原生事件缩进与完整路径不一致，解析拒绝。
- `m4-triage-20260925-02`：三次 unaltered 采集成功，最初窄区间非零指标均未稳定；
  随后源码核对证实该 marker 不对应 WorkControl。仅保留旧范围复核兼容性，
  不作为生产 WorkControl 计数证据。
- `m4-triage-20260925-03`：base clocks 采集/导出成功，但输出目录变为 BASE，
  原先固定 BASE_UNLOCKED 的解析失败。修正为唯一集合发现后使用新目录重跑。

未覆盖或改写失败批次。改范围和时钟策略是明确记录的新配置，不能把前后差异
归因于单独的时钟因素，也没有反复执行相同失败配置直到出现通过结果。

## 按需扩展的决定与验证

前一切片未引入完整 NvPerf SDK backend：现有官方 CLI 已能无人操作导出当前需要的
真实阶段级计数器。当前问题已经得到新的编译资源证据；引入另一个采集后端还
不能自动解决生产 dispatch 的正确隔离与状态恢复。

isolated production replay、SASS 依赖视图暂缓。若下一候选必须区分 WorkControl
自身的 occupancy、memory/atomic 行为，再优先建立其可写资源恢复契约，单独
验收 isolated 与 in-frame，一次只增加所需指标。Full Zorah 驻留、多 case 与
M1 通用 UI adapter 的既有边界保持独立，未被本轮标成完成。

Release 样例构建通过。Perf CTest **6/6**，共 **85** 项 Python 测试，其中本轮新增
13 项，覆盖原生层级/重复列、范围错配、空值/sentinel、shader 映射冲突、SDK
完成与 snapshot、重复性/非诊断误用及归档篡改。资源与正式 triage 的离线复核通过。
事务锁已释放，没有遗留本轮 renderer/CLI 进程。

## 后续：NvPerf backend

按后续要求，已增加可选的进程内 Vulkan Range Profiler。使用 D 盘提供的
Nsight Perf SDK 2025.1 配套头文件与运行库，直接标注生产 WorkControl early/late
的间接 dispatch，支持单 pass 采集与三进程一致性验证。
这一扩展不改变上面 M3 候选已拒绝的结论，也未完成 isolated replay 或多 pass 状态恢复。
详见 [NvPerf 验收记录](AgenticShaderOptimizationNvPerf.md) 与
[运行方式和证据契约](../Tools/Perf/NvPerf.md)。
