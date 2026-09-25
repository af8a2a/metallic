# M2 补充跑测（2026-09-24 至 25）

MiniZorah 的固定视角与非零 late 相机转换 case 均通过三进程 A/A，SDK 也生成了新的独立 GPU Trace。Full Zorah 的 OOM 与退出挂起没有在本轮宿主执行中复现，但两批三进程试验仍因纹理驻留差异失败，**M2 总体验收仍为 partial**。失败批次保持原始 failed manifest，没有重写成成功。

## 正常计时

GPU 为 RTX 5070 Ti，driver 616.92。输出 1797×660，内部渲染 1198×440，WorkControl mode 5，同步 graphics queue；未改变 10% A/A 门槛。普通计时检查了已知 Nsight/RenderDoc 注入标记。每批独立进程串行运行，所有本轮正常采集进程均 exit 0，无 teardown 回收。

| Case / 证据 | 每进程 | Graph GPU 三次中位数 ms / 相对极差 | SW early+late 三次中位数 ms / 相对极差 | 判定 |
| --- | --- | --- | --- | --- |
| [Mini 固定视角 v2](../build/m2-mini-aa-20260925-01/Manifest.json) | 3×256 帧；预热 15 秒，恢复 32 帧 | 5.086432 / 5.055760 / 5.076288；0.6042% | 0.139424 / 0.139424 / 0.139360；0.0459% | stable |
| [Mini history v1](../build/m2-mini-history-20260925-01/Manifest.json) | 3×32 个目标帧；预热 15 秒 | 4.343440 / 4.361408 / 4.442128；2.2628% | 0.175264 / 0.176000 / 0.175648；0.4190% | stable |
| [Full 15 秒预热](../build/m2-full-aa-20260925-02/Manifest.json) | 3×256 帧，恢复 32 帧 | 输入身份失败，不出具合格 A/A | 同左 | failed |
| [Full 60 秒预热](../build/m2-full-aa-20260925-03/Manifest.json) | 3×256 帧，恢复 32 帧 | 输入身份失败，不出具合格 A/A | 同左 | failed |

统计单位仍是独立进程的运行中位数。两个 Mini 包的离线 `verify` 均 exit 0，输入、绑定与 depth/visibility 在前后、轮间和进程间一致。两种 case 的上下文不同，不应直接相减解释为性能收益。所有结果均 `candidateAccepted=false`。

固定视角：early 18,542 clusters，late 0。新 history case 每个目标帧之前运行 4 个未计时的相机平移 `[0,0,3]` 历史帧，drain 后恢复目标相机；目标帧也在下一次预置前 drain。early 为 19,773 clusters / `[19773,1,1]`，late 为 2,745 / `[2745,1,1]`。两阶段是实际生产 dispatch，软件列表 hash、cut、page mappings、depth/visibility 均保持一致，没有使用 coverage replay 充当计时。它验证特定相机转换，不是 isolated replay，缓存受到预置历史帧影响。

## Full Zorah 的明确失败项

15 秒预热批次三次纹理驻留为 177,353,216 / 178,008,576 / 178,008,576 bytes。60 秒批次为 176,370,176 / 176,370,176 / 176,632,320 bytes；累计 upgrades 为 403 / 403 / 404，downgrades 均 12。延长预热没有消除跨进程的发布差异。

两批每个进程内部冻结检查通过；几何/CLAS 驻留、cut、page mappings、bin、生产绑定和 depth/visibility 在跨进程也一致，差异字段只有 `identity.residency.textureBytes`。但 Deferred 消费的纹理状态仍然不同，不能把整图时间当作同输入 A/A。需要显式保存并恢复纹理 mip 驻留计划，或建立独立、明确声明固定纹理质量的 case；继续增加等待时间不能保证确定性。

[最初探索批次](../build/m2-full-aa-20260925-01/Manifest.json)还发现检查器将生命周期纹理 upgrades/downgrades 错当成每帧计数，要求为零。根据实现修正为测量期间不增长，累计历史不参与跨进程身份；实际 textureBytes 检查保留。原失败证据保留并新增回归测试。

## 资产与环境证据

[WorkloadAssets.py](../Tools/Perf/WorkloadAssets.py) 对声明依赖顺序 SHA-256：

| Manifest | 文件数 | 总 bytes |
| --- | --- | --- |
| [mini.json](../build/m2-assets-20260925/mini.json) | 5 | 70,928,795,715 |
| [full.json](../build/m2-assets-20260925/full.json) | 6,458 | 288,698,369,369 |

范围为 graph 中的 Asset 路径、glTF images/buffers URI 和现存 scene sidecar。运行前后复查 size/mtime；manifest 被复制进每批证据。没有复制全部 360 GB 原件，不宣称跨机器可移植资源包，也不将声明依赖范围扩大为任意隐式运行时依赖。

`GpuProcesses.csv` 保存 PDH 逐进程引擎样本，`Competition.json` 按 Capture 中的测量时间窗筛选重叠的一秒区间。Mini 固定视角三次分别有 6 / 7 / 7 个本进程样本，均 covered；可见 msedge、dwm 等背景活动，未关闭用户程序。history case 的 host 时间窗包含目标帧之间的预置操作，竞争报告是保守覆盖，不是只覆盖 GPU kernel 的精确时间段。

本轮 GPU 进程和 ngfx 通过工具审批在宿主执行。正常退出日志包含 graph、device、SDL 与 task system 清理完成。此前沙箱运行挂起没有在此路径复现；SDK 成功命令还去除了显式 platform 参数，因此不能将原因唯一归为沙箱。没有修改全局 GPU 时钟或驱动设置，也没有通过关闭用户程序腾挪显存。

## SDK 与 Nsight

[无源码符号探针](../build/workload-m2-sdk-20260925-03/Probe.json)和[history 源码符号探针](../build/workload-m2-sdk-20260925-04/Probe.json)均 CLI exit 0，应用 `sdkTrace.complete=true`，生成新 `.ngfx-gputrace` 和 5 个 `.xls` 表格文件（其中 FRAME 空表不作为有效指标证据）。完整参数、工具/应用 hash、METALLIC 环境与进程记录均保留。`--auto-export` 的此次输出没有 Source/IL CSV；源码导出通过下面的独立 UI 路径完成。

history 探针启用 `METALLIC_SHADER_CAPTURE_SYMBOLS=1`，保持优化编译；产物 `MetallicGPUDrivenSample_2026_09_25_00_07_58.ngfx-gputrace` 为 43,553,864 bytes。SDK 快照的 early/late 列表和 dispatch 与合格 history case 一致。符号版 SPIR-V FNV1a64 为 `7710823934669845759`，普通计时版为 `14699073418322314354`，两者不冒充相同二进制或 Nsight module hash；采集数据不参与正常 A/A。

Nsight Graphics 2026.3.1.0 build 38722833 已打开这份 trace；外部核验记录、原始 CSV、解析包和 11 张截图归档于 [Validation.json](../build/nsight-m2-ui-20260925-01/Validation.json)。trace SHA-256 为 `f112e3dc47e3be4ab2e607b0fc0a0050a4bac5872b6ead342b6c98e60b464b7a`。没有改写应用原始 `artifactVerified` 字段来替代外部验证。

可见队列为 `Vulkan Graphics Q:0 [VkDevice]`。early 的 compute pipeline bind event 123 与 late 的 event 222 均绑定 `0x00000198d1498cd0`；从 Shader Source 选择同一 handle 后，对应 module 为 `comp.10000.spv (4d0378657da7b7d1)`，CSV 的 `OpEntryPoint` 为 `GLCompute %streamClusterRasterWorkControlMain "main"`。软件阶段 label 分别从 event 122、221 开始。生产入口、阶段与 Nsight 模块已对齐。

| UI 选择 / 导出 | 范围（UI 舍入值） | WorkRaster 文件 Samples | 全部导出源码 Samples / IL Samples |
| --- | --- | --- | --- |
| [Entire Trace](../build/nsight-m2-ui-20260925-01/imported-entire-v2/manifest.json) | 0.00–5.07 ms | 5,100 | 5,100 / 5,100 |
| [Stream early](../build/nsight-m2-ui-20260925-01/imported-early-v2/manifest.json)，events 70–143 | 0.08–0.80 ms | 4,416 | 6,515 / 6,515 |
| [Stream late](../build/nsight-m2-ui-20260925-01/imported-late-v2/manifest.json)，events 169–242 | 0.86–1.06 ms | 684 | 1,117 / 1,117 |

三个包均原生格式解析成功、无未映射非空记录，重新导入后 `verify` 通过。整帧 CSV 只导出 WorkRaster 文件；阶段 CSV 包含其引用源码文件，因此不能直接比较最后一列的整帧与阶段总和。WorkRaster 文件的两阶段 Samples 相加等于 5,100。源码与 IL 是不同表示，不能相加；上述范围是父阶段窗口，不是单次 dispatch 的 GPU 时间。indirect groups 的证据来自引擎检查点，未宣称由 Nsight 解码参数独立验证。部分 marker 表中的零 duration 不代表零 GPU 工作。

阶段 CSV 还展示了一个真实格式变体：缺少 `Cooperative Vector Fusion` 列。解析器现在严格接受这两个已观察到的有序表头，其他未知列仍拒绝；新增保留原始记录与来源 hash 的真实夹具。旧解析包保持原状，解析器版本改变后生成新的 `*-v2` 包。三份 CSV 对应三个不同范围，不作为三次相同选择的重复性测试，也不用于声称性能优化收益。

## 代码与验证

- 新增历史相机预置及目标帧单独计时；诊断读回采用同一预置序列，检查器拒绝零 late。
- 添加测量 UTC 时间窗、已知注入检测和逐进程竞争归档；修正累计纹理计数语义。
- [Release 构建](../build/m2-rerun-build-03.log)通过；[最终 Perf CTest](../build/m2-rerun-perf-ctest-final.log) 4/4，共 55 项 Python 测试通过，覆盖资产缺失、范围、注入、计数、非零 late 和新增真实 CSV 表头等情况。
- 已验证配置：[Mini 固定视角](../Tools/Perf/WorkloadCase.MiniZorahWorkControl.json)、[Mini history](../Tools/Perf/WorkloadCase.MiniZorahHistory.json)。Full 的新配方作为失败实验保留在各批 Case.json，未替换成合格默认 case。
