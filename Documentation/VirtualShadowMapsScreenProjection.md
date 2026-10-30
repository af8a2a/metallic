# VSM 屏幕输出与点光准备共享

本次减少公开屏幕输出、Legacy 中间拷贝、点光六面表准备和 Deferred 的灯表查找。普通 GPUDriven 光栅、页请求/分配/光栅、过滤数学和 SMRT 时域历史保持原有路径；GPU 投影继续每面一次 dispatch。

## 输出与消费

`VirtualShadowMapLightsPass.debugOutput` 默认 `true`，兼容现有图的 `debug` pin。关闭后不反射或分配 RGBA8 调试数组，也不创建它的视图、绑定、清理、屏障或 shader 写入。运行时设置 **Publish Page Debug** 会重编 RenderGraph；连接了 `debug` 的图应保持开启。

五份生产 compact 示例显式关闭此输出：`virtual_shadow_map_sponza`、`gpu_driven_realtime_vsm`、`gpu_driven_tessellation_vsm`、`gpu_driven_zorah_full_vsm`、`realtime_lighting_vsm`。P3/P4 调试图继续保留输出。

Legacy 和 compact 均直接投影到公开数组：Legacy 层为 `baseLayer + face`；compact 点光六面共享一层，仅 canonical face 写入归属像素。LightsPass 不再创建私有屏幕纹理或执行逐面 shadow/debug 拷贝。低层独立 `VirtualShadowMaps` 调用仍支持私有输出。公开数组的初始化保留，覆盖未准入和禁用层。

投影内核先完成归属与过滤，再对归属像素写一次 visibility 和可选的 debug。原有接收点/背景计数、fallback、过滤计数和同步次序保留；这不包括公开数组初始化及后续时域处理的写入。

Deferred 直接使用发布表的槽位契约：Sun/Moon 为 0/1，局部光源槽 `s` 为 `s + 2`，包括禁用和预算拒绝条目。稳定 ID 排序仅影响准入，不重排发布表。消费者保留 `header.count` 边界及 admission 判断，省去每次采样的线性查找。该契约已写入共享头文件，并用源槽重排及 Sun/Moon 独立消费验证。

## 点光表共享

`VirtualShadowMaps::projectShared()` 接受按灯分组、按 canonical face 排列的录制结果。PCF/SMRT 点光只在首面准备一次六面表，其他五面复用表句柄；原先每面各准备一次。Hard 不准备此表。单面调用继续委托 `project()`。

首面保存表资源和跨面的描述符 leases；命令缓冲保留所有被引用的 face resource owner，每面还独立保留自身状态和管线，直到 GPU completion。共享依赖仅准备和绑定一次，逐面的统计和输出同步继续执行。没有新增全屏 receiver cache、GPU 批量表或原生 GPU 能力需求。

隐藏属性 `benchmarkLegacyProjection=true` 使用原逐面表准备路径，用于同一二进制对照；默认使用共享准备。该开关不出现在运行时设置中。两臂都使用原有 push-constant GPU 内核，每面仍有一个 `VSM projection` scope。CPU scope `VSM shared projection` 发布 `faces`、`lights` 和实际 `pointTables`，空输入也记录零。

## 内存与正确性

以下为 RTX 5060、1920×1080、compact 的公共输出逻辑字节，不代表整个 VSM 或实际 DRAM 流量：

| 工作负载 | 数组容量 | 保留 debug | 关闭 debug | 减少 |
| --- | ---: | ---: | ---: | ---: |
| Sponza，单太阳 | 2 层 | 19.775 MiB | 3.955 MiB | 15.820 MiB（80%） |
| LookDev，Sun/Moon/Point/Spot | 8 层 | 79.102 MiB | 15.820 MiB | 63.281 MiB（80%） |

图集、页预算和历史配置不变。对应历史分配为 50.625 / 202.5 MiB，没有计入 80%。两种布局的私有屏幕输出均为零。混合场景有 4 盏准入灯、9 个面，GPU projection 仍为 9 次；过滤点光表准备由 6 次减为 1 次。

`virtual_shadow_map_projection_equivalence` 包含 78 个状态：Hard/PCF/SMRT × Legacy/compact，同帧对比 Legacy 准备+debug、Shared 准备+debug、Shared 无 debug。深度、R8 阴影、每面的完整 32 项统计、HDR 必须逐字节一致，两条有 debug 的路径也精确比较。覆盖冷启动、固定相位缓存、强制重画、实际 jitter/相机移动、连续时域历史、源槽重排、预算拒绝、禁用、工作溢出/恢复、三面点光和接缝；最后把 Spot 原位改成第二盏 Point，要求 4 灯、14 面全部准入。

Shared 臂单独开启 CPU profiling，逐状态检查实际 `pointTables`：Hard 为 0，过滤点光每灯为 1，禁用为 0；GPU profiling 同时关闭。双点光的 14 面超过现有 profiler 每节点 256 段发布上限，整棵 CPU 记录被拒收：测试明确记录计数不可用，不把期望值 2 当作实测；14 面的全部画面/统计/内存检查仍执行。若该状态有完整记录，也必须严格验证实际值。可选 pin、公共输出字节和私有输出为零也有严格断言。现有 local lights 测试增加稳定 ID 重排及 Sun/Moon 独立消费；重排后的 HDR 沿用原有格式容差，三臂同帧 HDR 保持精确比较。

固定相位缓存与轮转相位分开验证。SMRT 相位改变可能请求新页，不能把每个相邻帧都零 raster 当作缓存契约；测试仍严格要求固定相位 warm 状态零重画，并独立推进时域阶段。

实景诊断比较帧 16/32/63 的 PNG、线性 HDR、depth、shadow，以及第 16–63 帧全部统计和每面稳定 workload/build。`markPageRequests` 是并发共享缓存实际发出的原子次数，依照现有 bitset 协议保留原值但不要求逐位相等；其他门槛不放宽。

最终实景诊断两臂精确一致；与修改前 `f3a4ca3e` 二进制也精确一致，包括上述三个图像检查点、第 16–63 帧全部 32 项逐面统计及稳定 workload/build。图像已目视检查。恢复原接收点计算位置与顺序后，早期批量候选出现的少量边界像素差异也不再出现。完整结果见本机 `SharedCorrectness.json`，没有采用放宽容差的比较。混合场景还少持有 960 字节点光表缓冲（此次串行回放的逻辑字节）。

## GPU 批量实验与最终选择

尝试过多面共享一次 GPU 调度，并进一步增加 compact 点光的工作组归属掩码以跳过空面。这两种候选均通过同帧画面和接收面质量检查，但未通过性能门槛，因此已全部移除，没有保留实验入口或默认开关。

混合灯光场景中，第一版 projection 收益估计为 -14.25%，95% 区间 [-71.90%, +24.06%]，不足以证明收益。掩码版本更明确地回退：projection 收益 -59.09%，区间 [-59.28%, -58.90%]；完整 graph 收益 -21.58%，区间 [-29.92%, -13.77%]。负值表示耗时增加。这些是已撤下候选的数据，不是最终版本的性能。

SPIR-V 检查显示批量方案增加共享同步，并把部分 push-constant 访问转成 StorageBuffer 访问。不能据此断言原生寄存器或 spill 是原因。Nsight GPU Trace 尝试超时，未产生有效 capture，没有引用其硬件计数。

最终方案保留原逐面 GPU 内核，只共享 CPU 准备。确定的结构收益是表准备 6→1、输出字节减少和移除中间拷贝；不把这些计数直接换算成帧率提升。

## 性能方法与复查

最终 A 使用 `benchmarkLegacyProjection=true, debugOutput=true`，B 使用两项 `false`。其他设置与二进制相同。比较涵盖共享准备及省略 debug；单次最终写入、Legacy 直接输出和灯表索引在两臂中相同，不能从此对照单独归因其收益。

两场景均为原生 1920×1080、SMRT 2×16、32 帧历史、2048 页，关闭 jitter；第 16 帧冻结流送并 camera cut。诊断每进程 64 帧，启用 GPU work/验证/回读；普通计时每进程 96 帧、前 32 帧预热，关闭验证和回读。保留磁盘缓存，每样本独立进程，不锁频、不停止用户应用。

预先固定 2 个 ABBA discovery block；有希望才运行独立 3 个 confirmation block，不追加轮次追逐显著性。每进程使用预热后 GPU 中位数，95% 区间以完整 block 的 log time ratio 为单位。projection 目标 3%，graph 回退限额 2%；pass 耗时不代表端到端帧率。源/二进制/输入 hash、实际 projection SPIR-V SHA256、计划、原始进程与失败均保留在 `.cache/VSMOutputs/`。

最终两场景均完成 2 个 ABBA block（各 8 个独立进程），没有达到预设 projection 3% 门槛，未运行 confirmation。下表 A/B 为各进程中位数再取中位数；收益及区间来自独立 block 的 log ratio，因此不直接用两列相除计算。

| 场景 | GPU 范围 | A（ms） | B（ms） | 收益估计 | 95% 区间 |
| --- | --- | ---: | ---: | ---: | ---: |
| Sponza | projection | 2.3594 | 2.3585 | +0.042% | [+0.018%, +0.065%] |
| Sponza | Shadows | 13.5675 | 13.5264 | +0.071% | [-3.490%, +3.509%] |
| Sponza | graph | 19.2039 | 19.1620 | -0.126% | [-4.612%, +4.167%] |
| Mixed | projection | 3.9811 | 3.9650 | +0.858% | [-12.652%, +12.747%] |
| Mixed | Shadows | 9.1697 | 9.0758 | +1.261% | [-14.281%, +14.689%] |
| Mixed | graph | 11.8593 | 11.6362 | +1.496% | [-3.537%, +6.284%] |

这些结果未证明有实际意义的 GPU 总收益；graph 区间也不足以排除 2% 回退，不能宣称通过该置信门槛或获得帧率提升。结构计数和输出字节的减少由独立正确性/内存检查确认。未追加采样轮次追逐显著性。

```powershell
cmake --build build-release --target MetallicRHITests MetallicVSMSample Metallic --parallel 8
python -B Tools/CheckShaderModules.py
python -B -m unittest discover -s tests/perf -p 'TestVirtualShadowMap*.py'
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_projection_equivalence:*virtual_shadow_map_local_lights:*virtual_shadow_map_profiling:*virtual_shadow_map_receiver_quality' --rhi-bindless --rhi-validation --output-dir .cache/vsm-projection-check
```

构建沿用 x64 MSVC `build-release` 的编译器、Ninja 和 SDK。Matplotlib 仅安装在本地 `.cache/VSMOutputs/Python`，没有修改系统 Python。66 项性能工具测试及 Slang 模块审计（387 files / 275 modules / 112 implementation units）通过，编辑器、Sample、RHI Release 构建成功。原有 19 项补充 GPU 回归记录在 `Regression`；最终实现重跑 local lights、profiling、receiver quality 共 3 项通过（后者 36 cases / 108 frames），`SharedFinalTests2` 的完整 78 状态矩阵通过，无跳过。其中 72 状态验证真实 CPU 计数，6 个双点光状态显式报告现有 profiler 上限。首次遇到该上限的失败保留在 `SharedFinalTests`，未掩盖或删除。

最终本地证据为 `SharedFinalTests`、`SharedFinalTests2`、`SponzaSharedDiagnostic`、`MixedSharedDiagnostic`、`SharedCorrectness.json`、`SponzaSharedDiscovery`、`MixedSharedDiscovery`、`SharedPerformanceSummary.json` 和 `SharedProjectionPerformance.png`。`RunSharedComparison.py` 及每轮 `Plan.json` 记录输入、固定顺序和完整性检查；已撤下的 GPU 候选、旧工具副本和所有失败另存原目录，没有混入最终数据。

仍未合并点光六面的 mark 轨迹生成，也未引入 tile light list 或改变可见灯光/页需求策略。
