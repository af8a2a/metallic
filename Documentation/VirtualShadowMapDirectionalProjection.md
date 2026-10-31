# VSM 方向光 Projection 正交计算专用编译

状态（2026-10-31）：已实现，`directionalProjection` 默认关闭。GPU 等价与 Sponza 图像检查通过；独立确认 Projection 收益 13.80%、Shadows 收益 4.33%。RenderGraph 收益区间下界为 −2.19%，未达到预定 −2% 保护门槛，判定为 `inconclusive / uncertain / retain_candidate`，不声明整图稳定加速。

## 实现边界

两个 VSM Pass 均支持生产属性 `directionalProjection`，显式布尔 `benchmarkDirectionalProjection` 优先。`VirtualShadowMapSettings::directionalProjection` 默认 `false`。有效开关为方向光类型与设置的交集，并进入 `Programs` 缓存键。持久 pass 可以逐帧切换，已录制状态继续持有对应程序。

只有 per-face `vsmProjectMain` 编译 `VSM_DIRECTIONAL_PROJECTION=1`。方向光 `makeProjections` 生成的全部 fine/coarse projection 都满足 `metric.w=0`；局部光不启用该宏。独立的混合多灯 `vsmProjectOnePassMain` 不传该宏，继续使用通用程序。Sponza 单方向光即使开启 one-pass 设置，也通过单 face 分支使用 per-face 程序。

本轮仅裁剪四处透视计算分支：

- `vsmProjectPosition` 的透视除法。
- `vsmComparisonDepth` 的透视 receiver-plane 深度。
- `vsmRayDepth` 的透视径向深度换算。
- Hard Projection 的透视深度比较。

保留正交表达式、采样位置与顺序、dither、偏移、页选择、域边界、资源访问及动态 `lightType` 分支。没有实施 uniform 资源访问候选，也没有改变 GPU 资源布局。

## 正确性与构建

`virtual_shadow_map_directional_projection_equivalence` 复用既有 GPU 矩阵：82 个主状态、24 个 clipmap 1–12 Cold/Warm 状态、10 个混合 one-pass 状态，共 116 个状态；其中 10 个状态增加默认、显式开关、benchmark 优先级和逐帧切换控制。覆盖 Hard/PCF/SMRT、resident/stream、方向光与局部光、冷热缓存、失效、溢出及恢复。图像、深度、stats32、45 项 build diagnostics 和 15 项稳定工作量精确一致；仅保留既有 Mark page-request 原子调度计数例外。Validation 为 0。

VSMSample-Sponza 的第 16/32/63 帧 PNG、线性 HDR、depth、shadow 逐字节一致；16–63 帧 stats32、build diagnostics、稳定工作量均精确一致。候选第 63 帧图像已目视检查。该结果证明测试范围内相对 baseline 等价，不代表独立参考渲染器验证。

Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过；shader ownership audit 388 文件通过；CPU runner 回归 54/54、build attribution 8/8 通过。10 个实际 SPIR-V 模块通过 Vulkan 1.4 `spirv-val`。未采集移动相机完整场景性能。

## 性能与编译资源

RTX 5070 Ti，驱动 621.01，1920×1080，固定 Sponza 相机，6 层 clipmap、2048 页、SMRT 2×16、temporal 开启。两臂保留当前默认 Mark clipmap specialization 与 ray-page 快路径。每进程 96 帧，前 32 帧预热；第 16 帧冻结 streaming 并 camera cut。计时前已完成回归、诊断及独立编译资源采集，shader/PSO 缓存已热。普通计时关闭 validation、工作量 readback 和 native resource instrumentation。

预定预算为 2 个 discovery ABBA 区组，通过筛选后运行 3 个独立 confirmation 区组（12 进程）。下面只使用 confirmation；没有合并 discovery，没有剔除进程或追加样本，无 spread 警告。绝对时间为每臂 6 个进程中位数的中位数，收益与区间采用完整 ABBA 区组 log 比值，因此不必等于表中绝对时间的直接比值。

| GPU 范围 | A ms | B ms | 区组收益 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Projection | 1.0163 | 0.8767 | 13.80% | [12.56%, 15.03%] |
| Shadows | 3.4547 | 3.2907 | 4.33% | [3.35%, 5.31%] |
| RenderGraph | 7.0871 | 6.9370 | 1.76% | [−2.19%, 5.56%] |
| Mark | 1.3103 | 1.3165 | −0.14% | [−2.12%, 1.81%] |
| Allocator | 0.0852 | 0.0851 | 0.23% | [−1.76%, 2.19%] |
| Build | 0.2034 | 0.3262 | −18.81% | [−393.79%, 71.41%] |
| Raster | 0.0482 | 0.0481 | 1.51% | [−2.84%, 5.69%] |

Build 离散较大，不能从中确认回退，也不能把其他未改程序的波动归因于正交裁剪。各阶段独立中位数不可求和解释 Shadows。未控制 GPU clocks，也不声明系统级 GPU 独占。

Projection 原生寄存器为 96→96，共享内存 4880→2832 bytes，原生代码体积 119424→106496 bytes（−10.83%）。统计通过实际 SPIR-V SHA256、缓存提取及 native fingerprint 关联。Mark/Build/raster mesh/depth fragment 的实际 SPIR-V 在两臂相同。静态资源变化不能证明 occupancy 或 stall 原因。

## 证据与复核

本机证据目录：`build/vsm-directional-projection-20261031/`，不纳入源码。

- `Regression/`：GPU 矩阵与 validation。
- `Diagnostic/`、`Discovery/`、`Confirmation/`：密封原始证据。
- 对应 `*Report/`：当前 verifier 复核及 Python 图表。
- `CompilerReport-v2/`：原生资源与提取的 SPIR-V。
- `SummaryReport/report.md`、`SummaryReport/Confirmation.png`：汇总及进程分布。
- `PlotReport.py`、`SummaryReport/PlotProvenance.json`：绘图脚本、数据映射、版本与输入哈希。
- `FinalVerification.json`：816 个源文件与 20 个复制二进制的最终身份核验。

运行入口与完整协议见 [Tools/Perf/VirtualShadowMapDirectionalProjection.md](../Tools/Perf/VirtualShadowMapDirectionalProjection.md)。复核用当前仓库 runner 的 `report` 命令输出到新目录；不要执行证据包内来源不明的归档脚本。
