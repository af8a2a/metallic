# VSM 方向光 Projection 正交计算专用编译

2026-11-04 更新：同一开关已扩展为方向光资源访问与输出归属专用化，使用 uniform 描述符；采样数学继续保留与 Mark 共享的表达式。按用户决定默认开启，显式关闭仍可用。以下 2026-10-31 的结果属于旧版正交裁剪，不能作为新版本的性能证据；本次实现与测量见文末。

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

## 2026-11-04：方向光 uniform 资源访问

复用 `directionalProjection` / `benchmarkDirectionalProjection`，不新增设置、资源或 ABI。CPU 仍仅为真实方向光的 per-face Projection 编译该排列；混合 one-pass 和局部灯继续使用通用程序。单太阳即使开启 one-pass 设置，也走单面分支。

方向光过滤查询不再编译点光六面资源选择，选中 projection、projection table、page table、statistics 和 atlas 五处解析使用 `resolveUniform`。fine/ancestor/coarse 选择只改变同一资源中的数据索引，不改变描述符。该合同由真实光源类型门控、每次 dispatch 的固定 push 资源句柄和现有录制状态/描述符租约保证。方向光输出也不再执行点光共享层的 canonical-face 归属判断。

没有同时实施物理页 XY 打包、wave 统计、UVZ ray 重构或采样设置调整。原有四处正交计算裁剪保留。

### 数值边界与实现收敛

最初尝试进一步删除采样辅助函数的所有局部灯分支，在 Room PCF 中改变了阴影与查询工作量。保留 `vsmReceiverDomain` 原表达式后，126 状态矩阵通过，但 Sponza 实景仍出现极少查询差异，并经历史累积改变 PNG/HDR/shadow。将入口光源类型直接写成常量也复现了实景差异。具体原生指令原因未证明，未将其确定归因于 FMA 或寄存器变化。

最终仅专用化资源与输出归属，保留与 Mark 共用的 light-type 采样数学表达式、域判断、页表映射和 receiver-plane 修正。没有增加 epsilon、放宽图像阈值或删除异常帧。最初失败、逐项隔离与实景失败证据保存在本地 `build/vsm-directional-uniform-20261104/{Regression,Bisect,Diagnostic}`；它们不属于最终通过的计时证据。

### 验证

- Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过；Slang module ownership audit 392 文件通过。
- `virtual_shadow_map_directional_projection_equivalence` 最终 **126 状态通过**，validation 为 0。保留原局部灯跨面、混合 one-pass、开关切换、缓存、溢出恢复和 clipmap 边界覆盖；新增 resident/stream 的 mapping-disabled SMRT 十状态，并用计数确认实际执行原页表查询。
- 新增 `virtual_shadow_map_directional_projection_room` 四状态快速复现入口，复用 Room PCF 的 resident/stream、Cold/Warm；阴影失败时输出首个像素差异。最终完整矩阵包含对应 Room 状态。
- 最终 Sponza A/B 在第 16/47/79 帧的 PNG、线性 HDR、depth、shadow 精确相同；第 16–79 帧完整 stats32、全部 55 项 Build 和 29 项稳定 Work 逐面逐帧相同。Work 共 31 项，预先保留 `markPageRequests` 与 `markReceiverMaskAtomicOrs` 两项调度型 OR 次数例外，实际差异完整记录，没有按结果追加豁免。
- 两个诊断进程开启 core/synchronization validation，消息均为 0；已检查最终 Sponza 输出。没有新增移动相机的完整 Sponza 巡检或长时间显存压力验证。

### 有界普通计时

RTX 5060、驱动 621.01；默认 Sponza 1920×1080、SMRT 2×16、history32、分层缓存、2048 物理页，无 jitter/debug 输出。第 16 帧冻结 streaming 并 camera cut；预热 48 帧，确保历史完成累积。两臂仅切换 `benchmarkDirectionalProjection`，比较当前通用默认与本次专用路径，不能单独归因于 uniform 一项。

预算在采集前固定为两个 80 帧诊断进程和两个 ABBA 区组、八个 96 帧普通计时进程。普通计时关闭 validation、readback、GPUWork 和 Build 子阶段标记，保留阶段 GPUProfile；每进程使用预热后 48 帧中位数。保留磁盘 shader/PSO 缓存，不锁频，不停止其他应用，不声称 GPU 独占。

| GPU 范围 | A 通用 ms | B 专用 ms | ABBA 区组收益 |
| --- | ---: | ---: | ---: |
| Projection | 2.046880 | 1.839800 | 10.11% |
| Shadows | 4.162400 | 3.957544 | 4.95% |
| RenderGraph | 10.255888 | 10.018120 | 2.64% |

A/B 绝对时间为各四个进程中位数再取中位数；收益为两个完整 ABBA 区组的平均 log ratio，不必等于表中绝对时间直接相除。Projection 两个区组分别为 10.04%、10.18%；Graph 为 3.74%、1.52%，波动更明显。仅报告有界探索结果，没有计算置信区间、追加确认样本或默认推广，也不是呈现 FPS 结论。

十个进程的源文件、二进制、实际 shader 身份和输入封包保持一致；Mark、Build、raster mesh、depth fragment 的 SPIR-V 在 A/B 相同，只有 Projection 改变。诊断与计时的场景/相机/质量以及共享帧的几何与驻留输入已交叉核对；timing 不提供纹理语义哈希，未声称验证完整纹理内容清单。

实际绑定的普通 Projection SPIR-V 从 253712 B 缩至 217300 B，NonUniform 装饰从 68 条降为 0；两份模块均通过 Vulkan 1.4 `spirv-val`。提取核对当前 producer v3 缓存的长度、body checksum 和日志 SHA256。没有新增原生寄存器、occupancy 或 ISA 采集；静态 IR 变化不能代替这些证据。

最终证据位于 `build/vsm-directional-uniform-20261104/Measurement/`：`Summary.json`、`Correctness.json`、逐进程密封包和 `IR/Report.json`。`FinalPlan.json` 与 `RunFinal.py` 记录固定预算及复核规则。该轮测量时默认 false；随后按用户决定改为默认 true。可通过 `directionalProjection=false` 显式关闭；局部灯和混合 one-pass 的通用回退不变。默认值提升不增强既有两区组性能证据。

### 默认启用验证

`VirtualShadowMapSettings::directionalProjection` 默认改为 true。两个 Pass 继承该缺省值，显式 production / benchmark 属性仍保留原优先级；未改 shader 实现或采样设置。

- Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 重建通过。
- 默认提升后，126 状态等价回归通过，`productionDefault=true`，包含十个默认/显式开关控制状态、十个 mixed one-pass 状态、十个 mapping-disabled 状态，validation 为 0。`Default` arm 的工作量参照同步改为已接受候选。
- 默认 Sponza 诊断使用没有 `directionalProjection` 和 `benchmarkDirectionalProjection` 属性的图，80 帧、预热 48 帧。实际 Projection 日志为 true，实际 shader SHA 与此前显式候选完全一致。
- 第 16/47/79 帧 PNG、HDR、depth、shadow，以及第 16–79 帧 stats32、29 项稳定 Work、全部 Build55 和 canonical 输入与此前候选精确相同；两个既定调度型 OR 计数继续独立记录，validation 为 0。

新的源码和二进制身份独立保存在 `build/vsm-directional-default-20261104/Capture/Identity.json`，默认选择与跨版本等价复核见 `Verification.json`，回归见 `Regression/`。本轮没有追加性能 A/B，沿用前述有界性能结果。
