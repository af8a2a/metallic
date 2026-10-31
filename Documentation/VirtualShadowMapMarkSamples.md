# VSM Mark 样本生成与投影复用

> 当前状态（2026-10-31）：本文对应的已接受优化已固定为生产路径，对应运行时 A/B 开关及参考分支已退役。下文保留历史实验设置和验收结果，不再作为当前开关使用说明；当前范围与验证见 [生产优化固化](VirtualShadowMapProductionOptimizations.md)。

本次在已有共享 Mark 请求去重和生产 cluster dirty-page bitset 之上，减少局部光 Mark 的重复计算。`VirtualShadowMapSettings::optimizeMarkSamples` 现在默认 **true**，但 CPU 侧有效值为 `settings.optimizeMarkSamples && light.type != Directional`。方向光始终选择原有 shader 排列，Spot/Point 默认复用不变量；两个 VSM Pass 的显式布尔属性 `benchmarkMarkSamples` 可关闭或开启这一局部光策略，不能强制方向光进入已拒绝的排列。

有效值统一进入程序缓存比较、Programs 状态、shader 宏和日志。日志同时记录 `lightType`（0 directional、1 spot、2 point）；这是 CPU 选择独立排列，不是在同一 shader 内加分支掩盖方向光代价。最终生产策略已完成独立回归与 A/B 验证；初轮试验与最终结果分开保留。

## 实现与生产者契约

`Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang` 保留每条 ray 的采样数量、顺序、世界坐标插值及逐级页请求范围，没有投影端点插值、跨页近似或提前省略样本。

- SMRT 接收点的光源基、dither 基、texel size、disk scramble 和 step random 移到 ray 循环之外；透视插值的 `startW` 按接收点计算，`endW` 按 ray 计算。点光的 canonical face 与 forward 仍由接收点确定。
- `makeLocalProjections` 为同一光面所有 mip 生成完全相同的 rowX/Y/Z/W。Mark 对每个世界采样点只计算一次完整透视投影，各 mip 继续独立执行域检查及 texel/page 地址计算。
- 初轮也依据 `makeProjections` 的同一方向光 rowZ 相同契约复用 Z 点积，各级 XY 独立计算；最终 CPU 策略不为方向光选择该候选分支，保留原逐级计算。
- Endpoint 和 dither 分别沿用原来的 projection 输入与归一化过程；不因方向看似等价而合并不同输入的浮点运算。公用 endpoint/disk helper 的重载只接受已计算的不变量，其公式不变。

这些矩阵行相同的约束由 `Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp` 中两个 projection producer 保证。若将来引入逐级不同的深度范围或局部光投影，必须同时修改这一复用路径及其等价测试。

宏 `VSM_OPTIMIZE_MARK_SAMPLES` 与 runtime setting 一起进入 shader/program 缓存身份，切换设置会重新选择程序。独立 shader 入口默认宏为 0。资源布局、push constants、32 项 statistics ABI 和 build diagnostics ABI 均不改变。实际编译日志在既有 `[VSMShader]` 字段末尾记录 `markSamples` 与 SPIR-V SHA256，不能仅凭图属性断言候选已执行。

## 计数含义

本次没有减少逻辑采样点与待测试级数。`markSamplePositions` 仍记录进入 Mark 的采样位置数，`markLevelTests` 仍记录逐级域/地址测试数；它们不是随机数生成、矩阵点积或实际 ALU 指令数，不能将其不变误读为计算未减少，也不能用它们推算 ALU 收益。

A/B 的 15 项确定性 workload counters 应保持一致。共享 Mark 缓存的 hash collision 和线程调度可能改变 `markPageRequests` 原子请求次数；保留原始数据，但不要求这一个计数逐帧相等。完整 stats32、输入和画面等价要求不因此放宽。性能收益只由关闭诊断后的同负载 GPU 计时判断。

## 固定预算的 A/B 验证

初轮任务局部 runner 为 `.cache/VSMMarkSamples/MarkSamples.py`，最终策略使用独立的 `.cache/VSMMarkSamplesLocal/MarkSamples.py`，复用已有 Acceptance receipts、封装校验与 ABBA 统计；它不是新增的通用生产性能工具。原始 cases、runner、tests、日志和报告留在该目录，不纳入源码提交。

两个臂显式固定 shared mark、cluster culling 和生产 bitset 开启，仅 `benchmarkMarkSamples` 为 false/true。源码、二进制与输入在 GPU 实验期间冻结，实验串行执行；每个样本使用独立进程。

- Diagnostic：两臂各 64 帧，warmup 32，第 16 帧 freeze streaming 并 camera cut，检查第 16/32/63 帧输出及 canonical 输入、stats32 和稳定 workload。
- Discovery：2 个 ABBA 区组，共 8 个进程。普通计时关闭 validation、work diagnostics 和 capture/readback，本次每进程固定 96 帧，前 32 帧预热。
- Confirmation：仅在 discovery 满足继续条件时执行独立 3 个 ABBA 区组，共 12 个进程；不与 discovery 混算，不追加轮次追逐显著性。
- 主指标为 Mark GPU pass，目标收益 3%，RenderGraph 最大回退 2%，95% 区间。统计单位是独立四进程 ABBA 区组，收益按区组 log ratio 计算；不得用展示的两列中位数直接代替收益估计。

保留磁盘缓存，不锁频，不停止用户应用；串行执行不代表系统 GPU 独占。pass/graph 耗时不等同于端到端呈现 FPS。诊断带来的计时不得用于普通性能结论。

## 构建与正确性验证

Release 编辑器、VSM Sample、RHI tests 构建通过。Slang ownership audit 通过（387 files / 275 modules / 112 implementation units），68 项现有 VSM 性能工具测试通过；初轮局部 runner 的 22 项 CPU fixture 测试和最终 runner 的 23 项测试均通过。

初轮 6 项 GPU 回归全部通过、无跳过：mark samples equivalence、filter demand、receiver quality、SMRT temporal、SMRT search、content。质量测试 helper 显式启用候选；content 是原路径覆盖。最终 CPU 策略变更后重跑 mark samples equivalence、filter demand、content，3 项全部通过、无跳过。

最终等价矩阵为 82 状态，涵盖 resident/stream、Hard/PCF/SMRT、单/多灯、方向/Spot/Point、屏外/近面/点光 seam、cold/warm/dirty/overflow/recover；10 个状态另验证关闭共享 Mark 缓存。Prepared 臂省略 `benchmarkMarkSamples`，实际测试生产默认 true；Baseline 显式 false。图像、深度、页准入、完整 stats32 与确定性 work 精确一致，validation 为 0。

最终 Sponza 与 Mixed 独立 diagnostic 的 16/32/63 帧 PNG、线性 HDR、depth、shadow 精确一致；第 16–63 帧 canonical 输入、完整 stats32 与 15 项稳定 workload counters 一致。最终两场景截图已目视检查。Sponza 的两个显式开关臂均编译 `markSamples=false`，实际 Mark 与 Build SPIR-V 相同；它只作方向光策略控制，不运行新的性能试验，也不声明方向光加速。

最终 Mixed 的候选臂同时包含方向光原排列及局部光 prepared 排列。prepared Mark 的实际 SHA256 为 `bc28c3e99b664b0588e4a25d6727af0be7f776bc921010ce97906a314693d140`，与初轮质量回归使用的候选完全相同；方向光与 Build 排列在最终 A/B 间完全相同。证据见 `LocalShaderContinuity.json`。

最终方向光 Mark SHA256 `1e6ccfc9d9a0d5ddea5c7211859a084b2fd191fcb35aadf7395db0da74d07a3a` 与先前 bitset 生产验证的 `f97e050bf8f89dfd07834469d58017dd587ef1fc8c746227b13399d575f85937` 不同。公共 helper 提取后不能声称与前一发布版本字节相同；本次仅验证最终同源 A/B 的方向光排列及输出一致，没有测量相对前一发布版本的 Sponza 性能。

## 初轮候选结果与策略收敛

首轮所有光类型均可启用的候选通过 6 项 GPU 回归和完整场景诊断。Sponza discovery 的 Mark 从约 4.7741 ms 变为 5.0373 ms，耗时回退约 5.325%，因此没有进入 confirmation，也不默认启用方向光路径。

Mixed（Sun/Moon/Point/Spot）在独立 confirmation 中取得约 5.65% 的 Mark 收益，95% CI 为 4.60%–6.68%。这只是混合负载中原候选的确认结果，并未单独归因到某个 helper；不能断言方向光回退由寄存器压力引起，也不能把整体收益当作单个局部灯的加速比例。

据此仅改变 CPU 选择策略：局部光默认启用，方向光保持旧排列，shader 代码保持冻结以保留已测局部光程序身份。最终策略另起独立预算，保留原始 trial，不混用源码、二进制或统计样本，结果如下。

## 最终局部光策略的独立性能确认

RTX 5060、原生 1920×1080、SMRT 2×16、32 帧历史、2048 页、关闭 jitter/debug 输出。Mixed 为 Sun/Moon/Point/Spot 共 4 灯 9 面，预热后的固定快照没有 dirty build 工作。源码、二进制、实际 shader、输入与封装 hash 验证通过；最终核对 739 个源码与 18 个二进制均与采集时完全一致。

Mixed discovery 的 2 个 ABBA 区组得到 Mark 收益 7.075%，随后独立 confirmation 的 3 个区组（12 个进程）达到 `valid / confirmed_gain / accept`。下面列出所有阶段；A/B 是进程中位数再取中位数，收益和区间是区组 log ratio 估计。确认样本未与 discovery 合并。

| GPU 范围 | A：原计算（ms） | B：局部光复用（ms） | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Mark | 3.726072 | 3.437216 | 7.9056% | 6.0430%–9.7313% |
| RenderGraph | 12.083632 | 11.604856 | 3.9016% | 0.1358%–7.5254% |
| Shadows pass | 9.323112 | 8.930184 | 4.1265% | -0.6304%–8.6586% |
| Reset | 0.041760 | 0.041696 | 0.3171% | -1.6952%–2.2897% |
| Allocator | 0.013120 | 0.013720 | -2.6001% | -8.6936%–3.1518% |
| Build | 0.289168 | 0.225176 | 21.9775% | 21.1849%–22.7621% |
| Raster | 0.063120 | 0.063136 | -0.0001% | -0.4121%–0.4102% |
| Projection | 4.056928 | 3.980024 | 1.2436% | -1.8120%–4.2075% |

Mark 达到至少 3% 的确认收益，graph 未超过 2% 回退限额。Shadows、reset、allocator、raster、projection 的区间跨零，不能声称这些阶段获得了确认收益。Build 的次级计时虽下降，但该静态快照没有 dirty build 工作且实际 Build SPIR-V 相同，不将其解释为本次改进了 Build 算法或降低了 build 工作量。上述结果属于固定 Mixed 快照的 pass/graph 测量，不是单灯、动态流送场景或呈现 FPS 的承诺。

## 本地证据与复查

- 初轮：`.cache/VSMMarkSamples/{Regression,Sponza,Mixed}`；保留方向光候选被拒绝的 discovery 和 Mixed 的原 confirmation。`CandidateSource` 保存被测源码改动，`ToolSnapshot` 保存工具源文件与 hash。
- 最终：`.cache/VSMMarkSamplesLocal/Regression`、`Sponza/Diagnostic`、`Mixed/{Diagnostic,Discovery,Confirmation}` 及各自 `*Report`。`Cases`、`Budget.json`、`RunEvidence.py` 保留输入与预声明流程，`ToolSnapshot` 保留实际工具；`FinalIdentityCheck.json` 记录最终核对。
- `Mixed/ConfirmationReport/{ProcessMedians,BlockGains}.{json,csv,png}` 保留全部阶段数据与区间。`Summary.json`、`Decision.json` 记录身份与采用决定，生成数据未纳入版本控制。

本地复查依赖上述缓存、固定工具版本与现有 matplotlib 依赖：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='E:/metallic/.cache/VSMOutputs/Python'
$env:MPLCONFIGDIR='E:/metallic/.cache/VSMOutputs/Matplotlib'
python -B .cache/VSMMarkSamplesLocal/TestMarkSamples.py
python -B .cache/VSMMarkSamplesLocal/MarkSamples.py report .cache/VSMMarkSamplesLocal/Mixed/Confirmation .cache/vsm-mark-local-recheck --diagnostic .cache/VSMMarkSamplesLocal/Mixed/Diagnostic --reference .cache/VSMMarkSamplesLocal/Mixed/Discovery
```
