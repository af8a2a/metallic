# VSM 方向光 build 编译特化

`VirtualShadowMapSettings::directionalBuild` 及两个 VSM Pass 默认启用方向光专用 build，已有图无需添加属性。`directionalBuild: false` 可恢复通用 build；显式布尔 `benchmarkDirectionalBuild` 优先于生产属性。只有方向光生效，Spot 和 PointFace 继续使用通用程序。本项验证以当前默认的 cluster/triangle dirty-page bitset 为基线。

## 编译与数据契约

`VSM_DIRECTIONAL_BUILD` 只传给 `vsmBuildMain`，有效值进入 `Programs` 缓存身份。其余入口保留原宏集合。实际 shader 日志记录 `directionalBuild` 与 SPIR-V SHA256，不能仅以图属性证明候选执行。

`makeProjections` 为方向光生成 `metric.w=0`、`rowW=(0,0,0,1)`。方向光缓存复用仅替换深度行和偏移相关字段，不改变该类型标志；`compatibleLight` 禁止跨灯光类型复用缓存。基于这一契约，专用 build 编译移除：

- 精确三角形阶段的透视 polygon 裁剪分支。
- Cluster 包围盒的透视近面保守保留与 xy 除法分支。
- `vsmProjectPosition` 中的透视除法选择。

方向光仍执行原来的 dot、min/max、guard、floor/clamp 和页枚举。没有改变顶点坐标计算次序、资源布局、诊断 ABI、dispatch、barrier 或原子 work 发射语义。Mark、allocator、raster 和 projection 不启用此宏。

## 预声明验证

生成证据位于 `build/vsm-directional-build/`，不加入版本控制。

- GPU 等价矩阵 72 状态，覆盖 resident/stream、单灯/混合灯、Hard/PCF/SMRT、近面/屏外/点光接缝、cold/warm/dirty/overflow/recover、VPA 16/31/33、12 levels。其中 10 状态增加关闭位图及关闭 cluster culling 的成对控制。
- 比较图像、depth、完整 stats32、全部 45 项 build diagnostics 与 15 项确定性 workload。共享 Mark 的 `markPageRequests` 保留原有调度相关计数边界，不要求精确相等。
- Sponza 固定 1920×1080、SMRT 2×16、6 clipmaps、128×128 virtual pages、2048 physical pages、32 帧历史，关闭 jitter。第 16 帧 freeze streaming 并 camera cut，前 32 帧预热。
- Diagnostic 两个独立进程、各 64 帧；第 16/32/63 帧检查 PNG、线性 HDR、depth、shadow，逐 canonical 帧检查输入与计数。
- Discovery 两个 ABBA 区组、8 个独立进程、每进程 96 帧。有希望时才运行独立 confirmation 三个区组、12 个进程。普通计时关闭 validation、GPUWork、图像回读；不混合两个阶段。
- 目标为 build 收益至少 3%，RenderGraph 回退不超过 2%。以完整 ABBA 区组 log ratio 为统计单位，95% Student-t 区间；不追加轮次追逐显著性。

保留磁盘缓存，不锁频，不停止用户应用；串行实验不保证系统 GPU 独占。静态 SPIR-V 规模不是原生 ISA、寄存器占用或实际执行指令数，GPU pass/graph 耗时也不是端到端呈现 FPS。

复查入口：

```powershell
python -B Tools/CheckShaderModules.py
python -B -m unittest discover -s tests/perf -p 'TestVirtualShadowMap*.py'
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment directional-build --phase diagnostic --cases-dir build/vsm-directional-build/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-directional-build-new-diagnostic
```

独立协议为 `virtual-shadow-map-directional-build-v1`，与历史 cluster bitset 协议分开；report 自动识别原始 Plan，并验证输入、源码、二进制及实际 shader 身份。

## 2026-10-30 验证结果

本轮机器为 NVIDIA GeForce RTX 5070 Ti，驱动 621.01；不与历史 RTX 5060 的 VSM 测量混用。Release `Metallic` / `MetallicRHITests` 构建通过，Slang 模块审计通过（387 files / 275 modules / 112 implementation units），74 项相关 Python 测试通过。

4 项 GPU 回归全部通过、无跳过：directional build equivalence、triangle page bitset equivalence、cache、profiling。新矩阵完成全部 72 状态和 10 个控制状态；单灯方向光累计解码 96 个三角形、多灯方向光 212 个、局部光 1076 个，证明各路径实际有 build 工作。图像、depth、stats32、45 项 build diagnostics 与 15 项确定性 workload 全部符合等价门槛，validation 为 0。

Sponza diagnostic 第 16/32/63 帧 PNG、线性 HDR、depth、shadow 精确相同；第 16–63 帧 canonical 输入、完整 stats32、15 项稳定 workload 与全部 45 项 build diagnostics 相同。场景截图和报告图表已查看。每帧预热后中位数均为 4098 个 cluster 组、1072 个 surviving 组、24588 次 level 测试、166530.5 次 cluster bitset word 读取、236345 次 triangle bitset word 读取、5821.5 次置位页命中；本项减少编译进程序的透视代码，不减少这些逻辑工作量。

### 实际编译产物

从 shader cache 提取与实际运行日志 SHA256 完全匹配的原始 SPIR-V。诊断版和普通版分别存放；以下为普通计时程序，两臂均通过 Vulkan 1.4 scalar-layout `spirv-val`。

| 静态指标 | A：通用 build | B：方向光专用 |
| --- | ---: | ---: |
| SPIR-V 大小（B） | 109896 | 91024 |
| 函数指令数 | 6597 | 5226 |
| 基本块 | 1068 | 783 |
| 循环合并指令 | 33 | 19 |
| 条件分支 | 275 | 216 |
| OpFDiv | 15 | 3 |
| OpDot | 29 | 21 |

候选中 `ShadowClipVertex[9] polygon` 函数变量及其裁剪访问消失。普通 build 的 SHA256 为 A `d74c7cd9be3fcfd35c0f21fe1b747827e62667a9df9987552810f73b0a3f0b98`、B `4b3dc12d797029f4b551915c96cb9d19a95ac5d968ff741926a9ba1c582b842a`。Mark/projection 实际 SPIR-V 在两臂相同；build 身份在 discovery 和 confirmation 之间保持一致。以上静态规模不解释原生寄存器占用，也不能单独证明运行时加速。

### 独立 GPU 性能确认

Discovery 两个 ABBA 区组的 build 收益估计为 18.57%，满足进入独立确认条件。Confirmation 完成预定三个区组、12 个独立进程，仅使用自身样本。判定为 `evidenceValidity=valid / effectAssessment=confirmed_gain / decision=accept`，无主指标/graph 的进程极差警告。

A/B 列为各臂六个进程中位数的中位数；收益和区间来自完整 ABBA 区组 log 比值，不能直接用两列相除替代。

| GPU 范围 | A（ms） | B（ms） | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Build | 0.181712 | 0.148824 | 18.0816% | 16.6116%–19.5257% |
| Shadows pass | 3.898408 | 3.903816 | -0.4346% | -7.6108%–6.2630% |
| RenderGraph | 6.926328 | 6.934088 | -0.0380% | -1.7109%–1.6074% |
| Reset | 0.008312 | 0.008304 | -17.7180% | -133.1540%–40.5649% |
| Mark | 1.998408 | 1.998680 | -1.4977% | -8.2546%–4.8374% |
| Allocator | 0.088392 | 0.088032 | 0.1907% | -1.4902%–1.8437% |
| Raster | 0.045792 | 0.044592 | 2.7439% | 1.8704%–3.6096% |
| Projection | 1.016376 | 1.016512 | -0.0066% | -0.0679%–0.0547% |

Build 约节省 0.033 ms，达到至少 3% 收益门槛；graph 的区间满足最大 2% 回退限额，但没有确认 graph 或整个 Shadows pass 加速。Reset 很短且第三个区组波动较大，完整保留该宽区间；Raster 虽观察到小幅下降，其程序和工作量没有改变，不能据此宣称本项优化了 raster 算法。其他阶段的不确定性同样保留。

上述候选阶段采集时默认关闭，通过 `benchmarkDirectionalBuild: true` 启用；后续按用户请求切换生产默认，验证记录见下节。性能结论限于本机固定 Sponza workload；没有测量动态流送场景、其他显卡或端到端呈现 FPS。

### 证据与复现

最终核对 739 个源码和 20 个二进制与采集封装的 hash 一致，见 `build/vsm-directional-build/FinalIdentityCheck.json`。原始采集、预算、Cases 和执行脚本均保留；没有修改封装或追加区组。

- [正确性诊断报告](../build/vsm-directional-build/DiagnosticReport/report.md)
- [Discovery 报告](../build/vsm-directional-build/DiscoveryReport/report.md)
- [独立确认报告](../build/vsm-directional-build/ConfirmationReport/report.md)、[逐进程图](../build/vsm-directional-build/ConfirmationReport/ProcessMedians.png)、[全部阶段区间图](../build/vsm-directional-build/ConfirmationReport/BlockGains.png)
- [确认报告复现脚本](../build/vsm-directional-build/ConfirmationReport/Reproduce.py)
- [普通版编译报告](../build/vsm-directional-build/CompilerEvidence/Ordinary/report.md)、[诊断版编译报告](../build/vsm-directional-build/CompilerEvidence/Diagnostic/report.md)、[提取与分析脚本](../build/vsm-directional-build/CompilerEvidence/ExtractAndAnalyze.py)
- GPU 回归：`build/vsm-directional-build/Regression/Process.log` 及各测试 JSON；原始采集：`Diagnostic/`、`Discovery/`、`Confirmation/`。

## 2026-10-30 生产默认路径验证

按用户请求，将 runtime settings 默认值改为 true，并为单灯、多灯 Pass 添加 `directionalBuild` 生产属性，沿用显式 benchmark 属性优先的规则。只修改默认与属性读取，shader 算法、宏选择范围和资源布局不变。独立 shader 的缺省宏仍为 0；生产录制按灯类型与有效 setting 显式选择 build 排列。

Release 编辑器与 RHI tests 增量构建通过。重新运行 cache、profiling、directional build equivalence 三项 GPU 回归，全部通过，无跳过。方向光矩阵保持 72 状态、10 个控制状态，但 `Directional` 臂现在省略两个开关；控制组验证 `directionalBuild:false`、两个方向的 benchmark 优先级及禁用 cluster culling 的默认路径。图像、depth、stats32、45 项 build diagnostics 与 15 项确定性 workload 相同，validation 为 0。日志另外核验 252 条局部光 build 记录均选择 generic，所有非 build 入口均未启用方向光宏。

完整 Sponza 场景独立采集生产关闭与省略开关的默认诊断各一次（64 帧，32 帧预热），默认普通排列一次（96 帧）。三个封装均校验通过；默认图的实际 resolved properties 不含 `directionalBuild` 或 `benchmarkDirectionalBuild`。诊断第 16/32/63 帧 PNG、线性 HDR、depth、shadow 精确一致，第 16–63 帧 canonical 输入、stats32、全部 build diagnostics 和稳定 workload 精确一致；最终截图已查看。

默认诊断 build SHA256 为 `9c0fdedc414a17b9b5345e4a24e4fc62acfb56a85e0dc0ed91975225623961d7`，普通 build 为 `4b3dc12d797029f4b551915c96cb9d19a95ac5d968ff741926a9ba1c582b842a`，分别与上轮已验证诊断版和独立性能确认版相同。Mark/projection 身份也一致。本轮验证默认选择与正确性，没有重新做性能 A/B，不将普通排列验证的单次计时当成新的收益结论。

证据保存在 `build/vsm-directional-build/ProductionDefault/`：

- [完整验证记录](../build/vsm-directional-build/ProductionDefault/Verification.json)；[离线复核脚本](../build/vsm-directional-build/ProductionDefault/VerifyDefault.py)，运行时加 `--verify-only` 可复核既有封装。
- `Regression/` 保存 GPU 测试及完整日志；`Generic/`、`Default/`、`DefaultOrdinary/` 保存三次原始封装；`Cases/` 保存输入。
- `FinalIdentityCheck.json` 核对 739 个源码、20 个二进制与采集身份一致。

离线比较脚本首次只在逐 face 数据中排除了既有非确定性 `markPageRequests`，遗漏了同一计数的 totals 字段，导致汇总比较拒绝。修正后按原有 15 项确定性 workload 门槛复核同一批封装通过；原始采集未修改、未重跑，初版脚本与失败日志保留为 `VerifyDefaultInitial.py`、`VerificationInitial.log`。
