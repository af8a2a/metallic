# 方向光 Mark 按 clipmap 数编译裁剪

2026-10-31。按当前 VSMSample-Sponza 完整图先采集分阶段 GPU 时间与工作量，再实现方向光 Mark 的编译级数裁剪。独立确认得到 **Mark 收益 22.96%，Shadows 总 GPU 收益 5.32%**；RenderGraph 收益区间跨零，不声明整图或端到端 FPS 提升。

方向光 Mark 的 clipmap 编译裁剪现已 **默认开启**，两个 VSM Pass 和已有图均继承该默认值。生产属性 `markClipmapSpecialization: false` 可恢复 12 槽路径，显式布尔 `benchmarkMarkClipmapSpecialization` 优先。已有方向光 Mark 特化仍默认开启。局部光保持 12 槽路径，不改变样本数、页预算、过滤质量或 LOD 设置。

## 先采集，再选择候选

通过 `MetallicRHITests` 的 production acceptance 回放执行当前 `Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json`，不是对用户截图进程的实时附加捕获。固定 Sample 相机、原生 1920×1080、一盏太阳、6 层 clipmap、2048 个物理页、SMRT 2×16、32 帧历史、关闭 jitter。第 16 帧 freeze streaming/camera cut，前 32 帧预热。设备为 RTX 5070 Ti，驱动 621.01；保留磁盘缓存，不锁频，不停止用户应用，也不保证系统 GPU 独占。

初始普通计时为 3 个独立进程、每进程 96 帧；诊断另用 1 个 64 帧进程。计时和工作量回读分别采集，以下时间是三个进程中位数的中位数，不是截图的平均值。

| GPU 范围 | 时间 ms | 进程中位数范围 ms |
| --- | ---: | ---: |
| Mark | 1.9260 | 1.9141–2.1017 |
| Allocator | 0.0961 | 0.0961–0.0965 |
| Build | 0.1509 | 0.1502–0.1512 |
| Raster | 0.0445 | 0.0439–0.0447 |
| Projection | 1.0188 | 1.0188–1.0193 |
| Temporal stage | 0.3692 | 0.3649–0.3710 |
| Shadows | 3.7598 | 3.7589–3.9269 |

各范围独立取中位数，不能要求分项中位数严格相加为父区间中位数。Temporal stage 包含该 stage 的命令及 publication，并非隔离 kernel 时间。

诊断预热后每帧中位数：2,070,497 个 receiver；70,396,898 次 Mark 采样位置；236,441,542 次 level 测试；约 374,154 次 page-request OR。Mark 每条射线另标记 receiver 端点，因此每 receiver 为 `2 × (16 + 1) = 34` 个位置。共享页请求去重已经大幅减少原子操作，但 level 地址判断仍执行。

请求 3629 个 fine pages，准入 2047、未准入 1582；每帧约 23 个新分配/dirty 页、2024 个 clean 页。Build 解码 68,603 个三角；Raster fine groups 约 5822。静态视角仍有采样需求及预算造成的页面更新，但 Build 已不是主热点。

初始 Timing/Diagnostic 的 744 项源码与 20 项二进制身份相同；核对 resolved graph、实际内部尺寸和第 16–63 帧的 canonical streaming 输入，差异仅为声明的诊断/输出注册。Timing 没有纹理语义内容回读，不把其缺失字段当成逐字节输入证明。最终 A 臂与初始默认路径的输出、页统计、Build 和稳定工作量也重新核验一致。

## 实现与数值契约

仅 Mark 编译入口传入 `VSM_MARK_CACHE_LEVELS`。关闭该选项及局部光取 12；方向光默认取已验证的 `settings.clipmapCount`。有效值进入 `Programs` 缓存键和 `[VSMShader] markCacheLevels` 日志，非 Mark 日志保持 12。独立 shader 的宏缺省仍为 12，由 CPU 录制时选择实际排列。

Shader 只替换 SMRT `lastPages` 参数/局部数组、初始化循环和固定展开循环的上限。Sponza 从 12 个编译缓存槽裁成 6 个，**两臂实际仍为同样的 6 层 clipmap**。保留 `level < firstLevel || level >= push.clipmapCount` 判断、`push.lightType` 数据流、所有浮点表达式及采样顺序。

CPU 边界保证 `clipmapCount` 为 1–12，push 直接复制该值。原程序在被移除的迭代中，先 continue 才可能执行投影、计数或缓存访问；这些迭代没有有效工作。裁剪不改变有效层的计算及缓存序列，不重试此前被拒绝的不变量外提、完整 lightType 常量传播或空 receiver tile 实验。

## 正确性验证

- Release `MetallicRHITests`、`MetallicVSMSample` 构建通过；shader module audit 通过（388 文件、276 module、112 implementation）。
- 新 GPU `virtual_shadow_map_mark_clipmap_specialization_equivalence`：82 个主矩阵状态、10 组控制，额外覆盖 1–12 层 SMRT 的 24 个冷/暖状态，共 106 个状态通过，validation 为 0。包含两个 Pass、resident/stream、局部光、PCF/SMRT/Hard、近面/屏外、dirty/overflow/recover、默认关闭、生产属性、benchmark 优先级及 shared Mark 关闭控制。
- 新 GPU `virtual_shadow_map_mark_clipmap_mutation`：同两份 `VirtualShadowMaps` 对象直接 record `1→12→6→1`，每档 transition/warm，共 8 个比较帧。Baseline 的 Programs cacheMiss 为 `[1,0,0,0,0,0,0,0]`，候选为 `[1,0,1,0,1,0,1,0]`；严格输出和计数比较通过，暖帧不重绘，validation 为 0。测试没有通过重建图或 Pass 掩盖程序缓存键问题。
- Sponza A/B 各 64 帧诊断：16/32/63 帧 PNG、线性 HDR、depth、shadow 完全一致；16–63 帧完整 stats32、45 项 Build diagnostics、15 项稳定工作量精确一致。原本受共享缓存调度影响的 `markPageRequests` 不要求逐帧相同。已查看候选场景图和工作量图。
- `VirtualShadowMapBitset.py --experiment mark-clipmaps` 使用新协议 `virtual-shadow-map-mark-clipmaps-v1`，核验 Mark A=12/B=实际级数、两臂已有 directionalMark 均开启、Build/Projection SPIR-V 不变。工具 51 项 CPU 测试通过，旧协议行为保留。

矩阵与完整场景 A/B 是当前候选的等价证据，不代表新增的长时间漫游或所有场景性能验证。

## 独立性能确认

事先固定 2 个 discovery ABBA 区组；Mark 收益点估计 26.56% 后进入预声明的 3 个独立 confirmation 区组。每进程 96 帧、预热 32 帧。普通计时关闭 validation、GPUWork、图像回读与 Build 子阶段时间戳；源码、二进制、输入及实际 shader 身份冻结。没有混合两阶段数据或追加采样。

确认阶段 12 个独立进程的结果为 **valid / confirmed_gain / manual_promotion**：

| GPU 范围 | A 12 槽 ms | B 6 槽 ms | 区组收益 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mark | 1.919752 | 1.437112 | 22.96% | [13.09%, 31.71%] |
| Shadows | 3.763456 | 3.558072 | 5.32% | [4.04%, 6.58%] |
| RenderGraph | 7.242120 | 6.972216 | 3.99% | [−1.28%, 8.99%] |
| Allocator | 0.096136 | 0.091704 | 3.73% | [−0.33%, 7.63%] |
| Build | 0.150768 | 0.150432 | 0.42% | [−0.35%, 1.18%] |
| Raster | 0.045080 | 0.046872 | −4.21% | [−6.75%, −1.72%] |
| Projection | 1.018608 | 1.019456 | −0.13% | [−0.29%, 0.03%] |

A/B 两列是每臂六个进程中位数的中位数；收益和区间使用三个完整 ABBA 区组的 log ratio，不是展示两列直接相除。Baseline Mark 的进程间相对极差为 21.53%，全部样本保留。Raster 区间出现小幅回退，展示中位数相差约 1.8 μs；没有改 Raster shader 或几何工作量，未单独证明原因，Shadows 总收益已包含该代价。RenderGraph 通过“不超过 2% 回退”的预定保护门槛，但区间跨零，不能声明整图加速。

## 编译资源核对

两次独立原生编译资源查询的实际 SPIR-V 与普通计时逐臂一致。实际日志 SHA256 对应缓存原始 SPIR-V，再用 FNV-1a 指纹连接 native pipeline；六个提取模块均通过 `spirv-val --target-env vulkan1.4`。

| Mark 原生资源 | A | B |
| --- | ---: | ---: |
| Register Count | 40 | 40 |
| Shared Memory Size | 2048 B | 2048 B |
| Binary Size | 68096 B | 55040 B |

原生程序体积减少 19.17%，寄存器及共享内存没有下降。数据支持“减少无效展开代码”的解释，但没有动态 occupancy/stall 证据，不能把收益归因为寄存器下降。Build/Projection 的 SPIR-V 与原生资源不变。资源查询时间不进入性能统计。

## 本机证据与复现

证据根目录为 `build/vsm-sponza-mark-20261031/`，捕获与图表不纳入源码。

- [初始分阶段与工作量报告](../build/vsm-sponza-mark-20261031/ProfileReport/report.md)、[跨诊断模式核验](../build/vsm-sponza-mark-20261031/ProfileReport/ProfileVerification.json)。
- [完整场景精确 A/B](../build/vsm-sponza-mark-20261031/DiagnosticReport/report.md)、[探索](../build/vsm-sponza-mark-20261031/DiscoveryReport/report.md)、[独立确认全部指标](../build/vsm-sponza-mark-20261031/ConfirmationReport/report.md)。
- [关键指标图表](../build/vsm-sponza-mark-20261031/SummaryReport-v2/report.md)、[绘图脚本](../build/vsm-sponza-mark-20261031/PlotSummary.py)、[原生编译资源](../build/vsm-sponza-mark-20261031/CompilerReport/report.md)。
- [最终重验与身份检查](../build/vsm-sponza-mark-20261031/FinalVerification.json)、[重验脚本](../build/vsm-sponza-mark-20261031/VerifyFinal.py)。当前 744 项源码、20 项二进制与 A/B 封存身份一致。
- `Budget.json`、`Candidate.json`、`Cases/`、`ABCases/`、`RunAB.py` 保存预算、候选与命令。新采集必须使用新输出目录；已有报告的 `Reproduce.py NEW_REPORT_DIRECTORY` 调用当前仓库工具离线重验，不运行封存的未知脚本。

上述候选 A/B 采集时默认关闭，随后按用户要求提升默认路径；默认验证与性能 A/B 分开记录。

## 默认开启验证

按用户要求将 `VirtualShadowMapSettings::specializeMarkClipmaps` 默认值改为 true。两个 Pass 继续继承 settings 默认，无需修改图资产；shader 算法、排列宏及资源布局不变。Default 测试臂省略生产与 benchmark 开关，直接与 Specialized 比较。

Release `Metallic`、`MetallicVSMSample`、`MetallicRHITests` 构建通过。106 状态 GPU 矩阵及 10 组选择控制通过，validation 为 0；日志中 399 条局部光 Mark 记录均保持 12 槽。

Sponza 独立运行默认、生产显式开启、生产显式关闭三份 64 帧诊断。16/32/63 帧 PNG、线性 HDR、depth、shadow 完全一致；16–63 帧 stats32、45 项 Build diagnostics 和 15 项稳定工作量精确一致，validation 为 0。默认与显式开启的 Mark 都是已验证的诊断版 `d28ffd45…`，显式关闭恢复 `c4845e9d…`。已查看默认输出图像。

另一次 96 帧普通默认路径执行 Mark `036ad786…`，Mark/Build/Projection SHA 三元组与上一轮独立性能确认的候选完全一致。本次仅验证默认选择和正确性，没有新增性能 A/B，不从该单进程计时重新声明收益。最终 744 项源码、20 项二进制身份核验一致。

证据位于 [默认路径验证报告](../build/vsm-mark-clipmaps-default/report.md)及[机器可读记录](../build/vsm-mark-clipmaps-default/Verification.json)。
