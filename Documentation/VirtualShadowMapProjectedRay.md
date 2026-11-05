# VSM 方向光投影空间射线状态

2026-11-05 更新：按用户指令，receiver 级平面参数/UE 式 ray-start bias 候选已撤回。
默认保留此前已接受的投影空间射线状态及 receiver-mask 移位。下文保留实验记录。

2026-11-04：按用户决定，已将 `codex/vsm-projected-ray-state` 的实现合入当前默认分支，作为方向光 per-face SMRT 的标准路径，没有新增生产或 benchmark 切换。固定 Sponza 工作负载的 Projection 收益点估计为 1.76%，仍未达到原先声明的 3% 门槛；本次采用决定不改变既有测量或证据范围。

## 参考与实现边界

参考本机 `F:/UnrealEngine` 的 UE 5.7.4，提交 `8284a0e654e82cee0ed716f16704b1c897e330c3`：

- `VirtualShadowMapProjectionDirectional.ush:73–144` 的 ray state、起点 `w=1` / 方向 `w=0` 投影及 `start + step * time` 查页。
- `VirtualShadowMapProjectionCommon.ush:237–284` 的整数祖先归属、target texel clamp 和 fallback 深度基准。
- `VirtualShadowMapSMRTTemplate.ush` 的采样时间和深度历史；本轮保留 Metallic 现有历史判定，不改成 UE 的不同命中边界表达式。

该标准实现只用于真实方向光的 per-face SMRT，沿用既有 `VSM_DIRECTIONAL_PROJECTION` 类型排列。PCF、Hard、局部灯及 mixed one-pass 保持原路径。

`VSMDirectionalRayState` 每条射线初始化一次，缓存带 dither 的查页起点、不带 dither 的比较起点、投影方向增量，以及 coarse 的独立 XY 起点/增量。fine clipmap 按二进制比例和相对平移换基，每个样本仍从 `receiverLevel` 重新查找；不移植 UE 的 receiver 预映射选级。采样仍使用原 `vsmRaySampleTime`，不采用累加步进。

CPU `makeProjections` 与兼容缓存保证全部方向光 fine/coarse 共用 `rowZ`，因此缓存 receiver depth 与 depth scale，沿用原减法/除法换算；不套用 UE 的额外 Z scale/bias。比较深度从最终 level 的独立未 dither 射线坐标计算，不复用查页后带 dither、祖先映射或 clamp 的坐标。propagated mapping 保持整数页归属和 target-page clamp，tag0 仍走显式层级查找。

## 正确性与验证范围

投影与插值换序会改变浮点舍入。本轮在首个候选 GPU 运行前明确采用数值质量门槛，不要求所有像素逐位一致，也不在看到候选结果后放宽门槛：独立 oracle 原门槛保留，每项 MAE 相对基线最多增加 `1/255`，已有聚合 lit/umbra interior error fraction 最多增加 0.001。缺少的独立 lit/umbra 指标不默认成零。

- Release 编辑器、sample、RHI tests 构建及 392 文件 shader 模块审计通过。
- 原实现和候选的 SMRT penumbra、search、真实 temporal 三项独立 oracle 通过，候选的原有 19 案 mapping 回归也通过，validation 为 0。
- 新 GPU probe 直接调用新状态和 lookup：9 fixture、594 个射线样本，覆盖 own/nonzero receiver/ancestor/负 origin/delta11/coarse/overflow/无效或越界 coarse。双精度线性坐标、比较坐标、深度单位、CPU 整数路由、texel 和唯一 atlas 像素检查通过。
- 原 Endpoint/Dither/Time 生成的受约束射线共进行 4268 次有效 fine-level Mark 区间检查，没有遗漏。这是有限 fixture 证据，不是任意浮点输入的通用证书。
- Sponza primary depth/visibility、非 Projection 的 stats32 槽、稳定工作量、Build55 及 canonical 输入保持一致；stats32 ABI/槽含义保留，实际 Projection 查询统计变化逐项记录。Mark/Build/raster 实际 shader 不变，Projection shader 改变。
- Sponza 16/47/79 帧活动 Sun 层分别有 0/47/94 个阴影像素变化，占比最高 0.00453%；最大 R8 差为 5/255。显著变化均在基线半径 2 像素、局部范围大于 8/255 的边缘带内，带外显著变化和严重内部变化均为零。非活动层不变，首帧 PNG/HDR 也完全一致；后两个检查帧不宣称逐位一致。

需求回归未声称整体通过。首次基线暴露旧查询下界未适配默认 adaptive ray count，测试现按设备能力使用至少一条射线的下界，PCF 九次及 SMRT 上界不变。第二次基线继续到既有 `Point-Finest-PCF-ZeroRadius` 失败：改变 Cached 的半径会清空需求历史，而 Hard 保留历史，导致请求同为 4765、总预算同为 1536 时，fine allocation 分别为 765/1530。两臂在不同 fine/coarse 覆盖上比较 raw 输出，未满足等资源条件；本轮不扩展修复此点光夹具，也不将其记为通过。

候选启用前即排除该非 SMRT 控制项，保留其完整失败记录；最终同二进制的三个通过 SMRT 报告作为 scoped 基线。候选回归五项均通过。没有把原失败父进程改写成通过。

## 固定预算性能结果

本机 Sponza 1920×1080、SMRT 2×16、history32、分层缓存、2048 页。第 16 帧 freeze streaming 并 camera cut。普通进程 96 帧、预热 48 帧，关闭 validation、readback/capture、GPUWork 和 BuildSubstages；保留磁盘缓存，不锁频，不声明 GPU 独占。

四个同二进制进程固定按 ABBA 顺序执行。A/B 列为各两个进程中位数再取中位数；收益为两个配对的平均 log ratio：

| GPU 范围 | A：世界空间逐样本投影（ms） | B：投影空间状态（ms） | 收益点估计 |
| --- | ---: | ---: | ---: |
| Projection | 1.835744 | 1.803472 | 1.76% |
| Shadows pass | 3.944488 | 3.911824 | 0.83% |
| RenderGraph | 9.657304 | 9.710088 | -0.54% |

Projection 两个配对分别为 1.752% 和 1.764%。采集前采用门槛是 Projection 至少 3%、两个配对均正、Shadows 不退、Graph 回退不超过 2%；仅 Projection 最低收益门槛未满足。随后依据用户指令固化到默认路径。一个 ABBA 区组没有置信区间或独立确认，Graph 的负点估计也不作为已确认回退结论，不推断呈现 FPS。

总共 8 个 GPU 进程，含两次基线夹具失败，没有追加性能样本。Sponza 正确性 A 复用已封存标准捕获，293 项 shader 源与原基线一致，旧 CPU/binary 身份差异明确只用于正确性参考；当前 scoped 回归 A、候选和全部普通计时严格使用同一最终二进制。

证据在 `build/vsm-projected-ray-20261104/`：`Plan.json`、`UnrealEvidence.json`、`ReferenceRegressionA.json`、`Correctness.json`、`Summary.json`、`Measurement/` 和两次原始记录 `Attempt-01/`、`Attempt-02/`。生成捕获、源快照和构建输出不提交版本控制。

默认路径合入沿用上述实现及验证，没有改动采样代码或资源布局。本轮仅核对合入内容、执行模块审计和 Release 构建，不追加 GPU 运行。

## Receiver 级平面斜率候选（2026-11-05，未采用）

实现了方向光 Projection helper 专用化和 receiver 级 fine/coarse 二维平面参数，
没有增加生产或 benchmark 开关。保持共享 Z、独立 coarse XY、严格
`abs(denominator)>1e-5`、未 dither 比较坐标及祖先整数页/clamp。
候选依次尝试归一化二维 slope、保留末尾乘除的二维 numerator，以及只保留
domain/contains/texel 三处专用化的版本。最后一版采样中仍有乘法和除法，
不能称为完整归一化系数全部外提。

两版完整物理回归通过半影、搜索、真实时域、19 案映射和 594 样本射线检查。
新增 288 样本 double 平面探针覆盖正负阈值邻域、零分母、fine 0/2/11 和独立
coarse 基底；六个倾斜/掠射 case 的 Hard/PCF/SMRT 共 18 张输出在首版 A/B
逐字节一致。首个新质量夹具在基线请求 2688 页、预算 1024 页时失败；仅将
focused 夹具预算调整到 4096，保留全部原质量和零丢页/fallback 门槛后重跑。
原失败父报告保留。其余已通过的旧二进制单项报告只作明确限定的正确性参考；
focused A2/B、完整场景 A/B 使用同一修正后的二进制。

三个版本均未通过预先固定的完整场景阴影门槛，没有放宽或重定义边缘带。
最终窄专用化版本的 Sun 层差异如下；非活动层不变：

| Sponza 帧 | 阴影变化像素 / 2073600 | 带外显著差异像素 | 严重内部差异像素 |
| --- | ---: | ---: | ---: |
| 16 | 3 | 0 | 0 |
| 47 | 13 | 4 | 0 |
| 79 | 22 | 5 | 0 |

局部像素差异虽少，但门槛要求带外显著差异为零，因此不采用二维变体。
末尾乘除改序不是已证明的唯一原因，仍不能排除 dot 分配、fine basis 复用及
编译器常量传播。最终版本只重新验证了完整场景，没有重跑完整物理 oracle；
相同 CLI 参数下其平面探针字节 hash 与上一版相同，不将它冒称为新的 GPU 探针结果。

总共执行八个 GPU 进程，包含首次基线失败。性能采样在正确性门槛前取消，
没有新的普通计时或性能收益结论。默认 shader 和测试源码恢复到本轮开始时的
状态，保留此前已接受的 receiver-mask 移位。恢复后重建 RHI 测试程序。

本机 `build/vsm-directional-receiver-plane-20261105/` 保留候选 `Prototype.patch`、
`Prototype/` 完整源码、各版快照、`Attempt-01/02/03`、全部原始回归/诊断、
`FinalCandidateQualityFailure.json` 和 `Adoption.json`，供独立复核；生成物不入库。

## UE 式 slope clamp 与 ray-start bias（2026-11-05，已撤回）

该候选曾按用户“接受 UE 的 slope clamp”的决定进行默认路径验证，随后按用户
“放弃候选”的指令撤回。以下为历史实验事实，不描述当前默认行为。候选包含
receiver 级二维平面参数，以及 UE 式 receiver-UV slope clamp / ray-start bias。
domain、contains、texel 三处 helper 同时使用已有方向光类型排列裁剪，不增加
生产或 benchmark 切换。Hard/PCF 的平面公式、局部灯及 mixed one-pass 保持原路径。

UE 5.7.4 `VirtualShadowMapProjectionDirectional.ush:16–35,105,223` 将
`d(normalized depth)/d(receiver UV)` 分量 clamp 到 ±0.05，并用于
`2*max(0,dot(clampedSlopeUV,TexelOffset))`，并非截断真实 texel-plane correction。
Metallic 中 `deltaWorld=2*receiverExtent*deltaUV`，所以从缓存的 world-plane
slope 乘 `2*receiverExtent` 得到 receiver UV slope。每个 receiver 计算 fine/coarse
真实斜率及 clamped UV slope；每条 ray 根据其实际 dither UV 只计算一次 bias。

真实的 signed texel-center plane correction 保持未截断，严格 denominator
门槛 `abs(d)>1e-5`、独立 coarse XY 归一化、未 dither comparison 坐标和祖先
整数页/clamp 保持。bias 只加到 comparison Z：UE 以 XY 查询，Metallic contains
还检查 Z，因此不移动 lookup Z，以保留既有 Mark/query 域。新增状态均为 shader
局部数据，CPU binding、push 和 stats32 布局不变。这是在 Metallic 平面修正上
叠加保守偏置，不声明与 UE 整套投影等价，也不是纯浮点等价改写。

首版直接截断真实平面修正的使用位置有误：虽然数学模型通过，Tilted-Clear
误阴影约 18.1%、Grazing-Clear 约 66.7%，五个 SMRT 几何 case 未通过原门槛。
该失败和源码均保留，未降低几何 oracle 的真值或门槛。修正使用位置后：

- 864 个独立 double 样本通过，包括 denominator 和 ±0.05 邻域、fine 0/2/11、
  非 dyadic coarse、正负 UV dither。真实 plane 和 clamped ray-start bias 分别验证；
  340 个正 bias、524 个零 bias，bias 最大绝对误差约 `9.32e-11`。
- 六个 Tilted/Grazing × Clear/Contact/Separated case、Hard/PCF/SMRT 原门槛
  全部通过，SMRT false-shadow fraction 均为零，最大 MAE 0.02614，最大 edge95
  为 1.313 像素。Validation 为 0，Release RHI 构建和 392 文件模块审计通过。
- 静态 Sponza 16/47/79 帧主 depth/visibility、非 Projection stats32、Build55、
  稳定工作量和 canonical 输入一致；保留已有两项 Mark 调度相关计数例外。
  Sun 阴影变化为 101/617/799 像素，最高占 0.03854%，非活动层不变。
  已查看第 79 帧场景图；用户接受算法偏置后，不再要求与未截断版本逐像素等价。

固定 Sponza 1920×1080、SMRT 2×16、history32、分层缓存、2048 页，freeze/cut16，
96 帧/预热48。普通计时关闭 validation/readback/GPUWork/BuildSubstages；保留
磁盘缓存，不锁频、不保证 GPU 独占。仅执行一个同二进制 A→B 配对：

| GPU 范围 | A 原默认 ms | B UE 式 bias ms | 耗时变化 |
| --- | ---: | ---: | ---: |
| Projection | 1.804800 | 1.880144 | +4.17% |
| Shadows | 3.837216 | 4.300208 | +12.07% |
| RenderGraph | 9.613504 | 10.331472 | +7.47% |

没有测到加速。同轮 allocator 也从 0.051872 增至 0.197088 ms，原因未定位；
不能因其 shader 未改就忽略回退。一个配对没有置信区间、ABBA 漂移平衡或独立
确认；不追加采样追逐结果，不把该版本称为已证实的性能提升或端到端 FPS 提升。
当时采用依据是用户接受算法选择及质量验证，不是性能获益判定；该候选现已撤回。

实际普通 Projection SPIR-V 为 216692→198364 B，NonUniform 装饰均为零，
其他四入口实际 hash 不变；保存了原始模块并通过 Vulkan 1.4 spirv-val。
静态体积和循环归类不代表原生寄存器、occupancy 或动态指令成本。

本轮共六个 GPU 进程：一次缺少已注册质量夹具的不完整进程、一次错误使用位置的
失败回归、最终回归/诊断各一次及普通 A/B 两次。保留全部父级失败状态；没有新增
完整场景移动相机或时域运动 oracle。诊断 A 引用旧二进制正确性参考，严格核对
runtime C++/shader 源；四个最终进程使用同一新封存二进制。普通计时缺少完整纹理
语义哈希，不声称完整纹理内容逐字节等价。

本机证据 `build/vsm-directional-slope-clamp-20261105/` 包含 Plan/Seal/Receipts、
Attempt-01/02、数学/几何报告、Correctness.json、FinalVerification.json、Summary.json
及 Audit/ 原始 SPIR-V。生成物不入库。
