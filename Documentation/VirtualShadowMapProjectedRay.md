# VSM 方向光投影空间射线状态

2026-11-04：已实现并验证原型，保存在 `codex/vsm-projected-ray-state` 分支。当前默认分支未采用该实现，没有新增生产或 benchmark 切换。固定 Sponza 工作负载的 Projection 收益点估计为 1.76%，未达到采集前声明的 3% 门槛。

## 参考与实现边界

参考本机 `F:/UnrealEngine` 的 UE 5.7.4，提交 `8284a0e654e82cee0ed716f16704b1c897e330c3`：

- `VirtualShadowMapProjectionDirectional.ush:73–144` 的 ray state、起点 `w=1` / 方向 `w=0` 投影及 `start + step * time` 查页。
- `VirtualShadowMapProjectionCommon.ush:237–284` 的整数祖先归属、target texel clamp 和 fallback 深度基准。
- `VirtualShadowMapSMRTTemplate.ush` 的采样时间和深度历史；本轮保留 Metallic 现有历史判定，不改成 UE 的不同命中边界表达式。

Metallic 原型只用于真实方向光的 per-face SMRT，沿用既有 `VSM_DIRECTIONAL_PROJECTION` 类型排列。PCF、Hard、局部灯及 mixed one-pass 保持原路径。

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

Projection 两个配对分别为 1.752% 和 1.764%。采集前采用门槛是 Projection 至少 3%、两个配对均正、Shadows 不退、Graph 回退不超过 2%；仅 Projection 最低收益门槛未满足。一个 ABBA 区组没有置信区间或独立确认，Graph 的负点估计也不作为已确认回退结论，不推断呈现 FPS。

总共 8 个 GPU 进程，含两次基线夹具失败，没有追加性能样本。Sponza 正确性 A 复用已封存标准捕获，293 项 shader 源与原基线一致，旧 CPU/binary 身份差异明确只用于正确性参考；当前 scoped 回归 A、候选和全部普通计时严格使用同一最终二进制。

证据在 `build/vsm-projected-ray-20261104/`：`Plan.json`、`UnrealEvidence.json`、`ReferenceRegressionA.json`、`Correctness.json`、`Summary.json`、`Measurement/` 和两次原始记录 `Attempt-01/`、`Attempt-02/`。生成捕获、源快照和构建输出不提交版本控制。
