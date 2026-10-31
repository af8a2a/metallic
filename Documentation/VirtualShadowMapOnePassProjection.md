# VSM GPU one-pass projection

**最终结论（2026-10-31）：正确性通过，性能证据有效但结论不确定。** 按用户后续要求默认开启，保留显式关闭；没有将 discovery 的正收益当作确认结果，也没有追加性能采样。默认值变更不代表已确认性能收益。

本轮按用户要求撤掉空 receiver tile 实验，保留本机归档；方向光 Mark 默认启用不变。`VirtualShadowMapSettings::onePassProjection` 和 `projectShared()` 参数默认开启，LightsPass 可用 `onePassProjection=false` 关闭，显式 `benchmarkOnePassProjection` 优先。`benchmarkLegacyProjection=true` 保留最高优先级，用于原逐面路径对照。

## 实现

原 `projectShared()` 只共享 CPU 点光表准备，GPU 每面一次 dispatch。新路径把所有已准入面放入一个 80 字节/面的参数表，一次全屏 `vsmProjectOnePassMain` 调度完成投影。每个活动 lane 按相同顺序遍历面，资源索引保持 uniform。共同过滤参数继续来自原 208 字节 push constants；表中存逐面资源、原 view 句柄、有效过滤模式、采样相位、点光归属和光源参数。接收点和法线保留原逐面重建。

过滤统计使用整数 subgroup 归约，不在每面之间执行工作组 barrier 或分配统计共享数组。背景/接收点计数仍在原归属判断前逐面记录，点光 PCF/SMRT 跨面查找及原统计归属保持不变。Legacy/compact 的公开数组布局、debug 可选输出、页需求、分配、光栅和时域处理不变。没有引入 light list、灯光准入或新的采样策略。

该路径要求现有 `computeSubgroupBallotArithmetic` 能力；零面不调度，单面直接调用原内核。`benchmarkProjectionStatistics=true` 的旧实验选项也沿用原逐面内核，不与 one-pass 组合。表使用现有 completion-aware resource pool，所有面状态、描述符 leases、跨面表和管线均由命令生命周期保留；输出和统计同步在批次边界执行，compact 别名输出按底层纹理去重。

仓库早期批量投影尝试见 [屏幕输出与共享准备](VirtualShadowMapsScreenProjection.md)。本候选的可检验差异是紧凑参数表、共同参数仍走 push constants，以及无逐面工作组同步的统计归约；不能直接引用旧候选数据评价本实现。

首版还尝试共用接收点/法线，12 个 Floor 状态通过后，在 Room Hard/Legacy 的像素 6764 出现点光 face 14→16 转移，阴影值不同。限定正确性修正恢复原 receiver helper 和每面原 view 访问，没有修改浮点表达式、添加编译属性或放宽门槛。原版本、失败及二进制保存在 `CorrectnessRepair-v1/`、`Regression-v2/`；此前没有普通性能采样。首轮 `Regression/` 的另一个失败是五臂同时 profiling 耗尽共享 query 容量，测试已改成轮换辅助臂采集时间戳，五臂正确性比较仍逐帧全部执行。

修正后普通 one-pass 和 NoDebug 通过此前失败状态，但与旧投影统计实验组合时，像素 7156 仍改变点光面归属，见 `Regression-v3/`。该组合被明确限制为原逐面路径，实际 scope 必须证明回退；没有忽略统计开关或放宽画面门槛。普通 A/B 两臂均关闭该旧实验选项。

## 验证政策

GPU 矩阵扩展为 86 状态、五个对照臂：Legacy、Shared、OnePass、NoDebug、Statistics 兼容回退。覆盖 Hard/PCF/SMRT、Legacy/compact、视口尾部、冷/热缓存、jitter、连续历史、灯槽重排、预算拒绝、禁用、溢出恢复、14 面双点光、近裁面及零/单面回退。严格比较阴影、debug、HDR、深度、灯表、stats32、build45 和 15 项稳定工作量；`markPageRequests` 保留原调度相关例外。

实际 GPU scope `VSM one-pass projection` 必须恰好一次，且不能与旧 `VSM projection` 同时出现；其他阶段仍逐面记录。Runner 支持多光源 Mark/Build 的实际 SPIR-V 身份集合，并要求跨臂不变，不能用编译了新 shader 代替实际执行证据。

预先固定 Mixed（主收益）与 Sponza（单面保护）两个独立工作负载。均为 1920×1080、SMRT 2×16、2048 页、32 帧历史，无 jitter；Mixed 使用 OpenPbrVSMQuality 场景和 P4 相机，Sponza 沿用既有密集视角。每场景 diagnostic 两进程，普通 discovery 两个 ABBA 区组，仅 promising 时运行独立 confirmation 三个区组，不合并统计或追加显著性采样。

Diagnostic 64 帧，检查点 16/32/63；普通 96 帧，预热 32 帧，关闭 validation 和诊断回读。第 16 帧冻结 streaming 并 camera cut。主目标 projection 收益 95% 下界至少 3%；单面保护允许最多 5% projection 回退；两者 RenderGraph 回退上限均为 2%。保留缓存、每样本独立进程，不锁频、不认证全局 GPU 独占。

## 最终验证

Release `Metallic`、`MetallicRHITests` 构建通过，109 项 VSM Python 测试、387 文件 Slang 模块审计通过。最终 `Regression-v4` 的 86 状态 × 五臂通过，无跳过；先前的 profiling 回归也通过。已有 14 面状态仍触及 CPU metadata 发布上限，12 个状态显式标记 CPU 记录不可用，没有把预期值当作实测；GPU 投影路径及正确性检查继续完整执行。

Mixed 与 Sponza diagnostic 均通过：16/32/63 的 PNG/HDR/depth/shadow 精确一致，第 16–63 帧的逐面 stats32、build45、15 项稳定工作量及输入一致。已目视检查两场景实际输出。Mixed 的实际 projection scope 为 A9/B1；Sponza 为 A1/B1，保留原内核。统计实验兼容臂的 31 个已采样状态均证明使用原逐面 scope。

默认开启后的独立功能回归位于 `build/vsm-one-pass-projection/DefaultEnabled/`。Release `Metallic`、`MetallicRHITests` 重建通过；projection equivalence 的 OnePass 臂省略两个启用属性，86 状态 × 五臂全部通过（303 秒），实际 GPU scope 验证默认多面单次调度、零/单面回退、显式关闭与统计兼容回退。Profiling 回归 20 状态 × 四臂通过（约 32 秒），默认多面一次投影，单面仍逐面，计时/工作量开关未改变输出。该轮只验证默认行为，不新增性能收益结论。

## 独立性能确认

每场景完成 2 个 discovery 和独立 3 个 confirmation ABBA 区组。下表只使用 confirmation，每臂六个进程中位数再取中位数；收益及 95% 区间来自三个独立区组的 log time ratio，不能用两列中位数直接替代该统计。

| 场景 | GPU 区间 | A / ms | B / ms | 收益估计 | 95% CI |
|---|---|---:|---:|---:|---:|
| Mixed | Projection | 1.865680 | 1.897880 | -3.83% | [-21.36%, +11.17%] |
| Mixed | Shadows | 4.462696 | 4.534936 | -1.28% | [-4.03%, +1.40%] |
| Mixed | RenderGraph | 5.966408 | 6.049568 | -1.36% | [-4.96%, +2.12%] |
| Sponza | Projection | 1.016568 | 1.015520 | +0.12% | [-0.06%, +0.30%] |
| Sponza | Shadows | 3.722968 | 3.702344 | -0.31% | [-8.97%, +7.66%] |
| Sponza | RenderGraph | 6.903848 | 6.916944 | -0.24% | [-9.67%, +8.38%] |

两场景均为 `evidenceValidity=valid / effectAssessment=uncertain / nextAction=retain_candidate`。Mixed discovery 曾给出 +7.06% 的 projection 点估计，但独立确认没有确认该收益。确认轮次 A/B projection 的进程极差分别为 11.35%/15.92%，已保留为波动警告，没有剔除样本。Sponza 的 projection 保护单项通过，RenderGraph 区间仍不足以排除 2% 回退。不能据此宣称无收益、确定回退或生产加速；RenderGraph 时间也不是整帧延迟。

## 编译资源与限制

计时完成后仅执行预算内两次原生资源查询。普通运行实际 SPIR-V 与资源查询输入通过 SHA256→FNV→PipelineStatisticsBinding 对接，20 个 Mixed 普通进程身份一致，Mark/Build 的方向光及局部光变体和原生资源均保持不变。

| Projection 原生编译资源 | 原逐面 | One-pass |
|---|---:|---:|
| Register Count | 96 | 128 |
| Shared Memory Size | 4880 | 6144 |
| Binary Size | 109312 | 83584 |

原生资源占用增加，但这不是动态 occupancy、spill 或回退因果证明。Local Memory Size 的异常原始值保留，不作 spill 推断。六个普通 SPIR-V 模块验证通过；one-pass 的 SPIR-V 无 ControlBarrier/MemoryBarrier/Workgroup 变量，80 字节 stride、view offset60、208 字节 push 均经实际二进制核对。原生共享存储增加不是新增源代码 groupshared 数组。

当前实现只合并 GPU 调度及同步、替换过滤计数归约；不复用接收点重建，也不减少过滤采样数。后续若继续优化，应独立评估参数表读取、寄存器生命周期及局部光筛选，保留目前的逐面数值契约；不能把额外优化或复测结果混入本轮确认。

## 本机证据

生成产物不纳入源代码，位于 `build/vsm-one-pass-projection/`：

- [固定预算](../build/vsm-one-pass-projection/Budget.json)
- [最终 GPU 矩阵](../build/vsm-one-pass-projection/Regression-v4/Process.json)
- [Mixed 实景诊断](../build/vsm-one-pass-projection/Mixed/DiagnosticReport/report.md)、[Sponza 实景诊断](../build/vsm-one-pass-projection/Sponza/DiagnosticReport/report.md)
- [确认汇总报告](../build/vsm-one-pass-projection/Summary/report.md)、[聚焦图表](../build/vsm-one-pass-projection/Summary/Confirmation.png)、[绘图脚本](../build/vsm-one-pass-projection/Summary/plot_report.py)
- [Mixed 最终复核](../build/vsm-one-pass-projection/Mixed/FinalVerificationReport/report.md)、[Sponza 最终复核](../build/vsm-one-pass-projection/Sponza/FinalVerificationReport/report.md)
- [最终诊断编译审计](../build/vsm-one-pass-projection/CompilerAudit/RegressionFinal/CompilerAudit.md)、[原生资源及普通运行身份](../build/vsm-one-pass-projection/ResourcesReport/report.md)

原始包和失败未覆盖。性能实验结束时使用当时的 verifier 重核两份 confirmation 及其 diagnostic/discovery 链接；739 个源码、20 个二进制及声明输入的哈希保持一致。后续默认值变更与默认路径验证单独记录，不改写该轮性能证据。
