# Metallic 长期路线图 / Long-term roadmap

这份路线图把 [issues/](issues/) 里的 63 条草稿排成一条可执行的长期路径。
它回答两个问题：**先做什么**，以及**什么算做完**。单条 issue 的正文负责"现在卡在
哪、怎么验收"；这里只负责阶段、主线与顺序。

## 当前快照（2026-11-01）

- **已发布**：63 条全部作为 GitHub issue 落在 `af8a2a/metallic` 的 **#9–#71**；编号映射见
  [publish-state.json](publish-state.json)，逐条链接见 [README 索引](README.md#issue-索引)。
- 台账合计 **63 条**：P0 **0** · P1 **14** · P2 **42** · P3 **7**；
  其中 `blocked` **10 条**，全部卡在厂商或上游（VVL 误报、Nsight/Aftermath 注入、
  驱动 pipeline cache、Slang hook、Streamline 互斥）。
- **没有 P0**：当前没有"已损坏且阻塞所有人"的条目。这是一个仍然可构建、可出图的主线，
  欠的是收口、证据与决策，不是救火。
- 投入分布：最近 30 个提交全部落在虚拟阴影贴图（VSM）上，因此 M1 的活跃线集中在
  VSM / 材质 / CLAS 三处；其余 4 个里程碑是本次盘点补出来的历史欠账。

## 里程碑

里程碑是**阶段关口**，不是并行队列。进入下一阶段的前提是上一阶段的退出条件成立。

### M1-Active · 8 条 —— 正在推进，需要收口或决策

退出条件：本里程碑内每条 issue 都有明确结论（采纳 / 回退 / 转成新的独立任务），
不允许"代码里有一半、文档里没有、默认值靠猜"的中间态。

| ID | 优先级 | 状态 | 标题 | 收口要点 |
| --- | --- | --- | --- | --- |
| [0001](issues/0001-vsm-propagated-mapping-default-flip.md) | P1 | in-progress | VSM: propagatedMapping 工作树默认值已翻转为开启，缺默认路径验收、单面 Pass 开关与文档同步 | 默认值与三份文档互相矛盾，先决策再收口 |
| [0021](issues/0021-clas-tessellation-template-shadow-regression.md) | P1 | in-progress | CLAS: 位移曲面模板实例化默认仍为 triangles，Shadow pass 回归在 C1–C5 全部候选下都未消除 | 最新活跃线，Shadow 回归未消除 |
| [0023](issues/0023-texture-feedback-wave-not-accepted.md) | P3 | in-progress | 纹理反馈: textureFeedbackWave 仍默认关闭，生产 A/B 未证明收益（Opaque +16.6%/+36.5%） | 已判定无收益，待正式退役 |
| [0041](issues/0041-material-closure-mix-layer-backend-gap.md) | P1 | in-progress | Material: Closure Mix/Layer 执行后端只有单面漫反射 Slab，M4 有界组合里程碑未闭环 | 材质 M4 有界组合里程碑未闭环 |
| [0043](issues/0043-material-ray-material-queue-unused-in-production.md) | P2 | in-progress | Material: M8 ray-hit 分类队列没有生产消费者，路径追踪仍只有 fused program loop | 队列已实现但无生产消费者 |
| [0047](issues/0047-material-graph-sdk-frontend-scope-limits.md) | P2 | in-progress | Material: MaterialGraph 与受限 SDK 前端只覆盖单材质 Surface 子集，任意原生 Slang 扩展未实现 | 受限前端范围是否接受 |
| [0048](issues/0048-fiber-strands-rt-and-model-coverage.md) | P1 | in-progress | Fiber/Strands: 原生曲线只有屏幕可见性后端，缺 RT 曲线求交、Far Field 模型注册与阴影/时域输运 | 曲线 RT 求交等后端缺口 |
| [0050](issues/0050-pathtracing-experimental-backends-unqualified.md) | P2 | in-progress | PathTracing: 五个实验后端（RT pipeline / SER / vkGltf / SimpleBSDF / splitPrimaryGuides）全部默认关闭且未过严格数值门槛 | 五个实验后端全部未过数值门槛 |

### M2-Correctness · 11 条 —— 正确性、确定性与验证可信度

退出条件：凡是"已知会失败/会出错"的路径，要么修复，要么被显式标记为不支持并在
测试与文档中可检索；不允许存在"稳定复现但无人跟踪"的失败。

| ID | 优先级 | 状态 | 标题 | 收口要点 |
| --- | --- | --- | --- | --- |
| [0009](issues/0009-vsm-marker-section-budget-overflow.md) | P2 | open | VSM: 多面/多灯下每 face 生成的 marker 会耗尽 RenderGraph 256-section 预算，Projection 归因不可用 | 多灯每 face marker 溢出 256-section 预算 |
| [0029](issues/0029-vvl-payload-limit-literal-stride-policy.md) | P2 | blocked | RHI: VVL 误报 task/mesh payload 上限，literal descriptor heap stride 策略被迫常驻 | VVL 误报导致的常驻规避策略 |
| [0031](issues/0031-aftermath-auto-checkpoints-under-injection.md) | P2 | blocked | Tooling: Nsight 注入期间禁用 Aftermath 自动 checkpoint，GPU 进度定位能力缺失 | Aftermath checkpoint 与注入冲突 |
| [0034](issues/0034-native-resident-tessellation-image-diff.md) | P1 | open | RHI: native resident 细分渲染图像与参考不一致（32/96 个 resident 组合失败） | 可复现的图像不一致，根因未最小化 |
| [0035](issues/0035-as-handle-heap-index-unification.md) | P2 | blocked | RHI: AS handle 无法统一为 heap index —— 驱动 heap 数值 load 错误与 Slang hook native 崩溃 | AS handle 两处硬阻塞 |
| [0036](issues/0036-shader-object-pipeline-cache-device-lost.md) | P1 | blocked | RHI: 跨 shaderObject 配置复用驱动内部 pipeline cache 触发 DeviceLost，根因未确认 | DeviceLost 根因未确认 |
| [0037](issues/0037-native-alpha-mask-coverage-assertion.md) | P2 | open | RHI: native alpha-mask 覆盖断言持续失败，前表面覆盖数远超上限且原因未证实 | 覆盖断言长期失败 |
| [0046](issues/0046-material-regression-credibility-gaps.md) | P2 | open | Material: 材质回归可信度缺口——长期失败用例、未定位的整组 HDR 漂移与 NTC alpha 跨路径不一致 | 长期失败用例与未定位漂移 |
| [0055](issues/0055-rendergraph-read-visibility-frontier.md) | P2 | open | RenderGraph: 同状态 read/read 快路径可能丢失 writer 可见性，尚未取得 GPU 级验证 | 未证伪的可见性洞，需 GPU 级判别 |
| [0056](issues/0056-rendergraph-hzb-preview-invalid-argument.md) | P1 | open | RenderGraph: Visibility 第二帧 HZB preview 稳定返回 InvalidArgument，回归未修复 | 已复现未修复的 HZB InvalidArgument |
| [0069](issues/0069-build-nrd-submodule-mapping.md) | P1 | open | Build: .gitmodules 缺少 External/nrd 映射，子模块初始化直接失败 | 全新检出无法初始化子模块 |

### M3-Hardening · 22 条 —— 生产化与基建

退出条件：默认路径可被任何人一键验证（CI lane + skip 口径 + 抓取前置检查），
且文档与实现不再冲突。

| ID | 优先级 | 状态 | 标题 | 收口要点 |
| --- | --- | --- | --- | --- |
| [0017](issues/0017-streamer-root-cut-full-rebuild-pending.md) | P2 | open | Streamer: 根页仍占 2.956 GiB / 3.5 GiB 几何池，root-reduction cook 后未执行 Full 全量重建 | root-reduction 后未做全量重建 |
| [0018](issues/0018-streamer-initial-load-main-thread-sync.md) | P1 | open | Streamer: 初始加载仍在主线程同步完成（loadingSeconds 7.7 s，无进度/取消），未达可交互目标 | 首帧主线程同步加载 |
| [0030](issues/0030-nsight-capture-runtime-integrity.md) | P1 | open | Tooling: Nsight 抓取子进程处于 Low 完整性级别，报告与场景缓存写入失败 | 抓取子进程完整性级别 |
| [0032](issues/0032-nsight-command-buffer-recycling.md) | P2 | blocked | RHI: 为规避 Nsight 悬垂引用，命令缓冲在抓取期永不释放并全局复用 | 抓取期命令缓冲常驻 |
| [0033](issues/0033-gputrace-d3d12-module-pin.md) | P2 | blocked | Tooling: GPU Trace 退出期依赖 D3D12.dll，靠 PIN 住模块维持注入进程存活 | GPU Trace 依赖 PIN 住模块 |
| [0039](issues/0039-streamline-descriptor-heap-workaround.md) | P2 | blocked | RHI: Streamline 与 VK_EXT_descriptor_heap 状态互斥，靠绑定空 legacy set 换取 DLSS 输出 | Streamline 互斥旁路 |
| [0044](issues/0044-material-m2-doc-vs-code-drift-sparse-program-bins.md) | P2 | open | Material: M2 实施记录与代码现状脱节（稀疏 Program 分箱已生产默认开启，文档仍写未实现） | 实施记录与代码脱节 |
| [0057](issues/0057-rendergraph-subresource-and-ownership.md) | P2 | open | RenderGraph: 精确 subresource 范围与 exclusive queue-family ownership 尚未纳入同步计划 | subresource / exclusive ownership |
| [0059](issues/0059-test-skip-semantics-baseline.md) | P1 | open | 测试基建: skip 与通过无法区分，缺少 skip 基线与已知失败跟踪 | skip 与通过不可区分 |
| [0060](issues/0060-verification-matrix-ci-coverage.md) | P1 | open | 构建与验证: 无 CI 工作流，metallic-ci 只跑 scene/task/debug，关键缓存用例仅 Windows | 无 CI 工作流，preset 覆盖不足 |
| [0063](issues/0063-metallicctl-perf-command-surface.md) | P2 | open | Tooling: metallicctl 缺少 perf 命令面与能力状态，实验链固定在 Tools/Perf 脚本 | metallicctl 缺 perf 命令面 |
| [0066](issues/0066-sky-atmospheric-pass-parked-branch.md) | P2 | blocked | Sky: 大气天空渲染通道停在未合并分支 modernization_/sky，需重新落地或明确废弃 | 天空通道停在未合并分支 |
| [0067](issues/0067-geometry-meshlet-compression-parked-branch.md) | P2 | blocked | Geometry: meshlet 自适应位置编码与关节切线帧停在未合并分支，测试已就绪但未进主线 | 压缩分支停在未合并分支 |
| [0068](issues/0068-build-branch-and-stash-archival.md) | P3 | open | Build: 9 个长期未合并分支与 10 个 stash 缺少归档策略，工作正在腐烂 | 分支与 stash 缺归档策略 |

表内为代表项，其余详见 [README 索引](README.md#issue-索引) 中同一里程碑的分组。

### M4-Performance · 14 条 —— 有测量支撑的性能跟进

退出条件：每条都有同 workload、同设置的 A/B 数据，并且结论落在"采纳 / 退役"之一；
只有单臂计时或只有一次 ABBA 区组的条目不允许推进到"已采纳"。

| ID | 优先级 | 状态 | 标题 | 收口要点 |
| --- | --- | --- | --- | --- |
| [0004](issues/0004-vsm-directional-projection-retain-candidate.md) | P2 | open | VSM: directionalProjection 停在 retain_candidate，RenderGraph 下界 −2.19% 未过保护门槛，默认值待决策 | retain_candidate 未过保护门槛 |
| [0005](issues/0005-vsm-compact-build-and-attribution-unconfirmed.md) | P2 | open | VSM: compact/indirect build 两次实验都未确认收益，需先拆 finalize 与逐页计数原子成本再决定去留 | 两次实验都未确认收益 |
| [0008](issues/0008-vsm-shadow-lod-family-default-off.md) | P2 | open | VSM: Shadow LOD 全家仍默认关闭，Refined receiver 暖缓存 +471%，运动与大场景质量未验证 | Shadow LOD 全家默认关闭 |
| [0010](issues/0010-vsm-arc-eight-ray-and-mapping-counters.md) | P2 | open | VSM: ARC 已按设备能力默认开启，但只有 2-ray 普通计时；8-ray 与 mapping 计数组合未测 | ARC 只有 2-ray 数据 |
| [0015](issues/0015-streamer-merged-demand-track-cost.md) | P2 | open | Streamer: Track merged demand 仍逐唯一请求查表，S1 延迟跟踪 ≤0.5 ms 目标未达 | S1 延迟跟踪未达标 |
| [0016](issues/0016-streamer-prefetch-watermark-blocked.md) | P2 | open | Streamer: Full 场景预测预取恒被内存水位阻断（预取请求=0），预取与准入缺少拒绝原因分类 | 预取恒被水位阻断 |
| [0019](issues/0019-sw-raster-scanline-and-setup.md) | P2 | open | SW raster: 仍无 scanline / 小 bbox 快路径，96.61% 三角形在 setup 阶段被汇总拒绝 | SW raster 缺快路径 |
| [0024](issues/0024-zorahfull-roam-30fps-gate.md) | P1 | open | ZorahFull: 生产 1440p 漫游三轮均 sampledRouteMeets30Fps=false，长帧只能逐帧归因 | 生产 1440p 30 fps 门禁未达 |
| [0045](issues/0045-material-performance-baseline-instability.md) | P2 | in-progress | Material: 材质性能基线只有单机单 GPU 数据、5 项不稳定且缺硬件计数器与程序规模扫描 | 材质性能基线不稳定 |
| [0049](issues/0049-pathtracing-cold-compile-and-pso-cost.md) | P1 | in-progress | PathTracing: 冷启动仍以单个 Guides PSO 50 s 量级为主成本，模块 IR 复用与 provider 请求扩展未落地 | PT 冷启动 50 s 量级 |
| [0061](issues/0061-shader-warmup-cold-cache-target.md) | P2 | open | ShaderWarmup: 冷缓存全目录预热约 56 s，未达到 10 s 目标 | warmup 冷缓存未达 10 s 目标 |

### M5-Research · 8 条 —— 长期探索

退出条件：要么转成有验收标准的 feature，要么明确写入"不做"及原因。

| ID | 优先级 | 状态 | 标题 | 收口要点 |
| --- | --- | --- | --- | --- |
| [0011](issues/0011-vsm-directional-local-parity-ledger.md) | P3 | open | VSM: 已固化的特化全部只覆盖方向光，局部光/混灯没有对等台账，两项对等候选停在「保留后续」 | 方向光/局部光对等台账缺失 |
| [0012](issues/0012-vsm-mark-empty-tile-and-cross-light-cost.md) | P3 | open | VSM: 方向光 Mark 的空 tile 与跨光重复准备成本未测量，唯一同类候选已被否决 | Mark 空 tile 成本未测 |
| [0020](issues/0020-stream-lod-cross-instance-traversal.md) | P2 | proposed | Streamer: 安全 frontier 仍每实例一个工作组，跨实例持久遍历队列未实现（空间 demand 候选已全部撤回） | 跨实例持久遍历队列未实现 |
| [0025](issues/0025-meshlet-lod-metric-and-encoding-research.md) | P3 | proposed | Meshlet LOD: 误差度量与 Nanite/参考不可直接比较，更紧的保守投影界从未量化 | LOD 误差度量不可比 |
| [0026](issues/0026-terrain-cluster-pipeline-feasibility.md) | P3 | proposed | 地形: GPU-Driven cluster 化 M0–M2 未开工，且可行性文档的技术前提已被统一流式 VBuffer 取代 | 地形 cluster 化未开工 |
| [0040](issues/0040-dlss-rr-psr-guide-gap.md) | P3 | proposed | DLSS-RR: PSR 与 path regularization 未实现，guide 输入与参考样本不对齐 | PSR / regularization 未实现 |
| [0042](issues/0042-material-packed-closure-split-backend-deferred.md) | P2 | proposed | Material: Phase 11 之后的 packed closure / split 调度无实现，M5 存储与调度决策缺证据 | packed closure / split 调度无实现 |
| [0064](issues/0064-nsight-headless-source-export.md) | P2 | blocked | Nsight: Source/IL 无头导出仍无验证后端，源码分析依赖交互 UI | Nsight Source/IL 无头导出无后端 |

## 四条跨里程碑主线

里程碑按"什么时候能算完"分组，主线按"为什么它们是一件事"分组。同一条 issue 只出现
在一个里程碑里，但可能同时属于多条主线。

### 主线 A：证据链可信度（最高杠杆）

**目标**：随机抽一条性能或质量结论，任何人能在 1 小时内独立复核。

[0059](issues/0059-test-skip-semantics-baseline.md), [0060](issues/0060-verification-matrix-ci-coverage.md), [0030](issues/0030-nsight-capture-runtime-integrity.md), [0031](issues/0031-aftermath-auto-checkpoints-under-injection.md), [0032](issues/0032-nsight-command-buffer-recycling.md), [0033](issues/0033-gputrace-d3d12-module-pin.md), [0064](issues/0064-nsight-headless-source-export.md)
（采集与工具链）

[0002](issues/0002-vsm-workload-diagnostics-abi-doc-drift.md), [0044](issues/0044-material-m2-doc-vs-code-drift-sparse-program-bins.md), [0052](issues/0052-hdr-output-calibration-and-doc-drift.md), [0058](issues/0058-rendergraph-self-submission-gpu-timing-doc.md)

理由：这一组是所有其他结论的前置条件。11-01 的 VSM 寄存器归因正是因为
NSight 抓取写入失败（[0030](issues/0030-nsight-capture-runtime-integrity.md)）而无法完成；而 skip 不等于通过（[0059](issues/0059-test-skip-semantics-baseline.md)）、
`metallic-ci` 只跑三个测试目标（[0060](issues/0060-verification-matrix-ci-coverage.md)）意味着"测试全绿"本身就是弱证据。
**先修证据，再谈性能**，否则后续所有采纳决策都要重做。

### 主线 B：默认关闭候选的去留（决策债）

**目标**：每个 benchmark 开关都有 owner、结论与登记位置，`git grep benchmark` 不再
出现"没人知道还要不要"的开关。

[0001](issues/0001-vsm-propagated-mapping-default-flip.md), [0003](issues/0003-vsm-projection-statistics-candidate-retirement.md), [0004](issues/0004-vsm-directional-projection-retain-candidate.md), [0005](issues/0005-vsm-compact-build-and-attribution-unconfirmed.md), [0006](issues/0006-vsm-cache-layering-perf-unmeasured.md), [0007](issues/0007-vsm-mark-page-shift-projection-regression.md), [0008](issues/0008-vsm-shadow-lod-family-default-off.md), [0010](issues/0010-vsm-arc-eight-ray-and-mapping-counters.md), [0023](issues/0023-texture-feedback-wave-not-accepted.md), [0043](issues/0043-material-ray-material-queue-unused-in-production.md), [0050](issues/0050-pathtracing-experimental-backends-unqualified.md), [0011](issues/0011-vsm-directional-local-parity-ledger.md), [0012](issues/0012-vsm-mark-empty-tile-and-cross-light-cost.md)

理由：这些条目有共同的失败模式——候选实现完整、开关齐全、有部分测量，但缺一次
"采纳或退役"的正式结论。它们持续消耗后续每一次批处理的判断力，也让性能对比失去
可比基线的定义。

### 主线 C：半成品接入生产

**目标**：已实现的机制要么真正被默认路径使用，要么明确标注为实验设施。

[0041](issues/0041-material-closure-mix-layer-backend-gap.md), [0042](issues/0042-material-packed-closure-split-backend-deferred.md), [0043](issues/0043-material-ray-material-queue-unused-in-production.md), [0047](issues/0047-material-graph-sdk-frontend-scope-limits.md), [0048](issues/0048-fiber-strands-rt-and-model-coverage.md), [0066](issues/0066-sky-atmospheric-pass-parked-branch.md), [0067](issues/0067-geometry-meshlet-compression-parked-branch.md), [0055](issues/0055-rendergraph-read-visibility-frontier.md), [0057](issues/0057-rendergraph-subresource-and-ownership.md)

### 主线 D：首帧与冷启动

**目标**：从进程启动到可交互帧的时间可预测、可取消、可归因。

[0018](issues/0018-streamer-initial-load-main-thread-sync.md), [0017](issues/0017-streamer-root-cut-full-rebuild-pending.md), [0049](issues/0049-pathtracing-cold-compile-and-pso-cost.md), [0061](issues/0061-shader-warmup-cold-cache-target.md), [0062](issues/0062-shader-warmup-graphics-pso.md), [0024](issues/0024-zorahfull-roam-30fps-gate.md)

理由：[0018](issues/0018-streamer-initial-load-main-thread-sync.md)（主线程同步 7.7 s）、[0049](issues/0049-pathtracing-cold-compile-and-pso-cost.md)（PT 单 PSO ≈50 s）、
[0061](issues/0061-shader-warmup-cold-cache-target.md)（warmup ≈56 s）是同一个用户体验问题的三个切面：谁在启动路径上同步
阻塞、谁在做不可复用的编译。分开修会互相掩盖。

## 依赖链

只有真实的阻塞关系才登记为 `depends_on`；"相关"不算。

- [0001](issues/0001-vsm-propagated-mapping-default-flip.md) → [0002](issues/0002-vsm-workload-diagnostics-abi-doc-drift.md)、[0007](issues/0007-vsm-mark-page-shift-projection-regression.md)、[0010](issues/0010-vsm-arc-eight-ray-and-mapping-counters.md)
  默认集合未定，则计数器口径、Mark 特化回归归因、ARC 与 mapping 计数组合都无法定稿。
- [0017](issues/0017-streamer-root-cut-full-rebuild-pending.md) → [0016](issues/0016-streamer-prefetch-watermark-blocked.md)
  根页不降到池容量之下，预测预取会一直被水位拒绝，预取策略无从评估。
- [0041](issues/0041-material-closure-mix-layer-backend-gap.md) → [0042](issues/0042-material-packed-closure-split-backend-deferred.md)
  Mix/Layer 执行后端不闭环，packed closure 的存储与调度就没有收益依据。

## 建议的执行顺序（未来两个季度）

1. **[0059](issues/0059-test-skip-semantics-baseline.md) + [0060](issues/0060-verification-matrix-ci-coverage.md)**：把"测试是否真的通过"和"CI 是否真的覆盖"变成
   事实。这是唯一能同时提升其余 61 条条目可信度的动作。
2. **[0001](issues/0001-vsm-propagated-mapping-default-flip.md)**：唯一一个"代码与文档互相矛盾"的活跃项，且它阻塞三条依赖。
   决策成本低、收益立即体现。
3. **[0030](issues/0030-nsight-capture-runtime-integrity.md)**：修好抓取完整性后，VSM 侧被卡住的寄存器/占用率归因可以一次补齐
   （[0008](issues/0008-vsm-shadow-lod-family-default-off.md)、[0010](issues/0010-vsm-arc-eight-ray-and-mapping-counters.md) 都在等同一类数据）。
4. **[0069](issues/0069-build-nrd-submodule-mapping.md) + [0068](issues/0068-build-branch-and-stash-archival.md)**：仓库卫生类小改动，成本低，防止新参与者踩坑。
5. 之后按里程碑推进：M2 先清"已复现但无人跟踪"，M3 补基建，M4 只接受有 ABBA 数据的
   结论，M5 保持低成本调研。

## 复审规则

| 时点 | 动作 |
| --- | --- |
| 一条 issue 关闭时 | 在 `Roadmap.md` 对应里程碑表里删除该行，或在正文记录迁移原因 |
| 每两周 | `python -B Tools/PublishGitHubIssues.py --check`：evidence 路径失效即修；新出现的 TODO / 失败测试补条目 |
| 每个里程碑收尾 | 按该里程碑的"退出条件"逐条核对，未达成的迁移到下一里程碑并写明原因 |
| 每季度 | 重跑证据采集（未提交改动、未合并分支、roadmap 文档里的"后续"段落），把新形成的挂起工作并入台账 |

台账的价值来自**可核对**。任何一条 issue 的 `evidence` 都应该能被重新打开验证；
一旦 evidence 失效，就要么更新它，要么关闭它。
