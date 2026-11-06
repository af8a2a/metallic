# Metallic Issue 台账 / Issue ledger

这个目录把仓库里**正在推进**和**尚未解决**的功能与缺陷，固化成可以直接发布到
GitHub 的 issue 草稿，作为长期跟进的主干清单。它不是设计文档，也不是 roadmap
的替代品：文档描述"想做什么、怎么验证"，这里描述"现在欠了什么、下一步谁来做、
怎么算做完"。

## 目录结构

| 路径 | 作用 | 谁维护 |
| --- | --- | --- |
| `ISSUE_CONVENTIONS.md` | 起草规范：front matter 字段、标签集、里程碑决策表、质量门槛 | 人工 |
| `issues/NNNN-slug.md` | 单条 issue 草稿（YAML front matter + 正文） | 按子系统分派 |
| `manifest.json` | 机器可读清单，由 `--sync` 从草稿生成，**不要手改** | 脚本生成 |
| `README.md` | 本文件；下方索引表由 `--index` 生成 | 脚本生成索引 |
| `Roadmap.md` | 长期里程碑路线图与跨里程碑依赖链 | 人工 |
| `publish-state.json` | 发布后 id → GitHub issue number 映射（`--apply` 生成） | 脚本生成 |

## 工作流

```powershell
# 1. 改完草稿后重建 manifest 并校验（evidence 路径必须真实存在、标签必须自洽）
python -B Tools/PublishGitHubIssues.py --sync
python -B Tools/PublishGitHubIssues.py --check

# 2. 刷新 README 里的索引表
python -B Tools/PublishGitHubIssues.py --index

# 3. 预览将要创建的内容（默认 dry run，不需要登录）
python -B Tools/PublishGitHubIssues.py

# 4. 真正推送（需要 gh 登录，或设置 GH_TOKEN）
gh auth login
python -B Tools/PublishGitHubIssues.py --apply            # 只补建缺失的 issue
python -B Tools/PublishGitHubIssues.py --apply --update   # 同时刷新已发布 issue 的正文
```

发布是**幂等**的：每条 issue 正文末尾带一行隐藏标记
`<!-- metallic-issue-id: NNNN -->`，脚本据此识别已发布的条目，不会重复创建。
`depends_on` 会在第二轮编辑中渲染成 `Blocked by #<number>` 区块。

仓库文件编号（`NNNN`）与 GitHub issue 编号是两套体系：前者稳定、用于跨文件引用，
后者由 GitHub 分配，记录在 `publish-state.json` 里。

## 发布状态（2026-11-01）

- 63 条已全部发布到 `af8a2a/metallic`，占用 **#9–#71**（#1–#7 是历史 PR，#8 是发布前的
  试写，已删除后重新整批发布）。
- 17 个标签与 5 个里程碑已按本文档的分类创建；每条 issue 都带
  `<!-- metallic-issue-id: NNNN -->` 标记与 `Blocked by #N` 依赖块。
- 校验方式：`manifest.json` 的标题、里程碑、标签集与线上逐条比对，63/63 一致。
- 后续同步：改完草稿跑 `--sync --check --index`，再用 `--apply --update` 覆盖正文；
  只改一条时也可以直接 `gh issue edit <N> --body-file <file>`。

## 标签与里程碑

- `area:<shadow|geometry|rhi|material|editor|tooling|build>`：子系统归属，一条 issue 只取一个主值。
- `kind:<defect|feature|perf|research|task>`：问题性质。
- `priority:<P0|P1|P2|P3>`：`P0` 阻塞全部工作或正确性已坏；`P1` 近期处理——阻塞他人工作，
  或正在推进且需要决策；`P2` 已规划，下一步明确；`P3` 探索性。优先级表示**现在要不要动手**
  的排序，严重程度写在 issue 正文里，两者不互相替代。
- `state:<in-progress|blocked>`：只有这两种状态要加标签；`open` / `proposed` 不加。

里程碑承载长期阶段，不用标签：

| 里程碑 | 含义 |
| --- | --- |
| `M1-Active` | 当前正在推进，需要收口或决策 |
| `M2-Correctness` | 正确性、确定性、验证可信度 |
| `M3-Hardening` | 生产化：默认开启、内存与生命周期、工具与测试基建 |
| `M4-Performance` | 有测量支撑的性能跟进 |
| `M5-Research` | 长期探索，尚无实现 |

评选规则见 [ISSUE_CONVENTIONS.md](ISSUE_CONVENTIONS.md) 第 4 节。

## 本轮整理的证据来源

1. **未提交的工作树改动**（`git status` / `git diff`）：正在推进的候选实现。
2. **未合并的本地分支与 stash**（`git rev-list --count modernization..<branch>`）。
3. **子系统 roadmap / next-steps / gap-analysis / closeout 文档**里明确写着
   "未验证 / 未采纳 / 默认关闭 / 待测量 / 后续"的条目。
4. **代码与测试中的客观信号**：`TODO`/`FIXME`、默认关闭的 benchmark 开关、
   被 skip 或按能力门控的测试、`Documentation/` 中自带 `Result.json` 的实测结论。

## Issue 索引

<!-- issue-index:begin -->
### M1-Active · 8 条

| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| [0001](issues/0001-vsm-propagated-mapping-default-flip.md) | #9 | P1 | in-progress | VSM: propagatedMapping 工作树默认值已翻转为开启，缺默认路径验收、单面 Pass 开关与文档同步 | — |
| [0021](issues/0021-clas-tessellation-template-shadow-regression.md) | #27 | P1 | in-progress | CLAS: 位移曲面模板实例化默认仍为 triangles，Shadow pass 回归在 C1–C5 全部候选下都未消除 | — |
| [0023](issues/0023-texture-feedback-wave-not-accepted.md) | #29 | P3 | in-progress | 纹理反馈: textureFeedbackWave 仍默认关闭，生产 A/B 未证明收益（Opaque +16.6%/+36.5%） | — |
| [0041](issues/0041-material-closure-mix-layer-backend-gap.md) | #44 | P1 | in-progress | Material: Closure Mix/Layer 执行后端只有单面漫反射 Slab，M4 有界组合里程碑未闭环 | — |
| [0043](issues/0043-material-ray-material-queue-unused-in-production.md) | #46 | P2 | in-progress | Material: M8 ray-hit 分类队列没有生产消费者，路径追踪仍只有 fused program loop | — |
| [0047](issues/0047-material-graph-sdk-frontend-scope-limits.md) | #50 | P2 | in-progress | Material: MaterialGraph 与受限 SDK 前端只覆盖单材质 Surface 子集，任意原生 Slang 扩展未实现 | — |
| [0048](issues/0048-fiber-strands-rt-and-model-coverage.md) | #51 | P1 | in-progress | Fiber/Strands: 原生曲线只有屏幕可见性后端，缺 RT 曲线求交、Far Field 模型注册与阴影/时域输运 | — |
| [0050](issues/0050-pathtracing-experimental-backends-unqualified.md) | #53 | P2 | in-progress | PathTracing: 五个实验后端（RT pipeline / SER / vkGltf / SimpleBSDF / splitPrimaryGuides）全部默认关闭且未过严格数值门槛 | — |

### M2-Correctness · 11 条

| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| [0009](issues/0009-vsm-marker-section-budget-overflow.md) | #17 | P2 | open | VSM: 多面/多灯下每 face 生成的 marker 会耗尽 RenderGraph 256-section 预算，Projection 归因不可用 | — |
| [0029](issues/0029-vvl-payload-limit-literal-stride-policy.md) | #33 | P2 | blocked | RHI: VVL 误报 task/mesh payload 上限，literal descriptor heap stride 策略被迫常驻 | — |
| [0031](issues/0031-aftermath-auto-checkpoints-under-injection.md) | #35 | P2 | blocked | Tooling: Nsight 注入期间禁用 Aftermath 自动 checkpoint，GPU 进度定位能力缺失 | — |
| [0034](issues/0034-native-resident-tessellation-image-diff.md) | #38 | P1 | open | RHI: native resident 细分渲染图像与参考不一致（32/96 个 resident 组合失败） | — |
| [0035](issues/0035-as-handle-heap-index-unification.md) | #39 | P2 | blocked | RHI: AS handle 无法统一为 heap index —— 驱动 heap 数值 load 错误与 Slang hook native 崩溃 | — |
| [0036](issues/0036-shader-object-pipeline-cache-device-lost.md) | #40 | P1 | blocked | RHI: 跨 shaderObject 配置复用驱动内部 pipeline cache 触发 DeviceLost，根因未确认 | — |
| [0037](issues/0037-native-alpha-mask-coverage-assertion.md) | #41 | P2 | open | RHI: native alpha-mask 覆盖断言持续失败，前表面覆盖数远超上限且原因未证实 | — |
| [0046](issues/0046-material-regression-credibility-gaps.md) | #49 | P2 | open | Material: 材质回归可信度缺口——长期失败用例、未定位的整组 HDR 漂移与 NTC alpha 跨路径不一致 | — |
| [0055](issues/0055-rendergraph-read-visibility-frontier.md) | #58 | P2 | open | RenderGraph: 同状态 read/read 快路径可能丢失 writer 可见性，尚未取得 GPU 级验证 | — |
| [0056](issues/0056-rendergraph-hzb-preview-invalid-argument.md) | #59 | P1 | open | RenderGraph: Visibility 第二帧 HZB preview 稳定返回 InvalidArgument，回归未修复 | — |
| [0069](issues/0069-build-nrd-submodule-mapping.md) | #71 | P1 | open | Build: .gitmodules 缺少 External/nrd 映射，子模块初始化直接失败 | — |

### M3-Hardening · 22 条

| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| [0002](issues/0002-vsm-workload-diagnostics-abi-doc-drift.md) | #10 | P2 | open | VSM: workload 诊断已到 v3/23 计数器，四份文档仍写 v1/16、v2/19 与旧字节布局 | 0001 |
| [0003](issues/0003-vsm-projection-statistics-candidate-retirement.md) | #11 | P2 | open | VSM: Projection 组内统计候选已被判定确认回退，实验开关仍留在两个 Pass 的生产代码里 | — |
| [0017](issues/0017-streamer-root-cut-full-rebuild-pending.md) | #23 | P2 | open | Streamer: 根页仍占 2.956 GiB / 3.5 GiB 几何池，root-reduction cook 后未执行 Full 全量重建 | — |
| [0018](issues/0018-streamer-initial-load-main-thread-sync.md) | #24 | P1 | open | Streamer: 初始加载仍在主线程同步完成（loadingSeconds 7.7 s，无进度/取消），未达可交互目标 | — |
| [0030](issues/0030-nsight-capture-runtime-integrity.md) | #34 | P1 | open | Tooling: Nsight 抓取子进程处于 Low 完整性级别，报告与场景缓存写入失败 | — |
| [0032](issues/0032-nsight-command-buffer-recycling.md) | #36 | P2 | blocked | RHI: 为规避 Nsight 悬垂引用，命令缓冲在抓取期永不释放并全局复用 | — |
| [0033](issues/0033-gputrace-d3d12-module-pin.md) | #37 | P2 | blocked | Tooling: GPU Trace 退出期依赖 D3D12.dll，靠 PIN 住模块维持注入进程存活 | — |
| [0039](issues/0039-streamline-descriptor-heap-workaround.md) | #42 | P2 | blocked | RHI: Streamline 与 VK_EXT_descriptor_heap 状态互斥，靠绑定空 legacy set 换取 DLSS 输出 | — |
| [0044](issues/0044-material-m2-doc-vs-code-drift-sparse-program-bins.md) | #47 | P2 | open | Material: M2 实施记录与代码现状脱节（稀疏 Program 分箱已生产默认开启，文档仍写未实现） | — |
| [0051](issues/0051-color-nrd-exr-ocio-materialx-gaps.md) | #54 | P2 | open | Color: ACEScg 迁移后仍缺原生宽色域降噪、EXR/OCIO 输入与完整 MaterialX 色彩管理 | — |
| [0052](issues/0052-hdr-output-calibration-and-doc-drift.md) | #55 | P2 | open | Output: HDR 输出缺显示器峰值检测与独立窗口 HDR，且 HdrOutput.md 与已实现的 HDR10 路径不一致 | — |
| [0053](issues/0053-debug-control-checkpoint-pass-coverage.md) | #56 | P2 | open | DebugControlPlane: 只有两个 pass 注册 checkpoint，其余 pass 无法 capture 或 probe | — |
| [0054](issues/0054-editor-debug-control-no-igui-entry.md) | #57 | P2 | open | Editor: 调试控制面没有编辑器内入口，生产 watch 仅限一次性专用进程 | — |
| [0057](issues/0057-rendergraph-subresource-and-ownership.md) | #60 | P2 | open | RenderGraph: 精确 subresource 范围与 exclusive queue-family ownership 尚未纳入同步计划 | — |
| [0058](issues/0058-rendergraph-self-submission-gpu-timing-doc.md) | #61 | P2 | open | RenderGraph: 文档仍称 self-submission 无 GPU 计时、跨队列边未支持，与实现不符 | — |
| [0059](issues/0059-test-skip-semantics-baseline.md) | #62 | P1 | open | 测试基建: skip 与通过无法区分，缺少 skip 基线与已知失败跟踪 | — |
| [0060](issues/0060-verification-matrix-ci-coverage.md) | #63 | P1 | open | 构建与验证: 无 CI 工作流，metallic-ci 只跑 scene/task/debug，关键缓存用例仅 Windows | — |
| [0062](issues/0062-shader-warmup-graphics-pso.md) | #65 | P2 | open | ShaderWarmup: graphics PSO 与 ImGui ABI 未预热，bounded 请求集外仍首帧按需编译 | — |
| [0063](issues/0063-metallicctl-perf-command-surface.md) | #66 | P2 | open | Tooling: metallicctl 缺少 perf 命令面与能力状态，实验链固定在 Tools/Perf 脚本 | — |
| [0066](issues/0066-sky-atmospheric-pass-parked-branch.md) | #68 | P2 | blocked | Sky: 大气天空渲染通道停在未合并分支 modernization_/sky，需重新落地或明确废弃 | — |
| [0067](issues/0067-geometry-meshlet-compression-parked-branch.md) | #69 | P2 | blocked | Geometry: meshlet 自适应位置编码与关节切线帧停在未合并分支，测试已就绪但未进主线 | — |
| [0068](issues/0068-build-branch-and-stash-archival.md) | #70 | P3 | open | Build: 9 个长期未合并分支与 10 个 stash 缺少归档策略，工作正在腐烂 | — |

### M4-Performance · 14 条

| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| [0004](issues/0004-vsm-directional-projection-retain-candidate.md) | #12 | P2 | open | VSM: directionalProjection 停在 retain_candidate，RenderGraph 下界 −2.19% 未过保护门槛，默认值待决策 | — |
| [0005](issues/0005-vsm-compact-build-and-attribution-unconfirmed.md) | #13 | P2 | open | VSM: compact/indirect build 两次实验都未确认收益，需先拆 finalize 与逐页计数原子成本再决定去留 | — |
| [0006](issues/0006-vsm-cache-layering-perf-unmeasured.md) | #14 | P2 | open | VSM: 静态/动态深度分层默认关闭，缺普通 GPU A/B，静态 atlas 翻倍的成本未量化 | — |
| [0007](issues/0007-vsm-mark-page-shift-projection-regression.md) | #15 | P2 | open | VSM: Mark 页尺寸特化已固定进生产，但同一轮确认里 Projection 范围 +35% 未归因 | 0001 |
| [0008](issues/0008-vsm-shadow-lod-family-default-off.md) | #16 | P2 | open | VSM: Shadow LOD 全家仍默认关闭，Refined receiver 暖缓存 +471%，运动与大场景质量未验证 | — |
| [0010](issues/0010-vsm-arc-eight-ray-and-mapping-counters.md) | #18 | P2 | open | VSM: ARC 已按设备能力默认开启，但只有 2-ray 普通计时；8-ray 与 mapping 计数组合未测 | 0001 |
| [0015](issues/0015-streamer-merged-demand-track-cost.md) | #21 | P2 | open | Streamer: Track merged demand 仍逐唯一请求查表，S1 延迟跟踪 ≤0.5 ms 目标未达 | — |
| [0016](issues/0016-streamer-prefetch-watermark-blocked.md) | #22 | P2 | open | Streamer: Full 场景预测预取恒被内存水位阻断（预取请求=0），预取与准入缺少拒绝原因分类 | 0017 |
| [0019](issues/0019-sw-raster-scanline-and-setup.md) | #25 | P2 | open | SW raster: 仍无 scanline / 小 bbox 快路径，96.61% 三角形在 setup 阶段被汇总拒绝 | — |
| [0022](issues/0022-texture-upload-batch-and-copy-merge.md) | #28 | P2 | open | 纹理上传: 仍按 128 regions / 64 MiB / 3 批在途合批，U2 批次组织与逐 mip copy 合批未落地 | — |
| [0024](issues/0024-zorahfull-roam-30fps-gate.md) | #30 | P1 | open | ZorahFull: 生产 1440p 漫游三轮均 sampledRouteMeets30Fps=false，长帧只能逐帧归因 | — |
| [0045](issues/0045-material-performance-baseline-instability.md) | #48 | P2 | in-progress | Material: 材质性能基线只有单机单 GPU 数据、5 项不稳定且缺硬件计数器与程序规模扫描 | — |
| [0049](issues/0049-pathtracing-cold-compile-and-pso-cost.md) | #52 | P1 | in-progress | PathTracing: 冷启动仍以单个 Guides PSO 50 s 量级为主成本，模块 IR 复用与 provider 请求扩展未落地 | — |
| [0061](issues/0061-shader-warmup-cold-cache-target.md) | #64 | P2 | open | ShaderWarmup: 冷缓存全目录预热约 56 s，未达到 10 s 目标 | — |

### M5-Research · 8 条

| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| [0011](issues/0011-vsm-directional-local-parity-ledger.md) | #19 | P3 | open | VSM: 已固化的特化全部只覆盖方向光，局部光/混灯没有对等台账，两项对等候选停在「保留后续」 | — |
| [0012](issues/0012-vsm-mark-empty-tile-and-cross-light-cost.md) | #20 | P3 | open | VSM: 方向光 Mark 的空 tile 与跨光重复准备成本未测量，唯一同类候选已被否决 | — |
| [0020](issues/0020-stream-lod-cross-instance-traversal.md) | #26 | P2 | proposed | Streamer: 安全 frontier 仍每实例一个工作组，跨实例持久遍历队列未实现（空间 demand 候选已全部撤回） | — |
| [0025](issues/0025-meshlet-lod-metric-and-encoding-research.md) | #31 | P3 | proposed | Meshlet LOD: 误差度量与 Nanite/参考不可直接比较，更紧的保守投影界从未量化 | — |
| [0026](issues/0026-terrain-cluster-pipeline-feasibility.md) | #32 | P3 | proposed | 地形: GPU-Driven cluster 化 M0–M2 未开工，且可行性文档的技术前提已被统一流式 VBuffer 取代 | — |
| [0040](issues/0040-dlss-rr-psr-guide-gap.md) | #43 | P3 | proposed | DLSS-RR: PSR 与 path regularization 未实现，guide 输入与参考样本不对齐 | — |
| [0042](issues/0042-material-packed-closure-split-backend-deferred.md) | #45 | P2 | proposed | Material: Phase 11 之后的 packed closure / split 调度无实现，M5 存储与调度决策缺证据 | 0041 |
| [0064](issues/0064-nsight-headless-source-export.md) | #67 | P2 | blocked | Nsight: Source/IL 无头导出仍无验证后端，源码分析依赖交互 UI | — |
<!-- issue-index:end -->

## 长期维护节奏

| 频率 | 动作 |
| --- | --- |
| 每次改动一个子系统 | 更新对应 issue 的 `state`/`evidence`；完成项按验收标准核对后关闭 |
| 每两周 | `--check` 保证 evidence 路径仍然有效；把新出现的 TODO/失败测试补成 issue |
| 每个里程碑收尾 | 复核 `Milestone` 内全部条目，未达成的迁移到下一个里程碑并写明原因 |
| 每季度 | 重跑本流程的第 1、2 项证据来源，把新形成的"挂起工作"并入台账 |

台账的价值来自**可核对**：任何一条 issue 的 `evidence` 都应该能被重新打开验证。
一旦某条 issue 的 evidence 已经失效（文件被删、代码被重构掉），就要么更新它，
要么关闭它 —— 不要留下无法验证的条目。
