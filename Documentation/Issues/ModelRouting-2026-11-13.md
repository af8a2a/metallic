# Metallic 开放 issue：难度、工作量与模型路由

评估日期：2026-11-13（Asia/Shanghai）。范围：`af8a2a/metallic` 当日开放的 63 个 issue，GitHub 编号 #9–#71，不含 PR。仅做分诊，不修改远端标签、状态或代码。以 GitHub 开放状态为准，结合仓库 issue 台账和关键源码静态核查；不是逐条复现报告。

## 1. 模型基准与适用边界

查询的是 Artificial Analysis Intelligence Index **v5.0.1**；分数不是百分比、修复成功率或 Vulkan 专项成绩。相差两分不应被解释为稳定、显著的能力差距。

| 用户可用模型 | 本次对应的 AA 版本 | 智能指数 | 路由用途 |
| --- | --- | ---: | --- |
| GPT6 Astra | [GPT-6 Astra max](https://artificialanalysis.ai/models/gpt-6-astra) | 57 | D5 的方案、根因推理与独立审查；不默认承担所有编码 |
| GPT6.1 Sol | [GPT-6.1 Sol max](https://artificialanalysis.ai/models/gpt-6-1-sol) | 55 | D3–D4 主力；D5 按已审查方案分阶段实现 |
| DeepSeek4.1 1017 | [DeepSeek V4.1 **Flash** 1017 max](https://artificialanalysis.ai/models/deepseek-v4-1-flash-1017) | 33 | D1–D2、局部且有确定测试 oracle 的任务 |
| Gemini 3.8Flash | [Gemini 3.8 Flash high](https://artificialanalysis.ai/models/gemini-3-8-flash) | 31 | D1–D2 文档、脚本、数据整理与受约束实现；与 DeepSeek 可替换 |

**型号假设：**用户的 DeepSeek 简称按 AA 的 V4.1 Flash 1017 处理；若实际部署不是此版本，不能沿用该分数。[DeepSeek high](https://artificialanalysis.ai/models/deepseek-v4-1-flash-1017-high) 为 29，不推理版本在[厂商汇总页](https://artificialanalysis.ai/models/creators/deepseek)为 19。[Sol high](https://artificialanalysis.ai/models/gpt-6-1-sol-high) 为 50；不能把 max 分数套到任意推理档位。AA 综合指数包括编码、科学和知识工作等九项评测，详见[方法说明](https://artificialanalysis.ai/methodology/intelligence-benchmarking)。本报告的具体模型分配是工程判断，不是 AA 对 Metallic 的测试结论。

## 2. 分级规则

### 难度（独立于优先级与工作量）

- **D1 机械型：**事实已知、局部文档/配置更改、验证方式明确。
- **D2 局部工程：**单一契约、小范围实现或测试，预期结果可直接判定。
- **D3 跨模块：**需要沿调用链理解多个模块、增补集成测试或稳定测量。
- **D4 专家型：**GPU 生命周期、并发、编译/缓存身份、数值/时序或跨后端语义。
- **D5 开放型：**根因不明、算法/架构选择、驱动/工具交互、需要多轮实验排除假设。

### 工作量

以熟悉项目的工程师完成“定位 + 实现 + 回归 + 证据”的**人力等效投入**粗估，不是 AI 运行时承诺：**S ≤0.5 人日；M >0.5–2 人日；L >2–5 人日；XL >5 人日，应先拆分**。硬件/厂商等待另计；测量型任务通常依赖指定 GPU、场景资产及工具。很多 issue 是能力建设或采纳决策，不是改几行代码就能完成的 bug。

### 调度原则

1. **Sol 默认，Astra 按需升级。**Astra 优先投入同步/驱动未知根因、算法决策与高风险审查；只高两分，不值得把所有任务整体迁走。
2. **Flash 与 DeepSeek 属于相近档。**本报告的区分偏向工作类型，不代表已经证明某者 C++ 或文档能力更强。
3. 低档模型只拿有明确范围、禁止事项和可执行验收的子任务。完整 issue 含 GPU 验证时，不能靠完成文档部分宣布关闭。
4. 每项先确认目标分支与当前实现。本地评估基线：`modernization_/vsm`、`866eba50a`（2026-11-11）；线上多数 issue/台账源于 2026-11-01，可能部分过时。
5. 现有 `priority:P1/P2/P3` 保留；优先级回答“先做什么”，难度回答“交给谁”，两者不等价。

## 3. 全量分类

### 一键选模型的队列摘要

按下表第一责任人计数（合计 **63**，编号 **#9–#71** 无缺漏/重复；角色可在拆分后变化）：

- **DeepSeek max：2 项** — #10、#71。D2/M；以权威事实和确定性验证为边界，Sol 复核。
- **Gemini Flash high：1 项** — #47。D2/M；完成事实清单后纯编辑可降 D1/S。可与 DeepSeek 互换，不把两分差距当决定性优势。
- **Sol：43 项** — #9、#11、#12、#13、#14、#15、#17、#18、#19、#20、#21、#22、#23、#26、#28、#29、#30、#32、#33、#34、#35、#36、#37、#41、#42、#48、#49、#52、#53、#55、#56、#57、#58、#59、#61、#62、#63、#64、#65、#66、#67、#68、#70。主要 D3–D4，M–XL；#26/#32 若转入新算法/架构实现，升级 Astra。
- **Astra 牵头：17 项** — #16、#24、#25、#27、#31、#38、#39、#40、#43、#44、#45、#46、#50、#51、#54、#60、#69。主要 D5/XL；#40 是 L 级有界诊断，不承诺厂商修复；#16 先让 Sol 做 M 级范围审计。

这里不是 17 个 issue 从头到尾都用 Astra：先用它冻结根因/方案/验收，再让 Sol 编码。全部交最高模型成本不合理；全部按 issue 标题交便宜模型，也会把文档票中的 GPU/生命周期工作漏掉。

表中“主模型”负责该行范围；Astra 行通常由其定方案/审查，再由 Sol 分段编码。**★** 表示当前实现已明显超前于旧 issue，先审计余额；**阻**表示原台账 blocked，不等于本地 AI 可以消除阻塞。标题为简写。验证栏仅列关键门槛，不替代原 issue 完整验收。

| GitHub issue / 工作范围 | 原优先级 | 难度 | 工作量 | 主模型 | 判断理由 / 关键验收 |
| --- | --- | --- | --- | --- | --- |
| [#9★ propagated mapping 标准路径验收](https://github.com/af8a2a/metallic/issues/9) | P1 | D3；补矩阵 D4 | M 审计；L 补验收 | Sol | 已固化且删除开关，先核对 mixed/local/pass 覆盖与当前整图/额外映射内存；不恢复旧 Off/On/Toggle。 |
| [#10★ workload / Build ABI 文档消歧](https://github.com/af8a2a/metallic/issues/10) | P2 | D2 | M | DeepSeek max，Sol 审查事实 | 按当前 workload v6/31、Build v7/60 与 parser 表更新当前布局和历史版本说明；不全局删历史 work16，不退回旧 v3。 |
| [#11 Projection 统计候选保留/退役](https://github.com/af8a2a/metallic/issues/11) | P2 | D3 | M；GPU 验收可 L | Sol | 涉及设置、Pass、cache identity、shader、日志、测试；11/6 已有过滤兼容修复，非字符串清理。撤除须默认图像精确，保留须标注期限。 |
| [#12★ directional projection 状态与证据](https://github.com/af8a2a/metallic/issues/12) | P2 | D3；性能 D4 | M 审计；L 重测 | Sol | 11/4 标准 uniform 实现与 11/9 新基线已超前；先重写剩余验收，10 月性能不能为新实现背书，不复活开关。 |
| [#13 compact build 采用/归档](https://github.com/af8a2a/metallic/issues/13) | P2 | D4 | L | Sol，Astra 审查 append/同步 | 先拆 finalize/native/诊断成本，再选单候选；精确回归、预声明 Build 门槛；预算耗尽可记录“未证明有益”。 |
| [#14 cache layering 收口](https://github.com/af8a2a/metallic/issues/14) | P2 | D4 | L | Sol | 需要 caching，仍默认关闭；先解决 #17，多光/冷热/动态 promotion ABBA，包含 atlas 旧代在途显存；memory 字段勿重复相加。 |
| [#15 Mark page-shift 的 Projection 回退归因](https://github.com/af8a2a/metallic/issues/15) | P2 | D4 | L | Sol | Projection SPIR-V 相同不支持指令因果；先重测当前基线，再分 prepare/dispatch/publish/scheduling，旧复现检出旧版本而非复活设置。 |
| [#16★ Shadow LOD 质量与 warm 成本](https://github.com/af8a2a/metallic/issues/16) | P2 | D5 优化；审计 D3 | XL 优化；M 审计 | Astra，Sol 先审计范围 | 多个旁路已删，仅功能开关仍关闭；先核对当前移动/resize/大场景与 warm 门槛余额，再做高数值/驻留风险优化；不用 erosion 掩盖 miss。 |
| [#17 VSM profiler section 溢出](https://github.com/af8a2a/metallic/issues/17) | P2 | D4 | L | Sol | 共享 CPU/GPU query 预算及生命周期，不只是提高 256 常量；至少双点光 12 faces，overflow/缺 Projection 必须显式拒绝，不能补零。报表 fixture 可给 DeepSeek。 |
| [#18 ARC 8-ray 默认路径验收](https://github.com/af8a2a/metallic/issues/18) | P2 | D4 | L | Sol | 2-ray 证据不能外推 8-ray；三轮普通 ABBA、图差位置/幅度、联合 mapping 质量；原质量豁免不无限扩展。 |
| [#19 方向光/局部光能力与成本矩阵](https://github.com/af8a2a/metallic/issues/19) | P3 | D3 | L | Sol | 需源码支持的研究，不是改文案；按当前标准路径列共享/特化/fallback，光锥覆盖不同不能做严格因果归因。表格整理可给 Flash。 |
| [#20 Mark 空 tile / 跨光成本研究](https://github.com/af8a2a/metallic/issues/20) | P3 | D3 | L | Sol | 先构造可验证空 tile 计数与 generate/sync/consume 全成本；Projection 被否决实验不证明 Mark。先 go/no-go，不改默认。 |
| [#21 Streamer 延迟统计热路径](https://github.com/af8a2a/metallic/issues/21) | P2 | D4 | L | Sol | 已有优化仍未达门槛；resident lookup 复用须保持 cancel/retry/pageID 复用生命周期和统计语义，固定路线压力/吞吐回归。 |
| [#22 预测预取与余量准入](https://github.com/af8a2a/metallic/issues/22) | P2 | D4 | L | Sol | 先细分水位/碎片/CLAS/upload 拒绝原因，再用 #23 新 cache 测实际收益；不靠增预算达标。counter 子任务 D3/M。 |
| [#23 Full 保底集全量 recook](https://github.com/af8a2a/metallic/issues/23) | P2 | D3 | L | Sol | 算法/版本已完成，重点为 Full 全量 cook、属性/覆盖/cache 验证及两轮漫游；小 probe 不能外推，长 cook 墙钟不全算人日。 |
| [#24 初始加载可交互与取消](https://github.com/af8a2a/metallic/issues/24) | P1 | D5 | XL | Astra | 主线程 pump/UI 状态机与 accepted GPU drain、未提交取消、reset/replace 契约；进度/取消/编译并行拆分，量连续阻塞与首图。 |
| [#25 SW raster setup / 微三角快路径](https://github.com/af8a2a/metallic/issues/25) | P2 | D5 | XL；诊断 M | Astra；Sol 先诊断 | SW32 已默认；先分类 setup reject，单候选实验、稳定 triangleID/depth/visibility 精确与边界 fixture，不放宽 ULP。 |
| [#26 streaming LOD frontier 负载分布](https://github.com/af8a2a/metallic/issues/26) | P2 | D4 评估；D5 新队列 | L 评估；XL 实现 | Sol 测量，Astra 定算法 | 已有 demand bounded queue，缺的是安全 frontier 负载策略；保持父依赖、稳定输出、完整 cut 与容量 fallback，覆盖 WG 边界/288 帧 LDS。 |
| [#27 CLAS 模板化 Shadow 回归](https://github.com/af8a2a/metallic/issues/27) | P1 | D5 | XL | Astra | 构建收益不等于 Shadow 收益；C1–C5 未过关，需 BVH/cache 证据与安全 bounds，精确 HDR/depth/shadow；允许结论为否决。 |
| [#28 纹理上传 U2 合批](https://github.com/af8a2a/metallic/issues/28) | P2 | D4 | XL | Sol，Astra 审查同步 | copy/barrier 合批与池、完成点、取消/失败生命周期一起处理；先测量再合批，验证 mip/hash/峰值内存与端到端交互。 |
| [#29 textureFeedbackWave 采纳决策](https://github.com/af8a2a/metallic/issues/29) | P3 | D4 | L | Sol | 微基准不代表生产；活跃 demand 漫游、至少三轮 ABBA、95% CI、键基数与精确 HDR/temporal/guide；可限定域或退役。 |
| [#30 Full 1440p 30FPS 长尾](https://github.com/af8a2a/metallic/issues/30) | P1 | D4 | L 归因；优化另拆 XL | Sol | 集成验收伞，不是单补丁；三轮固定路线和质量，逐长帧归因及显存预算；P99 达标不等于零超预算帧。 |
| [#31 meshlet 保守投影误差界](https://github.com/af8a2a/metallic/issues/31) | P3 | D5 | XL | Astra | 数值几何证明；near plane/jitter/shear/nonuniform scale，统一像素阈值对照 HDR/depth/visibility；容量减少不冒充性能。 |
| [#32 地形 cluster 可行性](https://github.com/af8a2a/metallic/issues/32) | P3 | D4 评估；D5 实现 | L 评估；XL 实现 | Sol 评估，Astra 定架构 | 旧前提已变化；先源码化缺口清单与 go/no-go，heightfield 探针验缝/skirts/error；离线和 GPU 生成拆票。 |
| [#33 VVL payload 误报与 stride 策略](https://github.com/af8a2a/metallic/issues/33) | P2·阻 | D4 | L | Sol，必要时 Astra | literal stride 关联合约与 cache ABI，不机械删除；合法/超限 payload、heap 矩阵与真实上游版本验证。 |
| [#34 Nsight Low integrity 采集](https://github.com/af8a2a/metallic/issues/34) | P1 | D3 | L | Sol | staging/preflight/manifest 与实际子进程 TokenIntegrityLevel；不能只看历史 JSON，验证 capture/replay，不无故提升权限。 |
| [#35 Nsight/Aftermath checkpoint 冲突](https://github.com/af8a2a/metallic/issues/35) | P2·阻 | D4 | L | Sol，未知根因升 Astra | 现有隔离不能证明根因；退役前覆盖异步 BLAS/OMM/compaction/TLAS、capture/replay，保留版本与 override 边界。 |
| [#36 Nsight 命令池寿命 workaround](https://github.com/af8a2a/metallic/issues/36) | P2·阻 | D4 | L | Sol | 不直接删除延迟回收；三次 capture、resize、replay loop、外部注入/普通运行；工具等待另计。 |
| [#37 GPUTrace D3D12.dll PIN](https://github.com/af8a2a/metallic/issues/37) | P2·阻 | D4 | M | Sol | 移除几行容易，卸载/注入/退出证明难；三 trace、resize/exit、失败退出不 hang、disabled 不加载，未获上游证据则保留。 |
| [#38 native resident 细分图像差异](https://github.com/af8a2a/metallic/issues/38) | P1 | D5 | XL；最小 probe L | Astra；Sol 做 probe | mapped 恢复仅强相关；逐 texel/mesh/heap/index probe 与 SPIR-V，完整矩阵/live edit，不按入口名绕路、不把 timeout 当 pass。 |
| [#39 AS handle 统一 heap index](https://github.com/af8a2a/metallic/issues/39) | P2·阻 | D5 | XL | Astra | driver uint64 load 与 Slang hook 两阻塞；address-cell PoC 可独立推进；B/8 与 B/16 单位、多个 AS、替换/refit/compaction/在途生命周期。 |
| [#40 跨配置 cache DeviceLost](https://github.com/af8a2a/metallic/issues/40) | P1·阻 | D5 | L 有界证据；厂商修复不定 | Astra；Sol 自动化复现 | app PSO/driver cache 分离、首次编译顺序反转、extension/feature 与 key 矩阵；nonce 不是修复，已知 hang 配置不进入日常回归。 |
| [#41 native alpha coverage 红项](https://github.com/af8a2a/metallic/issues/41) | P2 | D4 | M | Sol | native/mapped 都失败；先 materialID/winner 隔离填孔来源，再修分类或有几何依据的阈值，MASK/ray/shadow/BLEND 不退；不先放宽阈值。 |
| [#42 Streamline descriptor workaround](https://github.com/af8a2a/metallic/issues/42) | P2·阻 | D4 | M | Sol | 精确 SDK/driver 验证后再移除 legacy set/rebind；真实 SR/RR 非黑图、Reflex 与生命周期；上游 closed 不等于本地安全。 |
| [#43 DLSS RR PSR / guides](https://github.com/af8a2a/metallic/issues/43) | P3 | D5 | XL；coverage 调查 M | Astra | 提案非回归；先 GPU coverage 再有意义的反射 prototype，radiance/guide 共享路径契约，参考图/时域/disocclusion/GPU 验证；零激活可建议推迟。 |
| [#44 Mix/Layer 物理后端](https://github.com/af8a2a/metallic/issues/44) | P1 | D5 | XL | Astra | 双 basis、介质/涂层 eval/sample/pdf/互易与 white-furnace 需 reference；先澄清双 basis 预算矛盾，不把 diffuse 近似当完整后端。 |
| [#45 packed / split backend 采纳](https://github.com/af8a2a/metallic/issues/45) | P2 | D5 | XL；观测 M | Astra | 先 #44、观测与窄 DGC A/B，再决定整套 wavefront；历史 split 更慢且增内存，允许否决，不为阶段编号强实现。 |
| [#46 RayMaterialQueue 生产消费](https://github.com/af8a2a/metallic/issues/46) | P2 | D5 | XL | Astra | classifier 测试不等于 secondary-hit 调度；fullProgramKey/generation/RNG/path 状态、stale/overflow 整帧语义，先 opt-in parity 再性能。 |
| [#47★ M2 sparse-bin 文档](https://github.com/af8a2a/metallic/issues/47) | P2 | D2；纯编辑 D1 | M；纯编辑 S | Flash，Sol 审查事实 | 当前 sparse 默认已开；逐项 delivered/superseded/remaining 与源码链接、二次复核；仍保留 coneLOD/UV seam/grazing/offscreen 等未交付范围。 |
| [#48 材质性能基线稳定性](https://github.com/af8a2a/metallic/issues/48) | P2 | D4 | L | Sol | 同 workload/cache/clock，NaN/spike 与至少一组 counter、program×instance 规模；不把预算拒绝当支持。报告制图可给 Flash。 |
| [#49 材质回归可信度](https://github.com/af8a2a/metallic/issues/49) | P2 | D4 | XL | Sol；核心同步/数值升 Astra | live edit、热重载删文件、HDR suite 漂移与 NTC alpha 分票；三次全 suite/隔离运行及 winner 证明，不放宽误差或删断言。 |
| [#50 MaterialGraph / SDK 范围](https://github.com/af8a2a/metallic/issues/50) | P2 | D5 | XL | Astra | closed grammar 不是任意 Slang 平台；custom closure 跨 IR/schema/ABI/PT/VBuffer。先确定注册机制，多材质 apply/undo/export 可拆 D3/L。 |
| [#51 Fiber / Strands RT 覆盖](https://github.com/af8a2a/metallic/issues/51) | P1 | D5 | XL | Astra | screen-space 不等于曲线 RT transport；intersection/模型/阴影/时域拆分，端点/邻接/亚像素/root-tip GPU，先限定 coverage 等价比较范围。 |
| [#52★ PT 冷编译与 PSO 成本](https://github.com/af8a2a/metallic/issues/52) | P1 | D4；完整缓存重构 D5 | XL | Sol；重构设计升 Astra | compiler/module/codegen/downstream/native 分层且不相加嵌套时间，现已有 snapshot/session 演进；如改持久 IR，cache key/deps/ABI 必须完整。 |
| [#53 实验 PT backend 采纳/退役](https://github.com/af8a2a/metallic/issues/53) | P2 | D4；未知 driver/codegen D5 | L 决策；根因修复 XL | Sol | 逐 backend 允许 adopt/diagnostic/close+reopen；同条件 ABBA、guide/HDR/ULP 与新增 HitObject 四阶段证据，不必须修好所有实验路径。 |
| [#54 ACEScg / NRD / EXR 色彩边界](https://github.com/af8a2a/metallic/issues/54) | P2 | D5 | XL | Astra | AP1→Rec709 与 vendor sanitize/负分量；native AP1 不只是删矩阵。拆 denoiser 契约、EXR float/chromaticities 与 MaterialX 支持/拒绝表。 |
| [#55 HDR 能力与校准](https://github.com/af8a2a/metallic/issues/55) | P2 | D3；完整多窗口 D4 | L；完整多窗口 XL | Sol | HDR10/PQ 已实现，文档过时；最小可接受为人工/半自动校准和 detached SDR-only 声明；物理 HDR 三 profile/nit 验收不能由离屏读回替代。 |
| [#56 checkpoint pass 覆盖](https://github.com/af8a2a/metallic/issues/56) | P2 | D4 | L | Sol | 逐 pass provider 的资源、layout/copy 时点、dynamic rendering 外执行；双 checkpoint 独立 capture/probe/offline 与新 pass E2E，不是重复注册。 |
| [#57 编辑器调试面板与 watch](https://github.com/af8a2a/metallic/issues/57) | P2 | D4 | XL | Sol | capabilities/jobs/results/errors/artifacts 与有界 watch、reload/stale job 生命周期；先拆 UI 状态 D3 与控制协议 D4。 |
| [#58★ writer 可见性剩余验收](https://github.com/af8a2a/metallic/issues/58) | P2 | D4；齐证后状态同步 D2 | M | Sol | writer/layout/readers 已实现；补核专项 compute→fragment 像素链、生产图 inventory 与文档。不是已确认当前仍有原缺陷。 |
| [#59 HZB 第二帧 InvalidArgument](https://github.com/af8a2a/metallic/issues/59) | P1 | D4 | L | Sol | alias 开/关和旧 pass 都复现，先定位 history generation/view/output 生命周期；两帧 GPU 像素与 alias parity，不能新增 skip 冒充修复。 |
| [#60 subresource / exclusive ownership](https://github.com/af8a2a/metallic/issues/60) | P2 | D5 | XL | Astra | range overlap、同 allocation views、mip/layer/aspect、初始化证明、release/acquire/import/alias；拆范围模型→planner/backend→外部资源，传参数不等于自动分析。 |
| [#61 多队列 GPU timing 文档与回归](https://github.com/af8a2a/metallic/issues/61) | P2 | D3 | M | Sol | 已有三类队列 timing，Tracy 仍 graphics-only；含 async/copy/overflow/editor GPU 回归。极短负载判 availability，不盲断言时间大于零；文档子项给 Flash。 |
| [#62 skip 与已知失败闭环](https://github.com/af8a2a/metallic/issues/62) | P1 | D3 | L | Sol | tb/gtest/CTest 的 skip 语义不同；能力/版本 pass-skip-fail 矩阵和 strong lane，不能新增 skip 即绿、泛化 xfail 或降断言。 |
| [#63 CI 验证矩阵](https://github.com/af8a2a/metallic/issues/63) | P1 | D4 | L | Sol | 当前 CI preset 仅 Scene/Task/Debug；补 editor/rhi/tools/perf、JUnit、显式 GPU lane 与 #62 策略；CTest 不构建目标，不只是 YAML。 |
| [#64★ 冷预热速度与归因](https://github.com/af8a2a/metallic/issues/64) | P2 | D4 | L | Sol | 当前已有 session/snapshot 演进；重新分阶段、新 cache、jobs 4/8/12 各三次；可接受“目标不可达”的有界结论，硬保证 5 倍提速则 D5/XL。 |
| [#65 graphics PSO warmup 覆盖](https://github.com/af8a2a/metallic/issues/65) | P2 | D4 | XL | Sol | pass-owner 真正 render-state collector、ImGui ABI、single-flight/cache 格式；首帧 graphics create=0 仅针对声明场景，不猜状态或无界枚举。 |
| [#66 metallicctl perf 协议与编排](https://github.com/af8a2a/metallic/issues/66) | P2 | D4 | XL | Sol | 冻结 capabilities/case manifest 与四态能力，贯通 collect/analyze/错误/过期；引擎稳定协议、外层 Python 编排，先一条真证据链。 |
| [#67 Nsight Source/IL 无头导出](https://github.com/af8a2a/metallic/issues/67) | P2·阻 | D4 | L | Sol | 公开 SDK 缺 producer API；四个带 hash/进程/window 证据的有界实验，可行则同 trace 三次，否则 vendor 问题；CLI metrics 不能冒充 Source/IL。 |
| [#68 大气天空分支收敛](https://github.com/af8a2a/metallic/issues/68) | P2·决策阻 | D4 重集成；决策 D2 | L 集成；M 决策 | Sol | graph/editor/NaN/flicker/时域图像；仅归档或另开后续任务可降级；“看到天空”不等于物理/时域正确，等待维护者采纳决策。 |
| [#69 meshlet 压缩分支收敛](https://github.com/af8a2a/metallic/issues/69) | P2·决策阻 | D5 | XL | Astra | 编码/量化/tangent/BLAS 生命周期与真实资产误差；先拆 scratch 生命周期和误差契约。无独有 commit 不等于语义已验收。 |
| [#70 分支 / stash 归档](https://github.com/af8a2a/metallic/issues/70) | P3 | D3 | L | Sol；Flash 只读盘点 | 各项 provenance/语义与 owner 决策，不是删至指定数量；旧 9/10 快照须重盘点，保全备份，合并另拆 #68/#69，禁止自动删除。 |
| [#71 NRD 子模块映射](https://github.com/af8a2a/metallic/issues/71) | P1 | D2 | M；单配置改动 S | DeepSeek max | 核对 canonical URL 与 gitlink，干净 recursive clone/init 和 NRD-enabled build；本地 External 已填满的构建不算干净克隆验证，不碰 microprofile dirty。 |

### 低档模型的安全子任务池

不要为平均分配而让四个模型各接四分之一整票。对复杂票，保留 Sol/Astra 责任人，再切出这些可验收子任务：

| 模型 | 可委派子任务 | 必须先给出的约束 |
| --- | --- | --- |
| Flash | #19 能力表整理；#48 既有测量的图表/报告；#55/#61 已核实事实的文档同步；#70 只读分支/stash 清单 | 上级给定权威源码/版本、数据字段和结论边界；不能自行宣布性能收益、删除数据或关闭票 |
| DeepSeek max | #10 ABI 版本 fixture；#17 overflow 报告 fixture；#62 pass/skip/fail 汇总与样本测试；#66 CLI 参数/协议错误路径测试 | 先冻结 schema、错误语义、预期输出；不能擅改共享 RHI/调度/生命周期契约；Sol 审查集成 |

## 4. 静态核查发现

- **#9、#12 的旧修复方向已过时：**[propagated mapping 当前状态](../VirtualShadowMapPropagatedMapping.md#L3-L6)与[directional projection 当前状态](../VirtualShadowMapDirectionalProjection.md#L3)明确 2026-11-04 已固化标准实现并删除旧设置/benchmark 开关。应复核剩余验收证据与台账，不重新添加旧开关；#12 的 10 月旧实验不能证明 11 月新实现的收益。
- **#10 不能机械同步为 v3/23：**[当前常量](../../Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h#L44-L84)为 31 项 workload、60 项 Build；[最新文档](../VirtualShadowMapsGPUProfile.md#L3-L5)是 workload v6、Build v7，但[底部旧布局](../VirtualShadowMapsGPUProfile.md#L73-L79)仍写 16/25。任务应更新为版本化 ABI 文档消歧，不能退回旧验收数字。
- **#71 仍有直接证据：**[子模块配置](../../.gitmodules)没有 NRD 映射，而 `git ls-files --stage External/nrd` 仍有 gitlink。适合先修；源仓库 URL/固定提交必须确认，不能猜依赖版本。
- **#58 不应从零重写：**[访问规划器](../../Source/Runtime/Render/RenderGraph/RenderGraphAccessPlan.cpp#L14-L63)已存在 `MemoryProducer`、writer 与按队列记录的可见性；[生产者消费逻辑](../../Source/Runtime/Render/RenderGraph/RenderGraphAccessPlan.cpp#L347-L417)也保留 layout producer 与 reader frontier。[对应测试](../../tests/rhi/RenderGraphAccessPlanTests.cpp#L134-L164)明确 `requiresDevice=false`，覆盖 compute 写→compute 读→fragment 读的规划契约。另有[GPU fanout/readback 用例](../../tests/rhi/RenderGraphAccessPlanTests.cpp#L753-L816)，覆盖六消费者、跨/同队列及连续提交，但其存在不证明本轮已运行，也不自动证明原票要求的 compute→fragment 像素路径完整覆盖。因此应先盘点“已实现/剩余 GPU 与生产图证据”的差额；不能将 CPU 契约测试写成 GPU 验证通过，也不能据此直接关闭 issue。
- **#61 不是纯文档任务：**[验收条目](issues/0058-rendergraph-self-submission-gpu-timing-doc.md#L71-L82)还要求异步队列 GPU 计时、query 槽位耗尽与编辑器统计验证；文档部分可交低档模型，整项不能按 D1。
- **#52、#64 的编译实现也在演进：**[当前编译器](../../Source/Runtime/Render/Core/SlangCompiler.cpp#L1200-L1227)已有 `SourceEpochPool`/snapshot，并有[条件 session 共享](../../Source/Runtime/Render/Core/SlangCompiler.cpp#L1284-L1297)；不能将旧票“每请求都新建 session”当作普遍现状。先采集新冷启动分阶段基线，区分模块加载、Slang 编译、SPIR-V 与 native PSO。
- **#70 涉及用户数据：**分支/stash 的盘点与归档建议可低成本完成；合并、删除分支或清空 stash 必须先由用户明确选择，不能为了满足旧验收数量直接执行。

## 5. 执行顺序与升级规则

1. **低成本入口：#71 + #10 + #47。**先恢复可复现依赖初始化、校正文档事实。#70 只并行做只读盘点和保全/归档方案，不自动清空 stash。
2. **先让验证可信：#62 → #63。**先区分 pass/skip/fail，再建立 CPU 和真正执行 GPU 的独立 lane；无 GPU 的 CI 不冒充渲染验收。
3. **解除证据链阻塞：#34。**在不扩大系统权限、不关闭安全边界的前提下定位采集子进程完整性/输出路径问题，再补硬件计数器与 Nsight 证据。系统配置改动单独确认。
4. **活跃主线先查余额：#9、#12、#58。**先做当前代码/历史证据/旧 issue 的差额表，再决定补验收、更新描述或建议关闭；不能依据本报告自动关闭。
5. **正确性优先于新的优化：#38、#41、#49、#59。**先最小化稳定失败与跨路径差异；Astra 诊断，Sol 实现，保留原图像/HDR/时域证据。#40 的 DeviceLost 单独按驱动/cache 配置矩阵复现。
6. **再推进生产化与性能。**先冻结场景/相机/配置/commit、明确冷暖缓存，至少重复 ABBA 并报告置信区间；scope 级收益不写成整帧收益。#30 是集成验收伞，不是一个可以独立一次性“修好”的补丁。

### 硬依赖（来自台账，执行前复核是否仍成立）

- **#9 → #10 / #15 / #18：**原依赖源于默认集合未定；当前 #9 已固化实现，先核对版本和验收余额，不机械阻塞后续。
- **#23 → #22：**先降低/重构 root-cut 全量构建与驻留压力，再评估内存水位下的预取；仅加预取计数不等于实现有效预取。
- **#44 → #45：**先有闭环 Mix/Layer 执行后端，再决定 packed closure 存储与 split 调度。
- **#34 → 多项性能归因：**这是建议的工具准备顺序，不是台账声明的硬依赖。

### 并行边界

可并行文档清理、测试报告基建、独立 UI、厂商最小复现。共享 VSM shader/Programs、RenderGraph 同步计划、材质 evaluator、编译请求/缓存身份的任务不要无协调同时修改。#52/#64/#65 先共同盘点编译/PSO阶段，避免三路各做一套缓存。XL issue 先拆“oracle/复现→最小实现→生产验收”，主模型保持契约所有权，低档模型只承担独立可验证部分。

### 每次交给 AI 的任务模板

> 处理 Metallic GitHub #N。先确认该问题在目标分支仍存在；报告当前实现与 issue 的差额，不重复实现已完成部分。只修改指定子系统，保留无关工作树改动。先复述调用/生命周期契约和验收 oracle，再实现最小方案。禁止降低阈值、忽略测试、把 skip 当 pass、用单帧 smoke 代替视觉/时序验收。提交实际执行的构建/测试命令、通过/失败/跳过数量、GPU/驱动与不可用验证。若两个有证据的假设均被否定，或需要改变共享 RHI/同步/材质数值契约，停止扩写并给出最小升级包。

升级包包括：最小复现、预期/实际、目标 commit、已排除假设、原始日志/读回、受影响契约、建议下一实验。低档→Sol；Sol→Astra 做诊断/审查，不重复无证据尝试。外部工具/驱动阻塞转为带版本的最小复现与上游报告，而非要求 AI 无限修补。

## 6. 证据与限制

- 线上范围：[开放 issues](https://github.com/af8a2a/metallic/issues?q=is%3Aissue%20is%3Aopen)，GitHub REST 分页读取并与本地发布映射核对。部分 API 正文响应被截断，后续又触及匿名访问速率上限；这些条目的细节以仓库对应完整草稿补齐。并非所有线上最新正文或评论均被完整读取，执行前须复核变更。
- 台账：[索引](README.md)、[编号映射](publish-state.json)、[路线图](Roadmap.md)。本地草稿 ID **不等于** GitHub issue 编号；全文表格使用 GitHub 编号。
- 本轮不编译、不执行 GPU/视觉/性能实验、不修复 issue、不修改远端。难度与工作量是分诊估算，实际复现后应更新。
