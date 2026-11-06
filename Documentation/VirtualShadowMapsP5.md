# Virtual Shadow Maps P5：默认接入候选与验收

对应 [实施方案](VirtualShadowMapsResearch.md) 的 P5，基于 [P4](VirtualShadowMapsP4.md) 的多灯过滤与位移 caster。已新增实时图候选、编辑器入口、紧凑屏幕输出、并行页分配和可复现的验收工具，完成 LookDev / Sponza 各 512 帧完整 DLSS 图回放及各 3 次普通计时；**尚未提升为默认阴影实现**。MiniZorah / Zorah Full 的实际 BLEND 内容、Deferred 法线和 PT preview 回归、参考图质量及生产成本门槛仍未通过。P0–P4 的历史通过记录不代替本阶段验收。

## 候选入口与保留合同

| Sample ID | 图资产 | 用途 |
| --- | --- | --- |
| `realtime-lighting-vsm` | [realtime_lighting_vsm](../Pipelines/Samples/realtime_lighting_vsm.metallic_graph.json) | 通用实时 VSM 候选 |
| `gpu-driven-vsm` | [gpu_driven_realtime_vsm](../Pipelines/Samples/gpu_driven_realtime_vsm.metallic_graph.json) | MiniZorah 自适应主视图候选；受 BLEND 阻断 |
| `gpu-driven-zorah-full-vsm` | [gpu_driven_zorah_full_vsm](../Pipelines/Samples/gpu_driven_zorah_full_vsm.metallic_graph.json) | Zorah Full 流送候选；受 BLEND 阻断 |
| `gpu-driven-tessellation-vsm` | [gpu_driven_tessellation_vsm](../Pipelines/Samples/gpu_driven_tessellation_vsm.metallic_graph.json) | 独立位移 caster 候选 |

[RenderSample](../Source/Runtime/Render/RenderSample.cpp) 中名称均标明 `VSM Candidate`。编辑器 File 菜单提供 `New Realtime Graph (VSM Candidate)`，加载通用候选；通用 `New Render Graph` 和既有 GPUDriven 重置入口维持原来的 PT / RT 图。原 `realtime-lighting`、GPUDriven RT、路径追踪图继续可选，旧的显式 RT 阴影序列化语义没有被 VSM 替换。

候选把 `Shadows` 改为 `VirtualShadowMapLightsPass`，连接 `Shadows.shadows` / `Shadows.lights` 到 Deferred 的 `virtualShadows` / `virtualShadowLights`，不再连接 shadow acceleration structure。VBuffer 显式关闭 `enableClusterRtx` / `enableClas`，位移候选另关闭 `rtTessellation`。保留原主视图的自适应 LOD、几何和纹理流送设置，没有用强制 LOD0 掩盖主视图与独立 caster 的几何差异。

候选使用明确的 PCF 参数：128 texel 页、每轴 128 页（16K 虚拟分辨率）、6 层方向光 clipmap、首层 extent 4 m、bias 0.002 m、全灯物理页预算 2048、lightCapacity 8、`compactOutputs=true`。普通候选的每面 raster work 上限为 1,048,576；位移候选为 4,194,304，caster factor 8、额外 subdivision depth 2。图中仍保留 DLSS-SR Quality 与关闭的 DLSS-NR。

Pass Inspector 可显式选择 Hard / PCF / SMRT，并调节页预算、bias、footprint 与采样数。当前候选选 PCF；P4 曲面 SMRT 的近似误差与噪声尚未达到默认质量门槛，不能因为控件存在就视为 SMRT 已获默认批准。位移 depth 2 增加 16 倍微三角，属于明确的质量成本，不能由分辨率命名推断其生产性能。

## 紧凑输出与内存合同

`VirtualShadowMapLightsPass.compactOutputs` 缺省为 false，旧图继续使用每灯最多六层的屏幕数组。候选显式开启后，每灯只发布一层 `R8` visibility 和一层 `RGBA8` debug，投影直接写 RenderGraph 输出；省去各面私有屏幕纹理及其复制。Point 仍有六个独立深度投影，只有 receiver 所属的 canonical face 写该像素，共享输出层不改变点光跨面查询。

[灯表 ABI](../Source/Runtime/Render/Shadow/VirtualShadowMapLights.h) 的 version 仍为 1，header 和 entry 均为 64 B。`header.reserved[0]` 的 bit 0 表示紧凑输出；`entry.baseLayer` 在紧凑模式为 entry index，旧模式为 entry index × 6。`faceCount` 仍是 1 / 6，`totalFaces` 仍表示实际投影面数。Deferred 按标志消费两种布局，旧图与旧六层接口继续有效。统计布局仍为 `capacity × 6 × 32 uint32`，不能把输出层数当作统计面数。

新增 `Shadows.memory` 为 64 B、8 个 uint64：atlas 字节、私有输出字节、buffer 字节、公开输出字节、保留的 face 数、实际 atlas 页数、两个保留字段。它描述 VSM 当前拥有的逻辑资源；不包含原生分配对齐、已退役但仍在途的资源及其他 pass。关闭或未获准入的灯释放其缓存池，整设备峰值另以 RHI 分配域记录。

公开屏幕输出的逻辑字节为 `width × height × layerCount × 5`。lightCapacity 8 时，1920×1080 紧凑输出为 79.10 MiB，旧六层布局为 474.61 MiB；如果实际内部 extent 为 1280×720，则紧凑输出为 35.16 MiB。DLSS 协商的实际 extent 必须从运行记录读取，最终窗口分辨率不是阴影数组的分辨率证明。这些是布局计算，不是整帧显存实测或总显存降低比例。

2048 个 128² D32 有效页的深度数据为 128 MiB，atlas 排布可能额外取整。每个活动 face 的 1M 项 raster work buffer 为 8 MiB，位移候选的 4M 项为 32 MiB，另有页表、caster、每帧状态和在途资源。`lightCapacity` 必须覆盖全部局部灯 source slots 加 Sun/Moon 两个固定 slot，包括禁用的 source；不能只按活动灯数缩小。1080p 不自动要求增加页池，是否调整由 receiver 需求、coarse fallback、工作溢出和图像证据决定。

## 无 shadow TLAS 的验收边界

[SceneResourceManager](../Source/Runtime/Render/Streamer/SceneResourceManager.cpp) 的历史 `ScenePathTraceResources` 命名不等于无条件构建 RTAS；它也服务于材质与纹理准备。实际 streamer 的 RTAS 路径由 `AccelerationStructure` 资源需求或 `enableClusterRtx` 激活，`rtTessellation` 同样会提升该需求。因此必须检查全图需求与实际分配，而不能只检查 VSM shader 是否含 RayQuery。

验收分成两条路径：

- **严格核心路径**：生成明确标记的 `--core-only` 图，移除 DLSS-SR / NR 并重新接通 Deferred → AutoExposure → FinalBlit；关闭 Streamline，要求设备 RayQuery / RTAS capabilities 均为 false，且 RHI RayTracing / CLAS live 与 peak 分配始终为零。该路径可运行核心与 synchronization validation，但不代表完整候选图。
- **完整候选路径**：保留 DLSS 图，`enableStreamline=true`。Streamline 可能要求启用 RT 设备能力，因此记录实际 capabilities，并要求 RayTracing / CLAS live 与 peak 分配为零，不能把能力位误写成全部关闭。完整图还需满足 SDK 帧生命周期、输出质量与长期内存门槛。

`RenderGraphPreviewRenderer::initialize` 增加末尾可选 `enableStreamline=false` 参数，保持既有调用不变。完整图的验收显式开启，独立 VSM 质量测试继续关闭。

当前 bundled Streamline 的 `vkCreateInstance` 代理只枚举全局 instance extensions，无法识别由 validation layer 提供的 `VK_EXT_layer_settings`；开启显式 synchronization validation 时会在创建设备前失败。这是 [既有互操作限制](AgenticShaderPrintfP2.md)，不是 VSM 验证错误。验收 manifest 分开控制 core validation 与 `synchronizationValidation`，完整图与严格核心图的覆盖范围分别记录；不能把关闭完整图 validation 的结果合称“完整图零 VVL”。

## 验收协议与可复现记录

[PrepareVirtualShadowMapAcceptance.py](../Tools/Perf/PrepareVirtualShadowMapAcceptance.py) 为 LookDev、Sponza、MiniZorah、Zorah Full 生成独立图和 version 1 输入 manifest，不修改原场景或候选资产，也不移除不支持的 BLEND 材质。`Catalog.json` 仅为目录，每场景的 `LookDevCases.json` / `SponzaCases.json` 等文件才是可运行输入。初始输出为 768×512，预热 16 帧，可显式提供相机运动；目标分辨率、相机、resize / cut / scene reload 等事件应在输入 manifest 中保留。

[VirtualShadowMapAcceptanceTests.cpp](../tests/rhi/VirtualShadowMapAcceptanceTests.cpp) 注册 `virtual_shadow_map_acceptance`。每个进程只接受一个 case / device，每帧等待提交完成，输出 `virtual-shadow-map-acceptance-v2` 报告。它不是编辑器 present loop，也不测帧间 CPU/GPU 重叠。当前 Python verifier 明确拒绝 v1 输出，旧包保留其原有证据范围。

`diagnostic` 模式收集阴影统计、灯表、VSM 逻辑内存、真实 VBuffer extent、输出图像和显式 cache / force-redraw 对照。对照必须先确认 depth、visibility 及 streaming semantic / source / topology / primitive 身份一致；输入改变则标为 `incomparable`，不能算作 cache 等价通过。输入一致时要求阴影数组逐字节相等；HDR 一致性另行记录。两次 Deferred HDR 读回均验证实际尺寸、RGBA16/32 float 格式、字节数及全部分量有限性，且必须包含所有声明的比较帧。该检查不覆盖所有帧或 DLSS 输出的有限性。还要求有效的 receiver / 灯准入、零 work overflow、零 invalid coarse page，并把清理阶段纳入 validation 检查。

`timing` 模式关闭 validation、图像与诊断 readback；保留 CPU execution snapshot 来验证实际 raster extent，其 CPU 采集开销必须公开。报告 graph GPU envelope、各 pass GPU 和 CPU recording，不能把 shadow pass 时间称作整帧提速。诊断模式含 validation/readback 的时间不作为性能结论。

[VirtualShadowMapAcceptance.py](../Tools/Perf/VirtualShadowMapAcceptance.py) 串行运行独立进程，记录 exe / DLL、源码与 shader、case / graph 的 hash、commit、工作树状态和缓存策略；缺失、skip、部分完成、输入或二进制变更、运行时 / SDK 错误都不能作为有效验收。资产当前只记录路径，未对大型 cooked geometry / texture 全内容做 hash；也不强制独占 GPU，运行者须记录其他 GPU workload。

独立进程是重复试验单位，连续帧不是独立样本。报告每帧分位数、各进程中位数和 RHI live / peak allocation；RHI/VMA 字节不等于进程 VRAM 或总物理驻留。默认不推导速度提升。RT / VSM A/B 必须额外固定相同灯数、内容、相机、LOD、质量与缓存条件，现有单灯 RT 不能直接与多灯 VSM 作性能胜负比较。

示例命令中的输出目录必须尚不存在：

```powershell
python -B Tools/Perf/PrepareVirtualShadowMapAcceptance.py .cache/vsm-p5-core-cases --mode diagnostic --frames 512 --case LookDev --core-only
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases .cache/vsm-p5-core-cases/LookDevCases.json --exe build-release/tests/MetallicRHITests.exe --output .cache/vsm-p5-core-run --runs 3
python -B Tools/Perf/VirtualShadowMapAcceptance.py report .cache/vsm-p5-core-run .cache/vsm-p5-core-report
```

## 当前验收状态

本节记录 2026-10-27 当前实现的实测，GPU 为 RTX 5070 Ti 16 GB，驱动 621.01。测试串行运行，未关闭用户程序、未锁定频率，也没有系统级 GPU 独占保证。完整图使用 Release / Streamline 开启 / NRD 关闭；独立回归另使用 Release / NRD 开启 / Streamline 关闭。用户清理构建目录后，已重新构建 `build-release` 的 RHI / ShaderRequests 测试目标；最终 v2 包均来自重建后的二进制。两个 Release 变体构建成功；RelWithDebInfo 在受限环境触发 MSVC C1902，未算作通过。

| 项目 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 四个候选图、入口与 no-RTAS 资源需求检查 | 通过 | `render_graph_sample_load` 覆盖图连接、PCF、compact、关闭 RT 需求及旧 RT sample 保留 |
| 并行 allocator | 32 个配置 / 重复通过 | 与独立串行 CPU 策略逐页、逐 owner、全部 32 项统计精确一致；最大 196,609 页、4096 槽；core / sync VVL 零消息 |
| compact / legacy 质量及 ABI 对照 | 24 帧通过 | 三类灯 × Hard / PCF / SMRT、点光 seam / corner、禁用和零预算；合并旧 canonical 输出后逐字节相同，私有输出为零、公开输出为旧布局 1/6 |
| filter demand / cache | 30 / 52 帧通过 | demand 覆盖冷 / 热 / 移动 / coarse / 恢复；过滤 coarse query 为零；缓存逐字节重画对照通过 |
| 严格无 RT 能力核心图 | LookDev / Sponza 各 64 帧通过 | RayQuery / RTAS caps 关闭，RT / CLAS live / peak 零，core / sync VVL 零消息；旧 v1 包不包含新增 HDR 有限性凭据 |
| 含 DLSS 完整图回放 | LookDev / Sponza 各 512 帧通过 | v2；core VVL 零消息，sync validation 关闭；SDK 错误零；详见下节 |
| 含 DLSS 完整图普通计时 | 每场景 3 × 256 帧通过 | 每进程预热 16 帧，240 帧计入统计；无 validation / 图像 readback，SDK 错误零；没有 RT 对照或提速结论 |
| 含 DLSS 完整图 ordinary baseline | **128 帧 × 3 次原始记录失效** | `.cache/vsm-p5-timing-baseline/` 的 RHI 测试退出成功，但出现 SDK `presentCommon() was not observed` 错误 |
| 代表性 MiniZorah / Zorah Full | **被实际 BLEND 内容阻断** | 不是未使用材质导致的误拒，见下节 |
| 当前代码 P4 / P3 回归 | quality 83 帧、content 42 帧、local lights 88 帧通过 | 重跑当前实现；这些矩阵不代替代表性长期回放 |
| CPU / shader 验证 | 28 项 focused CPU 测试、14 个运行时 SPIR-V 通过 | 26 项静态及 2 项实际 shader 编译；`spirv-val --target-env vulkan1.4 --scalar-block-layout`；module ownership 通过 |
| 既有 RT / PT 回归 | **部分通过，2 项未解决失败** | RT+SIGMA、material shadow / transmission、天体固定 slot 通过；Deferred 法线与 PT preview 失败，见下节 |

完整图 baseline 暴露了离屏 Streamline 生命周期缺口：preview 不执行真实 present，未调用仓库已有的 `notifyStreamlineOffscreenFrame()`。bundled SDK 的 `presentCommon()` 每帧推进内部记账和 GC，`afterPresentCommon()` 回收帧末 tag；超过 60 次 evaluate 未见 present 会报错。现有 helper 通过 common plugin 导出的 Vulkan hooks 支持空 swapchain，不提交真实 present。本轮在 preview 的有 / 无 readback 两条 GPU 完成路径接入一次帧结束通知，仅在显式启用 Streamline 时调用并传播失败，不修改 SDK 或真实 present 路径。修复后的两套 512 帧回放及六次 256 帧普通计时均未记录该 SDK 错误；旧 baseline 保留且不参与有效统计。

本轮需求标记收敛到实际 Hard / PCF / SMRT tap 使用的起始层级，避免把已由 footprint 选择的粗层需求额外铺到不使用的细级；保底粗页仍存在。页分配改为五阶段 GPU 批量处理，保持原有确定性准入与物理映射规则。allocator 的 32 组精确 oracle、30 帧 demand 测试及真实场景缓存 / 重画对照均通过。旧 LookDev 诊断中 requested 从 16,485 降至 111、拒绝需求从 14,935 降至零，新长程的 filtered coarse query 全帧为零；这是需求正确性与覆盖变化，不能作为同质量性能 A/B。

本轮还修正了一个旧 CPU 测试的失效预期：`LocalMaterialProgramsWarmBeforeRuntimeCreatesTheInclude` 从 3 改为 2 个预热请求。现有 Deferred global / local view 已由 runtime flag 选择，不再生成不同的 shader macro key；请求合并后是一个 CLAS PT 和一个 flat Deferred。仍保留实际 request 存在、编译及 cache-hit 检查。诊断 semantic capture 的 texture sampling metadata 由 `VisibilityBufferPass` 显式请求 `textureFeedback`，使消费者读取前有对应 producer；不改变普通图的默认反馈开关。

## 完整图回放与成本记录

最终证据为 [LookDev 512 帧](../.cache/vsm-p5-lookdev-long-v2/run-1/VirtualShadowMapAcceptance.json) 和 [Sponza 512 帧](../.cache/vsm-p5-sponza-long-v2/run-1/VirtualShadowMapAcceptance.json)，各为一个独立进程。输出 768×512、DLSS Quality 的实际内部尺寸为 512×341，相机周期 128 帧、运动幅度 0.05。LookDev 在 128 / 256 帧切到 960×640（实际 640×427），192 / 320 帧恢复，384 帧执行 scene reload 与 camera cut。Sponza 本批没有 resize / reload。两场景的第 16 / 511 帧均满足输入身份稳定，缓存与强制重画的 shadow 和 Deferred HDR 逐字节相同，HDR 非有限分量均为零；全程 RT / CLAS live 与 peak 分配、work overflow、filtered coarse query 均为零。

| 回放 | 同状态稳定窗口（帧，左闭右开） | 窗口内 live RHI 分配 | 全程峰值 |
| --- | --- | ---: | ---: |
| LookDev | [208,256)、[336,384)、[400,500) | 每帧均为 1,113,543,568 B（1061.96 MiB） | 1751.51 MiB，reload 时几何资源暂时重叠 |
| Sponza | [32,500) | 每帧均为 1,466,245,632 B（1398.32 MiB） | 1436.56 MiB |

该结果证明所测窗口回到同一分配水平，不代表无限时长、所有场景或物理 VRAM 稳定。图表及原始逐帧数据见 [LookDev 回放报告](../.cache/vsm-p5-lookdev-long-report/report.md) / [Sponza 回放报告](../.cache/vsm-p5-sponza-long-report/report.md)。已打开检查最新 [LookDev 图像](../.cache/vsm-p5-lookdev-long-v2/run-1/LookDev-511.png) 和 [Sponza 图像](../.cache/vsm-p5-sponza-long-v2/run-1/Sponza-511.png)；LookDev 球面仍有明显斑块，没有同状态 RT / PT 参考来确定归因，视觉质量不判定通过。

普通计时使用相同输出 / 内部尺寸、开启 temporal jitter，每场景 3 个独立进程，每次 256 帧，仅排除声明的前 16 帧预热。保留已有磁盘缓存，每进程重新创建 preview；没有清缓存冷启动性能结论。以下均为各进程的逐帧中位数，单位 ms：

| 场景 | 整图 GPU，运行 1 / 2 / 3 | Shadows GPU，运行 1 / 2 / 3 | 整图 CPU 录制，运行 1 / 2 / 3 |
| --- | --- | --- | --- |
| LookDev | 2.107 / 2.246 / 2.214 | 0.691 / 0.690 / 0.686 | 32.421 / 31.393 / 34.121 |
| Sponza | 1.770 / 1.742 / 1.745 | 0.230 / 0.236 / 0.233 | 43.641 / 42.793 / 43.036 |

本轮 P5 基线的 Shadows CPU 录制中位数为 LookDev 29.56–32.32 ms、Sponza 41.05–41.88 ms。随后 [resident caster CPU 优化](VirtualShadowMapsCPU.md) 已通过独立 ABBA 对照收敛该成本，原始数字保留作历史记录。GPU envelope 不能替代 CPU / presentation / 帧间重叠，也不能据此推算实时 FPS。验收仍采集 CPU execution snapshot，不能把这套离屏 CPU 数字称为普通编辑器开销；但也没有证据把 Shadows 内部的全部耗时归因于该采集。当前没有同条件 RT A/B，不作 RT / VSM 提速或生产成本达标判断。

完整统计、三条进程曲线和复现脚本分别保存在 [LookDev 计时报告](../.cache/vsm-p5-lookdev-timing-report/report.md)、[Sponza 计时报告](../.cache/vsm-p5-sponza-timing-report/report.md)。四份报告均已使用当前 v2 verifier 重新检查包内文件集合 / SHA256、进程退出、帧数、比较凭据与错误日志，并实际生成、查看 PNG。报告目录保留 `Summary.json`、逐帧 JSON、`plot_report.py` 及 Python / numpy / matplotlib 版本；诊断与普通计时分图，逐帧 P95/P99 只描述本次回放，不作为独立重复实验。

## 回归和生命周期未解决项

`visibility_buffer_deferred_openpbr` 仍报告 4153 个比较像素、3812 个 outlier、mean 40.582230，与 [P2 记录](VirtualShadowMapsP2.md) 数值一致。该重复不建立“P5 前已存在”的独立 HEAD 基线，也未证明 LOD 或其他根因。`render_graph_openpbr_pathtracing_sample_preview` 在回归批次和独立 NRD Release 复测均发生 SEH `0xc0000005`，未取得可归因的崩溃栈。失败保留，不修改阈值或跳过后宣称通过。

scene / local lights 首轮遇到 SourceEpoch 变化，恰逢工作区清理；在源目录外保存输出、重新独立运行后均通过（local lights 88 帧）。有效复测与 PT 失败日志分别为 [VSM 复测](C:/Users/11252/.codex/visualizations/2026/10/22/01a162b8-d664-75e2-9c47-391ccd9ba656/vsm-p5/vsm-recheck.log) 和 [PT 复测](C:/Users/11252/.codex/visualizations/2026/10/22/01a162b8-d664-75e2-9c47-391ccd9ba656/vsm-p5/pt-recheck.log)。其余原始回归保留在 `.cache/vsm-p5-regression/`；allocator、demand/cache、CPU 与 SPIR-V 证据分别保留在 `.cache/vsm-p5-allocator/`、`.cache/vsm-p5-demand-cache/`、`.cache/vsm-p5-cpu-{static,compile}.log`、`.cache/vsm-p5-spirv-v3/Validation.json`。

另一次同进程先 LookDev 后 Sponza、销毁并重建设备的尝试出现 `0xc0000409`；独立 Sponza 通过。bundled SDK 的静态 helper 函数地址在 `slShutdown` 卸载插件后可能失效，但没有崩溃栈确认根因。当前协议因此显式限制每进程一个 case / device，同一个 preview 内的 scene reload 已测。这个限制不能视为同进程设备重建问题已修复，失败包 `.cache/vsm-p5-final-smoke/` 保留。

## BLEND 是实际内容阻断

按 glTF 默认 scene 的节点可达性、mesh primitive 引用及可见性扩展检查，结果如下。这里的“可见”指场景可见标记，不表示当前相机像素中一定出现；三角数也是原始 glTF 实例数，不是 cooked terminal geometry 数。

| 场景 | BLEND 材质 | 活动 scene 中引用 | 代表内容 |
| --- | ---: | --- | --- |
| MiniZorah | 30，全部被引用 | 34 个不同 mesh primitive，606 个 primitive 实例，605 个节点；无 `KHR_node_visibility=false` | Dirt decal、sub-UV reflection plane、透光灯罩、fog card |
| Zorah Full | 1，被引用 | mesh 294 / primitive 0，30 个 Plane 节点 / 实例；无 `KHR_node_visibility=false` | `M_Subuvreflections_01_6aed7b6831f7a263` |

MiniZorah 的代表引用包括：material 46 `MI_Dirt_AtlasCorner_Decal_A2` → mesh 41 / primitive 0，31 个实例；material 95 `M_Subuvreflections_01` → mesh 99 / primitive 0，35 个实例；material 208 `M_EmissiveSphericalLamp_Shadow` → mesh 206 / primitive 0，72 个实例；material 415 `MID_M_BP_EasyFog_0` → mesh 532 / primitive 0，71 个实例。Full 的该 primitive 为两个原始三角，实例化后共 60 个三角，但少量实际 BLEND 也必须遵守材质语义。

准确拒绝边界在 [VirtualShadowMaps::record](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp)：当前遍历材质表，遇到非 OPAQUE / MASK 返回 Unsupported，并记录 `VSM supports opaque/MASK caster Coverage with UV0 alpha textures.`。`gatherMaterialKeys` 只是收集缓存 key，不是 BLEND 拒绝点。遍历全表理论上也会拒绝未引用材质，但这两个目标场景的 BLEND 都有实际活动引用，缩小扫描范围不会解除阻断。没有静默丢弃这些 primitive、把 BLEND 当成 OPAQUE，或篡改资产来制造通过。

## 默认提升仍需满足的门槛

1. 明确 BLEND / 透射内容的真实阴影策略，保持现有材质语义，并使 MiniZorah / Zorah Full 候选完整运行；当前旧 RT / PT 图继续保留，不需要为了 VSM 候选改写它们。
2. LookDev / Sponza 已完成本轮完整离屏图的 512 帧回放，LookDev 已验证重复 resize / scene handoff 的分配回收；进一步解决同进程 Streamline 设备重建失败，并完成真实 editor present loop 的验收。旧 baseline 不能复用为修复后的证据。
3. 将验证扩展到代表性 MiniZorah / Full 场景和目标输出 / 内部分辨率，包括灯光与素材变化、流送细节迟到及位移回放；当前小分辨率 LookDev / Sponza 证据不能外推至 1080p 或 Full 场景。
4. 对独立 caster 与自适应 receiver 的几何差异、页 / clipmap / point seam、有限分辨率及曲面 SMRT 近似取得可接受的参考图证据；P4 中尚存的限制不因新增候选而消失。
5. 定位并收敛当前 Shadows CPU 录制成本，完成相同 workload / 相同灯数的 RT 或 PT 质量参考，以及 RT / VSM 成本对照。报告整图、shadow pass、CPU 与内存各自范围；当前没有生产提速结论。
6. 解决当前复现的 `visibility_buffer_deferred_openpbr` 法线对照与 `render_graph_openpbr_pathtracing_sample_preview` SEH 失败，不能以旧记录或修改预期掩盖未解决失败。
7. 无 shadow TLAS 的整图资源需求、运行时分配、验证层覆盖与设备能力报告一致。完整图的 SDK / validation 互操作限制须显式列出，不与严格核心测试合并为更强结论。

只有这些门槛有当前代码的实测证据后，才修改默认实时图或既有新图默认行为。本轮交付是候选接入、资源成本收敛和验收基础设施。
