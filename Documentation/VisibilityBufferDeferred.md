# Visibility Buffer 实时延迟着色

`VisibilityBufferDeferredPass` 是光栅化 Visibility Buffer 的 Compute resolve。
主可见性由 `VisibilityBufferPass` 生成，Deferred 重建表面、求值 OpenPBR 材质，
使用 ClusterLightGrid 直接光、SH 漫反射和预过滤 HDRI 镜面光照，输出未曝光的
Scene Linear Rec.709/D65 HDR。ACES、曝光和显示编码仍在后处理链路中执行。

```text
VBuffer.visibility / depth / rasterInfo / domain
                 ↓
             Deferred ← shadow / shadowParameters（可选）
                 ↓
          AutoExposure → FinalBlit ← ColorGrading.lut
```

## 2026-11-16：可过滤纹理预过滤环境光（采用）

Deferred 的镜面和折射 IBL 现在优先读取 RGBA32F `Texture2DArray`，以硬件双线性
过滤替代每层四次 buffer 读取。仍为八个 256×128 lat-long 层，U repeat、V clamp，
层间粗糙度插值显式执行；最低粗糙度的原始 HDRI 采样、HDR 精度和环境预计算算法不变。
设备缺少 RGBA32F linear-filter/transfer 支持时沿用原 buffer 路径。旧消费者继续使用
buffer；同一 publication 新增约 4 MiB 纹理，只在环境内容发布时复制，不逐帧复制。
物理天空八份缓存最多额外约 32 MiB，另有 HDRI publication；未测动态天空更新开销。

纹理、view 和 sampler 与既有 publication 同生命周期，覆盖 HDRI 发布以及物理天空
复用、取消和淘汰。Scene 资源包由 408 扩至 416 字节，新增 texture/sampler 位于
408/412，旧字段偏移不变；primary-split 与 diagnostic 包扩至 424 字节，后缀移至416。

实验基线为 `80160cc1`，RTX 5060 / 621.01，MSVC Release，Sponza 1920×1080、
SMRT 2×16/history32、分层缓存2048页、6 clipmaps、page128/VPA128，无 jitter。
第16帧冻结流送，计时窗口内两臂均为503页、380401664纹理驻留字节及相同拓扑，
不将冻结解释为全部资源驻留。真实前景为1个 material program，加背景共2个 bin。

两臂使用同一新二进制、相同双资源 publication，仅切换 Deferred 的过滤 shader。
因此下面量到的是**稳定帧采样收益**，不包含新增纹理的分配、发布和复制成本。
固定3组 ABBA、12个串行进程，每次96帧预热、128帧计时，保留全部1536帧；
224–226帧回读在计时窗口之外。保留磁盘缓存，编译与预热排除在计时之外，
未锁频、未强制 GPU 独占；关闭 validation、Nsight 注入和 VSM detailed GPU profile。
收益由每进程中位数的区组 log ratio 计算，95% CI 使用 Student-t（df=2）。

| GPU范围 | 耗时下降点估计 | 95% CI |
| --- | ---: | ---: |
| Deferred shading | 5.77% | [2.05%, 9.36%] |
| Deferred pass | 5.28% | [1.94%, 8.51%] |
| RenderGraph GPU envelope | 2.46% | [1.91%, 3.01%] |

达到预定的 Shading 点估计至少3%、Shading/Deferred收益下界大于零、Graph下界
不低于−2%的门槛，保留候选。Graph 包含 VBuffer、Shadows、Deferred、AutoExposure、
ColorGrading、FinalBlit，不含编辑器呈现；重叠节点不能求和，不外推为端到端帧率。
未采集 native 指令或硬件计数器，不据此断言寄存器、带宽或 TEX 吞吐的变化。

静止、运动、禁用阴影、禁用直射四例的16/47/49/63/79帧，共20个独立 checkpoint
在 core/synchronization validation 下通过，HDR全部有限。诊断显示代理为非负
RGB → `x/(1+x)` → sRGB；最大误差0.226576/255，最差RGB RMSE 0.000509/255，
均低于预声明的2/255与0.15/255门槛。原始HDR最大差异0.042079，alpha无差异。
直射贡献差异最大1.57e−7，阴影贡献差异最大1.88e−6。已查看静态/运动代理图和差分图；
代理不等同于实际 ACES/FinalBlit 图像。36份计时截图也全部通过，各臂重复截图逐字节一致。

新增 `environment_prefilter_texture_sampling` 使用真实发布数据：262144 texel 的
texture.Load及中心采样逐位一致，2048个跨层、接缝、极点、亚texel及HDR高亮样本
满足设备subTexelPrecisionBits与局部对比度推导的误差界。非中心采样不承诺逐位一致。
两臂各五项环境光/阴影历史回归、候选采样测试，以及trusted资源包、云层环境发布、
物理天空取消/淘汰、guide编译检查均通过，无skip和新增validation错误。
70项ShaderRequests/NativeShaderWarmupRequests测试通过；RHI、请求测试与VSMSample
构建通过，shader模块所有权审查通过393文件。未在不支持过滤的设备上实测fallback。

共享包额外用world绑定的Opaque1/Sky验证fused/split：64×64、16帧、8–15帧回读、
第11帧相机变更及逐帧运动，core/synchronization validation通过；HDR满足既有
`2e-5 + 2e-5 × abs(reference)`门槛，七类guide逐字节一致。原path-only配置在
首帧发生SEH访问异常，未完成split；源码指向缺少全局GPUScene场景绑定导致实例
buffer为空，但未取得调试栈，不能断言已确认根因，也不将原测试计为通过。
首次world配置的runtime通过，但检查器发现缺少同步验证配置；保留记录后显式启用
该配置，另存两臂完整结果于`SupplementalPrimaryValidated/`，未改变比较门槛。

一次A臂capture-view回归被源码监视器epoch变化中断，封存哈希未变，停止并行构建后
显式补验通过；原始失败与恢复记录均保留，该补验不是性能补采样。
本机证据在 `build/deferred-ibl-filter-20261116/`：`Plan.json`、`SealedIdentity.json`、
`QualitySummary.json`、`RegressionSummary.json`、`TimingSummary.json`、`TimingAudit.json`、`Decision.json`
及额外兼容验证目录。生成图像、二进制和捕获数据不纳入源码。

## 2026-11-16：shadow-first 直射求值实验（未采用）

以 `80160cc1` 为基线，只在 `VisibilityBufferLighting.slang` 的内置 Deferred
路径中提前读取 visibility：有效灯光的可见度精确为零时，跳过该灯的
`prepared.evalProjected`。其余权重、贡献检查和累加表达式保持原顺序；材质
evaluate/prepare、自发光、IBL、自定义材质与路径追踪不变。候选已完成验证，
但未达到预定性能门槛，生产 shader 已逐字节恢复；没有新增默认路径或开关。

使用同一新构建 Release RHI 程序，RTX 5060 / 621.01，Sponza 1920×1080、
SMRT 2×16/history32、分层缓存2048页、6 clipmaps、page128/VPA128，无 jitter。
固定截图相机，第16帧冻结流送；不将冻结时刻等同于全部资源驻留。两臂开启
core/synchronization validation，静止、运动、禁用阴影、禁用直射四例的
16/47/49/63/79帧 HDR 逐字节一致：20个独立 checkpoint、24份文件比较，
额外4份为重复的最终帧文件，全部有限。已查看候选 HDR 预览。

两臂 `visibility_buffer_deferred_shadow_history` 均通过。另两项原有回归在两臂
复现完全相同的失败，不能计为通过：`visibility_buffer_deferred_openpbr` 的
shadingNormal 对照为4153像素、3812 outliers、mean error 40.582230；
`visibility_buffer_abeautiful_game_transmission` 报告 FP16 显示误差超过2/255。
没有跳过或新增失败；正确性及回归日志未发现 Vulkan validation 错误。

普通计时为预先固定的3组 ABBA、12个进程，每进程96帧预热、128帧计时；
224–226帧 HDR 回读在计时窗口外。所有1536个有效帧保留，磁盘缓存保留，
不锁频、未强制 GPU 独占。以下收益由每进程中位数的区组 log ratio 计算，
95% CI 使用 Student-t（df=2）；负值表示耗时增加的点估计，不代表确认回退。

| GPU范围 | 耗时下降点估计 | 95% CI |
| --- | ---: | ---: |
| Deferred shading | −0.31% | [−4.76%, +3.94%] |
| Deferred pass | −0.29% | [−4.35%, +3.62%] |
| RenderGraph GPU envelope | −1.07% | [−4.57%, +2.32%] |

未达到 Shading 点估计至少3%、Shading/Deferred收益区间下界大于零、Graph下界
不低于−2%的预定门槛。所有普通计时 HDR 比较一致；实际前景 binned shader
编译产物为448252→448320 B，背景保持226776 B。静态产物变化不能证明原生
分支执行、寄存器或性能瓶颈；本轮未采集对应 native 指令和硬件计数器。
不能从旧阴影图的零值覆盖率推导节省比例，也不能推广为所有场景下的性能结论。

首个A进程正常完成，随后报告校验因配置遗漏 `expectedNodeQueues` 而失败。
原始封存包未改动；离线补充源码支持的明确队列合同后复核通过，并继续B02，
没有重跑、丢弃或补采样。A01与B02之间存在校验修复暂停，完整时间戳保留。
实际 Graph 保留 VBuffer、Shadows、Deferred、AutoExposure、ColorGrading、
FinalBlit；后两节点位于 compute 队列。选择 `Deferred.color` 只是增加回读输出，
并不裁剪 presentation 依赖。Graph 不含编辑器呈现，重叠节点不能求和解释。

证据位于本机 `build/deferred-shadow-first-20261116/`：`Plan.json`、
`CorrectnessAudit.json`、`RegressionComparison.json`、`Recovery.json`、
`TimingSummary.json`、`TimingAudit.json`、`Decision.json` 与 `Candidate.slang`。
候选和生成数据保留在忽略目录，不纳入源码。Shader 模块所有权审查通过；
本轮没有新增多灯实景或长期运行性能结论。

## LookDev 单路径观察

在 Render Graph Editor 的 Settings 中，选择 `lookdev-vbuffer` 或任一 Painter / Studio
PT/Deferred 对比场景，然后使用 **LookDev Render Path**：

- **Comparison**：Reference + VBuffer + Deferred，通过 Slider 对比。
- **Path Trace Only**：仅 Reference，直接连接原来的曝光与显示链路。
- **Deferred Only**：仅 VBuffer + Deferred，直接连接原来的曝光与显示链路。

Only 模式会从活动图中移除另一分支和 Slider；不是把分割线移到边缘。
切换保留当前场景、未保存的材质编辑、视口相机和现有 pass 参数，重新编译活动图并重置累积历史。
选择新的兼容材质场景时沿用模式。此选项只适用于 PT/Deferred 对比图；OpenPBR/Standard
双 BSDF 对比图、Fiber 专用图不会被自动改写。

```powershell
.\build-scheduling-release\Source\LookDev.exe --render-path pathtrace
.\build-scheduling-release\Source\LookDev.exe --render-path deferred
.\build-scheduling-release\Source\LookDev.exe --sample studio-white-M05_Fuzz-textured --render-path deferred
```

显式指定 `--render-path` 且未指定 sample 时，默认采用 `lookdev-vbuffer`。
可选值为 `comparison`、`pathtrace`、`deferred`；不指定参数保留原来的启动行为。
Graph Save 保存的是当前活动图；需要保留原始对比资产时，请先更换保存路径。
重新加载保存的单路径图可直接使用，但该文件不含已移除的分支；恢复对比应重新选择原始 sample。

验证入口：RHI `lookdev_render_path_contract` / `lookdev_render_paths`，以及 LookDev
`--smoke-test` 配合 `METALLIC_SMOKE_TEST_LOOKDEV_PATHS=1`。
RHI 渲染测试默认采用 shaderball，可用 `METALLIC_LOOKDEV_PATH_SAMPLE` 指定本地 Painter / Studio sample；
检查 32 帧执行记录、线性 HDR 有限性及与对比分支的逐字节一致性，输出独立图和显示 PNG。

## 光栅路径边界

- Deferred shader 不执行 ray query，不使用 TLAS，也不包含路径积分器。
  删除了主命中续追、随机环境光遮挡查询、逐灯阴影查询、体积吸收和多次散射续追。
  这不等于全局场景加载已完全免除 RTAS 分配：当前 `SceneResourceManager` 在支持光追的
  resident 设备上仍会预建完整场景资源；Deferred 本身不再声明或绑定该资源。
- 阴影仅消费显式连接的 `shadow` / `shadowParameters`。未连接时按无遮挡处理，
  不再隐式启动光追阴影。输入使用 SIGMA 的 sqrt(visibility) 编码，
  只影响参数指定的稳定光源槽位；其他灯光无遮挡。
  上游可选的 `RayTracedShadowPass` 仍是独立光追功能；连接它的整张图仍属于混合渲染。
- 透射材质保留预过滤环境折射近似，不显示玻璃背后的场景几何，不计算内部界面、
  Beer–Lambert 吸收、焦散或 SSS。BLEND 排序合成不在此 Pass 的职责中。
- 参考 `ScenePathTracePass` 继续提供完整路径追踪。原有 LookDev Slider 可用于比较
  光栅近似和参考结果，两者的间接光、遮挡、透射及轮廓像素积分会有差异。
- 实时路径不做渐进光照累积。抗锯齿与时域重建交给下游，
  `exportUpscalerGuides` 可输出 motion vectors 和 device depth。

旧图中 `lightingMode`、`environmentSamples`、`supplementaryPathTracing`、
`transmissionSamples`、`transmissionDepth`、`accumulate` 不再控制 Deferred 行为，
Inspector 也不再显示它们。仓库内图资产已迁移。外部旧图应删除连到
`Deferred.accelerationStructure` 的边；独立 Shadows 的 TLAS 边保留。

## Shader 分层与精度

- `VisibilityBufferDeferred.slang`：resident / StreamAsset 三角形重建、整屏与分箱调度、
  上采样 guides 和 HDR 输出。
- `VisibilityBufferLighting.slang`：实时 OpenPBR 直接光、外部阴影与 IBL。
- `SceneSurface.slang`：共享相机、纹理过滤、表面和 TBN 辅助函数。
- `OpenPBRSurface.slang`：共享 OpenPBR LUT 适配、材质求值与调试视图。
  两个表面文件不包含 ray-query 积分器；参考路径显式包含自己的追踪实现。

`halfPrecision` 默认 true，选择有界材质权重的半精度策略；关闭时使用相同算法的
FP32 权重，供精度回归对照。位置、深度、重心坐标、UV/LOD、TBN、IOR、折射、
GGX 分母和直接光 BSDF 保留 FP32；HDR 采样、乘光源和累加也保留 FP32。
因此高亮不会因为中间 HDR radiance 被压到 FP16 而溢出。

内置 ACEScg 的 flat 和四个非背景 binned 类共享精度源码，保留现有 float 运算及
pack/unpack 舍入，由 native Bool specialization 显式选择 `halfPrecision`。
flat 使用精度 Bool ID0；binned 保留金属度 Int32 ID0、不透明 Bool ID1，并增加
精度 Bool ID2。CPU 分别提供两个 flat 值和八组完整的 class/precision tuple，
使两种精度共享源码请求而拥有不同的 native pipeline 身份。自定义材质、Rec709、
背景类和显式精度覆盖继续使用各自的静态请求契约。
这项重构已通过同源码六目标构建和 122 项 CPU 测试。完整冷缓存 CLI 保留
132 个源码请求和 178 项逻辑策略，O1、默认 12 workers 下端到端实测 12.964 秒
（warmup 内部 12.610 秒），尚未达到 10 秒目标；GPU 输出/性能仍待验收。
下方历史 FP16 算术与性能记录不作为这版共享模块的验收结果。

## 分箱、资源与场景契约

`materialBinning` 默认 true，8×4 tile / wave32 分类为 Background、Dielectric、
Conductor、Opaque、Transmission 五类，以互斥 lane mask 执行 indirect dispatch。
所有类别使用同一光栅估计器；透射类不再有追踪特例。
分类要求 native subgroupSize=32 和 compute ballot/arithmetic；不支持时可关闭分箱，
使用整屏 8×8 调度。Custom Value 程序在 resident 场景使用一般内核。

`visibility`、`depth`、`rasterInfo` 必须来自同一个 producer，Deferred 继承其场景绑定。
位置和 TBN 仍保持 authored/world-space 方向，法线贴图求值后才对最终着色法线 face-forward。
StreamAsset 的光栅材质路径不需要启用 Cluster RTAS。

| 接口 / 设置 | 含义 |
| --- | --- |
| `visibility` | R32Uint 光栅 ID |
| `depth` | 同一 producer 的 D32 深度 |
| `rasterInfo` | 相机、分辨率、场景版本与 GPUScene view 元数据 |
| `domain` | 可选的位移重心坐标与几何法线 |
| `shadow` / `shadowParameters` | 必须成对连接，同分辨率阴影与对应光源参数 |
| `color` | Scene Linear RGBA32F；输出 upscaler guides 时为 RGBA16F |
| `halfPrecision` | 默认开启的 FP16 材质权重，可切换 FP32 对照 |
| `materialBinning` | 默认开启的 wave32 分类调度 |
| `debugDisableShadows` | 忽略外部阴影输入 |
| `debugDisableTransmission` | 关闭环境折射分量 |
| `debugView` | final、baseColor、geometryNormal、shadingNormal、tangent、material |

## 验证入口

`MetallicShaderRequestsTests` 覆盖 request 规范化和 24 个生产 Deferred 变体：
resident / stream、guides 开关、整屏 / 5 个材质类别。
编译测试检查 SPIR-V 不含 RayQuery/RTAS 类型或 ray-tracing capability，
非背景变体含原生 16-bit 浮点类型。

`visibility_buffer_abeautiful_game_transmission` 覆盖 15 种真实材质的
FP16/FP32 显示误差（各通道最多 2/255）、flat/binned 一致性、环境透射切换、
旧续追参数无效、volume 参数无效、材质刷新、guide 导出和奇数分辨率。
`stream_material_transmission` 验证无 Cluster RTAS 的 StreamAsset 材质输出。
`visibility_buffer_deferred_openpbr` 保留无阴影直接光及几何属性与 ray-primary 参考的对照。

性能 runner 当前只支持 WorkControl 候选；增加 FP16 类型本身不构成提速证据。
GPU 测试的 preview wall time 包含 CPU 提交与 readback，不能当作 GPU shader 时间。
完整 Zorah 场景的帧率、长期稳定性和显存驻留需要单独测量。

2026-10-03，MSVC Release 实测：8 项 shader 请求/编译测试通过；9 项 Vulkan GPU 回归通过，
包括棋盘材质、resident/stream、显式阴影、场景交接及 Custom Value 材质。
棋盘 385×257 的 FP16/FP32 对照最大通道误差 2/255、平均 0.002698/255，
无新增 Vulkan VUID；场景绑定测试包含预期的故障注入日志。
日志与图像保存在 `.cache/deferred-raster-shaders-final.log`、`.cache/deferred-raster-gpu/`
和 `.cache/deferred-raster-regression/`，上述运行开启 validation，不作为性能 A/B 证据。
实际 runtime 缓存中的非背景变体含 10 条 FP16 算术指令，解析结果及 SPIR-V 哈希见
`.cache/deferred-raster-precision.json`。Metallic、LookDev、MetallicGPUDrivenSample 构建通过；
`LookDev --sample lookdev-vbuffer --smoke-test` 完成提交和 present，启动预热 168 项、0 失败。
