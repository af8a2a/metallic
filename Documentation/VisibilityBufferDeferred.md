# LookDev：VBuffer 与 OpenPBR 路径追踪对比

```powershell
cmake --build cmake-build-debug-visual-studio --target LookDev --parallel 8
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample lookdev-vbuffer
# 15 种源材质，包括棋子顶部的 transmission / volume 玻璃
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample lookdev-abeautiful-game
```

在 Built-in Sample 中选择 **LookDev / VBuffer vs Path Tracing** 也可加载。
左侧为 OpenPBR 路径追踪参考，右侧为 GPUDriven Visibility Buffer 延迟渲染；
沿用 Slider 的拖动分界、上下比较和交换 A/B 操作。
`--scene <path>` 同时替换 Reference、VBuffer、Deferred 的材质场景。

## 渲染链路

```text
Reference: ScenePathTracePass (OpenPBR) ───────────────────────→ Slider.sourceA

VBuffer: VisibilityBufferPass ─ visibility / depth / rasterInfo
                                           ↓
                           VisibilityBufferDeferredPass ─────→ Slider.sourceB

Slider.color → AutoExposure (固定 EV100 0, None/sRGB) → FinalBlit
```

VisibilityBufferPass 保持可见性职责：GPUScene 实例／meshlet 剔除、Mesh Shader
光栅化、alpha-mask 判定、两阶段 HZB，以及 R32Uint ID 和 D32 深度输出。
`visualization: "none"` 关闭 ID 调试显示。

新增的 **VisibilityBufferDeferredPass** 是独立的 Compute RenderGraph 节点。
它按 VBuffer 获胜 record 找到实例、meshlet 和三角形，从深度和光栅相机重建表面位置，
计算重心坐标，插值 UV、法线与 tangent。TBN 保留 authored/world-space 方向，
只在法线贴图求值之后处理最终着色法线朝向。

材质贴图、glTF 到 OpenPBR 的映射、OpenPBR 能量 LUT、点光／聚光／方向光单位和
阴影逻辑复用现有参考渲染资源。直接光调用 `openpbr_eval`，环境照明调用
`openpbr_sample`，包含漫反射与镜面／金属反射。此路径不发射主可见性光线，
光线查询用于灯光和环境遮挡。不透明表面默认每像素每帧 64 次环境采样，并渐进累积线性 HDR，
仍只计算直接光与环境照明。透射表面从重建的主命中继续执行同一 OpenPBR 路径积分器，
默认 2 个样本、最多 8 层，支持折射、出射界面、内部全反射和 Beer–Lambert 体积吸收。
阴影连接复用参考路径的直线透射近似，玻璃不再一律作为不透明遮挡物。

## 场景绑定与帧前准备

场景来源由 Pass 的 `sceneDependency()` 声明，属于 RenderGraph 的依赖契约。
编辑器和 Sample 加载器按此声明发现场景消费者，不再维护 Pass 类型白名单或依赖
`scenePathTargets` 来保证路径同步。后者仍作为旧 Sample 的 UI／stream asset 目标元数据保留。

- `World`：有效的运行时文档是唯一来源，节点 `path` 仅在未绑定有效文档时作为启动资产。
  节点显式设置 `"sceneBinding": "asset"` 时，才独立使用自己的 `path`，不跟随编辑器切换。
- `Input`：Deferred 从 VBuffer 输入继承同一场景绑定，忽略自己的旧 `path`。
  `visibility`、`depth`、`rasterInfo` 必须来自同一个生产者，混接在任何 Pass 准备前被拒绝。
- `None`：没有场景依赖，例如 Slider 和后处理。

RenderGraph 为每个绑定保存源指针、路径，以及 resource identity、lifetime、structural、content、
transform、visibility、material 七项版本快照。两个执行入口均在录制图／子系统命令前检测变化：
等待旧提交完成、使历史失效、准备所有受影响 Pass，再发布新一代绑定。
同地址文档替换、同路径重载、不同路径切换不需要调用者额外设置 graph dirty。
未变化的场景不会重复编译；帧前刷新保留已导出的 graph texture／buffer 句柄。
场景更新不得改变 Pass 的资源反射契约；改变输出布局仍需走正常的 graph rebuild。

Reference／Deferred 的几何、材质和 RTAS 在这次准备中获取，`execute()` 不再自行解析路径或
重新选择资源快照。VBuffer 输出保存实际编译的 scene identity，避免文档原地替换后
把旧资源标成新文档。调试图同时记录每个 Pass 的实际路径、绑定来源和版本。
准备中途失败时不会执行任何 Pass；下次会重新准备整个场景依赖集合，
即使调用者切回了此前的文档，也不会复用已经部分改写的资源。

`render_graph_scene_binding_contract` 覆盖两个执行入口、自动重载／材质版本刷新、独立资产、
错误 consumer path、输出句柄稳定、场景不变时不编译、注入准备失败后恢复，以及混接输入提前拒绝。
`visibility_buffer_async_scene_handoff` 还通过实际 VBuffer 着色验证无需 dirty 的场景交接，
保留旧路径切换到 meet_mat、asset/world 切换和分箱／整屏逐像素一致。
故障注入测试会产生预期的 `Injected consumer scene preparation failure` 日志，不属于 Vulkan 错误。
2026-09-11 验证：15 项 RHI 回归通过，包含实际材质编辑、ABeautifulGame 透射、分类覆盖、
shader reload、图资源复用和提交路径；Debug 的场景切换／Inspector／Slider 三项编辑器回归通过，
Release 的真实编辑器场景切换回归也通过。均未出现 Vulkan validation 错误。
验证记录位于 `.cache/scene-binding-contract/`。

## GPU 材质分箱

`materialBinning` 默认开启，Inspector 中显示为 **Wave32 Material Tile Classification**。
可见性光栅化沿用原有 meshlet indirect draw；延迟着色按可执行的 BSDF 特征分类屏幕 tile，
每个类别执行一次 **indirect compute dispatch**。

```text
VBuffer ID → Reset → Classify 8×4 tiles → Build indirect XYZ → DispatchIndirect × 5 classes
```

参考 Unreal `SubstrateMaterialClassification.usf` 中的特征分类、tile 列表与间接调度思路。
Substrate 的复杂度分类说明见
[Epic 文档](https://dev.epicgames.com/documentation/en-us/unreal-engine/overview-of-substrate-materials-in-unreal-engine)。
这里针对当前 glTF → OpenPBR 映射采用如下类别；分类并不增加 Substrate 多层材质能力：

| 类别 | 保守分类条件 | 着色变体 |
| --- | --- | --- |
| Background / 0 | 无有效可见性或实例／材质映射 | 仅环境背景和历史输出 |
| Dielectric / 1 | 无透射，metalness 因子 ≤ 0 | 固定 metalness=0 的不透明 OpenPBR |
| Conductor / 2 | 无透射，metalness 因子 ≥ 1，且没有 metallic-roughness 贴图或 NTC 集 | 固定 metalness=1 的不透明 OpenPBR |
| Opaque / 3 | 其余无透射表面，包括混合金属度和金属度贴图 | 一般不透明 OpenPBR |
| Transmission / 4 | transmission 因子 > 0，优先于上述材质分类 | 完整 OpenPBR 主命中续追 |

- 每个 8×4 tile 对应一个 32-thread workgroup。所有 lane（包括越界边缘）先参与 ballot；
  每类最多由 leader 做一次原子追加，记录 `{tileIndex, laneMask}` 两个 uint。
  删除全屏像素 Count / Scatter 的两次分类扫描与逐材质 wave peeling。
- Unreal 将普通 tile 提升到其中最复杂的类型；这里对混合 tile 保留互不相交的类别 mask。
  玻璃边缘不会迫使不透明 lane 进入主光线续追；消费时从 tile 和 lane 恢复相邻屏幕坐标，
  不再把全屏同材质像素打散为线性队列。不同材质 ID 可以共享同一种着色变体。
- 分类读取与延迟着色相同的源材质 buffer；越界 source ID 使用同一个 fallback material 0。
  每帧重新分类，因此 Inspector 的金属度、透射和纹理修改无需重建材质 ID 列表。
  存在金属度贴图或 NTC 时不会猜测纹理像素值；不透明特化只写入数学上已确定的 BSDF 常量。
- 固定 5 个队列，每类最多保存一份每 tile 任务，容量为
  `ceil(width/8) × ceil(height/4) × 5 × 8 + 100` 字节。
  对齐分辨率下约为 **1.25 字节/像素**，原实现为 4 字节/像素加逐材质 metadata。
  容量和 dispatch 数量均不随材质数量增长，也不再有 16-bit source material ID 限制。
- 每个着色 workgroup 消费一个 tile mask；超过 65535 个 X 组时扩展到 Y。
  空队列 X/Y 为 0，无 CPU count readback。Scratch 按 GPU 完成点复用，参数明确转为
  `IndirectArgument`；不同类别只写各自 mask 覆盖的像素，无需类别间内存屏障。
- 独立编译 5 个 shader permutation。不透明变体不包含主命中续追调用，背景变体不包含
  几何重建。二次射线和透射阴影仍使用完整 OpenPBR 材质求值，保留跨材质命中语义。
  RNG 仍以屏幕像素为种子，输出和历史不依赖任务追加顺序。
- 5 个变体通过 `ComputeProgram::dispatchIndirectBatch` 共用一次描述符更新，批内仅切换 pipeline 和 push data。
  共享前校验 device、绑定顺序／类型／数量、实际 heap shader index 及完整 push-data 布局；
  不兼容变体在录制前返回 InvalidArgument。描述符表和所有变体按 frame completion 保留。

此实现要求 **native subgroupSize=32** 和计算阶段 subgroup ballot/arithmetic；运行时明确检查。
当前 RHI compute stage 未开启 varying subgroup size，因此 32-thread workgroup 恰好是一整个 wave。
mask 的构造和消费均使用 `WaveGetLaneIndex()`，不依赖它与 `SV_GroupIndex` 的对应关系。
其他 subgroup 大小设备应关闭 `materialBinning`，使用相同估计器的整屏 8×8 dispatch。
单个类别在 tile 中只占少量像素时仍有空闲 lane，性能取决于场景覆盖和 BSDF 成本。

## 接口与同步

| 字段／属性 | 约定 |
| --- | --- |
| `visibility` | 必需，同分辨率 R32Uint，来自 resident GPUScene 的 VisibilityBufferPass |
| `depth` | 必需，同一次光栅化的 D32Sfloat |
| `rasterInfo` | 必需，VisibilityBufferPass 发布的 HostUpload metadata buffer |
| `color` | 始终为未曝光的线性 RGBA32F，接到公共曝光链路 |
| `environmentSamples` | 每帧 1–256 次 OpenPBR 环境采样，默认 64 |
| `materialBinning` | 默认 true，wave32 tile 特征分类与 5 种着色变体；false 为整屏对照路径 |
| `transmissionSamples` | 透射表面每帧 1–16 个继续追踪样本，默认 2 |
| `transmissionDepth` | 透射继续追踪深度 2–16，默认 8，包含光栅主命中 |
| `accumulate` | 默认 true，可关闭以查看单帧延迟渲染结果 |
| `debugDisableShadows` | 关闭直接光和环境遮挡，用于隔离 BSDF 差异 |
| `debugDisableTransmission` | 关闭透射，用于对照玻璃和阴影效果 |
| `debugDisableVolumeAttenuation` | 关闭体积吸收 |
| `debugUseOpaqueShadows` | 使用二值遮挡对照阴影透射近似 |
| `debugView` | final、baseColor、geometryNormal、shadingNormal、tangent、material |
| `flipBitangent` | 与参考路径相同的材质 TBN 调试开关 |

`rasterInfo` 包含实际观察相机、分辨率、scene identity 和 producer 类型。
它由 CPU 在光栅 Pass 内写入，延迟 Pass 读取此元数据，不回读 GPU 图像。
延迟着色不维护第二套可编辑相机；冻结 culling camera 也不会改变其重建相机。
Reference 和 VBuffer 通过 `cameraSyncGroup: "LookDevComparison"` 联动。

相机、材质／几何、照明、环境或延迟着色设置变化会重置延迟路径的累积。
拖动 Slider 和调整统一显示曝光不会清除该累积。

## 当前范围与可解释差异

- 支持 resident GPUScene 的 opaque 和 alpha-mask 材质；沿用 VBuffer 的单／双面及 LOD 路径。
  Stream page geometry 暂返回 Unsupported，避免将 stream record 当作 resident record 解释。
- 两路使用同一 OpenPBR BSDF。延迟路径的不透明主表面只计算直接光与环境照明，
  参考路径默认每帧 4 spp、12 层反弹，因此凹槽、接触区和反射内的间接照明仍会不同。
- 环境采样累积只增加着色样本，不对 VBuffer 的主表面做像素抖动／抗锯齿；
  轮廓可能与路径追踪参考的像素积分存在差异。
- 支持 ABeautifulGame 的 `KHR_materials_transmission` 和 `KHR_materials_volume`，
  以及已有的金属／粗糙度、法线、遮蔽、发光与颜色贴图。两种棋子顶部使用 OPAQUE alpha mode，
  透射由 OpenPBR BSDF 处理；透明 BLEND 排序合成仍不支持。此次未增加 clearcoat / sheen 等资产未使用的扩展。
- 该资产没有设置 attenuationDistance，按无限距离处理，不能仅根据 attenuationColor 期待体积吸收。
  测试通过 Inspector 同一材质更新接口设定有限距离后验证吸收。
- 透射路径是有限深度的混合渲染：不透明主表面的镜面反射仍只采样环境，
  与完整路径追踪的间接照明、反射中的场景和焦散有差异。直线阴影透射不求解精确折射光源连接。
- 纹理重建使用现有材质采样器和每三角形的 ray-cone LOD 尺度；
  参考路径使用每 primitive 的平均尺度，纹理缩小过滤可有差异。

场景、HDRI、太阳和材质参考来源见 [OpenPbrLookDev.md](OpenPbrLookDev.md)。
默认 shader ball 渲染图由 `Tools/BuildOpenPbrLookDev.py` 随参考场景一起生成。
ABeautifulGame 图为 `Pipelines/Samples/lookdev_abeautiful_game.metallic_graph.json`，使用资产配套 HDRI，
默认 8 次环境采样、2 次透射采样，以便交互比较。

## 验证

```powershell
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = "disabled"
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter material_binning_indirect_coverage --rhi-validation --output-dir .cache/validation/material-binning
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter visibility_buffer_abeautiful_game_transmission --rhi-validation --output-dir .cache/validation/material-binning
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter visibility_buffer_deferred_openpbr --rhi-validation --output-dir rhi-test-output/vbuffer-lookdev
ctest --test-dir cmake-build-debug-visual-studio -C Debug -R "MetallicLookDev(SceneSwitch|VBuffer)Smoke" --output-on-failure
```

GPU 测试先用相同的 OpenPBR 直接光照比较光线／VBuffer 主可见性，覆盖透视、正交、
材质颜色、法线、HDR 曝光、实际光栅相机和移除灯光。随后生成 768×768 对比图和两路完整图，
参考路径累积 1024 spp，延迟路径累积 256 帧的环境采样。
编辑器测试覆盖 Slider 拖动、相机联动与历史保留。

回归测试关闭 Aftermath，覆盖正常驱动编译下的行为。
分类探针覆盖 257 个源材质映射到 5 类、每类每 tile 唯一任务、逐像素唯一覆盖和实际 wave32 宽度；
同时检查无效 ID、全背景、边界 mask、纹理／NTC 保守分类、跨帧材质特征修改、尺寸变化与缓冲复用，
并以超过 65535 个 tile 的单类别队列验证二维间接调度。
批次测试交替切换带输出标记的 shader permutation，验证真正执行了对应变体；
同时检查不兼容布局的提前拒绝，`frame_descriptor_snapshots` 验证帧内描述符快照的保留。
`visibility_buffer_async_scene_handoff` 覆盖正常编辑器的异步场景交接：先由图按路径加载临时场景，
再交给编辑器 SceneDocument，并覆盖同一对象重新加载和切回临时场景。图拓扑、路径和分辨率保持不变。
修复前，资源重建复用的 VBuffer 仍持有旧 scene/source lease，而 Deferred 已读取新场景，
导致 `InvalidArgument`（实测 raster scene=5、deferred scene=8）；这不是 GPU device-lost。
VisibilityBufferPass 现在在 `prepare()` 中重新检查场景来源／版本，复用匹配资源或重建对应视图。
场景或视图不匹配时，Deferred 日志会输出两侧分辨率和 scene ID。
修复后上述交接、分箱覆盖、材质编辑及 ABeautifulGame 四项回归通过；Release ABeautifulGame
正常异步启动的已完成执行序号从 1548 推进到 2245，无 validation／execute 错误，随后正常退出。
证据保存在 `.cache/binning-scene-handoff/`（`repro.stdout.log` 为修复前的预期失败）。

手动切换到不同路径的场景还需要同步所有场景消费者的 `path`。编辑器的
`isSceneAwareRenderPassType()` 曾遗漏 `VisibilityBufferDeferredPass`，导致从 ABeautifulGame
切到 `meet_mat.glb` 时，Reference／VBuffer 使用新场景，Deferred 仍按旧路径读取 ABeautifulGame；
日志实测 raster scene=17、deferred scene=13，随后返回 `InvalidArgument`。
已将 Deferred 纳入统一的路径和相机同步。
`MetallicLookDevSceneSwitchSmoke` 通过编辑器实际的异步加载、资源准备和提交入口，
覆盖 ABeautifulGame → meet_mat → 同路径重载 → ABeautifulGame，以及分箱关闭／开启后的渲染。
它同时检查三个消费者的有效路径和每次加载产生的新文档身份，避免测试辅助函数掩盖编辑器漏同步。
修复前该测试明确报告 Deferred 保留 ABeautifulGame 路径；修复后 Release 和 Debug 均通过，
无 Vulkan validation／execute 错误，Debug 的 Slider／相机联动回归也通过。
证据保存在 `.cache/meet-mat-scene-switch/`（`repro/` 为修复前的预期失败，
`fixed-release/` 和 `fixed-debug/` 为通过记录；Debug 完整编辑器日志由 CTest 保存在
`cmake-build-debug-visual-studio/tests/lookdev-scene-switch/editor.log`）。
ABeautifulGame 测试比较整屏／分箱的颜色、法线、材质 ID 和最终着色，检查透射开关、有限距离吸收及材质更新，
生成 `ABeautifulGameComparison.png`、两路完整图及 `ABeautifulGameTiming.txt`。
计时同时报告包含 CPU 提交与 readback 的 preview.render 墙钟时间，以及 Deferred 节点的 GPU timestamp。
后者包含分箱和所有材质着色；两者不能互相替代。

2026-09-10，RTX 5070 Ti / NVIDIA 616.64，Debug 构建、validation 开启、驱动内部缓存关闭，
ABeautifulGame 的 15 个源材质，8 次环境采样、4 次透射采样、深度 8、关闭累积。
最终 wave32 分类版本，预热后 12 帧的测量结果如下（毫秒）：

| 分辨率 | 整屏 GPU / 端到端 | Wave32 分类 GPU / 端到端 |
| --- | --- | --- |
| 385×257 | 0.560 / 6.041 | 0.617 / 5.731 |
| 1280×720 | 0.970 / 6.843 | 0.989 / 6.855 |

1280×720 下分类 GPU 成本比整屏高约 1.9%，端到端基本持平；当前资产不能据此宣称分类比整屏更快。
混合 tile 的空闲 lane 和固定分类／间接调度开销仍然存在，可关闭该开关选择整屏路径。
历史对照中，上一个按源材质全屏像素分箱版本在相同设置下的 GPU 时间为 0.624 / 1.217 ms
（385×257 / 1280×720）。本次 720p 的分类成本降低，但两次测量不是固定时钟下的配对基准。
后续材质数量、空间分布、着色复杂度和分辨率变化后应重新测量。

五个变体分别写描述符曾使 720p 端到端增加到 13.967 ms；共享批次描述符后为 6.855 ms。
这部分收益来自减少 CPU 描述符更新，不应计为 GPU BSDF 加速。
最终记录、测试日志及对比图位于 `.cache/material-classification-wave32/final/`。
12 个延迟着色组合（整屏／5 类 × position fetch 开关）均通过 Slang 编译和 Vulkan 1.4 SPIR-V 验证。

## Aftermath 排查

后续首次编译顺序对照已隔离到跨 Shader Object 特性配置复用不兼容 pipeline 缓存：
SO=false 编译后由 SO=true 命中相同 key 时失败，禁用缓存或先由 SO=true 编译时通过。
关闭驱动内部缓存并重新生成 VBuffer 应用 `.pso` 后，原始 shader 的完整 GPU 测试通过。
结构体复制消融被 generator / OpNop 阴性对照推翻，不能作为根因。
详见 [DescriptorHeapGpuFault.md](DescriptorHeapGpuFault.md)。以下保留最初捕获记录。

2026-09-08，本机 GB203-A / NVIDIA 616.64 上，未插桩的完整 GPU 回归仍然失败。
已撤回尝试过但未同时通过回归的 shader 改写、SPIR-V 优化和 robustness 配置，
保留原有着色器编译和 RHI 资源生命周期逻辑。
现有 28 项相关 RHI 回归通过；默认 Aftermath 配置下完整 VBuffer 测试通过，
重建后的 LookDev 正常启动、运行 10 秒并正常退出。这些结果不包含关闭 shader debug info 的失败用例。

在原始实现上启用资源追踪、自动检查点，关闭 shader debug info 后捕获到：

- `Error_DMA_PageFault`，读取 GPU 虚拟地址 `0`，Graphics Processing Cluster。
- 最早的正在执行检查点是 `EnvironmentLightingSubsystem::GpuPrecompute::build()`
  的环境 SH dispatch，经 `publishDecoded()` / `recordPreGraph()` 调用。
- 开启完整 shader debug info 时，同一完整 VBuffer 回归通过。
  这说明诊断配置改变了复现条件，不能视作修复或据此认定驱动存在 bug。
- CPU 自动检查点定位到命令录制位置；此次转储没有 GPU 指令到 Slang 源码的映射，
  尚未确定地址 0 的底层成因。后续应从该 SH dispatch 的 push data、描述符及驱动生成代码继续缩小。

复现并保存独立证据目录：

```powershell
.\Tools\CaptureLookDevAftermath.ps1
# 对照运行：启用完整 shader debug info（可能不再复现）
.\Tools\CaptureLookDevAftermath.ps1 -ShaderDebugInfo
```

脚本运行 `visibility_buffer_deferred_openpbr`，设置 `METALLIC_TEST_AFTERMATH=1`，
记录进程基址、可执行文件和 PDB 的 SHA-256、退出码、日志及 Aftermath 转储副本。
若 PATH 中存在 `llvm-symbolizer`，同时保存当前构建的 CPU 检查点解析结果。
默认输出 `.cache/aftermath/captures/`，可通过 `-OutputDirectory` 指定。
脚本退出码沿用 GPU 测试结果；捕获到崩溃时测试仍应报告失败。

正常 LookDev 启动默认启用 Aftermath；smoke test 会关闭它。
手动捕获正常启动故障时可使用：

```powershell
$env:METALLIC_AFTERMATH_SHADER_DEBUG_INFO = "0"
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample lookdev-vbuffer
```

此开关只关闭 shader debug info，保留资源追踪和自动检查点。
原始转储与解码 JSON 写入 `.cache/aftermath`；移除环境变量恢复默认配置。
