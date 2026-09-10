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

## GPU 材质分箱

`materialBinning` 默认开启。可见性光栅化沿用原有 meshlet indirect draw；延迟着色按材质
执行 **indirect compute dispatch**，无需为屏幕空间着色重新绘制几何。

```text
VBuffer ID → Count → Allocate / indirect XYZ → Scatter → DispatchIndirect × material bins
```

- 按 GPUScene 到源材质的映射分箱；背景为 0，材质为 source material ID + 1。
  ABeautifulGame 有 15 个源材质，对应 16 个箱。材质参数在 Inspector 修改后仍沿用该 ID。
- Count 和 Scatter 在 wave 内反复取出一种材质，使用 ballot count / prefix count 聚合，
  每个 wave 对其中一种材质仅做一次全局原子操作。混合材质 wave 和边界残余 lane 均参与正确计数。
- Allocate 用 wave prefix sum 分配互不重叠的区间，同时生成每箱 3 个 uint 的 indirect 参数；
  超过 65535 个 X 组时扩展到 Y。空箱的 X/Y 为 0，不执行着色。
- 像素队列共占 `width × height × 4` 字节，计数、范围和参数另占每箱 24 字节，
  无每材质预留整屏容量，也无 CPU 像素计数回读。最多支持 65535 个源材质。
- Scratch buffer 按帧完成点复用并保留到 GPU 完成。阶段间显式同步 compute 读写，
  参数输出转为 `IndirectArgument` 后才消费；分辨率变化会重建所需容量。
- 同一入口按箱调度，保持材质选择在 wave 内一致；像素 RNG 种子与队列顺序无关。
  `ComputeProgram::dispatchIndirectBatch` 为整批保留一份描述符表，每箱只更新 push 参数与间接参数偏移，
  避免重复写入完整材质纹理描述符。
  这不是每种材质生成独立 shader permutation，二次射线仍可能命中其他材质并发散。

分箱需要计算阶段的 subgroup ballot 和 arithmetic 支持，运行时会检查能力。
相关能力的定义见 [Vulkan subgroup limits](https://docs.vulkan.org/spec/latest/chapters/limits.html)。
关闭 `materialBinning` 可使用相同着色估计器的整屏 8×8 dispatch 进行结果和性能对照。
材质很少、分辨率很低或每箱像素很少时，分箱和多次提交的开销可能超过减少发散的收益。

## 接口与同步

| 字段／属性 | 约定 |
| --- | --- |
| `visibility` | 必需，同分辨率 R32Uint，来自 resident GPUScene 的 VisibilityBufferPass |
| `depth` | 必需，同一次光栅化的 D32Sfloat |
| `rasterInfo` | 必需，VisibilityBufferPass 发布的 HostUpload metadata buffer |
| `color` | 始终为未曝光的线性 RGBA32F，接到公共曝光链路 |
| `environmentSamples` | 每帧 1–256 次 OpenPBR 环境采样，默认 64 |
| `materialBinning` | 默认 true，按源材质间接调度；false 为整屏对照路径 |
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
ctest --test-dir cmake-build-debug-visual-studio -C Debug -R MetallicLookDevVBufferSmoke --output-on-failure
```

GPU 测试先用相同的 OpenPBR 直接光照比较光线／VBuffer 主可见性，覆盖透视、正交、
材质颜色、法线、HDR 曝光、实际光栅相机和移除灯光。随后生成 768×768 对比图和两路完整图，
参考路径累积 1024 spp，延迟路径累积 256 帧的环境采样。
编辑器测试覆盖 Slider 拖动、相机联动与历史保留。

回归测试关闭 Aftermath，覆盖正常驱动编译下的行为。
分箱探针检查 258 个箱的计数、区间、逐像素唯一覆盖、无效 ID、空箱、尺寸变化和缓冲复用，
并以超过 65535 个 X 组的单材质队列验证二维间接调度。
ABeautifulGame 测试比较整屏／分箱的颜色、法线、材质 ID 和最终着色，检查透射开关、有限距离吸收及材质更新，
生成 `ABeautifulGameComparison.png`、两路完整图及 `ABeautifulGameTiming.txt`。
计时同时报告包含 CPU 提交与 readback 的 preview.render 墙钟时间，以及 Deferred 节点的 GPU timestamp。
后者包含分箱和所有材质着色；两者不能互相替代。

2026-09-10，RTX 5070 Ti / NVIDIA 616.64，Debug 构建、validation 开启、驱动内部缓存关闭，
ABeautifulGame 的 15 个源材质，8 次环境采样、4 次透射采样、深度 8、关闭累积，
预热后 12 帧的测量结果如下（毫秒）：

| 分辨率 | 整屏 GPU / 端到端 | 分箱 GPU / 端到端 |
| --- | --- | --- |
| 385×257 | 0.565 / 6.581 | 0.630 / 6.554 |
| 1280×720 | 0.901 / 8.138 | 1.136 / 8.128 |

当前资产尚未体现分箱的 GPU 加速，端到端差异也不足以说明收益。分箱默认接入用于后续材质扩展和对比，
不应据此宣称更快；可关闭该开关选择整屏路径。后续材质数量、着色复杂度和分辨率变化后应重新测量。

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
