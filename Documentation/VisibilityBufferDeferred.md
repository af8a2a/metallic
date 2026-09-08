# LookDev：VBuffer 与 OpenPBR 路径追踪对比

```powershell
cmake --build cmake-build-debug-visual-studio --target LookDev --parallel 8
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample lookdev-vbuffer
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
光线查询用于灯光和环境遮挡。默认每像素每帧 64 次环境采样，并渐进累积线性 HDR；
不做多次反弹 GI。

## 接口与同步

| 字段／属性 | 约定 |
| --- | --- |
| `visibility` | 必需，同分辨率 R32Uint，来自 resident GPUScene 的 VisibilityBufferPass |
| `depth` | 必需，同一次光栅化的 D32Sfloat |
| `rasterInfo` | 必需，VisibilityBufferPass 发布的 HostUpload metadata buffer |
| `color` | 始终为未曝光的线性 RGBA32F，接到公共曝光链路 |
| `environmentSamples` | 每帧 1–256 次 OpenPBR 环境采样，默认 64 |
| `accumulate` | 默认 true，可关闭以查看单帧延迟渲染结果 |
| `debugDisableShadows` | 关闭直接光和环境遮挡，用于隔离 BSDF 差异 |
| `debugView` | final、baseColor、geometryNormal、shadingNormal、tangent |
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
- 两路使用同一 OpenPBR BSDF。延迟路径只计算可见表面的直接光与环境照明，
  参考路径默认每帧 4 spp、12 层反弹，因此凹槽、接触区和反射内的间接照明仍会不同。
- 环境采样累积只增加着色样本，不对 VBuffer 的主表面做像素抖动／抗锯齿；
  轮廓可能与路径追踪参考的像素积分存在差异。
- 当前未实现透明 BLEND 的排序合成、穿过玻璃后的场景折射或多次反弹透射。
  环境阴影使用现有的二值 shadow query。
- 纹理重建使用现有材质采样器和每三角形的 ray-cone LOD 尺度；
  参考路径使用每 primitive 的平均尺度，纹理缩小过滤可有差异。

场景、HDRI、太阳和材质参考来源见 [OpenPbrLookDev.md](OpenPbrLookDev.md)。
渲染图由 `Tools/BuildOpenPbrLookDev.py` 随参考场景一起生成。

## 验证

```powershell
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter visibility_buffer_deferred_openpbr --rhi-validation --output-dir rhi-test-output/vbuffer-lookdev
ctest --test-dir cmake-build-debug-visual-studio -C Debug -R MetallicLookDevVBufferSmoke --output-on-failure
```

GPU 测试先用相同的 OpenPBR 直接光照比较光线／VBuffer 主可见性，覆盖透视、正交、
材质颜色、法线、HDR 曝光、实际光栅相机和移除灯光。随后生成 768×768 对比图和两路完整图，
参考路径累积 1024 spp，延迟路径累积 256 帧的环境采样。
编辑器测试覆盖 Slider 拖动、相机联动与历史保留。

回归测试关闭 Aftermath，覆盖正常驱动编译下的行为。

## Aftermath 排查

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
