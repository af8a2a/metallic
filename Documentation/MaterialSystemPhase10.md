# Material System Phase 10 — Closure IR / Slab Prototype

本阶段提供独立的 `Slab / Mix / Layer` Closure DAG，以及可执行的 Single / Dual Slab 后端。
OpenPBR 保留 vendor reference；`OpenPBRCompositeClosure` 只是原 `OpenPBRClosure` 的类型别名，
现有 Surface Program 使用这个 canonical family 名称，散射、采样和 LUT 路径不变。

## IR 与后端边界

[MaterialClosureIR](../Source/Runtime/Material/MaterialClosureIR.h) 位于共享 CPU Material 层。
`create(nodes, root)` 接受 children-before-parent 的节点数组，验证活跃图，剔除不可达节点，
并按有序子节点遍历生成规范化表示及版本化 FNV-1a 身份。输入编号和无关节点不影响身份。
Slab 参数为已解析的 RGB reflectance 和 RGB optical depth；Mix 额外提供 `[0,1]` 的权重。
Layer 的 operands 明确按 top、bottom 排列。非法数值、循环/前向引用、未知节点均报错。

IR 可表达大于两个 Slab 的嵌套 DAG；4096 个输入节点与 32 位复杂度溢出检查只用于资源安全。
实时预算由 `RealtimeBackendProfile` 单独定义，默认：

| 预算 | 默认值 |
| --- | ---: |
| maxClosures | 2 |
| maxOperators | 1 |
| maxNormalBases | 1 |
| maxLayerDepth | 1 |
| maxPayloadBytes | 96 |

`lowerMaterialClosure` 检查预算，再选择 canonical family。当前执行后端只实现单个 Slab
或两个 Slab 加一个操作符。放宽 Profile 不会自动提供嵌套执行器；这种情况会给出单独的
“no executable family” 诊断。当前后端只接受共享的 SurfaceMaterialContext normal basis 0。

| Canonical family | 来源 | 已解析 Closure payload |
| --- | --- | ---: |
| SingleSlabClosure | 单个 Slab | 48 bytes |
| DualSlabClosure | Mix 或 Layer，参数值可以不同 | 96 bytes |
| OpenPBRCompositeClosure | 原有 OpenPBR Surface Program | 原参考布局，未转换成 Slab |

Slab upload packet 为 80 bytes：两个 32-byte SlabRecord 加 16-byte control。
法线来自命中上下文，所以 upload packet 与 resolved Closure 的大小不同。
GPU 测试通过 `sizeof(T.Closure)` 验证 CPU 记录的 48 / 96 bytes。

复杂度分别记录 `ClosureRecordCount`、`ScatteringLobeCount`、`NormalBasisCount`、
`LayerDepth`、`PayloadBytes`，另有 OperatorCount。一个原型 Slab 对应一个漫反射 lobe；
共享 DAG 节点被两个操作数引用时按两次散射出现计数。Basis 按独立符号 ID 去重，
LayerDepth 是最长路径上的 Layer 操作符数量。PayloadBytes 在 family lowering 后确定。

## 散射模型

[SlabClosure.slang](../Shaders/Modules/SlabClosure.slang) 实现三阶段接口：

1. `SingleSlabMaterialProgram<TSource>` / `DualSlabMaterialProgram<TSource>` 从 provider 取得已验证输入并解析法线。
2. Closure 保存不依赖视角和资源的 SlabRecord、normal 和操作符数据。
3. `prepare(wo, mode)` 生成临时状态；Lighting 通过现有 `IPreparedSurfaceClosure` 评估或采样。

这是单面、共享法线、仅反射的漫反射界面原型。所有方向应为归一化世界空间方向。
Slab 的反射为 `r / pi`；终端 Slab 是不透明基底，其 opticalDepth 只在它成为 Layer 顶层时使用。
不包含镜面、折射、薄膜、体散射或层间多次反射；被省略的能量不重新分配。

Mix 直接组合散射函数：

```text
fMix = (1-w) * fA + w * fB
```

Layer 使用顶层界面的 RGB 反射率 r、顶层下方的光学厚度 tau、底层反射率 b：

```text
T = 1-r
A(wi, wo) = exp(-tau * (1/mu_i + 1/mu_o))
fLayer = r/pi + T*T*A(wi, wo)*b/pi
```

这是一次向下、一次向上的穿透/吸收模型，角度越倾斜路径越长；没有把层参数做 lerp。
两个 cosine 都取共享法线的正半球值。倒数在 1e-6 处截断以避免数值溢出。
交换 wi/wo 后公式不变，且逐通道 `r+(1-r)^2*b*A <= 1`，因此不会产生超出入射白炉的能量。
无吸收时黑色顶层允许白色基底完全返回；白色顶层完全遮住底层；大吸收趋近顶层单独反射。

两个分量采用同一 cosine hemisphere proposal，完整 PDF 为 `mu_i/pi`。
Layer 的 cosine proposal 不是精确重要性分布，但无偏、归一化，Sample 返回完整 Layer 的 f。
Radiance / Importance 对当前 eta=1 的互易反射模型相同；背面或下半球返回零。
采样端点被夹到有效范围。模型保持命中法线，不改变 TBN 或几何法线约定。

## 运行与接入范围

[MaterialSlabTests.cpp](../tests/rhi/MaterialSlabTests.cpp) 展示 CPU DAG 构造、lowering、上传及 family dispatch；
[MaterialSlabProbe.slang](../tests/rhi/shaders/MaterialSlabProbe.slang) 提供读取 packet 的具体 source。
测试用同一 `SurfaceLighting` eval/sample 路径渲染球体与地面，64 samples/pixel、最多 3 次命中。

本阶段是可执行的独立原型。Slab 尚未成为编辑器可选的 scene material implementation，
也没有新增 Closure 图 JSON/编辑器或从 Value IR 自动生成 Slab inputs；provider 是两者的接入点。
Phase 9 的 Value IR 与现有 OpenPBR 作者路径保持原样。Program→Closure Family 两级 GPU 分箱
属于后续 Phase 11，本阶段不修改 Visibility / Material bin 的 ABI。

## 验证

2026-10-04，Windows / MSVC Release，复用 `build-scheduling-release`：

- `MetallicRHITests`、`MetallicSceneTests`、`Metallic` 构建成功；Scene CTest 通过。
- 新增 `material_closure_ir_lowering`、`material_slab_furnace_sample_render` 两项通过。
  GPU probe 的四个入口关闭磁盘缓存重新编译，并在 Vulkan validation 下执行，validation error 为 0。
- 23 组材质 × 3 个出射角 × 4096 个样本，共 282624 次采样，覆盖黑/白界面、权重 0/1、
  零/强吸收、法线倾斜和掠射方向；Sample/Eval/PDF、加权采样、互易性、传输模式、背面拒绝、
  端点有限性和 payload ABI 全部通过。独立 uniform-hemisphere 积分与 cosine sampling
  的逐通道能量差容差为 0.0002，能量上界为 1.00001；完整报告保存在 `build/material-phase10-slab/SlabEnergy.txt`。
- 三张 256×192 HDR / PNG 已生成并检查：`SingleSlab`、`DualSlabMix`、`DualSlabLayer`，
  路径为 `build/material-phase10-slab/`。Layer 在不同出入射角下显示不同的吸收。
- 既有 Closure / SurfaceLighting / MaterialProgram / Value IR / Deferred 回归：10 项中 9 项通过。
  `render_graph_openpbr_pathtracing_shader_compile` 的 native heap 资源契约断言失败，shader 本身编译成功。
  临时恢复两个改动前的 HEAD OpenPBR shader 后单独复测，得到相同失败，随后按字节恢复 Phase 10 修改。
  对照日志为 `build/material-phase10-prechange-check.log`；该已有失败未计为通过，未修改其断言。
- 独立运行 `material_asset_phase0_equivalence` 通过。OpenPBR PT、OpenPBR Deferred、RTXCR Chiang
  三组原始 RGBA32F 与冻结 Phase 0 基线逐位一致，RGB RMSE / max error 都为 0。
  比较记录为 `build/material-phase10-report/comparison.json`。

主要命令（MSVC developer shell，repo root）：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests MetallicSceneTests Metallic -j 8
ctest --test-dir build-scheduling-release -R '^MetallicSceneTests$' --output-on-failure
$env:METALLIC_NSIGHT_GRAPHICS_CAPTURE='0'
$env:METALLIC_SHADER_CAPTURE_SYMBOLS='0'
.\build-scheduling-release\tests\MetallicRHITests.exe `
  '--gtest_filter=*material_closure_ir*:*material_slab_*' --rhi-validation `
  --output-dir build/material-phase10-slab
.\build-scheduling-release\tests\MetallicRHITests.exe `
  '--gtest_filter=*material_asset_phase0_equivalence*' --rhi-validation `
  --output-dir build/material-phase10-baseline
```

系统 Vulkan loader 仍报告已有 EOSOverlay / `E:\Validation.json` 缺失警告。
本次没有声称 Slab 的编辑器交互、时域稳定性、大场景内存或性能已验证；
也没有运行完整 ShaderWarmup 或完整 RHI suite。
