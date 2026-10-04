# Phase 2：Slang Material Program 数据接口

外部路线图 Phase 2 建立 shader 侧的模型无关数据契约。Phase 1 的作者资产、共享 CPU Program 与旧 720 字节上传保持兼容；`ISurfaceMaterialProgram` / Closure / PreparedClosure 的三阶段散射调用属于 Phase 3，本阶段不预先引入另一套散射接口。

## 独立模块

`import MaterialProgram; using Metallic.Material;` 提供：

| 类型 | 契约 |
| --- | --- |
| `SurfaceMaterialContext` | 世界位置、几何法线、插值着色法线、TBN、两套 UV、footprint、UV 有效位与正反面 |
| `TextureFootprint` | UV0/UV1 的像素导数，或投影后的各向同性 normalized-UV LOD，以及明确的有效位 |
| `MaterialInstanceRef` | 相对当前 Program 绑定的参数表/资源表的两个 **字节偏移**；不保存 descriptor、实例 ID 或 shader variant |
| `BsdfEval` | `f`、`pdf`、事件 flags |
| `BsdfSample` | 向外的世界空间 `wiWS`、`f`、`pdf`、`eta`、事件 flags |

入口：[MaterialProgram.slang](../Shaders/Modules/MaterialProgram.slang)。此模块没有任何 import，不依赖 OpenPBR、RTXCR、Lighting、Core、Scene 或 Vulkan 资源声明，也不声明 push constants。资源由实际 Program 和调用方显式绑定。

Context 的 normal/tangent/bitangent 保持 authored/world-space 方向，不根据 ray 或 frontFace 翻转；`shadingNormalWS` 是 normal mapping 前的插值法线。`frontFace` 使用 uint 0/1，避免跨存储时依赖 bool 表示。`texCoordMask` 的 bit 0/1 表示对应 UV 可用；缺失数据不能假装另一套 UV。

Footprint 的导数以一个内部渲染像素为单位，位于纹理 UV transform 之前。`isotropicLodN` 是 normalized-UV footprint 的 log2，可以为负；还没有加入纹理分辨率的 `0.5 * log2(width * height)` 或 UV transform LOD bias。ray producer 先把 cone 投影到表面，再写入这个标量，因此 Context 不携带灯光、观察方向或 BSDF 参数。valid flags 区分“有效的零”与“数据不可用”，并允许消费者按自身过滤算法选择有效的梯度或各向同性表示。

散射数值约定：`f` 不乘 cosine，不除以 pdf；连续分布的 `pdf` 是包含 lobe 选择概率的完整方向采样密度。delta sample 设置 `kBsdfDelta`，pdf 使用离散概率，任意方向 eval 为零；消费者不能把离散与连续密度直接做 MIS。反射/透射、diffuse/glossy/delta 使用具名 bits。透射 `eta = eta_i / eta_t`，反射为 1。传输模式缩放由后续 prepared closure 负责。本阶段仅定义这些记录，没有改变既有 OpenPBR/RTXCR 的计算公式。

## 旧路径兼容接入

[MaterialRuntime.slang](../Shaders/Modules/Material/MaterialRuntime.slang) 提供 stateless `LegacyMaterialProgram` 与 `legacyMaterialInstanceRef()`。旧上传把参数与纹理描述嵌在同一 720 字节 record，因此该 adapter 的两个偏移必须相等、按 720 字节对齐、落在绑定表范围内；它只接收兼容转换函数生成的引用。偏移为 32 位，旧表不能超过 4 GiB 的可寻址范围。后续拆表由具体 Program/schema 定义两个偏移的布局，不修改作者 `.material`。

[SceneSurface.slang](../Shaders/Features/PathTracing/SceneSurface.slang) 的 `loadSceneMaterial()` 用同一 concrete storage program 读取任意实例。OpenPBR/standard PT、secondary medium/shadow、guides、stream ray 和 VBuffer lighting 都调用这条兼容入口；没有按 material instance ID 选择 concrete shader code，也没有增加生产 variant。原有按 Program identity 区分 Surface/Fiber 的逻辑仍保留，Phase 2 不宣称完成后续模型适配。

`sceneSurfaceMaterialContext()` 适配 ray-query / VBuffer 共用的 HitInfo。当前 HitInfo 只有 UV0，所以 UV1 为零、mask 为 1。旧 cone LOD 计算通过 Context 读取几何法线并写入 footprint，运算顺序、UV transform 与 mip clamp 保持原样；TBN 构造与 normal mapping 未改变。没有强行生成 raster ddx/ddy；当前 VBuffer 仍沿用原有 cone 近似。

## 验证入口

- `material_program_shader_contract`：禁用磁盘缓存，独立编译新模块并检查完整依赖不含模型/场景/光照。单个静态泛型 concrete 测试程序，8 组重复/乱序实例、独立参数与资源偏移，两次 dispatch 只改变数据；GPU 逐位读回全部 Context、BsdfEval、BsdfSample 与实例引用。两种 footprint 和正反面均覆盖。
- `material_runtime_gpu_abi`：旧 probe 通过 LegacyMaterialProgram 读取三个不同实例，逐 word 验证原 720 字节 payload，避免只验证 CPU 名称或结构大小。
- `material_program_guide_compile`：复用生产 `makeSceneShaderRequest()`，编译 standard/OpenPBR guides 的 hardware/fallback position-fetch 四个请求；此项只验证编译，不验证 denoiser 输出。
- Phase 0 三条 HDR、材质资产往返、streamed textured/masked/transmission/shadow 及原 footprint 测试验证生产兼容路径。

新模块中的 Context 是 shader 瞬态值，不是新 GPU 上传大结构；测试故意通过 scalar-layout buffer 传输，测得/约定 Context 128、Ref 8、Eval 20、Sample 36 字节。测试参数使用现有 ComputeKernel / ParameterWriter inline ABI，没有增加旧 slot 访问。

```powershell
cmake --build build-scheduling-release --target MetallicRHITests -j 8
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_program_shader_contract*:*material_runtime_gpu_abi*' --rhi-validation --output-dir build/material-phase2-contract
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_asset*:*material_value*:*material_runtime_generations*:*.stream_material_shading:*.stream_material_transmission:*.stream_material_shadow:*texture_primary_ray_cone_pixel_footprint*:*visibility_buffer_deferred_openpbr*' --rhi-validation --output-dir build/material-phase2-render
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_program_guide_compile*' --rhi-validation --output-dir build/material-phase2-guides
```

## 2026-10-04 验收记录

MSVC Release 构建成功。上述三组共 **15/15 个 RHI 测试通过，无 skip**（2 项契约/ABI、12 项生产及材质回归、1 项 guides 编译检查）；GPU 运行启用 Vulkan validation，未出现 VUID 错误。环境仍报告已有 EOS / `E:\Validation.json` loader manifest 缺失，以及旧 validation layer 下 OMM 沿用 shader alpha traversal 的提示。

三条固定 LookDev 图各运行 256 帧，全部 HDR 值有限，已检查相同曝光与 sRGB 显示变换下的预览：

| RGBA32F 输出 | 相对 Phase 0 run-0 | 进一步核对 |
| --- | --- | --- |
| OpenPBR PT | RGB RMSE 0.0004556465743，max absolute 0.05359750986 | 与冻结的 **run-1 逐位一致** |
| OpenPBR Deferred | 逐位一致，RMSE 0 | — |
| RTXCR Chiang | 逐位一致，RMSE 0 | — |

PT 不是相对 run-0 零误差；其差异与原 Phase 0 run-1 对 run-0 的差异完全相同，未另行放宽容差。Phase 0 的跨进程非确定性原因仍未定位。本次没有做性能结论，也不把 shader 重编译/PSO 创建时间当作渲染耗时。

证据位于忽略的 `build/`：`material-phase2-build-final.log`、`material-phase2-{contract,render,guides}.log`、三个同名输出目录的 `Tests.xml`，以及 `material-phase2-report/comparison.json`、`pt-repeat-comparison.json` 与三张 PNG 预览。

- [x] 新 shader 数据 ABI 可独立编译，依赖检查不含具体材质或光照。
- [x] Context 不依赖 OpenPBR，保留 authored TBN 与明确的 UV/footprint 有效位。
- [x] 一个 concrete Program 对应多个实例，参数更新复用同一编译 kernel。
- [x] 实例偏移仅选择数据；兼容 renderer 读取不按实例 ID 选择 concrete shader code。

未执行交互式编辑器、完整 DLSS/NRD guides 消费链、NTC 专用排列或大型场景长时间流送/显存压力测试。Closure 的 prepare/eval/sample 调用、OpenPBR 数学实现的搬迁和通用散射模型切换仍按后续阶段推进。
