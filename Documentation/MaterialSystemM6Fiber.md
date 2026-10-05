# M6 — Fiber Domain

对应外部路线图 **M6 / Phase 15**。2026-10-05 实现，继续使用现有 DOTS 三角形光追几何。

## 接口与执行

- `FiberMaterial` 定义 `IFiberMaterialProgram`、`IFiberClosure`、`IPreparedFiberClosure`，不继承 Surface BSDF 接口。Context 包含位置、稳定法线/TBN、半径、strand parameter、材质坐标及 texture footprint；有效位区分 DOTS 尚未提供的半径/strand parameter 和真实零值。
- `RTXCRFiber` 将加载参数、构造 Closure、按出射方向准备 interaction 分为三个阶段。Prepared 缓存吸收、粗糙度方差、cuticle、frame、局部出射方向，多灯求值不重复读取材质。
- `FiberLighting` 使用 projected scattering，不附加 Surface `NdotL`。`FiberSample.weight` 已除以采样 PDF，路径延续只乘一次。
- `SceneFiberPath` 负责共用 Fiber 灯光、遮挡、环境采样和路径延续。标准和 OpenPBR ray-hit 主循环通过 **domain** 分派；具体 Chiang 类型只在材质适配边界出现。
- `RTXCRHair` 为 vendored 原生 Slang 2026 模块，包含 Chiang、Separate Chiang、Far Field 及依赖。原版 HLSL、MIT 许可证、commit/hash 与符号映射保留，详见 [模块说明](../Shaders/Modules/RTXCRHair/README.md)。

## 材质资产

`Asset/Materials/RTXCRChiang.materialdef` 的 implementation 为 `RTXCRChiang.DOTS`。
示例 `Asset/Materials/Examples/ChestnutFiber.material` 使用 `Asset` 作为资源根。

共享 `MaterialAssetLibrary` 的 Definition/Instance、稀疏参数、parent 继承、版本校验、原子保存、SceneDocument 绑定和重载；共享 MaterialGeneration、ProgramKey、GPU 参数记录与生命周期。参数修改更新实例数据，不新建材质程序。

| 参数 | 范围 |
| --- | --- |
| melanin / melaninRedness | 0–1 |
| longitudinalRoughness / azimuthalRoughness | 0.02–1 |
| hairIor | 1.01–3 |
| cuticleAngle | -10–10 度 |

实例可配置 `features.doubleSided`。当前场景 Program 固定使用 normalized melanin absorption；不暴露在该模型中无效的 baseColor 或 Far Field diffuse 参数。其他吸收模型和 Far Field 在原生模块中可用，尚未注册为新的场景 Program。

Fiber 资产拒绝 Surface Value/Closure IR、Surface 资源槽、非默认 Surface feature policy。场景参数编辑、保存和重载保留 Fiber 参数并更新 material revision。现有 MaterialGraph 仍是 Surface 前端。

## 验收

Windows Release、RTX 5070 Ti、Vulkan validation 开启；复用 `build-scheduling-release`。

```powershell
cmake --build build-scheduling-release --target Metallic MetallicSceneTests MetallicRHITests
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter=MaterialAssets.*
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.material_fiber_native_and_stages:RHIRendering.material_fiber_asset_rendering
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.material_runtime_hdr_capture
```

- 资产 CPU 测试：20/20，通过 Fiber 继承、round-trip、域校验、失败不发布、SceneDocument 保存及重载。
- 扩展 CPU 回归 36/36：包含上述资产测试、MaterialGraph 和 6 项 SceneEditing 材质/场景序列化测试；旧的“忽略 Hair 属性编辑”断言已更新为 M6 的可编辑 Fiber 契约。
- 独立 GPU oracle：4096 组参数/方向，三个 Hair 模型的 eval/sample 与未修改上游比较；1/8 灯时每个 Program 只 load 一次。最终版本最大绝对差 `6.1035e-5`（对应参考值 `551.79584`，相对差约 `1.11e-7`），validation errors 0。原始 SPIR-V/readback 在 `build/fiber-m6-final/`。
- Claire groom：384×216、每次新历史 64 帧，导入材质与等值资产输出逐位一致；melanin 从 0.98 改为 0.15、保存重载后 RGB 总绝对差 2534.04，GPU 参数同步且共享同一 Program。OpenPBR ray path 与标准 Fiber ray path 输出相对 RMSE 约 `6.7e-9`。证据在 `build/fiber-m6-asset/`。
- 回归基线：768×432 groom 和 768×768 OpenPBR PT/Deferred，各 256 帧、原始线性 HDR；移植版与改动前 groom 相对 RMSE 约 `1.295e-4`，OpenPBR 两输出逐位一致。已检查 groom 对照图。证据在 `build/fiber-m6-baseline/`、`build/fiber-m6-final/`，不加入源码控制。
- 最终 RHI 回归 13/13：覆盖 Fiber、资产上传、generation/schema/publication、错误材质、in-flight reload、GPU ABI、两条 ray path 的 guides 编译及 RTXCR Sample 编译。`Metallic`、`MetallicSceneTests`、`MetallicRHITests` Release 构建通过。
- Fiber 场景测试使用同一输出目录再次运行通过；测试会清理自己上次生成的场景 sidecar，确保每次从导入材质开始。
- 修复原 HDR 验收遗漏：首帧记录编译日志，出现 shader 错误或 error-material fallback 即失败，不能再用有限但错误的颜色通过验收。

## 能力边界

本阶段支持 Radiance transport；Importance 返回无效样本。轴向/数值上接近轴向出射的 azimuthal frame 在 Prepared 阶段拒绝（横向分量平方 ≤ `1e-12`），防止 `normalize(0)` 传播 NaN。

保留上游 Chiang 的采样 PDF 和已有环境 NEE 的近似 standalone PDF/MIS 策略，运行时 `exactStandalonePdf=false`。本次数值一致性验收不证明该近似估计器无偏，也不声称性能提升。

Opaque VBuffer 明确拒绝 Fiber。原生曲线交互、真实 radius/strand parameter、strand visibility、多层覆盖与运动/LOD 验收属于 **M7**；本次只验证现有 DOTS。RTXCR geometry 导入和 Sample 的 Subsurface 部分继续依赖对应 SDK，Hair 数学模块本身不依赖 SDK include path。
