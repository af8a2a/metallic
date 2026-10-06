# Material System Phase 4 — OpenPBR 三阶段适配

Phase 4 将生产 OpenPBR 的参数/纹理求值、法线映射与散射准备接到 Phase 3 接口。当前调用原生 Slang `OpenPBR` 模块的 `openPBRPrepare`、`openPBREval`、`openPBRSample`、`openPBRPdf`；不修改 `External/openpbr-bsdf` 或 720 字节材质上传 ABI。

## 生产调用

[OpenPBRSurface.slang](../Shaders/Features/PathTracing/OpenPBRSurface.slang) 定义 `OpenPBRMaterialProgram<TSource> : ISurfaceMaterialProgram`；生产别名 `SceneOpenPBRMaterialProgram` 静态绑定 `SceneOpenPBRMaterialSource`：

1. `evaluate(context, instance)` 读取材质参数，依次求值 normal、base color、metallic/roughness、emission、occlusion、transmission、specular 和 specular color，完成 glTF/OpenPBR 参数映射，返回 `SurfaceMaterialResult<SceneOpenPBRClosure>`（自定义 family 由 `SceneSurfaceClosure` 包装）。
2. `OpenPBRClosure<TContext>` 保存已解析的 `OpenPBRResolvedInputs`、occlusion 和不可变 BSDF LUT context。生产 context 是静态绑定的空类型；不保留材质纹理求值 provider、实例引用、观察方向或随机状态。
3. `prepare(wo, mode)` 生成 `OpenPBRPreparedClosure<TContext>`。路径追踪使用额外的 `SurfaceSamplingContext` overload，明确传入原有 throughput、RGB wavelengths 和嵌套介质 exterior IOR；这些采样状态不会进入 view-independent Closure。
4. 光源循环只消费 Prepared；路径延续使用统一 `BSDFWeightSample` 和事件位，不再读取 vendor 的 lobe 类型或返回结构。

`ISurfaceMaterialProgram.evaluate` 标记为 `[mutating]`，允许 STF 在 Program 内推进 RNG。生产调用在 evaluate 后取回 RNG 和 mapped hit；法线 debug 数据只留在 Program，不污染 Closure。纯函数式 Debug Lambert 仍满足同一接口。

生产 PT、VBuffer Deferred、ray-query realtime/direct lighting 与 OpenPBR guides 共用同一个 Program 和 Closure。材质在光源循环之外求值/准备。PT 在 alpha 随机覆盖测试前保留一次轻量参数读取，以保持先前的采样顺序；被 alpha 拒绝的 hit 不求值全部纹理。旧 GPU-driven visibility preview 的独立资源布局暂时保留，散射准备改用同一个 Closure。

VBuffer/realtime 的环境 SH/prefilter 近似、guide albedo 近似、介质堆栈和阴影透射仍保留原算法。Phase 4 不把它们重写成 Phase 5 的完整统一 LightingKernel，也不合并 Fiber/OpenPBR 的管线入口。散射调用没有按实例 ID 选择模型的新分支。

## 法线、纹理与发光

- Context 从 authored hit 构造；使用 authored geometry normal 一次投影 ray-cone footprint。纹理分辨率、UV transform、驻留 mip floor、NTC/STF 和 deterministic glass mip0 仍由现有采样路径处理。Context 当前生产者提供 UV0/isotropic footprint；UV1 与梯度采样不在此阶段新增。
- 所有 normal mapping 在 evaluate 中进行。PT 保留双面法线供 vendor 的折射/内外判断；Deferred 和 guides 按原行为在 normal mapping 后 face-forward。几何法线和原始 TBN 不随观察方向翻转。
- 保留 PT normal-map debug overrides、原有贴图顺序、M2 Value Program 映射位置和分箱提供的 fixed-metalness / opaque 特化，不更改模型外观。
- `SurfaceMaterialResult.emission` 是 view-independent 已解析发光；生产仍消费 Prepared 的 vendor emission，以保留背面发光抑制。二者不能无条件互换。

## BSDF 语义桥接

[OpenPBRClosure.slang](../Shaders/Modules/OpenPBRClosure.slang) 通过 `import OpenPBRClosure;` 使用，只依赖 `OpenPBR` 和 `MaterialProgram`。泛型 `IOpenPBRContext` 显式提供静态 Feature 与 LUT 访问；准备结果保留同一 context。生产的 Texture LUT 辅助函数位于 `OpenPBRTextureLUT` 模块，不再依赖旧 HLSL adapter、`OpenPBR_*` 类型别名或 `openpbr_*` 包装函数。

统一 `eval().f` 不含 cosine：适配器把 vendor projected value 除以 shading normal 的绝对 cosine；`sample().f` 从 vendor 的 `weight * pdf / abs(cosine)` 恢复。失败样本只依据 vendor 保证有效的 pdf 判断，其余未定义输出不再被读取。Reflection、Transmission、Diffuse、Glossy、Delta 显式映射到统一位；透射 eta 取 vendor 实际准备后的、含 specular-weight 修正和 clamp 的相对 IOR 的倒数，反射/失败为 1。

新增可选 `IWeightedPreparedSurfaceClosure`，用于本来就返回 projected value / importance weight 的库。`SurfaceLighting.evaluateSurfaceProjected` 和 `sampleSurfaceWeight` 通过泛型约束消费它。生产路径使用这一接口以避免除 cosine 后又乘回、恢复 f 后又除 pdf 的浮点往返；原有直接光、MIS 和 throughput 运算顺序保留。基础 `IPreparedSurfaceClosure` 与 Lambert 消费者继续可用。

**TransportMode 边界：** 本适配保留 vendor 的现有相机路径传输约定，仅支持 Radiance。vendor API 没有 adjoint 模式；Importance prepared 的 `validTransport=false`，eval/pdf/sample/emission 返回零/无效样本。不能将此实现用于双向/光源起点积分器并假定已实现 Importance。未增加新的 eta 能量缩放来改变现有 PT 外观。

Prepared 的 vendor prepare/eval 可以查询 BSDF 固定 LUT；“不重复纹理求值”指材质参数和 base/normal 等材质资源，不包括 BSDF 数学查表。

## 验证

`material_closure_openpbr_stages` 用真实 GPU 参数缓冲和 RGBA32F 纹理，实例化同一个 `OpenPBRMaterialProgram<ProbeSource>`。测试 provider 替换资源来源并计数，Program 的 normal mapping 和参数映射、Closure 的 vendor 包装及泛型消费者均为生产实现。

- 1024 组样本覆盖 dielectric、metal、rough/smooth glass、emission、unlit、背面透射及零能量材质。
- 和直接 vendor 调用比较 projected eval、PDF、采样方向/weight/event、emission，核验统一 f 的投影/权重重建与透射 eta。
- 检查 authored TBN 法线映射与实例/纹理颜色的独立 CPU 预期，确认多次 prepare 和增加光源不增加材质读取。
- 原子计数为逻辑 source 调用次数，不是物理纹理访存或 cache miss。探针将原始 vendor uint16 LUT 以相同归一化和线性插值读取 GPU buffer，避免大数组嵌入 shader 的驱动编译成本；生产 texture LUT 描述符由完整 renderer 回归覆盖。

构建和执行命令（x64 VS developer shell）：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests -j 8
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_closure_openpbr_stages*' --rhi-validation --output-dir build/material-phase4-probe
```

## 2026-10-04 实测验收

MSVC Release 构建成功，最终 **18/18 测试通过，无 skip**。16 项生产/接口回归覆盖材质 asset roundtrip/upload、M2 Value Program、旧 GPU ABI、Lambert、OpenPBR/standard guides 编译、stream material/shadow/transmission、ray-cone footprint、authored tangents 和 VBuffer Deferred；另外通过 OpenPBR 三阶段探针及材质纹理 preview。保留的 legacy visibility shader 单独编译通过，不把这项编译检查当作其完整运行时验证。

| 1024 个已求值材质 | 参数读取 | 材质纹理请求 | 实际材质纹理 Load | 主视角 prepare | 灯光循环 eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 灯 | 1024 | 8192 | 2048 | 1024 | 1024 |
| 8 灯 | 1024 | 8192 | 2048 | 1024 | 8192 |

每个 hit 有 8 个纹理槽请求，其中 2 个绑定真实 GPU 纹理，其余返回缺省值。额外视角和不支持的 Importance prepare 不增加上述材质资源读取。探针的 vendor 对照与 f/weight 重建最大差异为 `1.73315e-7`；136 个失败样本、244 个进入和 116 个退出的透射样本均通过检查。

实际采样 event mask 为 15（Reflection/Transmission/Diffuse/Glossy）。当前 vendor 对非常光滑的玻璃仍使用有限 GGX、返回 Glossy，未产生 Specular/delta 事件。代码保留其到 `kBSDFDelta` 的显式映射，但本次不能宣称 delta 传输已经获得运行时覆盖。

三条 Phase 0 场景各渲染 256 帧；与冻结 `run-0` 原始 RGBA32F 比较：

| 场景 | 逐位一致 | RGB RMSE | RGB max absolute |
| --- | --- | ---: | ---: |
| OpenPBR PT | 否 | 0.0002632135 | 0.04733241 |
| OpenPBR Deferred | 是 | 0 | 0 |
| RTXCR Chiang | 是 | 0 | 0 |

所有值有限，三张同曝光预览已检查。PT 不是逐位回归；其差异低于已冻结的 Phase 0 `run-0` 对 `run-1/2` 的 A/A 差异（RMSE 0.000455647 / 0.000524175，max 0.05359751 / 0.05389053），符合本场景已有随机波动尺度。该比较不建立跨进程确定性，也不把一个场景的误差阈值推广到其他内容。

Vulkan validation 未报告 VUID 错误；环境仍有已有 layer manifest 缺失和旧 validation layer 的 OMM fallback 提示。未进行大型场景内存压力、长时间 temporal/denoiser 稳定性、Importance 或真实 delta 传输验证，没有渲染性能提升结论。

证据均在忽略的 `build/`：

- `material-phase4-build-final.log`：最终构建。
- `material-phase4-final.log`、`material-phase4-final/Tests.xml`：16 项回归与三条原始 HDR。
- `material-phase4-probe.log`、`material-phase4-probe/Tests.xml`、`OpenPBRClosureStages.txt` 及 readback `.bin`：新增探针、legacy 编译和纹理 preview。
- `material-phase4-report/comparison.json`、`phase0-aa.json` 和三张 PNG：HDR 对比与预览。

Phase 4 的生产接入完成；下一阶段可以统一完整 LightingKernel 和环境/guide 近似接口，继续保留这里建立的每 hit 求值一次与 Prepared-only 消费边界。
