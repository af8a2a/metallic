# Phase 3：Material → Closure → PreparedClosure

本阶段为独立 `MaterialProgram` 模块增加三阶段接口，并以 Debug Lambert 和独立 GPU 诊断渲染器验证完整调用链。沿用已有 `BSDFEval` / `BSDFSample` 的 acronym 拼写。现有 OpenPBR/RTXCR renderer 的三阶段迁移属于 Phase 4，当前默认渲染路径保持原有实现。

## 契约与静态特化

[SurfaceClosure.slang](../Shaders/Modules/MaterialProgram/SurfaceClosure.slang) 提供：

```slang
interface IPreparedSurfaceClosure
{
    BSDFEval eval(float3 wiWS);
    BSDFSample sample(float3 random);
}
interface ISurfaceClosure
{
    associatedtype Prepared : IPreparedSurfaceClosure;
    Prepared prepare(float3 woWS, TransportMode transportMode);
}
struct SurfaceMaterialResult<TClosure : ISurfaceClosure>
{
    TClosure closure;
    float3 emission;
    uint flags;
}
interface ISurfaceMaterialProgram
{
    associatedtype Closure : ISurfaceClosure;
    SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance);
}
```

`TransportMode` 定义 Radiance / Importance。方向采用向外、单位世界空间方向；`f` 不含 cosine/pdf，光照和积分器按 Phase 2 的密度约定消费。材质 emission 与 closure 分开，`SurfaceMaterialResult.flags` 为材质级标记（当前保留为零），不复用 BSDF event bits。

调用顺序：每个 hit/pixel 调一次 evaluate；每个 wo 调一次 prepare；每个光源/NEE 样本调用 Prepared.eval；每次路径延续调用 Prepared.sample。Closure 不能保存观察方向；Prepared 不得重新读取 baseColor/normal 纹理或参数表。接口文档明确这些约束，但 Slang 类型系统不会自动禁止任意第三方实现读取全局资源；实现和调用点仍需审核。

调用者使用 `T : ISurfaceMaterialProgram` / `TPrepared : IPreparedSurfaceClosure` 泛型约束，绑定 concrete type 后编译为 SPIR-V。没有 existential interface 容器、runtime virtual call 或按实例 ID 分派模型的 switch。实例引用仍只选数据。

## Debug Lambert

[DebugLambert.slang](../Shaders/Modules/DebugLambert.slang) 是独立模块，只导入 MaterialProgram：

- `DebugLambertMaterialProgram<TSource>` 在 evaluate 中调用一次 source，读取参数/纹理，生成 bounded albedo、emission 与单位法线。`IDebugLambertSource` 是资源读取适配器，便于实际调用方显式提供资源；不是额外的运行时求值阶段。
- `DebugLambertClosure` 仅保存反射率和已解析法线；多次 prepare 不再依赖 source。
- `DebugLambertPreparedClosure` 仅保存法线、反射率除以 π 和有效状态；eval 返回 Lambert `albedo / π`、cosine-weighted PDF；sample 使用余弦半球分布，第三个随机数预留给未来 lobe selection。
- 这是单面 Lambert，背面/切线 wo 无效，背面 wi 的 eval 为零；不修改 Context 的 authored TBN，不隐式面向相机翻转法线。退化输入法线先回退 geometryNormal，再回退 +Z。Radiance 与 Importance 在 eta=1 纯反射下结果相同。

纹理句柄和 source 只存在于 Material 阶段，两个 closure 类型都没有资源字段。source 可以完成自己需要的 UV transform、footprint 或 normal mapping；诊断 fixture 使用显式 2×2 纹理读取，不宣称实现了全部 OpenPBR 纹理导入规则。

## Prepared-only 消费端

[SurfaceLighting.slang](../Shaders/Modules/SurfaceLighting.slang) 只依赖 MaterialProgram：

- `evaluateSurfaceDirect<TPrepared>()` 仅接收 prepared、wi、入射辐亮度和 cosine；没有参数表、纹理、MaterialProgram 或 Closure 参数，无法再次调用 evaluate/prepare。
- `sampleSurfaceContinuation<TPrepared>()` 仅调用 prepared.sample，返回统一 BSDFSample。路径积分器仍负责 throughput、几何偏移、可见性、MIS 和终止。

诊断 kernel 用相同的 Program/Closure/Prepared 处理主命中和二次命中，在灯光循环外完成 evaluate/prepare。直接光照与路径延续都调用上述泛型消费者。这里验证的是 Lambert 的独立端到端渲染切片；生产 VBuffer、OpenPBR PT 和完整通用 lighting 的迁移仍由 Phase 4/5 负责。

## GPU 验证

`material_closure_lambert_stages` 使用真实 RGBA32F 2×2 GPU texture，覆盖两份实例参数与四个 texel：

1. 1024 组分层余弦半球样本；GPU/CPU 独立对比 f、PDF、方向长度、flags、eta、emission、白炉 throughput=albedo，检查平均 cosine 接近 2/3。
2. 同一已求值 Closure 为两个前向 wo、Importance 和背向 wo 分别 prepare；多次 eval/sample 不读取材质纹理。
3. 独立解析球体/平面场景的 192×128 HDR 渲染：1 灯、8 灯、8 灯加最多 3 次命中的路径追踪。GPU atomic counter 记录 evaluate-input、prepare、eval、sample、texture-load 的逻辑调用次数，并验证增加光源不会增加材质求值/纹理读取，路径追踪确实发生二次命中。

计数器和计数 wrapper 仅在测试 shader 中；生产模型无原子写入。这些是逻辑纹理求值次数，不是 GPU cache miss / 物理访存计数。测试用点读取是一个纹理调用，不能外推成其他过滤算法的物理取样成本。

`eval` 计数记录消费者对 Prepared.eval 的调用，不重复计入 Lambert sample 内部对纯算术 eval 的复用。

```powershell
cmake --build build-scheduling-release --target MetallicRHITests -j 8
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_closure_lambert_stages*' --rhi-validation --output-dir build/material-phase3-closure
```

测试生成原始 `.rgba32f`、相同曝光下的 sRGB PNG 与 `MaterialClosureStages.txt`。独立解析场景不等同于已验证完整生产 VBuffer/PT 集成；OpenPBR 数学包装、复杂法线映射、delta/transmission/MIS、完整 denoiser 消费链及大型场景压力测试不在本阶段的 Lambert 验收中。

## 2026-10-04 实测验收

MSVC Release 构建成功。新 Lambert GPU 测试及 5 项兼容回归共 **6/6 通过，无 skip**；兼容回归包含旧 720 字节 GPU ABI、Phase 2 shader contract、M2 Value Program GPU ABI、三条 Phase 0 HDR 图，以及 standard/OpenPBR guides 的四个编译请求（guides 项仅验证编译）。Vulkan validation 未出现 VUID 错误；环境仍有已有 layer manifest 缺失及旧 validation layer 下 OMM 回退提示。

| 工作负载 | Material evaluate-input / texture load | Closure prepare | Prepared eval | Prepared sample |
| --- | ---: | ---: | ---: | ---: |
| 1024 组数学/多视角探针 | 1024 / 1024 | 4096 | 13312 | 2048 |
| 192×128，1 灯 | 14508 / 14508 | 14508 | 14508 | 0 |
| 相同画面，8 灯 | 14508 / 14508 | 14508 | 116064 | 0 |
| 8 灯、最多 3 次命中的路径追踪 | 16947 / 16947 | 16947 | 135576 | 16947 |

多灯只增加 Prepared.eval，不增加材质/纹理求值。路径追踪增加了二次命中，仍保持每命中一次 evaluate、prepare 和材质纹理读取。白炉 throughput、Lambert f、sample/eval PDF、eta/flags、emission、前向多 wo 与背面零值通过绝对误差 `3e-5` 的检查；平均采样 cosine 为 `0.666669`，接近解析值 `2/3`（容差 0.001）。

三条旧 LookDev 图各渲染 256 帧，OpenPBR PT、OpenPBR Deferred、RTXCR Chiang 的 RGBA32F 均与 Phase 0 `run-0` **逐位一致，RGB RMSE/max absolute 为 0**；全部值有限。已检查新 Lambert 的 1 灯、8 灯和路径追踪预览。此处不把单次 PT 一致性扩展为跨进程确定性，也没有性能提升结论。

本地证据均在忽略的 `build/`：`material-phase3-build-final.log`、`material-phase3-closure.log`、`material-phase3-closure/Tests.xml`、`material-phase3-closure/MaterialClosureStages.txt`、三份 Lambert HDR/PNG；兼容回归在 `material-phase3-regression.log` 和同名目录，HDR 对比在 `material-phase3-report/comparison.json`。

- [x] Debug Lambert 完整实现 Material → Closure → PreparedClosure。
- [x] 同一像素从 1 灯增加到 8 灯，材质纹理仍只求值一次。
- [x] 新增泛型直接光照函数只接收 PreparedClosure contract。
- [x] 诊断 PathTracer 在主/次命中使用同一 Closure API。
- [x] concrete type 静态特化，无传统 runtime virtual call / material type switch。

下一阶段可以在保持这些消费者与计数约束的前提下，以 OpenPBR 的 evaluate/prepare/eval/sample 包装替换诊断模型。当前没有新增编辑器模型选择面板，也没有更改默认 OpenPBR/RTXCR 实现。
