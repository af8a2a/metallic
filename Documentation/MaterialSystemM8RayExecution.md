# M8 — Unified Ray Material Execution

对应外部路线图 **M8 / Phase 17**。resident 场景的 OpenPBR/Slab ray-hit 路径与 VBuffer primary 继续使用同一个 Material Program / Value IR / Closure 实现；M8 将二次命中的求值与准备边界显式化，并提供经过 GPU 验证的有界 Program 分类队列。

## 命中点执行

`Modules/RayMaterialExecution.slang` 是无资源的执行契约。`RayMaterialWork` 携带命中点的 Surface context、实例参数/资源引用、出射方向、path ID、seed 和 bounce。context 保存稳定 authored TBN、UV、front face 和本次 ray cone 投影出的纹理 footprint。

`evaluateRaySurface` / `evaluateRayFiber` 调用既有静态泛型 Material Program；随后 `prepareRaySurface` / `prepareRayFiber` 以本次 `wo` 创建临时 PreparedClosure。work 不存放屏幕坐标、Closure cache、PreparedClosure 或资源 provider。材质输入在每次命中重新解析，不能继承 raster primary 的静态 Program 特化。

- OpenPBR ray path 使用 `SceneOpenPBRMaterialProgram`，与 VBuffer 共享 Value IR、纹理求值、Single/Mix/Layer lowering 和 Closure。nested IOR、体积衰减及 path continuation 留在积分器。
- 两条现有 ray path 的 Fiber 分支使用相同通用执行契约和 RTXCRChiang Program。DOTS/triangle 的稳定 TBN、UV 和 ray footprint 转成 Fiber context，未提供的半径、strand parameter 和横截面 h 不伪造有效位。
- 当前为 Radiance transport。Fiber 的 projected scattering、weighted sample 与已有 MIS 近似保持不变，不额外套 Surface `NdotL`。
- legacy Standard Surface BSDF 仍是兼容积分器；完整 Value/Closure IR 的 Surface 执行后端是 OpenPBR/Slab。M8 不将旧模型的散射数学替换成 OpenPBR。

## Coverage

RT candidate、raster winner 和 opaque shadow 继续消费同代的独立 Coverage Program。OpenPBR 的 BLEND 续射概率和透明阴影衰减现在也调用 `evaluateSceneCoverage(...).opacity`，不再在消费者中重复实现 alpha 读取。当前自定义 Coverage 仍只允许 MASK Surface，BLEND 使用共享 legacy evaluator 的实例 alpha × 纹理 alpha；新增拒绝测试保持这一作者能力边界。

MASK 的候选接受、BLEND 的随机续射/透过率是不同消费者语义；共享 opacity 和 cutoff 并不表示把所有阴影改成相同输运算法。alpha layer 上限、直线连接的透明阴影近似以及现有 OMM 动态 Coverage 禁用策略均保持既有契约。

## Program 分类队列

运行时 `RayMaterialQueue` 和 `Features/PathTracing/RayMaterialQueue.slang` 提供 reset → count → prefix → scatter 四个 ComputeKernel，显式同步读写。分类表由**完整 MaterialProgramKey 相等比较**构建，包含 IR、domain、specialization、quality 和 target 等字段；实例参数与 Closure Family 不能代替 Program 身份。

| 数据 | 契约 |
| --- | --- |
| Hit key，16 bytes | material index、path ID、64-bit material generation |
| Bin，16 bytes | offset、accepted count、attempted count、overflow count |
| Indices | 回指原始 hit/work；排队不会用 slot 覆盖 path ID 或 seed |
| Status，4×uint | miss、无效材质/程序、过期 generation、容量丢弃数 |
| 容量 | 1…4096 个 Program；hit 和队列容量各最多 4,194,240；允许空 hit / 零容量 |

前缀分配只使用声明容量，GPU 原子 scatter 不越界；溢出按 Program 顺序保留受限数量，bin 内顺序不保证稳定。任何 invalid / stale / overflow 都要求调用者报告失败或完整重试，不得把部分结果当作完整帧。miss 是正常的未命中记录。

调用者拥有 GPU 缓冲，并负责保留对应的不可变 MaterialBindingGeneration / executable 到帧完成。队列只传 generation 标记，不能凭标记延长资源寿命。producer → classifier 的资源同步由调用者提供；classifier 内部和最终 compute consumer 的同步由队列提供。

这是可复用的分类基础设施，默认完整场景仍使用 fused path loop。尚未将所有场景积分器切换成 wavefront，也未声明吞吐或寄存器收益；大规模路径队列调度和 DGC 属于后续实测驱动的 M9。

## 验收

`RHIRendering.ray_material_execution_queue` 使用实际 Slab 与 RTXCR Fiber Program，比较串行 hit 顺序与分类后消费的 eval / sample / weight / RNG 输出。覆盖两个 bounce 的独立 position、UV footprint、wo，以及同 family 不同 Program、共享实例、domain/quality/target 差异。逐项检查唯一消费、分组归属、miss、非法材质、非法映射、generation 高位过期、容量 0/3 和 0/1/63/65/257 hit 的尾组。

`RHIRendering.material_value_closure_scene` 运行生产 PT/VBuffer，并额外创建不含 raster/closure cache 的 ray-only 图：Fiber 接收面和相机背后的 Slab 发光面使用同一代材质。depth=1 应全黑，depth=3 有间接辐射，隐藏发光量翻倍应使间接输出翻倍。保存原始线性 HDR 和 PNG，不以 shader compile 代替场景验证。

2026-10-05 最终验收：Windows Release / RTX 5070 Ti / Vulkan validation 开启，`Metallic`、`LookDev`、`MetallicRHITests` 构建通过。**11/11 回归通过，无跳过**：8 项 GPU 场景/数值回归、2 项 CPU 契约检查和 1 项 guide 编译检查。日志无 Vulkan validation error。队列 5 个 Program、20 组容量/尾组/bounce 用例，共 574 次有效求值，分组前后逐位一致。

实际混合域场景的 primary-only 输出为黑；开启 secondary 后 RGB 总能量为 228.4435，发光量翻倍的相对误差为 **0**。仅修改 Fiber 接收面的 melanin 后，相对 RGB 差异约 **0.86964**，排除了只有 Surface 接收面贡献间接光的假通过。已查看对应 PNG；该夹具仅 8 samples，图片用于检查执行/响应，不作为低噪声画质基线。Surface Program 分箱、Layer 边界以及独立 Slab secondary 发光线性对照误差也均为 0。

证据位于 `build/ray-m8-final.log`、`build/ray-m8-final/RayMaterialMetrics.json`、`ClosureSceneAcceptance.json` 和同目录 HDR/PNG/HTML 报告，不加入源码控制。本机旧 validation layer 会关闭硬件 OMM；Coverage 验证实际 shader alpha traversal，不能据此声明硬件 OMM 路径已通过。

另外回归 MASK 的 raster/RT/shadow winner、纹理 footprint、Fiber 资产和原生 strands、material generation / 在途重载。构建与测试命令：

```powershell
cmake --build build-scheduling-release --target Metallic LookDev MetallicRHITests
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.ray_material_execution_queue:RHIRendering.material_value_closure_scene:RHIRendering.material_fiber_asset_rendering --output-dir build/ray-m8-final
```

边界：M7 原生屏幕曲线尚无 RT 曲线求交后端，Fiber ray-hit 验证使用现有 DOTS/triangle。stream 材质、SDK 辐射缓存和 Importance transport 不因新增模块自动获得完整 Closure IR 支持。队列原型验证不等于完整 wavefront renderer 的端到端验收。
