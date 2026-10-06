# Material System Phase 8 — Coverage Program 分离

后续更新：[Phase 9 Value IR](MaterialSystemPhase9.md) 已统一 Surface/Coverage 前端并新增 v2 图引用。
下文保留 Phase 8 交付时的后端和验收记录。

Phase 8 将“命中是否存在”从 Surface/Closure 求值中分离。Slang 接口位于
[Coverage.slang](../Shaders/Modules/Material/Coverage.slang)：
`ICoverageEvaluator.evaluateCoverage(CoverageContext, MaterialInstanceRef)` 返回 opacity 和 accepted。
`CoverageContext` 只包含 UV、基础 alpha、cutoff 和 alpha mode；不依赖光照、法线、closure 或 pass 局部时钟。

## 作者入口与最小切片

沿用 [M2](MaterialValueProgramsM2.md) 的 `RenderMaterial::valueProgram` JSON 源码，增加独立的 `coverage` 根。
通过 `SceneDocument::setMaterialProperties()` 修改并随场景 sidecar 保存。
读取 [Dissolve.value.json](../Asset/LookDev/MaterialPrograms/Dissolve.value.json) 到源码字段，设置
`alphaMode="MASK"`、`alphaCutoff=0.5`，调整 `valueParameters[0]` 即可移动 UV.x 方向的溶解边界。
参数为 -1 时全部消失；没有 alpha 纹理且基础 alpha 为 1 时，参数为 1 则全部保留。

```json
{
  "version": 1,
  "coverage": {
    "op": "mul",
    "args": [
      { "op": "alpha" },
      { "op": "add", "args": [
        { "op": "uv" }, { "op": "parameter", "index": 0 }
      ] }
    ]
  }
}
```

- `alpha` 是基础颜色因子 alpha × 基础颜色纹理 alpha，纹理不存在时使用 1。
- `uv` 是当前几何的 TEXCOORD_0；基础颜色纹理仍应用原有 UV transform。读取其他 UV 集的既有限制不变。
- 参数是 Surface 与 Coverage 共用的四个 float4 槽。最终 `.x` 经 saturate 后与 cutoff 比较。
- 支持 scalar/float4 常量及 `add`、`mul`、`dot`、`mix`、`sin`、`fract`、`abs`、`saturate`。
- 编译器仅遍历 coverage 根，做相同子表达式去重和依赖分析；不执行 Surface 输出所需的无关节点。
  无 `alpha` 依赖时跳过基础颜色 alpha 纹理读取。
- 每个切片最多 64 条独立指令、深度 24、访问节点 256；场景最多 64 个不同 Coverage 程序。
  源码仍受 16 KiB 限制，常量和参数必须有限且绝对值不大于 1e6。

当前后端是有界、只读的 float4 字节码，Surface 继续生成静态 Slang。
这是现有 M2 表达式语言的 Coverage 切片；Phase 9 的通用 Value IR/图编译器尚未实现。
自定义 Coverage 仅开放给 MASK Surface 材质；OPAQUE、BLEND 和 Fiber 的自定义 coverage 被拒绝。
Coverage-only 可以搭配透射；同时写 Surface 输出时仍遵守 M2 的 lit、非透射、非 Fiber 限制。

后续 Painter LookDev 的显式 OpenPBR `attenuationColor` Surface 输入允许普通透射，
见 [当前接入说明](OpenPbrLookDev.md#painter-验证场景选项)；Coverage 字节码及独立执行约束不变。
没有新增编辑器图形界面，也没有放宽 `.material` 资产对自定义程序源码的既有限制。

## 共享快照与生产接入

[MaterialValueProgramSet](../Source/Runtime/Render/Material/MaterialValueProgram.cpp)
分别编译 Surface 与 Coverage，按规范化表达式共享 Coverage 指令。
现有 80 字节 instance 的三个保留 uint 用于 Coverage offset/count/flags；四个参数槽的偏移保持不变。
所有 instance 记录及去重后的 Coverage 指令尾部一起上传为一个不可变输入 buffer。
该 buffer 与 720 字节 legacy payload 一起由 `MaterialBindingGeneration` 发布和保留，使用同一个 material revision。
legacy payload 的 `attenuationColor.w` 保留通道用于标记 MASK 自定义输入；RGB 吸收颜色语义不变。

| 消费者 | 求值位置 |
| --- | --- |
| Resident VBuffer / tessellation | MASK fragment 写入 visibility/depth 之前 |
| Stream VBuffer / tessellation | `streamAlphaAccepts`，保留原有透明分类 |
| PathTrace / OpenPBR / stream ray query | 候选三角形命中处的共享 `evaluateSceneCoverage` |
| Ray-traced shadows | 与 path trace 相同的 alpha acceptance |
| RTXDI / 材质可视化 | 相同 Coverage 接口、参数 buffer 与普通纹理 alpha 采样规则 |

普通纹理使用原有 `AlphaCoverage.hlsli` 的确定性双线性读取，使用最精细驻留 mip（上传图像的 mip 0）。
OPAQUE 的 raster entry 不调用 Coverage；ray query 的 opaque 分支直接通过。
MASK 无自定义切片时保留基础 alpha/cutoff 行为。没有测量 CPU 绑定成本或声明性能提升。

动画/时间由调用方写入共享参数槽并发布一个材质 revision，不在不同 pass 内分别读取时间。
帧内消费者读取相同 generation，GPU 完成前保留其资源。编译或上传失败不会发布部分程序。
纯 Coverage 代码/参数修改不改变 Surface executable identity；Coverage-only 材质仍可使用原有 Program 分箱。

`VisibilitySignature` 升为 v2，包含 alpha mode 与规范化 Coverage 代码，不再包含 transmission。
ProgramSignature 排除 Coverage 代码，transmission 的 Feature 类别仅为 Closure。
透射描述有效命中后的光传输，不把玻璃自动变成透明裁剪。

## Opacity Micromap 与边界

含自定义 Value 源码的 MASK 材质保守禁用静态 OMM，防止 OMM 直接确认命中而绕过动态 Coverage。
首次加入或移除自定义程序会改变材质资源布局身份并重建加速结构；参数/代码编辑复用已禁用 OMM 的路径。
其他 legacy 材质继续使用 OMM。这一保守策略也覆盖只有 Surface 输出的 MASK 程序。

本阶段验收使用普通 alpha 纹理；NTC alpha 仍沿用原有 RT 神经纹理解码路径，
其与 VBuffer 普通 alpha 的跨路径一致性未建立，不作为本阶段已支持的 alpha authoring 路径。
需要跨路径一致的覆盖纹理应保留普通纹理。UV/参数程序本身不依赖 NTC。
VBuffer 仍不处理 alpha blend；Coverage 不引入半透明排序或随机透明度。

## 验证（2026-10-04）

复用 MSVC/Ninja Release `build-scheduling-release`，构建 `Metallic`、`MetallicSceneTests`、`MetallicRHITests`。
`ctest --test-dir build-scheduling-release -R '^MetallicSceneTests$' --output-on-failure` 通过。

- 带 Vulkan 验证层的最终材质回归：45 项，41 通过、4 跳过、0 失败。
  跳过项是缺少外部数据的 Zorah probe、未启用 NRD 的 RTXDI 完整图，以及旧验证层不支持 KHR OMM 的两项 OMM 测试。
  最终日志仍保留 loader 缺少 EOS overlay JSON / `E:\Validation.json` 的环境警告；未过滤这些消息。
- 新增 Coverage 编译测试验证最小切片、资源依赖消除、1000 个实例共享指令、Surface executable 身份独立、参数范围和错误拒绝。
- `material_coverage_winner_shadow_ray` 在生产 VBuffer 图和真实 RayQuery/阴影函数上检查八组状态、30,752 个内部像素。
  覆盖 legacy → 自定义 UV 裁剪 → 参数变化 → 完全保留/溶解 → 纹理 alpha → Surface 颜色与 Coverage 同时热更新 → legacy 恢复。
  几何后层在前层被裁剪后成为 VBuffer winner；RT 和阴影命中距离与之逐像素一致。
- 扩展 `stream_material_shading`：材质 ID 258 的 Coverage 参数热更新，resident、stream hardware、stream hybrid 的可见性一致。
- 显式 `--rhi-no-validation` 单独运行 Coverage、stream shading 和两项 OMM 测试：5/5 通过。
  同一场景实测 OMM 数量为 legacy 1、自定义 Coverage 0、移除程序后 1；该结果不来自旧验证层的 fallback。
- 最终 `MetallicShaderCompiler --filter Stream --debug-mode disabled --jobs 4`：38 个请求、0 失败。
  这是定向编译检查，没有声称完整 shader warmup 已完成；可选 warmup target 仍不属于默认构建。
- 修复了回归期间发现的流式空材质快照访问，以及 buffer 跨分支生成 `VariablePointersStorageBuffer` 的问题。
  保留初次失败日志，修复后重新跑完整测试。检查了 dissolve 后层与共享颜色更新的 PNG 输出。

256 帧冻结 HDR 对比中，Deferred 与 RTXCR Chiang 和 Phase 0 逐位一致；
完整套件上下文下 PathTrace 的 RGB RMSE 为 0.0002640552、最大绝对误差 0.04733187。
随后独立进程重跑 `*material_asset_phase0_equivalence*` 通过，但 PathTrace 仍非逐位一致：
RGB RMSE 为 0.00001984444、最大绝对误差 0.01300333，另外两项仍逐位一致。
差异原因尚未定位，不能归因于随机噪声，也不能声称所有输出逐位一致；原始 HDR 均为有限值。

验证记录、原始 HDR、预览 PNG 和 XML 位于本地 `build/material-phase8-*`，不纳入源代码。
没有执行长时间动画稳定性、NTC alpha 一致性或整场景显存压力验收。
