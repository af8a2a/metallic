# Material System Phase 9 — Value IR

Surface 和 Coverage 现在从同一个 [MaterialValueIR](../Source/Runtime/Material/MaterialValueIR.h) 自动切片。
JSON 只是作者入口；后端消费经过验证和优化的节点，不再分别递归解释两套 JSON 表达式。
纯 CPU IR 位于共享 Material 层，Feature System 与 render 后端使用相同的规范化结果。

## 节点与作者格式

IR 是不可变、拓扑排序的 float4 DAG。标量常量广播到四个分量；标量消费者显式使用 x。
保留 v1 嵌套表达式，新增 v2 的命名节点/引用格式，例如：

```json
{
  "version": 2,
  "nodes": {
    "scaled": { "op": "mul", "args": [
      { "op": "parameter", "index": 0 }, 0.5
    ] }
  },
  "outputs": {
    "baseColor": { "ref": "scaled" },
    "roughness": { "ref": "scaled" },
    "coverage": { "ref": "scaled" }
  }
}
```

通过既有 `RenderMaterial::valueProgram` 和 `SceneDocument::setMaterialProperties()` 发布，参数仍为四个 float4 槽。
v2 引用可以指向后定义的节点；活跃路径上的未知引用和环会被拒绝。
源码最多 16 KiB、JSON 嵌套 64、表达式/引用深度 24、访问预算 1024、最多 256 个独立节点。
常量和动态参数必须有限且位于 ±1e6，算术中间值保持既有有限范围限制。

| IR 节点 | JSON 表达 |
| --- | --- |
| Constant / Parameter | 数字、四分量数组；`parameter` + `index:0..3` |
| Add / Multiply / Lerp / Dot | `add`、`mul`、`mix`、`dot`；mix 权重 saturate，Dot 是四维点积 |
| Clamp | `clamp` 的三个 args：值、下界、上界；语义为 min(max(value, lower), upper) |
| Normalize / NormalMap | `normalize` 对 xyz 归一化；`normalMap` 先将 xyz 从 [0,1] 解码到 [-1,1] 再归一化，w=0 |
| UV Transform | `uvTransform` 的 args 为 UV、两行 float4；每行 xy 为线性部分、z 为平移，输出 xy00 |
| Swizzle | `swizzle` + 四字符 `components`（xyzw 可重复）和一个参数 |
| Select | `select` 的 args 为条件、真值、假值；条件 x>0 选真值 |
| TextureSample | `textureSample` + `texture`、`footprint` 和显式参数 |
| 兼容节点 | `sin`、`fract`、`abs`、`saturate`、`position`、`geometryNormal`、`uv`、`baseColor`、`metallic`、`roughness`、`emissive`、`alpha` |

Normalize/NormalMap 的极小向量退化为 (0,0,1,0)。NormalMap 产出切线空间数值，
不自行修改 `HitInfo` 或构造 TBN。现有法线贴图和稳定 authored/world-space normal 规则保持不变。
输出仍为 baseColor RGB、metallic、roughness、emissive RGB 和独立 coverage。

## IR pass 与身份

1. 从输出根遍历和验证；只对可达表达式做操作、类型、参数和资源策略检查。
2. 常量折叠；静态 Select 在访问未选分支前删除该分支。
3. CSE 将相同的规范化节点共享，包括不同输出和不同作者节点名称下的相同子表达式。
4. DCE 删除折叠后的孤立节点、未引用定义及死分支。死分支中的不存在纹理/不支持采样不会进入后端。
5. 从实际存活节点分析 parameter/input/texture/footprint mask，以及 geometry、texture、normal mapping、coverage 等 Feature。
6. 按输出根自动生成 Surface 与 Coverage 子图，再生成各自的最小 manifest 和后端代码。

规范化序列包含 IR 版本、操作、float32 常量、拓扑依赖、纹理槽、footprint 与输出，使用确定性 FNV-1a 64 位 hash。
节点名称、JSON 字段顺序、死代码、实例数量与参数值不进入身份。常量折叠后的等价代码得到相同 hash；
不承诺任意数学等价表达式相同，也不做可能改变浮点语义的代数重排。hash 不是安全散列，缓存仍比较完整生成文本。

`MaterialValueManifest` 新增 IR hash、texture/footprint/feature mask。
Surface 程序按规范化 Surface IR 去重，Coverage 指令按规范化 Coverage IR 去重。
Feature 的 ProgramSignature 升为 v2，VisibilitySignature 升为 v3，分别消费对应切片。
Coverage-only v2 图不会误选 Surface general 路径；Coverage 变化不会改变 Surface executable 身份。

## 显式纹理 footprint

[MaterialValueTextures.slang](../Shaders/Features/PathTracing/MaterialValueTextures.slang)
通过现有材质纹理槽采样，不增加 GPU descriptor ABI。可用槽为 baseColor、metallicRoughness、normal、occlusion、emissive、specular。

| 策略 | args | LOD 来源 |
| --- | --- | --- |
| ExplicitLOD | UV、LOD（x） | 作者提供的当前驻留图像 mip 层级 |
| SampleGrad | UV、dUVdx（xy）、dUVdy（xy） | 显式梯度经过材质 UV 变换后，按纹理尺寸计算各向同性 LOD |
| RayCone | UV、LOD bias（x） | 生产 Surface context 的 ray cone / 已重建 footprint，加纹理尺寸、材质 UV 变换和 bias |

不允许隐式采样策略，不调用 ddx/ddy。过滤为确定性 repeat、双线性加 mip 线性插值；
普通纹理复用格式解码和驻留 mip floor。缺失纹理返回 float4(1)。NTC 复用现有尺寸查询和 texel 解码入口。
SampleGrad 的输入必须由作者/生成器提供；没有新增自动微分。自定义非线性 UV 的梯度或 RayCone bias 需要作者显式处理。
当前没有增加各向异性过滤或独立的 IR 纹理 residency feedback 调度。

OpenPBR 的因子求值已由 IR 生成代码完成，仍发生在原有 glTF 纹理调制之前；
例如将 metallicRoughness 采样写入 roughness，写入的是 roughness 因子，后续原有贴图仍会相乘。
这保持 M2 的参数语义，不将其暗中改为最终 Closure 输入覆盖。
示例 [TexturedDissolve.value.json](../Asset/LookDev/MaterialPrograms/TexturedDissolve.value.json)
展示 RayCone 纹理因子与独立 Coverage；使用 MASK，参数槽 0.x 控制溶解偏移。

## Coverage 与兼容边界

Coverage 后端仍为 64 条指令的有界只读程序，现从同一 IR 自动切片并支持新增算术节点。
其 `alpha` 输入声明 baseColor 资源和 ExplicitLOD 策略，使用 Phase 8 固定的最精细驻留 mip alpha 采样。
Coverage 不接受任意 TextureSample、Surface 位置/法线输入或不同 pass 的 footprint，避免 VBuffer、RT、shadow 产生不同可见性。
已删除的无效采样不会触发这些后端约束。

Surface 仍限制为 lit、非透射、非 Fiber 的 OPAQUE/MASK；Coverage-only 的约束与 Phase 8 相同。

后续的 [Painter LookDev 场景接入](OpenPbrLookDev.md#painter-验证场景选项) 增加了
OpenPBR coat/fuzz/各向异性输入，以及显式 `attenuationColor` 的 Surface 透射例外；
未声明该输入的旧 Value 程序和 Slab 仍保留原有透射限制。
80 字节参数记录、720 字节 legacy 材质载荷、共享 generation 发布与 OMM 禁用/恢复协议不变。
无自定义程序的内置 OpenPBR 路径不运行 Value IR。
本阶段没有实现 MaterialGraph 编辑器、任意 shader language、Closure IR、资产格式扩展或 NTC alpha 跨路径一致性。

## 验证（2026-10-04）

复用 MSVC/Ninja Release `build-scheduling-release`，成功构建 `Metallic`、`MetallicRHITests`、`MetallicSceneTests`。

- 完整 Scene CTest 通过，包括 v1/v2 优化后签名一致、死节点不污染签名及 Coverage-only 保持 Surface 分类。
- 带验证层的材质 GPU/资源回归 **47 项：43 通过、4 跳过、0 失败**。
  跳过项为缺少外部数据的 Zorah probe、当前构建未启用 NRD 的 RTXDI 完整图、旧验证层不支持 KHR OMM 的两项测试。
- 新增 `material_value_ir` 验证环/未知引用/非法活跃节点拒绝，常量折叠、DCE、CSE、资源/Feature 分析、切片与稳定 hash。
  也检查静态未选分支中的无效 TextureSample 被删除，以及等价优化 IR 在生产程序集内复用身份。
- 新增 `material_value_ir_texture_footprints` 使用实际双 mip GPU 纹理（mip 0 红、mip 1 绿）和生产采样 adapter。
  ExplicitLOD 0/1/0.5、SampleGrad、RayCone 五组结果与解析预期一致；另五组动态 Clamp、Normalize、NormalMap、UV Transform、Swizzle/Select 通过。
  专用 Vulkan 验证错误计数为 0。初始探针的 mip 屏障/视图范围问题已修正，失败日志保留。
- 原有 Value GPU ABI、参数发布失败保留、LookDev 程序、内置闭包、Program 分箱和流式材质回归均通过。
- Coverage 的纹理 alpha 与共享 Surface 参数步骤改为 v2 节点引用，并实际执行新增 Swizzle/Clamp。
  八组状态合计 30,752 个内部像素的 VBuffer winner、ray query、生产阴影结果一致，已检查输出 PNG。
- 显式 `--rhi-no-validation` 的 OMM/IR/stream 定向组 **7/7 通过**。
  OMM 数量在 legacy → 自定义 Coverage → 移除程序时实测 **1 → 0 → 1**。
- `MetallicShaderCompiler --filter Stream --debug-mode disabled --jobs 4` 的 **38 个请求全部成功**；不代表完整 warmup 验收。

完整套件的 256 帧 HDR 对比：Deferred 与 RTXCR Chiang 和 Phase 0 逐位一致；PathTrace 的 RGB RMSE
为 0.0002632135、最大绝对误差 0.04733241。所有原始输出均为有限值，未将 PathTrace 报告为逐位一致。
随后独立进程重跑 `*material_asset_phase0_equivalence*`：PathTrace、Deferred、Chiang 三项原始 RGBA32F
均与 Phase 0 逐位一致。完整套件上下文差异的原因仍未定位；独立结果不用于掩盖该差异。
最终探针生命周期整理后，带验证层的 `*material_value_ir_texture_footprints*` 再次通过。

最终日志仍保留 EOS overlay JSON / `E:\Validation.json` 缺失的 loader 警告；未隐藏这些环境消息。
没有声称 NTC 纹理的新增采样策略、长时间动画、任意 MaterialGraph 编辑流程或性能压力测试已通过。
原始证据位于本地 `build/material-phase9-*`，不进入源代码。
