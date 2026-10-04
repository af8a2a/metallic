# M2 自定义 Value Program：首批实现

后续更新：[Phase 8 Coverage Program](MaterialSystemPhase8.md) 新增独立 coverage 根，并允许 MASK 上的 Surface 输出。
下文记录 M2 首批实现时的支持范围；当前 Coverage 约束和验证以 Phase 8 文档为准。

状态：**M2 进行中**。本批实现受控前端、静态场景程序集、独立实例参数，以及现有 LookDev 的 OpenPBR PT / reference VBuffer 接入。M0 场景及默认材质不变。完整 M2 的 coverage、TextureFootprint、稀疏 Program tile 和压力性能门槛仍按 [路线图](MaterialSystemRoadmap.md) 执行。

## 支持范围

- 仅 opaque、lit、非 transmission 的 Surface；散射仍由 OpenPBRComposite 实现。
- 输出 `baseColor` RGB、`metallic`、`roughness`、`emissive` RGB，修改模型因子后继续沿用原有 glTF 纹理计算。Alpha、法线和几何数据不被程序修改。
- PT 每次确认命中都调用同一个 evaluator，包括后续 bounce；reference VBuffer 使用同一 evaluator。
- 自定义场景使用未分类的通用 VBuffer kernel。旧五类分箱只适用于未修改的固定因子；M2 稀疏程序分箱尚未实现。无自定义程序的场景保持原有分箱。
- Standard BSDF、Fiber、StreamAsset、realtime deferred、MASK/BLEND 和 transmission 的自定义程序明确拒绝。原有内建材质不受此限制。
- 禁止任意 Slang、外部资源访问、纹理采样、coverage 输出和写操作。程序化坐标目前逐点求值，没有抗走样/梯度传播保证，不用于证明 footprint 门槛。

## 最小创作入口

`scene::RenderMaterial::valueProgram` 是 JSON 源码字符串；空字符串表示使用内建材质输入。`valueParameters` 是 16 个 float，按四个 float4 参数槽解释。通过 `SceneDocument::setMaterialProperties()` 编辑，随现有材质 overrides 保存/重载；编辑器专用 UI 属于后续 M3。

```json
{
    "version": 1,
    "baseColor": {"op": "parameter", "index": 0},
    "roughness": {"op": "parameter", "index": 1}
}
```

表达式为标量常量、四分量常量或 `op` 对象。标量广播到 float4；金属度/粗糙度读取 x。支持 `parameter(index:0..3)`、`position`（世界坐标）、`geometryNormal`（稳定几何法线）、`uv`、原始 `baseColor/metallic/roughness/emissive`。输入节点不接受 `args`；算术节点用 `args` 数组：`add/mul/dot` 两个参数、`mix` 三个、`sin/fract/abs/saturate` 一个。`dot` 是四维点积且广播结果；`mix` 权重 clamp 到 [0,1]。所有输出读取同一份原始输入，不依赖 JSON 字段顺序。

预算为每程序 16 KiB 源码、256 个表达式节点、表达式深度 24、JSON 嵌套 64，每场景最多 64 个自定义程序。常量、参数和算术中间值限制在 ±1e6；baseColor、metallic、roughness clamp 到 [0,1]，emissive clamp 到 [0,1e6]。程序 0 保留内建行为。

示例放在 `Asset/LookDev/MaterialPrograms/`：

- `ProceduralRust.value.json`：两个颜色按世界空间正弦斑块混合；参数 0/1 是底色/锈色，参数 2 是空间频率，参数 3.x 是粗糙度。这是用于验证计算路径的锈蚀外观示例。
- `Stripes.value.json`：世界空间条纹，使用相同参数约定。

例如锈蚀参数为 `[0.12,0.25,0.38,1, 0.85,0.12,0.015,1, 7,19,13,0, 0.7,0,0,0]`。读入示例文件文本到 `valueProgram` 后调用材质 setter。测试输出目录中的 `value-roundtrip.metallic_scene.json` 是持久化入口的可运行例子。

## 编译和发布

`MaterialValueProgramSet` 规范化 JSON 对象字段顺序，按源码排序去重，产生静态 Slang 函数和 switch。代码身份不包含实例数量、实例顺序、参数值或材质索引。不同 JSON 数值表示/不同表达式即使数学等价，也不承诺同键；通用优化 IR 留待 M3。

每个程序带 manifest：参数槽 mask、输入/输出 mask、节点数。该版本只有受控算术节点，外部资源和副作用集合固定为空。运行时参数绑定 97，CPU/Slang 记录为 80 字节：16 字节 ID/保留位，64 字节参数。旧 720 字节模型载荷不扩展。

生成 include 写入 `.cache/materials/<key>/MaterialValueDispatch.hlsli`，已有内容不符时拒绝使用。路径参与 Slang 请求，include 内容进入现有依赖追踪；场景程序集改变会重建 kernel，单纯参数更新保持 kernel key。缓存和测试图像均为本地输出。

程序集、参数缓冲和内建模型载荷由同一 `MaterialBindingGeneration` 发布并通过 frame 保留；上传/前端验证失败不替换原发布。场景参数变化沿用既有历史失效机制。初次 Slang/pipeline 编译失败仍走 M1 错误材质/诊断路径；本批并未扩大任意用户 Slang 热重载支持。

## 验证与剩余门槛

2026-10-01，Windows / MSVC Release，复用 `build-scheduling-release`：

- `MetallicRHITests`、`LookDev` 和 `MetallicSceneTests` 构建通过。
- 开启 Vulkan validation 的 17 项回归全部通过、无跳过、无 VUID：M2 四项、M1 材质运行时、现有材质分箱、VBuffer 材质编辑、StreamAsset 着色/透射/阴影；日志 `build/material-value-m2-final-tests.log`。
- 场景材质导入/编辑/文档回归 5 项通过；日志 `build/material-value-m2-scene-tests.log`。
- 增加完整光照检查后，M2 四项再次通过；日志 `build/material-value-m2-lit-tests.log`。`material_value_gpu_abi` 验证 GPU 80 字节步长、全部 16 个参数偏移、默认及两个自定义程序的数值、几何输入和 alpha 保留。首次实验发现的 `uint3` 对齐问题已用四个标量 uint 头修复。
- `material_value_lookdev` 在原 LookDev 上添加程序、编辑参数、替换程序，移除后两条路径的原始浮点画面逐字节恢复；另运行 32 帧完整光照并检查有限值，人工查看 PT/Deferred 输出。图像在 `build/material-value-m2-lit/`，`rust-lit-Reference.color.png` / `rust-lit-Deferred.color.png`；PT 的有限采样噪声仍可见，不能据此宣称长期时域稳定性或两种估计器等价。
- 默认 LookDev PT、Deferred、Claire RTXCR 的 256 帧 RGBA32F 输出，与 M1 已验收结果的 SHA256 全部相等。比较记录 `build/material-value-m2-final/hdr-baseline-comparison.json`。

复跑命令（先构建目标）：

```powershell
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_value*' --rhi-validation --output-dir build/material-value-m2-check
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter='SceneEditing.*Material*:SceneImport.*Materials'
```

尚未关闭：coverage 同代一致性和 opacity 缓存失效；三平面纹理及 footprint；稀疏任务容量与非 wave32；屏幕外反射的专门数值验证；真实 GPU 上 1/16/64/256 程序与不同 tile 混合度的编译时间、SPIR-V 体积、寄存器和运行成本扫描。当前前端预算测试接受 1/16/64、拒绝 256，不能代替这些性能实验。
