# M2 自定义 Value Program：状态快照与首批记录

状态（2026-11-13 静态对账）：**原路线图 M2 仍有剩余验收，不因稀疏分箱默认开启而宣布完整完成。**
受控前端、静态场景程序集、独立实例参数和 OpenPBR PT / VBuffer 接入已交付；
Coverage 同代发布/opacity 绕过由 [Phase 8](MaterialSystemPhase8.md) 交付，后端已被
[Native Slang Coverage](MaterialCoverageNativeSlang.md) 取代；统一 Value IR 和显式纹理采样由
[Phase 9](MaterialSystemPhase9.md) 交付。下方状态表区分已交付、被取代和仍缺的完整门槛。
本文 M2 指 [原路线图](MaterialSystemRoadmap.md#m2-跑通真正的自定义程序) 的自定义程序里程碑，
不等同于 [外部 M1–M4 验收](MaterialSystemM1M4Acceptance.md) 中的 M2 Unified Surface Material API。

## 当前生产稀疏分箱：默认值、生产者和消费者

- [ScenePathTracePass 设置](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L506-L514)
  在 Visibility Deferred 中默认 `materialBinning=true`、`programBinning=true`（Sparse Material Program Bins）。
  这是 Wave32 Deferred 的默认路径，不表示 PT 命中也使用屏幕 tile 队列。
- [updateMaterialProgramBins](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L2845-L2885)
  随 material generation 以完整编译 ProgramKey/ComputeKernel 去重，建立 instance→active executable slot 表；
  slot 0 是背景。OpenPBR Value 程序可以共享同一个场景 executable，Slab 有自己的特化 executable；
  **JSON/IR 程序数、实例数和 active bin 数不是同一个量**。
- [生产 record 调用](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L1960-L1978)
  将映射和 active table 长度传入 [MaterialBinning::record](../Source/Runtime/Render/MaterialBinning.cpp#L50-L66)。
  `programBinCount != 0` 选择 sparse，否则选择旧五类。
  [4096 上限](../Source/Runtime/Render/MaterialBinning.h#L11-L29) 是含背景的 active executable slots，
  不是 [64 个自定义 Surface/Coverage 程序预算](../Source/Runtime/Render/Material/MaterialValueProgram.cpp#L160-L185) 的扩大。
- [GPU 分类器](../Shaders/Features/VisibilityBuffer/VisibilityMaterialBinning.slang#L40-L183)
  从 visibility/source material 映射到 program bin，执行 count→prefix allocate→scatter→indirect arguments。
  [任务容量](../Source/Runtime/Render/MaterialBinning.cpp#L95-L109) 为
  `TileCount × min(ProgramBinCount, 32)` 条 8-byte task，控制/间接参数另按 bin 分配；分配失败向上传播，非静默丢像素。
- [真实间接消费者](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L2122-L2144)
  按 Family 分组顺序、原始 bin 的 mask/push/`bin*12` 参数偏移，调度 `materialProgramBins_[bin].program`，
  继续 fused evaluation + lighting；不是只注册了一个未消费的开关，也不承诺每份 JSON 都有独立 kernel。
  [Phase 11](MaterialSystemPhase11.md) 的 split 原型没有取代该生产分配器。
- [fallback 条件](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L2837-L2842)：
  关闭 `programBinning` 后，含自定义 Surface 程序的场景关闭分箱、走通用 kernel，避免旧固定因子误分类；
  无自定义 Surface 程序时保留五类对照。关闭 `materialBinning` 也走整屏路径。
  [native Wave32 检查](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp#L840-L845)
  在不支持的设备上明确返回 Unsupported、要求关闭 `materialBinning`，**不是自动回退或非 Wave32 已验收**。

## 首批实现的历史记录（2026-10-01）

以下支持范围、作者格式、编译细节和测试结果保留首批交付口径，不代表今天仍禁止纹理、MASK 或统一 IR。
当前扩展边界见 [Phase 9](MaterialSystemPhase9.md)、[场景 Slab 接入](MaterialSystemM4SceneClosure.md)
及 [MaterialGraph](MaterialSystemM5MaterialGraph.md)；这些后续交付不自动关闭本文末尾的数值/规模门槛。

### 首批支持范围

- 仅 opaque、lit、非 transmission 的 Surface；散射仍由 OpenPBRComposite 实现。
- 输出 `baseColor` RGB、`metallic`、`roughness`、`emissive` RGB，修改模型因子后继续沿用原有 glTF 纹理计算。Alpha、法线和几何数据不被程序修改。
- PT 每次确认命中都调用同一个 evaluator，包括后续 bounce；reference VBuffer 使用同一 evaluator。
- 首批自定义场景使用未分类的通用 VBuffer kernel；旧五类只适用于固定因子。该默认行为现已被上方的生产稀疏分箱接入取代，通用 kernel 保留为显式关闭分箱时的路径。
- Standard BSDF、Fiber、StreamAsset、realtime deferred、MASK/BLEND 和 transmission 的自定义程序明确拒绝。原有内建材质不受此限制。
- 禁止任意 Slang、外部资源访问、纹理采样、coverage 输出和写操作。程序化坐标目前逐点求值，没有抗走样/梯度传播保证，不用于证明 footprint 门槛。

### 首批最小创作入口

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

### 首批编译和发布

`MaterialValueProgramSet` 规范化 JSON 对象字段顺序，按源码排序去重，产生静态 Slang 函数和 switch。代码身份不包含实例数量、实例顺序、参数值或材质索引。不同 JSON 数值表示/不同表达式即使数学等价，也不承诺同键；通用优化 IR 留待 M3。

每个程序带 manifest：参数槽 mask、输入/输出 mask、节点数。该版本只有受控算术节点，外部资源和副作用集合固定为空。运行时参数绑定 97，CPU/Slang 记录为 80 字节：16 字节 ID/保留位，64 字节参数。旧 720 字节模型载荷不扩展。

生成的 `implementing SceneSurface;` 单元写入 `.cache/materials/<key>/MaterialValueDispatch.slang`，由 SceneSurface 通过 `__include` 加载；Slang 模块加载要求使用 `.slang` 扩展名。已有内容不符时拒绝使用。路径参与 Slang 请求，单元内容进入现有依赖追踪；场景程序集改变会重建 kernel，单纯参数更新保持 kernel key。缓存和测试图像均为本地输出。

程序集、参数缓冲和内建模型载荷由同一 `MaterialBindingGeneration` 发布并通过 frame 保留；上传/前端验证失败不替换原发布。场景参数变化沿用既有历史失效机制。初次 Slang/pipeline 编译失败仍走 M1 错误材质/诊断路径；本批并未扩大任意用户 Slang 热重载支持。

### 首批验证记录

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

## 原剩余清单逐项对账（2026-11-13）

| 原门槛 | 状态与可定位证据 | 尚需验收 / 不能外推的范围 |
| --- | --- | --- |
| Coverage 同代一致性、opacity 缓存失效 | **已交付（delivered），后端已取代（superseded）**：[Phase 8 共享快照与 OMM 协议](MaterialSystemPhase8.md#共享快照与生产接入)；当前 [MaterialBindingGeneration](../Source/Runtime/Render/Material/MaterialRuntime.h#L142-L161) 拥有同代参数和 Value 输入，[MASK 布局身份](../Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp#L991-L1020) 包含自定义程序存在性。Coverage 从统一 IR 切片 [生成原生 Slang](../Source/Runtime/Render/Material/MaterialValueProgram.cpp#L245-L270)，不再使用 Phase 8 指令尾部后端，当前协议见 [Native Coverage](MaterialCoverageNativeSlang.md#cache-and-live-edits)。 | 已有 ordinary-alpha 的 winner/ray/shadow 与 OMM 绕过/恢复历史记录；不是任意 NTC alpha、长时间动画或所有 visibility producer 都已验证。参见 [Coverage 边界](MaterialSystemPhase8.md#opacity-micromap-与边界)。 |
| 统一 Value IR、纹理采样和显式 footprint | **基础已交付（delivered），首批禁用纹理/无统一 IR 的限制已取代（superseded）**：[Phase 9](MaterialSystemPhase9.md#显式纹理-footprint) 记录 ExplicitLOD / SampleGrad / RayCone 探针；当前 [SceneSurface 回调](../Shaders/Modules/SceneSurface.slang#L899-L915) 调用 [MaterialValueTextureSampling](../Shaders/Modules/MaterialValueTextureSampling.slang#L48-L67)，[OpenPBRSurface](../Shaders/Modules/OpenPBRSurface.slang#L230-L244) 将生产 footprint 送入 Value evaluator。 | **仍缺（remaining）完整三平面及 RT cone/LOD 适配验收**，不是声称 RT 完全没有 adapter。需固定 mip/梯度参考，覆盖程序化 UV 的梯度或声明的近似 LOD、三平面混合、grazing 掠射、UV seam、跨 primitive 与 tile 边缘，比较 VBuffer 与 RT 的 footprint/数值误差。显式作者梯度不等于自动微分；现有双 mip 探针不能替代该矩阵。 |
| 稀疏 Program tile 与任务容量 | **已交付（delivered）**：见上方生产链；[Phase 6 的容量与 GPU 验收](MaterialSystemPhase6.md#稀疏任务生成) 覆盖 48 静态 Surface executable、mixed/partial tiles、prefix/indirect 和 4097 instances；[M4 场景接入](MaterialSystemM4SceneClosure.md#场景运行) 扩大到自定义 Slab。首批通用-only 描述已取代。 | 有界容量公式和合成 fixture 不等于完整规模/显存压力验收。需在目标分辨率、bin 数和混合度下记录实际 task 数、峰值容量/字节及分配失败行为；不能将 48 个静态 probe 当作 48 个生产作者材质。 |
| 非 Wave32 | **仍缺（remaining）分箱跨设备验收**：分箱器 [明确要求 native wave32](../Source/Runtime/Render/MaterialBinning.cpp#L40-L45)，调用方支持显式关闭 `materialBinning` 的整屏路径。 | 在非 Wave32 设备上分别确认默认分箱 Unsupported 诊断，以及显式关闭后的同场景自定义程序数值/覆盖正确性；没有声称非 Wave32 分箱已实现或自动 fallback。 |
| RT 每命中程序选择与屏幕外自定义材质 | **静态分发已交付（delivered），专门反射验收仍缺（remaining）**：[生成 Surface dispatcher](../Source/Runtime/Render/Material/MaterialValueProgram.cpp#L188-L245) 与 [M4 的 secondary-hit 接入/测试](MaterialSystemM4SceneClosure.md#验收)；屏幕 closure 缓存不替代 RT 命中求值。 | 已有相机背后 Slab 发光墙的 secondary-hit 间接照明数值记录，但它不是**屏幕外反射自定义程序**的专门 oracle。仍需固定镜面/粗糙反射场景、屏幕外程序对象及参数变化，对照独立参考数值，验证 secondary hit 不误用 primary 程序/屏幕缓存。 |
| 程序/实例规模及 tile 混合度压力 | **仍缺（remaining）完整扫描**：[前端 64 程序上限](../Source/Runtime/Render/Material/MaterialValueProgram.cpp#L184-L185) 与 [Phase 6 小型测量](MaterialSystemPhase6.md#2026-10-04-实测验收)、[Phase 11 的 2/8 Program split fixture](MaterialSystemPhase11.md#测量协议与结果) 只覆盖各自负载。 | 在真实 GPU 扫描 1/16/64/256 程序、1/100/1000 实例和不同 tile 混合度；记录实际编译 executable 数、冷/热编译时间、SPIR-V 体积、寄存器、运行 scope 与显存。当前 256 个自定义 Surface 程序应报告超预算拒绝，而不是伪装成支持；预算拒绝、静态 probe 或小型 timing 均不能代替完整压力实验。 |

以上只做源码/文档事实与链接对账；历史测试结果仍属于各阶段原记录，**本次未构建 renderer、未执行新的 GPU/视觉/性能测试**。
M2 保留 remaining 状态；Phase 11 的生产 fused 决策与这些剩余门槛独立，不能以 split 原型结果宣布 M2 全部完成。
