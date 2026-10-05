# M5 — MaterialGraph 与只读文本前端

本页对应用户外部路线图的 **M5 / Phase 13–14**，不使用仓库旧规划中调度阶段的 M5 编号。2026-10-05 实现首版 authoring 闭环：图或 SDK 表达式 → Value/Closure IR → `.materialdef` / `.material` → SceneDocument → 现有 PT / Deferred lighting。

## 使用

启动 `LookDev`，在 Scene Browser 选择一个材质，或只引用一个材质的模型，然后点击菜单栏 **Material Graph**（也可从 `Window` 打开）。

1. 默认图为 Parameter → OpenPBR → Output。`Add node` 添加节点，拖动输出与输入 pin 连线；选择节点后在右侧修改参数或删除。Surface 与 float4 是不同 pin 类型，编译时检查。
2. `Compile` 显示诊断和最后一次成功编译的 reflection；`Apply to selected material` 编译并应用。编译或运行时合同校验失败时保留当前场景。应用使用场景的 Ctrl+Z / Ctrl+Y 撤销栈，并重置渲染历史。
3. Material Inspector 的 `Material program parameters` 编辑四个 float4 实例槽；无需重新生成程序。图里的 Parameter default 是下一次应用/新实例的初始值，不是实时绑定。
4. `Save graph` / `Load graph` 保存和读取 `.materialgraph`。图结构编辑有独立 `Undo graph` / `Redo graph`；关闭面板不会销毁草稿。退出应用前需保存草稿。
5. `Export assets` 在 Graph path 同目录导出 `.materialdef`、`.material` 和 `.reflection.json`。导出实例初始为空资源覆盖；`Texture` 节点引用所选场景材质已有的语义资源槽。为可移植实例设置 `resources` URI，资源根目录遵循现有 MaterialAssetLibrary 规则。
6. `Ctrl+S` 保存场景中的已应用 IR 和实例值。场景不依赖 Graph 文件仍然存在；重新编辑需要单独保存的 authoring 文件。

可直接加载 `Asset/Materials/Graphs/CoatedPaint.materialgraph`。首版只对**一个选中材质**应用；多材质模型先在 Scene Browser 选择其具体材质。

批量或离线编译使用独立目标（无需启动编辑器）：

```powershell
cmake --build build-scheduling-release --target MetallicMaterialCompile
.\build-scheduling-release\Source\MetallicMaterialCompile.exe `
  Asset/Materials/Graphs/CoatedPaint.materialgraph build/MaterialGraph/CoatedPaint.materialdef
.\build-scheduling-release\Source\MetallicMaterialCompile.exe `
  Asset/Materials/Graphs/CoatedPaint.material.slang build/MaterialGraph/CoatedPaintSDK.materialdef `
  Asset/Materials/Graphs/CoatedPaint.parameters.json
```

## 节点和数值语义

| 节点 | 输入/作用 |
|---|---|
| Constant / Parameter | float4 常量；实例槽 0–3，默认值不进入程序身份 |
| UV / Position / GeometryNormal | 只读当前 Surface 上下文，世界空间位置/几何法线 |
| BaseAlpha | Coverage 专用的底色纹理 alpha；Surface 使用会报错 |
| Texture | 六个既有语义槽；UV、LOD 或 RayCone bias；ExplicitLOD / RayCone |
| Math | add、multiply、dot、lerp、clamp、select、sin、fract、abs、saturate、normalize |
| Swizzle | float4 通道选择；例如 packed metallic/roughness 的 z / y |
| NormalMap | 将编码 [0,1] normal 解码并归一化为切线空间方向 |
| OpenPBR | baseColor、metallic、roughness、normalTS、coatWeight、coatRoughness |
| Slab | reflectance、opticalDepth，沿用 M4 的漫反射 Slab prototype |
| Mix / Layer | 两个 Slab 的混合/有序组合，沿用已有预算和模型限制 |
| Output | Surface、emissive、coverage；指定一个根输出 |

所有 Value pin 为 float4；标量广播，标量消费者读取 x。dot 使用四个分量；normalize 使用 xyz 并将 w 清零；lerp 的权重被钳制到 [0,1]。有限值预算 ±1e6。

颜色使用 **linear Rec.709** authoring basis，发布表面值时转到 renderer working space。Graph 的 `textureSampleLinear` 对 Color 槽完成 sRGB transfer / primaries 转换，对 Data 槽保持原值；无 normal 资源时返回中性法线编码。原有 `textureSample` 保留历史语义，不改变旧 Value IR 的输出。

OpenPBR Graph 的四个 `surface*` 输出是最终表面值，不再乘一次 legacy 纹理或 factor。其他未由图暴露的属性（IOR、specular 等）沿用选中材质。normalTS 使用 authored normal 与稳定 TBN，最终阶段才根据调用方需要 face-forward。默认图使用 (0,0,1) 法线；要使用已有 normal 纹理，应连接 Texture(normal) → NormalMap → OpenPBR.normalTS。

Coverage 恒为 1 会消除；非恒 1 的 Coverage 自动选择 MASK。其程序继续走独立的只读 Coverage slice；通用纹理/世界空间读取不能混入该 slice，使用 BaseAlpha 或 UV/参数/数学。`Mix/Layer(OpenPBR,...)` 与多于两个 Slab 的组合明确拒绝。

## SDK expression profile v1

文本模式是 **Slang 语法的受限表达式前端**，不加载任意原生 Slang module。`import MaterialAuthoringSDK;` 是此前端识别的逻辑 SDK 名称；该文件不应直接交给 `slangc`。程序经同一 IR 编译器生成生产 Slang，与图程序共享 renderer contract。任意 Slang 函数、循环、预处理器、资源写入、动态资源声明和新增 BSDF 类型仍不属于此版本。

```slang
import MaterialAuthoringSDK;
Material evaluate()
{
    let color = parameter(0);
    let paint = openPBR(color, 0.0, 0.3);
    return withCoat(paint, 1.0, 0.08);
}
```

`Material evaluate()` 有一个隐式只读 Surface context 与当前 instance。局部声明支持 `let`、`float4`、`Material`；支持有限数字、float4 字面量、括号及 `+ - *`。标识符单次定义，不支持赋值或修改 context。接口/函数/参数数量错误报告 source byte offset 和邻近 token；IR/预算错误来自共用验证器。

| SDK 函数 | 合同 |
|---|---|
| parameter(slot) | 当前实例的四个 float4 槽；slot 必须是 0–3 字面量 |
| uv(), position(), geometryNormal(), alpha() | 对应 Graph 上下文节点；alpha 只供 Coverage |
| sampleBaseColor/MetallicRoughness/Normal/Occlusion/Emissive/Specular(uv,bias) | 固定语义资源，显式 RayCone footprint；Color 输出 linear Rec.709 |
| x/y/z/w(value) | 通道广播成 float4 |
| add/multiply/dot/lerp/clamp/select/sin/fract/abs/saturate/normalize/normalMap | 与 Graph 同名节点语义相同 |
| openPBR(color,metallic,roughness) | Surface 描述；withNormal(surface,normal)、withCoat(surface,weight,roughness) 修改其输入 |
| slab(reflectance,opticalDepth), mix(a,b,weight), layer(a,b) | 沿用 Closure IR 的有界 Slab 后端 |
| surface(material,emissive,coverage) | 显式 Output |

参数默认值在 CLI 使用独立 JSON，在编辑器取 Graph 的 Parameter 节点。文本模式可在 Graph path 的同名 `.material.slang` 保存/加载源码；导出时同时保存源码。实例仍可覆盖默认值。

这使新的程序化 Surface 可以不修改 renderer 而使用现有通用 lighting；它不等价于 Phase 14 设想的任意原生 Slang 扩展 SDK。完整 module/interface 加载及新增闭包实现仍是后续范围，不应以本页验收替代那部分能力。

## 编译和发布边界

运行时代码只收到 MaterialDefinition 的 IR 与实例资源/参数；不携带 Graph 节点、位置或文本前端标签。CSE、常量折叠、DCE、closure budget 沿用既有 IR。节点 ID 重命名、位置、未连接节点和默认参数值不改变 canonical signature；改变实际计算/闭包结构会改变程序身份。reflection 的 signature 是完整 IR 身份；可执行 key 另外包含 target、ABI 与 Surface/Coverage 分离后的实现。

当前预算：128 Graph 节点、16 KiB SDK/IR、256 唯一 Value 节点、4×float4 实例参数、最多两个 Slab；场景最多 64 Surface 和 64 Coverage 程序。无动态资源索引、无任意 side effect。反射声明活跃资源、参数/feature/footprint masks、closure family/records/payload；不把节点图变成 GPU ABI。已有 80-byte Value instance、720-byte legacy material record 不变。

## 验收

自动化入口：

```powershell
cmake --build build-scheduling-release --target LookDev MetallicSceneTests MetallicRHITests MetallicMaterialCompile
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter=MaterialGraph.*:MaterialAssets.*
.\build-scheduling-release\tests\MetallicRHITests.exe --filter material_graph_scene --output-dir build/material-graph-gpu
ctest --test-dir build-scheduling-release -R '^MetallicLookDevMaterialGraphSmoke$' --output-on-failure
```

CPU 检查图/文本 canonical 一致、pin/cycle/default/side-effect 拒绝、DCE、Math palette、packed 通道、Coverage stage 和 Definition → Instance → Scene 保存重载。GPU 场景测试比较 legacy / Graph / SDK 的实际纹理、倾斜 normal、PT / Deferred 输出，检查分箱一致性、实例值更新及程序 key 不变。编辑器测试绘制真实 ImNodes canvas，通过 ImGui 鼠标输入点击 Apply，验证 OpenPBR / Slab、撤销重做、文件往返与失败保留上次结果。

GPU 图像和机器可读误差保存在 `build/material-graph-gpu/`；编辑器日志在 `build-scheduling-release/tests/material-graph-smoke/editor.log`。测试产物不入源码库。编辑器 smoke 不等价于完整人工拖拽、布局和交互验收。

### 2026-10-05 实测结果

- MSVC / `build-scheduling-release`：Metallic、LookDev、MetallicMaterialCompile、MetallicSceneTests、MetallicRHITests 均构建成功。离线编译工具保持 `EXCLUDE_FROM_ALL`，不会成为编辑器构建的强制依赖。
- `MaterialGraph.*:MaterialAssets.*`：28/28 通过。
- RHI 选集：18/18 通过，无 skipped、无 Vulkan validation error。包含 Graph 场景、Value IR 与 Closure 场景、纹理 footprint、80-byte instance ABI、Coverage compiler/主命中/阴影/射线、OpenPBR 三阶段、通用 lighting、program/guide contract 和资产上传等价性。
- LookDev MaterialGraph / MaterialInspector 两个 smoke：2/2 通过。Graph 测试通过真实 ImGui Apply 点击；原有 Roughness 拖拽与材质间交互、撤销、保存重载也通过。暖缓存重跑合计约 6.5 秒；冷缓存会触发较长的 Slang/驱动 PSO 编译。
- Graph ↔ SDK 图像误差 0；binned ↔ unbinned 误差 0；Graph ↔ legacy 纹理/倾斜法线参考的 RGB 相对 L1 误差：Deferred `9.3678e-8`，PT `9.3614e-8`。
- 参数更新后 executable key/source 不变，实例字节变化，图像相对变化 `1.27137`；并验证图/材质/场景持久化。

完整回归证据：`build/material-graph-regression.log`、`build/material-graph-regression/MaterialGraphAcceptance.json` 及同目录 PNG/HTML report。CPU 与编辑器日志分别为 `build/material-graph-cpu.log`、`build/material-graph-editor-test.log`。本机用 Vulkan SDK 1.4.350 的显式 validation layer 与空 implicit-layer 目录避开系统的陈旧 layer 注册；测试未验证被该版本 layer 禁用的 KHR OMM 硬件路径。
