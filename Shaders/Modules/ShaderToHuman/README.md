# ShaderToHuman in Metallic

Metallic 使用 Electronic Arts [ShaderToHuman](https://github.com/electronicarts/ShaderToHuman)
v14 的原生 Slang 移植，支持 Gather 文本/2D 图形、交互控件、3D 调试几何和 Scatter 写入。

## 来源与维护

- 固定上游提交：`70ea227f3d504a19d9dd0539a832b68999426cca`。
- 本目录仅包含原生 Slang 移植；原始 vendored HLSL 已移除。
- [Upstream.json](Upstream.json) 记录来源提交、历史源文件校验值及移植差异。
- 模块入口为 [ShaderToHuman.slang](../ShaderToHuman.slang)。
- BSD-3-Clause [许可证](LICENSE.txt)和 [NOTICE](NOTICE.txt)随移植实现保留。

升级时按记录的上游来源核对算法变化，移植后运行模块编译及 GPU 行为测试。
上游 Gigi 工程、网页服务和示例资源不是 Metallic 构建依赖。

## 模块与 API

```slang
import ShaderToHuman;
using Metallic.ShaderDebug;

ContextGather ui;
s2h_init(ui, float2(dispatchId.xy) + 0.5);
s2h_setCursor(ui, float2(8, 8));
s2h_printTxt(ui, _I, _D);
s2h_printSpace(ui, 1);
s2h_printInt(ui, selectedMaterialId);
linearColor = composite(linearColor, ui);
```

原有 `s2h_*` 函数、字符常量及 Context 字段保留名称，统一放在 `Metallic.ShaderDebug`。
旧 `Features/Debug/ShaderToHuman.slang` / `ShaderToHumanScatter.slang` 文本入口已移除，
调用方改为上述 `import` 和 `using`；原 `ShaderDebug::composite` 改为此命名空间的 `composite`。
`S2H_VERSION` 现在是值为 14 的公开 `int` 常量，可用于表达式，不能用于预处理 `#if`。
库不声明 GPU 资源或 push constant，也不会自动向 RenderGraph 增加调试绘制。

模块实现分为 `Font.slang`、`Gather.slang`、`Geometry.slang`、`Scatter.slang`，由模块入口
通过 `__include` 组织。固定使用原版内嵌 8×8 字体，不再接受 `S2H_GLSL`、
`S2H_DISABLE_EMBEDDED_FONT` 或 `S2H_FLT_MAX` 宏定制；调用方局部宏不会影响模块。

Gather 在每个输出像素执行相同的绘制命令。Compute 整数像素坐标加 0.5；Fragment
的 `SV_Position.xy` 已经是像素中心。查看选中像素数据时，各 invocation 读取同一份数据。
`ContextGather.dstColor` 是预乘 Alpha 的线性颜色；`composite` 保持线性输出，交给曝光及
FinalBlit，不重复转换 sRGB。希望文字不受曝光影响时，在曝光后、最终输出变换前合成。

3D 使用 `Context3D` 和 `s2h_init(context, rayOrigin, normalizedRayDirection)`。
场景遮挡距离写入 `context.depth`，它是沿射线的距离，不是 Vulkan/Reversed-Z 深度。

## 显式回调

跨模块不再依赖调用方定义的同名全局函数，改为静态泛型接口：

| 功能 | 接口与签名 | 调用方式 |
| --- | --- | --- |
| Scatter | `IScatterSink`: `[mutating] void writePixel(int2 pixel, float4 color)` | `ContextScatter<MySink>`；`s2h_init(ui, sink)` |
| 整数表格 | `IIntTable`: `bool lookup(uint column, uint row, out int value)` | `s2h_tableInt(..., source)` |
| 浮点表格 | `IFloatTable`: `bool lookup(uint column, uint row, out float value)` | `s2h_tableFloat(..., source)` |
| 曲线 | `IFunctionPlot`: `float evaluate(uint functionId, float x)` | `s2h_function(..., source)` |
| 3D 阴影 | `IScene3D`: `void draw(inout Context3D context)` | `sceneWithShadows(context, scene)` |

表格和曲线的 source 参数追加在原有参数末尾。Scatter 的 sink 存在 context 中，其可变状态
通过 `ui.sink` 访问。调用方负责写入目标、边界裁剪、输出初始化、invocation 所有权和同步；
多个 invocation 必须写入互不重叠的区域。库不提供跨 invocation 同步。

完整 sink 实现见 [Scatter 示例](../../Features/Debug/ShaderToHumanScatterExample.slang)，表格、曲线、3D scene
实现见 [GPU 探针](../../../tests/rhi/shaders/ShaderToHumanProbe.slang)。

## 示例与验证

- [Gather/3D 示例](../../Features/Debug/ShaderToHumanExample.slang)：compute `shaderToHumanExampleMain`
  和 fragment `shaderToHumanExampleFragmentMain`，320×192 展示完整内容。
- [Scatter 示例](../../Features/Debug/ShaderToHumanScatterExample.slang)：`shaderToHumanScatterExampleMain`，
  单 invocation 在已初始化输出上绘制；相邻 dispatch 之间由调用方安排资源屏障。
- 运行时 program 名仍为 `Features/Debug/ShaderToHumanExample` 和
  `Features/Debug/ShaderToHumanScatterExample`，使用 Metallic 现有模块搜索路径。

在已有启用测试的构建目录中执行，例如：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=*shader_to_human* --rhi-validation
```

编译测试覆盖 compute、fragment、泛型 Scatter，以及唯一的 Slang 模块依赖列表，
并确认生产程序不依赖上游 HLSL。GPU 测试执行完整字体、数字、透明合成、2D 图形、
表格、曲线、交互控件、3D 阴影及带裁剪的 Scatter，检查有限值、关键像素和有效绘制覆盖。
结果图 `ShaderToHuman.png` 上半为 Gather/3D，下半为 Scatter。
原始 HLSL 移除前已完成原版/移植版逐像素对照；当前测试不再依赖或编译原版。
图像使用线性值直接量化，不替代 Metallic 的 HDR 显示变换。
