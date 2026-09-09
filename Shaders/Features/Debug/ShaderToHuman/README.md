# ShaderToHuman in Metallic

引入 [Electronic Arts ShaderToHuman](https://github.com/electronicarts/ShaderToHuman)
的核心 Shader 头文件，用于在输出图像中打印数字、绘制文本、2D 图形和 3D 调试几何。
这不是 CPU 日志或 GPU 崩溃转储工具。

## 来源

- 上游提交：[`70ea227f3d504a19d9dd0539a832b68999426cca`](https://github.com/electronicarts/ShaderToHuman/tree/70ea227f3d504a19d9dd0539a832b68999426cca)
- 接口版本：`S2H_VERSION = 14`
- `include/`、`LICENSE.txt` 和 `NOTICE.txt` 保留上游原始内容与文件名。
- [Upstream.json](Upstream.json) 记录固定提交和这六个文件的 SHA-256；当前没有本地补丁。
- `.gitattributes` 禁止转换这些上游文件的换行符，确保检出后校验值仍然一致。
- 许可证为 [BSD-3-Clause](LICENSE.txt)，随附上游 [NOTICE](NOTICE.txt)。

仅引入核心头文件；上游 Gigi 工程、网页服务和示例资源不是 Metallic 的构建依赖。
更新时从新固定提交同步上述六个文件，更新清单并运行下述编译测试，避免在供应商文件中直接修改实现。

## 引用入口

| 文件 | 用途 |
| --- | --- |
| [../ShaderToHuman.slang](../ShaderToHuman.slang) | Gather 文本/2D 图形、3D 图形，以及线性颜色合成辅助函数 |
| [../ShaderToHumanScatter.slang](../ShaderToHumanScatter.slang) | 从指定 Shader invocation 向输出纹理写调试文本，需要调用方实现写入回调 |
| [../ShaderToHumanExample.slang](../ShaderToHumanExample.slang) | 同一套数字、十六进制 ID、2D 标记与 3D 球体绘制的 compute/fragment 示例 |
| [../ShaderToHumanScatterExample.slang](../ShaderToHumanScatterExample.slang) | 单 invocation、有边界裁剪的 Scatter 示例 |

以上入口采用 Slang/HLSL 路径，不包含 `s2h_glsl.hlsl` 的 GLSL 兼容宏。
库头文件不声明 GPU 资源；按需 include 和调用即可，现有 RenderGraph 不会自动增加调试绘制。

## 在现有 Shader 中绘制

例如从 `Shaders/Features/Lighting/` 下引用：

```hlsl
#include "../Debug/ShaderToHuman.slang"

// 在每个输出像素上执行；Compute 的整数像素坐标需加 0.5。
// Fragment 的 SV_Position.xy 已经是像素中心，不要再次加 0.5。
ContextGather ui;
s2h_init(ui, float2(dispatchId.xy) + 0.5f);
s2h_setCursor(ui, float2(8.0f, 8.0f));
s2h_printTxt(ui, _I, _D);
s2h_printSpace(ui, 1.0f);
s2h_printInt(ui, selectedMaterialId);
s2h_printLF(ui);
s2h_printFloat(ui, selectedRoughness);
linearColor = ShaderDebug::composite(linearColor, ui);
```

`selectedMaterialId` 和 `selectedRoughness` 由调用方提供。Gather 中每个输出像素执行相同的
绘制命令；查看选中像素的数据时，各 invocation 应读取同一份选中数据，不能只让选中像素执行绘制。

`ContextGather.dstColor` 是预乘 Alpha 的线性颜色。`ShaderDebug::composite` 保留线性输出，
交给 Metallic 已有曝光/FinalBlit 链处理，不要再调用上游的线性到 sRGB 转换。若希望文字亮度
不受场景曝光影响，在曝光后的线性颜色阶段合成，再进入最终输出。

3D 绘制使用 `Context3D`、`s2h_init(context, rayOrigin, normalizedRayDirection)` 和
`s2h_drawLineWS` / `s2h_drawSphereWS` 等接口。需要场景遮挡时，将 `context.depth` 设为当前
可见表面沿该射线的距离；不要直接传 Vulkan/Reversed-Z 深度值。3D 示例使用不透明颜色。

Scatter 需实现 `void onGfxForAllScatter(int2 pxPos, float4 color)`。回调负责坐标裁剪和写入，
必须选择一个 invocation 或保证不同 invocation 的写入区域不重叠。输出纹理先初始化，
相邻 dispatch 的写入要有适当的资源屏障；Scatter 不提供跨 invocation 的同步或命令缓存。

## 示例编译与验证

`ShaderToHumanExample` 提供：

- `shaderToHumanExampleMain`：compute，`numthreads(8, 8, 1)`；set 0 / binding 0 为可写 `float4` 纹理。
  按输出尺寸向上取整 dispatch，内置边界检查；320×192 可显示完整内容。
- `shaderToHumanExampleFragmentMain`：fragment，从 `SV_Position` 获取像素坐标并输出线性颜色。
- `ShaderToHumanScatterExample.shaderToHumanScatterExampleMain`：compute，单组 `(1, 1, 1)`；
  binding 0 为已初始化的输出纹理，在 `(8, 160)` 写文本。示例只让全局 invocation 0 执行。

运行时模块名分别为 `Features/Debug/ShaderToHumanExample` 和
`Features/Debug/ShaderToHumanScatterExample`，搜索路径仍为 `PROJECT_SOURCE_DIR "/Shaders"`。

```powershell
cmake --build cmake-build-debug-visual-studio --target MetallicRhiTests
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --gtest_filter=RhiResource.shader_to_human_shader_compile --rhi-validation
```

测试通过运行时 Slang API 编译三个入口，并检查上游 HLSL 被纳入缓存/热重载依赖。
更多接口与交互控件参见 [上游文档](https://electronicarts.github.io/ShaderToHuman/)。
