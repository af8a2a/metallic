# LookDev 材质 Playground

独立 CMake 目标 **LookDev** 生成 `LookDev.exe`，无参数启动时默认加载
**LookDev / OpenPBR Default** 的场景、相机和渲染图。程序复用编辑器外壳，
可通过 RenderGraph 面板的 **Built-in Sample** 切换示例，或从命令行
选择其他场景，作为后续测试多种材质的 playground。

```powershell
cmake --build build --target LookDev --config Debug
.\build\Source\Debug\LookDev.exe
```

本地 Ninja 构建目录使用：

```powershell
cmake --build cmake-build-debug-visual-studio --target LookDev --parallel 8
.\cmake-build-debug-visual-studio\Source\LookDev.exe
```

`--list-samples` 列出注册的示例 ID，`--sample <id>` 选择启动示例，
`--scene <path>` 覆盖所选示例的场景。还支持 `--smoke-test`、
`--debug-control` 和 `--wait-for-graphics-debugger`。

```powershell
.\cmake-build-debug-visual-studio\Source\LookDev.exe --list-samples
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample openpbr-lookdev
.\cmake-build-debug-visual-studio\Source\LookDev.exe --scene Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf
```

另提供 [SliderDebugPass 着色路径比较](SliderDebugPass.md)：
`LookDev.exe --sample lookdev-shading-compare`，在同一材质上滑动比较 OpenPBR 与 Standard BSDF。

[VBuffer 延迟渲染比较](VisibilityBufferDeferred.md) 使用
`LookDev.exe --sample lookdev-vbuffer`，在相同 OpenPBR BSDF 下比较 GPUDriven 延迟着色与路径追踪。

默认示例
使用 `PathTrace → AutoExposure → FinalBlit`，明确选择 OpenPBR BSDF，
每帧 4 spp、最大深度 12，在线性 HDR 空间渐进累积，不启用降噪器。

参考为用户指定的
[MaterialX Web Viewer / Open Pbr Default](https://academysoftwarefoundation.github.io/MaterialX/?file=Materials/Examples/OpenPbr/open_pbr_default.mtlx)。
这个页面使用 MaterialX 生成的 GLSL 和预积分环境照明，是实时预览；
本场景使用 Metallic 的 OpenPBR 离线路径追踪。它提供相同输入条件下的
对照基线，不把网页截图当作离线路径追踪的逐像素真值。

## 场景与显示条件

| 项目 | 值 |
| --- | --- |
| 场景 | `Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf`，自动加载同名 scene sidecar |
| 渲染图 | `Pipelines/Samples/openpbr_lookdev.metallic_graph.json` |
| 几何 | MaterialX `shaderball.glb` 的两个原始网格，88,264 个三角形 |
| Base color | 线性 Rec.709 `(0.8, 0.8, 0.8)`，不是 sRGB 贴图值 |
| Base weight / metalness / diffuse roughness | `1 / 0 / 0` |
| Specular weight / color / roughness / IOR | `1 / (1,1,1) / 0.3 / 1.5` |
| Coat / fuzz / transmission / subsurface / emission | 权重为 0 |
| HDRI | `san_giuseppe_bridge_split.hdr`，强度 1，旋转 0°，背景可见 |
| 太阳光 | 方向光，RGB `(1, 0.894474, 0.567234)`，强度 `2.52776` |
| 光线传播方向 | `(-0.711269, -0.479014, -0.514434)` |
| 相机 | eye `(0, 1.047861, 3.644735)`，target `(-0.054851, 1.047861, -0.064490)` |
| 投影 | 垂直 FOV 60°，Y up，near 0.05，far 100 |
| 曝光 | 自动曝光关闭，EV100 0，补偿 0，输入/输出 multiplier 1 |
| 显示 | AutoExposure 的 **None (sRGB)**，不压缩高光，超过显示白的值裁切 |

太阳已从 HDRI 中拆出，必须连同 `san_giuseppe_bridge_split.mtlx` 的方向光
使用。保留源文件的相对标定值，不单独把太阳改成日光场景常用的高照度。
MaterialX Viewer 对方向光应用 Y 轴 +90° 旋转；其环境投影
`atan2(x, -z)` 经同一旋转后等价于 Metallic 的 `atan2(z, x)`，因此
Metallic 的环境旋转应为 **0°**，不能再加 90°。

对比截图使用 **768 × 768**，相机按参考模型包围盒拟合。
编辑器任意长宽比都能查看场景，但比较轮廓时应固定长宽比。
编辑器会保留已手动修改的全局环境覆盖；建议在新启动的编辑器中加载
此示例，或者按上表恢复 HDRI、强度和旋转后再比较。

新增的 `toneCurve: "none"` 仍应用手动/自动曝光，再使用分段 sRGB
传递函数。它不经过 Reinhard、Exponential 或 ACES。原有两种曲线不变。
`PathTrace.color` 保留未曝光的 RGBA32F 输出，供 HDR 数值检查。

## 复现与验证

资源已随场景提供，正常加载无需联网。重新生成资源只需 Python 标准库：

```powershell
python Tools/BuildOpenPbrLookDev.py
```

脚本从 `Reference.json` 记录的固定 MaterialX 提交下载，校验源资源
SHA256，从原始 MaterialX 文档读取材质和方向光，并重建相机、glTF、
scene sidecar 和渲染图。也可用 `--source-dir <已下载的源文件目录>`
离线生成。脚本会覆盖此示例的生成文件。

```powershell
cmake --build cmake-build-debug-visual-studio --target LookDev MetallicRhiTests --parallel 8
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter openpbr_lookdev --rhi-validation --output-dir rhi-test-output/openpbr-lookdev
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter auto_exposure --rhi-validation
```

LookDev 测试绑定与编辑器相同的 SceneDocument，输出
`rhi-test-output/openpbr-lookdev/OpenPbrDefault.png`（768²，1024 spp）。
测试检查参考配置加载、OpenPBR 选择、主体曝光，并关闭所有光源确认
输出为黑色。预览必须绑定该场景，不能把 scene sidecar 中的灯光再作为
独立 world 灯光追加一次。sRGB 测试覆盖暗部线性段、18% 灰、0.8、白色、
高光裁切及手动曝光补偿。生成图像保留在忽略的测试输出目录中。

独立程序加载冒烟检查：

```powershell
.\cmake-build-debug-visual-studio\Source\LookDev.exe --smoke-test
```

## 扩展材质测试场景

应用入口位于 `Source/LookDev/LookDevMain.cpp`。仅更换材质/模型时，
提供新的 glTF 及其 scene sidecar，使用 `--scene` 加载；要复用参考灯光，
在新 sidecar 中保留此示例的环境与曝光设置。直接 `.mtlx` 导入尚不支持。
需要独立渲染参数或灯光配置的预设，可以在 `RenderSample.cpp` 注册新的
`RenderSample`（建议归入 `LookDev` 分类）并添加对应的场景和渲染图。
它会自动出现在 `--list-samples` 和编辑器选择器中，无需再创建可执行目标。

## 对照结果与边界

已按 768² 抓取参考页面并渲染 1024 spp 对照图，模型轮廓、HDRI 背景、
受光方向和高光位置一致。Metallic 会计算内凹区域与底座之间的遮挡和
多次反弹；参考页面使用 irradiance 贴图及 16 个过滤环境高光采样，
不具有相同的可见性与间接照明求解。底座接触区域在路径追踪中明显更暗，
不能通过提高曝光来抹平这个差异。

对截图反解 sRGB 后，右侧受光面区域 `(480:515, 345:410)` 的平均线性
亮度为参考 `0.7083` / Metallic `0.6843`，主体前侧区域
`(330:420, 350:450)` 为 `0.4089` / `0.3709`。这些是视觉检查的局部
读数，不是跨渲染器一致性阈值，也不是独立离线渲染器的精度认证。
后续比较其他离线渲染器时，应复用这里的网格、材质、相机和光照，并输出
相同色域的线性 EXR；另行记录积分器、最大深度与降噪设置。

对齐还修正了两个 OpenPBR 着色路径中 glTF 的漫反射粗糙度映射：
glTF 的 Lambert 漫反射使用 `base_diffuse_roughness = 0`，高光粗糙度
仍来自 glTF roughness。该修正也会影响其他使用 glTF 的 OpenPBR 场景。

源资源和许可证见 [资源说明](../Asset/LookDev/OpenPbrDefault/README.md)。
