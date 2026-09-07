# LookDev 着色路径比较

独立 `LookDev` 程序增加 **LookDev / Shading Comparison** 示例：

```powershell
cmake --build cmake-build-debug-visual-studio --target LookDev --parallel 8
.\cmake-build-debug-visual-studio\Source\LookDev.exe --sample lookdev-shading-compare
```

也可在 **Built-in Sample** 中选择该示例。无参数启动仍使用原来的
OpenPBR 参考场景。通过 `--scene <path>` 测试其他材质场景时，两路路径会同时更新。

## 操作

- 左键拖动视口中的分界线，连续显示两路结果的对应区域。
- 视口工具栏的 **Compare** 调整分界位置，**Top / bottom** 切换上下比较，
  **Swap A/B** 交换来源。位置 0、1 分别显示完整 B、A；交换后相反。
- 分界线附近仍可使用 Alt + 左键旋转、右键／中键相机操作。
  相机面板及节点的运行时 Camera 控件会同步同组相机。
- 查看单独的 `OpenPBR.color` 或 `Standard.color` 时不显示分界线。
  分界线和来源标签属于编辑器叠加层，不写入渲染结果。

## 默认比较条件

```text
OpenPBR.color  ──→ Slider.sourceA ─┐
                                 ├─→ Slider.color → AutoExposure → FinalBlit
Standard.color ──→ Slider.sourceB ─┘
```

两路均使用 `ScenePathTracePass`，BSDF 分别为 `openpbr` 和 `standard`。
它们读取同一个 MaterialX shaderball glTF、同一个运行时场景及材质、HDRI、太阳光，
每帧 4 spp、最大深度 12，输出未曝光的线性 HDR。两路相机设置相同，
`cameraSyncGroup: "LookDevComparison"` 将编辑器的相机修改同步到组内其他场景 Pass。
场景、参考材质和照明来源见 [OpenPbrLookDev.md](OpenPbrLookDev.md)。

合成后只经过一次曝光和显示转换，默认固定 EV100 0、补偿 0、None (sRGB)。
调整分界、方向或交换来源不会重编译渲染图或清除两路累积。
相机移动仍会清除累积，以避免旧视角残留。

两种 BSDF 的材质模型不同，图像差异也包含模型本身的差别。渐进渲染初期还包含采样噪声；
比较细微差异时应等待收敛。该样例用于并排检查，不将 Standard 路径作为 OpenPBR 真值。

## Pass 接口与扩展

`SliderDebugPass` 是不依赖光追的 Compute Pass，通用 RenderGraph 也可添加。

| 字段／属性 | 约定 |
| --- | --- |
| `sourceA`, `sourceB` | 必需的 sampled texture 输入，必须与输出分辨率一致、采用相同颜色空间和曝光尺度 |
| `color` | RGBA32F，原样保留被选像素的 RGB 和 alpha，包括 HDR 和负值 |
| `splitPosition` | 浮点数，默认 0.5，夹取到 [0,1]；非有限值回退 0.5 |
| `orientation` | `vertical`（默认，左右）或 `horizontal`（上下） |
| `swapSides` | 布尔值，默认 false |

支持输入格式：RGBA8Unorm、BGRA8Unorm、RGBA16Sfloat、RGBA32Sfloat、
RG32Sfloat、R32Sfloat、B10G11R11UfloatPack32。整数 ID、深度、sRGB 格式需先显式转换。
缺失输入或尺寸不一致会使图编译失败；不支持的格式在执行时返回 InvalidArgument。
按像素中心决定分界归属，不缩放、插值、混合或压缩任一侧图像。

后续可将两种着色路径的同分辨率线性 HDR 输出分别接入 A/B，保留共同的
`AutoExposure → FinalBlit` 链路，并为两路场景节点设置相同的非空 `cameraSyncGroup`。
该相机组是编辑器约定；直接执行图时由调用者提供一致的相机属性。
若启用自动曝光，测光会读取分界后的混合画面，拖动分界可能改变目标曝光；
需要稳定比较时应保持手动曝光。

图资产 `Pipelines/Samples/lookdev_shading_compare.metallic_graph.json` 与参考场景一起由
`Tools/BuildOpenPbrLookDev.py` 生成。

## 验证

```powershell
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --filter slider_debug --rhi-validation --output-dir rhi-test-output/slider-debug
ctest --test-dir cmake-build-debug-visual-studio -C Debug -R MetallicLookDevSliderSmoke --output-on-failure
```

GPU 测试逐像素读取 float4，覆盖方向、交换、端点、奇数尺寸、1×1、负值、HDR、alpha、
缺失／错误输入和运行时更新。实际场景测试生成 768×768、每路 1024 spp 的
`LookDevShadingComparison.png`。编辑器测试检查真实 ImGui 拖拽、Alt 相机手势、
相机同步以及调整 Slider 时历史资源不失效。
