# White Studio 02 材质 LookDev

运行 LookDev 后，在 **RenderGraph → Graph Editor Settings → Studio LookDev Scene**
切换。也可以用：

```powershell
.\build-scheduling-release\Source\LookDev.exe --sample studio-white-overview --skip-shader-warmup
```

本地包在 `build/MaterialValidation/WhiteStudio02`，不会加入默认构建，也不会覆盖
原来的 Painter / NeutralStudio 包。目录在程序启动时读取，生成后需要重启。

## 场景

- **Material Overview**：18% 灰、80% 白、90% 镜面、粗糙金属、铜色金属、
  红色涂层、各向异性、fuzz、光滑电介质，共九球，带名称标牌。
- **Photographic Chart (Unlit)**：左侧为提供的实拍色卡，右侧为下半部分
  ColorChecker 24 色块中心 ROI 的 RGB 中位数，使用不受光照影响的材质。
  此场景用于输入/显示链路检查，不参与反射率或灯光标定。
- 原有 24 组 Painter 用例的 White Studio 版本：复用网格与材质/贴图，
  独立保存场景、相机与图设置。玻璃路径的原有限制仍适用。

Reference / Deferred 共享 HDRI、相机、固定 EV 0、强度 1、旋转 0°，
没有额外点光源；默认 Slider 的 Split Position 为 1（全幅渐进 PT），改为
0.5 可左右对照。最终输出使用现有 ACES 2 显示链路。当前 Deferred 不做
渐进累积，coat/fuzz 的单样本环境项有明显噪声，不能把它当成收敛参考。
Studio 选项会恢复预设环境设置；场景修改未保存时仍会提示保存/放弃/取消。

## 来源和色彩

[White Studio 02](https://polyhaven.com/a/white_studio_02) 由 Grzegorz Wronkowski
发布，Poly Haven 标注 CC0。使用用户提供的 4096×2048 EXR 与配套 ZIP，
SHA256 和转换误差保存在 `ImportReceipt.json`。解压时仅读取三个指定成员，
保留 EXR / TIFF / NEF；不执行附件内容。

HDRI EXR 没有 chromaticities 字段，按常见线性 Rec.709/D65 约定导入（属于导入假设）；
色卡 EXR 的 chromaticities 明确对应 Rec.709/D65。当前引擎环境解码不支持
EXR，因此生成 RGBE HDR，显式标记 `lin_rec709`，再由引擎转换至 ACEScg。
没有归一化、LDR 截断或 gamma 烘焙。原始最大通道值 44.0622，RGBE 最大值
44.0；相对 RMS 转换误差 0.3261%，最大绝对误差 0.249981。

ZIP 中是**已受现场光照与相机处理影响的照片**，不是独立反射率表。
`ChartMeasurements.json` 保存原始线性 RGB 中位数、标准差与 72×72 ROI。
`ChartROIs.png` 可检查取样位置。为了让参考照片可读，单独应用约 10.3829
的显示系数，将照片 Neutral 5 的 Y 映射到 0.18；不应用白平衡矩阵，也不将
该系数应用于 HDRI。这个约定不等于认定实物该色块的反射率为 18%。

照片面板使用 sRGB PNG8 预览，独立色块使用浮点常量；原始 EXR 数值保留。
没有将照片二次当作受光照底色，也没有根据未知相机曝光反推物理照度。
这套场景不构成光谱、Delta-E、绝对光度或 Painter 物理一致性认证。

## 复现

Python 依赖 numpy、OpenEXR、Pillow。已有 Catalog 的输出目录不会被覆盖：

```powershell
python Tools/MaterialValidation/BuildStudioLookDev.py --hdri D:/white_studio_02_4k.exr --chart-zip D:/white_studio_02.zip
cmake --build build-scheduling-release --target LookDev MetallicRHITests
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-bindless --rhi-validation --filter studio_lookdev --output-dir rhi-test-output/white-studio
ctest --test-dir build-scheduling-release -R '^MetallicLookDevStudioSceneSwitchSmoke$' --output-on-failure
```

测试读取所有场景并检查资源解析、显式环境色彩标签和手动曝光。
对总览、色卡、M03 textured 执行 PT / Deferred / 显示输出，确认 4K HDRI
已就绪、线性输出有限，并以 EXR 测量值检查两条着色路径的 24 个无光照色块。
截图、线性 float32 及结果写入测试输出目录，不进入源码控制。

2026-10-05 GPU 验收：26 个场景加载通过，三组场景的 PT / Deferred / 显示输出
共 9 张图通过有限值与环境就绪检查。PT 和 Deferred 的 24 色块最大绝对误差
均为 1.1921e-7（仅检查输入辐射值传递，不是与实物的色差）。Vulkan validation
开启，无 validation 错误。最终日志为 `build/white-studio-gpu-final.log`，
报告为 `rhi-test-output/white-studio/StudioLookDev.json`。

编辑器验收：Studio 场景连续切换、预设环境恢复、返回总览、未保存修改保护
通过（47.93 秒）；原 Painter 场景切换回归通过（8.77 秒）。日志为
`build/white-studio-editor.log` 和 `build-scheduling-release/tests/lookdev-studio-switch/editor.log`。
测试调用 UI 共用入口并执行真实编辑器渲染；未做鼠标点击自动化。
