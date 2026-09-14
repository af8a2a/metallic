# scRGB HDR 输出

Windows 中启用显示器 HDR 后，Metallic 主窗口默认请求 scRGB HDR。`Display Output` 菜单显示实际输出模式，可关闭 HDR、调整纸白亮度、显示峰值和额外 HDR 曝光。默认纸白跟随 Windows 的 SDR 白电平，峰值默认 1000 nits，需要根据显示器调整；这不是自动检测的峰值。

在 Samples 中选择 `HDR / scRGB Calibration` 可绕过场景渲染。上半部依次为 80、203、400、1000 nit 色块，下半部为 0–1000 nit 的灰阶和 RGB 渐变。校准图中的数值是绝对亮度，不受曝光、纸白或峰值映射影响。SDR 模式会显示经过裁切的预览。菜单中的 Calibration pattern 可临时替换当前 FinalBlit 的画面；独立校准样例才会完全省去上游场景工作。

## 渲染路径

```text
Scene-linear HDR → AutoExposure (曝光后的 FP16)
                 → FinalBlit (显示映射 / nits ÷ 80)
                 → FP16 ImGui 合成 → scRGB 交换链
```

- RHI 的 `SwapchainDesc.outputMode` 支持 `Sdr` / `HdrScRgb`，`Swapchain::outputMode()` 返回协商结果。HDR 请求只接受 `R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR_EXT`；默认不支持时回退到 SDR，`allowSdrFallback = false` 则返回 Unsupported。
- 可用时启用 instance extension `VK_EXT_swapchain_colorspace`。支持情况以 surface 报告的完整格式/色彩空间组合为准。[Vulkan 扩展说明](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_swapchain_colorspace.html)
- Windows scRGB 的 1.0 对应 80 nits，所以四个校准色块的 RGB 值分别为 1.0、2.5375、5.0、12.5。无需 PQ 编码或 Rec.2020 转换。[Microsoft Advanced Color](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range)
- `RenderGraphCompileOptions.displayOutput` 传入实际输出模式和显示参数。参数变化、shader 热重载及窗口 HDR/display-change 事件均保留正确的显示上下文。
- `AutoExposure.color` 在 SDR 下保留既有 RGBA8 色调映射行为；HDR 下输出带 `ExposedLinear` 标记的 FP16。FinalBlit 将曝光后的 1.0 映射到纸白，超过纸白的亮部平滑接近峰值，按 RGB 最大分量统一缩放以保持通道比例。
- FinalBlit 的 Automatic 输入模式读取资源的颜色语义，不通过 FP16/FP32 格式猜测颜色空间。旧节点默认视为 sRGB 显示色；自定义裸线性输入可选择 `Exposed scene-linear`，绝对 scRGB 输入选择 `scRGB (absolute)`。sRGB 纹理的硬件解码也会被计入，避免重复转换。
- ImGui 普通 UI 从 sRGB 转为线性并按纸白缩放，FinalBlit 的 scRGB 图像使用独立兼容管线保留超过 1.0 的值。窗口重建会更新 Vulkan 附件格式；不改动 vendored ImGui。

## 当前范围

主窗口支持 HDR。ImGui 拖出的独立窗口仍使用 SDR 交换链，其中 scRGB 视口按纸白转换并裁切高光。实验性 DLSS-NR 目前要求 SDR RGBA8，因此 HDR 下按 `fallbackToInput` 设置透传 FP16；关闭回退则明确返回不支持。DLSS-SR/RR、场景光照与材质不需要修改。

HDR10/PQ、HDR metadata、自动显示器峰值检测与独立窗口 HDR 尚未实现。

## 验证

构建 `Metallic` 和 `MetallicRhiTests`，运行：

```powershell
MetallicRhiTests.exe --gtest_filter="*hdr_display_output*:*hdr_surface_format*:*hdr_editor_imgui*:*final_blit*:*auto_exposure*"
$env:METALLIC_SMOKE_TEST_SAMPLE = 'hdr-calibration'
$env:METALLIC_DEBUG_VALIDATION = '1'
Metallic.exe --smoke-test --debug-control
```

GPU 回读测试覆盖精确校准值、渐变、曝光、高光范围、纸白更新、shader 热重载、HDR/SDR 往返切换，以及 ImGui FP16 合成中的 UI 白电平、视口高光和回调后的状态恢复。离屏测试无需 HDR 显示器；显示器的实际亮度与 Windows HDR 开关/跨屏切换仍需 HDR 硬件验收。
