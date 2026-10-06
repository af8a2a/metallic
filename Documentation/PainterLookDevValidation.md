# Painter 整组场景切换验收

本机生成包位于 `build/MaterialValidation/PainterLookDev`，来源为已验证的
`build/MaterialValidation/PainterValidation`。生成资产、日志和图片均为本地输出。

在 LookDev 的 **RenderGraph → Graph Editor Settings → Painter Material Scene**
选择场景；每次切换同时替换网格、材质实例、贴图、相机、环境和比较图。
目录包含 M01–M08 的 uniform/textured、S01–S06 参数扫描、H01 两个版本，
共 24 项。修改尚未保存时，切换沿用保存/放弃/取消确认。

## 2026-10-05 验证

- MSVC Release：LookDev、MetallicRHITests、MetallicSceneTests 构建成功。
- MaterialAssets：18 项通过。
- Value / Closure / Slab 兼容性回归：13 项通过，含 GPU ABI、Value LookDev、
  Value → Closure → 场景执行及 Slab 参数/纹理/传输边界检查。
  日志：`build/painter-lookdev-regression.log`。
- `RHIRendering.painter_lookdev_scenes`：24 个完整场景连续加载，分别执行
  Reference PT 和 Deferred，共 48 个输出；检查 float32 有限值、非全黑、
  编译失败状态及错误材质棋盘，通过。检查图使用 256²、4 spp，未做收敛比较。
  本次线性输出最大值约 100.87，覆盖大于 1 的 HDR 响应；30 张预览贴图的
  最大 UNORM 量化误差为 0.00196073。
- GPU 证据：`rhi-test-output/PainterScenes.json` 及对应 PNG；最终运行日志
  `build/painter-lookdev-gpu-final.log`。较早的 `painter-lookdev-gpu.log` 不是验收结果。
- 编辑器 CTest：`MetallicLookDevMaterialInspectorSmoke`（44.11 秒）与
  `MetallicLookDevPainterSceneSwitchSmoke`（127.23 秒）均通过，无 Vulkan
  validation 错误。后者从 M01 uniform 切换到 M03 textured、M08 textured、
  H01 textured，再回到 M01，实际渲染并检查未保存修改保护。
  日志：`build/painter-lookdev-editor-tests.log`、
  `build-scheduling-release/tests/lookdev-painter-switch/editor.log`。
  测试调用下拉框共用加载入口；computer-use 工具初始化失败，未执行鼠标点击验收。

## 验证边界

以上检查确认导入和执行路径，不代表已经证明 Painter 与 Metallic 像素一致、
能量守恒或采样收敛。当前 resident 预览贴图为 RGBA8，原始 32f EXR 保留；
量化误差见 `ImportReceipt.json`。玻璃绝对吸收尺度尚未与 Painter Iray 标定，
Deferred 的体积表现也不能替代 PT 验收。MASK 阴影及时间稳定性需要专门对照。

Vulkan 检查按进程指定 SDK 1.4.350.0 layer 路径及空 implicit-layer 目录，
绕开本机失效的外部 layer 注册；没有修改系统设置。该 SDK 不支持本次
KHR OMM 验证路径，测试使用 shader alpha traversal。

使用和重新生成步骤见 [LookDev 文档](OpenPbrLookDev.md)。
