# Sponza VSM Sample

`MetallicVSMSample` 使用仓库内的 Sponza，展示同一视图内从近处地砖、柱脚到远处柱廊和端墙的 Virtual Shadow Map 阴影。

## 构建与启动

在仓库根目录的 x64 Visual Studio developer shell 中，复用已配置的构建目录：

```powershell
cmake --build build-release --target MetallicVSMSample --parallel 8
.\build-release\Source\MetallicVSMSample.exe
```

也可以在通用编辑器的内置 Sample 列表中选择 **Virtual Shadow Maps / Sponza**，ID 为 `virtual-shadow-map-sponza`。首次运行按需编译当前图的 shader/PSO。

## 场景与构图

- 场景：[SponzaVSM.metallic_scene.json](../Asset/Sponza/SponzaVSM.metallic_scene.json)，引用原始 `glTF/Sponza.gltf`，保留原始模型和材质。
- 相机：[SponzaVSMCamera.gltf](../Asset/Sponza/SponzaVSMCamera.gltf)。这是独立的相机源，通过组合场景加载，使编辑器加载场景时也能选择展示相机。
- 管线：[virtual_shadow_map_sponza.metallic_graph.json](../Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json)，保存相同的视图预设。

相机位于 `(8.7, 1.65, 0.15)`，朝向 `(-8, 1.3, -0.3)`，垂直视场角 60°。低机位沿庭院 X 轴长向观察，轻微俯视以保留前景地面，同时让重复的柱廊向远端收拢。

太阳的光线传播方向为 `(-0.30, -0.94, -0.16)`，即光源约位于 70° 仰角。较高的太阳可以照进狭窄庭院，沿 X/Z 的分量让建筑投影斜穿地面。HDRI 强度为 0.25，太阳照度为 6 lux；这组数值用于固定曝光下的阴影展示，不代表正午室外实测照度。月亮关闭，手动曝光 EV100 为 0，避免曝光适应干扰阴影比较。

## 渲染配置

图使用常规 GPUDriven mesh shader 光栅路径：

```text
VBuffer ── depth / rasterInfo ── Shadows (VirtualShadowMapLightsPass)
   └── visibility / depth / rasterInfo / domain ── Deferred
Shadows ── shadows / lights ── Deferred
Deferred → AutoExposure → FinalBlit ← ColorGrading
```

使用固定 LOD 0、硬件光栅和实例/HZB/meshlet 裁剪，关闭 Hybrid Software Rasterization、CLAS 和 cluster ray tracing。阴影走 VSM；图中没有 RT shadow、NRD 或 DLSS pass。输出为原生 1920×1080，关闭 temporal jitter。

VSM 使用 128×128 texel 页、每轴 128 个虚拟页、2048 个物理页预算、6 层 clipmap，开启页缓存和 receiver footprint。默认 SMRT 为 2 条射线 × 16 个搜索样本，并启用最多 32 帧的独立阴影历史；PCF 备选半径为 1 texel。第一层在灯光平面的半宽为 1 m，后续为 2/4/8/16/32 m；这些是灯光投影范围，并非相机深度分割距离。每层虚拟分辨率为 16384×16384，实际物理页按接收点需求分配。

观察时先看前景地面上的阴影边缘，再沿重复柱列看向中远景。沿长廊前后移动，可检查细节与缓存更新。选择 Shadows 节点可切换 Hard / PCF / SMRT，也可使用 Force Shadow Redraw 对照缓存。三种模式共享接收点 footprint 起始级，零 PCF 半径或零光源尺寸退化到 Hard 时保持该分辨率策略。关闭 Match Receiver Footprint 可诊断最细级需求，可能增加页预算压力。`Shadows.debug` 的红色强度表示中心查询的 clipmap，蓝色表示 coarse fallback；它不代表全部过滤样本实际使用的层级。

接收面足迹包含法线和光投影，掠射区域可能请求更细的页面。需求超过 2048 页时先保留较粗层覆盖，再分配细节页；丢弃部分细页不等于退回全场景粗图。斜面、掠射、凹凸曲面的独立质量门槛及当前实景结果见 [P4 接收面质量验收](VirtualShadowMapsP4.md#接收面质量门槛2026-10-29)。

低射线配置应先停留约 32 帧观察收敛，再平移相机检查边缘与显露区域。`SMRT History Frames = 1` 显示同一逐帧采样序列的原始结果；关闭 `SMRT Temporal Stabilization` 则回到固定种子的单帧模式。切换相机、光源几何或 caster 内容会清空相关历史，连续变化的 caster 目前不能持续累积。这里的时域解析直接处理线性阴影可见度，不要求主视图开启 jitter 或 TAA。

搜索默认保留 10 m 上限，`SMRT View Distance Scale (0 = fixed) = 0` 关闭视距比例限制。设为正数后，实际搜索长度为 `min(10 m, 相机距离 × 比例)`；它会限制半影展开，过小会使远处半影变窄。局部光还受 `SMRT Local Max Angle (rad)` 限制，默认 0.03 rad。具体合同、质量和生命周期验收见 [P4 低射线时域与搜索范围](VirtualShadowMapsP4.md#低射线时域与搜索范围2026-10-29)。

## 验证方式

```powershell
$env:METALLIC_SMOKE_TEST_FRAMES = '128'
.\build-release\Source\MetallicVSMSample.exe --smoke-test --debug-control
Remove-Item Env:METALLIC_SMOKE_TEST_FRAMES
```

使用 `METALLIC_DEBUG_VALIDATION=1` 可开启 Vulkan validation；结合 `metallicctl` 捕获 `FinalBlit.color`、`Shadows.debug` 和 `Shadows.statistics`，可以检查实际输出和页面统计。`Shadows.shadows` 为 R8 格式，当前 debug capture 不支持直接读取。截图、捕获和日志保存在本地 `.cache/`，不进入源码。

2026-10-29 验证：Windows x64 / MSVC Release 构建通过，实际加载组合场景并检查画面。启用 Vulkan validation 连续观察至第 4509 次图执行，无 validation 事件；第 272 与 3902 次执行的最终输出逐字节相同。静态视图请求并分配 654 页，缓存命中 654 页，无页面分配失败、raster work overflow 或过滤粗级回退；debug 图包含 3 个 clipmap 层级。内部渲染为 1920×1080，截图是编辑器视口缩放后的 653×282 输出。本次未覆盖移动相机巡检或 Force Shadow Redraw A/B。

地板误剔除回归：Sponza 的 0.008 缩放会使流式 normal-cone 的 cofactor 法线落入旧归一化函数的长度阈值，错误替换为 +Z，低机位下剔除上层地板并露出下层砖面。`transformStreamNormal` 现在先按最大分量缩放再归一化，保留有效方向和原有 winding 符号。

在启用测试的构建目录中运行：

```powershell
cmake --build build-release --target MetallicRHITests --parallel 8
.\build-release\tests\MetallicRHITests.exe --gtest_filter=*gpu_driven_cone_scale_invariance*:*gpu_driven_sponza_vsm_floor*:*gpu_driven_two_pass_occlusion*:*gpu_driven_temporal_occlusion_equivalence* --rhi-bindless --rhi-validation --output-dir .cache/sponza-floor/after
```

修复后上述 4 项 GPU 测试通过，无跳过：32 组法线探针覆盖普通和实际流式 helper；SponzaVSM 默认及两个录制低机位在预热后分别比较 4 帧，1920×1080 VBuffer 在 cone 开关两侧逐像素相同。完整 VSM 管线另渲染每个视点 8 帧并检查 960×540 最终截图，地板与阴影正常。移动／jitter／resize／透视及正交 HZB 回归覆盖 30 帧。本次回归未进行长时间显存压力测试。
