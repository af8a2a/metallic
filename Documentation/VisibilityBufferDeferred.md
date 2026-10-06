# Visibility Buffer 实时延迟着色

`VisibilityBufferDeferredPass` 是光栅化 Visibility Buffer 的 Compute resolve。
主可见性由 `VisibilityBufferPass` 生成，Deferred 重建表面、求值 OpenPBR 材质，
使用 ClusterLightGrid 直接光、SH 漫反射和预过滤 HDRI 镜面光照，输出未曝光的
Scene Linear Rec.709/D65 HDR。ACES、曝光和显示编码仍在后处理链路中执行。

```text
VBuffer.visibility / depth / rasterInfo / domain
                 ↓
             Deferred ← shadow / shadowParameters（可选）
                 ↓
          AutoExposure → FinalBlit ← ColorGrading.lut
```

## LookDev 单路径观察

在 Render Graph Editor 的 Settings 中，选择 `lookdev-vbuffer` 或任一 Painter / Studio
PT/Deferred 对比场景，然后使用 **LookDev Render Path**：

- **Comparison**：Reference + VBuffer + Deferred，通过 Slider 对比。
- **Path Trace Only**：仅 Reference，直接连接原来的曝光与显示链路。
- **Deferred Only**：仅 VBuffer + Deferred，直接连接原来的曝光与显示链路。

Only 模式会从活动图中移除另一分支和 Slider；不是把分割线移到边缘。
切换保留当前场景、未保存的材质编辑、视口相机和现有 pass 参数，重新编译活动图并重置累积历史。
选择新的兼容材质场景时沿用模式。此选项只适用于 PT/Deferred 对比图；OpenPBR/Standard
双 BSDF 对比图、Fiber 专用图不会被自动改写。

```powershell
.\build-scheduling-release\Source\LookDev.exe --render-path pathtrace
.\build-scheduling-release\Source\LookDev.exe --render-path deferred
.\build-scheduling-release\Source\LookDev.exe --sample studio-white-M05_Fuzz-textured --render-path deferred
```

显式指定 `--render-path` 且未指定 sample 时，默认采用 `lookdev-vbuffer`。
可选值为 `comparison`、`pathtrace`、`deferred`；不指定参数保留原来的启动行为。
Graph Save 保存的是当前活动图；需要保留原始对比资产时，请先更换保存路径。
重新加载保存的单路径图可直接使用，但该文件不含已移除的分支；恢复对比应重新选择原始 sample。

验证入口：RHI `lookdev_render_path_contract` / `lookdev_render_paths`，以及 LookDev
`--smoke-test` 配合 `METALLIC_SMOKE_TEST_LOOKDEV_PATHS=1`。
RHI 渲染测试默认采用 shaderball，可用 `METALLIC_LOOKDEV_PATH_SAMPLE` 指定本地 Painter / Studio sample；
检查 32 帧执行记录、线性 HDR 有限性及与对比分支的逐字节一致性，输出独立图和显示 PNG。

## 光栅路径边界

- Deferred shader 不执行 ray query，不使用 TLAS，也不包含路径积分器。
  删除了主命中续追、随机环境光遮挡查询、逐灯阴影查询、体积吸收和多次散射续追。
  这不等于全局场景加载已完全免除 RTAS 分配：当前 `SceneResourceManager` 在支持光追的
  resident 设备上仍会预建完整场景资源；Deferred 本身不再声明或绑定该资源。
- 阴影仅消费显式连接的 `shadow` / `shadowParameters`。未连接时按无遮挡处理，
  不再隐式启动光追阴影。输入使用 SIGMA 的 sqrt(visibility) 编码，
  只影响参数指定的稳定光源槽位；其他灯光无遮挡。
  上游可选的 `RayTracedShadowPass` 仍是独立光追功能；连接它的整张图仍属于混合渲染。
- 透射材质保留预过滤环境折射近似，不显示玻璃背后的场景几何，不计算内部界面、
  Beer–Lambert 吸收、焦散或 SSS。BLEND 排序合成不在此 Pass 的职责中。
- 参考 `ScenePathTracePass` 继续提供完整路径追踪。原有 LookDev Slider 可用于比较
  光栅近似和参考结果，两者的间接光、遮挡、透射及轮廓像素积分会有差异。
- 实时路径不做渐进光照累积。抗锯齿与时域重建交给下游，
  `exportUpscalerGuides` 可输出 motion vectors 和 device depth。

旧图中 `lightingMode`、`environmentSamples`、`supplementaryPathTracing`、
`transmissionSamples`、`transmissionDepth`、`accumulate` 不再控制 Deferred 行为，
Inspector 也不再显示它们。仓库内图资产已迁移。外部旧图应删除连到
`Deferred.accelerationStructure` 的边；独立 Shadows 的 TLAS 边保留。

## Shader 分层与精度

- `VisibilityBufferDeferred.slang`：resident / StreamAsset 三角形重建、整屏与分箱调度、
  上采样 guides 和 HDR 输出。
- `VisibilityBufferLighting.slang`：实时 OpenPBR 直接光、外部阴影与 IBL。
- `SceneSurface.slang`：共享相机、纹理过滤、表面和 TBN 辅助函数。
- `OpenPBRSurface.slang`：共享 OpenPBR LUT 适配、材质求值与调试视图。
  两个表面文件不包含 ray-query 积分器；参考路径显式包含自己的追踪实现。

`halfPrecision` 默认 true。使用 native FP16 保存和计算有界的金属度、透射权重、
IBL 镜面能量权重、漫反射与透射颜色系数。位置、深度、重心坐标、UV/LOD、TBN、
IOR、折射、GGX 分母和直接光 BSDF 保留 FP32；HDR 采样、乘光源和累加也保留 FP32。
因此高亮不会因为中间 HDR radiance 被压到 FP16 而溢出。
关闭 `halfPrecision` 使用相同算法的 FP32 权重，供精度回归对照。

## 分箱、资源与场景契约

`materialBinning` 默认 true，8×4 tile / wave32 分类为 Background、Dielectric、
Conductor、Opaque、Transmission 五类，以互斥 lane mask 执行 indirect dispatch。
所有类别使用同一光栅估计器；透射类不再有追踪特例。
分类要求 native subgroupSize=32 和 compute ballot/arithmetic；不支持时可关闭分箱，
使用整屏 8×8 调度。Custom Value 程序在 resident 场景使用一般内核。

`visibility`、`depth`、`rasterInfo` 必须来自同一个 producer，Deferred 继承其场景绑定。
位置和 TBN 仍保持 authored/world-space 方向，法线贴图求值后才对最终着色法线 face-forward。
StreamAsset 的光栅材质路径不需要启用 Cluster RTAS。

| 接口 / 设置 | 含义 |
| --- | --- |
| `visibility` | R32Uint 光栅 ID |
| `depth` | 同一 producer 的 D32 深度 |
| `rasterInfo` | 相机、分辨率、场景版本与 GPUScene view 元数据 |
| `domain` | 可选的位移重心坐标与几何法线 |
| `shadow` / `shadowParameters` | 必须成对连接，同分辨率阴影与对应光源参数 |
| `color` | Scene Linear RGBA32F；输出 upscaler guides 时为 RGBA16F |
| `halfPrecision` | 默认开启的 FP16 材质权重，可切换 FP32 对照 |
| `materialBinning` | 默认开启的 wave32 分类调度 |
| `debugDisableShadows` | 忽略外部阴影输入 |
| `debugDisableTransmission` | 关闭环境折射分量 |
| `debugView` | final、baseColor、geometryNormal、shadingNormal、tangent、material |

## 验证入口

`MetallicShaderRequestsTests` 覆盖 request 规范化和 24 个生产 Deferred 变体：
resident / stream、guides 开关、整屏 / 5 个材质类别。
编译测试检查 SPIR-V 不含 RayQuery/RTAS 类型或 ray-tracing capability，
非背景变体含原生 16-bit 浮点类型。

`visibility_buffer_abeautiful_game_transmission` 覆盖 15 种真实材质的
FP16/FP32 显示误差（各通道最多 2/255）、flat/binned 一致性、环境透射切换、
旧续追参数无效、volume 参数无效、材质刷新、guide 导出和奇数分辨率。
`stream_material_transmission` 验证无 Cluster RTAS 的 StreamAsset 材质输出。
`visibility_buffer_deferred_openpbr` 保留无阴影直接光及几何属性与 ray-primary 参考的对照。

性能 runner 当前只支持 WorkControl 候选；增加 FP16 类型本身不构成提速证据。
GPU 测试的 preview wall time 包含 CPU 提交与 readback，不能当作 GPU shader 时间。
完整 Zorah 场景的帧率、长期稳定性和显存驻留需要单独测量。

2026-10-03，MSVC Release 实测：8 项 shader 请求/编译测试通过；9 项 Vulkan GPU 回归通过，
包括棋盘材质、resident/stream、显式阴影、场景交接及 Custom Value 材质。
棋盘 385×257 的 FP16/FP32 对照最大通道误差 2/255、平均 0.002698/255，
无新增 Vulkan VUID；场景绑定测试包含预期的故障注入日志。
日志与图像保存在 `.cache/deferred-raster-shaders-final.log`、`.cache/deferred-raster-gpu/`
和 `.cache/deferred-raster-regression/`，上述运行开启 validation，不作为性能 A/B 证据。
实际 runtime 缓存中的非背景变体含 10 条 FP16 算术指令，解析结果及 SPIR-V 哈希见
`.cache/deferred-raster-precision.json`。Metallic、LookDev、MetallicGPUDrivenSample 构建通过；
`LookDev --sample lookdev-vbuffer --smoke-test` 完成提交和 present，启动预热 168 项、0 失败。
