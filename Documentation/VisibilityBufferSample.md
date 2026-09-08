# Visibility Buffer Sample

`MetallicGPUDrivenSample` 无参数启动时默认加载 `gpu-driven-sample`，图中使用 `VisibilityBufferPass`，无需传入 `--visibility-buffer`。默认场景使用仓库附带的
`Asset/SuperSponza/NewSponza_Main_glTF_003.gltf`。入口命令：

```powershell
cmake-build-release-visual-studio\Source\MetallicGPUDrivenSample.exe --smoke-test
```

## 帧内数据流

```text
GPUScene → instance cull → Wave32 AS meshlet cull → MS + visibility PS
                                                        ↓
                                                 R32Uint ID + D32 depth
                                                        ↓
                          HZB → late cull/raster → final visibility/depth
                                                        ↓ (optional)
                                          ID / depth / coverage display
                                                        ↓
                                     GPUDriven.color → FinalBlit → Viewport
```

1. `GPUDrivenCulling.slang` 先进行实例视锥/HZB 剔除，使用上一帧相机与上一帧 HZB。
2. `VisibilityBuffer.slang` 的 amplification shader 每组处理 32 个 meshlet，执行 bucket、producer ownership、meshlet 视锥与 normal-cone 测试。支持时请求完整 Wave32，并以 wave prefix/count 压缩 payload；不支持时保留 subgroup/groupshared fallback。
3. Mesh shader 每组输出一个 meshlet 的共享顶点与索引（上限 128 vertices / 128 triangles），通过 per-primitive `SV_PrimitiveID` 输出可见性 ID。Opaque 变体仅输出 position；`VISIBILITY_BUFFER_ALPHA_MASKED=1` 变体另外输出 UV / material index 供 alpha test 使用。Fragment shader 只写入 `R32Uint` ID，不进行材质着色。
4. 深度附件使用 `D32Sfloat`。Opaque fragment 允许 early depth；masked fragment 先按 alpha cutoff discard，不能强制 early depth 写入。四个 raster bucket 分别覆盖 opaque/masked 与单面/双面材质，BLEND 暂不进入此路径。
5. Compute 将第一阶段深度归约成当前 HZB：Reversed-Z 用 min，普通 Z 用 max。第二阶段只重测之前的 HZB 遮挡候选，AS/MS 将恢复可见的 meshlet 补绘到同一 visibility/depth。
6. 完整深度再生成下一帧 HZB，直接保留最终 `GPUDriven.visibility`（R32Uint）和 `GPUDriven.depth`（D32Sfloat），不执行属性重建或材质着色。
7. 可选的 `VisibilityBufferComposite.slang` 直接读取原始 ID / depth，输出 `GPUDriven.color`（Rgba8Unorm）供视口调试显示，不改变原始输出。关闭可视化时只清空 color，不绘制全屏三角形。内置图将 `GPUDriven.color` 连接到 `FinalBlit.source`，样例的 `previewOutput` 为自动呈现输出 `FinalBlit.color`，JSON 的 `outputs` 数组为空。原始 visibility / depth 仍是节点输出，可供后续 Pass 使用；整数 ID 应先经过可视化再呈现。

本 Pass 不再创建 OpenPBR compute 管线、LUT 或 deferred color buffer，也不依赖环境光子系统。材质贴图只上传 MASK 几何所需的 base-color alpha 贴图；这属于可见性判定，不是着色。`VisibilityBufferShading.slang` 暂保留源码供后续独立着色阶段使用，当前 Pass 不编译、不调度它。

独立着色阶段现由 [VisibilityBufferDeferredPass](VisibilityBufferDeferred.md) 提供，
使用新的 `VisibilityBufferDeferred.slang` 和共享 OpenPBR 光照函数。
本 Pass 另发布 `rasterInfo` buffer，供延迟节点读取实际观察相机及 scene identity；
resident LookDev 比较通过 `LookDev.exe --sample lookdev-vbuffer` 启动。

RenderGraph 类型为 `VisibilityBufferPass`，内置图节点仍名为 `GPUDriven`。加载旧 JSON 时自动将 `GPUDrivenPreviewPass` 迁移到新类型；下次保存时写入新名称。

当前路径不创建 RTAS、不发起 ray query，也不计算几何阴影或几何环境遮挡。双面材质跳过 normal-cone backface 剔除。

## Visibility 数据契约与 Nanite 参考

参考本地 Unreal Engine 的 `NaniteDataDecode.ush::UnpackVisPixel` 与
`NaniteWritePixel.ush::WritePixel`，沿用“可见 cluster record + cluster 内 triangle”的索引式 visibility 思路：

- `ID = ((recordIndex + 1) << 7) | triangleIndex`，低 7 位容纳 128 个三角形，编码值 0 为背景。
- Record 引用 `VisibleClusterRecord`，而不是直接引用 meshlet：record 同时确定实例、cluster、geometry/page 和 producer。Resident 与 stream 使用互不重叠的 record 地址区间。
- `GPUDrivenRasterCommon.slang` 与 C++ `GPUDrivenRaster.h` 定义 ID 范围和记录格式。
- `GPUDrivenSceneCommon.slang` 为剔除、光栅和材质重建共享同一 GPUScene / push / params 布局，避免重复声明发生 ABI 漂移。
- 这里实现的是硬件 Mesh Shader visibility 路径：`R32Uint` ID + 固定功能深度附件。并未实现 Nanite 的软硬件混合光栅，也未引入它的 `uint64(depth, ID)` 原子竞争写入方式。

## 管线与固定剔除相机

Shader 职责按模块拆分：

| 模块 | 职责 |
| --- | --- |
| `GPUDrivenCulling.slang` | Reset、实例剔除、HZB 归约 |
| `VisibilityBuffer.slang` | AS meshlet 剔除、opaque/masked MS、visibility PS |
| `VisibilityBufferComposite.slang` | 可选的 ID / device depth / coverage 全屏调试显示 |

PSO 共用 `.cache/pso/VisibilityBufferPass.pso`，shader 内容及变体宏参与缓存键。
首次加载新 shader 会创建对应缓存；日志中的 `PSO cache status/hits/misses/stored/bytes` 可用于检查复用情况。

启用固定剔除相机后，Pass 锁存相机 pose、投影和裁剪参数。实例视锥/HZB、
meshlet 包围球和 normal cone 使用这台相机，viewport 相机仅投影幸存的 meshlet。
固定模式使用独立内部 visibility/depth 生成两阶段 HZB，避免误用观察相机深度。

## 可调开关

`VisibilityBufferPass` 暴露以下运行时设置；剔除默认开启，固定相机默认关闭：

- `visualization`：默认 `meshlet`，运行时切换无需重建图，不使 HZB 历史失效。
- `instanceFrustumCull`
- `instanceHzbCull`
- `meshletFrustumCull`
- `meshletNormalConeCull`
- `freezeCullingCamera`：勾选时捕获当前相机作为固定剔除相机；取消勾选后恢复使用实时 viewport 相机剔除。

| Visualization | JSON 值 | 显示内容 |
| --- | --- | --- |
| Meshlet ID | `meshlet` | 可见 cluster record ID 的哈希色（区分实例与 producer） |
| Triangle ID | `triangle` | 完整 packed ID 的哈希色，区分 meshlet 内三角形 |
| Depth | `depth` | 当前观察相机的原始 device Z 灰度，不做线性化 |
| Coverage | `coverage` | ID 非零为白色，背景为清屏色 |
| Off (VBuffer Only) | `none` | 仅生成原始 visibility / depth，color 保持清屏色 |

旧 `mode` 属性仍可读取；旧 Shaded / Base Color 回退为 Meshlet ID，不会重新启用材质着色。Depth 在固定剔除相机模式下仍显示观察相机深度，而非内部剔除深度。

## 验证

Shader 编译测试覆盖 Wave32 / atomic fallback 以及 opaque / masked 两种 MS，并检查 SPIR-V：opaque 不含用户 varying，masked 恰好导出 UV / material 两个 location。渲染测试覆盖 alpha 裁剪、固定相机、两阶段 HZB、奇数尺寸、五种可视化及环境光独立性，并读回验证 resident/stream 共用的原始 visibility/depth 不因可视化切换而改变。

```powershell
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --filter render_graph_gpu_driven_preview_shader_compile
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_preview_pass_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_sponza_visibility_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_alpha_mask_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_mixed_producer_render
```
