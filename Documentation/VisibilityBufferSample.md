# Visibility Buffer Sample

`MetallicGPUDrivenSample` 默认加载 `gpu-driven-sample`，使用仓库附带的
`Asset/SuperSponza/NewSponza_Main_glTF_003.gltf`。入口命令：

```powershell
cmake-build-release-visual-studio\Source\MetallicGPUDrivenSample.exe --smoke-test
```

## 帧内数据流

1. 第一阶段实例剔除以当前相机做包围球视锥测试，并以相机上一帧参数查询上一帧 HZB。
2. 可见实例对应的 meshlet ID 被 compute compact 到列表，并生成 `drawMeshTasksIndirect` 参数。
3. mesh shader 对每个 meshlet 继续执行包围球视锥剔除和 normal-cone backface 剔除，写入 `R32Uint` visibility ID 与深度。
4. 深度被 compute 归约成当前帧 HZB。Reversed-Z 使用 min reduction，普通 Z 使用 max reduction。
5. 第二阶段只重测第一阶段的 HZB 遮挡候选，并把新可见 meshlet 补绘到同一 visibility/depth buffer。
6. 完整深度再次生成 HZB，供下一帧使用。
7. Mesh shader 把最多 128 个三角形的 meshlet 分成两个 64-triangle chunk；每个三角形复制三个输出顶点，以普通 flat varying 写入稳定的 `meshlet + local triangle` visibility ID，避免依赖片元阶段未定义的 primitive ID。
8. Deferred compute 根据 visibility ID 读取 meshlet 顶点/索引，重建 perspective-correct barycentrics、world position、normal/tangent/UV，并采样 glTF base-color、metallic-roughness、normal、AO、emissive 和 transmission 纹理。
9. Deferred compute 把 glTF 参数映射到 `OpenPBR_ResolvedInputs`，调用 `openpbr_prepare` 与 `openpbr_eval` 完成 BSDF 计算。HDR 环境贴图在 CPU 端预计算为 9 系数低阶球谐，shader 用 24 个局部 Fibonacci 球面方向做稳定的环境近似积分；背景仍显示原始 HDR。
10. 全屏 pass 只负责把 ACES tone-map 后的 compute 颜色缓冲复制到最终 `Rgba8Unorm` 输出。

当前路径刻意不创建 RTAS、不发起 ray query，也不计算阴影或环境遮挡。双面材质会跳过 normal-cone backface 剔除。

启用固定剔除相机后，Pass 会在切换瞬间锁存当前相机的完整 pose、投影和裁剪参数。实例视锥/HZB、meshlet 包围球和 normal cone 都继续使用这台虚拟相机；viewport 相机只负责投影幸存的 meshlet，因此可以自由移动到视锥外观察剔除结果。固定模式使用独立的内部 visibility/depth 生成 2-pass HZB，避免把观察相机的深度误用于虚拟相机剔除。

## 可调开关

`GPUDrivenPreviewPass` 暴露以下运行时设置，默认全部开启：

- `mode`：`Shaded`（OpenPBR）、`Base Color`、`Meshlet`、`Primitive`、`LOD Group`。
- `instanceFrustumCull`
- `instanceHzbCull`
- `meshletFrustumCull`
- `meshletNormalConeCull`
- `freezeCullingCamera`：勾选时捕获当前相机作为固定剔除相机；取消勾选后恢复使用实时 viewport 相机剔除。

## 验证

```powershell
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --filter render_graph_gpu_driven_preview_shader_compile
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_preview_pass_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_sponza_visibility_render
```
