# Visibility Buffer Sample

`MetallicGPUDrivenSample` 默认加载 `gpu-driven-sample`，使用仓库附带的
`Asset/SuperSponza/NewSponza_Main_glTF_003.gltf`。入口命令：

```powershell
build-msvc-x64\Source\MetallicGPUDrivenSample.exe --smoke-test
```

## 帧内数据流

1. 第一阶段实例剔除以当前相机做包围球视锥测试，并以相机上一帧参数查询上一帧 HZB。
2. 可见实例对应的 meshlet ID 被 compute compact 到列表，并生成 `drawMeshTasksIndirect` 参数。
3. mesh shader 对每个 meshlet 继续执行包围球视锥剔除和 normal-cone backface 剔除，写入 `R32Uint` visibility ID 与深度。
4. 深度被 compute 归约成当前帧 HZB。Reversed-Z 使用 min reduction，普通 Z 使用 max reduction。
5. 第二阶段只重测第一阶段的 HZB 遮挡候选，并把新可见 meshlet 补绘到同一 visibility/depth buffer。
6. 完整深度再次生成 HZB，供下一帧使用。
7. 空 deferred compute 根据 visibility ID 生成 meshlet 调试颜色；全屏 pass 只负责把颜色缓冲复制到最终 `Rgba8Unorm` 输出。

当前 deferred 阶段不读取材质，也不做实际光照。双面材质会跳过 normal-cone backface 剔除。

## 可调开关

`GPUDrivenPreviewPass` 暴露以下运行时设置，默认全部开启：

- `instanceFrustumCull`
- `instanceHzbCull`
- `meshletFrustumCull`
- `meshletNormalConeCull`

## 验证

```powershell
build-msvc-x64\tests\MetallicRhiTests.exe --filter render_graph_gpu_driven_preview_shader_compile
build-msvc-x64\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_preview_pass_render
build-msvc-x64\tests\MetallicRhiTests.exe --filter render_graph_gpu_driven_sponza_visibility_render
```
