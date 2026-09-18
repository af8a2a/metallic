# ZorahFull Z1：导入与探针

2026-09-18。实现基于 `e041dcf9e`，本次改动尚未提交。Z1 已完成外部 glTF 实例化、纯 metadata 导入和可重复生成的探针集。完整属性着色、KTX2 上传及 Full 全量 cook 仍按 Z2–Z5 推进。

## 导入结果

实际运行 `Scene::loadStreamMetadata(Asset/ZorahFull/zorah_textured_public.v1.gltf)`：

| 项目 | 验证值 |
| --- | ---: |
| 原始节点 | 13,079 |
| 扩展实例生成的子节点 | 7,654 |
| 展开后节点 | 20,733 |
| mesh 实例 | 16,118 |
| primitive 实例 | 43,068 |
| 含实例三角形 | 18,930,392,835 |
| 材质 | 1,514 |
| texture / image 描述 | 4,418 / 4,418 |
| 实例 accessor 范围读取 | 306,160 B，约 299 KiB |
| 几何 / 图像 payload | 未读取、未解码；CPU 数组为空 |

计数未应用参考 cfg 的 `skipmeshes`。这是完整源资产的导入口径，不是每帧可见量。Full 单次 metadata 加载约 0.8–0.9 秒，仅供本机功能检查，未建立冷盘性能基准。

源 glTF SHA-256：`4ae9247e8bbb5f5ef3cce6d7aa9cdf283bfba8cb765ddf2f3cf421d750313406`。机器可读结果见 [ZorahFullZ1Validation.json](E:/metallic/Documentation/ZorahFullZ1Validation.json)，原始日志见 [tests.log](E:/metallic/build-release/zorah-z1/tests.log)。

## 实现协议

- [GltfGpuInstancing.cpp](E:/metallic/Source/Runtime/Scene/GltfGpuInstancing.cpp) 将 `EXT_mesh_gpu_instancing` 投影为普通子节点，供 metadata、resident `.gltf` 和外部范围读取 cook 共用。源节点保留原编号和层级变换，移除其普通 mesh，避免多画一份；实例世界矩阵为 `node world × instance TRS`。原有子节点只保留一次，不随实例复制。
- 生成节点的源 node / instance 编号保存在 `LoadResult.gpuInstancing`，同一输入的展开顺序稳定。primitive 顺序沿用 mesh 中的顺序；metadata 与离线实例表逐项核对 render node、material 和 16 个矩阵分量。
- 只按实例 accessor 范围读取外部 buffer。检查类型、归一化、stride、offset、声明及物理文件边界、count 一致性、有限值和四元数；几何 buffer 和 KTX2 不进入 metadata 载入路径。
- metadata 保留图片 URI、MIME、sampler、texture 及已有材质字段，并在 `LoadResult.gltfMaterialDescriptions` 保存全部源材质 JSON。`specular/unlit` 尚未接入着色的部分不丢源描述，仍明确报告 ignored-extension 警告。MIME 是原始提示，Full 中错误的 PNG MIME 没有被伪装成已验证的容器类型。

Z1 实例 accessor 支持 float T/S、float 或归一化 signed byte/short R，以及缺省 TRS 分量。当前明确拒绝实例 accessor 的 sparse、meshopt 压缩、嵌入数据和自定义实例属性；metadata 仍拒绝 skin/animation/morph 和嵌入图像。ZorahFull 的实际外部静态 TRS 数据符合该范围。此次没有宣称支持 `.glb` 内的 GPU instancing。

## 代表性探针

[PrepareZorahFullProbes.py](E:/metallic/Tools/PrepareZorahFullProbes.py) 保留选中节点的祖先链，并裁剪、重映射 mesh/accessor/bufferView/material/texture/image/sampler。全部 payload 使用原文件引用，不复制几十 GiB 数据。源材质扩展、texture transform 和各张 UDIM 文件引用保持完整。

脚本另用源节点和原始实例 accessor 独立计算列主序参考世界矩阵，与 C++ 导入结果比较；不以 C++ 展开结果生成期望值。manifest 保存源 glTF/cfg 哈希、完整 cfg 参数和源索引映射。`MirroredInstances` 的额外父级为 `scale=(-1,2,0.5)`，在 manifest 中明确标记。

| 探针 | 源 mesh | 源三角形合计 | primitive 实例 | cook 页面 | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| StoneUdim | 12 | 22,430 | 11 | 20 | 石材、authored tangent、normal map、显式 UDIM 文件 |
| InstancingNoTangent | 49 | 2 | 25 | 1 | 扩展实例、缺 authored tangent 的 normal map |
| MaskedLeaves | 1312 | 369 | 2 | 3 | MASK、双面、alpha cutoff |
| TextureTransformBc4 | 398 | 936 | 2 | 8 | UV transform、BC4 specular、masked 叶片 |
| Glass | 334 | 15,872 | 1 | 30 | material 424 transmission / IOR |
| Blend | 294 | 2 | 1 | 1 | material 395 BLEND |
| Unlit | 109 | 19,200 | 1 | 17 | material 1513，无 authored normal |
| SharedGeometry | 12、844 | 44,860 | 19 | 40 | 相同 accessor 几何、不同材质 |
| MirroredInstances | 49 | 2 | 25 | 1 | 镜像、非均匀父级缩放、实例组合 |
| LargestMesh | 2322 | 32,054,623 | 12 | 未 cook | 独立内存压力探针，仅验证 metadata |

输出及逐实例参考值见 [probes.json](E:/metallic/build-release/zorah-z1/probes/probes.json)。9 个小探针已完成 cook，全部 **121 页**通过 payload 解码校验。其离线实例数量、材质和世界矩阵与 metadata 一致。这里验证结构和映射，不表示法线/UV/tangent 的 LOD 保真或 MASK/透明着色已经通过。

`SharedGeometry` 当前仍产出两份带不同材质约束的几何，保留为 Z2 几何键分离的对照。镜像矩阵已核对；镜像绕序、TBN、alpha 覆盖的完整像素对照需要后续属性/材质消费者，不把此次结果记为渲染保真验收。

## 验证与复跑

- Release 构建通过：`MetallicSceneTests`、`MetallicMeshletCook`、`MetallicGPUDrivenSample`、`MetallicRhiTests`。
- 30 项定向 Scene 回归全部通过，含 5 项新增实例化测试、完整 Full metadata、10 个探针、已有普通 glTF/GLB、材质、层级、外部图像延迟加载和 cook 持久化/恢复测试。
- `RhiRendering.stream_metadata_contract` 通过：128×128 离屏验证 metadata → GPUScene、双面 HW/SW coverage 及 resident/stream 材质解析兼容性。它是已有小场景回归，不是 ZorahFull GPU 渲染。
- 探针脚本连续生成的 10 个 glTF 哈希一致；`git diff --check` 通过。
- 初次从仓库根执行旧测试时，旧 `scene-test-output` 无写权限且读到陈旧夹具；改在新的 `build-release/zorah-z1/regression` 目录执行后全部通过，无需更改旧目录权限或内容。

生成探针（仓库根）：

```powershell
python Tools/PrepareZorahFullProbes.py --directory build-release/zorah-z1/probes
$manifest = Get-Content build-release/zorah-z1/probes/probes.json -Raw | ConvertFrom-Json
foreach ($probe in $manifest.probes) {
    if (!$probe.cookRecommended) { continue }
    $output = [IO.Path]::ChangeExtension($probe.path, '.meshstream.bin')
    & build-release/Source/MetallicMeshletCook.exe --source $probe.path --output $output --workers 4 --memory-mib 4096 --validate-payloads
    if ($LASTEXITCODE -ne 0) { throw "Cook failed: $($probe.name)" }
}
```

验证 Full 和探针（先完成上述小探针 cook）：

```powershell
$env:METALLIC_TEST_ZORAH_FULL = '1'
$env:METALLIC_ZORAH_Z1_PROBES = 'E:/metallic/build-release/zorah-z1/probes/probes.json'
$env:METALLIC_ZORAH_Z1_REPORT = 'E:/metallic/build-release/zorah-z1/full-metadata.json'
& E:/metallic/build-release/tests/MetallicSceneTests.exe '--gtest_filter=GltfInstancing.*'
```

下一阶段按 Z2 推进：先使用 StoneUdim / InstancingNoTangent / TextureTransformBc4 / SharedGeometry 固定 LOD0 属性契约，再处理粗 LOD 的 UV seam、切线符号和属性误差，之后才启动最大 mesh 及 Full 全量 cook。
