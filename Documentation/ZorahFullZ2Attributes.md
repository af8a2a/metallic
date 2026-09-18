# ZorahFull Z2：属性 cook 与保真

2026-09-18，基于 `4460a58a6` 的工作区改动。Z2 的属性 cook、数据保真、几何复用和单 primitive 内存验证已完成。真实 KTX2 材质的像素验收依赖 Z3 纹理资源和 Z4 流式材质消费者，本轮没有将数据校验记作完整材质渲染通过，也没有启动 Full 全量 cook。

精确结果见 [ZorahFullZ2Validation.json](E:/metallic/Documentation/ZorahFullZ2Validation.json)。源 glTF SHA-256 为 `4ae9247e8bbb5f5ef3cce6d7aa9cdf283bfba8cb765ddf2f3cf421d750313406`，与 Z1 相同。

## 已实现的属性契约

- resident 导入和离线范围读取 cook 共用 [GeometryAttributes.cpp](E:/metallic/Source/Runtime/Scene/GeometryAttributes.cpp)。缺失切线由 vendored meshoptimizer 的 `meshopt_TangentCompatible` 生成，使用 MikkTSpace 兼容规则；镜像 UV 或不同角点切线需要不同 TBN 时拆顶点，保留正确的 `tangent.w`。已有 authored tangent 原样保留。生成器只归一化临时法线，不改源 normal。
- P/N/UV/T 的 accessor 数量、有限值、法线及切线有效性、索引范围在 cook 前检查；错误的可选属性不能静默变成“缺失”。静态 cook 拒绝 morph target，轻量外部 glTF 读取器也保留该标记用于检查。合法缺 UV/normal 的源对象仍可 cook；unlit 探针没有被补造 normal/tangent。
- CLOD 同时接收 normal 和 UV，逐分量权重均为 `0.5`。保护 normal/UV 不连续点及切线手性边界；切线方向的误差权重为零。属性网格关闭忽略 seam 的 sloppy fallback，无法继续保真简化时保留终止组。
- CLOD 仍输出源顶点索引。所有 LOD 顶点的 P/N/UV/T 都取自已准备好的源属性，未引入 half 或有损量化。UV 保留原 tile 坐标，`KHR_texture_transform` 留给材质采样消费，不能在 cook 和 shader 重复应用。
- 当前完整布局仍为 position 12 + normal 16 + UV 8 + tangent 16 = **52 B/顶点**，另计 meshlet 局部重复、三角形索引、cluster/header/directory 和对齐。缺失属性不占对应流空间。

## 几何复用与持久化

离线 cook 在同一源内按完整 attribute accessor 映射、index accessor、primitive mode、material 和 morph targets 建立精确键。相同 accessor 且材质相同的条目只 cook 一次，各实例继续指向各自 render node；不同 UV accessor 或不同 material 不合并。恢复 checkpoint 时重建复用映射，包含暂停前已处理的别名。

按当前 Full 元数据，该策略将 7,277 个 primitive 条目变为 **5,715 个几何任务**，任务对应源三角形由 3,310,614,344 降至 **1,951,260,908**。这是元数据计划数，未执行全量 cook；不表示驻留显存或每帧三角形下降同样比例。跨材质共享仍需拆分 page/cluster 材质绑定，本轮保留两份几何。

缓存契约同步更新：

- resident meshlet 缓存版本 `1 → 2`，几何指纹加入 UV 和完整 tangent；修改外部 buffer 中的 UV 会使旧缓存失效。
- partial checkpoint 版本 `8 → 9`，避免续写旧简化规则的结果；meshopt 解码缓存仍可复用原格式。
- stream 容器/页布局不变，文件头原保留字段记录 `cookRevision=1`，`isCurrentForSource` 检查该值。旧 stream 文件仍可打开和解码，但需重新 cook 才具有新属性规则；单纯 transcode 不会升级 cook revision。旧 Z1 StoneUdim 的 20 页已验证可读，报告 revision 为 0。
- 运行时通过 `isRuntimeCompatibleForSource` 单独判断兼容性：允许复用 revision 0 的纯位置静态 glTF 缓存（例如既有 MiniZorah），前提是源文件/外部 buffer 指纹、构建参数匹配，所有页和源 primitive 均无额外顶点属性，且源不含 morph、skin 或 GPU instancing。带属性的旧 ZorahFull 缓存仍需重新 cook。离线 cook 的严格版本检查保持不变；运行时失败日志包含资源路径、失效原因和重建命令。

## 验证结果与代价

新增 `MetallicMeshletCook --validate-attributes`：逐 primitive 范围读取源数据，逐页解码，对所有 LOD 做 P/N/UV/T 位级元组核对；LOD0 还核对含重复计数的三角形集合与绕序。JSON 分别报告 P/N/UV/T、索引、cluster 字节数、源/准备后顶点数和负手性数量。该检查器面向小探针，额外映射占用不属于 cook 内存预算保证。

9 个小探针共 **106 页**通过全部属性和 LOD0 校验。另一路测试将原始 meshopt bufferView 独立解码，再由普通 TinyGLTF resident reader 读取 accessor，与每个 cooked LOD 对照，避免只比较同一离线读取器的两次输出。

下表属性字节为所有 LOD 的解码后流合计，包含 meshlet 内重复顶点，不含索引、cluster、header 和对齐；全部使用精确 B。

| 探针 | 页数 | Position | Normal | UV | Tangent |
| --- | ---: | ---: | ---: | ---: | ---: |
| StoneUdim | 20 | 357,120 | 476,160 | 238,080 | 476,160 |
| InstancingNoTangent | 1 | 48 | 64 | 32 | 64 |
| MaskedLeaves | 3 | 10,284 | 13,712 | 6,856 | 13,712 |
| TextureTransformBc4 | 8 | 18,204 | 24,272 | 12,136 | 24,272 |
| Glass | 15 | 395,664 | 527,552 | 263,776 | 527,552 |
| Blend | 1 | 48 | 64 | 32 | 64 |
| Unlit | 17 | 304,200 | 0 | 202,800 | 0 |
| SharedGeometry | 40 | 714,240 | 952,320 | 476,160 | 952,320 |
| MirroredInstances | 1 | 48 | 64 | 32 | 64 |

MaskedLeaves 从 399 个源顶点变成 403 个准备后顶点，新增 4 个切线拆分；合成镜像共享顶点四边形从 4 拆为 6，两个 UV chart 的 TBN 手性符合解析期望。粗级测试在平面、常法线、非线性 UV 的双 chart 上确认 UV 产生简化误差、三角形不跨 chart，常量切线空间法线经 TBN 后不翻向。

**保真有实际驻留成本。** Glass 的密集不连续边界使简化停止，15 页全部成为终止页，256 B 对齐后的根集为 **1,803,264 B，约 1.72 MiB**。Z1 的 30 页中含进一步粗化结果；本轮减少到 15 页不等于性能优化，反而会提高这个对象的保底驻留量。全量 cook 前需结合 Z3/Z4 实际材质对照决定是否有安全放宽空间，不能仅为减少 root 数量跳过 seam。

### 最大单 primitive 压力检查

脚本新选项 `--include-stress-primitive` 单独隔离 **mesh 942 / primitive 0**，这是源资产三角形数量最多的单 primitive：17,297,796 个三角形、8,807,104 个源顶点。它与 Z1 的最大 mesh 2322（多个 primitive 合计 32,054,623 个三角形）是不同口径。

| 项目 | 本机实测 |
| --- | ---: |
| workers / process memory budget | 2 / 4,096 MiB |
| cook + 页校验总时间 | 35.116 s |
| 峰值进程提交内存 | 2,985,291,776 B，约 2.78 GiB |
| 峰值工作集 | 2,478,358,528 B |
| 页数 / 终止页 | 11,610 / 3 |
| 文件字节 | 1,387,274,360 B |
| 终止页字节，256 B 对齐 | 269,056 B |
| payload 校验 | 11,610 页全部通过 |

这是单次功能压力检查，未建立冷盘基准，也不能据此保证所有 primitive 的峰值均小于此值。大探针只运行完整 payload 校验，没有运行需要额外源元组/三角形映射的 `--validate-attributes`。最大 mesh 2322 和完整 Full 均未 cook。

## 回归与复跑

- Release 构建通过：`MetallicSceneTests`、`MetallicMeshletCook`、`MetallicGPUDrivenSample`、`MetallicRhiTests`。
- **37 项 Scene 回归通过**，含 6 项新属性测试、Full metadata/实例化、小探针独立 resident 对照、UV 缓存失效、同键复用和断点恢复、损坏属性/morph 拒绝，以及原有 glTF/GLB、材质、场景层级测试。
- `RhiRendering.stream_metadata_contract` 通过，验证已有小场景的 metadata → GPUScene、HW/SW coverage 和材质解析兼容性；它不是 Full 贴图像素验收。
- 日志与逐探针原始报告位于 [zorah-z2](E:/metallic/build-release/zorah-z2)。Scene 回归在独立 `regression` 目录执行，避开仓库根的旧测试输出。

在仓库根生成和验证小探针：

```powershell
python Tools/PrepareZorahFullProbes.py --directory build-release/zorah-z2/probes --include-stress-primitive
$manifest = Get-Content build-release/zorah-z2/probes/probes.json -Raw | ConvertFrom-Json
foreach ($probe in $manifest.probes) {
    if (!$probe.cookRecommended) { continue }
    $output = [IO.Path]::ChangeExtension($probe.path, '.meshstream.bin')
    $report = [IO.Path]::ChangeExtension($probe.path, '.cook.json')
    & build-release/Source/MetallicMeshletCook.exe --source $probe.path --output $output --report $report --workers 4 --memory-mib 4096 --validate-attributes
    if ($LASTEXITCODE -ne 0) { throw "Cook failed: $($probe.name)" }
}
```

单独执行压力 cook（显式选择，不进入默认小探针循环）：

```powershell
& build-release/Source/MetallicMeshletCook.exe --source build-release/zorah-z2/probes/LargestPrimitive.gltf --output build-release/zorah-z2/probes/LargestPrimitive.meshstream.bin --report build-release/zorah-z2/probes/LargestPrimitive.cook.json --workers 2 --memory-mib 4096 --validate-payloads
```

接下来推进 **Z3：KTX2/BC4/BC5/BC7、按 mip 解压上传、swizzle 与色彩空间、受预算的纹理资源和统一句柄**。Z4 接通普通 stream 的 normal/UV/tangent、法线贴图、MASK 和透明消费者后，使用这套小探针补齐 resident/stream 图像对照，再启动 Z5 全量 cook。
