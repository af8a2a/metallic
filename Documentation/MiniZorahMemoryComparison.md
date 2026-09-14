# MiniZorah：Metallic 与 vk_lod_clusters 的内存差异

2026-09-13。依据用户截图、本地两套源代码、两份实际 cook 缓存，以及新增的一次 65 秒几何缓存漫游采样。本次只定位和记录，没有修改运行时策略或缓存格式；之前的崩溃修复保持原样。

截图 Geometry 为 975.470 / 14 MiB（约 69.7 倍），CLAS 为 1498.381 / 60 MiB（约 25 倍）。这不是同一相机、渲染分辨率、LOD 阈值与漫游历史的对照，不能把全部倍数归因于单一算法。已确认差异来自统计内容、驻留范围和每项数据的存储方式。

## 1. 图表统计的内容不相同

| 内容 | Metallic Streaming | vk_lod_clusters Streaming memory |
|---|---|---|
| Geometry | 几何页池已分配字节，包含保底页、缓存冷页及仍持有分配的上传/退役页 | 动态流送分配器 requestedSize；不含 persistentDataBytes |
| 位置数据 | VBuffer 需持续读取，保留在几何页内 | 光追路径仅上传到临时 CLAS 构建位置缓冲，动态持久 Geometry 不保留 |
| CLAS | 每 cluster 最坏尺寸槽位的已分配量，含退役待释放槽 | GPU 实际尺寸分配后的动态池占用，含分配器碎片；不含 persistentClasBytes |
| 未画入此图的部分 | 空闲预留容量、CLAS scratch、LOD 拓扑/状态等 | 常驻低精度几何/CLAS、拓扑、位置 scratch、其他 Operations、空闲预留容量等 |

参考图表直接读取 `stats.usedDataBytes` 和 `stats.usedClasBytes`，不是 Statistics 的总 Geometry/CLAS。它的 Statistics 总量另加 `persistentDataBytes` / `persistentClasBytes`。因此 14 + 60 MiB 也不等于程序总显存。

依据：[参考图表](E:/vk_lod_clusters/src/lodclusters_ui.cpp:1512)、[参考总量口径](E:/vk_lod_clusters/src/scene_streaming.cpp:1651)、[Metallic 图表取值](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:3653)。

## 2. 实际几何布局：位置保留方式最关键

直接解析当前 `MiniZorah.meshstream.bin` 的全部 1,356,959 个 page 记录：**所有页 attributeFlags 都为 9，即仅位置与材质标识**。实际 `.nvsngeo` 的 2,068 个 GeometryBase 也全部没有可选顶点属性。当前差距不能归因于 Metallic 多存了法线、UV 或切线。

| 数据 | Metallic | 参考 |
|---|---:|---:|
| 最高精度三角形总数 | 1,627,207,159 | 1,627,207,159 |
| 最高精度 clusters | 16,086,545 | 16,121,034 |
| 全 LOD clusters | 33,020,491 | 33,160,684 |
| 全 LOD 顶点记录 | 2,955,357,756 | 2,984,886,176 |
| 位置记录 | float4，16 B | float3，12 B；光追动态持久池不保留 |
| 局部三角索引 | uint8，3 B/triangle | uint8，3 B/triangle |
| 每 cluster 元数据 | cluster 96 B + 材质表 4 B | cluster 16 B + generating-group 4 B + BBox 32 B |
| 每 page/group 主头 | 112 B | 32 B |

**两份 cook 的最高精度三角数完全一致，cluster 数只差约 0.2%。** Geometry 数 3,163 对 2,068 与按 primitive/mesh 组织、参考的 geometry 去重有关，不能直接当成几何放大倍数。两边都在 cluster 内复用顶点、使用 8 位局部索引；Metallic 未按实例复制整份流送几何。

全 LOD 数据布局求和（这是离线总和，绝非运行时全量驻留）：

- Metallic 解码后页 payload：60,494,871,392 B（56.340 GiB）。其中位置 47,285,724,096 B，占 **78.16%**；索引 16.11%，cluster 头约 5.24%。256 B 页对齐额外增加约 **0.27%**。
- 参考保留位置的 runtime group：47,358,078,784 B（44.106 GiB）。Metallic 为其 **1.277 倍**。
- 参考 group 移除位置后的 runtime 前缀：11,531,305,832 B（10.740 GiB）。Metallic 为其 **5.246 倍**。此处对全部 group 使用相同布局计算；实际参考最粗 LOD 常驻组单独保留，不属于动态流送图表。

所以，VBuffer 与参考光追的 Geometry 比较自带很大的布局差异。仅把 float4 改为 float3，按当前全缓存组成估算可减约 **19.5%** 页 payload；它不能解释或消除 69.7 倍截图差异。VBuffer 仍需要位置，不能直接照搬参考光追的“构建后不保留位置”。参考命中着色器使用 `gl_HitTriangleVertexPositionsEXT` 读取 CLAS 内的顶点。

当前 Metallic 文件的 storedBytes 与 decodedBytes 相同，没有启用磁盘压缩；参考磁盘 group 总计约 25.440 GiB，位置压缩配置为丢弃 7 位尾数，但运行时会解码。只增加磁盘压缩主要改善磁盘/I/O，不会自动降低 Metallic 的 GPU 页分配。

依据：[Metallic payload 编码](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.cpp:843)、[参考几何结构](E:/vk_lod_clusters/src/scene.hpp:328)、[参考位置拆分上传](E:/vk_lod_clusters/src/scene_streaming.cpp:830)、[命中位置读取](E:/vk_lod_clusters/shaders/render_raytrace_clusters.rchit.glsl:163)。

## 3. CLAS 的固定最坏尺寸槽位与实际尺寸分配

Metallic 查询最大 128 vertices / 128 triangles CLAS 的空间，将返回值对齐为固定 `clusterStride`。本机 RTX 5070 Ti 为 **5,760 B/cluster**。每页按 `clusterCount × 5,760` 分配，即使某个 cluster 很小也占整槽。

已有实测 207,700 clusters 的池占用恰好为：

```text
207,700 × 5,760 = 1,196,352,000 B = 1,140.9 MiB
```

用户截图的 1,498.381 MiB 与约 272,772 个槽位一致。这个数表示分配空间，不能当作实际编码尺寸。槽位内部未用部分仍是真实的池容量浪费，单改 Profiler 显示不能回收它。

参考路径先在 scratch 构建，输出每个 CLAS 的 `dstSizesArray`；GPU 分配器累计组内实际尺寸，按实际字节量分配，再通过 `MOVE_OBJECTS_NV` 搬入持久池。当前 Metallic 的三角 CLAS RHI 描述没有尺寸输出或隐式目标选项，也没有接入这一步搬移。扩展本身支持尺寸输出与对象搬移，参见 [Khronos 扩展说明](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cluster_acceleration_structure.html)。

另一个待对齐项：MiniZorah 自带 `.cfg` 指定 `--claspositionbits 8`，Metallic 构建使用 0。当前参考截图没有显示该项，不能确认截图时仍为 8。这个精度选项与磁盘位置压缩的 7 位配置是两个独立设置。

没有读取两边同一批 clusters 的 GPU 实际 CLAS 尺寸，因此不能声称“改紧凑分配必然减少 25 倍”。目前只能确定固定最坏尺寸槽位存在，以及两边没有使用相同的分配/精度条件。

依据：[最坏尺寸槽位](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamClasPool.cpp:239)、[按槽位分配](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamClasPool.cpp:537)、[参考实际尺寸输出](E:/vk_lod_clusters/src/scene_streaming.cpp:1343)、[GPU 按组累计实际尺寸](E:/vk_lod_clusters/shaders/stream_allocator_load_groups.comp.glsl:185)、[场景精度配置](E:/metallic/Asset/MiniZorah/zorah_main_public.v2.cfg:7)。

## 4. 保留的驻留集合不同，图表会随漫游历史累积

Metallic 收到 GPU 的 unused 页列表后，将其标记为冷页并清空实际卸载请求，只有新的几何页分配遇到容量压力时才淘汰冷页。CLAS 跟随这些页保留；CLAS 容量不足会推迟构建，不会独立触发冷几何/冷 CLAS 回收。用户截图的几何/CLAS 已分别达到默认预算的 **95.3% / 97.6%**。

参考默认 `unloadThreshold=0`、`ageThreshold=16`，会主动卸载超过年龄阈值未用的组。常驻最粗层和仍被 BLAS cache 需要的层例外。因此它的曲线更接近近期工作集，Metallic 曲线则包含走过区域的缓存。

本次用现有几何漫游测试跑 65 秒、1920×1080、1.5 px、1 GiB 几何预算，验证通过且日志无 Vulkan validation 错误。该测试 fixture 没有启用 CLAS 设备能力，所以这些数值只用于验证几何缓存保留策略：

| 轨迹时间 | 几何池 MiB | 驻留页 | GPU 反馈为 unused 的缓存页 |
|---|---:|---:|---:|
| 5 s | 364.7 | 14,675 | 3,181 |
| 35 s | 597.4 | 21,370 | 11,835 |
| 60 s | 671.7 | 23,533 | 12,385（52.6%） |

unused 来自已完成的 GPU 反馈，有帧延迟；这是页数比例，不能直接当成字节比例。它明确表明已分配量不等于当前视角必需量。两张截图的相机、参考 1001×621 / 1 px、Metallic 截图的未知渲染分辨率，以及各自流送历史未对齐，剩余截图倍数还不能精确归因。

依据：[保留 unused 页](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:858)、[容量触发淘汰](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:1328)、[参考年龄阈值](E:/vk_lod_clusters/src/scene_streaming.hpp:126)、[参考年龄过滤](E:/vk_lod_clusters/shaders/streaming.glsl:36)。

## 建议顺序

1. **补齐同口径测量。** 分开展示动态热页、缓存冷页、保底页、退役页、实际 CLAS 编码字节、已分配字节、池预留与构建 scratch；固定相机/分辨率/LOD/流送重置条件对照。保留现有预算指标，避免把数字变小误当成回收显存。
2. **CLAS 改按实际尺寸分配并搬移。** 扩展已有 RHI 的尺寸输出/隐式构建及移动路径；地址表继续按逻辑页关联，维持退役延迟和重载保护。先保留当前精度评估分配收益，再单独评估尾数截断的质量影响。
3. **几何与 CLAS 采用共同的缓存回收目标。** 加冷页年龄、低/高水位和 CLAS 压力反馈；保留短期回看窗口与保底 cut，限制每帧回收量，避免重新引入页面重复装卸和请求尾延迟。只缩小几何预算会把过量保留转成装卸压力。
4. **优化 VBuffer 几何格式。** 先处理 float4 的无效第四分量、精简 cluster 头，再评估量化位置；未来独立光追消费路径可采用临时位置构建方式，VBuffer 路径继续保留可访问的位置。

[结构化审计结果](E:/metallic/Documentation/MiniZorahMemoryComparison.json) · [原始缓存统计](E:/metallic/build-relwithdebinfo/memory-comparison/AssetMemoryAudit.json) · [漫游原始数据](E:/metallic/build-relwithdebinfo/memory-comparison/roaming/MiniZorahRoamingReport.json) · [漫游日志](E:/metallic/build-relwithdebinfo/memory-comparison/roaming.log)
