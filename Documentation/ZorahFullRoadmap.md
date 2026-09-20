# ZorahFull：完整属性与贴图加载路线

2026-09-17。Metallic 源码基线 `5b5a3c3cc`；参考 `E:/vk_lod_clusters` 基线 `1febfa7694ebdc8f97005a07897029272ca9e85a`。目标资产为 `Asset/ZorahFull/zorah_textured_public.v1.gltf`，以同目录 `.cfg` 建立相机与内容筛选基准。

本轮完成 glTF 元数据、外部依赖、全部 4418 张 KTX2 的头/level index/元数据审计和当前源码核对。未启动全量 cook，未加载 GPU 场景，也没有新的帧时测量。可复跑工具：[AuditZorahFull.py](E:/metallic/Tools/AuditZorahFull.py)；精确计数、源 glTF SHA-256 和纹理预算估算：[ZorahFullAssetAudit.json](E:/metallic/Documentation/ZorahFullAssetAudit.json)。

2026-09-18 更新：**Z1 导入与探针已完成**，实测计数及验证见 [ZorahFullZ1ImportAndProbes.md](E:/metallic/Documentation/ZorahFullZ1ImportAndProbes.md)。**Z2 属性 cook、数据保真与预算验证已完成**，见 [ZorahFullZ2Attributes.md](E:/metallic/Documentation/ZorahFullZ2Attributes.md)；真实 KTX2 材质的像素对照依赖 Z3/Z4。以下资产数据保持 Z0 审计口径，实现状态随阶段更新。

## 结论

沿用当前几何流送、VBuffer、CLAS 和实时 OpenPBR 主线，先闭环 **完整实例/材质语义 + 低 mip 全场景首帧**，再实现 **受预算约束的纹理细节流送**。不需要先重写几何遍历，也不需要把完整虚拟纹理系统作为首帧的前置条件。

2026-09-18 Z3 更新：**受预算的 KTX2/BC 纹理资源与寻址已完成**，4418 张纹理的 512 mip 尾链已上传并验证；实际 image allocation 1.369 GiB，峰值 staging 192 MiB。见 [ZorahFullZ3Textures.md](E:/metallic/Documentation/ZorahFullZ3Textures.md)。

2026-09-19 Z4 更新：流式属性、OpenPBR specular/unlit、MASK 光栅与 CLAS alpha、BLEND/玻璃续追已接入。实现、探针验收及透明阴影边界见 [ZorahFullZ4StreamingShading.md](E:/metallic/Documentation/ZorahFullZ4StreamingShading.md)。

2026-09-20 Z5 更新：**全量 cook 与完整材质首帧已完成**。5715 份去重几何、43068 个实例、1676913 页通过原始/紧凑布局全页校验；GPU 根页由 5.232 GiB 降到 2.956 GiB。独立 asset 和编辑器 world 入口均通过首帧及释放检查，4418 张纹理使用 512 mip 尾链。运行 `MetallicGPUDrivenSample.exe --zorah-full` 或选择 `GPUDriven / ZorahFull`，采用 Full 专用的 3.5 GiB geometry / 2 GiB CLAS 预算。详情和图像见 [ZorahFullZ5FirstFrame.md](E:/metallic/Documentation/ZorahFullZ5FirstFrame.md)。当前几何池细化后会触顶，尚未验收 1.5 px 收敛或持续纹理细节流送。

2026-09-18 编辑器入口修复：在 GPUDrivenSample 的流式 VBuffer 图中使用 File/Open，现在先异步校验所选场景的 meshstream，再导入 metadata，不进入普通常驻 glTF 几何/图像解码和静态 RTAS 准备。重载同一场景保留图中显式缓存路径；切换场景使用 `<source.gltf>.meshstream.bin`，校验成功后一起提交源路径、缓存路径和 world 绑定。缓存缺失或过期会显示具体路径和 `MetallicMeshletCook --source ... --output ...` 命令，保留当前场景。ZorahFull 的无 URI buffer 是 meshopt 解压目标，原先 File/Open 的 “uri is missing” 是入口未接入流式加载的误导性错误；普通常驻入口也会提前明确报告不支持 meshopt，而非报缺文件。Full 全量缓存尚未生成，完整材质和首帧仍按 Z4/Z5 推进。

入口修复验收：Release 构建通过；17 项 Scene 测试通过（另有 1 项可选 SuperSponza 测试跳过），覆盖真实 Full metadata/缺失 cook、小型 meshopt 场景的异步流式导入、无几何/贴图 CPU payload、缓存失效、文档保存/重载及普通场景回归。2 项 RHI 测试通过，覆盖源/缓存切换和流式/常驻混合渲染。原始日志和 GoogleTest JSON 位于 `build-release/zorah-full-open/`；未执行 Full 全量 cook 或完整材质渲染。

## 1. 资产实况与口径

以下采用本机实际文件，不照抄 README 的近似数值。README 的 2813 meshes / 13080 nodes / 4462 纹理文件，与当前解压内容有小幅差异；本轮引用的所有外部 buffer/image 均存在。

| 项目 | 实际值 | 对路线的影响 |
| --- | ---: | --- |
| mesh / primitive / node | 2812 / 7277 / 13079 | 有 3736 个带子节点的节点，不能沿用 MiniZorah 扁平节点探针脚本 |
| `EXT_mesh_gpu_instancing` 节点 | 1092 | 必须读取实例 TRS；仅保留普通 node 会漏实例 |
| mesh 实例 / primitive 实例 | 16118 / 43068 | CPU/GPU 映射与拓扑容量按 primitive 实例核对 |
| mesh 条目累计三角形 | 3,310,614,344 | 含引用同一 accessor 的重复 mesh 条目，不能直接视为全新独立几何 |
| 精确 accessor 几何去重 | 5408 组 / 1,630,979,739 三角形 | 仅按属性 accessor、索引 accessor 和 primitive mode 去重，未做内容哈希 |
| 几何 + material 精确去重 | 5715 组 / 1,951,260,908 三角形 | 同材质的重复处理已有明显可避免空间；跨材质共享需分离几何与材质绑定 |
| 含实例三角形 | 18,930,392,835 | 未应用 cfg 的 skipmeshes；不是驻留或每帧绘制数量 |
| 材质 / 引用纹理 | 1514 / 4418 | 显著超过当前 256 张纹理上限 |
| OPAQUE / MASK / BLEND 材质 | 1403 / 110 / 1 | alpha 覆盖属于必要功能，不能仅验证不透明墙体 |
| 双面 / transmission / unlit | 157 / 1 / 1 | 玻璃、双面叶片及 unlit 各需明确消费者 |
| `KHR_materials_specular` | 1388 个材质 | 不是可忽略的少数扩展；含 61 个 specular texture 引用 |
| 外部几何及实例文件 | 30.372 GiB | 已拆成 2034 个几何 buffer + 1 个实例 buffer，可继续有界分片读取 |
| KTX2 磁盘 / BC 全 mip payload | 47.258 / 85.024 GiB | Zstd 解压后仍是 BC 数据；磁盘大小不是 GPU 需求 |

几何复用实例：mesh 43 / 135 / 257 复用相同 POSITION、NORMAL、UV 和 indices accessor，但使用不同 material。当前 page/header 带有 material 约束，不能只按 POSITION 合并；需保持 UV、法线、切线、primitive mode、简化设置及材质覆盖策略的兼容性。同几何共享 CLAS 时还要核对 opaque/masked flags 和材质寻址。

### 属性与特殊纹理

- NORMAL：7276/7277 primitives；UV0：7237/7277；TANGENT：1713/7277。缺切线时需要与法线贴图一致的生成规则。缺 UV 的 40 个 primitive 使用两个无纹理 placeholder 材质，不应因此拒绝整个资产；唯一缺 NORMAL 的 primitive 使用 unlit 材质。
- 所有材质纹理引用使用 UV0；存在 **501 个 `KHR_texture_transform`**。应正确变换 UV 与导数，不必为该资产先做 UV1 支持。
- 3420 张引用纹理带 UDIM tile 文件名，但 glTF 已逐张引用这些文件。先验证其 primitive/material/UV 绑定，不应仅因文件名就引入完整 UDIM atlas 或虚拟纹理页表。
- 1583 张 BC7 sRGB、2774 张 BC5 UNORM、61 张 BC4 UNORM；全部使用 Zstd supercompression。1375 张声明 `KTXswizzle=1rg1`，61 张声明 `111r`，其余没有 swizzle。
- 金属度/粗糙度原始 BC5 为 R=roughness、G=metallic；`1rg1` 映射后正好恢复 glTF 的 G/B。BC4 specular 的 `111r` 将强度放进 alpha。**应用一次文件 swizzle 后继续使用标准 glTF 材质语义，不能在 shader 再交换一次。** 参考实现见 [scene_textures.cpp](E:/vk_lod_clusters/src/scene_textures.cpp:364)。
- 4069 个 image 仍声明 `mimeType=image/png`，实际文件为 KTX2。加载按容器 magic 校验，并报告 MIME 不一致，不能交给 PNG decoder。
- 存在 4096×4200 等非二次幂纹理和最长边 8192 的纹理。BC 上传布局应按压缩块计算，并覆盖小于 4×4 的末尾 mip。

KTX2 的每个 mip 可独立 Zstd 解压，适合按需读入；容器提供 format、level offsets 和 swizzle 元数据。[Khronos KTX2 规范](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)

## 2. 当前已有能力与缺口

| 环节 | 已有，可复用 | 仍需补齐 |
| --- | --- | --- |
| 几何/CLAS | 可恢复 cook、页请求、完整 cut、混合光栅、紧凑 CLAS、联合冷页回收 | 全属性数据量、MASK 的光栅及 RT 覆盖不能用 MiniZorah position-only 验证替代 |
| 新位置格式 | 新 cook 已写 Float32x3；旧 float4 可加载时转换 | 不把“16→12 B 位置”重复列为待办；完整属性采用何种编码另行决定 |
| 场景 metadata | Z1 保留外部 image/texture 描述和源材质 JSON；Z3 从 metadata 独立准备纹理，无全量几何载入 | 未消费的材质扩展仍在 Z4 |
| glTF 实例化 | Z1 已将外部 `.gltf` 的 TRS 实例统一展开；runtime metadata、resident 与 cook 共用规则，保留源 node/instance 映射 | 当前边界为静态、外部、非 sparse/非压缩的实例 accessor；不等同于已支持所有扩展存储形式 |
| cook 属性 | Z2 共用兼容 MikkTSpace 的切线生成与镜像拆分；normal/UV 参与简化并保护 seam/手性；9 探针全部 LOD 属性核对通过；同 accessor+material 复用及缓存失效/恢复已验证 | Glass 的保真约束使 1.72 MiB 终止页常驻；真实法线贴图、texture transform 和 alpha 的像素质量须由 Z3/Z4 验证 |
| stream resolve | Z4 普通 surface 与 CLAS ray hit 均解码 normal/UV/tangent；保留镜像手性、稳定 TBN 与纹理 footprint | Full 全场景质量及持续漫游验证属于 Z5/Z6 |
| 纹理资源 | Z3 支持 KTX2+Zstd 的 BC 尾链直传，512 cap / 2 GiB 默认分配预算，保留 PNG 路径 | 视图驱动 mip 提升/回收、跨场景和几何/CLAS 总预算属于 Z6 |
| RHI | Z3 已实现 BC4/5/7、压缩块行距/边缘上传、view 与 bindless swizzle | 更广 KTX2 类型及 Basis transcoding 不在本资产 Z3 范围 |
| 纹理寻址 | Z4 alpha raster、deferred 与 stream shadow 共享纹理 owner；流式独立 heap 使用自己的 descriptor 映射；移除其固定 4096 cap | 通用 sampler 与纹理动态升降 mip 仍待后续 |
| PBR 语义 | Z4 接入 specular 因子/颜色/纹理及 unlit；BC5 Z、BC4 alpha swizzle、单次 sRGB 解码及 stream UV/TBN 可用于着色 | glTF 与 OpenPBR 中间金属度的 specular 插值是参数化映射，不宣称跨渲染器逐像素相同 |
| alpha/RT | Z4 MASK 写深度前测试并路由 HW；CLAS 命中可重建 UV/material；BLEND/玻璃使用有界 OpenPBR 续追 | 独立 SIGMA 通道仍为二值透明阴影；alpha query 的全量 non-opaque 回调可后续优化 |

关键源码：[metadata](E:/metallic/Source/Runtime/Scene/scene.cpp:2640)、[cook 扩展](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.cpp:3296)、[实例生成](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.cpp:4087)、[CLOD 属性](E:/metallic/Source/Runtime/Scene/scene.cpp:1504)、[stream 解码](E:/metallic/Shaders/Features/VisibilityBuffer/VisibilityStreamDecode.slang:74)、[surface 重建](E:/metallic/Shaders/Features/VisibilityBuffer/VisibilityBufferDeferred.slang:100)、[纹理上传](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:521)、[容量上限](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.h:16)、[RHI format](E:/metallic/Source/Runtime/Render/GAPI/rhi.h:68)。

旧的 2026-09-15 差距报告不再完全代表当前源码。Float32x3、whole-cut BLAS 复用、MOVE 提交内发布和部分 frame overlap 已落地，见 [MiniZorahGpuDrivenReuse](E:/metallic/Documentation/MiniZorahGpuDrivenReuse.md)。本次重点是内容扩展，先保留这些机制。

## 3. 推进路线与验收门槛

| 阶段 | 交付 | 验收，满足后进入下一阶段 |
| --- | --- | --- |
| Z0：资源审计 | **本轮已完成**元数据/文件头报告、预算估算、兼容性缺口 | 所有引用文件存在；4418 个 KTX2 头及各 BC mip 长度符合尺寸；不将此记为完整 decode/render 验证 |
| Z1：导入与代表性探针 | **2026-09-18 已完成**；共用实例展开、metadata 描述、10 个探针及 cfg manifest | 未筛选时 16118 / 43068 实例；父级/镜像/非均匀变换与独立参考矩阵一致，9 个小探针 cook 实例表一致；未载入全量图像/几何。镜像着色及 alpha 的像素验收继续归 Z4 |
| Z2：属性 cook 与保真 | **2026-09-18 cook/数据验收完成**；normal/UV/tangent 契约、seam/手性保护、缺失切线拆分、缓存恢复、同材质精确几何复用 | 9 探针 106 页全 LOD 属性及 LOD0 绕序一致，粗级合成 UV/TBN 检查通过；最大单 primitive 17.3M 三角形 cook 峰值提交 2.78 GiB；真实材质像素对照随 Z3/Z4 完成，未全量 cook |
| Z3：KTX2 与受预算的纹理资源 | **2026-09-18 资源验收完成**；BC4/5/7、Zstd、swizzle、色彩、mip-tail；实际数量描述符与 stream raster/deferred 共享 owner | 4418 引用均有效；合成图案 GPU 采样、小块/NPOT 回读、低预算降 mip 通过；512 cap payload 1.358 GiB、image allocation 1.369 GiB、峰值 staging 192 MiB。完整材质/stream shadow 像素验收属 Z4 |
| Z4：完整流式材质消费者 | **2026-09-19 主链路及探针验收完成**；stream normal/UV/tangent、OpenPBR specular/unlit、MASK/双面及 CLAS alpha、BLEND/玻璃续追 | 8 个真实探针分批对照、HW/混合覆盖、MASK 阴影及 ID >255 通过，详情见 Z4 报告；透明 SIGMA 阴影仍为明确边界 |
| Z5：全场景 cook 与首帧 | **2026-09-20 已完成**；全量 revision 2 cook、上传时紧凑 N/T、Full cfg 相机和 512 cap 全材质首帧 | 原始/紧凑全部 1676913 页通过校验，51764 根页完整 ready；43068 实例、4418 纹理及有效 descriptor 完整，asset/world 两入口加载释放通过；持续质量收敛归 Z6 |
| Z6：持续漫游与细节流送 | 保底 mip 常驻，按可见纹理 footprint 提升/降低 mip；几何、CLAS、纹理共享总预算协调及上传节流；profiling 图表 | 固定路线往返、急转、近景、低预算测试；无 DeviceLost/悬空 descriptor；质量收敛、请求尾延迟、重载量和 P95/P99 可解释 |

依赖为 **Z1 → Z2**，**Z1 → Z3**，**Z2+Z3 → Z4 → Z5 → Z6**。Z2/Z3 是独立模块；全量 cook 应等格式和小场景验证稳定，避免昂贵的反复烘焙。

Z1 探针至少包括：普通贴图石材、缺 authored tangent 的 normal map、带 texture transform 的 UDIM 文件引用、MASK 双面叶片、BC4 specular、扩展实例化及负缩放、两个不同材质共享几何的对象。另含 material 424 彩玻璃、395 BLEND 反射贴片、1513 unlit 喷泉球，以及最大 mesh 2322 的单独内存压力探针。不把最大 mesh 与所有纹理一起交给旧 resident 加载器。

### 几个实现决策

1. **实例展开是导入协议。** 实例世界矩阵为 node world × instance TRS，扩展节点不能再额外画一份普通实例；cook 与 runtime 需保留稳定源 node / instance / primitive 对应。metadata 不应删除实例 accessor 所需的 buffer 后再试图展开。[Khronos 实例化规范](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_gpu_instancing/README.md)
2. **先属性正确，再定全量编码。** 现有全属性浮点布局约为 position 12 + normal 16 + UV 8 + tangent 16 = 52 B/顶点，尚未计局部重复、索引、元数据及对齐。不能将 position-only 预算原封不动用于 Full。先测小样，再决定 normal/tangent 压缩；UV 是否能用 half 需验证纹理变换、精度和边界，不能默认安全。
3. **纹理只保留一份资源所有权。** 材质着色、alpha raster 和 shadow 使用统一 provider/逻辑 ID；缓存键区分文件内容、view 色彩空间、swizzle 和 sampler。上传完成后发布版本，在途帧结束后退役旧 image/view/descriptor，避免漫游期间重演资源生命周期崩溃。
4. **MASK 在写 visibility/depth 前求值。** 第一版可将 masked cluster 明确归入已实现 alpha 的 HW 路径，不强求立刻加 SW alpha；HW-only 与混合模式对照必须保持覆盖一致。仅在 deferred 丢弃像素无法修复已经被前景叶片遮掉的背景。RTAS 的 opaque 标志和阴影 alpha 同步处理。
5. **颜色空间和 normal 格式显式化。** BC7 sRGB 由硬件解码时跳过旧 shader 的手动 sRGB 转换；保留既有 PNG 语义。BC5 normal 从 XY 重建 Z，normal scale、镜像 tangent sign 和非均匀缩放遵循稳定 TBN，不能对 authored normal 提前 face-forward。
6. **少量透明材质单独闭环。** BLEND 与 transmission 分别有一例。可走明确的透明/折射消费者并继续使用有界流式几何，或为这少量内容建立有预算的独立表示；在其完成前，阶段结果只能称“opaque/masked 首帧”，不能称“完整材质完成”。无 skin/animation，因此动态角色/风动/通用 WPO 不必成为此次前置条件。

## 4. 纹理预算与流送选择

下表读取全部引用 KTX2 的 mip index，选择最长边不超过上限的首个 mip，并累计它与所有更粗 mip 的 BC 数据。包含边缘压缩块；**不含 Vulkan allocation 对齐、descriptor、环境图或 staging**，也未裁掉 cfg 跳过对象的纹理。

| 全局最长边上限 | 全场景 BC mip-tail payload |
| --- | ---: |
| 128 | 0.085 GiB |
| 256 | 0.340 GiB |
| 512 | 1.358 GiB |
| 1024 | 5.406 GiB |
| 2048 | 21.531 GiB |
| 全分辨率 | 85.024 GiB |

当前设备确认是 RTX 5070 Ti / 16303 MiB。建议 Z5 从 **512 cap** 开始，设置 **2 GiB 纹理资源预算**并把实际 allocation 纳入准入；Z6 初始纹理预算再放到 **4 GiB**，按视图逐张细化，保留至少 256/更粗 mip 作稳定保底。几何/CLAS/BLAS 先沿用已有默认 512/256/256 MiB 做压力验证，再根据 Full 根级集合和可见工作集调节，不能预先保证这些预算足够。

整卡需给 RT/拓扑、帧资源、DLSS、纹理切换双份驻留和桌面留余量，建议初始目标至少 2 GiB 余量，并以实际可用显存动态准入。本轮读取时整卡已有 13824 MiB 使用量；它不是 Full 的占用或可比较基准，正式测量需先恢复独占渲染条件。

Z6 首选 **普通 mip streaming**：按纹理的期望 mip 管理 KTX2 尾链，异步读盘/解压/上传，使用 image 替换或受控复用并正确退役。只上传低 mip 到一张仍按完整尺寸分配的普通 VkImage，不会达到节省物理显存的目标。

若普通 mip streaming 实测仍因大 UDIM tile 的局部近景需求浪费大量显存，再升级 tile virtual texturing / sparse image；这需要新的离线 tile cache、边框和页表。KTX2 的独立 mip 压缩提供 mip 级随机读取，并不直接提供任意 tile 的随机解压。

vk_lod_clusters 当前纹理在初始化加载，按 4 GiB 默认预算轮流丢弃细 mip；不是视图驱动的纹理流送。它适合作为 Z3/Z5 的受预算首帧参考。[本机参考实现](E:/vk_lod_clusters/src/scene_textures.cpp:175)。Nanite 的几何虚拟化与虚拟纹理是独立系统，Epic 推荐一起使用，但不要求先实现虚拟纹理才能渲染 Nanite 几何。[Epic 官方说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine)

## 5. 同条件验证与 profiler 证据

固定 Full 自己的 cfg 与相机：eye `(7.729439, 2.278318, -11.146186)`，center `(-0.484482, 4.163491, -8.020232)`，FOV 60°、near/far 0.02/29999.998。`.scene.json` 是另一套相机/HDRI 配置，不能与 cfg 混为同一基准。

cfg 还启用属性 7、多材质、法线/UV 简化权重 0.5、材质权重 32、adaptive error、geometry/CLAS 各 3096 MiB、pathtrace+DLSS，并跳过两类 kite mesh（本机审计为 52 mesh instances）。这些是参考程序配置；Metallic 必须记录可映射项、未支持项和差异，不直接照搬全部数值。尤其不能把 1.5 px 与参考 adaptive error 后的质量、PT+DLSS 与实时光栅的 FPS 直接相除。

沿用既有 cfg replay 的前进—停留—返回—停留脚本框架，替换为 Full cfg，并增加近景纹理检查点。先验证路线没有穿墙再冻结路线文件和 hash。运行冷缓存、热缓存、正常预算、小预算四组，分开“不带 RT 的几何/材质”和“完整实时图”两条基线。

新增观测项应接入现有可折叠 Streaming 面板：

- 纹理：物理分配/有效 mip payload/保底/冷缓存/退役/staging 字节；期望 mip 与 resident mip 差、屏幕未收敛比例。
- 延迟：请求→IO→Zstd→上传→GPU 可采样；完成请求延迟与未完成请求年龄分开，P50/P95/P99 与样本数一起报告。
- 流量：读盘、解压、上传 MiB/s，预算推迟原因，回看命中率、重复提升/降低次数。
- 几何/RT：全属性页字节，LOD 误差、缺页、CLAS/BLAS fallback 屏幕占比；MASK coverage 不一致计数。
- CPU/GPU：texture demand、decode、upload、material resolve、alpha raster/shadow、总图区间；并发 scope 不简单相加。报告首个完整粗级帧时间，以及视图几何和纹理各自的收敛时间。

下一阶段推进 **Z5 全量 cook 与首帧**，沿用 Z2 属性格式、Z3 的 512 cap / 2 GiB 纹理预算和 Z4 实时着色链路。先验证完整根级工作集的几何/CLAS/纹理总驻留，再建立 Full cfg 相机的首帧证据；小探针不替代完整场景或持续漫游验收。
