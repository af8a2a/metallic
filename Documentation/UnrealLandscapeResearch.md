# Unreal Engine 5.6 地形系统调研报告

> 目标：为 Metallic 的 GPU-Driven 地形系统设计提供 UE5.6 参考实现分析。
> 调研对象：`E:\UnrealEngine`（UE 5.6.1，BranchName "UE5"）。所有 `file:line` 引用均以该源码树为准。
>
> **后续决策已更新**：经与 Metallic 现有 meshlet/mesh shader 管线比对，最终方案改为"地形以 cluster
> 进入现有 GPU-Driven 管线（Nanite-like）"，不再单独复刻 tile culling 路径。详见
> `Documentation/TerrainMeshShaderFeasibility.md`（本报告 §9 的路线已由该文档取代）。

## 1. 执行摘要

UE 5.6 的地形（Landscape）有 **三条并存的渲染路径**，对一个从零实现的 GPU-Driven 引擎而言，最值得借鉴的组合是：

1. **数据模型**：地形源数据是"每组件一张 BGRA8 纹理"（高度 uint16 存 R/G、法线存 B/A），完整 mip 链同时充当 LOD、碰撞源与流式单元。权重每纹素 4 层、每组件多张权重纹理。这个模型与 GPU 高度契合。
2. **非 Nanite 渲染路径（经典路径）**：顶点缓冲只有 4 字节网格坐标（x, y, subX, subY），顶点位置在 VS 里从高度纹理逐 mip 采样重构；LOD 是 CPU 算的连续值，shader 内做双 mip 插值 morph。每个组件每 LOD 一次 draw。
3. **5.6 新增的 Landscape GPU Culling（重点参考）**：把 LOD0 组件切成 4×4 quad 的 tile，compute pass 逐 tile 视锥剔除 + InterlockedAdd 压缩可见实例 + indirect draw。这是官方"非 Nanite 地形走向 GPU-Driven"的样板，与 Metallic 现有 GPUDriven 管线结构高度相似。
4. **Nanite 地形**：**不是运行时程序化位移**，而是在编辑器里把高度场按指定 LOD 导出为带裙边的静态网格，走标准 Nanite 烘焙管线（DDC 缓存、异步重建），材质按"每组件一个 material slot"评估。运行时只是一个启用了 Nanite 的 UStaticMesh。
5. **Virtual Heightfield Mesh（VHM）**：真正"无网格、运行时位移"的 GPU 地形：把高度场渲染进运行时虚拟纹理页，以 VT 页表为四叉树做 GPU LOD，VS 逐顶点采样高度页位移（注意：在 5.6 中 VHM 是经典光栅 + VS 位移，**不是** Nanite）。需要完整的 VT 系统支撑。
6. **UE 5.6 已删除 `MD_Landscape` 材质域**：地形材质现在是普通 Surface 域材质（`MaterialDomain.h` 枚举中已无 Landscape），地形专用节点在 Surface 域下编译。这对自研材质系统是个好消息：不需要专门的地形材质域，只需要"带层混合的材质"。

**建议的 Metallic 路线**（详见第 9 节）：先做"经典路径 + 5.6 tile culling"的 GPU-Driven 地形（复用 GPUScene 实例/网格记录与 HZB 遮挡剔除），数据用每组件高度纹理 + mip LOD + shader morph；虚拟纹理（权重层、RVT）作为第二阶段；"VHM 式光栅期位移"作为远期目标，等 Metallic 的 meshlet 光栅管线成熟后再做。

---

## 2. 数据模型（源数据、层级、格式）

### 2.1 Actor / 组件层级

```
ALandscape (ALandscapeProxy 子类, Landscape.h:250)
 └─ 非 WP 时直接拥有全部组件
 └─ World Partition 下: ALandscapeStreamingProxy (每个空间 cell 一个, LandscapeStreamingProxy.h:18)
     └─ TSoftObjectPtr<ALandscape> LandscapeActorRef (共享的全局父 actor)
     └─ ULandscapeComponent (Within=LandscapeProxy, LandscapeComponent.h:413)
         └─ NumSubsections × NumSubsections 个 subsection (1×1 或 2×2)
```

- `SectionBaseX/Y`：组件在全局网格中的 quad 坐标（LandscapeComponent.h:419-424）。
- 合法尺寸由 `FLandscapeConfig` 强制：`NumSubsections ∈ {1,2}`，`SubsectionSizeQuads ∈ {7,15,31,63,127,255}`（LandscapeConfigHelper.cpp:23-24）。组件边长 = `(SubsectionSizeQuads+1)*NumSubsections` quads。
- 最大 LOD = `CeilLog2(SubsectionSizeQuads+1) - 1`（Landscape.cpp:707）。
- 上限：`MaxComponents = 256`/边（LandscapeSettings.h:87），编辑器允许 8192 顶点/边。

### 2.2 高度数据

- **每组件一张 `UTexture2D`**（`HeightmapTexture`，LandscapeComponent.h:543），BGRA8、非 sRGB、无压缩，尺寸 = `(SubsectionSizeQuads+1)*NumSubsections`。
- **像素编码**（LandscapeDataAccess.h:26-59）：
  - 高度 uint16：R = 高字节、G = 低字节；`GetLocalHeight(H) = (H - 32768) * (1/128)` 世界单位（`LANDSCAPE_ZSCALE = 1/128`，即 128 级/米，MidValue=32768，MaxValue=65535）。
  - 法线：B = X、A = Y（Z 由单位化重构）。
  - XY 偏移纹理（可选）：scale 1/256（LANDSCAPE_XYOFFSET_SCALE）。
- **mip 链即 LOD**：`mip = floor(log2(size))+1`，最后 1~2 个退化 mip 不算"相关 mip"（Landscape.cpp:1970-1979）。渲染时 LOD 直接对应高度纹理 mip。
- **每编辑层（edit layer）另有自己的 per-component 高度纹理**（`LayersData: TMap<FGuid, FLandscapeLayerComponentData>`，LandscapeComponent.h:520-522），最终数据是各层 GPU 合并结果。
- **洞（holes）**：特殊权重层 `__LANDSCAPE_VISIBILITY__`，采样阈值 `2/3`（`LANDSCAPE_VISIBILITY_THRESHOLD`，LandscapeDataAccess.h:19）；Marching-squares 网格化（LandscapeEdit.cpp:4615）；同一通道同时驱动物理碰撞的 per-quad flags。

### 2.3 权重数据（材质层）

- 每组件 `WeightmapTextures[]` + `WeightmapLayerAllocations[]`（LandscapeComponent.h:548-553）。
- **4 层/纹素**（BGRA 四通道，`ULandscapeWeightmapUsage::NumChannels=4`），每组件可有多张权重纹理；`FWeightmapLayerAllocationInfo` = (纹理索引, 通道)（LandscapeComponent.h:140-187）。
- 编辑层上限 8（clamp 32，LandscapeSettings.h:80）。
- 渲染期由材质实例绑定为材质纹理参数；`LODIndexToMaterialIndex` 让不同 LOD 可用不同材质（LandscapeComponent.h:462-464）。

### 2.4 编辑管线（GPU Render-to-Texture）

- 雕刻/绘制只改**激活编辑层**的 per-component 纹理；最终数据 = 各层 GPU 合并（`RegenerateLayersHeightmaps/Weightmaps`，LandscapeEditLayers.cpp:2479-2491）。
- 合并是纯 RDG 管线：每组件 scratch 纹理 → `MergeEditLayersPS` 合成 → `StitchHeightmapPS`（用 9 邻居修补 2×2 subsection 边缘）→ `FinalizeHeightmapPS` 写回 → `GenerateMipsPS`（LandscapeEditLayers.cpp:4869-5116）。
- 异步 readback：staging 纹理 + GPU fence + `MapStagingSurface`（LandscapeEditReadback.cpp:52-181）。
- 批次上限：每批 ≤16 组件、≤1024 px（LandscapeEditLayers.cpp:299-315）。
- **对自研引擎的直接启示**：GPU 地形编辑 = "每组件渲染目标 + 邻居边缘缝合 + mip 重生成 + 异步回读"，无需 CPU 中间表示。

### 2.5 碰撞

- 每个渲染组件对应一个 `ULandscapeHeightfieldCollisionComponent`（1:1）。
- 高度采样自高度纹理的 **`CollisionMipLevel` mip**；`CollisionSizeQuads = (SubsectionSizeQuads >> CollisionMipLevel) * NumSubsections`（LandscapeEdit.cpp:1780-1782）。
- UE5.6 用 **Chaos**：`Chaos::FHeightField(Heights, MaterialIds, CollisionSizeVerts², FVec3(1))`（LandscapeCollision.cpp:1105），shape 缩放 `(Scale.X*CollisionScale, Scale.Y*CollisionScale, Scale.Z*LANDSCAPE_ZSCALE)`（382, 430-431）。
- per-quad flags：`QF_NoCollision=128`（洞）、`QF_PhysicalMaterialMask=63`、`QF_EdgeTurned=64`（对角线翻转，LandscapeHeightfieldCollisionComponent.h:180-185）。
- 运行时允许异步创建物理状态，编辑期同步（LandscapeCollision.cpp:348-355）。

### 2.6 流式与内存

- 高度/权重纹理走标准 UTexture2D mip 流式；5.6 新增 `LandscapeTextureStorageProvider`：高度纹理可改存压缩格式，流式 mip 到达时用邻居快照**修补边缘纹素**避免低 mip 接缝（LandscapeTextureStorageProvider.cpp:27-98）。
- World Partition 下按 proxy cell 粒度加载（`FLandscapeActorDesc`，WorldPartition\Landscape\LandscapeActorDesc.cpp:20-26）；`ULandscapeSubsystem` 注册/跟踪组件与移动的 streaming proxy（LandscapeSubsystem.cpp:335-601）。
- `ULandscapeInfo` 维护全局 `XYtoComponentMap`（邻居查询、层信息共享）。

---

## 3. 经典（非 Nanite）地形渲染路径

### 3.1 渲染组织

- **一个渲染 section = 一个组件**：每个组件每 LOD 一条 mesh batch 一次 draw，覆盖所有 subsection（LandscapeRender.cpp:2449-2507）。
- **共享顶点/索引缓冲（按几何尺寸全局去重）**：
  - Key = `(SubsectionSizeLog2) | (NumSubsections<<4) | (NumRayTracingSections<<8) | XYOffset位`（LandscapeRender.cpp:1691-1692），进程级 `SharedBuffersMap` 引用计数（LandscapeRender.h:754-757）。
  - 顶点缓冲：`FLandscapeVertex { uint8 VertexX, VertexY, SubX, SubY }` —— **每顶点仅 4 字节网格坐标**（LandscapeRender.h:319-325），`NumVertices = (SubsectionSizeVerts)² * NumSubsections²`（LandscapeRender.cpp:3511）。
  - 索引缓冲：**每个 LOD mip 一份**（`CeilLog2(SubsectionSizeQuads+1)` 份），含 per-subsection min/max 索引范围（LandscapeRender.cpp:3320-3442）。
- **顶点工厂**：普通 / XYOffset / FixedGrid（RVT、水面、Lumen、草地用，无 morph）/ Tile（5.6 culling 用）四种，共用同一顶点缓冲。

### 3.2 位置重构与 morph

- 顶点位置在 VS 里从高度纹理重构：`GetLocalPosition = LocalPosition + InputPosition.zw * SubsectionOffsetParams.ww`（LandscapeVertexFactory.ush:333-336），高度/法线来自高度纹理采样。
- **连续 LOD morph**：CPU 给每 section 算连续 LOD 值；VS 中 `LodValue = floor(LOD)`，`MorphAlpha = frac(CalcLOD)`，高度/法线在 mip LOD 与 LOD+1 之间 lerp（LandscapeVertexFactory.ush:641-701）。XY 偏移纹理同样双 mip 采样。
- **不同 LOD 邻居的接缝处理**：不做几何缝合，而是在 section 边缘用 `EdgeLOD = max(CenterLOD, NeighborLOD)`（LandscapeVertexFactory.ush:102-122），两侧边缘 quad 收敛到同一 LOD，配合连续 morph 消除裂纹。

### 3.3 LOD 选择公式（CPU，每 view × section 一次）

- `SectionScreenSizeSquared = ComputeBoundsScreenRadiusSquared(包围球) / LODScale²`（LandscapeRender.cpp:4566-4567）。
- 阈值表由 `LOD0ScreenSize`（默认 1.0，可用 Scalability 档位）、`LOD0Distribution`、`LODDistribution`（默认 2.0 几何级数）预计算（LandscapeRender.cpp:1531-1553）。
- 分段映射（`ComputeLODFromScreenSize`，LandscapeRender.cpp:544-559）：
  - `screenSize² ≤ LastLOD²` → LastLOD；
  - `> LOD1²` → LOD0→1 线性插值；
  - 其余 → `1 + LogX(distribution², LOD1²/screenSize²)`。
- 流式偏置：`ComputeLODBias()` = 高度纹理当前已驻留 mip（LandscapeRender.cpp:4574-4590），上传为 `PF_R32_FLOAT` SRV，shader 里作为 SampleLevel 偏移。
- 结果经 task graph 并行计算后打包进 per-view SRV + 间接表，绑定进 view uniform buffer（LandscapeRender.cpp:1218-1286；SceneRendering.cpp:2161-2170）。

### 3.4 绘制结构

- 每组件静态批次：`(LastLOD-FirstLOD+1)` 个 LOD batch（LOD0 无限屏幕尺寸，其余按 `sqrt(LODScreenRatioSquared[LOD])*2` 参与缓存 draw list 的屏幕尺寸剔除）+ RVT fixed-grid 批次 + 水面信息批次 + Lumen capture 批次（DrawStaticElements，LandscapeRender.cpp:2509-2603）。
- 动态批次路径（编辑器工具、debug 视图等）每帧按 `RenderSystem.GetSectionLODValue(view, coord)` 选 LOD（LandscapeRender.cpp:2631-3056）。
- 纹理绑定：高度/法线/参数在一个 `FLandscapeUniformShaderParameters` UB（LandscapeRender.h:116-142）；权重纹理由材质实例绑定。
- 洞：`1 - StaticTerrainLayerWeight(VisibilityLayer)` 编译进 OpacityMask（MaterialExpressionLandscapeVisibilityMask.cpp:42-47）。

---

## 4. UE 5.6 新增：Landscape GPU Culling（Tile Culling 路径）

> 这是本调研中**与 Metallic GPU-Driven 管线最直接对应的官方实现**，建议重点复刻。
> 代码：`Runtime/Landscape/Private/LandscapeCulling.cpp`（858 行）+ `Shaders/Private/Landscape/LandscapeCulling.usf`。

### 4.1 设计

- 仅对 **LOD0** 生效（`SetupMeshBatch` 对 LOD≠0 直接返回，LandscapeCulling.cpp:520）；把每个组件切成 **4×4 quad 的 tile**，用 compute 逐 tile 视锥剔除，通过 indirect 实例化 draw 绘制可见 tile。
- 一个 tile = 5×5 顶点共享网格（`FLandscapeTileMesh`，LandscapeCulling.cpp:227-298）；每 tile 一条 `ubyte4(QuadX, QuadY, SubX, SubY)` 实例数据（`FLandscapeTileDataBuffer`，323-357）。
- CVar：`landscape.SupportGPUCulling`（只读，平台编译期）、`landscape.EnableGPUCulling` / `.EnableGPUCullingShadows`（运行时可切，13-29）。
- **不启用条件**：VSM 或 Lumen GI 激活时（需要 VF PrimitiveID 支持，LandscapeCulling.cpp:171-182）。

### 4.2 每帧流程

1. `PreRenderViewFamily` 重置缓存状态（484-487）；view extension 收集视图并发起 LOD 计算 task（LandscapeRender.cpp:1199-1288）。
2. `ComputeSectionIntermediateData`：对每个 LOD0 section 上传 `FLandscapeSection`（LWC 矩阵、TilePosition、LocalZ/HalfHeight、**NeighborLODExtent = max(1, (1<<邻居最大LOD)-1)**）（LandscapeCulling.cpp:606-663）。
3. `DispatchCulling`：`FBuildLandscapeTileDataCS`，dispatch 维度 = (每 section 的 tile 数, ×views×sections)；每线程一个 tile：
   - tile AABB 中心/范围由 tile 坐标与 section 高度范围构成，**按邻居 LOD 膨胀范围**（避免与高 LOD 邻居间出现 pop 裂纹）；
   - `BoxCullFrustum`（标准视锥剔除，LandscapeCulling.usf:74；文件虽 include 了 Nanite 的 HZB 剔除 helper，但当前未做 HZB 遮挡测试）；
   - 可见 → `InterlockedAdd(IndirectArgs[z].InstanceCount, 1)` 压缩输出 tile 实例数据（LandscapeCulling.usf:44-91）。
4. 渲染器在可见 mesh 命令收集时 `ApplyViewDependentMeshArguments` → 用缓存好的 indirect args 替换 batch 的 instance 数/args（LandscapeRender.cpp:3058-3069；LandscapeCulling.cpp:829-856）。
5. 阴影视图同流程（`InitShadowViews`，815-827）。
6. 最终 LOD0 = **一次 indirect 实例化 draw**；LOD>0 回落到经典"每组件一 draw"。

### 4.3 对自研引擎的要点

- 顶点缓冲只存网格坐标 + 高度在 VS 采样 → 地形几何数据量趋近于零，GPU 内存只花在高度纹理上。
- Tile 级 GPU 剔除 + atomic 压缩 + indirect args 是 GPU-Driven 的完整最小闭环，无需 BVH/meshlet。
- 与 Nanite 不同，tile 路径**没有 HZB 遮挡剔除**（视锥 only），因为地形作为 occluder 通常先渲染；在 Metallic 中可以接入现有 HZB 做 second-phase occlusion。

---

## 5. Nanite 地形（UE 5.6）

> 代码：`Runtime/Landscape/Private/LandscapeNaniteComponent.cpp`（717 行）、`LandscapeEdit.cpp:4192-4252, 4289+`（导出）、`Landscape.cpp:7160-7204`（运行时切换）。

### 5.1 核心结论：Nanite 地形是"烘焙静态网格"，不是运行时程序化位移

- 编辑器里把高度场（按 `NaniteLODIndex` 指定的 mip/LOD）**导出为带裙边的 MeshDescription**，喂给标准 `NaniteBuilder` 生成 Nanite 资源；结果缓存在 DDC，内容变更时异步重建（LandscapeNaniteComponent.cpp:235-612）。
- 运行时它就是一个 **启用了 Nanite 的 `UStaticMeshComponent`**（`ULandscapeNaniteComponent`），渲染管线对它没有任何地形专用代码。
- 证据：`Renderer/Private/Nanite/` 全目录搜 "Landscape" 仅命中 `NaniteCullRaster.cpp:3972-3974` —— 一个 `EFilterFlags::Landscape` 显示标志（对应 Show>Landscape 开关，`NaniteSceneProxy.h:178-185`）。Nanite 剔除/光栅/着色对地形完全无感知。
- 与 §6.5 VHM（真·光栅期位移）对比：UE 官方没有把"逐像素高度位移"做进 Nanite 地形；想要运行时可编辑的 GPU 地形，VHM 才是参考。

### 5.2 编辑器导出管线

1. **数据准备**（`MakeAsyncNaniteBuildData`，LandscapeEdit.cpp:4192-4252）：按导出 LOD 读高度纹理 mip 到 CPU 数组（`HeightAndNormalData`）+ 洞层数据（`Visibility`）；每个组件取 `GetMaterialInstance(0)` 作为 material slot（默认 `MD_Surface` 域材质 —— 5.6 已删除 Landscape 材质域）；材质数上限 `NANITE_MAX_CLUSTER_MATERIALS`。
2. **网格导出**（`ExportToRawMeshDataCopy`，LandscapeEdit.cpp:4289+）：
   - 一个 polygon group = 一个组件（= 一个 material slot）；
   - 每组件按导出 LOD 的网格三角形化，顶点位置 = 高度数据 + 组件变换（相对 proxy）；
   - **裙边**：无邻居组件（或不在导出集合内）的边缘加 1 quad 裙边、按 `SkirtDepth` 向下挤出（LandscapeEdit.cpp:4383-4399, 4494-4509）；
   - **洞**：可见度阈值 170（≈255×2/3）以下的 quad 被移除（LandscapeEdit.cpp:4484-4485）；
   - **UV**：4 组 —— XY 地形坐标、XZ、lightmap UV、权重图 UV（per-component scale/bias，LandscapeNaniteComponent.cpp:337-340）；
   - NaniteSettings：`PositionPrecision = Log2(scale max) + NanitePositionPrecision`、`MaxEdgeLengthFactor`；`FallbackPercentTriangles = 0.01`（几乎不保留回退网格）、`bRecomputeNormals/Tangents = false`（LandscapeNaniteComponent.cpp:307-322）。
3. **异步构建**：DDC 缓存导出结果（按 ProxyContentId+组件集合哈希，LandscapeNaniteComponent.cpp:284-305）；`BatchBuild` → `NaniteBuilder`；带 stall 检测（LandscapeNaniteStallDetectionTimeout）与取消（内容 ID 变化时丢弃过期构建，538-545）。
4. **内容变更**：`NaniteContentId` 变化 → `InvalidateOrUpdateNaniteRepresentation` → 后台重建（可选 live rebuild：延迟 `LandscapeNaniteBuildLag` 秒触发，Landscape.cpp:7134-7141）。
5. **代理拆分**：一个 `ALandscapeProxy` 可拆成多个 Nanite 组件/网格（`NaniteComponents[]`），组件按 section base 分组（LandscapeNaniteComponent.cpp:297-305）。

### 5.3 运行时路径

- `ULandscapeNaniteComponent` 构造时 `bVisibleInRayTracing=false`、`bEvaluateWorldPositionOffset=false`（LandscapeNaniteComponent.cpp:111-115）；NaniteRT（Mega Geometry）下例外允许光线追踪（154-160）。
- 碰撞**不来自 Nanite 网格**：由 `ULandscapeHeightfieldCollisionComponent`（Chaos heightfield，§2.5）承担；Nanite 网格 BodySetup 标记 `NoCollision`、不烘焙碰撞数据（LandscapeNaniteComponent.cpp:551-558）。
- `UpdateRenderingMethod`（Landscape.cpp:7160-7204）逐组件切换 Nanite/经典路径：`CVarRenderNaniteLandscape && HasNaniteComponents() && UseNanite(平台) && (编辑器下内容 ID 匹配，避免渲染过期网格) && AuditNaniteMaterials()` → `Component->SetNaniteActive(...)`。
- 材质：每组件一个 surface 域材质实例（地形专用节点如 LandscapeLayerBlend/Weight 编译为普通纹理采样 + 混合表达式）；权重/高度纹理经标准材质纹理参数绑定，经 Nanite 材质管线评估。
- 流式：Nanite 网格数据走 DDC + 平台缓存（`BeginCacheForCookedPlatformData`，LandscapeNaniteComponent.cpp:681-715）；运行时页流式与普通 Nanite 网格一致。

### 5.4 对自研引擎的启示

- **不推荐直接复刻**：烘焙静态网格方案牺牲了地形最大优势（极小内存 + 运行时编辑 + 流式粒度），且导出/重建开销大。
- 真正值得借鉴的是其**材质侧简化**：地形材质 = surface 域材质 + 层混合（采样权重 VT），无需专用材质域。
- "Nanite 式地形"的正确打开方式是 VHM 模式（§6.5）：把高度场当虚拟纹理、页表当四叉树，VS 逐顶点位移 —— 若未来做进 meshlet 光栅器，则变为光栅期逐像素位移。

---

## 6. 运行时虚拟纹理（RVT）与 Virtual Heightfield Mesh（VHM）

> 代码：`Runtime/Renderer/Private/VT/`、`Runtime/Engine/Classes/Components/RuntimeVirtualTextureComponent.h`、
> `Plugins/Experimental/VirtualHeightfieldMesh/`。

### 6.1 VT 系统架构（Renderer/Private/VT）

- **三个核心对象**：`FVirtualTextureSystem`（单例）→ 最多 16 个 `FVirtualTextureSpace`（页表）+ 若干 `FVirtualTexturePhysicalSpace`（物理图集）+ `FVirtualTextureProducerCollection`（生产者）。每帧流程 `BeginUpdate → 异步 gather → EndUpdate → FinalizeRequests`（VirtualTextureSystem.h:105-308）。
- **Space = 页表**：页表是 2D 纹理，**每个虚拟 mip 一级**（VirtualTextureSpace.cpp:304-318），每纹理存 4 层（LayersPerPageTableTexture=4）；`EVTPageTableFormat` UInt16/UInt8，16 位格式每 tile 坐标仅 6 bit → 物理池 ≤64×64 tile（VirtualTexturePhysicalSpace.h:101-102）；自适应页表用 PageTableIndirection 间接寻址。
- **Physical space = 图集**：池大小由 MB 预算反推（默认 64 MB，VirtualTexturePoolConfig.h:64），除以每 tile 字节后开方得边长（VirtualTextureSystem.cpp:958-1026）；页尺寸含边框（UE 默认 border 4px）。
- **LRU 页池**：`FTexturePagePool`（TexturePagePool.h:23-248）用二叉堆做空闲 LRU 表；每帧 `UpdateUsage` 保护本帧用过的页；`Lock()` 页永不淘汰；池过载时提升 **residency mip bias** 强制请求更粗 mip（VirtualTexturePhysicalSpace.cpp:242-286），池可自动增长（GrowPhysicalPools，VirtualTextureSystem.cpp:2045-2079）。

### 6.2 Feedback 闭环（最小 VT 系统的核心）

1. **GPU 写请求**：每 N×N 像素一个采样（抖动），请求打包 `ID<<28 | (vLevel+1)<<24 | vPageY<<12 | vPageX`（VirtualTextureSystem.cpp:222-232；vLevel+1 让 CPU 识别"想要比现有更细"）。
2. **GPU 压缩**：哈希表去重为 `(page, count)` 对 + 原子计数器头 + indirect dispatch（VirtualTextureFeedbackResource.cpp:127-245）。
3. **回读**：8 个 staging 缓冲环形队列 + GPU fence（VirtualTextureFeedback.cpp:109-146）。
4. **CPU 分析**：`FeedbackAnalysisTask` 多线程合并唯一页列表 → `GatherRequestsTask` 解码（Morton 码），驻留页只刷新 LRU，缺失页生成加载请求 → 按 64 位优先级排序（Locked > Streaming > ProducerPriority > InvalidatePriority > PagePriority）→ **按每帧预算节流**（SubmitThrottledRequests，VirtualTextureSystem.cpp:2221-2284）。
5. **Mip 计算**（shader 侧）：`MipLevelAniso2D(dUVdx, dUVdy)` + 材质 bias + 全局 bias + 随机抖动；三线性时请求 floor/ceil 两级，跨帧交错采样（VirtualTextureCommon.ush:259-287, 353-358）。

### 6.3 页更新执行（把场景渲染进页）

- `SubmitRequests`：`IVirtualTexture::RequestPageData` → 每物理组分配池页 → 构造 `FVTProduceTargetLayer`（物理 RT + pPageLocation）→ `ProducePageData` → `FinalizeRequests` 里 `RenderFinalize` 后更新页表 + 纹理转回 SRV（VirtualTextureSystem.cpp:2286-2641）。
- **RVT 每页渲染**（RuntimeVirtualTextureRender.cpp:2034-2134）：每页建一个临时正交相机（相机中心 = 页 UV 中心，正交宽度 = 页尺寸+边框×2），`View->bIsVirtualTexture=true`，viewport 按页设置；`GatherMeshesToDraw` 收集与该 RVT 关联的 primitive（球-视锥剔除、按 mip/像素覆盖率剔除、按屏幕面积选 LOD），只画 `bRenderToVirtualTexture && RuntimeVirtualTextureMaterialType 匹配` 的 batch（:1573-1676）。页渲染进数组切片 RT → BC 压缩 compute → blit 进物理图集（CopyPagesToOutput）。
- **接缝处理**：物理 UV = `(pPageX,pPageY)*pPageSize + frac(vUV)*vPageSize + vPageBorderSize`（VirtualTextureCommon.ush:789-807），额外边框纹素让双线性/各向异性采样安全跨页。

### 6.4 RuntimeVirtualTextureComponent（场景 → VT）

- 体积 = 组件变换决定的 AABB（支持 `bSnapBoundsToLandscape` 吸附地形、`ExpandBounds`）。
- 默认 256px/页、256×256 tile 页表（65536² 虚拟纹理）、4px 边框（RuntimeVirtualTexture.h:29-38）。
- 材质按 `ERuntimeVirtualTextureMaterialType`（BaseColor/Normal/Roughness/Specular/Mask4/**WorldHeight**/Displacement）输出打包（VirtualTextureMaterial.usf:65-141）。
- **地形向 RVT 的渲染**：§3.4 的 fixed-grid 批次（每材质类型 × 每 LOD 一条 `bRenderToVirtualTexture` batch；`landscape.RuntimeVirtualTextureRenderWithQuad` 开启"单 quad + 逐像素高度位移"模式，LandscapeRender.cpp:230-233, 2400-2543）。
- **材质采样 RVT**：`RuntimeVirtualTextureSample` 节点编译出页表纹理 + `VirtualTextureWorldToUV` 平面基（HLSLMaterialTranslator.cpp:8928-8951），采样链 = 页表查询（带 feedback 写入）→ 物理 UV → 纹理采样，未映射页用 FallbackValue（VirtualTextureCommon.ush:782-856）。
- 地形编辑会失效对应 RVT 区域（Landscape.cpp:5401-5437）。

### 6.5 VirtualHeightfieldMesh（真·无网格 GPU 地形，重点参考）

- **不存任何网格**：`UVirtualHeightfieldMeshComponent` 只引用 RVT 体积（必须 WorldHeight 类型）、`UHeightfieldMinMaxTexture`、着色材质（VirtualHeightfieldMeshComponent.cpp:83-90）。
- **数据路径**：
  1. 高度数据 = RVT 的 WorldHeight 页；编辑器里把 WorldHeight 页做 min/max 降采样生成 `HeightMinMaxTexture`（供裁剪/LOD 用的保守界）与 LodBias 纹理（HeightfieldMinMaxTextureBuild.cpp:169-231）。
  2. 每帧 `AddPass_CollectQuads`：**persistent-wavefront compute** 遍历页表四叉树 —— 采样 min/max 高度 → 距离 LOD → 细分或发射 quad render item → **同时为 VS 可能读到的页写 VT feedback**（VirtualHeightfieldMesh.usf:219-350）。这就是"以 VT 页为四叉树节点"的 GPU LOD。
  3. `CullInstancesCS` 按视图剔除并 append `QuadRenderInstance`。
  4. 一次 `DrawInstancedIndirect`：**无顶点缓冲的自定义顶点工厂**，网格顶点由 `SV_VertexID` 生成，实例数据含 UV rect + LOD，VS 经页表两次采样（floor/ceil mip）高度并 lerp（morph），位置 = `mul(float4(NormalizedPos, Height, 1), VirtualHeightfieldToWorld)`（VirtualHeightfieldMeshVertexFactory.ush:73-149）。
- **在 5.6 中 VHM 不是 Nanite**：插件里没有任何 Nanite 代码；VHM 是经典光栅 + VS 位移。Nanite 自己的位移是独立材质路径（NaniteRasterizationCommon.ush:376-454 `ApplyFallbackDisplacement`）。
- **对自研引擎的启示**：VHM 是"光栅期位移地形"的完整参考实现，其核心循环（cluster/quad 采样高度页 → 写 feedback → VT 系统流式供页 → min/max mip 金字塔提供保守裁剪界）可以平移进任何 Nanite-like 管线。

### 6.6 自研最小 VT 系统清单

1. Feedback：子采样（N×N 像素 1 请求、逐帧抖动）+ GPU 去重压缩 + fence 守护回读 + 2~3 帧延迟。
2. 页表：每虚拟 mip 一级的 2D 纹理（或数组），texel = 打包物理页坐标；16 位格式限 64×64 tile 池。
3. 物理图集：LRU 页池 + lock 位 + 每帧使用标记；页尺寸含边框，shader 必须偏移进边框采样。
4. Mip 公式：`0.5*log2(max(dot(dUVdx), dot(dUVdy)))` + 各向异性钳制 + bias。
5. 节流：每帧页上传/页生产预算 + 优先级排序。
6. Producer 接口：RequestPageData → 分配池页 → GPU 渲染/上传 → finalize（页表更新 + 纹理转换）。
7. 典型规模：256px 页、64MB 池 ≈ 4096²(无压缩)~8192²(BCn) 物理图集。

---

## 7. 草地与其它周边系统

- **草地有两种机制**（LandscapeGrass.cpp）：
  1. **Grass map 烘焙**：把组件完整顶点网格当作 `PT_PointList` 用 FixedGrid VF 渲染，正交投影渲染到条带 RT（LandscapeGrassWeightExporter.cpp:439-486），异步读回每草种权重 + 每 mip 高度到 `FLandscapeComponentGrassData`。状态机 `FLandscapeGrassMapsBuilder`（Pending→TextureStreaming→Rendering→AsyncFetch→Populated）。
  2. **实例化**：`ALandscapeProxy::UpdateGrass` 按距离排序组件、按 `MaxInstanceDiscardDistance` 剔除，向 `UHierarchicalInstancedStaticMeshComponent`（HISM）喂实例 —— 草地不是从地形缓冲 GPU 实例化，而是经典 HISM 程序化植被。
- **光照**：`FLandscapeLCI` + `FLightMap2D`；lightmap UV 由 `StaticLightingLOD` 与扩展 patch 计算（LandscapeRender.cpp:2299-2360）。
- **非 Nanite VSM 失效**：用 `WorldSpaceMipToMipMaxDeltas` 估计高度误差 + 屏幕尺寸衰减判定阴影缓存失效（LandscapeRender.cpp:4633-4712）。

---

## 8. 关键常数速查

| 项目 | 值 | 出处 |
|---|---|---|
| 高度格式 | uint16，R=高字节 G=低字节；法线 B/A | LandscapeDataAccess.h:38-59 |
| 高度量化 | ZScale=1/128（128 级/米），MidValue=32768 | LandscapeDataAccess.h:13-14,26-36 |
| XY 偏移量化 | 1/256 | LandscapeDataAccess.h:16 |
| 洞阈值 | 2/3 | LandscapeDataAccess.h:19 |
| 组件网格 | subsection quads {7,15,31,63,127,255} × subsections {1,2} | LandscapeConfigHelper.cpp:23-24 |
| 每组件纹理尺寸 | (SubsectionSizeQuads+1)*NumSubsections | Landscape.cpp:1972 |
| 最大 LOD | CeilLog2(SubsectionSizeQuads+1)-1 | Landscape.cpp:707 |
| 权重层 | 4 层/纹素，多张纹理/组件 | LandscapeWeightmapUsage.h:17 |
| 编辑层上限 | 8（clamp 32） | LandscapeSettings.h:80 |
| LOD 分布 | LOD0ScreenSize=1.0，distribution=2.0（几何级数） | LandscapeRender.cpp:1531-1553 |
| Tile 尺寸 | 4×4 quads（5×5 顶点） | LandscapeCulling.cpp:145 |
| Tile 路径最小 section | 31 quads | LandscapeCulling.cpp:146 |
| 碰撞 quad flags | 材质 mask 63 / 边翻转 64 / 无碰撞 128 | LandscapeHeightfieldCollisionComponent.h:180-185 |
| 编辑合并批次 | ≤16 组件、≤1024 px/批 | LandscapeEditLayers.cpp:299-315 |

---

## 9. 与 Metallic 现状的映射与实现建议

### 9.1 Metallic 现有能力盘点

- `Source/Runtime/Render/Subsystem/GPUScene.*`：几何/材质/实例 GPU 记录、draw key、bucket、meshlet LOD（GPUScene.h:80-147）。
- `Shaders/GPUDrivenDeferred.slang`：实例剔除 + HZB（hzbBuffer0/1）、meshlet 两阶段剔除 + visibility buffer、indirect draw、延迟着色（OpenPBR）。**已具备 GPU-Driven 基座**。
- `Source/Runtime/Render/MeshletStream*`：meshlet 流式页加载（Nanite 式流式基础）。
- RenderGraph（JSON 描述、typed pass）+ Vulkan RHI + Slang。

### 9.2 推荐实现路径

**阶段 0 —— 数据与流式（对齐 UE §2）**
- 地形数据：每 patch（对应 UE 组件）一张高度纹理（R16 或 R8G8 打包）+ 完整 mip 链；权重纹理（4 层/纹素）后续再加。
- 全局 patch 网格 + 按相机距离的 mip 驻留管理（对齐 UE 的 texture mip streaming + LOD bias）。Metallic 已有 MeshletStreamPageLoader，可复用其页加载模式。

**阶段 1 —— GPU-Driven 渲染（对齐 UE §3 + §4，最高性价比）**
- 复刻 5.6 tile culling：静态 5×5 顶点网格 + per-tile 实例数据（patch 坐标）+ compute 视锥剔除 + atomic 压缩 + indirect draw。
- 顶点/像素阶段从高度纹理采样重构位置（Metallic GPUDriven 管线需要加一个"地形材质/顶点工厂"概念；Slang 里直接写）。
- LOD：CPU 端按 §3.3 公式算每 patch 连续 LOD；shader 双 mip 采样 + morph（`EdgeLOD = max(CenterLOD, NeighborLOD)` 处理接缝）；LOD>0 的 patch 可以走"每 patch 一 draw"或同样 tile 化。
- 接入现有 HZB 做二次遮挡剔除（UE tile 路径没有这步，Metallic 可以做得更好）。
- 碰撞：复用物理侧 heightfield（Chaos 式），采样自高度纹理的碰撞 mip。

**阶段 2 —— 虚拟纹理**
- 权重层材质混合需要 VT：page table（间接寻址）+ 物理 atlas（128×128 页）+ feedback buffer + 异步页渲染（对齐 UE RVT，详见 §6）。
- 在此之前可以用"每 patch 直接绑定权重纹理"过渡（正是 UE 非 VT 模式的做法）。

**阶段 3 —— VHM 式光栅期位移（远期）**
- 高度场渲染进 VT 页 → meshlet 光栅器按材质域逐像素位移。Metallic 的 meshlet 流式管线成熟后再考虑；短期价值低于阶段 1+2。

### 9.3 决策要点对照

| 问题 | UE 5.6 的做法 | Metallic 建议 |
|---|---|---|
| 地形网格存哪 | 不存：4 字节网格坐标 VB，VS 采样高度纹理 | 同（最省显存） |
| LOD 与接缝 | 连续 LOD + 双 mip morph + EdgeLOD | 同（成熟且 GPU 友好） |
| Nanite 地形 | 编辑器烘焙成静态网格资产 | **不采用**——烘焙失去运行时编辑与流式粒度优势 |
| GPU 剔除 | 5.6 tile 路径：tile 级视锥 + indirect | tile 级视锥 + 现有 HZB 遮挡 |
| 材质层 | 权重纹理 + VT 采样 | 先权重纹理直绑，后 VT |
| 碰撞 | Chaos heightfield @ 碰撞 mip | 物理 heightfield @ 碰撞 mip（复用现有物理） |
| 编辑 | 每组件 RT + GPU 层合并 + 异步回读 | 阶段 1 之后按需引入 |
