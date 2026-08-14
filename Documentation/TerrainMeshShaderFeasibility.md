# GPU-Driven 地形 Mesh Shader 整合可行性分析（Nanite-like Terrain）

> 目标：地形不建独立 tile 渲染路径，而是以 **cluster（meshlet）为单元进入 Metallic 现有 GPU-Driven mesh shader 管线**，
> 与普通场景共用剔除 + 光栅 + 延迟着色 + 流式 + RT，实现 Nanite-like 地形。
> 前提阅读：`Documentation/UnrealLandscapeResearch.md`（UE 5.6 地形调研）。本文件所有 `file:line` 引用基于
> `E:\metallic` 当前源码（`Shaders/GPUDrivenPreview.slang`、`Shaders/GPUDrivenDeferred.slang`、
> `Shaders/GPUDrivenStreamAsset.slang`、`Source/Runtime/Render/MeshletStream*.{h,cpp}`、
> `Source/Runtime/Scene/MeshletStreamAsset.h`）。

## 1. 结论先行

**可行，且 Metallic 的底子比预想的好得多** —— 引擎已经有一条几乎完整的 Nanite 式流式管线
（`GPUDrivenStreamAsset`：BVH 层级 + 像素误差 DAG-cut + 持久化 GPU 遍历 + 页流式 + fallback + 动态 BLAS/TLAS），
但**光栅侧是玩具级的**（只输出调试颜色，无 visibility buffer、无属性、无材质、无 HZB 遮挡）；
另一条 `GPUDrivenDeferred` 管线有完整的剔除 + visibility buffer + 延迟 OpenPBR 着色，但**没有 DAG-cut LOD 和流式**。

两条管线的**交集恰好就是地形需要的能力**。推荐路径：

| 里程碑 | 内容 | 新代码量 | 收益 |
|---|---|---|---|
| **M0** | 把 DAG 遍历 + 页流式移植进 Deferred 路径（统一管线），补 HZB 遮挡 + per-cluster 材质 | 中（shader 为主） | 普通场景获得 Nanite 级 LOD/流式；地形获得入口 |
| **M1** | 地形离线烘焙为 `.meshstream.bin`（扩展离线 builder 支持高度场输入 + 裙边） | 小（builder） | 端到端验证：地形与普通场景同管线剔除/渲染/RT |
| **M2** | 地形 cluster GPU 程序化生成（高度页流式 + compute 生成 cluster payload） | 中 | 运行时编辑、内存 = 高度数据量级 |
| **M3** | 权重层虚拟纹理（VT）+ 地形材质层混合 | 大 | 材质表现对齐 UE |
| **M4（可选/远期）** | 虚拟 cluster（不存顶点，mesh shader + deferred 双端采样高度）+ 连续 morph | 大 | 真·VHM/Nanite-displacement 级 |

**M1/M2 即满足"地形与普通场景整合剔除 + 渲染"的目标**；M4 是渐进演进而非前提。

---

## 2. Metallic 现有管线盘点（已逐行核对）

### 2.1 GPUDrivenDeferred 路径（完整着色，无流式/LOD 选择）

帧流程（`Shaders/GPUDrivenPreview.slang` + `GPUDrivenDeferred.slang`，CPU 侧 `GPUDrivenPreviewPass.cpp`；
pass 注册 `GPUDrivenPreviewPass.cpp:613` + `BuiltinRenderPasses.cpp:47-49`，图 `Pipelines/Samples/gpu_driven_sponza.metallic_graph.json:43`；
`execute()` 帧序 `GPUDrivenPreviewPass.cpp:1002-1225`：sync → early cull(reset→instanceCull→compact) → drawVisibility(0) → buildHzb →
late cull → drawVisibility(1) → buildHzb → dispatchDeferred → composite）：

1. **实例剔除 compute**（`gpuDrivenPreviewInstanceCullMain`，Preview.slang:580-652）：
   - 两阶段 HZB：pass 0 用**上一帧 HZB**（`currentHzb ^ 1`）做视锥 + 遮挡测试，被遮挡 → visibility=2；
     pass 1 只对标记 2 的实例用**当前帧 HZB** 复测 → 3（可见）/ 0（剔除）；
   - 可见实例经 `InterlockedAdd` 压缩进 `visibleInstanceIdsBuffer`（不回读，仅调试）。
2. **Meshlet 压缩 compute**（`gpuDrivenPreviewCompactMain`，654-737）：每个 `MeshletDraw` 按实例可见性过滤，
   每 bucket 独立 indirect args（4×uint32/bucket `{count,1,1,overflow}`，`kDrawBucketCount=4`、`kIndirectArgumentUintCount=4`），
   原子 CAS 抢占容量；每个 meshlet 切成 **2 × 64 三角形 chunk**（`kMeshletTriangleChunkSize=64`），
   写 `(meshletIndex<<1)|chunkIndex`。CPU 侧 `meshletDraws` 缓冲布局 = **baseRange + 每 LOD 的 lodRanges**
   （GPUSceneSubsystem.cpp:492-540）—— LOD 区间已上传，只是选择逻辑未启用（见 2.3 差距表）。
3. **Mesh shader 光栅**（`gpuDrivenPreviewMeshMain`，739-843）：每 chunk 一个 DispatchMesh，
   CPU 侧 per-bucket `drawMeshTasksIndirect`（GPUDrivenPreviewPass.cpp:1985-1988）；
   - 解码：三角形索引打包 1 字节/角（`loadPackedMeshletTriangleIndex`，121-130），
     顶点从全局 position buffer 取 `GPUDrivenPreviewVertex`（float4 pos/normal/tangent + float2 uv + flags = 64B，52-60）；
   - **meshlet 级剔除在 mesh shader 内**：视锥 + 法线锥背面（`meshletVisible`，525-551，调用点 794）——
     没有 compute 级 meshlet 剔除、没有 HZB、没有 LOD 选择；
   - 输出：`visibilityId = ((meshletIndex+1)<<7) | meshletTriangleIndex`（7 bit 三角形 id，828-829）、
     texcoord、**per-instance 材质索引**（`instance.identity.y`，831）。编码上限：draw id 25 bit。
   - 上限：128 顶点 / 128 三角形 / meshlet，chunk 输出 ≤192 顶点（符合 Vulkan mesh shader 限制）。
4. **HZB 构建 compute**（`gpuDrivenPreviewHzbMain`，877+）双缓冲 ping-pong。
5. **延迟着色 compute**（`gpuDrivenPreviewDeferredMain`，Deferred.slang:614-843）：
   - visibilityId → meshletDraw → 重取三角形 3 顶点 → 屏幕空间重投影 →
     `perspectiveBarycentrics`（Deferred.slang:386-407）→ 插值 worldPos/uv/normal/tangent
     （**每像素重变换三角形角点，无三角形压缩/属性流**）；
   - per-instance 材质记录（`GPUDrivenPreviewMaterial`，9 个纹理槽 + UV transform + OpenPBR 参数）；
   - `sampleMaterialTexture` 经 `materialTextureRemapBuffer` 间接绑 bindless 纹理（LUT 描述符连续排列）；
     OpenPBR 直接光照 + 24 采样 SH/map 环境光 + ACES。
6. **关键事实**：`meshlet.lod`（`GPUDrivenPreviewMeshlet.lod`，Preview.slang:73）**只用于 debug 着色**
   （Deferred.slang:686-689），lodError 已上传但**没有任何 LOD 选择逻辑** —— 全部 meshlet 都参与光栅。
   Meshlet 与 LOD 在加载期由 `meshopt_buildMeshletsSpatial` + clodgen 构建（scene.cpp:1337-1527）。

### 2.2 GPUDrivenStreamAsset 路径（Nanite 式流式，但光栅是玩具）

数据（`Source/Runtime/Scene/MeshletStreamAsset.h`，运行时 `MeshletStreamRuntime.h/.cpp`，驻留 `MeshletStreamResidency.cpp`）：

1. **资产 `.meshstream.bin`**（**纯离线构建**，格式 v7；运行时自动重建已移除，需 `Metallic --build-meshstream`，
   且 `isCurrentForSource` 校验源尺寸+写时间+依赖指纹，MeshletStreamRuntime.cpp:859-865）：
   - LOD 由 NVIDIA **clusterlod** 生成（`buildMeshletLods`，scene.cpp:1411-1542）：`partition_size≤24`
     （保证 group ≤32 cluster）、`simplify_error_merge_previous=1.5`；quadric error 直接取自 clod
     （group.simplified.error / cluster.bounds.error，scene.cpp:1472/1514）；法线以权重 0.5 参与简化。
   - **node BVH**：每 primitive 一个根（lodLevel=invalid）→ 子节点 = 每 LOD 级一个 width-8 BVH 根
     （空间排序、bounds 聚合、maxQuadricError=子节点最大值，`buildStreamLodNodeTree` MeshletStreamAsset.cpp:1134-1208）。
   - **1 页 = 1 个 LOD group**（≤32 cluster，`appendStreamPrimitivePages` MeshletStreamAsset.cpp:1242-1470，
     pageCount==groupCount 强制）；**fallback 页 = 页数最少的（最粗）LOD 层**，运行时 init 即分配并锁定
     （`lockFallbackPages` MeshletStreamResidency.cpp:473-530）。
   - 页 worst case ≈ 96 + 32×36 + 32×128×16 + 32×128×3 + 32×4 ≈ **80KB**（无硬上限，驻留预算必须容纳最大页）。
2. **页 payload**（`MeshletStreamPayloadHeader`，24 word 头：word2 clusterCount、word9 clusterOffset、
   word10 positionOffset、word11 triangleOffset、word12 payloadBytes；**格式已支持 normal/texcoord/material
   属性偏移**（word17-20 + attributeFlags，默认 Position|Material）与 ByteRle 压缩）；
   cluster 记录 9 word：vertexOffset/Count、triangleOffset/Count、primitiveIndex、**materialIndex**、
   lodLevel、lodGroupIndex、**refinedGroupIndex**（构建期解析为全局 group 索引、指向更早 group 的 DAG 边）。
3. **页流式运行时**：
   - 页表条目 = **3-bit 状态（Empty/PendingUpload/Resident/LockedFallback）+ 29-bit device 偏移**（8B/页）；
   - 分配器 = 256B 对齐 first-fit free-list，**LRU 淘汰（年龄阈值 1 帧）**，卸载延迟 1 帧，锁定 fallback 页永不淘汰；
   - CPU 帧序（`GPUDrivenStreamAssetPass.cpp:389-434`）：syncRuntimeScene（transform 修订号变化时拷贝可见标志）
     → cmdBeginFrame（读回消费 → ≤64 页/帧经 `Streamer` 直接写 pageBuffer → **+3 帧存储延迟** → 页表 patch）
     → cmdPreTraversal（页表一次性 init → ApplyUpdates patch → 请求清零 → **active build 4 阶段 dispatch**
     Reset(1)/Seed(workers)/Run(持久化)/Finalize(1) → **Unload 阶段遍历只 dispatch 一次**
     （residentPageCount 线程，扫未使用页）→ CLAS 构建 → fallback/dynamic BLAS → TLAS）
     → draw → cmdEndFrame（请求缓冲拷回 host）；
   - **流式端到端延迟 ≈ 4-5 帧**（GPU 请求 → 回读 → CPU 调度 → 上传 → +3 帧驻留 → patch → 可画），
     靠锁定 fallback 页防 pop。
4. **GPU 遍历**（`gpuDrivenStreamAssetTraversalMain` / `BuildActiveMain`，StreamAsset.slang:1167-1534）：
   - **Seed**：每可见实例先 append 全部 fallback（最粗 LOD）groups 保证有底可画，再入队根节点；
   - **Run**：**持久化 wave-cooperative worker**（`runActiveTraversalWorker`，1024-1165）：工作队列
     （readyFrame 标记）→ 循环取活 → 内部节点入队子节点 / 叶节点做页需求 + cluster 选择掩码；
   - **DAG-cut**：`nodeNeedsTraversal`/`groupNeedsTraversal`（549-601）：
     `pixelError = maxQuadricError × uniformScale × pixelScale / distance > 1.5px`（`targetPixelError=1.5`）；
   - **cluster 目录细化**（`buildGroupClusterSelectionMask`，645-713）：cluster 的 refinedGroupIndex 指向更细 group，
     若该 group 已画/不可画则不画本 cluster —— 组内自适应 LOD；
   - 页需求/加载请求（`updateGroupPageDemand`）+ unload 请求（未使用页）；
   - **`instance.visible` 由 CPU 的 `renderNode.visible` 决定**（MeshletStreamRuntime.cpp:2544）——
     **全路径没有任何视锥/HZB 剔除，纯 LOD 遍历**。
5. **光栅**（`gpuDrivenStreamAssetMeshMain`，1776-1924）：indirect DispatchMesh，
   `groupCountX = activeGroupCount × maxPageClusters`，每 cluster 一个 workgroup；
   从页缓冲解码 float3 位置 + uint8 三角形索引 → 输出**位置 + 调试颜色**（无 visibility buffer/属性/材质）。
6. **RT**（可选 `enableClusterRtx`）：**CLAS 池**（固定步长存储，CPU 写页表：30-bit 偏移 + 2-bit 状态
   Empty/Active/Retiring；`vkCmdBuildClusterAccelerationStructureIndirectNV`；cluster id =
   `pageIndex×clusterIdStride + clusterIndex`；**位置未量化**，`minPositionTruncateBitCount=0`）；
   4 阶段 BLAS input builder（Reset/Count/Setup/Insert）→ **一次 indirect cluster-BLAS 构建**（隐式目标）；
   fallback BLAS per locked primitive；TLAS 实例选择 dynamic/fallback 地址。
7. **上限**：32 cluster/group、128/128 顶点/三角形、262144 active groups（默认）、4096 驻留页（默认）、
   traversal workers 1024 / work items 1M、512MiB RT 预算、全套 overflow 计数器。

### 2.3 两条路径的差距清单（要"Nanite 化"必须补的）

| 能力 | Deferred 路径 | StreamAsset 路径 | 目标（统一后） |
|---|---|---|---|
| 实例视锥剔除 | ✅ compute | ❌（CPU 标志） | ✅ 已有 |
| 实例 HZB 遮挡 | ✅ 两阶段 | ❌ | ✅ 已有 |
| BVH/DAG 层级 | ❌ 平铺 meshletDraw | ✅ node 树 + 组细化 | 移植到统一管线 |
| 像素误差 LOD 选择 | ❌（meshlet.lod 未用） | ✅ 1.5px 判据 | 移植到统一管线 |
| cluster 级剔除（视锥/HZB/背面） | 视锥+法线锥（mesh shader 内） | ❌ | 加 HZB，放 compute/遍历 |
| 页流式 + fallback 防 pop | ❌ 全驻留 | ✅ | 移植到统一管线 |
| visibility buffer + 延迟着色 | ✅ | ❌（调试色） | 统一输出 visibilityId |
| 顶点属性（normal/uv） | ✅ 全局 vertex buffer | 格式支持、shader 未用 | mesh shader 解码 + deferred 解码 |
| per-cluster 材质 | ❌（per-instance） | 格式支持（materialIndex） | deferred 读 cluster.materialIndex |
| 运行时/程序化数据注入 | ❌ 静态 GPUScene | ❌ 全离线烘焙 | **M2 增加程序化页生成 hook** |
| draw id 编码空间 | 25 bit（`(id+1)<<7`） | 活跃组×cluster 线性 id | 统一后需扩展/分区编码 |

---

## 3. 与 UE 5.6 Nanite 的对照（常量与算法已验证）

见 `UnrealLandscapeResearch.md` §5；关键对照（`Engine/Shaders/Shared/NaniteDefinitions.h`、
`Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf`）：

| 项 | UE Nanite | Metallic StreamAsset | 评估 |
|---|---|---|---|
| cluster 大小 | 128 tri / 256 vert | 128 / 128 | 兼容（128 tri 满足） |
| group | target 128 cluster（编码上限 511） | ≤32 cluster | 32 偏小但够用 |
| 页 | root 32KB / streaming 128KB（≤256 cluster） | 1 页 = 1 group，≤32 cluster | 更细粒度，反而灵活 |
| 层级 | BVH fanout 4 + DAG group parts | width-8 node 树 + refinedGroupIndex 组细化 | 同构 |
| LOD 判据 | `ProjectedEdgeScale > UniformScale × LODError × LODScale`（NaniteClusterCulling.usf:288-338） | `quadricError × scale × pixelScale / distance > 1.5px` | **数学同构**（都是"投影误差像素数"） |
| 遮挡剔除 | 实例两阶段 HZB + 持久化遍历内 cluster HZB | 实例两阶段 HZB（Deferred 路径）+ 遍历无 HZB | 缺遍历内 HZB |
| 光栅 | HW（小三角形）+ SW 光栅器（大三角形）+ 材质深度桶 | 纯 HW mesh shader | 地形面密度高，HW 足够，SW 可后补 |
| 顶点数据 | 全局 unified buffer + 页转码（NaniteTranscode） | 页缓冲自包含 | 建议保持页缓冲自包含（见 §5） |

**结论**：Metallic StreamAsset 已实现 Nanite 的"骨架"（层级 + 误差 LOD + 流式 + 持久化遍历），
缺的是"血肉"（遮挡剔除、visibility buffer、材质、程序化数据源）。地形整合不需要重造轮子，只需要**合流 + 补肉**。

---

## 4. 统一管线目标架构（M0）

把两条路径合为一条（Deferred 为渲染后端，Stream 为数据/遍历前端）：

```
1. GPUScene 实例记录（transform/bounds/材质/类型标志）
2. 实例剔除 compute（已有）：视锥 + 两阶段 HZB → visible 标志 + 压缩 id
3. 持久化 DAG 遍历（StreamAsset 移植，输入 = 可见实例）：
   - 每 node：像素误差判据 + 【新增】node 包围球 HZB 遮挡测试（mip 由投影半径选）
   - 叶 node → group → cluster 选择掩码（refinedGroupIndex 组细化）
   - 页 demand/load/unload 请求（流式）
   - 【新增】cluster 级视锥 + 背面（法线锥）剔除并入遍历或紧跟的 compact pass
4. Active group 构建 → per-bucket indirect DispatchMesh args（每 cluster/chunk 一个任务）
5. Mesh shader：页缓冲解码 cluster → 输出 visibilityId + 属性（texcoord/材质 id）；
   地形 meshlet 与普通 meshlet 同一条代码路径（顶点已含位移）
6. 延迟着色：visibilityId → activeGroup/cluster → 页缓冲解码顶点/属性 →
   重投影 barycentrics → 【新增】per-cluster 材质索引 → 材质求值（地形材质 = 层混合）
7. RT：CLAS 页表 + 动态 BLAS（已有）自动覆盖地形 cluster；fallback BLAS 防 pop
```

关键设计决策：

1. **页缓冲作为统一的 cluster 数据源**。延迟 pass 增加一条"从页缓冲解码"的路径
   （活跃组表 + 页表已可绑定），与现有 GPUScene 常驻 meshlet 并存：
   - 常驻资产（小场景）继续走 GPUScene position buffer（现有代码不动）；
   - 流式资产（含地形）走页缓冲解码。visibilityId 编码区分两者（例如 meshletIndex 高位加 1 bit 来源标志）。
   - 远期可像 Nanite 一样把页上传进全局 unified buffer（NaniteTranscode 模式），消除双路径；本期不必。
2. **mesh shader 输出扩展**：StreamAsset mesh shader 改为输出 visibilityId（沿用 `(id+1)<<7|tri` 编码，
   增加 chunk 位）+ 需要的属性（地形网格 UV 可程序化生成，不必存）。
3. **per-cluster 材质**：payload 已有 cluster.materialIndex 与页级材质表（materialOffsetBytes/materialCount），
   deferred pass 改读 cluster 材质索引（普通 meshlet 保持 per-instance 回退）。
4. **遍历内 HZB**：`nodeNeedsTraversal` 增加 HZB 遮挡测试（复用 `sphereOccluded`，
   Preview.slang:505-523 的 mip 选择逻辑），对地形尤其重要（大面积被地形自身遮挡）。

---

## 5. 地形整合方案（M1/M2 细节）

### 5.1 数据模型（沿用 UE 调研结论）

- 源数据：**每 patch 高度纹理 + 完整 mip 链**（UE 格式：R16 或 R8G8 打包高度，`UnrealLandscapeResearch.md` §2.2）；
  patch 网格对齐 UE 组件概念（subquads 63/127/255 可选，先支持 1×1 subsection 简化）。
- 权重层（材质）：M3 之前直接绑定 per-patch 权重纹理（UE 非 VT 模式做法）；M3 换 VT。

### 5.2 M1：地形离线烘焙进 `.meshstream.bin`（最快端到端验证）

- 扩展离线 builder（`buildMeshletLods` 所在链路的 scene.cpp/MeshletStreamAsset.cpp）：新增"高度场 primitive"输入；
  - cluster = 8×8 quad 网格（81 顶点 / 128 三角形，恰好卡满 meshlet 上限）；
  - **裙边**：cluster 边缘向外扩 1 环 quad 并向下挤出 `SkirtDepth`（UE Nanite 地形同款，
    `LandscapeEdit.cpp:4383-4399`）解决 DAG-cut 邻居 LOD 不一致的裂缝；
  - 简化：按 quad 合并（2×2→1）逐级生成 LOD，`maxQuadricError` = 该 LOD 相对下一级的**最大高度偏差**
    （地形不需要 quadric 网格简化器，高度差就是几何误差；min/max mip 金字塔可精确给出保守误差界；
    直接沿用 clusterlod 亦可，把高度差作为 per-cluster error 注入）；
  - node 树 = patch 四叉树（挂到现有"每 LOD 级一个 width-8 BVH 根"结构下）；fallback 页 = 最粗 LOD。
- 注意：**运行时自动重建已移除**（`Metallic --build-meshstream` 离线构建 + `isCurrentForSource` 校验），
  所以本方案与现有资产工作流完全一致。
- 结果：**地形零 GPU 代码改动**进入统一管线：流式、DAG LOD、HZB、visibility buffer、延迟着色、CLAS/BLAS 全自动。
- 局限：编辑地形 = 重新烘焙（适合资产流式工作流；不做运行时雕刻）。

### 5.3 M2：GPU 程序化 cluster 生成（推荐目标态）

- **高度页**：把 patch 高度 mip 数据打包为特殊 payload 页（页表/驻留/淘汰机制 100% 复用 StreamAsset）；
  patch 自身也是 primitive：其"cluster 页"的 payload 由 **compute pass 程序化生成**。
  **程序化注入接口**：运行时目前没有 hook（节点/组/页全部离线烘焙），但 residency/页表 patch/CLAS 机制
  已具备注入所需的全部原语 —— 需新增：程序化页源（驻留管理器 PendingUpload 时除 `Streamer` 拷贝外
  注册 GPU 生成 pass）+ 生成后页表 patch + 生成 payload 的 CLAS 构建（现有 `MeshletStreamResidency.cpp`
  upload/evict 接口与 `applyPageTablePatches` 可直接复用）。
- **生成 pass**：读高度页（对应 mip）→ 写 cluster 页 payload：
  - positions：高度采样 + 裙边；
  - normals：高度场有限差分（无需存储，生成时算出）；texcoord：网格坐标程序化（不用存）；
  - cluster 记录、页头、refinedGroupIndex（四叉树关系）、材质表照常写。
- **编辑**：高度页脏 → 重新生成 cluster 页 → 页表失效/刷新；运行时雕刻 = 改高度纹理 + 重生成（可局部）。
- 内存：只有高度页 + 少量生成缓冲；cluster 数据随需生成。
- **裂缝**：裙边方案不变（生成时加环）。连续 morph（双 mip 插值）推迟到 M4。

### 5.4 M4：虚拟 cluster（真·VHM 式，远期探索）

- cluster 页不存顶点；mesh shader 与 deferred pass 都按 cluster 描述符（patch 坐标 + 网格尺寸 + LOD）
  **双端采样高度纹理**重构位置/法线；min/max mip 金字塔给遍历提供保守 bounds 与 error。
- 收益：连续 LOD morph（双 mip lerp，UE 经典路径同款）、几何内存趋零；
  代价：两条 shader 路径引入地形分支 + 高度纹理 bindless 绑定 + 光线追踪/阴影需同步重构（VHM 先例可参照）。

### 5.5 材质（地形层混合）

- 地形材质 = per-cluster material index 指向的"地形材质记录"：权重纹理数组（4 层/纹素）+
  层参数；UV = 世界 XY × scale/bias（payload texcoord 或程序化）；
- M3 引入 VT：feedback + 页表 + 物理图集（`UnrealLandscapeResearch.md` §6.6 最小清单），
  地形权重页可复用同一套 StreamAsset 驻留/淘汰机制（**页系统统一：几何页 + 高度页 + 权重页**）。

---

## 6. 风险与验证点

| 风险 | 影响 | 缓解 / 验证 |
|---|---|---|
| 页缺 → pop（**端到端 4-5 帧延迟**） | 视觉突跳 | 已有 fallback 锁定页机制；验证 worst-case 相机高速飞行 |
| 驻留预算抖动（页 worst case ≈80KB） | 流式抖动 | `maxResidentPages=4096` 默认需按地形工作集调大；年龄阈值 1 帧的 LRU 已防瞬时抖动 |
| active group / 遍历容量溢出（262144 / 1M work items） | 漏画 | 已有 overflow 计数器 + 统计；地形面密度高，需压测 |
| DAG-cut 邻居 LOD 差 >1 裂缝 | 破洞 | 裙边（M1/M2）+ 树结构保证；UE 同方案 |
| 遍历内 HZB 误剔除（地形自遮挡） | 漏画 | `sphereOccluded` 已有 depth bias；地形保守 bounds（min/max 金字塔） |
| mesh shader 192 顶点上限 | 光栅瓶颈 | 128 tri cluster 固定 81 顶点，天然安全 |
| 高度页与几何页共用 LRU 互相挤压 | 流式抖动 | 分池预算（几何/高度/权重各自 maxResidentPages） |
| per-cluster 材质状态切换 | PSO/描述符压力 | 地形 patch 通常单材质；保持 per-patch 材质合并 |
| CLAS 位置未量化（float3） | RT 显存放大 | 地形可后补量化 + `minPositionTruncateBitCount`（Nanite 21-bit 方案参照） |

**验收标准（M1）**：一个 8×8 km 高度场 + 普通 glTF 场景混排，同一帧内：地形与普通网格同走
实例 HZB 剔除 + DAG LOD + visibility buffer + 延迟着色；相机拉近时地形页流式加载无 pop；
RT 模式下地形 cluster 进入 BLAS 可被 ray query 命中。

---

## 7. 与 UE 方案对照总结

| 决策点 | UE 5.6 | 本方案 |
|---|---|---|
| 地形几何 | 经典路径 VS 采样高度 / Nanite 路径烘焙静态网格 | M1 烘焙（对齐 Nanite 地形）、M2 GPU 程序化 cluster（超越：可编辑+流式） |
| LOD | 经典路径连续 morph；Nanite 路径 DAG-cut | DAG-cut（现有 1.5px 判据）；连续 morph 留 M4 |
| 裂缝 | 经典：EdgeLOD morph；Nanite 地形：裙边 | 裙边（M1/M2） |
| 剔除 | Nanite：实例两阶段 HZB + cluster HZB；5.6 tile 路径：仅视锥 | 实例两阶段 HZB（已有）+ 遍历内 HZB（新增） |
| 流式 | Nanite 页流式（128KB 页）+ fallback | StreamAsset 页流式 + fallback（复用） |
| 材质层 | Surface 域材质 + 权重 VT | per-cluster 地形材质 + 权重纹理（M3 换 VT） |
| 碰撞 | 独立 Chaos heightfield | 独立 CPU heightfield（不依赖渲染 cluster） |
| 编辑 | GPU 层合并 render-to-texture | M2 后：高度页局部重生成 |

---

## 8. 建议的第一步（最小验证闭环）

1. **M0-a**：StreamAsset mesh shader 输出 visibilityId + per-cluster 材质索引（去掉颜色输出），
   deferred pass 增加页缓冲解码路径 —— 用现有 `.meshstream.bin` 资产验证流式几何进延迟着色。
2. **M0-b**：`nodeNeedsTraversal` 加 HZB 遮挡测试（复用 `sphereOccluded`）。
3. **M1**：离线 builder 支持高度场 → 烘焙 4×4 patches 地形资产，与 glTF 场景同帧渲染、RT 命中。
4. 之后按 §1 里程碑推进 M2（程序化页生成 + 高度页流式）。
