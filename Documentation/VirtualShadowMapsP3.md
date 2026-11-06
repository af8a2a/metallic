# Virtual Shadow Maps P3：局部光与多灯查询

对应 [实施方案](VirtualShadowMapsResearch.md) 的 P3，沿用 [P2](VirtualShadowMapsP2.md) 的提交后缓存、完整场景 caster、MASK Coverage 与硬阴影。新增 spot 透视投影、point 六面、稳定光源身份、统一多灯 Pass 和 Deferred 查询。旧 P1/P2 图继续使用原接口；本阶段没有切换默认阴影系统，也没有性能收益结论。

本文记录实现合同及 2026-10-25 本机验证结果。88 帧局部光缓存对照、独立遮挡参考、四灯 OpenPBR 示例与所选回归已通过；原型成本和未覆盖范围在下文明确列出。

## 投影与覆盖范围

`VirtualShadowMapLight` 区分 Directional、Spot 和 PointFace。旧 `GPUCelestialLight` record 重载保留为方向光适配器。每个 spot 使用一个 core 实例；每个 point 使用六个独立 core 实例，每面保留一个完整粗级页。所有灯共用一次 caster preparation，继续从完整 resident 场景或已完成初始加载的 StreamAsset terminal cut 生成遮挡物，不使用主相机可见列表代替 caster。

方向光保留 camera-centered clipmap。局部光所有 mip 使用相同透视视锥，各级虚拟页轴数为 `max(1, virtualPagesPerAxis >> level)`，另外保留单页粗级；`clipmapCount` 在局部光上表示细 mip 层数。Receiver 标记所有覆盖它的层，查询从最细有效准入页逐级回退到有效粗级。当前不是基于 receiver footprint 的最优 mip 选择器。

Spot 的 `outerConeAngle` 是半角，必须严格满足 `0 < angle < pi/2`；不支持该范围之外的 spot 投影。Point 每面为 90° 视锥，面序及基向量如下，满足 `right × up = forward`：

| face | forward | right | up |
| --- | --- | --- | --- |
| 0 / +X | (1,0,0) | (0,0,-1) | (0,1,0) |
| 1 / -X | (-1,0,0) | (0,0,1) | (0,1,0) |
| 2 / +Y | (0,1,0) | (1,0,0) | (0,0,-1) |
| 3 / -Y | (0,-1,0) | (1,0,0) | (0,0,1) |
| 4 / +Z | (0,0,1) | (1,0,0) | (0,1,0) |
| 5 / -Z | (0,0,-1) | (-1,0,0) | (0,1,0) |

Demand、core query 与 Deferred 使用同一个 `VirtualShadowMapCommon.virtualShadowMapPointFace()`：按 receiver 相对光源的位置选择绝对值最大的轴；相等时优先 X、再 Y、最后 Z，零向量归 +X。选面不依赖 point 的 direction。

局部投影默认 `nearPlane=0.001m`。光源附近被 near plane 截去的 caster 不承诺阴影正确，包括其可能遮挡 near 外 receiver 的情况；这是当前覆盖限制。选定面的轴向 receiver 深度不大于 near 时返回中性可见度并增加 face `statistics[29]`，不伪装成已验证的 near 内遮挡结果。光源 range/cone 外 receiver 不申请页。

局部三角形先在齐次空间裁 near/far 和页内四边，再映射 atlas；保留 `clip.w` 和未预除的 UV。近面相交时保持插值后的 `z`，令 `w=z`。跨光源的大三角形可能在计算插值 `w` 时发生消减误差，反过来令 `z=w` 会把误差写入深度；本次整体平移地面的实际 GPU 测试曾因此产生大片自阴影，已通过该修复消除。原方向光四边裁剪不走这个近面分支。

Manager 从现有 GPU 灯数据的 outer cosine 恢复 spot 半角；极窄角若已量化为 cosine=1，同样明确返回 Unsupported，不伪造一个更宽的视锥。

`range > 0` 决定局部投影的基础 far extent；`range == 0` 是合法无界光，此时使用完整 caster bounds 到灯位置的最大角点距离。Far 增加余量；开启缓存时保留仍覆盖当前范围的旧 far，不因场景缩小而收缩，从而避免对旧深度使用新映射。位置、spot 方向/锥角、range、near、face 或影响深度的资源/设置变化会重置相应缓存。当前继承 P2 的 Coverage、双面和材质限制，未新增透明透射、位移或软阴影能力。

## 身份与缓存所有权

`PunctualLight.stableId` 是不序列化的运行时身份。默认创建分配，编辑与重排的值复制保留。`SceneDocument::setLighting()`、文档加载和 `RenderWorld::setLighting()` 在发布列表前规范化：保留每个首次出现的非零 ID，零 ID 或后续重复项分配新 ID。因此复制一盏灯作为新灯不会与原灯共享缓存。所有 virtual lights（包括导入绑定）均保留该身份；原生 RenderLight 使用独立 tag 加完整 EnTT entity/version；Sun/Moon 使用另一独立 tag 加固定槽位。

`SceneLightRecord` 透传身份，GPUScene 将身份变化纳入 light topology generation。当前 GPUScene source slot 仍用于 LightGrid 与 shader 查表，但不作为缓存 key。Manager 按 `(scene resourceIdentity, stableId)` 持有缓存；每帧重建 source-to-entry 映射，纯重排不会把上一盏灯的页绑定到新灯。

移除、禁用或预算拒绝的灯释放持久 pool；point 改为 spot 时释放多余五面。改变场景身份清空整个 manager。每面保留 P2 的 pending/commit/cancel 与 completion 依赖，未提交 recording 不发布有效缓存。Manager 和 core 均不宣称支持帧重叠；`supportsFrameOverlap()` 为 false。预算重建期间旧提交资源可能因 completion/command lease 短暂与新资源共存，配置预算不是进程瞬时显存上限。

## 全局预算与退化

`VirtualShadowMapLightsPass` 的 `lightCapacity` 是编译属性，默认 8、允许 2–64。表包括固定 Sun/Moon 和全部当前局部源槽，包含 disabled 与导入的受抑制旧槽；超出容量返回明确的 Unsupported，不能默默漏灯。容量决定数组层数和 buffer 大小。

`physicalPageBudget` 默认 256。先按 stable ID 升序遍历灯，为每个启用 spot/方向光保留 1 页，为每个 point 整体保留 6 页。放不下整灯粗级时将其标为 `budgetDropped`，继续考虑后面的灯。随后按同一稳定顺序把剩余页均分到 admitted faces，余数优先分给前面的 face，每面最多 4096 页。当前排序是确定性的配置政策，不是屏幕重要性优先级算法。

细页不足使用有效较粗层；细页 work overflow 使用独立完整粗级。整灯粗级预算拒绝时不执行该灯 core，表显式标记并计数，六层 debug 保持洋红、shadow 保持 1。Deferred 对该显式状态返回中性可见度；不会读取未完成纹理。禁用与预算拒绝由 metadata 区分，不能仅从白色 shadow 判断原因。

## Graph 与 shader 接口

新 Pass 输入匹配的 `depth` / `rasterInfo`，输出：

| 输出 | 格式与尺寸 | 含义 |
| --- | --- | --- |
| shadows | R8 Texture2DArray，`capacity × 6` 层 | 每灯线性硬可见度，entry 固定六层 |
| debug | RGBA8 Texture2DArray，同层数 | 每面页选择及未生成层标记 |
| lights | `64 + capacity × 64` B | header 与 source-to-layer entries |
| statistics | `capacity × 6 × 128` B | 每面 32 uint，未执行面全零 |

每次执行先录制全部 admitted face 的 core 工作，保存各自结果和具备独立名字的 imports；随后只调用一次 `executeStages()`。Initialize 阶段清 shadows 为 1、debug 为洋红，上传完整表并清统计；Publish 阶段将全部结果复制到互不重叠的 `baseLayer + face`，并复制对应 128 B 统计。阶段规划器负责初始化与发布之间的写后写依赖。RenderGraph 显式声明 texture arrays 与 table 的边，private core 输出通过 import/stage transfer-read 和目标 transfer-write 发布；没有把 atlas descriptor 隐藏在孤立 table 依赖中。

Deferred 新增成对 optional 输入 `virtualShadows` / `virtualShadowLights`。`SceneResourceParameters` 在原 400 B 后追加 sampled-array handle（offset 400）和 table handle（404），总计 408 B；primary split / diagnostic 扩展包为 416 B，旧字段偏移不变。存在 multi-light table 时它拥有整个灯域，忽略旧单灯/Moon shadow pins；没有 table 时保持旧接口行为。新数组读取线性值，旧 SIGMA sqrt 编码只在旧接口平方解码。

`VisibilitySurfaceLights.visibility()` 按当前 `SurfaceLightSample.source` 查 entry，point 再按位置选面。全屏与 material-binned Deferred 共用该函数，不采用两份灯身份或阴影逻辑。

## 表和统计 ABI

共享声明位于 [VirtualShadowMapLights.h](../Source/Runtime/Render/Shadow/VirtualShadowMapLights.h)。所有整数为 uint32；64 位身份以 low/high 两个 uint 存储。

Header 为 64 B，字段依次为 `version=1, count, capacity, totalFaces, admittedCount, budgetDroppedCount, disabledCount, physicalPageBudget, sceneIdentity[2], frameIndex[2], lightGeneration, sourceSlotCount, reserved[2]`。

Entry 为 64 B，字段依次为 `source, kind, baseLayer, faceCount, stableId[2], pageBudget, status, positionRange[4], reserved[4]`，其中 positionRange 是 float4；其他字段是 uint。`kind` 为 0 directional、1 spot、2 point；`status` 为 0 disabled、1 admitted、2 budgetDropped。`baseLayer = entryIndex × 6`；point faceCount 为 6，其余为 1。`source` 的天体域为 `0x80000000 | slot`，局部域为当前 GPUScene 源槽。`pageBudget` 是该灯所有面的总页预算。

统计块 byte offset 为 `(entryIndex × 6 + face) × 128`。0–28 继承 [P2 统计定义](VirtualShadowMapsP2.md#统计-abi)，29 记录该局部 face 的 near-plane receiver 排除，30–31 保留。Header 的 totalFaces 包括 disabled/rejected entries；admittedCount 等为灯数，不是面数。未执行面的全零统计不代表一次已运行且正确的 shadow 查询。

继承的 face `statistics[6]` 在 range/cone/canonical-face 筛选前计数，所以是全屏有效 receiver 数；`statistics[7]` 只计本面实际 coarse 查询数。局部光不能用两者相等判断是否完全回退，应结合本面 debug 的 coarse/fine 像素判断。

## 原型成本与示例

P3 目前选择显式每灯 screen-space arrays 作为集成原型，尚未实现调研方案中的直接逐灯页表查询。仅公开 R8 + RGBA8 数组的裸存储为 `width × height × capacity × 6 × 5` B；默认 8 槽在 512² 下为 60 MiB，在 1920×1080 下约 474.6 MiB。还需加上每个 core 的输出、D32 图集、页表、work buffers 和提交中的资源。每个 admitted face 都执行 receiver resolve 和全屏结果复制；点光通常有六次。不能把全热 cache 无 raster work 等同于整帧无阴影成本。

可选 sample [virtual-shadow-map-p3](../Pipelines/Samples/virtual_shadow_map_p3.metallic_graph.json) 使用 [OpenPbrVSMLights 场景](../Asset/LookDev/OpenPbrDefault/OpenPbrVSMLights.metallic_scene.json)：Sun/Moon 加暖色 point 与冷色 spot。内部固定 512²、temporal jitter 关闭、receiver `autoLod=false/lodLevel=0`、`initialLoad=true`；lightCapacity=8、总页预算512、pageSize=128、每面 maxWorkItems=1,048,576、cacheEnabled=true、nearPlane=0.001m。主视图固定 LOD 0 继承 P1/P2 避免 receiver/caster 几何差异的限制。

这是一份独立 sample，未修改既有默认图。没有同工作负载的 RT/VSM 性能比较，也没有大场景长期显存、SMRT 质量或默认替换验收结论。

## 验证

使用已有 `build-pass-stages-nrd` Release Ninja 配置、MSVC 14.51.36231 / Windows SDK 10.0.26100.0；没有改变编译器、生成器或 SDK 配置。

| 检查 | 实际结果 |
| --- | --- |
| `Metallic`、`MetallicRHITests`、`MetallicShaderRequestsTests`、`MetallicSceneTests` 构建 | 通过 |
| `python -B Tools/CheckShaderModules.py` | 385 files / 273 modules / 112 implementation units，通过 |
| Shader tests：`*Deferred*:*Trusted*:NativeShaderWarmupRequests.*` | 28/28 通过 |
| Scene tests：`*Light*:*Lighting*` | 9/9 通过 |
| RG array layer propagation / constraints、sample load | 3/3 通过；真实 6 层 texture/view/consumer，拒绝单层错接、零层输出、3D array 和 array presentation |
| `virtual_shadow_map_local_lights` + Vulkan/synchronization validation | 通过；88 帧 paired array 对照，所有 48 层逐像素相同，validation 0 messages |
| 实际 P3 runtime cache SPIR-V 提取与 `spirv-val --target-env vulkan1.4 --scalar-block-layout` | 10 个入口、7 个 CPU/Slang 布局通过；Push 160 B、Projection 112 B；依赖原文字节核对通过，无 RayQuery/RT capability |
| P0 / P1 / P2、RT+SIGMA、GPUScene 提交恢复/Coverage/stream cache、celestial publication、trusted ABI、RG extent | 13/13 通过；P2 的 52 帧缓存矩阵继续通过 |
| GPUScene light collection / frustum / camera frustum / world light sync | 4/4 通过 |

局部光测试在 257×193、关闭 RayQuery 的独立 preview device 上执行，使用真正光栅化的 glTF receiver depth。CPU 参考从 authored triangles 做双精度 camera ray 与 light-segment 求交，不复制 VSM 投影矩阵。总计 5,211,628 个检查样本、0 个遮挡错误；resident 和 Stream 各覆盖六个面中心、12 条双面接缝，分别包含 6,730 个接缝样本，其中 736 个被遮挡。接缝本身没有被排除。

参考区分硬件单 texel 采样和几何射线的阴影轮廓差异：按每像素实际 fine/coarse debug，以 fixture 的 0.035 m / 0.24 m 邻域光线判断几何阴影边缘，只排除这些轮廓歧义并单独计数；剩余样本必须零错误。两面墙公共边的双精度重心坐标允许 `1e-12` 舍入误差，以免 `u+v=1.0000000000000002` 被两面同时拒绝；没有额外排除该 receiver 或接缝像素。

88 帧矩阵覆盖 point 位移、spot 方向/锥角/range 变化、caster 整体平移的 old/new bounds 失效、重排、复制/移除光源、只改辐射属性、7/6/0/512 页预算，以及 `maxWorkItems=1` 的溢出、重试和恢复。Point 六面整体准入；投影变化重画对应灯，纯重排或辐射变化复用既有缓存。near 内 receiver 统计及两个不可表示的 cone 拒绝也有独立断言；日志中的对应 Unsupported 是预期负例。

额外 28 个 flat/binned Deferred 消费者帧分别隔离两盏灯，核对实际线性 HDR 与各灯 mask：不该受影的像素没有被另一灯遮蔽，单灯贡献相加最大误差 `2.384185791015625e-7`；两条路径的 cache/force HDR 均完全一致。

四灯 OpenPBR 示例在 512² 下比较 Cached / Force / Flat 三种配置：6 帧序列中抓取第 0、1、5 帧 HDR，再抓取显示输出和 receiver depth，所有对应输出逐像素一致且有限。已人工检查生成的 PNG。该 baseline harness 启用了 RayQuery，不能由此推断全局没有 RTAS 构建成本；专用局部光测试的无 RayQuery 配置与其分别记录。硬边与固定无 jitter 的锯齿仍属当前原型质量。

主要复现入口：

```powershell
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_local_lights --rhi-validation --output-dir .cache/vsm-p3
.\build-pass-stages-nrd\tests\MetallicShaderRequestsTests.exe '--gtest_filter=*Deferred*:*Trusted*:NativeShaderWarmupRequests.*'
.\build-pass-stages-nrd\tests\MetallicSceneTests.exe '--gtest_filter=*Light*:*Lighting*'
```

本机证据保留在忽略的 `.cache/` 中，不作为源文件提交：

- `vsm-p3-run7.log`、`vsm-p3/VirtualShadowMapLocalLights/VirtualShadowMapLocalLights.json` 与各场景 GPU/CPU/Error PNG。
- `vsm-p3/spirv/Validation.json`：实际 runtime artifact、source proof、SPIR-V 和布局验证。
- `vsm-p3-openpbr-config/`：三变体 graph、Cases 与比较脚本；`vsm-p3-openpbr/Comparison.json`、`OpenPBRVSMCached-FinalBlit.color.png`。
- `vsm-p3-regression.log`、`vsm-p3-light-regression.log`、`vsm-p3-rg.log`、`vsm-p3-shader-tests.log`、`vsm-p3-scene-tests.log`。

尚未单独验收三面交汇的 8 个 corner、局部透视 MASK 纹理专用矩阵、多灯长程大场景/峰值显存与软阴影质量。本次不重跑整套历史测试；[P2](VirtualShadowMapsP2.md) 已记录的 Deferred 法线对照失败与 CPU shader suite 陈旧计数断言仍是未解决项，不能用本次所选回归通过覆盖它们。P4/P5 的质量、生产成本和默认切换门槛继续保留。
