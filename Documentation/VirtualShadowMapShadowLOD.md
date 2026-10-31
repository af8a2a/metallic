# VSM 阴影专用 LOD

`VirtualShadowMap` 与 `VirtualShadowMapLights` 提供运行时属性 `shadowLOD`，默认 `false`；`shadowLODErrorTexels` 默认 `1.0`，界面范围为 0–16。误差单位为阴影虚拟纹理 texel，不是主相机屏幕像素。`0` 请求最细几何：resident 保留原始完整三角与顺序，stream 请求最细层级并在页到达前使用完整驻留回退 cut。

本次加入独立的阴影几何选择，不改变现有 one-pass projection 的默认开启状态，也不改变静态/动态深度分层的默认关闭状态。入口见 [VirtualShadowMapCasters.h](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.h)，误差计算见 [VirtualShadowMapLOD.h](../Source/Runtime/Render/Shadow/VirtualShadowMapLOD.h)。

## 选择范围与误差

每次 Pass 先准备原始 caster 快照。resident 快照是完整 authored 三角，stream 快照是所有 terminal 分支的完整 cut。随后从完整 bounds 构造实际阴影投影的 X、Y、W 行及虚拟分辨率；多灯 Pass 汇总全部准入光源、点光六面、各 clipmap 层和 coarse 投影，仅选择一次共享 cut。

每个 group 使用所有投影中的最大误差。方向光按正交投影 texel 密度计算；局部光使用误差扩展球上的透视 Jacobian 上界，近裁剪面相交时保守要求细化。变换尺度使用包括非均匀缩放与 shear 的保守上界。方向光 coarse 的密度上界依据完整 caster bounds 的未 padding 范围，覆盖旧 coarse 缓存投影被复用时的情况。

该需求独立于主相机的 visible/active cut，不以主相机屏幕外为由删除投影 caster。某个球位于一个阴影视图以外时，仅该视图的细化需求为零。最终遍历仍覆盖全部 terminal 分支；需要细化的 group 会把需求向所有 DAG 祖先传播，替换仅在所有 parent owner 生效时发生，以保留多父关系的完整覆盖。

当前实现为 CPU 元数据选择与 CPU 打包上传。resident 利用已有 `meshletLodGroups/Clusters/Vertices/Triangles`，stream 使用资产完整 group/refinement 元数据，不复用主视图压缩列表。尚未加入 GPU LOD 遍历或 cut hysteresis。

## 材质与 streaming

MASK 与 displacement caster 保留基准几何；这一版本不对 alpha 覆盖或位移误差做额外简化。没有 resident hierarchy 的 primitive 同样保留完整原始三角，并计入 fallback。这里的 stream 基准是原有完整 terminal cut，不表示所有原始三角已驻留。

Stream 只从 `pageResident()` 为真的 group 选 drawable cut，缺页会保留安全的驻留祖先，不把缺页当成空几何。需要的细化页通过 `requestShadowPages` 独立请求，选中的驻留页先 touch；资源录制先登记 geometry-use completion，页范围不能在本次 GPU 使用完成前回收。页仍沿用普通 stream 预算、异步加载及 publication 路径，不永久锁住 refined 页。

请求成功或分配成功不表示页已经 drawable，真正细化需要后续正常 frame 完成页发布。`freezeSourceResidency` 或 `freezeRasterSnapshot` 生效时不新增 shadow 请求或刷新驻留，缺页需求计入 blocked，当前可绘制 cut 继续使用。需要按现有 runtime 契约在 frame 的页发布后消费 shadow geometry；shadow producer 本身不替代主 stream 的更新与 publication 阶段。

## 缓存身份

选中 cut 的 cluster ID 序列单独形成每个 caster 的 `lodKey`，原始 cut 使用零。`contentKey`、`transformKey`、`materialKey`、完整 caster bounds 与几何 publication 依赖保持来自 base authored 快照；不使用全局 page-table mutation serial 作为阴影内容键。

仅 `lodKey` 变化时，使旧、新 footprint 的合成深度失效；若 caster 仍属于静态层，也使对应静态深度失效。LOD 切换本身不会将静态 caster 晋升为动态。真正的内容、变换或阴影材质变化仍沿用原缓存分类逻辑。选择后的连续 slot 范围用于现有静态/动态 layer mask；提交与取消沿用原有缓存事务，未提交 cut 不成为下一帧的已提交缓存依据。多灯 temporal content key 也包含 `lodKey`。

## 诊断

两个 Pass 新增独立 `lodStatistics` 输出，固定 32 个 uint。它记录一次共享选择，不按光源面重复累计，也不改变现有 `statistics`、`workload` 或 `buildDiagnostics` 的字段含义。

| 字段 | 内容 |
|---|---|
| 0 | 格式版本，目前为 1 |
| 1 | `shadowLOD` 配置是否启用 |
| 2 | 汇总的 shadow projection 数 |
| 3 | CPU selector 测试的 group 数 |
| 4 | 已选择的可简化层级 cluster 数；stream 受保护 caster 的 terminal cluster 也计入 |
| 5 | 最终 geometry triangle slot capacity |
| 6 | 本次缺页需求中已分配/排队的页数，不等于新驻留页数 |
| 7 | 冻结或 admission 预算等原因阻塞的缺页数 |
| 8 | 无 resident hierarchy 或 stream 缺细化页的 caster 数 |
| 9 | MASK/displacement 受保护 caster 数 |
| 10 | 原始 base geometry triangle slot capacity |
| 11–15 | 保留，零 |
| 16–31 | 所选 cluster 的 LOD 0–15 直方图，15 包含更高层级 |

CPU resources 还保留 `(render node, primitive-local group, local cluster)` 列表以核查 cut。没有可选 hierarchy 的 resident caster 或材质保护的 resident caster 不生成伪造 cluster ID，因此 cluster/histogram 不能直接等同于所有 caster 的完整三角工作量。Stream triangle slots 按 cluster 的 128 槽 padding 计数；base producer 还可保留隐藏 instance 的候选，不能把 slot capacity 当成真正光栅化三角数。

`cacheDiagnostics[29]` 为 `lodChangedCasters`，只在兼容缓存的 caster 差分中累计。整缓存重建仍由全局原因位解释；该字段为零不保证没有发生 LOD 配置或投影变化。参见 [缓存分层说明](VirtualShadowMapCacheLayers.md)。

## 验证与限制

2026-10-31 本机 Release 验证通过：`Metallic`、`MetallicRHITests` 构建完成，shader module audit 为 387 files / 275 modules / 112 implementation units，`git diff --check` 通过。复用 build-release，仅启用该树的测试目标，没有更换编译器或 SDK。

- `virtual_shadow_map_shadow_lod`：真实 Bunny + plane，54 pages / 54 groups / 12 levels；resident/stream 各 7 状态，每状态对比 layered cache、single cache、force redraw。请求/准入集合逐页相同，可查询有效页 D32 与全屏 visibility 精确相同；最细 resident 与 authored base、terminal stream 与原 base 也精确相同。包含阈值切换、取消回滚、屏外 caster、解析投影误差及 16 种 DAG 缺页组合。
- `virtual_shadow_map_shadow_lod_pass`：实际 Single/Lights RenderGraph 路径，4 分支 × 3 帧；零误差与关闭 LOD 精确一致，读取 `lodStatistics`，验证方向光 + 点光的 35 个投影、移除点光后的 5 个投影及缓存重复帧。
- `virtual_shadow_map_cache_layers`：原缓存分层回归 76 状态 × 3 分支通过。以上三项 Vulkan validation count 均为 0。
- 主 stream 相机看向场景外且未提交主视图光栅可见列表，shadow 请求 52 个缺页；正常发布后 2 → 551 clusters，达到独立全驻留 CPU reference。raster-only fixture 使用 `freezeRasterSnapshot` 验证冻结（0 请求、52 blocked）；本次没有运行需要 BLAS 的 `freezeSourceResidency` 路径。

1 texel 配置下，resident 为 551 → 20 clusters、69,453 → 2,166 triangle slots；stream 为 551 → 20 clusters、70,528 → 2,560 padded slots。相对各自最细图，193 × 157 hard-shadow 输出分别有 45 / 69 个像素变化，归一化 visibility MAE 为 0.0014851 / 0.0022772。近/远 spot light 的 cluster 数为 73 / 4。已检查导出的阴影图与差分图：1 texel 保留主体轮廓，差异集中在边缘；这不是对任意场景或运动序列的误差保证。

证据位于本地 `build/vsm-shadow-lod/`：`Run3/virtual_shadow_map_shadow_lod`、`Run1/virtual_shadow_map_shadow_lod_pass`、`Run2/virtual_shadow_map_cache_layers`，以及 `Summary/report.md`、`Geometry.png`、`QualityDiffMontage.png` 与可重生成的 `Summarize.py`。前两次 direct 测试失败记录保留：首次把未准入缓存页的有效性也纳入精确比较；第二次在 raster-only fixture 使用了要求 RT 资源的冻结模式。修正了测试契约，没有放宽可查询深度或最终可见性比较。

未测 GPU 时间，不声称性能收益；尚未验证大型场景长时间驻留压力、连续运动的 LOD 跳变和软阴影质量。

需要区分三类比较：相同 cut 的 cached/force-redraw 可以检查精确一致；resident 零误差路径保留原三角顺序，可与原完整几何作精确基线比较；不同 LOD 的输出允许近似差异，需单独检查轮廓、细小遮挡、跨层切换和阴影采样结果，不能要求与最细几何逐像素相等。

这版继续准备 base 快照，CPU 遍历、顶点打包和上传可能抵消 build/raster 工作量下降。多灯共用所有投影中要求最细的 cut，未实现逐灯或逐页独立几何选择；stream 预算不足时也可能持续使用较粗回退。原有 build 的 dirty-page bitset、候选压缩与 indirect 路径继续消费该 cut，但减少候选或三角槽本身不证明 GPU 或整帧加速。性能结论必须来自同场景、同设置、相同预热与驻留条件下的受控 A/B。
