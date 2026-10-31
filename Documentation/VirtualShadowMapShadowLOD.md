# VSM 阴影专用 LOD

`VirtualShadowMap` 与 `VirtualShadowMapLights` 提供运行时属性 `shadowLOD`，默认 `false`；`shadowLODErrorTexels` 默认 `1.0`，界面范围为 0–16。误差单位为阴影虚拟纹理 texel，不是主相机屏幕像素。`0` 请求最细几何：resident 保留原始完整三角与顺序，stream 请求最细层级并在页到达前使用完整驻留回退 cut。

开启 `shadowLOD` 后默认使用 `shadowLODHierarchy=true` 的自适应层级遍历；设置 `shadowLODHierarchy=false` 可使用原平铺选择作为 A/B 对照。两个路径的误差预算、稳定 cluster ID 和缺页回退契约相同。

`shadowLODDepthGuard=true` 默认开启，在 texel 预算之外增加世界空间替换误差指标预算 `0.5 × shadowBias`。超限 group 继续细化，GPU 深度写入、接收面位置和比较 bias 均不改变；`shadowLODDepthGuard=false` 保留旧选择用于 A/B。它可能要求恢复完整几何，不能假定仍保留旧 LOD 的 GPU 收益。

本次加入独立的阴影几何选择，不改变现有 one-pass projection 的默认开启状态，也不改变静态/动态深度分层的默认关闭状态。入口见 [VirtualShadowMapCasters.h](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.h)，误差计算见 [VirtualShadowMapLOD.h](../Source/Runtime/Render/Shadow/VirtualShadowMapLOD.h)。

## 选择范围与误差

每次 Pass 先准备原始 caster 快照。resident 快照是完整 authored 三角，stream 快照是所有 terminal 分支的完整 cut。随后从完整 bounds 构造实际阴影投影的 X、Y、W 行及虚拟分辨率；多灯 Pass 汇总全部准入光源、点光六面、各 clipmap 层和 coarse 投影，仅选择一次共享 cut。

每个 group 使用所有投影中的最大误差。方向光按正交投影 texel 密度计算；局部光使用误差扩展球上的透视 Jacobian 上界，近裁剪面相交时保守要求细化。变换尺度使用包括非均匀缩放与 shear 的保守上界。方向光 coarse 的密度上界依据完整 caster bounds 的未 padding 范围，覆盖旧 coarse 缓存投影被复用时的情况。

深度保护复用同一变换尺度上界，将 `group.error` 转为世界单位；只对至少一个阴影视图有投影需求的 group 增加细化条件。层级节点仅在投影和世界误差预算均允许时剪枝，平铺与层级叶使用相同条件。零预算仍允许零误差替换，terminal sentinel 不参加误差计算。这里的 `group.error` 来自 meshoptimizer absolute-error QEM 并合并前级误差，描述下一次替换，不能视作任意接收面或掠射角上的严格深度误差上界。固定一半 bias 的策略为原光栅误差保留余量，仍需图像验证。

该需求独立于主相机的 visible/active cut，不以主相机屏幕外为由删除投影 caster。某个球位于一个阴影视图以外时，仅该视图的细化需求为零。最终遍历仍覆盖全部 terminal 分支；需要细化的 group 会把需求向所有 DAG 祖先传播，替换仅在所有 parent owner 生效时发生，以保留多父关系的完整覆盖。

当前实现为 CPU 自适应层级选择与 CPU 打包上传。resident 利用已有 `meshletLodGroups/Clusters/Vertices/Triangles`，stream 使用资产完整 group/refinement 元数据，不复用主视图压缩列表。尚未加入 GPU LOD 遍历或 cut hysteresis。

## 自适应层级遍历

每个 primitive 在首次需要 LOD 时创建不可变元数据：原始 group/range/refinement、去重父关系、cluster→group 映射以及非 terminal group 的 BVH4。Pass 持有缓存，scene identity、结构版本或 stream geometry publication 变化时更换；transform/material 改变不重建 object-space 元数据。SceneLoader 完成 deferred meshlet 构建及 finalize 后才发布场景；旧录制通过 resource snapshot 保留自己的缓存生命期。

遍历先用所有准入光源的最大投影误差上界判断子树，再对未剪枝的叶 group 使用原误差函数。透视 group 度量在合并球上并不单调，因此节点上界分别约束 error、球半径、最小 W、clipXY 和原 Jacobian 的各项，而非直接把合并球代入 group 公式。近面穿越保守下降，大坐标计算使用向外余量，阈值边界只允许增加遍历。Terminal stop sentinel 不进入 BVH，所有 terminal 分支始终单独纳入需求。

命中的需求沿缓存父链求闭包，再按 group ID 降序进行 all-parent 驻留激活；缺页请求不受父页是否就绪限制。仅遍历实际 active 的 cluster range，并以原升序输出稳定 cut。零误差直接请求所有 group，跳过误差计算。Stream 打包通过缓存的 cluster→group 映射访问选中 cluster，不再完整扫描所有 cluster；resident 最细路径也不生成最终不会上传的替换顶点数组。

仍然保留 dense active/wanted 数组，stream 每个实例仍需读取全部 group 的 drawable residency。因此减少的是昂贵误差计算、拓扑边遍历、元数据重建及完整 cluster 重扫，不能称为消除了全部线性工作。本版不包含 GPU 层级遍历、LOD hysteresis 或逐页独立 cut。

## 材质与 streaming

MASK 与 displacement caster 保留基准几何；这一版本不对 alpha 覆盖或位移误差做额外简化。没有 resident hierarchy 的 primitive 同样保留完整原始三角，并计入 fallback。这里的 stream 基准是原有完整 terminal cut，不表示所有原始三角已驻留。

Stream 只从 `pageResident()` 为真的 group 选 drawable cut，缺页会保留安全的驻留祖先，不把缺页当成空几何。需要的细化页通过 `requestShadowPages` 独立请求，选中的驻留页先 touch；资源录制先登记 geometry-use completion，页范围不能在本次 GPU 使用完成前回收。页仍沿用普通 stream 预算、异步加载及 publication 路径，不永久锁住 refined 页。

深度保护提高细化需求，但不绕过驻留约束：细页缺失时仍保留完整祖先 cut，因此启用 guard 不表示当前回退几何已经满足预算。不得为了满足预算而丢弃 caster。

请求成功或分配成功不表示页已经 drawable，真正细化需要后续正常 frame 完成页发布。`freezeSourceResidency` 或 `freezeRasterSnapshot` 生效时不新增 shadow 请求或刷新驻留，缺页需求计入 blocked，当前可绘制 cut 继续使用。需要按现有 runtime 契约在 frame 的页发布后消费 shadow geometry；shadow producer 本身不替代主 stream 的更新与 publication 阶段。

## 缓存身份

选中 cut 的 cluster ID 序列单独形成每个 caster 的 `lodKey`，原始 cut 使用零。`contentKey`、`transformKey`、`materialKey`、完整 caster bounds 与几何 publication 依赖保持来自 base authored 快照；不使用全局 page-table mutation serial 作为阴影内容键。

仅 `lodKey` 变化时，使旧、新 footprint 的合成深度失效；若 caster 仍属于静态层，也使对应静态深度失效。LOD 切换本身不会将静态 caster 晋升为动态。真正的内容、变换或阴影材质变化仍沿用原缓存分类逻辑。选择后的连续 slot 范围用于现有静态/动态 layer mask；提交与取消沿用原有缓存事务，未提交 cut 不成为下一帧的已提交缓存依据。多灯 temporal content key 也包含 `lodKey`。

## 诊断

两个 Pass 新增独立 `lodStatistics` 输出，固定 32 个 uint。它记录一次共享选择，不按光源面重复累计，也不改变现有 `statistics`、`workload` 或 `buildDiagnostics` 的字段含义。

| 字段 | 内容 |
|---|---|
| 0 | 格式版本，目前为 2；字段 11–15 新增层级统计 |
| 1 | `shadowLOD` 配置是否启用 |
| 2 | 汇总的 shadow projection 数 |
| 3 | CPU selector 实际访问的拓扑 group 数；平铺为全部 group，层级为需求闭包 |
| 4 | 已选择的可简化层级 cluster 数；stream 受保护 caster 的 terminal cluster 也计入 |
| 5 | 最终 geometry triangle slot capacity |
| 6 | 本次缺页需求中已分配/排队的页数，不等于新驻留页数 |
| 7 | 冻结或 admission 预算等原因阻塞的缺页数 |
| 8 | 无 resident hierarchy 或 stream 缺细化页的 caster 数 |
| 9 | MASK/displacement 受保护 caster 数 |
| 10 | 原始 base geometry triangle slot capacity |
| 11 | 执行原始 group 误差测试的非 terminal group 数 |
| 12 | 层级 BVH 节点访问数；平铺为零 |
| 13 | 子树剪枝跳过的 group 误差测试数；零误差 fast path 不计入 |
| 14 | 本次新建的 primitive 元数据数量 |
| 15 | 本次复用的 primitive 元数据次数，按实例消费计数 |
| 16–31 | 所选 cluster 的 LOD 0–15 直方图，15 包含更高层级 |

CPU resources 还保留 `(render node, primitive-local group, local cluster)` 列表以核查 cut。没有可选 hierarchy 的 resident caster 或材质保护的 resident caster 不生成伪造 cluster ID，因此 cluster/histogram 不能直接等同于所有 caster 的完整三角工作量。Stream triangle slots 按 cluster 的 128 槽 padding 计数；base producer 还可保留隐藏 instance 的候选，不能把 slot capacity 当成真正光栅化三角数。

`cacheDiagnostics[29]` 为 `lodChangedCasters`，只在兼容缓存的 caster 差分中累计。整缓存重建仍由全局原因位解释；该字段为零不保证没有发生 LOD 配置或投影变化。参见 [缓存分层说明](VirtualShadowMapCacheLayers.md)。

## 层级版本验证

2026-10-31 最终 Release 构建与 `git diff --check` 通过。本次没有修改 shader 或 GPU 资源布局。`build/vsm-shadow-hierarchy/Run2/` 下两个测试通过，Vulkan validation count 均为 0：

- `virtual_shadow_map_shadow_lod`：768 个多父 DAG/缺页/变换/阈值组合、256 个真实 Bunny 确定性案例及 32 个大平移消减案例，与平铺参考的 selected/requested/active 集合一致。resident/stream 共 14 状态，每状态四路（layered、single、forced hierarchy、forced flat）比较，cut 身份、可查询 D32 和最终 visibility 精确一致；仍覆盖独立屏外 streaming、冻结/解冻、LOD 失效与取消。
- `virtual_shadow_map_shadow_lod_pass`：真实 Single/Lights graph 的 Off、Zero、Flat、Adaptive 八分支 × 三帧，验证默认层级路径、平铺开关、v2 统计、重复帧 metadata hit，以及点光移除后准入投影数变化。

固定 Bunny 输入、预建 metadata 的 CPU selector 微基准每 workload 为 12 个 ABBA 区组、每臂 128 次调用，32 次预热；所有区组保留。正交 1 texel 的单次调用中位数为平铺 8.019 μs、层级 1.208 μs，group 误差测试 52 → 7，拓扑访问 53 → 8。五种 workload 的 paired ratio 中位数为 0.095–0.400；冷 metadata 构建另计，中位数 14.5 μs。数值仅覆盖该 CPU selector，不含打包、上传、Pass 或 GPU；没有整帧性能结论，也没有跨进程置信区间。

图表、IQR、完整原始数据与脚本位于 `build/vsm-shadow-hierarchy/Summary/report.md`、`CPUTraversal.png`、`WorkCounts.png` 和 `Summarize.py`。初轮 Run1 也保留；最终 Run2 包含数值余量及大平移回归。不同 LOD 的质量仍沿用下方初版边界，层级优化本身不改变选中 cut。

## 自阴影修复与正确性、性能 A/B

2026-10-31：旧 LOD 只限制阴影平面 X/Y 误差，简化表面可能位于主视图高精度接收面之前，形成错误自遮挡。修复增加上述世界误差预算，拒绝不满足深度一致性策略的替换；没有修改 shader、GPU 资源布局或比较 bias。选择结果继续使用既有 `lodKey` 失效，切换 guard 后相同 cut 不重复失效、不同 cut 使对应深度缓存重绘。

Release `Metallic` 和 `MetallicRHITests` 构建通过，以下三个 GPU 回归均通过且 VVL=0：

- `virtual_shadow_map_shadow_lod`：新增 12 个预算边界/变换/缺页/投影案例、1,536 个有限预算多父 DAG 案例、1,024 个真实 Bunny 平铺/层级等价案例；resident/stream 共 18 状态，每状态维持 layered/single/forced hierarchy/forced flat 四路比较，包含 guard 切换后的缓存失效与暖缓存。
- `virtual_shadow_map_shadow_lod_pass`：Single/Lights 各 Off、Zero、Flat、Adaptive、Legacy、Guarded 六臂 × 3 帧，验证默认 guard 与显式开启一致、旧开关可用。该小 fixture 没有非零误差替换，不作为 Bunny 自阴影质量证明。
- `virtual_shadow_map_receiver_quality`：方向/spot/point、斜面/掠射/凸凹面、clear/contact/separated 的 36 case / 108 frame 原有 authored-triangle oracle 回归通过；此测试不产生真实 LOD，验证的是既有 receiver/filter 契约没有回归。

生产路径复现使用 RTX 5070 Ti / driver 621.01，768 × 512，Bunny + plane，主视图固定最细且 54 页全驻留。三臂为完整几何 Off、1 texel Legacy（guard=false）、1 texel Guarded（guard=true），其余输入相同。六个诊断进程各 48 帧 / 32 预热，frame 32/47 导出原始 R8 shadow、D32 receiver depth、R32 visibility 和浮点 HDR，保留格式/stride/hash；没有为普通计时增加读回。

以同状态 Off 的阴影可见性为参考，Bunny 接收区域由其位于 z=0 plane 之前的实际主深度识别。下表为每个稳定捕获帧的像素数，dark 指参考受光而候选遮挡，lost 指参考遮挡而候选受光；这些是原始阴影像素，与旧报告的最终着色 PNG 差异数不是同一指标。

| 状态 / 臂 | Bunny 新增暗像素 | Bunny 丢失遮挡 | 受光内部新增暗像素 | 阴影内部漏光（Bunny / plane） |
|---|---:|---:|---:|---:|
| 重绘 / Legacy | 1,077 | 349 | 606 | 7 / 0 |
| 重绘 / Guarded | 0 | 0 | 0 | 0 / 0 |
| 暖缓存 / Legacy | 1,078 | 349 | 607 | 7 / 0 |
| 暖缓存 / Guarded | 0 | 0 | 0 | 0 / 0 |

预声明质量门槛使用参考受光/遮挡 mask 的 3×3 erosion 区分内部和边界，要求内部新增暗点/漏光均为 0；全部边界仍计入全图指标并画出，没有丢弃差异。实际 Guarded 更强：两种缓存状态、两个捕获帧的完整 shadow、depth、visibility、HDR 和最终输出均与 Off 逐字节相等。Legacy/Guarded 的暖缓存另在 frame 47 保持 `cacheEnabled=true` 做 force-redraw 对照，shadow/HDR 精确一致。六臂 VVL=0、overflow=0、coarse receiver fallback=0。

这份 Bunny 的最小首级替换指标约 0.001291 m，高于本次 `shadowBias=.002` 导出的 0.001 m 预算，Guarded 恢复到 551 clusters / 69,453 triangles；Legacy 为 146 clusters / 17,348 triangles。Guarded 的 GPU 工作量与 Off 相同，消斑来自恢复必要几何。原始完整几何本身已有的阴影采样差异不属于这次 LOD 修复。

普通计时固定每 workload 3×ABCCBA、共 36 独立进程，每进程 96 帧、32 帧预热；validation、工作量计数、CPU profile 关闭，只用标准 GPU stage 时间戳。数据/工具/源码/二进制/资产身份固定；磁盘缓存保留，未锁时钟或停止用户程序。没有追加性能样本。下表为六个进程中位数的中位数，单位 ms：

| 状态 / GPU 范围 | Off 完整几何 | Legacy 旧 LOD | Guarded 修复后 |
|---|---:|---:|---:|
| 重绘 / Build | 0.043072 | 0.034304 | 0.043064 |
| 重绘 / Raster | 1.021424 | 0.279736 | 1.020744 |
| 重绘 / Shadows | 1.149560 | 0.398208 | 1.148360 |
| 重绘 / RenderGraph | 1.713696 | 0.940360 | 1.693200 |
| 暖缓存 / Shadows | 0.110032 | 0.112216 | 0.111400 |
| 暖缓存 / RenderGraph | 0.661848 | 0.688192 | 0.690808 |

以完整 ABCCBA 区组的 log ratio 为单位，95% CI 使用 t(df=2)=4.303：Guarded/Legacy 重绘 Shadows **+188.37% [183.48%, 193.35%]**，Raster +265.07% [258.61%, 271.65%]；Guarded/Off 重绘 Shadows **−0.20% [−0.90%, +0.50%]**、RenderGraph +0.21% [−5.89%, +6.70%]。暖缓存 Guarded/Legacy Shadows −0.82% [−2.02%, +0.40%]，Guarded/Off +1.22% [−0.74%, +3.22%]，没有明确变化。普通 GPU graph span 不含 present，不能改述为应用 FPS。

结论：修复通过该复现场景的图像和缓存验收，但它不再保留此前约 65% 的重绘收益；收益来自接受更粗几何，而本场景该近似不满足质量预算。`shadowLOD` 整体仍默认关闭，启用它时 depth guard 默认开启。尚未证明任意场景/运动/缺页期间无斑，也没有证明保持低模成本的通用自阴影修复；后者需要接收面与阴影表示的一致性方案。

证据位于 `build/vsm-lod-self-shadow/`：`FinalVerification.json`、`Regression/`、`Diagnostic/`、`Timing/`、`Quality/Report.md` 与原始可见性对照图、`Summary/Report.md` 与 GPU 图表。可重生成脚本为仓库 `Tools/Perf/VirtualShadowMapLODSelfShadow.py`，以及本地 `AnalyzeQuality.py` / `VerifyAll.py`。维护的 runner 提供 `prepare`、`run --phase diagnostic|timing`、`verify`、`report`；它复用前轮固定 fixture，普通计时必须引用诊断包，质量门槛另由 `AnalyzeQuality.py` 验证。

## 深度保护前的 GPU 影响验证

以下保留新增 depth guard 之前的历史结果，不能代表当前默认保护路径的性能。2026-10-31，RTX 5070 Ti / driver 621.01，Release，768 × 512，真实多级 Bunny + plane。阴影输入为完整 resident authored 几何；主视图固定最细级，预建 stream 的 54 页全部驻留，geometry 3,171,840 bytes，frame 16 冻结驻留。相机、光源、资源内容、接收面深度与流送身份在各臂相同，没有 RT/CLAS 分配。单方向光只有一个 face，`onePassProjection=true` 按既有默认实现走单面 projection。

三臂为关闭 LOD、1 texel 平铺、1 texel 层级。分别测 `cacheEnabled=false` 的每帧重绘与 `cacheEnabled=true` 的静止暖缓存。每种 workload 固定 3 个 ABCCBA 区组，共 36 个独立计时进程；每进程 96 帧、前 32 帧预热。普通计时关闭 validation、工作量原子计数、CPU profile 和 build substage，只保留标准 GPU stage 时间戳。磁盘缓存保留，不停止用户程序，不锁 GPU 时钟。区组内使用两个进程中位数的平均 log ratio，95% 区间使用 3 个区组、t(df=2)=4.303；帧不作为独立样本，未追加性能样本。

下表是六个进程中位数的中位数，单位 ms；RenderGraph span 不包含 present，也不能转换为应用 FPS 收益：

| 状态 / GPU 范围 | Off | Flat | Hierarchy |
|---|---:|---:|---:|
| 重绘 / RenderGraph | 1.987000 | 1.176408 | 0.987728 |
| 重绘 / Shadows | 1.160032 | 0.406712 | 0.407472 |
| 重绘 / Mark | 0.027528 | 0.027944 | 0.027952 |
| 重绘 / Build | 0.047152 | 0.036064 | 0.036064 |
| 重绘 / Raster | 1.025752 | 0.284840 | 0.284992 |
| 重绘 / Projection | 0.043840 | 0.043600 | 0.043640 |
| 暖缓存 / RenderGraph | 0.677320 | 0.668816 | 0.666000 |
| 暖缓存 / Shadows | 0.110144 | 0.110504 | 0.110112 |

重绘时 Hierarchy/Off 的 Shadows 为 **−64.91%**，95% CI **[−65.01%, −64.81%]**；Build 为 −23.60% [−26.07%, −21.04%]，Raster 为 −72.23% [−72.43%, −72.04%]，RenderGraph 为 −48.52% [−55.25%, −40.76%]。收益主要来自减少 raster 几何，不代表 Mark/allocator 同样减少；Mark 约增加 0.4 μs。这里仅证明该固定 fixture 的 GPU 时间变化。

层级相对平铺的重绘 Shadows 为 **+0.06% [−0.36%, +0.47%]**，RenderGraph 为 −10.21% [−27.78%, +11.65%]；均不能证明额外 GPU 加速。两者提交相同 cut，层级优化减少 CPU 选择工作。暖缓存 Hierarchy/Off 的 Shadows 为 −0.04% [−3.26%, +3.27%]、RenderGraph 为 −0.30% [−6.48%, +6.28%]，同样没有明确收益。暖缓存三臂 dirty/build/raster 有效工作均为零，相关 GPU span 是空 dispatch、同步等固定开销。

独立诊断为六进程 × 48 帧，前 32 帧预热，frame 32/47 保存输出；均无 VVL 消息、work overflow 或 coarse receiver fallback。Off→LOD 的实际计数如下，Flat 与 Hierarchy 完全一致：

| 每帧重绘工作 | Off | LOD |
|---|---:|---:|
| 选中 triangle slots / build 解码三角 | 69,453 | 17,348 |
| Build 线程组 | 1,086 | 272 |
| Cluster level 测试 | 4,344 | 1,088 |
| Cluster dirty-bitset word 测试 | 3,270 | 828 |
| Fine raster 工作项 | 211,611 | 54,157 |
| Raster 解码三角 | 281,064 | 71,505 |

接收像素均为 376,832，请求/分配页均为 52；LOD 选中 146 clusters。层级将 CPU group 误差测试 50→20、拓扑访问 52→18，但不改变 GPU 工作。两种缓存状态内，Flat/Hierarchy 的深度、阴影、HDR、最终输出哈希完全相同；日志提供的 Mark/Build/Project SPIR-V 在同计数模式的各臂相同。

`cacheEnabled` 会改变方向光深度 padding 和 coarse 投影范围，因此重绘/暖缓存不是相同投影的精确缓存对照；本例 LOD 最终 PNG 跨这两种状态相差 1 像素。另加两个仅正确性诊断进程，在缓存保持开启的 Flat/Hierarchy 上，使用 frame 47 内置 `forceRedraw` 比较：几何/可见性/深度身份稳定，shadow changed bytes=0、HDR byte equal，VVL=0。这两个进程不进入性能统计。

已查看质量对照图。1 texel LOD 相对 Off 在重绘最终图改变 1,356 像素（全图 0.345%，非黑区域并集约 9.36%），Bunny 自阴影斑点增加；plane 最终着色为黑色，当前图不能证明其投影轮廓质量。没有连续运动、复杂场景或软阴影质量保证。`shadowLOD=false` 默认保持不变；不因本例 GPU 收益忽略质量代价。

证据位于 `build/vsm-lod-gpu-impact/`：`Diagnostic-v2/`、`Timing/`、`CacheEquivalence-Flat/`、`CacheEquivalence-Hierarchy/`、`Summary/Report.md`、`Summary/Redraw-GPU.png`、`Summary/Warm-GPU.png` 和 `QualityReview/BunnyShadowLODComparison.png`。初次诊断目录 `Diagnostic/` 保留：渲染已通过，脚本误要求单 face 场景编译 one-pass shader，修正路径验收后重跑；没有修改 renderer。

复用工具为 `Tools/Perf/PrepareVirtualShadowMapLODImpact.py` 和 `Tools/Perf/VirtualShadowMapLODImpact.py`。准备器依赖本地 `build/vsm-shadow-hierarchy/Run2/.../VirtualShadowMapLOD` 的真实 fixture；所有输出目录必须新建。`run --phase diagnostic` 通过后才可 `run --phase timing --diagnostic <diagnostic-directory>`；`verify` / `report` 可从原始 receipt 重建验证与图表。额外缓存等价诊断使用 `VirtualShadowMapAcceptance.py` 和已保存的 `CacheEquivalenceCases/`。

## 初版验证与限制

2026-10-31 本机 Release 验证通过：`Metallic`、`MetallicRHITests` 构建完成，shader module audit 为 387 files / 275 modules / 112 implementation units，`git diff --check` 通过。复用 build-release，仅启用该树的测试目标，没有更换编译器或 SDK。

- `virtual_shadow_map_shadow_lod`：真实 Bunny + plane，54 pages / 54 groups / 12 levels；resident/stream 各 7 状态，每状态对比 layered cache、single cache、force redraw。请求/准入集合逐页相同，可查询有效页 D32 与全屏 visibility 精确相同；最细 resident 与 authored base、terminal stream 与原 base 也精确相同。包含阈值切换、取消回滚、屏外 caster、解析投影误差及 16 种 DAG 缺页组合。
- `virtual_shadow_map_shadow_lod_pass`：实际 Single/Lights RenderGraph 路径，4 分支 × 3 帧；零误差与关闭 LOD 精确一致，读取 `lodStatistics`，验证方向光 + 点光的 35 个投影、移除点光后的 5 个投影及缓存重复帧。
- `virtual_shadow_map_cache_layers`：原缓存分层回归 76 状态 × 3 分支通过。以上三项 Vulkan validation count 均为 0。
- 主 stream 相机看向场景外且未提交主视图光栅可见列表，shadow 请求 52 个缺页；正常发布后 2 → 551 clusters，达到独立全驻留 CPU reference。raster-only fixture 使用 `freezeRasterSnapshot` 验证冻结（0 请求、52 blocked）；本次没有运行需要 BLAS 的 `freezeSourceResidency` 路径。

1 texel 配置下，resident 为 551 → 20 clusters、69,453 → 2,166 triangle slots；stream 为 551 → 20 clusters、70,528 → 2,560 padded slots。相对各自最细图，193 × 157 hard-shadow 输出分别有 45 / 69 个像素变化，归一化 visibility MAE 为 0.0014851 / 0.0022772。近/远 spot light 的 cluster 数为 73 / 4。已检查导出的阴影图与差分图：1 texel 保留主体轮廓，差异集中在边缘；这不是对任意场景或运动序列的误差保证。

证据位于本地 `build/vsm-shadow-lod/`：`Run3/virtual_shadow_map_shadow_lod`、`Run1/virtual_shadow_map_shadow_lod_pass`、`Run2/virtual_shadow_map_cache_layers`，以及 `Summary/report.md`、`Geometry.png`、`QualityDiffMontage.png` 与可重生成的 `Summarize.py`。前两次 direct 测试失败记录保留：首次把未准入缓存页的有效性也纳入精确比较；第二次在 raster-only fixture 使用了要求 RT 资源的冻结模式。修正了测试契约，没有放宽可查询深度或最终可见性比较。

初版未测 GPU 时间；后续固定 fixture 的测量见上方“GPU 影响验证”。仍未验证大型场景长时间驻留压力、连续运动的 LOD 跳变和软阴影质量。

需要区分三类比较：相同 cut 的 cached/force-redraw 可以检查精确一致；resident 零误差路径保留原三角顺序，可与原完整几何作精确基线比较；不同 LOD 的输出允许近似差异，需单独检查轮廓、细小遮挡、跨层切换和阴影采样结果，不能要求与最细几何逐像素相等。

这版继续准备 base 快照，CPU 遍历、顶点打包和上传可能抵消 build/raster 工作量下降。多灯共用所有投影中要求最细的 cut，未实现逐灯或逐页独立几何选择；stream 预算不足时也可能持续使用较粗回退。原有 build 的 dirty-page bitset、候选压缩与 indirect 路径继续消费该 cut，但减少候选或三角槽本身不证明 GPU 或整帧加速。性能结论必须来自同场景、同设置、相同预热与驻留条件下的受控 A/B。

## 保留简化几何的 receiver 修正（2026-10-31，可选）

新增 `shadowLODSmartBias`（默认 false）和 `shadowLODScreenRayLength`（默认 0.015，receiver 处竖直 view extent 的比例）。显式使用 `shadowLOD=true, shadowLODDepthGuard=false, shadowLODSmartBias=true` 才是本次保留简化几何的候选；原有默认值保持不变。

参考本机 Unreal **5.7.4** 的 `VirtualShadowMapScreenRayTrace.ush:12`、`VirtualShadowMapProjectionCommon.ush:411–474`：屏幕深度限定 receiver 附近的 ray-start offset，方向光只改变比较深度，不改变原 receiver 的 clipmap/page。UE 的 `r.Shadow.NaniteLODBias` 默认 1，确实同时使用简化阴影几何；不能把其传统 atlas 的 `NaniteEmitShadow.usf` depth bias 误当成 VSM LOD 专用修复。Epic 也说明过大的 normal bias 会损失细节，参见 [官方 VSM 文档](https://dev.epicgames.com/documentation/unreal-engine/virtual-shadow-maps-in-unreal-engine)。

Metallic 实现只作用于**有效 Hard 模式的方向光**：先查 atlas，已有遮挡且位于修正距离内时才做最多 4 次主深度采样。搜索区间由当前深度冲突和已有世界 bias 缩短，receiver-plane 高度检查保留可能被采样步长跨过的薄遮挡；屏外、裁剪面外和无效朝向保留原阴影。PCF、SMRT、spot、point 不应用此修正。开关与长度改变会重置 temporal history，无需因 projection 参数重绘 atlas。这里是基于 UE 思路的适配，不是其公式、单位或质量保证的直接复制。

RTX 5070 Ti / 621.01、Release、固定 resident Bunny、768×512、finest 主视图。Legacy/Smart 仍严格相同：**146 clusters / 17,348 triangles**，完整几何为 69,453 triangles，Mark/Build/Raster 工作量不变。没有全几何回退或额外 AS 构建。36 个普通计时进程，Redraw/Warm 各 3 个 ABCCBA 区组、每进程 96 帧/预热 32 帧，诊断和 validation 关闭，不追加样本；磁盘缓存保留，时钟未锁定。

| GPU 范围，六个进程中位数的中位数（ms） | Off | Legacy | Smart |
|---|---:|---:|---:|
| 重绘 Shadows | 1.150280 | 0.402912 | 0.403392 |
| 重绘 Projection | 0.044088 | 0.043648 | 0.045288 |
| 暖缓存 Shadows | 0.107544 | 0.107544 | 0.108360 |

配对区组 Smart/Off 重绘 Shadows **−65.05% [−65.58%, −64.52%]**；Smart/Legacy Shadows **−0.06% [−3.01%, +2.97%]**，保留 LOD 收益。Projection 额外约 **1.64 μs**，区组变化 **+3.61% [+3.25%, +3.97%]**。暖缓存 Smart/Off Shadows **+1.02% [−0.33%, +2.40%]**，没有明确改善。95% CI 以三个完整区组而不是帧为样本，t(df=2)=4.303；不是应用 FPS 结论。原 atlas load 计数不包含新增主深度读取。

原视角、反向光照、近视角分别完成 6 个诊断进程，三臂主 depth/visibility 相同，Legacy/Smart 工作一致；同投影 cached/forceRedraw shadow/HDR 精确相同，VVL=0。相对 Off 的内部和地面边缘门槛通过，但 Off 本身不是独立真值。

新增独立 authored-Bunny CPU BVH oracle：69,453 三角形，camera/shadow brute-force 子集交叉检查无错误。23,947 个 Bunny 像素中 **23,917** 满足原 `2e-5` 主深度阈值；另 **30** 个单列 unresolved，量化与光栅差异未完全归因，没有降低阈值。ε=0/1e-6/1e-5 m 指标一致，1e-4 m 仅一个像素敏感。Redraw frame47 的真实误遮挡 **Off 880 / Legacy 1,677 / Smart 318**；真实漏影 **602 / 669 / 732**。Smart 修复 **1,359** 个真实 acne，同时新增 **63** 个真实漏影。相对 Off 的 778 个黑→白像素中，633 个修复原有 acne、142 个丢失真实遮挡、3 个 unresolved，不能只用 erosion 或“暗点减少”宣布正确。

独立 5/10/20/40 mm 零厚度 blocker 测试包含 Nominal、实际 120 mm Stress、关闭和零长度对照，以及 all-lit/all-dark 负对照；最终 false-shadow/leak 均为 0，缓存与重绘一致。早期两个候选的漏光失败保留在 `Contact1/Contact2`，最终为 `Contact3`，没有放宽门槛。Release 构建、shader module audit、receiver quality、LOD pass、temporal lifecycle、多灯 projection equivalence 回归通过；后者覆盖 Hard/PCF/SMRT 和默认关闭路径，VVL=0。

**结论：保留可选实现，不默认启用，不宣布无损修复。** 屏外/隐藏/亚像素遮挡、连续运动和大型场景仍需验证；下一步先解决已定位的 63 个新增真实漏影，不能继续扩大 bias 或退回完整几何来掩盖问题。

工具：`Tools/Perf/VirtualShadowMapLODSmartBias.py`、`VirtualShadowMapLODSmartBiasQuality.py`、`VirtualShadowMapLODSmartBiasOracle.py`。证据：`build/vsm-lod-smart-bias/{Diagnostic2,DiagnosticReverse,DiagnosticClose,Contact3,Oracle2,Timing,Regression}`；汇总、Python 图表与重验脚本在 `Summary/{Report.md,plot_report.py,verify_all.py,Verification.json}`。图像/捕获保留为本地输出，不纳入源码。
