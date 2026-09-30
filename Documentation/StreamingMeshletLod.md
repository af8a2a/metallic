# 统一流式 Meshlet LOD

`VisibilityBufferPass` 的常驻和 StreamAsset 几何，以及独立 `GPUDrivenStreamAssetPass`，共用 `MeshletLodMetric.slang` 的屏幕误差公式和四个运行时设置：`autoLod`、`lodPixelError`（默认 1.5 display px）、`lodBias`（默认 0）、`lodLevel`。旧 `enableGpuLodSelection`、`selectedLodLevel` 配置仍可读取，新键优先；可视化模式不再决定是否启用自动选择。

像素预算以最终显示视口高度为基准。RenderGraph 的 `displayWidth()/displayHeight()` 与 pass 的内部 `width()/height()` 分开，即使使用局部相机也保留图输出尺寸。`MeshletStreamFrameDesc::displayHeight` 传入显示高度，零值供独立调用者按原生分辨率回退。CPU 通过 `meshletLodRenderPixelThreshold()` 将带 bias 的显示像素阈值乘以 `renderHeight / displayHeight`，GPU 的 viewport、光栅、HZB 和抖动保护仍使用内部像素。需求、预取和最终 cut 使用同一个换算后的阈值；默认数值 1.5 和 cook 格式不变。DLSS 降低内部尺寸时不再自动放宽几何误差，旧图可能因此选择更多几何。

误差阈值范围 0.05–16，bias 范围 -4–4，有效阈值为 `error * exp2(bias)`。正交、近裁剪面、偏轴和非均匀实例变换与常驻路径一致。冻结相机只固定裁剪/LOD 相机，光栅及 Deferred 使用实时渲染相机；内部尺寸变化时重捕获冻结相机。两条光栅路径使用相同的 temporal jitter。

## 元数据与资产兼容

- StreamAsset v9 的 group 元数据为 44 字节：原 36 字节之后增加每 cluster 的 refinement offset 和 terminal flags。
- `refinedGroups()` 按 cluster 保存全局 group ID，独立于 payload 页驻留；运行时据此构造去重的父组 CSR。
- `primitiveTerminalGroups()` 返回完整终端集合，包括在较细 LOD 提前终止的分支。旧连续 fallback 字段仅作格式兼容；光栅和 CLAS fallback BLAS 使用新的完整集合。
- v9 打开时不加载几何 payload。v8 可读取，但首次打开必须解码各页以恢复拓扑；大资产建议重建为 v9。旧 partial checkpoint 自动失效，新 partial v8 保存拓扑。
- 检查引用范围、同 primitive、`refined < owner`、严格降低的 LOD、误差单调和 terminal 标记。当前 builder 合并前代 group sphere，保证全驻留条件下的投影误差单调。

## 有效 frontier

```text
desired(g) = terminal(g) OR projectedError(g) > targetPixels
reachable(g) = terminal(g) OR all(active(parent) for parent in parents(g))
active(g) = desired(g) AND reachable(g) AND drawable(page(g))
emit(cluster) = active(owner) AND (no refined OR NOT active(refined))
```

手动模式以 `group.level >= lodLevel` 替代误差比较。组可以有多个父组；所有父组必须生效，才能原子替换它们对应的粗表示。存在缺失祖先时，即使更细页仍驻留，也不能绕过祖先参与输出。细节页需求独立于祖先驻留状态，按屏幕误差与视图需求提前请求；可绘制 cut 仍遵守完整祖先依赖。PendingUpload 不可绘制，也不会重复请求加载。

每个 primitive 的所有终端页就绪前，不发布不完整 cut。初始化按全部实例（含隐藏实例）预留终端页和输出容量，预算不足返回明确错误，不跳过部分几何。页面预算还预留一个普通流式页（资产全为终端页时不需要）。

GPU 依次执行 reset → sparse clear / demand seed → distributed demand → ordered frontier → mask → prefix → sparse emit → indirect finalize。需求阶段通过全局有界子树队列和 wave 内可变长度分配跨实例并行；安全 frontier 仍按父级优先拓扑评估组，输出按 instance/group 顺序稳定排列。压缩后的 active group 和 cluster mask 继续转换为现有 `(instanceId, clusterId)` 可见记录，供 cluster 分箱、早晚 HZB、Mesh Shader 和异步软件光栅消费。关闭 `distributedPageDemand` 或队列容量不足时使用 ordered demand。实现与测量见 [StreamWorkDistribution.md](StreamWorkDistribution.md)。

精细输出超过容量时，prefix 在写出记录前按实例分配预算，使超出预算的实例退回完整终端 cut；连终端容量也不足的非法 GPU 输入产生空输出和错误标记，禁止越界写入。正常运行时初始化已拒绝该预算。CLAS 动态 BLAS 使用选中 mask，备用 BLAS 使用同一终端集合，备用 BLAS 存储不足也明确失败。

## BVH 遍历加速

运行时从常驻 group 元数据构建每 primitive 的有序 BVH4，随拓扑上传一次，无需读取 payload 或修改 v8/v9 资产格式。每个节点 40 字节，包含包围球、最大误差、最大 LOD、终止标记 OR、逃逸索引及叶子 group 范围；叶子最多 4 个 group。

节点采用前序布局，叶子按 group ID 降序访问。聚合球用双精度构建并将半径向外舍入，节点的投影误差是子组误差的保守上界。透视判断还根据矩阵、坐标范围和平移量增加世界空间半径余量，避免大坐标浮点消减使变换后的父球漏掉子球；该余量只用于 BVH。节点误差满足阈值时直接跳到逃逸索引；手动模式按最大 LOD 剪枝。终止标记确保完整回退集合始终被访问。浮点比较留有保守余量，叶子仍使用原来的精确选择条件。

这个顺序保证所有父组先于子组生效，不需要有限容量的遍历栈或排序；共享父组、跨层终止、缺页与 PendingUpload 的规则均保留。此 BVH 按误差裁剪，视锥/HZB 继续由下游光栅流程处理，因而加速前后的完整 cut、请求和稳定输出顺序一致。未使用 BVH 的线性路径保留为差分测试对照。

frontier 同时记录所有 active group 的稀疏列表，包括已被细化替代的祖先。mask 计算及 emit 只遍历当前列表；每帧只清除前一帧列表中的状态，随后再处理隐藏或恢复，避免旧祖先错误放行后代。首次使用通过 phase 8 初始化设备状态缓冲，支持跨越单行 dispatch 上限的二维派发。

## 调试与验证

现有 `AfterTraversal` / `AfterPass` checkpoint 提供：

- `streaming.<pass>.activeHeader`：live count、capacity、overflow、terminalFallback、invalidCapacity。
- `streaming.<pass>.activeGroups`：page、instance、LOD 和逐 cluster 选择 mask。
- `streaming.<pass>.lodState`：所有实例的四 word header（fine count、output prefix、terminal count、ready）位于缓冲开头。随后每实例有 `2 * groupCount` 个 active/mask word、四个 sparse header word（all-active count、visited BVH nodes、tested groups、demand seed root tests），以及最多 `groupCount` 个降序 active group ID。
- `streaming.<pass>.demandStats`：分布式需求启用时提供前 32 个统计 word，包含任务数、节点/组测试数、wave 分配槽位和任务大小直方图；不回读完整需求位图。布局见 [StreamWorkDistribution.md](StreamWorkDistribution.md)。
- `streaming.<pass>.pageTable`：关联请求和可绘制状态。列表容量中的空闲条目不是有效几何。

测试入口：

```powershell
build/tests/MetallicSceneTests.exe --gtest_filter=SceneImport.MeshletStreamAsset:SceneImport.MeshoptCompressedMeshletStreamAsset
build/tests/MetallicRhiTests.exe --gtest_filter="*meshlet_lod*"
build/tests/MetallicRhiTests.exe --gtest_filter="*gpu_driven_mixed_producer_render*"
```

合成参考测试检查原子几何覆盖、共享父组、跨层 terminal、缺页、PendingUpload、容量回退；GPU 差分测试对照选择、请求和间接参数。真实 Bunny 测试对照 CPU frontier 与 GPU group mask，并核对同帧 VBuffer 的每个有效 ID，同时切换透视/正交、标准/Reversed Z、误差、手动 LOD 和硬件/异步混合光栅。

## 换层诊断与预测预取

两个流式入口都支持 `enableLodTransitionTelemetry`（默认 false）和 `predictivePrefetch`（默认 true）。也可在 pass 的运行时设置中切换；切换会重建流式资源，比较时应在采样前设定并重新预热。诊断在既有 frontier/emit 遍历中记录历史，增加每 instance-group 8 字节、每 instance 4 字节 GPU 存储；关闭时不分配这些历史。计数随既有异步反馈读回，不增加 GPU 等待。Profiler 的 `LOD Transition Diagnostics` 和 Full roam 的 `Frames.jsonl` 提供同一口径。

| 计数 | 含义 |
| --- | --- |
| `ownPageBlockedGroups` | 当前需要细化，但本组几何页尚不可绘制；包含 PendingUpload |
| `dependencyBlockedGroups` | 本组页可绘制，但父级链本帧因缺页而受阻 |
| `catchupSelectedGroups/Clusters` | 上一帧已需要且因几何页或依赖受阻，本帧进入输出 cut |
| `thresholdSelectedGroups/Clusters` | 连续自动 LOD 可见帧中，前相机下 refinement 可见且误差未超阈值，本帧跨阈值进入输出 cut |
| `unclassifiedSelectedGroups/Clusters` | 其它新增输出，包括首次出现、重入视野等；不强行归因于正常换层 |

计数单位是 instance-group 和该组输出 mask 的 cluster 数，不是去重页数，也不是变动像素数。Selected 类计数只统计非 terminal 的实际输出，排除容量回退和被更细层完全覆盖的 active 祖先；另保留 `catchupActivatedGroups` 观察尚未进入最终 cut 的激活。这里的输出位于光栅/HZB 裁剪之前，只代表几何，CLAS/BLAS readiness 和纹理升级仍须分别看各自指标。手动 LOD、冻结诊断及历史不连续不推断正常跨阈值。`feedbackFrame` 标明已完成反馈的源帧；离线汇总须按源帧去重，不能将重复观测累加。

预取保留原来的 6.25% 视锥扩展与 0.95 倍误差预算，在 `predictivePrefetch` 开启时用相机位移和完整朝向（含 roll）速度外推请求相机。提前量为近期 demand-to-drawable P95 加一帧，限制在 25–250 ms；样本仅来自最近 5 秒内最后 64 个已完成需求，纯预取不参与，预取晋升按首次实际需求计时。没有有效样本时用 100 ms 加一帧。该小窗口查询不扫描待加载页表。位移上限为场景半径的 2%，转向上限 20°；大位移（超过半径 10%）、大转角（超过 60°）、投影/缩放变化或超过 250 ms 的停顿会重置预测，静止时不残留外推速度。当前 runtime 依靠这些跳变检测，不传递单独的编辑器 camera-cut 事件。

预测相机仅供请求阶段，最终 cut、正常需求、光栅与显示像素误差预算仍使用真实选择相机。需求先提交；预取继续受请求名额四分之一和空闲上传名额约束，并须满足下述真实空闲空间规则，不增大已有内存预算。关闭 `predictivePrefetch` 可恢复原来的静态视锥预取作对照；关闭 `prefetchPages` 则禁用所有预取。Profiler/JSON 同时给出预测是否实际启用、提前量、近期 P95、预取请求/丢弃与累计接纳/使用数。不同阶段的统计不要直接当成同一批页面的命中率。

Full 路线配置可加 `"enableLodTransitionTelemetry": true, "predictivePrefetch": false/true, "routeFrames": 360`，传给 `Tools/RunZorahFullRoam.ps1 -RouteConfig <json>`。两组使用相同显示/内部尺寸、像素阈值、相机轨迹、预热与缓存条件；`routeFrames` 固定采样姿态序列，实际帧耗时与 I/O 竞争仍可能不同。预取请求更多不等同于缺页追赶减少，须观察上述追赶计数及负载失败、溢出、内存压力。

## 页面保留与回收

`adaptivePageRetention` 默认 true，在 `coldPageRetentionFrames != 0` 的 runtime 中生效；Full 已开启冷页维护。两个流式入口的运行时设置提供 `Adaptive Page Retention`，切换会重建资源。设为 false 可对照原有几何 85% 进入压力、70% 退出压力及预取 75% 水位策略。这个对照开关仍共用新的设备 payload 字节口径，并不等同于历史二进制。

自适应策略按上传预算留出有限周转空间：目标为 8 个上传批次与 2 个最大设备页中的较大值，上限是几何容量的 1/8，向下对齐到分配粒度；其中一半为需求预留。Full 当前 8 MiB/帧上传预算对应 64 MiB 回收目标、32 MiB 需求预留。可用空间跌至需求预留时开始回收，达到目标后停止。无压力时保留曾被实际需求使用的冷页作为回访缓存；未使用的预取页仍按原有保留期过期。候选优先较久未使用的页，同龄时先淘汰未使用的预取页。CLAS 仍独立使用 85%/70% 压力水位。

每帧的分配压力回收与主动回收共用最多 1024 页、最多一个回收目标的字节配额；字节配额至少能容纳最大对齐设备页。已排队释放的页可抵扣回收目标，防止延迟释放期间重复过量淘汰；预取和分配只认可分配器已经释放的空间，并逐页复查需求预留及连续块可用性。预取不会主动淘汰需求页。保底页、仍被完整需求反馈标为使用中的页、上传未完成的页，以及刚驻留/刚使用的保护窗口继续受保护；几何和 CLAS 保持现有延迟释放时序。

Profiler 的 Page Prefetch 与 Full 的 `retention` 记录提供策略开关、目标/需求预留、冷页候选字节、待释放字节、每帧淘汰字节与预取页数，以及压力/到期回收计数。`coldResidentBytes` 是当帧候选扫描快照，可能包含随后排队回收的页；`pendingFreeBytes` 尚不可分配。完全被当前需求占满的工作集可能没有任何可回收冷页，此时仍会缺页，不会为凑齐预留而淘汰热页。

2026-09-30 的 Full 对照按旧→新、新→旧两个顺序，各运行 360 个固定姿态，10 秒预热，2560×1440 输出 / 1707×960 内部尺寸，1.5 display px，开启换层诊断。四组二进制、shader、资产和路线一致，未清空 OS 文件缓存。两对结果如下：

| 指标 | 全程变化 | 回程变化 |
| --- | --- | --- |
| 本页缺失阻塞的 group-frame 累计量 | -17.30% / -17.44% | -26.85% / -26.85% |
| 缺页追赶进入 cut 的 cluster 次数 | -4.11% / -4.47% | -2.08% / -2.53% |
| 淘汰页数 | -6.39% / -6.43% | -2.66% / +1.19% |
| 上传字节 | 两对均在 ±0.11% 内 | 两对均在 ±0.06% 内 |

依赖阻塞累计量较小但有所上升（5,795→6,242；5,756→6,906），不能仅用追赶次数下降推断所有依赖都已改善。新策略内存门槛阻塞从每组 360 帧降到 262/271 帧，但队列与内存条件没有同时放行，四组预取请求及使用数仍为 0；本轮收益来自页面保留/回收。CPU 冷页回收范围耗时下降，接纳范围耗时上升，总维护耗时两对方向不同，因此不宣称总 CPU 或整帧加速。按反馈源帧去重 GPU 计数，CPU 工作按采样帧统计；路段边界存在反馈延迟。原始数据位于 `build-release/retention-{legacy,adaptive}{,-repeat}-20260930`，完整汇总为 `build-release/retention-comparison-20260930.json`。

最终 25 项流送/LOD/CLAS 回归通过 Vulkan validation，其中新增预取字节预留、需求缓存/延迟释放、跨回收入口共享字节与页数配额三项测试。Full 短程验证只覆盖启动上传，随后扩展至 360 姿态验证层运行，实际淘汰 24,796 页、上传 48,283 页，无加载失败、请求/BLAS 溢出或验证层错误，记录在 `build-release/retention-validation-full-20260930`。Bunny GPU 输出为完整的有效 ID 图像；另导出 Full 的中心 1280×720 最终颜色区域，FP16 像素均为有限值，静态结构检查未见破洞，原始像素和仅供检查的 SDR 预览保存在该目录的 `image/`。此验证包含调试拷贝，不用于性能比较；静态图与计数不能替代相同视点的连续画面检查，尚未据此确认肉眼可见跳变消失。

## 遍历成本

2026-09-30 验证：14 项预测/近期延迟 CPU 测试、19 项 LOD/流式 RHI 回归通过；连续 GPU fixture 覆盖 4 条遍历路径各 20 帧，包括缺页/父依赖恢复、严格阈值证据、首次/重入、容量回退、手动/冻结恢复及历史 tag 回绕。预测 GPU case 要求出现仅由预测视图触发的带标记请求，同时真实相机的 cut 和普通需求仍匹配 CPU oracle。

Zorah Full 在相同二进制、shader、2560×1440 输出 / 1707×960 内部尺寸、10 秒预热和 360 个固定相机姿态下，静态/预测各运行一次。按反馈源帧去重，正常阈值细化分别为 74,884/74,709 次 instance-group，缺页追赶为 47,157/47,058 次；首次或其它变化另行计数。两组实际预取请求均为 0：几何驻留约 3.30–3.76 GB，预算 3.76 GB，始终超过原有 75% 预取水位。新增 `memoryWatermarkBlocked` / `queueBlocked` 明确这一限制。不能把两组计数或帧耗时差异解释为预测收益；此场景下一步应评估保留/回收策略及可用预取空间。原始证据在 `build-release/lod-prefetch-{static,predict}-20260930`，汇总为 `build-release/lod-prefetch-comparison-20260930.json`。这是几何诊断验证，未据此宣称 Full 漫游跳变已从视觉上消除。

frontier 临时状态为每实例 32 字节，加该实例每组 12 字节；只扫描访问的 BVH 节点及叶子组，mask/emit 成本随 active 列表大小变化。由精细切换到粗糙 LOD 的首帧仍需清除前一帧的精细列表，后续帧不再清理所有组。BVH 在 primitive 实例之间共享，每节点 40 字节。

有序 BVH 保持拓扑及输出顺序，但空间聚合可能比重新排序的空间 BVH 更松；近景或高误差要求下可能访问全部节点。当前 safe frontier 使用每实例一个 64 线程组，需求阶段独立跨实例调度；父子可绘制状态尚未改为全局持久化队列。测试中的 group 检查数减少不等同于整帧耗时收益。

当前可达性方案保留已激活的祖先页。很小的预算可以维持完整粗 cut，但可能无法达到指定像素误差；进一步释放完全被替代的祖先 payload、合并请求和优先级调度仍是后续优化。

2026-09-12 的 RTX 5070 Ti 验证中，80 组 CPU 覆盖检查、37 组 GPU 差分与 16 组收敛后的真实 Bunny 对照通过。透视 0.05 px 达到 550 个 cluster，16 px 为 18 个；手动最粗为 1 个。硬件与异步光栅的有效 ID 完全一致，无需使用深度 tie 容差；透视和正交冻结选择相机后移动渲染相机，cut 保持不变且画面正确变化。结果保存在 `StreamMeshletLodSceneReport.json` 与对应 PNG 中。

独立 StreamAsset pass 在可见性、变换和尺寸更新时复用 Runtime 与驻留缓存；场景内容或资源配置变化才重新初始化。最终主程序、RHI 和 SceneTests 构建通过，CLAS/隐藏恢复/两次尺寸变化烟测，以及混合 resident/stream 生产者回归通过（开启 Vulkan validation）。

BVH 版本验证：主程序、RHI 和 SceneTests 构建通过，9 个相关回归测试全部通过，包括 146 组线性/BVH GPU 差分、16 组真实 Bunny 对照、冻结相机、CLAS 和混合生产者。CPU 额外覆盖分散包围球、剪切、镜像、阈值附近比较，以及 `1e8` 坐标消减导致近裁面漏选的回归；GPU 初始化检查跨越 65535 工作组的第二行末尾哨兵。

同一 Bunny 资产共有 50 个 group、19 个 BVH 节点。透视 16 px 阈值时访问 9 个节点、检查 7 个 group，输出仍为 18 个 cluster；正交 16 px 和手动最粗层级检查 4 个 group。0.05 px 近景仍检查全部 50 个 group。硬件与异步光栅选择及可见 ID 一致。对应结果在 `.tmp/bvh-lod/StreamMeshletLodSceneReport.json`，回归日志在 `.tmp/bvh-lod/regression2.log`；这些是选择工作量数据，没有据此推断整帧加速比例。
