# Virtual Shadow Maps P2：持久页缓存与 Sun/Moon

对应 [实施方案](VirtualShadowMapsResearch.md) 的 P2，沿用 [P1](VirtualShadowMapsP1.md) 的完整场景 caster、硬阴影、独立粗级和无 RayQuery 路径。P2 增加按灯持久页缓存，以及 Sun/Moon 各自的 Pass 与 Deferred 输入；没有切换实时默认阴影系统。

本文记录当前实现合同及 2026-10-25 本机验证结果。专用缓存测试和 OpenPBR sample 对照已通过；未定位的 Deferred 法线对照失败与未覆盖范围在“验证”和“当前限制”中单独记录。

## 缓存与投影

每个 `VirtualShadowMaps` 实例拥有独立缓存：D32 图集、上一份 GPU 页表、投影、整数页原点、caster 内容快照和提交状态。physical page 0 始终属于完整场景粗级。每次录制仍创建新的参数、工作列表、间接参数、统计和输出图；只有兼容缓存的图集被复用。

`cacheEnabled=false` 保留 P1 每帧重画的行为，也是 Pass 默认值。`forceRedraw=true` 在缓存投影和页分配机制中将可用页标脏，供同投影硬阴影对照使用；这与关闭缓存后重新计算 P1 紧包围投影不同。

缓存开启时，初始 Z 范围在既有 bias margin 外，两端各增加 `0.25 * max(sceneDepthSpan, 1m)`。粗级半幅在 P1 范围上增加 25% 比例余量，并加入 margin。只要当前完整场景 bounds 仍位于上一粗级投影的安全内部，所有层继续使用上一帧完整 `rowZ` 和深度 bias，粗级 XY 与 extent 也保持不变。场景缩小不会收缩这些投影。

安全内部要求粗级 XY 保留 `2/pageSize` 的 NDC 边界，Z 保留对应世界 bias 的深度边界。超出保留范围时建立新投影并重置缓存，避免在旧深度值上使用新的 Z 比例或偏移。

细 clipmap 的半幅逐层倍增，中心按光源 XY 空间中的整页网格移动。CPU 保存各层 `int64` 页原点，由原点差将当前虚拟页映射到上一虚拟页；重叠区域保留 physical page，移入区域重新申请。重映射不移动图集 texel。未被当前 receiver 请求的缓存页可以保留，但不能挤占已按细到粗顺序准入的请求页；池满时确定性回收未准入页。当前没有 LRU、静动分层或页内容压缩。

## 内容变化与失效

caster 生产者发布按 ID 排序的 `{id, bounds, contentKey}` 快照，以及 `sceneIdentity`、`structuralRevision`、`dependencyKey`。ID 是结构代次内稳定的 render-node 索引。隐藏 caster 不进入快照，因此上一份已提交快照仍能提供其旧范围。

- resident：哈希实际发出的世界空间三角形位置、UV、当前变换及相关 Coverage 内容；不依赖主相机可见列表或 previous-world 历史。
- StreamAsset：使用生产者同步后的运行时节点映射、当前变换/显隐/材质，以及完整 locked terminal cut。资产重新初始化发布新的几何代次；普通 refined-page 流送不改变该代次，因为 terminal 页不能被卸载，其内容在该 runtime 生命周期内保持不变。
- 材质：跟踪单双面、MASK 状态、cutoff、Coverage IR 和实际引用的参数槽；使用 base alpha 时也跟踪 alpha factor、纹理逻辑绑定和 UV transform。Surface RGB、粗糙度、发光颜色等不进入该内容键。
- 纹理：物理图像初始发布、重载和完成的 mip 迁移产生全局唯一发布代次。Pass 存在实际绑定的 MASK alpha 纹理时，将它组合到依赖键。当前按整灯保守失效，因此无关纹理迁移也可能触发重画；不会仅用 descriptor 地址判断内容是否未变。VSM 当前直接采样发布图像，不消费纹理 sampling-floor 渐变元数据。

在场景和全局依赖兼容时，CPU 比较已提交快照与当前快照：新增使用新 bounds，移除/隐藏使用旧 bounds，内容变化同时使用旧、新 bounds。每个 AABB 的八角点投影到当前细级 XY，保守标记相交页；方向光的该步骤不按 Z 剔除阴影列。任意 caster 变化同时使粗级变脏。

Device、光源方向、有效启用状态、页尺寸/池容量/clipmap 设置、bias、场景身份、结构代次或全局依赖键变化会重置该灯缓存。越过上述投影范围也会重置。只改变灯颜色或强度且不改变有效启用状态时，几何深度可继续复用。Sun 与 Moon 分别拥有上述状态，一个灯的方向或预算变化不主动重置另一灯。

## GPU 清页、光栅与查询

GPU 页表使用 `uint4`：physical index、请求/准入位、clipmap level、valid/dirty 位。每帧按上一页表和 CPU remap 恢复缓存，使用真实 receiver depth 标记需求，再由 GPU 分配页。triangle/page 工作只写入本帧需要绘制的脏细页。`vsmBuildMain` 在 `statistics[16] == 0` 时于几何解码前返回，因此全热缓存仍录制 compute dispatch，但不解码三角形或生成页工作。

新图集整体清为 reversed-Z 空深度 0。复用图集使用 `Load`，新增 `vsmClearMeshMain` 对每个物理页检查 owner 与 dirty/admitted 状态；需要清除的 tile 输出两个三角形，以 Always 深度写入 0，其他页输出零图元。清页管线同时提供空的 `vsmClearFragmentMain`，满足当前 Vulkan RHI graphics pipeline 必须具有 fragment module 的合同。随后批量光栅 coarse 与细页间接工作。CPU 不逐页发 draw，已有页内四平面几何裁剪继续隔离相邻 tile。

coarse 绘制独立遍历完整 caster 容量，不消耗细页工作预算。细页工作溢出时，该帧 receiver 全部查询粗级，部分绘制的细页不会提交为有效缓存；后续帧可继续补画。页池不足时查询可用祖先或 coarse。投影外 receiver 仍输出洋红 debug 并计数。

## 提交与取消

缓存开启要求录制中的 `RenderFrameContext`。录制结束只创建 pending 状态，不立即替换上一份已提交缓存。下一次录制检查 `SubmissionTransaction` 与 completion：已提交的 pending 才成为当前缓存；已取消的 pending 被丢弃，保留上一份已提交状态；尚未提交也未取消时明确报错。

复用时将上一提交 completion 加入新命令的依赖，State 持有前后缓存及资源所有权。提交不等于 GPU 已完成，跨提交执行顺序由该依赖保证。取消录制不会发布其 caster 快照、页表或投影。当前 `VirtualShadowMapPass::supportsFrameOverlap()` 仍为 false，不将此机制描述为任意帧重叠支持。

## 双天体接口与示例

`lightIndex=0` 选择 Sun，`lightIndex=1` 选择 Moon。每个 Pass 的输出为 `shadow`（R8）、`parameters`（320 B 单灯 tagged identity）、`debug`（RGBA8）和 `statistics`（128 B，32×uint32）。Deferred 保留 `shadow`/`shadowParameters`，另接收 `moonShadow`/`moonShadowParameters`；各输入只作用于匹配的天体直接光，不相乘成为所有灯共用的遮蔽。

可选 [virtual-shadow-map-p2 图](../Pipelines/Samples/virtual_shadow_map_p2.metallic_graph.json) 使用 [OpenPbrVSMCache 场景](../Asset/LookDev/OpenPbrDefault/OpenPbrVSMCache.metallic_scene.json)，同时配置两个不同方向/颜色的天体。两个 VSM Pass 各开启缓存，各有 256 个 128×128 D32 页（每灯裸图集 16 MiB）和 1,048,576 个细页工作槽（每灯工作 buffer 8 MiB）。这些是资源尺寸，不是测得的进程显存峰值。

sample 固定主视图几何 `autoLod=false, lodLevel=0`，关闭 temporal jitter，StreamAsset 使用 `initialLoad=true`。固定 LOD 0 继承 P1 用于避免 receiver 简化几何与完整 resident caster 不一致的限制；StreamAsset caster 仍为 terminal cut，没有新增 shadow footprint 驱动的 refinement。

## 统计 ABI

每个天体分别输出以下 128 B；不是把 Sun/Moon 合并到一个统计 buffer。索引 0–15 保留 P1 含义，16–31 为扩展。

| 索引 | 含义 |
| --- | --- |
| 0 | 请求的细页数量 |
| 1 | 本帧准入的细页数量 |
| 2 | 未准入细页数量，即页池溢出 |
| 3 | 请求的脏细页 triangle/page 工作数 |
| 4 | 实际存储的细页工作数 |
| 5 | 超出工作容量的细页工作数 |
| 6 | 非背景 receiver 数 |
| 7 | 使用 coarse 的 receiver 数 |
| 8 | 背景像素数 |
| 9 | 位于所有投影外的 receiver 数 |
| 10 | triangle slot capacity；stream 为 cluster count×128，含尾 slot |
| 11 | 细 clipmap 层数 |
| 12 | 物理页容量，包含 coarse 页 |
| 13 | 使用祖先细级的 receiver 数 |
| 14 | 灯/Pass 无效启用状态，禁用为 1 |
| 15 | 保留，0 |
| 16 | 本帧准入且脏的细页数 |
| 17 | 本帧准入且干净有效的细页数 |
| 18 | 复用映射中被 CPU dirty 标记的已有页数，可能包含 coarse 与未请求页 |
| 19 | 已有物理映射对应的虚拟页索引发生变化的页数 |
| 20 | 本帧新分配物理槽的细页数，包含回收未准入页后的分配 |
| 21 | 本帧是否重画 coarse，0/1 |
| 22 | 本帧是否未复用缓存，0/1；关闭缓存时每帧为 1 |
| 23 | 缓存 epoch；兼容复用保持，重置时递增 |
| 24 | finalize 后仍有物理映射且有效的细页数，包含未请求缓存页 |
| 25 | CPU dirty bounds 条目数；单个变化 caster 可贡献旧、新两项 |
| 26 | `forceRedraw` 开关，0/1 |
| 27 | `cacheEnabled` 开关，0/1 |
| 28 | 本次录制观察并丢弃了上一 pending 取消状态，0/1 |
| 29 | 保留，0 |
| 30 | 保留，0 |
| 31 | 保留，0 |

## 验证

2026-10-25，`Metallic`、`MetallicRHITests` 与 `MetallicShaderRequestsTests` 构建完成；shader module ownership audit 检查 384 个文件通过。主要复现入口：

```powershell
cmd.exe /d /s /c '"E:\VisualStudio\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=14.51.36231 -winsdk=10.0.26100.0 && cmake --build build-pass-stages-nrd --target Metallic MetallicRHITests -j 8'
python -B Tools/CheckShaderModules.py
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_cache --rhi-validation --output-dir .cache/vsm-p2
```

`virtual_shadow_map_cache` 通过。resident 与 metadata-only StreamAsset 各 26 帧，共 52 帧；每帧分别比较 Sun/Moon 的 257×193（49,601）个像素，缓存结果与独立 force-redraw 结果全部逐像素相同。覆盖静止复用、灯 radiometry 与材质 RGB 修改、整页相机平移、caster 移动/显隐/Coverage 编辑、单灯方向/预算变化、页池溢出和需求变化后的回收。49,601 是每幅比较图的像素数，不保证每个测试场景均无背景。

实际统计示例（resident；对应 StreamAsset 行得到相同计数）：

| 场景 | Sun 脏/干净细页 | Moon 脏/干净细页 | Sun/Moon 页工作数 | Sun/Moon coarse 重画 |
| --- | --- | --- | --- | --- |
| Initial | 139 / 0 | 149 / 0 | 326 / 328 | 1 / 1 |
| Stable0 | 0 / 139 | 0 / 149 | 0 / 0 | 0 / 0 |
| CameraPageShift | 24 / 102 | 28 / 120 | 48 / 55 | 0 / 0 |
| CameraPageShiftAgain | 0 / 126 | 0 / 148 | 0 / 0 | 0 / 0 |
| CasterMoved | 25 / 101 | 24 / 124 | 80 / 76 | 1 / 1 |

完整取消录制、接受前缀后取消尾部录制两项恢复检查均通过，场景替换也与重画参考一致。flat 与 binned Deferred 消费检查分别确认 Sun 改变 4,345 个像素、Moon 改变 2,043 个像素。专用 Preview 设备开启同步 validation，捕获到的消息为 0；RayQuery 与 acceleration structure 功能关闭。测试关闭 jitter，不能据此宣称长程 temporal 验收。

10 个实际运行入口的 SPIR-V 均通过 `spirv-val --target-env vulkan1.4 --scalar-block-layout`；7 类 CPU/Slang ABI 布局与当前源码一致。实际依赖文件字节校验通过，无 RayQuery/RT capability，两个 fragment 入口均无 `EarlyFragmentTests`。

实际 OpenPBR P2 sample 另在 512×512 下比较缓存与强制重画：帧 0、1、5 的 HDR，以及全部导出捕获均完全一致；HDR 全部有限，日志检查无异常。两灯各有 60,933 个细级像素，coarse 与 outside-all 均为 0。该 MaterialBaseline harness 开启设备 RayQuery，且没有读取 cache counter；无 RT 结论来自上面的专用 P2 测试，热缓存工作量结论来自专用测试的统计。

回归与未通过项：

- 同轮 RHI 回归 9 项中，P0、P1、`realtime_ray_traced_sigma_shadows`、`gpu_scene_submission_recovery`、`render_graph_sample_load`、`gpu_scene_coverage_classification`、`stream_scene_cache_routing` 共 7 项通过。
- `trusted_scene_resource_contract_and_bounds` 初次失败是测试将固定 100 个 uint 的输入视为 oversized，而扩展后的 ABI 恰为 400 B。将 fixture 改为按当前 ABI 构造多一个元素的输入后，独立重跑通过。
- `visibility_buffer_deferred_openpbr` 的 shading-normal raster/ray 对照仍失败：4,153 个像素中 3,812 个 outliers，mean error 40.582230。原因尚未定位，也没有 HEAD 复现证据，不能归为既有失败或归因于 LOD。P2 硬阴影对照通过不等于该回归通过。
- CPU shader suite 首次 131 项为 124 通过、1 跳过、6 失败；其中 5 项日志报告测试期间源码或目录 epoch 变化，停止并行修改后串行重跑全部通过。跨两轮合并为 129 通过、1 跳过、1 未通过，而非一次完整运行全绿。
- 剩余 `LocalMaterialProgramsWarmBeforeRuntimeCreatesTheInclude` 期望 3 个请求而实际 2 个。静态比较 HEAD 确认 Deferred `globalView` 请求早已合并，测试仍保留旧计数；该断言未修改、未计为通过。跳过项是 unavailable optimizer 路径。

本地证据均位于忽略的 `.cache/`，不纳入源码：

- [52 帧对照、缓存计数、取消及双灯消费](../.cache/vsm-p2/VirtualShadowMapCache/VirtualShadowMapCache.json)
- [同步 validation](../.cache/vsm-p2/VirtualShadowMapCache/PreviewValidation.json)
- [实际 SPIR-V 与 ABI](../.cache/vsm-p2/spirv/Validation.json)
- [OpenPBR 比较结果](../.cache/vsm-p2-openpbr/Comparison.json)与[缓存/强制重画对照图](../.cache/vsm-p2-openpbr/Cached-Force.png)
- [RHI 回归](../.cache/vsm-p2-regression.log)与[trusted ABI 修正后重跑](../.cache/vsm-p2-trusted.log)
- [CPU shader suite 首轮](../.cache/vsm-p2-shader-tests.log)与[5 项串行重跑](../.cache/vsm-p2-shader-tests-retry.log)
- [module ownership audit](../.cache/vsm-p2-module-audit.log)

## 当前限制

CPU 仍逐帧扫描完整 resident 几何、生成并上传世界空间三角形与内容指纹；stream 路径仍逐帧枚举 terminal candidates。每个灯独立执行 caster preparation。CPU 还逐页生成 remap/dirty 输入。GPU build dispatch 仍按完整 triangle slot 容量录制；无脏细页时在解码前返回，有脏细页时仍遍历所有 slot 以生成工作。因此，热缓存节省解码/页工作与深度绘制，并未消除 CPU caster preparation、GPU demand/allocation 或 receiver 查询成本。

没有性能提升、CPU/GPU 帧预算或长期显存稳定性的结论。尚未宣称完成 jitter/长程相机及物体运动、真实纹理流送编辑矩阵、独立 shadow LOD 质量、SMRT、局部光、位移/细分、BLEND/透射/native strands 或默认切换验收。完整质量和默认实现门槛仍属于 P4/P5。

## 源码入口

- [缓存、投影与录制](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp)
- [caster 快照与内容键](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.cpp)
- [页管理、tile clear、光栅与查询](../Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang)
- [Pass 与统计输出](../Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp)
- [缓存及场景验收用例](../tests/rhi/VirtualShadowMapSceneTests.cpp)
