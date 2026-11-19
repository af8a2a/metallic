# VSM 动态失效统计与静态/动态深度分层

两种 VSM Pass 提供独立的 `cacheDiagnostics` 输出，单面为 32 个 uint；多灯按灯表条目分配六个面块，未准入面清零。字段与原因位定义见 [VirtualShadowMapCacheDiagnostics.h](../Source/Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h)。原 `statistics` 的 32 项含义保持不变。

| 字段 | 内容 |
|---|---|
| 0 | 失效原因位。场景、几何 publication、投影等可使整面失效；纹理 publication 与细分也可仅影响局部 caster。旧结构原因位保留。 |
| 1–4 | 分层是否生效、缓存是否复用、静态/动态 caster 数 |
| 5–10 | 新增、移除、内容变化、变换变化、阴影材质变化、首次动态晋升数 |
| 11–15 | 合成/静态 dirty bounds 与 rectangles、取消录制数 |
| 16–19 | 静态/合成层实际 admitted dirty fine/coarse 页 |
| 20–23 | 两层实际 raster 解码三角实例与输出 primitive 数；需字段 28 为 1 |
| 24–27 | 从静态恢复页、静态复用 fine 页、静态/合成失效 resident 页 |
| 28 | 分层 raster 工作量计数是否有效 |
| 29 | 阴影 LOD cut 变化的 caster 数；30–31 保留 |

`cacheLayering=true` 开启静态基础深度与动态合成深度分层；需要同时启用 `cacheEnabled`。新开关默认关闭，现有 one-pass projection 默认开启保持不变。

## 分类与提交

场景目前没有 authored mobility 字段。分类来自已提交的 caster 内容快照。初始 caster 进入静态层，内容发生变化后迁移到动态层。首次迁移使静态层旧 footprint 失效，以清除旧阴影。后续移动只使合成层的旧、新 footprint 失效。全局缓存重置时重新分类；启用 `framesStaticThreshold` 时，连续稳定达到阈值的 caster 可以合并回静态层。

阴影专用 LOD 的 `lodKey` 独立于 authored `contentKey`。仅 cut 变化会使对应静态/合成深度失效，不触发静态向动态晋升；细节见 [阴影专用 LOD](VirtualShadowMapShadowLOD.md)。

pending 录制只有在提交后才能发布分类和缓存。caster 的 render node 索引在同一 scene identity 内稳定，SceneGraph 结构版本变化后仍可对照已提交的快照，识别新增、移除及变更的 caster。场景重载、重新组合或 stream 发布新几何时仍重建深度缓存。

## 场景变化的失效范围

Single 和 Lights Pass 按材质记录实际参与 MASK 覆盖和位移采样的纹理版本。纹理重发布或细分采样密度变化时，使用该资源的 caster 以旧、新范围更新深度；静态 caster 保持静态。改变生效的 authored 遮挡材质或变换仍按原规则晋升动态。直接调用者仅提供全局纹理版本时，继续使用全局失效。诊断中的纹理和细分原因位也可伴随 `reused=1` 出现，此时页计数表示局部更新。

spot 和 point 的 coarse 页与本面 fine mip 具有相同视锥。合成层、静态层和静态合并各自复用本面 level 0 的投影结果，只有与对应旧、新范围相交的灯面需要更新。穿过近面的包围盒若完全位于同一侧平面之外，也可排除。仍可能进入视锥的跨近面包围盒保守覆盖整个灯面。方向光 coarse 继续覆盖完整场景，投影范围增长超出缓存时仍重建。逐 view GPU LOD 变化也按对应局部灯面的范围处理。候选槽位布局变化使旧 GPU LOD 掩码无法直接比较时，当前 caster 的覆盖范围会保守失效。

对照本机 UE 5.7.4，`VirtualShadowMapCacheManager.cpp` 在场景更新前后收集旧实例和新实例，`VirtualShadowMapCacheInvalidation.ush` 按 VSM 视锥和已分配页筛选。Metallic 使用已提交的 caster 快照与当前快照的世界 AABB，检查驻留页，包括本帧未请求的页。RenderGraph 的场景绑定刷新仍可重置时域历史；stream runtime 重新发布几何也有独立的缓存代次。

2026-11-18，Release，RTX 5060。扩展的 `virtual_shadow_map_cache_layers` 通过 102 个状态，比较分层缓存、普通缓存、强制重绘的可见性和可查询 D32，Vulkan validation 消息为 0。点光 caster 从 −Z 进入 +X 后，两面的 coarse 各重绘 1 页，其余四面为 0；聚光灯锥外的显隐均复用 68 个静态 fine 页。resident / stream 的实际采样高度或 alpha 纹理换版均重绘 8 个静态 fine 页，复用 14 页，未采样纹理换版的脏页为 0。另以 `ktx2_texture_feedback_submission_contract` 验证真实 mip 迁移只更新被替换槽位的版本。原始结果见 [缓存回归报告](../build/vsm-invalidation/Run1/VirtualShadowMapCacheLayers/VirtualShadowMapCacheLayers.json)。

实际 Single/Lights RenderGraph、逐 view GPU LOD 的 22 个状态、共享物理池的 60 个状态及 remap 独立参考也通过，以上 VSM 报告的 Vulkan validation 消息均为 0。共享池首次运行因 Slang 源码目录变更通知而中断，单独复跑完整通过。[共享池复跑结果](../build/vsm-invalidation/PoolRetry/VirtualShadowMapPhysicalPool/Report.json)。

共享物理页池在提交时为当帧所有灯面保留同一份不可变 caster 快照。新录制只在首个可复用灯面确实遇到新 caster 发布时比较一次新旧快照。其余可复用灯面读取同一变化列表和公共脏边界。各面按自己的动态 caster 历史应用静态失效与降级，并继续独立投影边界和处理 GPU Shadow LOD 的变化。

CPU profile 的 `Prepare VSM projections and dirty bounds` 中，`comparedCasterStates` 只由实际比较快照的灯面报告，`reusedCasterChanges` 标记复用差分的灯面。同一 caster 发布复用原有快照。取消的录制不更新已提交快照，暂时缺席的灯面回归时重新验证其物理页历史。

## 深度与工作量

两层共用虚拟页到物理页映射、页需求、分配和原 build 工作表。静态层只更新静态脏页；合成层脏页先从静态 atlas 恢复深度，再绘制动态 caster。coarse 页同样分层。最终 projection、PCF、SMRT 读取合成后的深度，过滤算法和点光跨面采样保持不变。

该实现减少重复的静态几何解码和光栅工作，不减少完整 build 扫描。新增静态 atlas 及其恢复操作有显存和执行成本，不能仅凭重绘计数宣称 GPU 加速。多灯 `memory[0]` 包含两层 atlas 总量，`memory[7]` 为其中静态 atlas 的字节数，不应重复相加。

失效统计区分全局原因位、caster 新增/移除/变化/晋升、旧新 bounds 与实际页计数；同一 caster 可同时发生变换与材质变化，分类原因并非互斥。resident invalidation、admitted dirty fine pages、coarse redraw 属于不同统计范围，不能直接相加当成唯一失效页数。

局部 caster 差分只在旧缓存可复用时执行；`reused=0` 的整缓存重建由原因位解释，局部差分字段的零值不表示场景未变化。这里的“动态”是当前缓存代次观察到的变化分类，不是资产永久 mobility 标签。

页计数始终采集；字段 20–23 的分层三角/primitive 计数只有在分层与 `benchmarkGPUWork` 同时开启时有效，由字段 28 `rasterWorkAvailable` 明确标识。未采集的零值不代表实际零工作。

## 验证

2026-10-31，Release `Metallic` 和 `MetallicRHITests` 构建通过，Slang 模块审计通过（387 文件）。生成证据保存在 `build/vsm-cache-layers/`，不纳入源码。

- `virtual_shadow_map_cache_layers`：76 状态 × 三臂通过，无跳过，validation 为 0。resident/stream 方向光 Hard/PCF/SMRT，以及 resident Spot PCF/Point SMRT；覆盖取消迁移、首次/连续移动、显隐、MASK 材质、纹理 publication、位移、clipmap scroll、真实 work overflow 及恢复。分层对原 cached 路径的 visibility、stats32（有意取消计数除外）和有效页 D32 精确相等；对 forceRedraw 比较 visibility、coarse 和共同有效 fine 页。
- `virtual_shadow_map_cache_layer_projection_equivalence`：86 状态 × 五臂通过，无跳过，validation 为 0。新增 `cacheDiagnostics` 实际经 RenderGraph 输出回读；shadow/debug/HDR/depth、stats32、build 及非 raster 工作量精确相等。78 个多面状态实际出现一次 one-pass scope，其他状态验证零/单面回退。分层 atlas 总量实测为单层的两倍，静态 subset 与公开内存输出一致。
- 原 `virtual_shadow_map_content`（50 个 frame 记录及资源池/生命周期检查）、`virtual_shadow_map_profiling`（20 状态 × 四臂）通过，无跳过，validation 为 0。四项测试使用同一最终二进制；11 个已修改源码文件在最终运行前后哈希一致。

八种配置的连续动态移动均观察到静态 dirty 页与静态 decoded triangles 为 0，合成层仍更新 9–36 页。这里的页数按光源面汇总；三角计数是按页重复的解码实例，不是唯一模型三角数。图矩阵只对分层臂采集 GPU scopes，其他四臂仍完整执行正确性对照，不采集辅助 CPU attribution。

本轮没有普通 GPU 性能 A/B，不宣称帧耗时收益。静态 atlas 加倍、每录制两份筛选 mask/参数上传、额外 mesh task 启动及恢复操作均有成本；后续性能验证应使用受控动态场景，单独测量。

首轮保留的两项失败属于测试契约/采集容量：`Run1` 对 forceRedraw 的 overflow 页错误要求与 cached 页有效位一致；修正后仍严格比较分层与原 cached 路径，仅对 overflow forced fine 页选择共同有效部分。`Run2` 图矩阵的辅助臂 timestamp query 溢出，改为仅 profile 分层臂，五臂正确性比较不变。失败包未覆盖。

证据与复现：

- [汇总报告](../build/vsm-cache-layers/Summary/report.md)、[最终身份与验证](../build/vsm-cache-layers/FinalVerification.json)
- [底层最终结果](../build/vsm-cache-layers/Run3/virtual_shadow_map_cache_layers/VirtualShadowMapCacheLayers/VirtualShadowMapCacheLayers.json)
- [多灯图最终结果](../build/vsm-cache-layers/Run3/virtual_shadow_map_cache_layer_projection_equivalence/VirtualShadowMapCacheLayerProjectionEquivalence/VirtualShadowMapCacheLayerProjectionEquivalence.json)
- [工作量图](../build/vsm-cache-layers/Summary/Workload.png)、[绘图脚本](../build/vsm-cache-layers/Summary/plot_report.py)、[原始字段与哈希](../build/vsm-cache-layers/Summary/PlotMetadata.json)
