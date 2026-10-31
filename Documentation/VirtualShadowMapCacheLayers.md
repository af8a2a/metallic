# VSM 动态失效统计与静态/动态深度分层

两种 VSM Pass 提供独立的 `cacheDiagnostics` 输出，单面为 32 个 uint；多灯按灯表条目分配六个面块，未准入面清零。字段与原因位定义见 [VirtualShadowMapCacheDiagnostics.h](../Source/Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h)。原 `statistics` 的 32 项含义保持不变。

| 字段 | 内容 |
|---|---|
| 0 | 全局原因位：禁用缓存、冷启动、设备、光源/启用状态、设置、场景、结构、几何 publication、纹理 publication、位移依赖、投影范围、强制重画、分层模式变化 |
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

场景目前没有 authored mobility 字段。分类来自已提交的 caster 内容快照：初始 caster 进入静态层，内容发生变化后单向迁移到动态层。首次迁移必须使静态层旧 footprint 失效，以清除旧阴影；后续移动只使合成层的旧、新 footprint 失效。全局缓存重置时重新分类，不按若干静止帧自动回迁。

阴影专用 LOD 的 `lodKey` 独立于 authored `contentKey`。仅 cut 变化会使对应静态/合成深度失效，不触发静态向动态晋升；细节见 [阴影专用 LOD](VirtualShadowMapShadowLOD.md)。

pending 录制只有在提交后才能发布分类和缓存；取消不能使未执行的迁移成为后续帧依据。caster 身份仅在同一结构代次内有效。变换键与阴影材质键由 producer 分别生成，不能从旧混合 `contentKey` 推断原因；材质 RGB 等与阴影无关的变化仍沿用现有阴影内容契约。

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
