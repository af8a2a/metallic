---
id: "0006"
title: "VSM: 静态/动态深度分层默认关闭，缺普通 GPU A/B，静态 atlas 翻倍的成本未量化"
area: shadow
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:shadow", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapCacheLayers.md:17
  - Documentation/VirtualShadowMapCacheLayers.md:29-31
  - Documentation/VirtualShadowMapCacheLayers.md:43-49
  - Documentation/VirtualShadowMapCacheLayers.md:51
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:89-90
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp:30-31
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:198
---

## 背景 / Context

`cacheLayering` 把静态基础深度与动态合成深度分成两套 atlas：静态层只更新静态脏页，合成层
脏页先从静态 atlas 恢复深度再绘制动态 caster，coarse 页同样分层。工程动机是减少重复的
静态几何解码与光栅工作，`cacheDiagnostics` 提供 32 个 uint/面的原因位与页计数。

## 现状 / Current state

- 正确性已经做得比较完整：`virtual_shadow_map_cache_layers` 76 状态 × 三臂、
  `virtual_shadow_map_cache_layer_projection_equivalence` 86 状态 × 五臂、
  `virtual_shadow_map_content` 50 帧、`virtual_shadow_map_profiling` 20 状态 × 四臂
  全部通过，validation 为 0（`VirtualShadowMapCacheLayers.md:43-45`）。
- 但性能证据是空的：文档明确写「本轮没有普通 GPU 性能 A/B，不宣称帧耗时收益」，并列出
  已知成本（静态 atlas 加倍、每录制两份筛选 mask/参数上传、额外 mesh task 启动与恢复操作），
  要求「后续性能验证应使用受控动态场景，单独测量」（`VirtualShadowMapCacheLayers.md:49`）。
- 代价是确定存在的：分层 atlas 总量实测为单层的两倍（`:44`），`memory[0]` 含两层、
  `memory[7]` 是其中静态 subset（`:31`）。
- 该候选默认关闭，且依赖另一个默认关闭的设置：`cacheLayering` 需要同时开启
  `cacheEnabled`（`VirtualShadowMapCacheLayers.md:17`）；
  `VirtualShadowMapSettings::cacheLayering = false`、`cacheEnabled = false`
  （`VirtualShadowMaps.h:89-90`），两个 Pass 各自解析属性
  （`VirtualShadowMapPass.cpp:30-31`、`VirtualShadowMapLightsPass.cpp:198`）。
- 采集能力仍是薄弱点：首轮 `Run2` 图矩阵的辅助臂 timestamp query 溢出，只能改为
  「仅 profile 分层臂」，五臂正确性比较不变（`VirtualShadowMapCacheLayers.md:51`）。

## 问题或缺口 / Problem

1. 这是一个「静态几何不重复解码」的优化，但唯一能量化它的场景是持续移动的动态 caster；
   当前证据只有 8 种配置的页数观察（静态 dirty 页为 0、合成层更新 9–36 页，`:47`），
   没有时间。
2. 显存成本与收益完全没配对：atlas 翻倍是确定的，收益（重绘时少解码多少三角、少多少
   光栅组）只是计数，没有换算成任何时间或带宽指标。
3. 「缓存重绘」与「暖缓存」两种 workload 的结果会完全不同，文档没有预声明这两种状态
   分别的准入条件，因此任何后续 A/B 都可以被解释成有利结论。
4. 采集侧限制（timestamp query 溢出）尚未在生产样例规模下解决，直接做多灯 A/B 会再次
   落到「只 profile 一个臂」的降级采集。

## 期望结果 / Expected outcome

用受控动态场景给出分层的 pass 级耗时与显存代价配对数据，并据此决定采纳或归档；
采纳时必须同时说明「静态 atlas 翻倍」在显存预算里如何摊销。

## 验收标准 / Acceptance criteria

- [ ] 固定一个持续移动 caster 的受控场景（可用现有 P4/OpenPBR 动态用例），分别报告
      cold（首帧全建）与 warm（持续移动）两种状态的 Shadows / RenderGraph 区间。
- [ ] 预声明预算：≥3 个独立 ABBA 区组、每进程 ≥96 帧与 32 帧预热，`Budget.json`
      在首个 GPU 进程前写入；未达条件不进 confirmation、不追加样本。
- [ ] 显存配对数据：分层前后稳态与峰值的 atlas bytes（`memory[0]`/`memory[7]`）以及
      `Device::memoryBudget()` 的 live/peak，明确峰值是否包含在途旧代 atlas。
- [ ] 采集侧：多灯图矩阵在 `benchmarkGPUProfile` 下 `profilingOverflow` 必须为 0，
      不再依赖「仅 profile 分层臂」的临时降级（对应
      `VirtualShadowMapCacheLayers.md:51`）。
- [ ] 判定二选一：采纳则更新 `VirtualShadowMaps.h:90` 默认值、
      `VirtualShadowMapCacheLayers.md:17` 与 `VirtualShadowMapProductionOptimizations.md`
      固化表；归档则在文档顶部写明「正确性完成、性能未验证，保留为显式实验」。
- [ ] 无论结果如何，`virtual_shadow_map_cache_layers` 76 状态与
      `virtual_shadow_map_cache_layer_projection_equivalence` 86 状态必须继续通过、VVL=0。

## 相关文档 / References

- [VirtualShadowMapCacheLayers.md](../../VirtualShadowMapCacheLayers.md)
- [VirtualShadowMapShadowLOD.md](../../VirtualShadowMapShadowLOD.md)
- [VirtualShadowMapDemandBudget.md](../../VirtualShadowMapDemandBudget.md)
