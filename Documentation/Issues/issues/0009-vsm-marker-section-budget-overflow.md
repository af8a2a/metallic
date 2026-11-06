---
id: "0009"
title: "VSM: 多面/多灯下每 face 生成的 marker 会耗尽 RenderGraph 256-section 预算，Projection 归因不可用"
area: shadow
kind: defect
priority: P2
state: open
milestone: M2-Correctness
labels: ["area:shadow", "kind:defect", "priority:P2"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapProfiling.md:42-45
  - Documentation/VirtualShadowMapProfiling.md:16-25
  - Documentation/VirtualShadowMapProjectionStatistics.md:37
  - Documentation/VirtualShadowMapCacheLayers.md:51
  - Documentation/VirtualShadowMapLightGridProjection.md:60-64
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:1215-1341
  - Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:2432
  - Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp:2565
---

## 背景 / Context

`benchmarkGPUProfile` 让 VSM 在每个 face 的 `record()` 里按阶段发 GPU marker：
`VSM reset` / `VSM mark` / `VSM allocator` / `VSM build` / `VSM raster` /
`VSM projection`，每个父区间再带 2–5 个子区间（`VirtualShadowMaps.cpp:1215-1341`、
`VirtualShadowMapProfiling.md:14-24`）。点光六面、多灯场景会把这一整套重复
`lightCapacity × 6` 次，同时 CPU scope 与 GPU scope 共享同一个每 node 预算。

## 现状 / Current state

- 预算是硬编码的 256：`RenderGraphExecutor.cpp:2432`（追加 section 时
  `stats.sections.size() >= 256` → `profilingOverflow = true`）与 `:2565`
  （加入 samples 时 `size() + samples.size() > 256`）。溢出后该 node 的时间戳直接不发布。
- 文档已经明确要求使用者检查它：「The profiler has a 256-section per-node budget shared by
  CPU/GPU sections……inspect `profilingOverflow` when scaling beyond the
  single-directional-light Sponza case. Overflow or missing timestamps are not zero-cost
  work.」（`VirtualShadowMapProfiling.md:42-45`）。
- 触发是实际发生过的，不止一次：
  - 双点光 + 后续近面共 12 状态的 CPU profile 树超过 256 上限，`pointTables` 计数被标为
    明确 unavailable（`VirtualShadowMapProjectionStatistics.md:37`）；
  - 缓存分层的图矩阵辅助臂 timestamp query 溢出，只能改成「仅 profile 分层臂」
    （`VirtualShadowMapCacheLayers.md:51`）；
  - grid projection 的实验里出现「timestamp-capacity 失败」，并给出规避办法：用
    `benchmarkProjectionProfile: true` 配 `benchmarkGPUProfile: false`
    （`VirtualShadowMapLightGridProjection.md:60-64`）。
- 现状的规避手段是**关掉**详细 GPU profile 或缩小 profile 范围，也就是在需要归因的地方
  关掉归因。

## 问题或缺口 / Problem

1. 「溢出不等于零成本」写在文档里，但消费方（报告工具、测试夹具）对缺失 section 的处理
   是标 `missing`/`unavailable`，不是失败；同一个节点可能一部分阶段有值、一部分缺失，
   很容易被读成「该阶段不耗时」。
2. 每 face 一套 marker 的设计与「点光六面」的几何形状天然冲突：单方向光 Sponza 恰好
   放得下，多灯就溢出，这不是偶发配置问题。
3. 现有规避手段（`benchmarkProjectionProfile`）只保住 Projection，其它阶段的子区间仍然
   不可用，且它改变了被测量的 workload（少一层 instrument），跨配置比较不成立。
4. 预算 256 是 RenderGraph 侧的常量，VSM 每 face 生成的 section 数没有上限也没有校验，
   两边没有约定，未来加一个 marker 就可能让某类场景静默进入不可用状态。

## 期望结果 / Expected outcome

把「VSM 在多面/多灯下如何降级」写成明确契约：要么一次性失败并报出具体 face/section
数，要么明确降级规则（例如按 light 聚合、只保留父区间），并且报告工具在这种降级下必须
报错而不是简单地标 missing。同时给出每 face 的 section 预算基线，作为扩 scope 的依据。

## 验收标准 / Acceptance criteria

- [ ] 在 `VirtualShadowMapProfiling.md` 写明契约：每 face 生成的 section 数上限、
      多面/多灯下的降级或失败行为、以及 `profilingOverflow` 的消费规则。
- [ ] 报告工具（`Tools/Perf/VirtualShadowMapMarkerProfile.py` /
      `VirtualShadowMapGPUProfile.py`）在 `profilingOverflow` 为真或缺 Projection 子区间时
      返回失败，而不是输出 `missing`；对应 Python fixture 测试覆盖该路径。
- [ ] 至少一组 2 点光 × 6 面的实测数据（每 face 实际 section 数 + 是否溢出）写入文档，
      作为预算基线；若溢出，必须给出降级后仍可用的阶段列表。
- [ ] `virtual_shadow_map_projection_equivalence` 的多灯状态在 `benchmarkGPUProfile`
      下 `profilingOverflow` 必须为 0，否则测试失败并在失败信息中报出 face 与 section 数。
- [ ] 若决定改预算而非降级，需说明对其它 pass 的 section 预算影响（该 256 是每 node
      共享的），并给出同时采集 CPU profile 时的上限核算。

## 相关文档 / References

- [VirtualShadowMapProfiling.md](../../VirtualShadowMapProfiling.md)
- [VirtualShadowMapsGPUProfile.md](../../VirtualShadowMapsGPUProfile.md)
- [VirtualShadowMapProjectionStatistics.md](../../VirtualShadowMapProjectionStatistics.md)
- [VirtualShadowMapLightGridProjection.md](../../VirtualShadowMapLightGridProjection.md)
- [Marker profile tool](../../../Tools/Perf/VirtualShadowMapMarkerProfile.py)
