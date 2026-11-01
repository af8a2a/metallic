---
id: "0002"
title: "VSM: workload 诊断已到 v3/23 计数器，四份文档仍写 v1/16、v2/19 与旧字节布局"
area: shadow
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:shadow", "kind:task", "priority:P2"]
depends_on: ["0001"]
evidence:
  - Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h:29-37
  - Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h:58-60
  - Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h:90-96
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:858-863
  - tests/rhi/VirtualShadowMapAcceptanceTests.cpp:192
  - Tools/Perf/VirtualShadowMapGPUProfile.py:44-52
  - Documentation/VirtualShadowMapsGPUProfile.md:29-31
  - Documentation/VirtualShadowMapsGPUProfile.md:53-55
  - Documentation/VirtualShadowMapAdaptiveRayCount.md:21-31
  - Documentation/VirtualShadowMapBuildAttribution.md:3
---

## 背景 / Context

VSM 的诊断 ABI 由三层组成：每 face 32 个公开 `uint32` statistics（语义不变）、紧随其后的
workload 计数器段（low/high `uint32` 对）、再后面的 build 诊断段。三段长度都由
`VirtualShadowMapProfiling.h` 的常量决定，acceptance JSON 里的 `shadow.gpuWork.version`
标识 workload 段布局。

## 现状 / Current state

- 代码（工作树，尚未进入 HEAD）：workload 段已扩到 23 项
  （`kVirtualShadowMapWorkCounterCount = 23`，`VirtualShadowMapProfiling.h:29-37`），
  新增 `projectionPageTableLoads`、`projectionMappingLoads`、
  `projectionMappedAncestors`、`projectionMappingSlowPaths` 四项（名称表 `:90-96`）。
  HEAD 里仍是 `kVirtualShadowMapWorkCounterCount = 19`
  （`git show HEAD:Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h`），
  本文行号与版本号均按工作树版本核对。
- build 诊断段已是 v3 / 45 项 / 360 B（`VirtualShadowMapProfiling.h:56-60`），
  workload 段字节偏移改为 `32 + 23 * 2` 个 `uint32`（`:60`）。
- acceptance 输出已写 `version 3`（`VirtualShadowMapAcceptanceTests.cpp:192`；
  HEAD 版本同行为 `version 2`，见 `git show HEAD:tests/rhi/VirtualShadowMapAcceptanceTests.cpp`）。
- 报告工具已支持 v1/v2/v3 三套计数器表（`VirtualShadowMapGPUProfile.py:44-52`），
  并新增两个 fixture 测试覆盖 v1/v2 前缀兼容与 totals 校验。
- 文档仍未同步：
  - `VirtualShadowMapsGPUProfile.md:29-31` 写「每 face 16 个 `uint64`」「byte 256 起的
    25 个 `uint64`」「work 开启时每 face 私有 statistics 总长为 456 B」；
  - `VirtualShadowMapsGPUProfile.md:53-55` 写 build 诊断 v1 200 B、v2 344 B；
  - `VirtualShadowMapAdaptiveRayCount.md:21-31` 写「version 2 追加三项到十六项」
    「公开 32-word 统计不变」；
  - `VirtualShadowMapBuildAttribution.md:3` 写「不改变 stats32、work16 或 build
    diagnostics v3 的 ABI」。

## 问题或缺口 / Problem

1. 三处数字互相矛盾且都与代码不符：现网私有 statistics 长度是
   `32*4 + 23*8 + 360 = 672` B，文档写 456 B；workload 是 23 项，文档写 16 项。
2. 偏移公式只存在于源码常量里。任何按文档手算 `buildDiagnostics` 偏移的读者都会读到
   错误位置，而 `kVSMReceiverCoverageStatisticsWordOffset`（`:68-69`）等相邻偏移也随之
   变化。
3. 「work16」这类历史名称在多个文档里被当作 ABI 契约反复引用（
   `VirtualShadowMapBuildAttribution.md:3`、`VirtualShadowMapsGPUProfile.md:31`），
   读者无法判断当前版本。
4. GPU profile 文档没有写 workload 段有版本号，也没有给 v1/v2/v3 的差别表，导致
   「读历史包」与「读当前包」的规则散落在工具代码里。

## 期望结果 / Expected outcome

`VirtualShadowMapsGPUProfile.md` 成为诊断 ABI 的唯一说明源：一处列出 v1/v2/v3 的
计数器数量、字节布局、版本字段与历史包读取规则；其他文档只引用它，不再复述数字。

## 验收标准 / Acceptance criteria

- [ ] `VirtualShadowMapsGPUProfile.md:29-31` 改为 v3 描述：23 项 workload、45 项 build
      诊断、360 B，并给出私有 statistics 总长 672 B 的算式（
      `32*4 + 23*8 + 360`）。
- [ ] `VirtualShadowMapsGPUProfile.md:53-55` 的 v1/v2/v3 小节补齐 v3，并说明 histogram
      与 bitset 两组计数的关系未变。
- [ ] `VirtualShadowMapAdaptiveRayCount.md:21-31` 改为「v2 追加三项、v3 再追加四项」，
      保留 `projectionAdaptiveSavedRays` 在 2-ray 配置下的语义说明。
- [ ] `VirtualShadowMapBuildAttribution.md:3` 去掉 `work16`，改为引用 ABI 章节。
- [ ] `python -B -m unittest tests/perf/TestVirtualShadowMapGPUProfile.py` 通过；
      CTest `MetallicVirtualShadowMapGPUProfileTests` 通过。
- [ ] `MetallicRHITests --filter virtual_shadow_map_profiling` 与
      `virtual_shadow_map_acceptance` 相关状态通过，`Shadows.workload` 长度断言
      （`VirtualShadowMapAcceptanceTests.cpp:196-197`）在 v3 下成立。
- [ ] 文档中不再出现 `work16` 或「16 个 `uint64`」字样（`grep -rn "work16\|16 个 uint64"
      Documentation/` 为空）。

## 相关文档 / References

- [VirtualShadowMapsGPUProfile.md](../../VirtualShadowMapsGPUProfile.md)
- [VirtualShadowMapAdaptiveRayCount.md](../../VirtualShadowMapAdaptiveRayCount.md)
- [VirtualShadowMapBuildAttribution.md](../../VirtualShadowMapBuildAttribution.md)
- [VirtualShadowMapProfiling.md](../../VirtualShadowMapProfiling.md)
