---
id: "0066"
title: "Sky: 大气天空渲染通道停在未合并分支 modernization_/sky，需重新落地或明确废弃"
area: material
kind: feature
priority: P2
state: blocked
milestone: M3-Hardening
labels: ["area:material", "kind:feature", "priority:P2", "state:blocked"]
evidence:
  - "git:modernization_/sky 0e4e16cf4 Add atmospheric sky render pass and samples"
  - Source/Runtime/Render/RenderPass/BuiltinPass
  - Documentation/ProjectArchitecture.md
---

## 背景 / Context

`modernization_/sky` 分支上有一个完整的大气天空渲染通道实现（单个提交
`0e4e16cf4`），相对当前 `modernization` 主线是 **+1690 / −4 行、11 个文件**。
当前主线不存在任何 `AtmosphericSky` 代码：在 HEAD 上
`git grep -i AtmosphericSky -- Source Shaders` 无结果，也就是说这个功能
既没有合入，也没有在主线留下任何开关或占位。

## 现状 / Current state

分支包含的完整内容：

| 文件 | 规模 | 说明 |
| --- | --- | --- |
| `Shaders/AtmosphericSky.slang` | +579 | 天空/大气散射着色器 |
| `Source/Runtime/Render/RenderPass/BuiltinPass/AtmosphericSkyPass.cpp` | +669 | 渲染通道实现 |
| `Pipelines`/`Asset` 下两个 `*.metallic_graph.json` | +59 / +118 | 渲染图样例与 Sponza GPU 驱动样例 |
| `tests/rhi/RenderGraphTests.cpp` | +191 | 通道接入渲染图的测试 |
| `BuiltinPasses.h` / `BuiltinRenderPasses.cpp` / `BuiltinPassCommon.h` / `Source/CMakeLists.txt` / `EditorApplication.cpp` | +13 | 注册与编辑器接线 |

## 问题或缺口 / Problem

1. 主线缺少大气/物理天空，因此 `PhysicalEnvironment*`、`SphericalHarmonics`
   等环境光照路径只能依赖外部 HDRI，场景配置与 LookDev 缺少程序化环境来源。
2. 该实现带着自己的一组渲染图样例与测试，与主线这一个月内的渲染图、任务图、
   VSM 改动已经产生漂移风险；分支越久越难合并。
3. 由于代码不在主线，它不参与任何 CI 或验证，实际状态无法判断（是可用、半成品
   还是原型）。

## 期望结果 / Expected outcome

给出一次明确决策，并把结论写回本 issue：

- **重新落地**：在主线视角下 rebase，补齐注册与编辑器接线，使
  `MetallicRHITests -R RenderGraph` 覆盖该通道，跑通至少一个内置样例场景的截图验证；
- **或明确废弃**：删除分支并把结论（为什么不做、替代方案是什么）记录到
  `Documentation/` 下的对应文档，避免后续重复调研。

## 验收标准 / Acceptance criteria

- [ ] 决策记录写入本 issue（保留或废弃，含理由）
- [ ] 若保留：`git log modernization..modernization_/sky` 为空，且分支内容已进入主线
- [ ] 若保留：`AtmosphericSkyPass` 已注册进 builtin pass 列表，编辑器可开关
- [ ] 若保留：`ctest --test-dir build-ci -R '^MetallicRHITests$'` 中渲染图用例覆盖该通道并通过
- [ ] 若保留：至少一个样例场景（Sponza 或内置样例）出图，天空可见且无 NaN/闪烁
- [ ] 若废弃：分支已删除，`Documentation/` 有对应的废弃说明

## 相关文档 / References

- 分支提交：`0e4e16cf4 Add atmospheric sky render pass and samples`
- [ProjectArchitecture.md](../../ProjectArchitecture.md)
- [PhysicalEnvironmentCheckpointA.md](../../PhysicalEnvironmentCheckpointA.md)、[PhysicalEnvironmentCheckpointB.md](../../PhysicalEnvironmentCheckpointB.md)
- [SphericalHarmonics.md](../../SphericalHarmonics.md)
