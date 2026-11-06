---
id: "0001"
title: "VSM: propagatedMapping 工作树默认值已翻转为开启，缺默认路径验收、单面 Pass 开关与文档同步"
area: shadow
kind: perf
priority: P1
state: in-progress
milestone: M1-Active
labels: ["area:shadow", "kind:perf", "priority:P1", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/VirtualShadowMapPropagatedMapping.md:3-7
  - Documentation/VirtualShadowMapPropagatedMapping.md:147-165
  - Documentation/VirtualShadowMapPropagatedMapping.md:231-234
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.h:63
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp:380
  - Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp:39-47
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:1315-1321
  - Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang:4-6
  - tests/rhi/VirtualShadowMapPropagatedMappingTests.cpp:170-232
  - tests/rhi/VirtualShadowMapSceneTests.cpp:3354-3364
  - Pipelines/Samples/virtual_shadow_map_sponza.metallic_graph.json:36-64
---

## 背景 / Context

`propagatedMapping` 是投影侧的常量表读取快路径：每帧在 Finalize 之后重建一张
`uint4(physicalPage, targetLevel, packedParentPage, mode)` 派生表，让本页/整数父页/
局部光 dyadic mip 命中时用一次派生表读取代替原生层级搜索。文档记录了三次迭代
（精确边界证书 → 整数父页相对坐标 → compact 查找状态），三次都只有一个方向光
ABBA 区组的 Projection 数据。

## 现状 / Current state

- 文档三处明确写「默认关闭、等待采纳」：`VirtualShadowMapPropagatedMapping.md:3-7`
  写 "It is off by default"，`:147-165` 写 "remains disabled by default; no performance
  adoption or independent statistical confirmation is claimed"，`:231-234` 写
  "disabled by default pending adoption"。
- 整条候选（设置字段、Pass 属性、shader 分支、探针、文档、4 个新计数器）都还没有进入
  HEAD：`git show HEAD:` 下的 `VirtualShadowMaps.h` 没有 `propagatedMapping` 字段、
  `VirtualShadowMaps.slang` 没有 `VSM_PROPAGATED_MAPPING`、
  `VirtualShadowMapLightsPass.cpp` 没有该属性解析，
  `VirtualShadowMapPropagatedMapping.md` 本身也还是未跟踪文件。
- 矛盾就发生在这批未提交材料内部：工作树
  `Source/Runtime/Render/Shadow/VirtualShadowMaps.h:63` 是 `bool propagatedMapping = true;`
  （本文所有源码行号均按工作树版本），而文档三处写默认关闭；没有任何默认切换提交，也没有
  `build/vsm-*-default-*` 形式的默认路径验证包。
- 关闭开关只挂在多面 Pass 上：`VirtualShadowMapLightsPass.cpp:380` 解析
  `benchmarkPropagatedMapping`；单面 `VirtualShadowMapPass.cpp:39-47` 的属性表里没有这一项。
- 自动化覆盖只有白盒探针：生产模块在 `VSM_PROPAGATED_MAPPING_PROBE` 下 `__include` 测试
  探针 `tests/rhi/shaders/VirtualShadowMapPropagatedMappingProbe.slang`
  （`VirtualShadowMaps.slang:4-6`），19 个 case 直接调用 `vsmBuildPropagatedMapping` 与
  `vsmFilterLookup`（`VirtualShadowMapPropagatedMappingTests.cpp:170-232`），没有任何状态
  经过 `VirtualShadowMaps::record/project`。现成的六臂控制机制
  （`VirtualShadowMapSceneTests.cpp:3354-3364` 的 `selectionControls` 与
  `productionDefault`）也没有被用于该候选。
- 作者自述的边界：默认路径验证缺失（doc:216-219）、局部光/点光面/混灯 one-pass 整场景
  验收未做（doc:231-233）、每帧多出 `totalPages * 16` B（Sponza 约 1.5 MiB）与一次
  propagation dispatch（`VirtualShadowMaps.cpp:1315-1321`）。

## 问题或缺口 / Problem

1. 默认值与三处文档互相矛盾，任何对照臂、Profiler 读数与用户报告都无法确定当前基线包含
   哪个候选；采纳方式也与仓库惯例不一致（ARC、grid projection、mark hash coverage 各有
   一份 `build/vsm-*-default-2026xxxx/` 默认路径验证，本轮没有）。
2. 单面 Pass 没有开关，该候选在单面图（编辑器/样例 `VirtualShadowMapPass`）里既不能
   A/B 也不能回退，回归时只能改源码。
3. mode 2（方向光整数父页）与 mode 3（局部光 dyadic mip）只在 19 个探针 case 与单方向
   光 Sponza 上验证过；改写了数值契约（doc:102-109 承认会跨页边界取到另一个父页），
   却没有 pass 级等价矩阵。
4. 新增的 `VSM propagate mappings` scope 嵌在 build/finalize 里（doc:143-145），默认开启
   后 Build 与 Projection 的区间都变了，但没有任何开启后的整图指标复核。

## 期望结果 / Expected outcome

先决策再收口：要么完整采纳（默认 true + 默认路径验证 + 等价矩阵 + 单面 Pass 属性 +
文档改写），要么回退（默认 false + 文档保持原样）。两者都不允许「代码 true、文档 false」
的组合随这批改动一起提交。

## 验收标准 / Acceptance criteria

- [ ] 决策写入 `VirtualShadowMapPropagatedMapping.md` 顶部状态行，并在
      `VirtualShadowMapProductionOptimizations.md` 的固化表或排除列表中登记。
- [ ] 若采纳：`virtual_shadow_map_projection_equivalence` 增加
      `propagatedMapping` 的 Off/On/Default/Toggle 四臂（复用
      `VirtualShadowMapSceneTests.cpp:3354-3364` 的 `selectionControls` 与
      `productionDefault` 断言），图像、depth、HDR、stats32、45 项 build diagnostics
      与全部 workload 计数器逐字节一致。
- [ ] 若采纳：`VirtualShadowMapPass.cpp` 增加 `benchmarkPropagatedMapping` 解析并与
      `VirtualShadowMapLightsPass.cpp:380` 对齐，补一条单面 pass 等价状态。
- [ ] 若采纳：新增 `build/vsm-propagated-mapping-default-<date>/`，使用不含
      `benchmarkPropagatedMapping` 的 graph，给出 ≥12 帧 PNG/HDR/depth/shadow
      逐字节对比 + ≥2 个 ABBA 区组的 Projection/Shadows/RenderGraph 区间。
- [ ] 至少一个局部光/点光面 one-pass 整场景状态覆盖 mode 3 与 mode 2 两条分支
      （可用现有 Mixed fixture），不复用探针结果充数。
- [ ] 文档 `:3-7`、`:147-165`、`:231-234` 改写为当前默认值，并保留「只有单方向光
      ABBA 数据」这一证据边界。
- [ ] 收尾同批提交：默认值与文档改写不再出现「字段 `true` + 文档 off by default」的组合
      （用 `git diff HEAD -- <paths>` 与 `git show` 自查，不依赖暂存区状态）；探针新增的
      实现单元使模块审计计数变为 389 文件 / 113 实现单元
      （`python -B Tools/CheckShaderModules.py`），`VirtualShadowMapProductionOptimizations.md:25`、
      `VirtualShadowMapShadowLOD.md:258`、`VirtualShadowMapDirectionalProjection.md:26`
      处的旧计数同步更新。

## 相关文档 / References

- [VirtualShadowMapPropagatedMapping.md](../../VirtualShadowMapPropagatedMapping.md)
- [VirtualShadowMapProductionOptimizations.md](../../VirtualShadowMapProductionOptimizations.md)
- [VirtualShadowMapProfiling.md](../../VirtualShadowMapProfiling.md)
- [VirtualShadowMapDirectionalProjection.md](../../VirtualShadowMapDirectionalProjection.md)
