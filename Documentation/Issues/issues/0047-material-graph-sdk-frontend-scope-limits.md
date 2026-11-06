---
id: "0047"
title: "Material: MaterialGraph 与受限 SDK 前端只覆盖单材质 Surface 子集，任意原生 Slang 扩展未实现"
area: material
kind: feature
priority: P2
state: in-progress
milestone: M1-Active
labels: ["area:material", "kind:feature", "priority:P2", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/MaterialSystemM5MaterialGraph.md:16
  - Documentation/MaterialSystemM5MaterialGraph.md:51
  - Documentation/MaterialSystemM5MaterialGraph.md:55
  - Documentation/MaterialSystemM5MaterialGraph.md:82
  - Documentation/MaterialSystemM5MaterialGraph.md:88
  - Source/Runtime/Material/MaterialSlangFrontend.cpp:14-16
---

## 背景 / Context

路线图 M3 的结束条件是「图能保存、重载、诊断并产生稳定编译产物」，并明确「参数编辑
无需代码重编译」「材质在编辑器之外也能编译和验证」。M5 文档记录的是这个方向的首版
authoring 闭环。

## 现状 / Current state

- 已交付：图或 SDK 表达式 → Value/Closure IR → `.materialdef` / `.material` →
  SceneDocument → 现有 PT / Deferred lighting；有 ImNodes 画布、编译诊断与反射、
  撤销/重做、`.materialgraph` 保存加载、`MetallicMaterialCompile` 离线编译。
- 但应用范围是**单个选中材质**：「首版只对**一个选中材质**应用；多材质模型先在
  Scene Browser 选择其具体材质」（`MaterialSystemM5MaterialGraph.md:16`）。
- 组合能力受限：「`Mix/Layer(OpenPBR,...)` 与多于两个 Slab 的组合明确拒绝」
  （`MaterialSystemM5MaterialGraph.md:51`）。
- 文本前端**不是** Slang 编译器：「文本模式是 **Slang 语法的受限表达式前端**，
  不加载任意原生 Slang module……任意 Slang 函数、循环、预处理器、资源写入、动态资源
  声明和新增 BSDF 类型仍不属于此版本」（`MaterialSystemM5MaterialGraph.md:55`）。
  代码侧同样写明文法是封闭的：`This grammar is deliberately closed. Source text never
  enters generated shader includes.`（`Source/Runtime/Material/MaterialSlangFrontend.cpp:14-16`）。
- 文档自己定性：「它不等价于 Phase 14 设想的任意原生 Slang 扩展 SDK。完整
  module/interface 加载及新增闭包实现仍是后续范围」
  （`MaterialSystemM5MaterialGraph.md:82`）。
- 预算固定：128 图节点、16 KiB SDK/IR、256 唯一 Value 节点、4×float4 实例参数、
  最多两个 Slab；场景最多 64 Surface + 64 Coverage 程序
  （`MaterialSystemM5MaterialGraph.md:88`）。

## 问题或缺口 / Problem

1. **作者无法表达「新模型」。** 现有前端只能重排已有节点（OpenPBR / Slab / 数学），
   不能引入新的 BSDF 类型或自定义闭包实现；路线图「自定义参数程序」的目标只完成到
   「参数计算不同、散射模型相同」。
2. **单材质应用阻断批量工作流。** 真实资产常常一个模型多个材质；逐一手选材质把
   authoring 变成手工操作，也无法在离线编译流程里整体验证场景。
3. **预算与能力的对应关系只写在文档。** 超出预算时的诊断是否定位到具体节点/槽位没有
   验收项，而这是 M3 门槛（「非法 domain/operator 组合能定位到节点」）的一部分。

## 期望结果 / Expected outcome

- 至少一条「新增闭包实现」的路径可用：受约束的 Slang 模块/接口可以注册为新 BSDF，
  经同一 IR、同一 schema/manifest、同一 GPU 上传与生命周期，并进入
  `MaterialGraph` 与 SDK 前端。
- 图编辑支持一次应用到多材质（或至少支持「按材质名批量设置」的脚本/CLI 入口）。
- 预算诊断可定位：哪个节点、哪个槽位、哪条预算超限。

## 验收标准 / Acceptance criteria

- [ ] `MetallicSceneTests --gtest_filter='MaterialGraph.*:MaterialAssets.*'` 保持全绿，
      并新增至少一个「多材质应用」用例：一个模型 N 个材质，一次操作后 N 个材质都更新，
      undo 可整体回滚。
- [ ] `MetallicMaterialCompile` 支持输入包含多个材质的场景并整体导出，输出每个材质的
      `.materialdef` / `.reflection.json`，失败时指出具体材质。
- [ ] 新增一个自定义闭包注册样例：能被 `.material` 资产引用、在 PT 与 VBuffer 求值、
      且编译键稳定（重命名节点/改布局不改变 key）。
- [ ] 超预算用例（>128 节点、>2 Slab、>16 KiB、>64 程序）各自返回定位到节点/材质的
      诊断，且不发布部分结果。
- [ ] 文档把「任意原生 Slang module 加载」从「后续范围」更新为具体状态（已实现 /
      明确不做），并说明替代路径。

## 相关文档 / References

- [MaterialSystemM5MaterialGraph.md](../../MaterialSystemM5MaterialGraph.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
- [MaterialSystemM4SceneClosure.md](../../MaterialSystemM4SceneClosure.md)
