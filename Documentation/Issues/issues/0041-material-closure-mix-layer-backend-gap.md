---
id: "0041"
title: "Material: Closure Mix/Layer 执行后端只有单面漫反射 Slab，M4 有界组合里程碑未闭环"
area: material
kind: feature
priority: P1
state: in-progress
milestone: M1-Active
labels: ["area:material", "kind:feature", "priority:P1", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/MaterialSystemPhase10.md:26-29
  - Documentation/MaterialSystemPhase10.md:53-55
  - Documentation/MaterialSystemM4SceneClosure.md:79-85
  - Documentation/MaterialSystemRoadmap.md:201-208
  - Source/Runtime/Render/Material/MaterialRuntime.cpp:44-47
---

## 背景 / Context

路线图把 M4 定义为「有界 Mix 和 Layer」：先做 M4a MixClosures（至多两个 closure
record + 各自 normal basis），再做 M4b Layer（一个 dielectric coating 加一个受约束
Surface substrate，含介质 IOR、厚度、吸收、底层透射与内部多次反射）。M4 关闭后才
算达到首个架构验收点（三案例：OpenPBR 迁移、双层程序化表面、Fiber adapter）。

## 现状 / Current state

- Closure IR 与预算校验、`Mix`/`Layer` 的接线、`SingleSlab`/`DualSlab` canonical
  family 都已落地，并且已经进入生产 PT / VBuffer：场景资产可以声明
  `implementation: "Slab.Surface"` 的 v3 值图 + Slab 拓扑，命中点构造场景 Closure。
- 但执行后端只实现「单个 Slab」或「两个 Slab 加一个操作符」
  （`Documentation/MaterialSystemPhase10.md:26-29`）。放宽 `RealtimeBackendProfile`
  不会产生执行器，只给出 `no executable family` 诊断。
- 散射模型是单面、共享 normal basis 0、仅反射的漫反射界面：
  `fMix = (1-w)·fA + w·fB`，Layer 是 `r/π + T·T·A(wi,wo)·b/π` 的一次下/上穿透吸收。
  没有镜面、折射、薄膜、体散射或层间多次反射，被省略的能量也不重分配
  （`MaterialSystemPhase10.md:53-55`）。
- 生产注册表同样把这个边界写死成材质能力描述：`Slab.Single` 标注
  "no transmission"，`Slab.Dual` 标注 "no multiple inter-layer scattering"
  （`Source/Runtime/Render/Material/MaterialRuntime.cpp:44-47`）。
- 能力边界明确写着：不支持单独法线、镜面/折射、多次层间散射、任意深度图，也不支持
  BLEND/透射/Fiber/unlit（`MaterialSystemM4SceneClosure.md:79-85`）。

## 问题或缺口 / Problem

M4 的两个增量都只完成了「拓扑与执行骨架」，没有完成路线图定义的验收内容：

1. **M4a 缺少真正的双 normal basis。** 路线图要求「至多两个 closure record 和各自
   normal basis」，当前预算 `maxNormalBases = 1`，后端只接受共享
   `SurfaceMaterialContext` 的 basis 0。双法线（例如清漆法线与基底法线不同）无法表达。
2. **M4b 的 Layer 不是路线图要求的介质层。** 现有 Layer 是「反射率 + 光程吸收 + 底层
   反射率」的标量近似，没有介质 IOR/厚度/eta，也没有内部多次反射的处理范围。路线图
   明确禁止用 BSDF lerp 冒充 Layer，而当前实现无法给出与物理参考的误差预算对照。
3. **`Layer(Surface, Fiber)` 被拒绝**（合同上仍允许拒绝），但没有任何诊断路径说明
   该组合何时可解，M6/M7 的 Fiber 因此无法参与组合。
4. Surface 侧缺 BLEND/透射：Slab 材质不能表达玻璃或半透明，导致「双层玻璃 + 涂层」
   这类最典型的 Layer 用例在 Slab 后端根本不可用。

## 期望结果 / Expected outcome

- `RealtimeBackendProfile` 至少支持 2 个 normal basis，并提供至少一个真实的
  dielectric coating / substrate Layer 执行后端，其参数包含介质 IOR、厚度与吸收，
  且明确声明是否包含层间多次反射。
- 超预算的拓扑（更深 DAG、更多 basis、更大 payload）给出可定位的编译诊断，且与
  「拓扑合法但后端不支持」分开报告。
- 与物理参考的比对方法（白炉、能量上界、互易性、掠射角、高吸收）写入文档，误差指标
  可复算。

## 验收标准 / Acceptance criteria

- [ ] `MetallicRHITests --filter *material_closure_ir*` 通过，并新增至少一个覆盖
      `maxNormalBases = 2` 的 lowering 用例。
- [ ] `material_slab_furnace_sample_render` 扩展为新的双层介质材质：白炉能量上界
      ≤ 1.00001，Sample/Eval/PDF 一致性、互易性、掠射角、零权重与端点有限性全部通过。
- [ ] PT 与 VBuffer 对同一个新 Layer 材质产生相同语义（相对差 ≤ 已声明的误差指标），
      并保留 `material_value_closure_scene` 现有 Layer 边界断言。
- [ ] 超预算拓扑（3 个 closure / 2 个操作符 / 2 个 basis）返回明确的编译诊断，
      不产生静默近似。
- [ ] 文档更新 `MaterialSystemM4SceneClosure.md` 能力边界表，逐项标注「已实现 /
      明确拒绝」，不再出现「仍是原型」与「已闭环」并存的表述。

## 相关文档 / References

- [MaterialSystemPhase10.md](../../MaterialSystemPhase10.md)
- [MaterialSystemM4SceneClosure.md](../../MaterialSystemM4SceneClosure.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
- [MaterialSystemPhase11.md](../../MaterialSystemPhase11.md)
