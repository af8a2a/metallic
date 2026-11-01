---
id: "0044"
title: "Material: M2 实施记录与代码现状脱节（稀疏 Program 分箱已生产默认开启，文档仍写未实现）"
area: material
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:material", "kind:task", "priority:P2"]
depends_on: []
evidence:
  - Documentation/MaterialValueProgramsM2.md:14
  - Documentation/MaterialSystemRoadmap.md:168-170
  - Source/Runtime/Render/MaterialBinning.cpp:50-51
  - Source/Runtime/Render/MaterialBinning.h:12
  - Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp:514
---

## 背景 / Context

`MaterialSystemRoadmap.md` 是材质系统的主线文档，阶段状态全部由它和各 Phase /
里程碑文档描述。M2（自定义 Value Program 闭环）被标注为「进行中」，而 M2 门槛之一
是「稀疏 Program tile 任务 / 每程序精确分类」。

## 现状 / Current state

- `MaterialValueProgramsM2.md` 第 7 行把 M2 状态写成「**M2 进行中**……完整 M2 的
  coverage、TextureFootprint、稀疏 Program tile 和压力性能门槛仍按路线图执行」。
- 同一文档第 14 行更具体：「自定义场景使用未分类的通用 VBuffer kernel。旧五类分箱
  只适用于未修改的固定因子；**M2 稀疏程序分箱尚未实现**。无自定义程序的场景保持
  原有分箱。」（`MaterialValueProgramsM2.md:14`）
- 但代码里稀疏 Program 分箱已经是生产默认路径：
  - `MaterialBinning.h:12` 定义 `kMaxMaterialProgramBins = 4096`；
  - `MaterialBinning.cpp:50-51` 用 `programBinCount != 0` 决定 sparse/general 分配，
    sparse 时 bin 数为真实 Program 数，否则退回 `kMaterialClassCount` 固定五类；
  - `ScenePathTracePass.cpp:514` 注册运行时设置
    `programBinning` / **Sparse Material Program Bins**，默认值 `true`。
- 同一文档第 69 行的「尚未关闭」清单同时混有已经被后续 Phase 关闭的项（coverage 同代
  一致性由 Phase 8 + native coverage 覆盖，统一 Value IR 与显式 footprint 由 Phase 9
  交付），也没有标注哪些项仍然成立。
- Phase 11 文档在描述 split 原型时反而写着它「不替换 Phase 6 的**生产**稀疏分配器」
  （`MaterialSystemPhase11.md:57`），与 M2 文档的说法直接冲突。

## 问题或缺口 / Problem

1. **里程碑状态不可信。** 读文档的人会得到「M2 稀疏分箱未实现、coverage 一致性未关闭」
   的结论，而代码里两者都已进入生产默认路径。这直接影响「M2 是否已完成」的判断。
2. **真正的 M2 剩余项被噪音淹没。** 经过代码核对，M2 仍未关闭的实际上是：
   TextureFootprint 在 RT 侧的 cone/LOD 适配与掠射/UV seam/跨 primitve/tile 边缘测试、
   程序规模压力扫描（1/16/64/256 程序、1/100/1000 实例的编译时间/SPIR-V 体积/寄存器/
   运行成本）、以及屏幕外反射自定义程序的专门数值验证。
3. **文档互相矛盾。** `MaterialValueProgramsM2.md` 与 `MaterialSystemPhase11.md` 对
   「稀疏 Program 分箱是否存在」给出相反陈述。

## 期望结果 / Expected outcome

`MaterialValueProgramsM2.md` 成为一份与代码一致的 M2 状态快照：逐条列出已交付项
（附代码入口）、仍缺项（附验收方法）、以及被后续 Phase 取代的项。

## 验收标准 / Acceptance criteria

- [ ] `MaterialValueProgramsM2.md` 删除或改写第 14 行的「稀疏程序分箱尚未实现」，改为
      指向 `programBinning` 运行时设置与 `MaterialBinning.cpp` 的 sparse 分配分支。
- [ ] 第 69 行「尚未关闭」清单逐项标注状态（仍缺 / 由 Phase 8 关闭 / 由 Phase 9 关闭），
      每项给出可点击的文档或代码路径。
- [ ] `MaterialSystemPhase11.md` 与 `MaterialValueProgramsM2.md` 关于生产稀疏分配器的
      表述一致，不再出现「不替换生产稀疏分配器」与「尚未实现」并存。
- [ ] `MaterialSystemRoadmap.md` M2 段的「当前状态」更新为与上述一致的结论。
- [ ] 更新后由第二人按文档描述的入口逐一打开核对（文档内不出现无法定位的路径）。

## 相关文档 / References

- [MaterialValueProgramsM2.md](../../MaterialValueProgramsM2.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
- [MaterialSystemPhase9.md](../../MaterialSystemPhase9.md)
- [MaterialSystemPhase11.md](../../MaterialSystemPhase11.md)
