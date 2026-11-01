---
id: "0048"
title: "Fiber/Strands: 原生曲线只有屏幕可见性后端，缺 RT 曲线求交、Far Field 模型注册与阴影/时域输运"
area: material
kind: feature
priority: P1
state: in-progress
milestone: M1-Active
labels: ["area:material", "kind:feature", "priority:P1", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/MaterialSystemM7Strands.md:53
  - Documentation/MaterialSystemM8RayExecution.md:59
  - Documentation/MaterialSystemM6Fiber.md:27
  - Documentation/MaterialSystemM6Fiber.md:53-56
  - Documentation/MaterialSystemRoadmap.md:227-237
  - Shaders/Features/Strands/NativeStrands.slang:71-113
---

## 背景 / Context

Fiber（M6）、原生 Strand（M7）与统一 ray 执行（M8）构成同一条毛发/纤维管线：
M6 交付 DOTS 三角形上的 Fiber 材质适配，M7 交付原生变半径曲线 → 有限多层屏幕可见性 →
共享 Fiber lighting，M8 把 ray-hit 的求值与准备边界统一到同一 Program 契约。路线图 M6
阶段要求「静态与运动下分别检查轮廓、排序/覆盖、阴影、透射、LOD 切换、重投影和超容量
场景」，并强调 DOTS 光追结果不能替代 strand 光栅验证。

## 现状 / Current state

- M6 已有可用的 Fiber 材质后端：三阶段接口（加载参数 → 构造 Closure → 按 `wo` 准备）、
  prepared 缓存、projected scattering（不叠加 Surface `NdotL`）、vendor RTXCRHair 原生
  Slang 模块与 4096 组参数/方向的 GPU oracle（最大绝对差 `6.1035e-5`）。
- M7 已有原生曲线后端：72 根 / 576 段 / 显式 8 层的 sample groom，`NativeStrands.slang`
  以 8×8 compute 组遍历段并做有界层插入（`Shaders/Features/Strands/NativeStrands.slang:71-113`），
  含容量/溢出诊断、覆盖合成、稳定身份、形变与密度 LOD。
- 但可见性只在屏幕空间成立：「这是可复用的小型 groom 验证后端；大规模 groom 的 tile
  binning、GPU 曲线导入/动画、LSS / secondary ray 是后续工作」，且光照是「固定方向主光
  和补光……不是 deep opacity map，未接入网格阴影、HDRI、多次散射或全局 Fiber 输运」
  （`MaterialSystemM7Strands.md:53`）。
- M8 明确写出缺口：「M7 原生屏幕曲线**尚无 RT 曲线求交后端**，Fiber ray-hit 验证使用
  现有 DOTS/triangle」（`MaterialSystemM8RayExecution.md:59`）。
- 材质模型侧只注册了一个场景 Program：「当前场景 Program 固定使用 normalized melanin
  absorption；不暴露在该模型中无效的 baseColor 或 Far Field diffuse 参数。其他吸收模型
  和 Far Field 在原生模块中可用，**尚未注册为新的场景 Program**」
  （`MaterialSystemM6Fiber.md:27`）。
- 输运能力仍只有 Radiance：「本阶段支持 Radiance transport；Importance 返回无效样本」，
  且「保留上游 Chiang 的采样 PDF 和已有环境 NEE 的近似 standalone PDF/MIS 策略，
  运行时 `exactStandalonePdf=false`」，并声明这不证明估计器无偏
  （`MaterialSystemM6Fiber.md:53-55`）。

## 问题或缺口 / Problem

1. **原生曲线无法参与光追求交。** 屏幕可见性只覆盖主视角；反射中、阴影中、相机外
   以及被遮挡后的 groom 全部缺失。这是「原生 strand 后端」与「生产可用」之间最大的
   断层，也意味着 M7 的可见性结果不能被 M8 的 Fiber ray 路径复用。
2. **材质模型覆盖面过窄。** Far Field 与其他吸收模型已在 vendored 模块中可用，但没有
   注册成场景 Program，作者无法选用。
3. **阴影与输运是近似而非验证过的估计器。** 主光解析自阴影 + `exactStandalonePdf=false`
   + Radiance-only，意味着毛发阴影与多次散射既无物理验收，也无 Importance 支持。
4. **时域完全未接。** `Fiber.motion` / `Fiber.identity` 已经提供重投影所需的稳定信息，
   但「本阶段未直接接入 DLSS / NRD / 时域重建」（`MaterialSystemM7Strands.md:48`）。

## 期望结果 / Expected outcome

- 至少一种曲线的 RT 求交后端（BVH 内曲线 primitive 或 LSS），使原生 strand 能出现在
  二次命中、反射与阴影查询中，且与屏幕可见性使用同一 Fiber Material Program。
- Far Field 与至少一个替代吸收模型注册为可选场景 Program，并在 `.material` schema
  中可声明。
- 毛发阴影有明确的质量声明与对照（deep opacity vs 解析自阴影），时域重投影接入
  至少一条重建路径。

## 验收标准 / Acceptance criteria

- [ ] 新增 GPU 回归：同一 groom 在「屏幕可见性 only」与「带 RT 曲线求交」两种模式下，
      主视角合成色在声明的容差内一致；反射/二次命中场景下 RT 模式出现可测量的间接贡献
      （关闭该贡献时输出全黑）。
- [ ] `RHIRendering.native_strand_visibility` 与
      `RHIRendering.material_fiber_native_and_stages` 保持通过，并新增 RT 曲线命中用例
      （相邻段、段端、亚像素半径、背面、root-to-tip 参数正确性）。
- [ ] 新增 Far Field material Program 的资产与渲染用例：可在 `.material` 中声明并通过
      `MaterialAssets.*` 的域校验；PT 输出与 Direct 模型可区分。
- [ ] 毛发阴影对照：给出解析自阴影与至少一种对照实现（例如 opacity map）在同一 groom
      上的差异图与误差指标，并写明采纳哪一个作为默认。
- [ ] `Fiber.motion` / `Fiber.identity` 接入至少一条时域路径，给出运动场景下的重投影
      误差与遮挡处理说明。

## 相关文档 / References

- [MaterialSystemM6Fiber.md](../../MaterialSystemM6Fiber.md)
- [MaterialSystemM7Strands.md](../../MaterialSystemM7Strands.md)
- [MaterialSystemM8RayExecution.md](../../MaterialSystemM8RayExecution.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
