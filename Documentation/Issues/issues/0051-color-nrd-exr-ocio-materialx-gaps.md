---
id: "0051"
title: "Color: ACEScg 迁移后仍缺原生宽色域降噪、EXR/OCIO 输入与完整 MaterialX 色彩管理"
area: material
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:material", "kind:feature", "priority:P2"]
depends_on: []
evidence:
  - Documentation/ColorPipeline.md:110-118
  - Documentation/ColorPipeline.md:65-66
  - Documentation/ColorPipeline.md:90-91
  - Shaders/Interop/NRDEncoding.slang:19-22
  - Documentation/ACEScgMigrationValidation.md:113-117
  - Documentation/ACEScgMigrationFixes.md:93-95
---

## 背景 / Context

渲染器工作空间已切换为 scene-linear ACEScg（AP1 / ACES white），资产侧保留 authoring
linear Rec.709，转换发生在上传边界。迁移的验证范围是 RGB 变换、纹理语义、曝光与显示
链路；资产生产侧（EXR、OCIO、MaterialX）与 SDK 侧的宽色域支持是另外的问题。

## 现状 / Current state

1. **NRD 只有 Rec.709 适配器，会裁掉 AP1 的负分量。** 前端把工作 RGB 转成
   linear Rec.709 再交给 vendor 打包：
   `RELAX_FrontEnd_PackRadianceAndHitDist(WorkingColor::toLinearRec709(radiance), ...)`
   （`Shaders/Interop/NRDEncoding.slang:19-22`）。文档确认「NRD's sanitization can clip
   negative Rec.709 components of AP1 colors outside the Rec.709 gamut; full native AP1
   denoiser specialization needs separate validation」
   （`ColorPipeline.md:116-118`）。
2. **没有 EXR 摄入。** 引擎现有 stb 解码器支持 HDR 与既有格式，「EXR ingestion is not
   added here」（`ColorPipeline.md:90-91`）；`Source/` 下也没有任何 `.exr` 处理路径。
   White Studio / Painter 资产都是先用 Python 转成 RGBE HDR 或 PNG 再导入
   （`WhiteStudioLookDev.md:36-40`）。
3. **离线条纹与 OCIO 图被推迟。** 「Runtime conversion is an interim cost; offline
   texture cooking, compression changes and a full OCIO graph are deferred」
   （`ColorPipeline.md:66`）。
4. **USD / MaterialX 色彩管理只覆盖 Preview Surface 的标签。** 迁移与修复两轮都声明：
   「does not establish complete USD/MaterialX color-management support」、
   「Painter/OCIO authoring and full MaterialX color management remain unvalidated」
   （`ACEScgMigrationFixes.md:75-78`、`:93-95`；`ACEScgMigrationValidation.md:113-117`）。
5. **物理 HDR 标定未验证。** 「Physical HDR monitor calibration, display-profile
   switching and extended interactive viewing were not validated」
   （`ACEScgMigrationValidation.md:108-112`）。

## 问题或缺口 / Problem

- **降噪路径与渲染工作空间不一致。** 这是最实质的正确性缺口：ACEScg 下合法的 AP1 颜色
  （Rec.709 坐标为负）经过 NRD 会被裁切，导致饱和材质在高能区域出现色偏或能量损失。
  当前只能靠「测试场景在 gamut 内」规避。
- **资产入口依赖外部预转换。** 缺少 EXR 摄入意味着所有参考图（HDRI、色卡、渲染对照）
  必须离线转码，转码本身引入量化与色彩假设（例如 HDRI 缺少 chromaticities 时按
  Rec.709/D65 假定），这些假设无法在引擎侧验证。
- **材质资产侧色彩管理不完整。** MaterialX/USD 的颜色空间图（而非单个纹理标签）会
  产生与引擎不同的调制顺序，当前迁移只保证「标签 + authoring Rec.709 调制」这一层。

## 期望结果 / Expected outcome

- NRD（至少 RELAX/Reblur 前端）在 ACEScg 模式下不再裁切 AP1 负分量，或明确声明
  降噪前的色域变换与可接受损失，并在 saturated / HDR 场景给出误差指标。
- 引擎可直接摄入 EXR（保留 chromaticities 与浮点精度），参考图不再依赖 Python 转码。
- OCIO / MaterialX 色彩管理有一个明确的范围声明：支持哪些配置、哪些节点、不支持时
  如何诊断。

## 验收标准 / Acceptance criteria

- [ ] 新增 GPU 回归：在 `METALLIC_WORKING_COLOR_SPACE=acescg` 下用超色域材质
      （AP1 坐标含负 Rec.709 分量）渲染 NRD 开关两路，报告是否出现裁切以及
      色度/能量差异；测试需在 Rec.709 模式下同样运行以作对照。
- [ ] `NRDEncoding.slang` 或其后端适配层给出 AP1 直通与 Rec.709 适配的显式选择，
      并在文档记录该选择对 vendor 约束的影响。
- [ ] 引擎支持直接加载 EXR 环境与纹理（至少浮点 + chromaticities），并有单元测试
      证明读入值与源文件逐像素一致；现有 RGBE 路径保留为兼容。
- [ ] USD/MaterialX：给出「已支持 / 明确不支持」矩阵，并对不支持的输入返回可定位诊断。
- [ ] `ColorPipeline.md` 的「外部边界」章节更新，删除已被本 issue 关闭的条目。

## 相关文档 / References

- [ColorPipeline.md](../../ColorPipeline.md)
- [ACEScgMigrationValidation.md](../../ACEScgMigrationValidation.md)
- [ACEScgMigrationFixes.md](../../ACEScgMigrationFixes.md)
- [WhiteStudioLookDev.md](../../WhiteStudioLookDev.md)
- [PainterLookDevValidation.md](../../PainterLookDevValidation.md)
