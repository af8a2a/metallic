---
id: "0040"
title: "DLSS-RR: PSR 与 path regularization 未实现，guide 输入与参考样本不对齐"
area: rhi
kind: research
priority: P3
state: proposed
milestone: M5-Research
labels: ["area:rhi", "kind:research", "priority:P3"]
evidence:
  - Documentation/PsrAssessment.md:7-56
  - Documentation/PsrAssessment.md:89-103
  - Documentation/DlssRayReconstruction.md:49-64
---

## 背景 / Context

DLSS-RR 的时域输入（相机、motion、jitter、天空 motion）已与参考样本
（`E:/vk_denoise_dlssrr` @ `f8113a84`）对齐，并有
`RHIRendering.dlss_motion_vector_reprojection` 与 `MetallicEditorDLSSCameraSmoke` 覆盖。
但参考实现默认开启的两项引导机制 Metallic 没有实现：Primary Surface Replacement（PSR）
与 path regularization（沿路径传播最大粗糙度以降低间接高光方差）。参考还提供 tone-mapped
环境色作为天空漫反射 albedo guide 与零天空法线，Metallic 使用默认天空 guide 值。

## 现状 / Current state

- PSR 评估结论是「推迟」：参考镜像门槛为 metallic==1 且平方粗糙度 ≤
  `MICROFACET_MIN_ROUGHNESS² + 0.001`（巡游粗糙度约 0.03165，最多 5 次命中）。
  对 ABeautifulGame 的 15 个材质做源贴图审计（读 ORM 的 G/B 并按 glTF factor 还原、不作 sRGB
  转换）：满足该门槛的源 texel 数为 **0**；棋盘粗糙度 min/median/max = 0.0549/0.2706/0.4784；
  放宽到 roughness ≤ 0.05 且 metallic ≥ 0.98 也只在两个 bishop 材质里各约 0.000119%。
- 该审计是源贴图统计，不是 GPU 屏幕覆盖率测量；未覆盖未保存的运行时材质覆盖、纹理过滤与
  运行时压缩。
- 实现成本位置明确：改动属于 path tracing 与 guide 生成，不是 DLSS 选项。正确实现必须让
  radiance 路径与替换 guide 表面共享同一镜面前缀（throughput、emission、path length、
  composed mirror transform 一起给出），并定义材质边界、最大路径长度、天空出口与相机切换时的回退。
- 目前没有任何 PSR / regularization / 天空 albedo guide 的代码、开关或测量；也没有 GPU 时间
  或图像质量改善的数据。

## 问题或缺口 / Problem

这是一个**已知未开工**的能力缺口，不是回归：RR 的 guide 集合与参考实现不一致，镜面与
rough reflection 的降噪质量无法与参考样本对等比较。风险在于把「照参考实现抄一遍」当成目标——
在当前 authored 场景里忠实移植几乎不会激活。需要先有 mirror 内容的测量，再决定是否投入。

## 期望结果 / Expected outcome

先做评估文档建议的下一步测量：在可重复的相机轨迹上检查 reflected hit distance、反射几何 motion
与天空 guide（以及可选的 specular motion vectors）；若加入镜面内容，再实现仅覆盖 1–2 个不透明
近完美金属交互的可切换 PSR 原型，并在相同条件下 A/B。path regularization 与天空 albedo guide
单独决策，不与 PSR 捆绑。

## 验收标准 / Acceptance criteria

- [ ] 记录一次 mirror 覆盖率测量（屏幕空间，而非源贴图统计），说明当前场景在参考门槛下的
      实际激活比例；不使用该测量不得给出 PSR 收益结论。
- [ ] PSR 原型仅覆盖不透明近完美金属（1–2 个交互），有独立开关，radiance 与 guide 复用同一
      镜面前缀；提供 fallback 定义（材质边界 / 路径长度 / 天空出口 / 相机切换）。
- [ ] A/B 使用相同相机路径、分辨率、sample 数、曝光与 RR preset，并报告镜边稳定性、
      收敛参考下的时域误差、disocclusion 恢复与 GPU 时间。
- [ ] `RHIRendering.dlss_motion_vector_reprojection`、`MetallicEditorDLSSCameraSmoke` 与
      `render_graph_pathtracing_guides_shader_compile` 在无 PSR 与有 PSR 两种配置下均通过。
- [ ] path regularization 与天空 albedo guide 各自形成决策记录（做或不做，附依据）。

## 相关文档 / References

- [PsrAssessment.md](../../PsrAssessment.md)
- [DlssRayReconstruction.md](../../DlssRayReconstruction.md)
- [NRDColorContract.md](../../NRDColorContract.md)
- [RTXDI.md](../../RTXDI.md)
