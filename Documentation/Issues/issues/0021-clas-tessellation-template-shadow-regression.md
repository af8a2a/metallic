---
id: "0021"
title: "CLAS: 位移曲面模板实例化默认仍为 triangles，Shadow pass 回归在 C1–C5 全部候选下都未消除"
area: geometry
kind: perf
priority: P1
state: in-progress
milestone: M1-Active
labels: ["area:geometry", "kind:perf", "priority:P1", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/CLASTessellation.md:6
  - Documentation/CLASTessellation.md:174-180
  - Documentation/CLASTessellation.md:254-268
  - Documentation/CLASTessellation.md:296-313
  - Documentation/CLASTessellation.md:612-625
  - Source/Runtime/Render/RenderPass/BuiltinPass/SceneAccelerationStructurePass.cpp:18
  - Source/Runtime/Render/Streamer/StreamCLASTessellation.h:58
---

## 背景 / Context

流式 CLAS 位移细分（`rtTessellation`）能从已发布的流式 RT cut 构建可光追的位移微几何。模板实例化把固定拓扑的一次性构建换成「引用模板 + 提供当帧位移顶点」，在固定样本上把 CLAS 构建/实例化操作打到 -81% 量级。

## 现状 / Current state

- `rtTessellation` 默认 `false`（`SceneAccelerationStructurePass` 读取属性时默认 false），`rtTessellationBuildMode` 默认 `triangles`（`StreamCLASTessellation.h` 的默认值是 `Triangles`）。模板模式需要显式设置才会启用。
- 2026-10-25 的受控 A/B（RTX 5070 Ti / 621.01、内部固定 1444×694、32 warmup + 64 timed、residency 第 16 帧冻结、HDR/depth/shadow 逐字节一致、每轮 2048 patches / 1 个生成 BLAS / 0 拒绝）：
  - CLAS 操作与屏障 0.171 ms → 0.032 ms（-81.42%，CI 81.15–81.69）；
  - 整图 5.527 → 5.327 ms（-3.40%，CI 1.27–5.49）；
  - **Ray-traced Shadows 反而 +7.67%（CI -10.93 至 -4.52，即更慢）**。
- 后续三个候选都没消除 Shadow 回归：C1（right/none）Shadow -6.58%、C2（equilateral/fastTrace）-7.11%、C3（geometry bounds）-7.16%、C4（rightDyadic）-14.30%、C5（rightCorner）仍为 -1.24%，且整图下界 -2.1562% 越过 -2% 护栏，判定 **inconclusive**。
- 原生尺寸诊断：triangles 3,616,896 B（每 cluster 1536–2176 B），三种模板策略全部 5,242,880 B（每 cluster 恰好 2560 B），即模板活跃 CLAS 数据多 44.96%；即便 C4 把原生尺寸压到 3,824,640 B，Shadow 反而更慢。

## 问题或缺口 / Problem

1. 结论卡在「CLAS 更快、消费者更慢」：现有证据只能排除「总 CLAS 字节」这一充分条件，无法定位 BVH 内部布局、cache 事件或求交次数差异（未采集 RT 硬件计数器）。
2. 占位约束仍未定论：零 `instantiationBoundingBoxLimit` 是可疑编码约束，但 C3 的全场景有限框、C4 的 dyadic 引导都没有解决；per-patch tight bounds 只在采集帧上成立，不能作为未来帧限制。
3. 结构上限：count/emit 走查虽受容量约束，但**在实例工作组内是串行的**，大的位移实例仍然昂贵；没有混合 mini-cluster 打包、持久 patch 队列或 temporal geometry cache。

## 期望结果 / Expected outcome

- 要么给出能同时改善 CLAS 操作与 Shadow pass 的表示/编码方案（含 native BVH 层面的证据），要么明确记录模板路径不可采纳的结论并把默认值、文档与工具一并收口。

## 验收标准 / Acceptance criteria

- [ ] 用 `Tools/Perf/MeasureCLASTemplateShadow.py` 的 v5 协议跑新的候选：三块 ABCCBA 探索 + 五块独立确认，正收益阈值与 -2% Shadow/整图护栏预先声明，样本不丢弃不合并。
- [ ] 至少采集一次 Shadow dispatch 的 RT 硬件计数器（BVH node/三角形测试数、cache 事件），把「CLAS 变快但消费者变慢」定位到具体机制。
- [ ] `scene_clas_tessellation{,_templates}`、`*_c1_right_none`、`*_c2_equilateral_fast_trace`、`*_c3_geometry_bounds`、`clas_template_patterns`、`clas_indirect_triangle_build`、`StreamCLASTemplateBounds.*` 全部通过严格同步校验且零差异。
- [ ] 所有实验进程 HDR/depth/shadow 逐字节一致，patch/BLAS/拒绝计数可复核。
- [ ] 若采纳新默认：同步更新 `CLASTessellation.md` 的默认值表、runtime identity 与 shader 模块审计（`python -B Tools/CheckShaderModules.py`）。

## 相关文档 / References

- [CLASTessellation.md](../../CLASTessellation.md)
- [GPUDrivenTessellation.md](../../GPUDrivenTessellation.md)
- [MeasureCLASTemplateShadow](../../../Tools/Perf/MeasureCLASTemplateShadow.md)
