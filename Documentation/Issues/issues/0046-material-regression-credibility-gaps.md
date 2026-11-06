---
id: "0046"
title: "Material: 材质回归可信度缺口——长期失败用例、未定位的整组 HDR 漂移与 NTC alpha 跨路径不一致"
area: material
kind: defect
priority: P2
state: open
milestone: M2-Correctness
labels: ["area:material", "kind:defect", "priority:P2"]
depends_on: []
evidence:
  - Documentation/MaterialCoverageNativeSlang.md:169-171
  - Documentation/MaterialSystemM4SceneClosure.md:136-139
  - Documentation/MaterialSystemPhase8.md:88-91
  - Documentation/MaterialSystemPhase8.md:114-118
  - Documentation/MaterialSystemPhase9.md:124-128
  - tests/rhi/MiniZorahStreamAssetTests.cpp:965
  - tests/rhi/ShaderHotReloadTests.cpp:104
---

## 背景 / Context

材质系统的每个 Phase 都用「N 项回归全绿」作为验收结论。但如果同一批回归里存在已知
失败项、无法定位的数值漂移，或某条能力被明确排除在验收之外，那么「全绿」这个结论
就不能作为后续判断的基础。

## 现状 / Current state

1. **已知失败用例被保留且未计为通过。**
   - `stream_metadata_contract` 的 live-color-edit 断言「with both the original and
     fixed binaries still fails」，文档明确说它「is preserved unchanged and is not
     counted as passing coverage here」
     （`MaterialCoverageNativeSlang.md:169-171`；测试注册于
     `tests/rhi/MiniZorahStreamAssetTests.cpp:965`）。
   - `slang_shader_background_hot_reload` 报 `deleted shader dependency was not
     reported`；独立复测 5 次通过，但综合组仍记为失败，「现有日志不足以区分根因」
     （`MaterialSystemM4SceneClosure.md:136-139`；测试注册于
     `tests/rhi/ShaderHotReloadTests.cpp:104`）。
2. **整组上下文下的 HDR 漂移原因未定位。**
   - Phase 8：完整套件上下文的 PathTrace 与冻结基线 RGB RMSE `0.0002640552`、
     最大绝对误差 `0.04733187`；独立进程重跑后仍非逐位一致（RMSE
     `0.000019844`），「差异原因尚未定位，不能归因于随机噪声」
     （`MaterialSystemPhase8.md:114-118`）。
   - Phase 9 复现了同一现象：「完整套件上下文差异的原因仍未定位；独立结果不用于
     掩盖该差异」（`MaterialSystemPhase9.md:124-128`）。
3. **能力被明确排除在验证之外。** 自定义 Value 程序路径下，「NTC alpha 仍沿用原有 RT
   神经纹理解码路径，其与 VBuffer 普通 alpha 的跨路径一致性未建立，不作为本阶段已支持
   的 alpha authoring 路径」（`MaterialSystemPhase8.md:88-91`）。

## 问题或缺口 / Problem

- **失败用例没有 owner。** 两个失败断言分别涉及流式材质元数据与后台 shader 热重载，
  都不是材质数学问题，但都直接影响材质编辑的正确性，且都已存在多个 commit 未被处理。
- **数值漂移破坏「逐位一致」这一验收语言。** 材质基线反复使用「与冻结 Phase 0 逐位
  一致」作为最强证据；一旦同进程多用例序列会产生非零差异，这条证据就只在单用例隔离
  上下文成立，而文档没有把上下文条件写成必填项。
- **alpha 覆盖的两条解码路径（普通纹理 vs NTC）可能给出不同可见性。** Coverage 是
  「命中是否存在」的判定，跨路径不一致会同时污染 VBuffer 与 RT 的 winner。

## 期望结果 / Expected outcome

- 两个失败用例要么修复，要么从回归集合中移除并记录替代覆盖，不再以「已知失败」长期
  共存于验收组。
- 材质基线的每个「逐位一致」结论都显式记录执行上下文（单用例 / 完整组、进程数、
  环境就绪帧），并查明整组漂移的根因。
- NTC alpha 与普通纹理 alpha 的跨路径一致性有明确的判定（一致 / 明确不支持并禁用）。

## 验收标准 / Acceptance criteria

- [ ] `stream_metadata_contract` 连续 3 次运行通过，或该断言被移除并新增等价覆盖；
      两种结果都需在文档中写明。
- [ ] `slang_shader_background_hot_reload` 连续 3 次（含完整回归组上下文）通过；
      失败时日志能区分「文件删除失败」与「通知等待失败」两种根因。
- [ ] 用同一可执行文件连续运行 `*material_asset_phase0_equivalence*` 两次，并给出
      同进程多用例序列与单用例隔离两种上下文的 HDR 对比，解释差异来源（例如随机种子
      推进、环境解码完成帧、sampleFrame 计数）。
- [ ] 若根因属于基线抖动而非渲染缺陷，在 `MaterialBaselineTests` 中记录上下文，
      并把「逐位一致」的适用范围写进 `MaterialSystemM1M4Acceptance.md`。
- [ ] NTC alpha 跨路径：要么建立一致（同一材质在 VBuffer 与 RT 下 winner 逐像素一致），
      要么在材质侧明确拒绝 NTC alpha 并给出诊断。

## 相关文档 / References

- [MaterialCoverageNativeSlang.md](../../MaterialCoverageNativeSlang.md)
- [MaterialSystemPhase8.md](../../MaterialSystemPhase8.md)
- [MaterialSystemPhase9.md](../../MaterialSystemPhase9.md)
- [MaterialSystemM4SceneClosure.md](../../MaterialSystemM4SceneClosure.md)
- [MaterialSystemM1M4Acceptance.md](../../MaterialSystemM1M4Acceptance.md)
