---
id: "0067"
title: "Geometry: meshlet 自适应位置编码与关节切线帧停在未合并分支，测试已就绪但未进主线"
area: geometry
kind: feature
priority: P2
state: blocked
milestone: M3-Hardening
labels: ["area:geometry", "kind:feature", "priority:P2", "state:blocked"]
evidence:
  - "git:modernization_/meshlet-compression 8d8bbe4bb Add adaptive position encoding with boundary and topology safeguards"
  - Source/Runtime/Scene
  - tests/CMakeLists.txt
---

## 背景 / Context

`modernization_/meshlet-compression` 分支相对 `modernization` 是
**+5869 / −195 行、48 个文件**，包含三块互相独立但被一起挂起的工作：

| 提交 | 内容 |
| --- | --- |
| `8d8bbe4bb` | 自适应（自适应 bit 宽度）位置编码，带边界与拓扑保护 |
| `349ab5641` | 关节切线帧流送与诊断 |
| `e88bd04c9` | 回退 BLAS 构建工作区内存的跟踪与构建后释放 |

在 HEAD 上 `git grep -i AdaptivePosition -- Source Shaders` 无结果，
说明这套编码**没有进入主线**，也没有对应开关。

## 现状 / Current state

分支自带相当完整的测试与工具，这是它与"纯原型分支"的关键区别：

- `tests/scene/PositionEncodingTests.cpp`（+228）、`tests/rhi/PositionEncodingTests.cpp`（+156）
- `tests/scene/AdaptivePositionFixtureTests.cpp`（+136）、`tests/scene/GeometryAttributesTests.cpp`（+214）
- `tests/rhi/StreamMaterialTests.cpp`（+89）、`tests/rhi/shaders/PositionEncodingProbe.slang`
- `Tools/ValidatePositionEncoding.py`（+170）、`Tools/ExportAdaptivePositionFixture.py`（+74）
- `tests/CMakeLists.txt` 已登记新测试目标

## 问题或缺口 / Problem

1. **压缩能力缺失**：主线的几何属性带宽优化缺少位置量化这一环，流送与
   顶点解码的显存/带宽收益无法兑现。
2. **工作处于三提交混合状态**：位置编码、关节切线帧、BLAS 工作区内存在同一
   分支上，难以单独评估哪一块已经可用、哪一块还有正确性风险。
3. **测试漂移**：分支改动了 `tests/CMakeLists.txt` 与多个既有测试文件
   （`StreamerTests.cpp`、`FrameContextTests.cpp`、`MemoryBudgetTests.cpp`、
   `ZorahFullFirstFrameTests.cpp`），与主线这一个月的改动已可能冲突。
4. **BLAS 工作区内存释放**是独立且有价值的修复（构建后未释放工作区会造成
   峰值显存抬高），被一起挂起会持续影响大场景显存。

## 期望结果 / Expected outcome

1. 拆解：把 **BLAS 工作区内存释放** 单独摘出，作为独立小改动进主线（与压缩能力无关，风险低）。
2. 位置编码 + 关节切线帧合并为一个可评审的特性分支，补一份
   `Documentation/MeshletPositionEncoding*.md` 记录编码格式、误差界与验证数据。
3. 用分支自带的 `Tools/ValidatePositionEncoding.py` 给出量化误差与重建质量的实测结论。

## 验收标准 / Acceptance criteria

- [ ] BLAS 工作区内存在构建完成后释放，`MetallicRHITests` 内存预算用例通过并记录峰值下降
- [ ] 位置编码进入主线后 `ctest -R 'PositionEncoding|GeometryAttributes|AdaptivePositionFixture'` 全绿
- [ ] `Tools/ValidatePositionEncoding.py` 在至少一个真实场景资产上产出误差报告
- [ ] 关节切线帧流送有对应测试与文档，或有明确的"不做"决策记录
- [ ] 分支 `modernization_/meshlet-compression` 最终为空或已删除

## 相关文档 / References

- 分支提交：`8d8bbe4bb`、`349ab5641`、`e88bd04c9`
- [MeshletLodDesign.md](../../MeshletLodDesign.md)、[ResidentMeshletLod.md](../../ResidentMeshletLod.md)
- [StreamingMeshletLod.md](../../StreamingMeshletLod.md)、[StreamRasterCost.md](../../StreamRasterCost.md)
- [GPUBLASAllocation.md](../../GPUBLASAllocation.md)（BLAS 工作区内存的既有约束）
