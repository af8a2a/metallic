---
id: "0060"
title: "构建与验证: 无 CI 工作流，metallic-ci 只跑 scene/task/debug，关键缓存用例仅 Windows"
area: build
kind: task
priority: P1
state: open
milestone: M3-Hardening
labels: ["area:build", "kind:task", "priority:P1"]
evidence:
  - CMakePresets.json:82
  - CMakePresets.json:86
  - tests/CMakeLists.txt:108-109
  - tests/CMakeLists.txt:139
  - tests/editor/ShaderWarmupBatchTests.cpp:653-654
  - tests/editor/CompilerProducerSealTests.cpp:103
  - tests/editor/SlangSourcePreparationTests.cpp:74
---

## 背景 / Context

仓库有完整的 CTest 注册表（scene/task/debug/rhi/editor/perf 六类标签、上百个用例）
和多个 configure/build presets，但「谁来定期跑它们」这件事没有落到仓库里：
根目录没有 `.github/`，也没有任何 CI 配置文件。

## 现状 / Current state

- `metallic-ci` 的 build preset 只构建
  `MetallicSceneTests / MetallicTaskTests / MetallicDebugTests`；对应的 test preset
  用 `filter.include.label = "^(scene|task|debug|build)$"` 限定运行范围。
  也就是说 **editor、rhi、tools、perf 四类标签不在 CI 形状的入口内**，而
  `Documentation/ShaderWarmup.md`、`RenderGraph*`、`DebugControlPlane` 的验证记录
  几乎全部依赖这些标签。
- Nsight 集成回归默认关闭：`option(METALLIC_NSIGHT_INTEGRATION_TESTS ... OFF)`，
  且还要求 `WIN32 AND METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE`，因此 GPU Trace
  启动/采集/导出/无效指标四个用例默认既不可用也不被执行。
- 编辑器/着色器缓存语义基本是 Windows-only：`ShaderWarmupBatchTests.cpp` 有 11 处
  `#ifndef _WIN32` 守卫（第 192、241、399、519、653、713、784、833、861、883、909 行），
  其中绝大多数直接 `GTEST_SKIP() << "Windows snapshot proof fixture"`；
  `CompilerProducerSealTests.cpp:103` 与 `SlangSourcePreparationTests.cpp:74` 同样
  以「需要 Windows 目录通知 / 共享语义」跳过。`tests/CMakeLists.txt:139` 起的大批
  编辑器 smoke 也整段包在 `if(WIN32)` 里。
- 结论：源码级缓存 epoch、producer seal、路径/符号链接语义这些**最容易出正确性
  问题**的部分，在非 Windows 平台上没有等价验证；而在 Windows 上也没有自动执行。

## 问题或缺口 / Problem

1. 「本地跑一次 ctest」是当前唯一的验证方式，PR 之间的回归靠人自觉；`metallic-full`
   又不构建全部注册目标，容易漏测。
2. 因为没有 CI，`metallic-ci` 的标签范围从未被挑战，perf/tooling 侧的 Python 证据
   用例（`tests/perf/*.py`，20+ 个）实际上处于「注册了但没人跑」的状态。
3. 平台矩阵缺失使 Windows-only 假设（完成端口目录通知、文件共享语义、reparse
   point）无法被发现和记录，跨平台移植时会集中爆发。

## 期望结果 / Expected outcome

- 仓库内落地至少一条自动化 lane：构建 + 运行 `scene|task|debug|build|tools` 标签，
  并在 GPU 可用时追加 `rhi|editor` lane（允许 skip 但必须记录）。
- 提供一份明确的验证矩阵文档：哪些标签在哪个平台/能力下必须通过，哪些是
  known-skip（与 0059 的基线共享同一份清单）。
- 对 Windows-only 的缓存/密封语义，给出非 Windows 的可运行替身（例如用注入式
  目录变更源替代完成端口）或明确标注「不支持并拒绝启用持久缓存」。

## 验收标准 / Acceptance criteria

- [ ] 仓库出现 CI 配置（如 `.github/workflows/verify.yml`），至少包含
      `cmake --preset metallic-ci` → `cmake --build --preset metallic-ci` →
      `ctest --test-dir build-ci --output-on-failure` 三步，并归档 JUnit 报告。
- [ ] `metallic-ci` 的 test preset 标签范围扩展到 `^(scene|task|debug|build|tools)$`，
      使 `tests/perf/*.py` 证据用例进入自动化；如某用例需要 GPU 则显式标注并允许 skip。
- [ ] Windows-only 用例清单化：`grep -rn "GTEST_SKIP" tests/editor/` 的结果逐条
      给出「保留原因 / 替身计划 / 平台限制」结论，写入验证矩阵文档。
- [ ] Nsight 集成用例在装有 Nsight Graphics 的机器上可通过
      `-DMETALLIC_NSIGHT_INTEGRATION_TESTS=ON` 跑通，并在文档中记录所需的
      外部进程与超时（`MetallicNsightGPUTrace.collection` 已配置 620 s 超时）。
- [ ] `python -B Tools/CheckShaderModules.py` 与
      `python -B tests/perf/TestShaderPipelineBench.py --executable <bench>` 在 lane 中执行。

## 相关文档 / References

- [Build.md](../../Build.md)
- [ProjectArchitecture.md](../../ProjectArchitecture.md)
- [TracyGpuProfiling.md](../../TracyGpuProfiling.md)
