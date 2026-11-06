---
id: "0062"
title: "ShaderWarmup: graphics PSO 与 ImGui ABI 未预热，bounded 请求集外仍首帧按需编译"
area: build
kind: feature
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:build", "kind:feature", "priority:P2"]
evidence:
  - Documentation/ShaderWarmup.md:113-121
  - Documentation/ShaderWarmup.md:244-249
  - Documentation/ShaderWarmup.md:269-274
  - Documentation/ShaderWarmup.md:1582-1584
---

## 背景 / Context

启动预热分两相：源码/SPIR-V 批处理（`MetallicShaderWarmupCore`，无需 GPU）与
native PSO 阶段（在真实设备上创建 compute PSO）。目标是「首帧不再触发按需编译」。

## 现状 / Current state

- native 阶段只会枚举 **compute** 排列：文档写得很直白——「Compute permutations can
  be enumerated at startup; graphics PSOs require their own render-state collectors」，
  且当前实现「waits synchronously for the compute phase before making the rendering
  device available」。
- graphics shader stage 只是**报告**，不预创建："Graphics shader stages are reported
  separately and are not given guessed PSO state. Render target/depth formats, stage
  combinations and external ImGui ABIs remain owned by their rendering paths."
- 实测一次 `pathtracing-sample` 设备相：109 个 compute PSO 创建并保留、1 个重复请求
  复用，**skip 了 35 个 graphics stage**；未支持/未知 ABI 计数为 0。
- 请求集本身是 bounded："This is a bounded warmup set, not every combination of
  runtime properties. Optional NRD SDK permutations, local-view deferred variants,
  and other unlisted combinations continue to compile on demand."

## 问题或缺口 / Problem

1. 用户可感知的首帧卡顿没有被这批工作覆盖：光栅 pass（VSM、FinalBlit、
   MaterialResolve 类输出、raster comparison）与 ImGui 显示管线的 graphics PSO
   仍按需创建，`--full-shader-warmup` 完成后仍然会掉帧。
2. ImGui 的 vertex/fragment 组合被明确排除在预热之外，而它是编辑器里最先被使用的
   图形管线之一；其 ABI 由编辑器适配器拥有，任何契约改动都会让预热结果失效。
3. 未列变体（NRD 排列、局部视图 deferred）没有覆盖统计：无法回答「这次预热到底
   覆盖了多少实际会在首帧被请求的 PSO」。

## 期望结果 / Expected outcome

- graphics PSO 预热与 compute 一样进入可枚举、可验证的路径：每个 raster pass 提供
  render-state 收集器（颜色/深度格式、混合、dynamic state、stage 组合），
  ImGui 适配器提供自己的 ABI 版本化收集器。
- 预热输出报告覆盖率：catalog 覆盖的 PSO 数 / 首帧实际请求的 PSO 数，以及
  覆盖率未达标时的显式 warning。

## 验收标准 / Acceptance criteria

- [ ] `--full-shader-warmup` 在一次编辑器实时启动后，首帧的 graphics PSO 创建调用
      为 0（对照 `--skip-shader-warmup` 的基线计数），可通过
      `.\build-release\Source\Metallic.exe --full-shader-warmup --smoke-test` 的日志断言。
- [ ] 预热报告新增字段：`graphicsPsoCreated`、`graphicsStagesSkipped`、
      `uncoveredFirstFrameRequests`；非零 uncovered 时打印可复制的 catalog 缺失项。
- [ ] 新增/扩展 `MetallicNativePipelineCacheTests` 或 `NativeShaderWarmupTests` 用例，
      覆盖 graphics PSO 的 single-flight、格式变体与 ABI 版本不匹配时拒绝复用：
      `ctest --test-dir build-ci -R '^MetallicNativePipelineCacheTests$' --output-on-failure`
- [ ] `Documentation/ShaderWarmup.md`「Native pipeline startup phase」与「Coverage and
      maintenance」两节更新为 graphics 已覆盖的范围与仍未覆盖的组合。

## 相关文档 / References

- [ShaderWarmup.md](../../ShaderWarmup.md)
- [ShaderRegistry.md](../../ShaderRegistry.md)
- [Build.md](../../Build.md)
