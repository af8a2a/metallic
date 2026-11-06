---
id: "0061"
title: "ShaderWarmup: 冷缓存全目录预热约 56 s，未达到 10 s 目标"
area: build
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:build", "kind:perf", "priority:P2"]
evidence:
  - Documentation/ShaderWarmup.md:338-348
  - Documentation/ShaderWarmup.md:349-356
  - Documentation/ShaderWarmup.md:280-284
---

## 背景 / Context

`MetallicShaderWarmup` / `--full-shader-warmup` 的目标是让首个可呈现帧不再触发
按需编译。为此实现的并发源码批处理、缓存事务、native PSO 阶段与进度统计都已经
落地，并有冷缓存测量流程。

## 现状 / Current state

文档记录的三次独立冷缓存进程（8 worker、清空 SPIR-V 缓存、174 个请求、
O1 / 关闭 shader debug、真实 stride 32/32）：

- 完整进程墙钟 **57.022 / 55.395 / 55.866 s**，中位数 **55.866 s**；
- 每次零命中、零失败，522 个最终产物与原始 catalog oracle 逐字节一致；
- 文档明确写：「This does not meet the ten-second target or establish a production
  speedup over earlier measurements taken in different runs.」
- 另一次带持久缓存的正确性运行中，native 阶段最大单次创建 67.261 s、求和
  697.797 s、完整 native 阶段 727.279 s；文档强调这是「把工作搬进启动」，不是
  端到端加速。

也就是说：**正确性已经成立，性能目标 10 s 差了 5 倍以上，且没有归因数据**。

## 问题或缺口 / Problem

1. 56 s 的启动成本对「冷机首次启动编辑器」是用户可感知的严重退化，因此该功能只能
   默认关闭（`--full-shader-warmup` opt-in、`MetallicShaderWarmup` 为
   `EXCLUDE_FROM_ALL`），能力实际上没被用起来。
2. 缺少阶段归因：现有统计只有 setup/module/code-generation 与文件发布区间，
   「77 个 session group」「Slang 前端 vs 后端」「native PSO 创建 vs 源码编译」
   各自占用多少墙钟没有可核对的分解。
3. `METALLIC_SHADER_WARMUP_JOBS` 的默认上限 12 是否合适，文档要求「先比较冷缓存
   耗时与峰值内存」，但这项比较还没做；峰值已到 6.20–6.21 GiB working set /
   7.03–7.08 GiB commit。

## 期望结果 / Expected outcome

- 给出可复核的阶段分解（SLANG 前端 / IR 降级 / SPIR-V 优化 / 缓存发布 / native
  PSO），并在此基础上至少把冷缓存全目录预热压到接近 10 s 目标，或明确记录
  「该目标在现有 Slang 版本下不可达」的实测依据与替代策略（增量预热、按样例子集、
  预置缓存分发）。

## 验收标准 / Acceptance criteria

- [ ] 冷缓存测量脚本可重复产出分阶段耗时与峰值内存（现有工具已带
      `--threshold-seconds` 默认 10，即目标已编码在工具里）：
      `python -B Tools/Perf/MeasureShaderWarmup.py --executable build-release/Source/MetallicShaderCompiler.exe --output-dir .tmp\warmup --jobs 8 --repeats 3 --threshold-seconds 10`
      以及 `python -B tests/perf/TestMeasureShaderWarmup.py` 通过。
- [ ] 报告包含每阶段（setup / module / codegen / publish / native）的墙钟与占比，
      并标注并发求和与墙钟的区别。
- [ ] 至少完成一次 B/A 对照：`METALLIC_SHADER_WARMUP_JOBS` 取 4 / 8 / 12，
      同请求集与同缓存条件，记录墙钟、峰值 working set、commit，给出默认值建议。
- [ ] 结论写入 `Documentation/ShaderWarmup.md`「Complete cold-cache measurements」
      一节：达标则以新数字替换，未达标则写清瓶颈所属阶段与下一步。
- [ ] `cmake --build build-release --target MetallicShaderWarmup` 后
      `MetallicShaderWarmupEvidenceTests` 通过（`ctest -R '^MetallicShaderWarmupEvidenceTests$'`）。

## 相关文档 / References

- [ShaderWarmup.md](../../ShaderWarmup.md)
- [ShaderPipelineBench.md](../../ShaderPipelineBench.md)
- [ShaderCompilationComplexity.md](../../ShaderCompilationComplexity.md)
