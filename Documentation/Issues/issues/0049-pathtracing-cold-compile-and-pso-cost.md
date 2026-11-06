---
id: "0049"
title: "PathTracing: 冷启动仍以单个 Guides PSO 50 s 量级为主成本，模块 IR 复用与 provider 请求扩展未落地"
area: material
kind: perf
priority: P1
state: in-progress
milestone: M4-Performance
labels: ["area:material", "kind:perf", "priority:P1", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/PathTracingShaderEncapsulation.md:93-110
  - Documentation/PathTracingShaderEncapsulation.md:176-182
  - Documentation/PathTracingLinkTimeSpecialization.md:402-433
  - Documentation/OpenPBRPrimaryGuidesSplit.md:64-75
  - Documentation/ShaderDeadBranchCleanup.md:1-14
---

## 背景 / Context

路径追踪的生产入口 `OpenPBRRayQueryPathTraceGuides` 在 O1 下生成约 6.9 MB、44.7 万条
指令、单函数 44.6 万条指令的 SPIR-V（`PathTracingShaderEncapsulation.md:126-141`）。
冷启动成本因此主要集中在两个地方：Slang 前端/下游优化，以及驱动侧 native PSO 创建。
仓库里为此做了至少五轮独立实验（封装、link-time specialization、material backend Type、
SPIR-V dead-branch cleanup、primary-guides split），但都没有进入生产默认。

## 现状 / Current state

- **前端重编译无法跨请求复用。** `SlangCompiler.cpp` 为每个未命中缓存的请求新建
  global/session，因为 session 会缓存导入模块；因此「把 include 改成 import」并不会带来
  跨请求 IR 复用（`PathTracingShaderEncapsulation.md:93-97`）。封装实验的结论是：Common
  依赖裁剪与原 SPIR-V 逐字节相同，module facade 只减少了 `OpName` 数量，配对
  `compileInner` 中位数约低 1%，不构成稳定收益（同文件 `:126-163`）。
- **编译器阶段耗时量级。** 单次 Guides 请求的 compiler wall 20.656 s，其中
  module load 716.876 ms、code generation 19.752 s、SDK downstream 18.842 s
  （`PathTracingShaderEncapsulation.md:176-182`）。下游优化约占 18 s，是关键瓶颈。
- **驱动侧 native 创建仍是最大的单项。** Primary-guides split 实验里 fused create
  三次为 59.566 / 51.127 / 47.542 s，中位数 51.127 s；拆成 primary（约 2.1 s）+
  continuation（约 44 s）后最长创建下降 14.0%，但 continuation 仍主导
  （`OpenPBRPrimaryGuidesSplit.md:64-75`）。
- **dead-branch cleanup 有效但默认关闭。** `dead-branches` 路径在单一 Guides 请求上给出
  7.72% 的配对中位数下降（含额外编译开销），生产默认仍是 `none`，且只支持 mapped
  descriptors（`ShaderDeadBranchCleanup.md:1-14`、`:59-63`）。
- **稳定模块缓存的前置条件已被明确列出但未实现。** 需要 provider 列表进入
  `ShaderRequest`/`SlangShaderDesc` 与缓存键、内存 provider 需要内容哈希、需要收集
  program 与 provider 双方传递依赖、`isBinaryModuleUpToDate`  freshness 检查、
  worker-owned session（`PathTracingLinkTimeSpecialization.md:402-433`）。

## 问题或缺口 / Problem

1. **没有一条改动被采纳。** 五轮实验分别证明「封装不减少可执行体」「link-time 选择不
   提供更强 DCE」「split 降低最长创建但增加运行成本与显存」——结论都正确，但项目仍缺少
   一个能降低冷启动的可交付改动。
2. **缺少「前端缓存」与「驱动 PSO」的分离账本。** 现有测量混在一起：compiler wall、
   native create、application startup 分开记录，但没有一份把「源码未变 → 命中哪一层缓存 →
   省掉多少」写成可执行规则的文档。
3. **`dead-branches` 已是可用收益却因适用范围（仅 mapped descriptors）与默认值停在实验
   状态。** 这是一个低风险、已测量的候选，缺的只是采纳决策与 native descriptor-heap
   路径的适配。

## 期望结果 / Expected outcome

- 冷启动有分层账本：前端编译、SPIR-V 后处理、native PSO 创建、应用启动四项分别可测，
  且每层都有「命中时省掉多少」的实测数字。
- 至少一项落地为默认（建议：`dead-branches` 在 mapped descriptors 下的默认化，或
  独立编译的 split kernel 在特定 workload 下的显式开关）。
- 若要做稳定模块 IR 缓存，必须先实现 provider 列表与缓存键扩展，再测 preparation
  与 relink 成本，不得只报「链接更快」。

## 验收标准 / Acceptance criteria

- [ ] `MetallicShaderPipelineBench` 对同一个 Guides 请求输出四项分层耗时
      （compiler wall / codegen / downstream / native create），三进程独立运行，
      中位数与区间可复算。
- [ ] 明确决定 `METALLIC_SPIRV_POSTPROCESS` 的默认值，并在
      `ShaderDeadBranchCleanup.md` 记录决策与适用条件；若采纳，需补 native
      descriptor-heap 路径的支持或明确拒绝诊断。
- [ ] 若实现模块 IR 缓存：`ShaderRequest`/`SlangShaderDesc` 至少包含 provider 标识与
      内容哈希，缓存键覆盖 SDK 版本、target、宏、descriptor ABI；并新增用例证明
      provider 变化与传递依赖变化都会失效缓存。
- [ ] 冷启动改善必须附同 workload 的 HDR/guides 等价性证据（不得只报编译时间）。
- [ ] `PathTracingShaderEncapsulation.md` 增加「分层账本」章节，把上述四项定义为固定
      报告字段。

## 相关文档 / References

- [PathTracingShaderEncapsulation.md](../../PathTracingShaderEncapsulation.md)
- [PathTracingLinkTimeSpecialization.md](../../PathTracingLinkTimeSpecialization.md)
- [PathTracingMaterialBackendExperiment.md](../../PathTracingMaterialBackendExperiment.md)
- [ShaderDeadBranchCleanup.md](../../ShaderDeadBranchCleanup.md)
- [OpenPBRPrimaryGuidesSplit.md](../../OpenPBRPrimaryGuidesSplit.md)
- [ShaderCompilationComplexity.md](../../ShaderCompilationComplexity.md)
