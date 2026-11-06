---
id: "0042"
title: "Material: Phase 11 之后的 packed closure / split 调度无实现，M5 存储与调度决策缺证据"
area: material
kind: research
priority: P2
state: proposed
milestone: M5-Research
labels: ["area:material", "kind:research", "priority:P2"]
depends_on: ["0041"]
evidence:
  - Documentation/MaterialSystemPhase11.md:3-5
  - Documentation/MaterialSystemPhase11.md:98-100
  - Documentation/MaterialSystemPhase11.md:123-125
  - Documentation/MaterialSystemRoadmap.md:210-225
  - Source/Runtime/Material/MaterialClosureClassification.h:8-18
---

## 背景 / Context

路线图 M5 是「按证据选择存储和调度后端」，包含四条独立实验：fused 对比 packed
closure + lighting 分类、普通 indirect 对比 DGC active-program、静态 RT 分发对比
wavefront、payload 压缩及专用 kernel。每条都以「触发条件 + 决策依据」定义，不是
必须依次启用的功能列表。

## 现状 / Current state

- Phase 11 只交付了**逻辑两级分类**：`MaterialClosureClassification` 把 Program slot
  映射到 dense Family slot，并给出稳定的 program 分组顺序；注释明确「Program slots
  still select material evaluation; family slots select a compatible closure layout
  and lighting implementation」（`Source/Runtime/Material/MaterialClosureClassification.h:8-18`）。
- 生产仍使用 fused dispatch。Phase 11 原文：「本次数据不支持把全屏 Closure Buffer
  升为生产默认路径，因此没有进入 Phase 12 的生产 Packed Closure Buffer 实现」
  （`MaterialSystemPhase11.md:3-5`）。
- split 原型在 1024×512 的全部四种负载上都更慢（2:1 与 8:1，条带与逐像素混合），
  且 payload 占 48 MiB、合计新增 50.125 MiB
  （`MaterialSystemPhase11.md:98-100`）。
- 决策记录为「生产保留 fused，只落地逻辑两级分类和统计」，并声明不排除 payload
  压缩、更昂贵材质图、更高光源数或其他 GPU 上 split 可能受益
  （`MaterialSystemPhase11.md:123-125`）。
- 仓库里没有 `MaterialSystemPhase12.md`：Phase 编号停在 11。
- DGC 侧只有既有能力（execution set / preprocess / 读回测试），从未接入材质
  active-program 调度；路线图把它列为「后续任务是接入并测量收益」。

## 问题或缺口 / Problem

1. **文档承诺与代码之间缺一个 Phase 12。** 路线图 M5 的四条实验里，只有 packed
   closure 做过原型（且结论是「不采纳」）；DGC active-program、wavefront、payload
   压缩三项在仓库中没有任何实现或测量记录。现有 `MaterialClosureClassification`
   容易被误读为「两级调度已上线」，实际只是分组顺序，不改变每像素执行路径。
2. **缺少触发条件的量化监测。** M5 的决策依据是「Program 扫描成本、编译时间、
   代码体积、寄存器成本变得显著」，但仓库没有任何持续统计这些量的入口；没有数据就
   无法判断何时该重开这条线。
3. **payload 预算只存在于 prototype。** `maxPayloadBytes = 96` 是实时 profile 预算，
   而 split 后 96-byte payload 的带宽与显存代价是原型实测的唯一硬约束，却没有写入
   任何长期预算表。

## 期望结果 / Expected outcome

- 增加可长期运行的「材质调度观测」输出：每 generation 的 Program 数、Family 数、
  实际 SPIR-V 体积、PSO 创建时间、fused 材质 pass 时间。数值来自生产 pass，不是
  prototype。
- 至少一条 M5 路线获得等负载测量并给出采纳/否决结论；若否决，把触发条件写成可
  自动检查的阈值。
- 文档层面明确 Phase 12 状态（未开始 / 已否决 / 已采纳），避免路线图与 Phase 编号
  之间出现空档。

## 验收标准 / Acceptance criteria

- [ ] `MaterialClosureClassification` 的日志行（`generation=... programs=...
      families=... mode=fused`）在至少 3 个生产场景中被采集并归档，作为基线数值。
- [ ] 新增一个只读的调度观测报告（CSV/JSON），包含 Program 数、SPIR-V 字节数、
      PSO 创建时间与材质 pass GPU 时间；连续两个进程的数值可复算。
- [ ] 对 DGC active-program 或 wavefront 之一完成等负载 A/B：正确性门槛与原路径
      逐位一致，且报告 CPU 提交/GPU 时间，明确「收益是否超过测量噪声」。
- [ ] 文档新增 Phase 12 条目（含结论或无实现原因），并从
      `MaterialSystemRoadmap.md` M5 表格链接过去。

## 相关文档 / References

- [MaterialSystemPhase11.md](../../MaterialSystemPhase11.md)
- [MaterialSystemRoadmap.md](../../MaterialSystemRoadmap.md)
- [MaterialSystemM8RayExecution.md](../../MaterialSystemM8RayExecution.md)
- [MaterialPerformanceBaseline.md](../../MaterialPerformanceBaseline.md)
