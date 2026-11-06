---
id: "0034"
title: "RHI: native resident 细分渲染图像与参考不一致（32/96 个 resident 组合失败）"
area: rhi
kind: defect
priority: P1
state: open
milestone: M2-Correctness
labels: ["area:rhi", "kind:defect", "priority:P1"]
evidence:
  - Documentation/NativeResidentImageInvestigation.md:1-64
  - Documentation/NativeDescriptorHeapStrideWorkaround.md:196-228
  - tests/rhi/TessellationTests.cpp:202-205
  - Source/Runtime/Render/Core/SlangCompiler.cpp:484-496
---

## 背景 / Context

在 typed uint64 指针链修复解除 native `streamClusterRasterMain` 的驱动编译崩溃之后，
`tessellation_displacement_render` 与 `tessellation_recursive_render` 在 native
（`METALLIC_SLANG_DESCRIPTOR_MODE=native`，默认仍是 mapped）下能运行，但 resident 路径的
图像与显式位移参考几何不一致；同轮的 stream 组合没有图像差异。

## 现状 / Current state

- 驱动 617.42 复测（与更新前 SHA-256 相同的测试二进制 + Slang 2026.18.2）：
  位移 64 组合中 32 个 resident 组合失败、32 个 stream 组合无差异；递归位移 192 组合中
  96 个 resident 组合失败、96 个 stream 组合无差异。mapped 侧 64/192 全通过。
- 差异覆盖 `baseColor` 与 `shadingNormal` 两个 debug view，并横跨不同投影、负 X 缩放镜像
  （同时关 `clusterPrebin`）、`edgePixels`、递归拆分深度与 legacy reversed-Z 设置。
- 隔离对照（只替换本地 shader cache 并逐字节恢复）：仅把 resident mesh 换成 mapped 字节码
  → 64 组合全通过；**只**把 height texture 换成 mapped descriptor lowering、保留 native
  buffer → 同样全通过。反之，Depth=2→0、显式 NonUniform 装饰、
  `OpConstantSizeOfEXT` 换成实测 stride 32 都不能消除差异。
- 受影响 resident mesh 模块不含 uint64；新旧 normalizer 对它产出逐字节相同的 60,836 B
  字节码（SHA-256 `7b10c41…`），因此 typed uint64 策略与该模块无关。
- 这些完整像素矩阵在 native/mapped 两侧都关闭 VVL；首次开启官方 VVL 1.4.363 的 native
  递归运行 300 s 超时，不计作通过。

## 问题或缺口 / Problem

差异与 native resident mesh 的**图像 handle lowering** 强相关（只改这一部分即恢复图像），
但改 lowering 同时改变 shader 代码生成，所以还不是排除了其他代码生成影响的最小根因证明，
不能直接断言为已确诊的驱动 image-load 缺陷。它使 native 迁移的整体验收无法全绿，
也让「native 与 mapped 语义等价」这一前提在流式光栅路径上仍未成立。

## 期望结果 / Expected outcome

构造独立 GPU 回读：在 mesh 阶段读取图像尺寸与逐 texel 值，与 compute 阶段、非零 descriptor
索引、不同 heap 布局分别对照，判定问题出在 Slang 输出还是驱动执行。在结论之前不引入
「按入口切 mapped」「BDA」「按 shader 名称识别」之类的生产特例。

## 验收标准 / Acceptance criteria

- [ ] 新增 mesh 阶段图像尺寸 / 逐 texel GPU 回读对照（compute 与 mesh、index 0 与 index≠0、
      两种 heap 布局），结论写入文档并与 SPIR-V 字节对照关联。
- [ ] `METALLIC_SLANG_DESCRIPTOR_MODE=native build/tests/MetallicRHITests.exe --rhi-no-validation
      --gtest_filter=*tessellation_displacement_render:*tessellation_recursive_render`
      的 32 + 96 个 resident 组合全部无图像差异；对应 stream 组合保持无差异。
- [ ] mapped 侧 64/192 组合与后续 live material edit 检查继续通过。
- [ ] 修复不得依赖按 shader 入口 / 名称切换 mapped、BDA 或关闭 normalization。
- [ ] 报告记录驱动版本、Slang 版本与是否启用 VVL（含 VVL 超时未完成的说明）。

## 相关文档 / References

- [NativeResidentImageInvestigation.md](../../NativeResidentImageInvestigation.md)
- [NativeDescriptorHeapStrideWorkaround.md](../../NativeDescriptorHeapStrideWorkaround.md)
- [NativeAtomicPointerFixPlan.md](../../NativeAtomicPointerFixPlan.md)
- [DynamicResourceUpgradeStatus.md](../../DynamicResourceUpgradeStatus.md)
