---
id: "0022"
title: "纹理上传: 仍按 128 regions / 64 MiB / 3 批在途合批，U2 批次组织与逐 mip copy 合批未落地"
area: geometry
kind: perf
priority: P2
state: open
milestone: M4-Performance
labels: ["area:geometry", "kind:perf", "priority:P2"]
depends_on: []
evidence:
  - Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1043
  - Documentation/ZorahFullTextureUploadPlan.md:11
  - Documentation/ZorahFullTextureUploadPlan.md:35-39
  - Documentation/ZorahFullTextureUploadPlan.md:57
  - Documentation/ZorahFullTextureUploadPlan.md:65-72
---

## 背景 / Context

ZorahFull 首帧的材质上传是一条约 60 秒的长链：4418 张 KTX2、512 cap 下保留 44,140 个 mip，最终 image allocation 约 1.369 GiB。U0（可归因基准）与 U1（每图一次 reader、每 worker 一个 Zstd context）已完成，有界并行准备也已落地（默认 4 worker，预热后 wall 中位数 2346 → 1095 ms）。

## 现状 / Current state

- 合批仍是 `kUploadBatchRegionLimit = 128` regions、64 MiB staging 页、最多 3 批在途；历史日志 369 批、平均 3.77 MiB/批，12 张常见纹理的 512 尾链恰好 4,194,624 B 就触发提前提交。
- 结构性的重复工作仍然存在：整场景约 44,140 条单 region 纹理 copy（同一 image 的多 mip 可以合成一次 multi-region copy，理论降到约 4418 条）；每个 copy batch 还各建一套 upload/acquire pool + command buffer + semaphore，369 批意味着约 738 次 queue submit 及对应对象创建/销毁。
- 已完成的部分不算缺口：逐 mip 重开文件与 decoder 创建（U1）、有界并行准备（U3 的一部分）。
- 长尾证据仍在：首末批提交跨度 59.886 s，≥250 ms 的间隔有 65 段共 44.536 s，其中 43 段发生时上一批在途数 <3，说明仅提高在途批次数不是有证据的首选。

## 问题或缺口 / Problem

1. U2（批次组织与对象复用）没有实施：没有 512/1024 regions 与 16/32/64 MiB 的 A/B 结论，pool/command buffer/timeline 复用、按 image 合并多 mip copy、按 image 合并 barrier 都还没有代码。
2. staging 页利用率约 6.25%（只用约 4 MiB 却独占一页），但这只是预留容量口径，没有实测证明改批大小能省时间——所以必须先量后改。
3. U4（交互加载与可选缓存）未做：图准备仍走同步 pump，没有进度/取消；若 U1–U3 后仍由文件读取主导，才需要带内容签名的 mip-tail pack/cache（见 0018 的加载阻塞问题）。

## 期望结果 / Expected outcome

- 在固定 512 cap、不降低 mip 质量、不扩大 staging 峰值的前提下，用可归因数据决定批大小与 copy 合并，并复用上传对象。

## 验收标准 / Acceptance criteria

- [ ] A/B 至少覆盖 128 / 512 / 1024 regions 与 16 / 32 / 64 MiB 两个维度，报告批次/regions 数、copy 与 acquire 时延 P50/P95/max、CPU 与 staging 峰值。
- [ ] 若合并多 mip copy：同一 image 的 mip 内容哈希与逐 mip copy 完全一致；截断/坏数据仍明确失败。
- [ ] 若复用 pool/command buffer/timeline：等待完成后才 reset，不误用在途资源；取消与失败路径统一撤销 pending。
- [ ] 端到端记录 `alpha-test resources`（或新的分段 scope）冷/热加载 wall、首批到首帧可用时间、UI 最长无响应间隔。
- [ ] `MetallicRHITests --rhi-validation --rhi-bindless --gtest_filter='*ktx2*:*material*:*streaming*'` 全绿，覆盖 BC4/5/7、NPOT、小 mip、swizzle/sRGB、MASK/玻璃、文件截断、取消、resize。
- [ ] 不以扩大 staging 峰值或降低 mip 上限换取加载时间来通过验收。

## 相关文档 / References

- [ZorahFullTextureUploadPlan.md](../../ZorahFullTextureUploadPlan.md)
- [ZorahFullBoundedTextureUpload.md](../../ZorahFullBoundedTextureUpload.md)
- [ZorahFullU01Upload.md](../../ZorahFullU01Upload.md)
