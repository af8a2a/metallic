---
id: "0023"
title: "纹理反馈: textureFeedbackWave 仍默认关闭，生产 A/B 未证明收益（Opaque +16.6%/+36.5%）"
area: geometry
kind: perf
priority: P3
state: in-progress
milestone: M1-Active
labels: ["area:geometry", "kind:perf", "priority:P3", "state:in-progress"]
depends_on: []
evidence:
  - Documentation/TextureFeedbackWave.md:3
  - Documentation/TextureFeedbackWave.md:48-52
  - Documentation/TextureFeedbackWave.md:56-62
  - Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp:656
  - Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp:732
---

## 背景 / Context

纹理流送的 mip 需求由着色端反馈写入。默认每个 lane 各做一次原子 `min`/`add`；`textureFeedbackWave` 实验改为按 wave 归并同 texture 的 lane，由选举出的一个 lane 做每 texture 一次原子。

## 现状 / Current state

- `textureFeedbackWave` 是 opt-in，默认 `false`（`ScenePathTracePass` 读取属性时默认 false），只作用于未拆分的 compute 路径追踪（含 OpenPBR/标准与 guide 变体），且要求 compute subgroup 运算与 `VK_KHR_shader_maximal_reconvergence`。其他设备、ray pipeline、deferred、realtime 与 split-primary 仍走逐 lane 原子。
- 代码已把 shader 置换 `METALLIC_TEXTURE_FEEDBACK_WAVE` 与 `[MaximallyReconverges]` 入口选择绑定在一起，并纳入 shader cache identity；native warmup 还会检查发出的 execution mode。
- 2026-10-18 首次实验：九种 key/活动模式的探针与 CPU oracle 逐字一致、SPIR-V 校验通过；隔离微基准里两/三 key 快约 93%/90%，但**每 lane 一个独立 key 的场景慢约 30%**，单 key 基本不变。
- 生产四进程 ABBA（512×512、2 spp、depth 12、cache off、全驻留 KTX、demand 采集开启）：图像逐字节一致，但paired shading time 变化为 Opaque +16.6%/+36.5%、Mixed -1.8%/+28.1%、Sky +15.2%/+15.6%（约 20 微秒），且 Opaque/Mixed 存在明显的基线漂移。

## 问题或缺口 / Problem

1. 现状只能得出「生产无收益」，无法定位是归并迭代次数、`[MaximallyReconverges]` 对 codegen 的影响，还是测量噪声；也没有 Nsight warp/stall 级别的归因。
2. 接受标准本身依赖「同场景/相机/设置 + 活跃流送 demand + 精确 HDR/temporal/guide readback + 多进程重复计时」，而首次实验用的是全驻留纹理与冻结之外的 host policy，未覆盖真实漫游中 demand 活跃的稳态。
3. 归并收益与 key 基数强相关，而 Full/Zorah 的材质分箱与纹理多样性正是「每 lane 多 key」的最坏情形，缺少该场景的 key 基数直方图。

## 期望结果 / Expected outcome

- 用真实漫游（非冻结 demand）的多进程计时给出结论：采纳、限定适用域（如仅 guide/标准材质）或彻底退役该实验并删掉置换。

## 验收标准 / Acceptance criteria

- [ ] `MetallicVulkanDeviceFeaturesTests --gtest_filter=DeviceFeatures.MaximalReconvergenceIsOptionalAndRequiresExtensionAndFeature` 通过。
- [ ] `MetallicRHITests --rhi-bindless --rhi-validation --gtest_filter=RHIRendering.texture_feedback_wave` 通过，覆盖九种 key/活动模式的逐字对照。
- [ ] 生产验收：同场景/相机/设置、活跃流送 demand、HDR/temporal/guide 逐字节一致，至少三轮 ABBA，报告每轮 paired shading time 与 95% 区间；基线漂移必须单列。
- [ ] 导出每个 wave 的同 texture key 基数分布，作为「适用域」判断依据。
- [ ] 若结论为不采纳：删除 shader 置换与属性，或将其明确标记为仅诊断；更新 `TextureFeedbackWave.md` 的默认值描述。

## 相关文档 / References

- [TextureFeedbackWave.md](../../TextureFeedbackWave.md)
- [ZorahFullTextureStreaming.md](../../ZorahFullTextureStreaming.md)
