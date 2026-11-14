# Debug source / capture layout 说明对账（2026-11-14）

范围：已有 Debug provider 的文档准确性，关联
[#56](https://github.com/af8a2a/metallic/issues/56)；基线 `fc3375f73499159b48b526c6dc85a1c6af5a1d3e`。
仅修改 [DebugControlPlane 主说明](../DebugControlPlane.md) 与本记录，不修改运行时代码、
shader、测试、ABI、配置或 checkpoint 注册。不是新增 pass capture/probe 能力。

## 已确认的过时声明

主说明原来宣称 global GPUScene tables 包含 meshlets / meshlet draws
(`VisibleClusterRecord`)、支持 Resident / StreamPage typed decoding，并给出
`render_graph_gpu_driven_mixed_producer_render` 命令和 Resident+streamed 图像断言说明。
这些不能直接沿用到当前代码：

- [实际 global bindings](../../Source/Runtime/Render/Debug/RenderDebugProviders.cpp#L137-L148)
  只发布 instances、geometries、drawInstanceIds。注册 layout 不等于发布该 layout 的实时 buffer。
- [当前 streaming capture binding](../../Source/Runtime/Render/Debug/RenderDebugProviders.cpp#L183-L190)
  是 `CompactStreamVisibleRecord`，仅 AfterPass 可抓；
  [compact schema](../../Source/Runtime/Render/Debug/RenderDebugProviders.cpp#L57-L65)
  提供 packed、clusterIndex、groupIndexPlusOne、rasterFlags。
- [decoded producer 类型](../../Source/Runtime/Render/GPUDrivenRaster.h#L93-L126)
  仍保留 16-byte `VisibleClusterRecord`，但唯一受支持 source 为 StreamPage=1；
  [Slang unpack](../../Shaders/Modules/GPUDriven/GPUDrivenRasterCommon.slang#L25-L40)
  只有 group 非零才写 source 1，清零记录不构成 Resident producer。
- [关系分析器](../../Source/Runtime/Debug/DebugAnalysis.cpp#L138-L162) 的 source / recordValidity
  关系针对 decoded `VisibleClusterRecord`，不能描述成 compact capture 已自动获得这些关系。
  source 1 标为 StreamPage，其他值标为 Unknown；source 分类和 captured visibility pixel
  引用的有效性是两回事。现有 source-0 geometry/meshlet 分支仍可能产生 raw-storage 关联，
  这不恢复已退休的 Resident source，也不在本次文档变更中删除该分支。
- [page-table schema](../../Source/Runtime/Render/Debug/RenderDebugProviders.cpp#L77-L82)
  中 Resident residency state 仍合法，不能把同名但不同字段的概念一起删掉。

[历史提交 b91fecf61](https://github.com/af8a2a/metallic/commit/b91fecf61ee722b3e7d2435b124a56dc7ff4c90b)
移除了 Resident source，并将旧 mixed-producer 回归改为 composed-stream。
前一轮 [CPU fixture 修正](DebugCaptureFixtureFollowup-2026-11-14.md) 已锁定
Unknown + Unverified，同时保留原有 raw-storage 关联断言；本轮不重复修改测试。

## 当前测试契约与文档修正

[实际 RHI 名称](../../tests/rhi/RenderGraphTests.cpp#L8849-L8858) 为
`render_graph_gpu_driven_composed_stream_render`，且
[类已注册](../../tests/rhi/RenderGraphTests.cpp#L10832-L10847)；旧名称不在当前源码中。
主说明的命令随之更新，而不是让一个不存在的 filter 继续冒充当前回归入口。

[checkpoint 断言](../../tests/rhi/RenderGraphTests.cpp#L9044-L9105) 覆盖
AfterTraversal / AfterEarlyCull / AfterLateCull / AfterPass、四 job 的 execution 一致性，
以及 active-header min/max GPU probe 返回的 minimum 与 captured activeGroupCount 一致性。
[图像断言](../../tests/rhi/RenderGraphTests.cpp#L9384-L9444) 通过 compact record 解包后的
active-group instance ID 区分两个 composed stream instance，并检查共享 color/depth/visibility；
[可视化模式检查](../../tests/rhi/RenderGraphTests.cpp#L9457-L9543) 保持 raw visibility/depth 不变。
它们不是 Resident typed decoding 或旧 visible-record-base 断言。

## 验证与余额

- 本地文档检查器扫描两文件：30 个 Markdown 链接，0 错误；本地目标、标题锚点与源码行范围均检查。
  外部 URL 仅识别并跳过，不声称已联网验证。3 个负例（缺失目标、缺失标题、越界行号）均被拒绝。
- 3 个源码/文档 filter 检查通过：主说明使用当前 filter、当前 RHI 类注册存在、
  旧 filter 不在主说明或当前 RHI 源码中。这是静态一致性检查，不是 3 个 GPU 测试。
- 独立只读审查 `d2-debug-doc-oracle`：APPROVE；按意见将 probe 的比对精确为 minimum，
  不把 min/max 两个结果都描述成已被断言。
- root 复核最终 scoped diff；`git diff --check` 无输出。检查器为 ignored 本地构建输出，未纳入源代码提交。

本轮本地检查命令（在仓库根目录）：

```powershell
python -B build/d2-round4-20261114/ValidateDebugSourceDocs.py
git diff --check -- Documentation/DebugControlPlane.md Documentation/Issues/DebugSourceDocumentationFollowup-2026-11-14.md
```

独立审读还发现：旧 legacy filter 没有匹配时，默认 GoogleTest 策略允许零测试并警告后成功退出。
这不是本轮运行证据，也未在此修改 harness；后续应单独建立 fail-on-empty-selection 的
CPU 回归，避免废弃文档命令被零测试退出码掩盖。

这是文档修正，不重新构建 renderer，不执行 GPU/editor/E2E。上述 RHI 说明来自当前测试源码，
不是本轮运行通过的证据；上一轮 CPU fixture 的结果也不能证明 GPU 图像正确性。
compact capture 的自动 decoded source/validity relations 不在本次实现范围内。
#56 的新增 pass/checkpoint、能力提示与新 pass E2E 验收没有因此完成，保持开放。
