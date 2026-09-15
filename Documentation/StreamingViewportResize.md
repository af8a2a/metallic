# 流式加载期间的视口尺寸调整

2026-09-15。GPUDrivenSample 首帧之后，ImGui 停靠布局会改变视口尺寸。用户日志显示：首次图尺寸为 1584×800，DLSS 内部渲染尺寸为 1056×533；随后视口变为 1076×486，VisibilityBuffer 又初始化了完整的 MiniZorah 流式会话。第二次初始化在 10:13:20.548～10:13:21.818 之间耗时约 1.27 秒，并重新创建了管线和页面资源。

原因是 `VisibilityBufferPass::compile()` 把屏幕尺寸作为场景资源复用条件，而且用实际渲染尺寸与图的显示尺寸比较。视口调整或 DLSS 图资源重建因此会调用 `resetStreamIntegration()`，丢弃已经加载的页面和遍历进度。

场景资源复用现在只依据场景身份、结构与内容版本、帧槽和 StreamAsset 来源。尺寸变化由 `execute()` 中既有的 `ensureFrameResources()` 处理，按解析后的渲染尺寸更新 HZB、光栅目标及绑定，重置屏幕历史，并按 GPU 完成状态退役旧资源。Streamer 会话、页池和 CLAS 驻留状态继续使用。

`streamed_realtime_pipeline` 和 `minizorah_realtime_pipeline` 在第 2、4 帧模拟停靠布局尺寸变化，第 6 帧重建同尺寸图资源，并在相机移动后再次调整尺寸。每次检查流送 generation 不变、帧号连续、累计上传字节不回退且只存在一个会话；后续继续验证照明、运动向量和会话回收。MiniZorah 版本同时覆盖 DLSS 显示/渲染尺寸不相等的情况。

验证（RTX 5060，Release，当前构建未启用 NRD）：

- 新增检查在旧实现第 2 帧尺寸调整时失败；修复后小场景和完整 MiniZorah 的断言均通过。MiniZorah 的四次图重建分别为 9.576、8.216、5.725、9.562 ms，流送帧号连续为 3、5、7、182。这些是 CPU 图重建耗时，不包含随后的渲染和屏幕资源分配。
- `render_graph_resize_reuses_compiled_passes`、`render_graph_scene_binding_contract`、`visibility_debug_stable_geometry_identity`、`hzb_spd_visibility_equivalence_timing` 共 4 项通过，退出码 0。
- 默认 Sample 完成 120 帧呈现；停靠布局调整时图重建为 8.85 ms，`refresh reusable passes` 为 1.52 ms，整个运行只初始化一次 MiniZorah 流式会话。
- 本机 Streamline 退出阶段仍出现已有的挂起，离屏测试和 Sample 均在完成渲染后由测试端终止；这两次进程不能记为退出成功。测试未启用 Vulkan 验证层。本次修复不涉及首次建图、DLSS 首次求值或 SDK 关闭成本。

日志保留于 `.cache/streaming-resize-hitch/`：`Before.log`、`After.log`、`Regression.log` 和 `Sample.log`。
