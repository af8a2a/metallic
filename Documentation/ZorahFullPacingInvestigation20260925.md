# ZorahFull pacing 等待增长定位（2026-09-25）

结论：增长发生在 Reflex On 的 `slReflexSleep` 等待中，并伴随 GPU 利用率下降和帧吞吐降低；不是驻留位图增加了 CPU 工作，也不只是等待从帧槽迁移到 Sleep。**慢 On 状态的具体触发条件尚未复现并锁定，不能据此宣称修复了驱动 pacing。** 本轮完成 6 次新采样，保持产品默认 Reflex On / VSync。

## 新证据

1. 使用上一轮优化后二进制原样复测，SHA256 均为 `81C9D7570F650B1DF150273C413544A0D63E91835F2153C4E1B226160AE36A79`，同样 Reflex On / FIFO，Sleep 从历史慢轮 15.41–15.82 ms 回到 6.28 ms，整帧 P95 从 34.21 ms 回到 26.71 ms。驻留优化没有撤回，因此不能把慢轮视为该代码改动的确定性回归。
2. 原四轮按轮次均值平均：Sleep 增加 9.76 ms，帧槽等待减少 1.50 ms，前端合计等待仍增加 **8.26 ms**；其他帧内墙钟时间减少 3.16 ms，最终整帧均值增加 **5.11 ms**。这是均值分解，不相减独立 P95。GPU envelope 均值反而减少约 2.62 ms。
3. 近似采样窗口中，快轮整卡利用率约 93%，慢轮约 66%–68%；图形频率分别约 2894–2896 / 2903–2904 MHz，没有降频证据。该利用率不是应用独占硬件计数，但进程遥测中也未发现另一个大型渲染负载。
4. 同一旧二进制 Off 对照：Sleep 约 0.045 ms，帧槽等待约 9.55 ms，整帧 P95 25.07 ms。只看 Sleep 会夸大收益；驱动报告的 SimulationStart→GPU end 均值反而由快 On 的 35.77 ms 增为 40.21 ms。这不包含显示延迟，也不是实际鼠标到屏幕延迟。
5. 添加诊断字段后，On / FIFO 与 On / MAILBOX 的整帧均值几乎相同（20.365 / 20.364 ms），后续两轮 FIFO P95 也约 26.1 ms。**这些快状态对照没有证明关闭 VSync 可以修复慢状态。**

## 全部轮次

共同条件：RTX 5070 Ti / 616.92，Full 固定路线 30 秒、就绪后预热 10 秒，1797×660 输出 / 1198×440 内部，DLSS Quality、LOD 1.5 px，隐藏编辑器窗口、2 frame slots。相机关键点、graph、shader SHA256、资产、渲染尺寸逐项一致。实时流送和 cut 未冻结；前两轮为位图修改前，后续为修改后。

|轮次|整帧均值 / P95 ms|Sleep 均值 ms|帧槽+Sleep 均值 ms|GPU envelope 均值 ms|>33.33 ms 帧数|
|---|---:|---:|---:|---:|---:|
|修改前 On 1|20.830 / 26.242|6.099|8.074|20.440|0|
|修改前 On 2|20.980 / 26.838|5.619|7.758|20.559|0|
|历史慢 On 1|26.091 / 34.205|15.824|16.363|17.933|108|
|历史慢 On 2|25.931 / 34.215|15.409|15.993|17.835|103|
|同二进制 On 重测|20.696 / 26.706|6.284|8.386|20.297|2|
|同二进制 Off 对照|20.115 / 25.074|0.045|9.599|19.966|0|
|诊断 On / FIFO|20.365 / 26.028|7.497|9.540|19.988|1|
|诊断 On / MAILBOX|20.364 / 25.893|7.415|9.417|19.981|0|
|诊断 On / FIFO 重复1|20.597 / 26.130|6.911|8.927|20.179|4|
|诊断 On / FIFO 重复2|20.671 / 26.188|6.694|8.710|20.277|1|

所有 On 有效 mode=1，Off=0；frameLimitUs=0、采样中 optionsUpdates=0、suspendedFrames=0。历史慢轮非 Sleep 的帧起始开销仅约 0.004 ms，排除应用侧 token、互斥锁和每 60 帧状态查询是主要来源。帧槽等待在 Sleep 前、输入在 Sleep 后；RenderSubmit 包围 RenderGraph 提交及编辑器提交，Present 标记紧邻实际 present。未发现本次缓存改动改变这些调用顺序。

## 定位边界与下一步

证据支持“低延迟 pacing 改变跨帧排队/重叠，慢状态产生吞吐损失”的解释：慢轮等待更长、GPU 使用不足，而驱动报告的渲染延迟更低。但当前没有 SDK 内部阻塞栈或 CPU/GPU 对齐调度时间线，无法区分驱动预测策略、swapchain/窗口状态与 OS 唤醒延迟。采样未锁频，全部为隐藏窗口，未完成可见编辑器的对应验收。

不要永久关闭 Reflex 或按固定阈值跳过 Sleep；SDK 文档要求持续调用，且 Off 对照显示渲染延迟代价。下一步针对慢状态保留短时循环时间线：记录 Sleep 入口/返回、底层 semaphore 等待、CPU 首次提交与前一帧 GPU 完成，并在**同一进程、同一慢状态**切换 On→Off→On。只有捕获到额外空档的边界和切换是否清除它，才决定调整帧槽/提交顺序或向驱动侧定位。目前无需撤回 Streamer 查询优化。

本轮新采样全部未复现历史慢状态；继续盲目启动相同快状态样例的收益有限。默认 On 的四轮新采样仍各有少量超过 33.33 ms 的帧，不能把 P95 达标当作每帧至少 30 fps。

## 新增诊断与验证

- 缓存并导出 Reflex 最新有效报告的 Simulation、RenderSubmit、Present、GPU 起止原始时间，以及 GPU active/frame 字段；仍按既有 60 帧节奏查询，不新增逐帧 GPU 同步。分析按 reportFrameId 去重，不能把缓存报告和当前帧配对。
- 本地 Streamline 的两种 Vulkan 后端把 `gpuActiveRenderTimeUs` 填为 GPU 起止跨度，把 `gpuFrameTimeUs` 填为相邻 GPU end 的间隔；本轮该 active 字段也与 gpuRenderMs 相等。它不是硬件实际忙碌时间，不能用字段名直接推断 shader 工作量或利用率。
- Full runner 增加仅供复测的 `-NoVSync`，保存 Reflex/VSync 参数并恢复环境变量。运行时只在 Full benchmark 环境生效，普通编辑器默认不变。实际 swapchain 创建日志记录 presentMode、image count 与 extent；确认本轮 FIFO=2、MAILBOX=1，均为三图像 swapchain。
- Release MetallicGPUDrivenSample 构建成功；4 轮新诊断构建 Full 捕获完成（另有 2 轮原二进制控制组），全部 10 组旧/新数据通过分析器；无 GPU 计时缺失、loadFailures/requestOverflows=0。没有启用 Vulkan validation，非新的 validation-clean 验收。既有 BLAS overflow 最大 12425 未在本次解决。

证据：[结构化结果](E:/metallic/Documentation/ZorahFullPacingInvestigation20260925.json)。新原始采样在 `build-release/full-pacing-{on-recheck,off-recheck,diag-fifo,diag-no-vsync,diag-fifo-repeat}-0925`。构建日志 `build-release/pacing-diagnostic-build-0925.log`；汇总脚本 `build-release/AnalyzePacingInvestigation0925.py`。整卡遥测取 Capture complete 日志之前 31 秒近似采样窗口，包含导出边界；不是逐帧对齐的 GPU 竞争跟踪。

```powershell
$env:METALLIC_REFLEX_MODE='on'
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-dir> -Runs 1 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660
# 对照非 VSync 时单独增加 -NoVSync；默认保持 FIFO。
python -B Tools/AnalyzeZorahFullReflex.py build-release/<new-dir>/run1
```
