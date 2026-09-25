# ZorahFull：慢状态 CPU / GPU 调度时间线

2026-09-25。承接 [上一轮 pacing 调查](E:/metallic/Documentation/ZorahFullPacingInvestigation20260925.md)。本轮已经复现慢状态，并保存了 ETW 调度、阻塞栈、应用提交标记和 GPU 队列通知。

**已定位的额外空档主要在下一帧首次图形提交之前：Reflex 尚未返回，加上返回后串行执行的 CPU 准备。长 Sleep 的主要部分是驱动调用内的阻塞，而非主线程已就绪却抢不到 CPU。** 仍未证明驱动为什么在部分运行中选择了这种节奏，也不能据此宣布整卡完全空闲或永久关闭 Reflex。

![两帧对齐时间线](E:/metallic/build-release/pacing-etw-focused-0925/Timeline.png)

## 1. 条件与复现

RTX 5070 Ti，驱动 616.92；Release Full 隐藏编辑器固定路线，输出 1797×660 / 内部 1198×440，DLSS Quality、LOD 1.5 px、SW 8 px、FIFO、2 frame slots。每轮就绪后预热 10 秒，采集 30 秒。流送、cut 和相机按既有路线运行，没有冻结。新增标记不改变 Sleep / 提交顺序及策略。

|轮次|帧均值 / P95 ms|Sleep 均值 ms|>33.33 ms 帧数|用途|
|---|---:|---:|---:|---|
|初次 On，内置 WPR CPU+GPU|21.791 / 29.702|4.384|16|快状态参考|
|focused On|26.018 / **33.986**|**14.417**|95|慢状态主要证据|
|focused Off|18.973 / 24.753|0.053|2|辅助对照，非同进程因果实验|
|queues On|23.010 / 32.483|见原始摘要|42|后段复现长 Sleep，取得 GPU 包通知|
|queues On repeat|22.116 / 30.441|6.105|14|快状态参考，ETW 提前超时|

Off 仍有帧槽等待，Sleep 减少不等于等量整帧收益。该轮 SDK SimulationStart→GPU end 均值 37.396 ms，高于慢 On 的 29.441 ms；这不是输入到显示延迟。

采样限制：focused Off 后段开始了本地 ETL 导出；queues On 期间本地 Python 分析意外持续占用一个 CPU 核，之后已停止。因此不把这两轮整帧差值当作干净的 A/B 优化收益。queues On 的具体帧调度顺序、提交和完成时间仍可直接核查；该帧主线程 Sleep 后仅有约 0.018 ms 的调度离核间隔，不能将 8.682 ms 的准备阶段解释成该线程抢不到 CPU。

## 2. Sleep 内部究竟在等什么

focused On 的 CPU 环形记录保留了 ETL 相对时间 **76.871797–84.543933 s**，即采样最后约 7.67 秒。其中完整覆盖 **105 次 Sleep >20 ms**：

|Sleep 内状态|均值 ms|P95 ms|
|---|---:|---:|
|总跨度|23.125|25.141|
|阻塞，尚未 Ready|**23.057**|25.036|
|Ready / 被抢占后等待 CPU|**0.019**|0.044|
|运行时间余量|0.048|0.120|

所有 105 次主要阻塞都有 ReadyThread 对应；主线程 TID 36512 被同进程 TID 36760 唤醒。阻塞栈一致包含：

```text
MetallicGPUDrivenSample → sl.reflex.dll → sl.common.dll → nvoglv64.dll
                       → KernelBase.dll → ntdll.dll / ntoskrnl.exe
```

记录的等待原因为 `UserRequest`。Release 没有应用 PDB，以上是模块级调用链，不能冒充已经符号化的驱动私有函数或具体 Vulkan semaphore。

queues On 的 frame 2090 进一步记录了驱动工作线程 TID 39428：Ready 83.165850 s，上 CPU 83.165855 s，随后于 83.165908 s 唤醒主线程。主线程于 83.165915 s 上 CPU，83.165934 s 从 Sleep 返回。其工作线程栈也包含 `nvoglv64.dll`。该次主线程 Ready 延迟仅 7 µs，驱动工作线程为 5 µs。唤醒事件的 `InDPC` 当前进程名不能当作等待发起者；没有定时器目标到期时间，因此尚不能区分驱动预测与定时器实际到期偏差。

## 3. 额外空档的边界

### 慢状态 frame 1851：12.725 ms 的 GPU 帧包络间隔

CPU 来自 ETW；GPU 边界来自同一帧 ID 的 Reflex 原始报告。每份报告以自己的 SimulationStart 标记对齐，不能套用一个全程固定时钟偏移，也不能把缓存报告当成导出时的当前帧。

|事件|ETL 相对时间 ms|
|---|---:|
|SleepBegin|83589.137|
|前一帧 GPU end（SDK 推导）|83607.223|
|SleepEnd|83611.090|
|RenderSubmitStart|83611.924|
|首次图形 vkQueueSubmit2 入口|83619.911|
|本帧 GPU start（SDK）|83619.948|

拆分：**3.867 ms** 仍在 Sleep 中，**8.821 ms** 为 Sleep 返回到首次提交，之后到 SDK GPU start 约 **0.037 ms**。此次 Sleep 的 21.953 ms 中，21.843 ms 为阻塞，Ready 等待 0.032 ms。

本地 Streamline Vulkan 后端用相邻 GPU end 的间隔填充 `gpuFrameTimeUs`，故 `previousGpuEnd = gpuEnd - gpuFrameTimeUs`。该指标是帧包络，包含跨队列重叠；负差值表示包络重叠，不能裁零后冒充硬件空闲率。采样内 19 个去重、匹配的报告，带符号间隔均值 7.386 ms、最大 15.407 ms。报告每约 60 帧更新，并不覆盖每一帧。

### frame 2090：DxgKrnl 同时确认“旧包完成，新包尚未提交”

主图形上下文 `0xffffb28fac517a00`；依据该进程实际 HAGS 提交关联，不依据中断执行时的 PID 归属。

|事件|ETL 相对时间 ms|
|---|---:|
|旧主队列包完成，progress fence 19601|83164.697|
|主线程 SleepEnd|83165.934|
|首次图形 vkQueueSubmit2 入口|83174.616|
|新主队列包提交通知，progress fence 19602|83174.646|
|新包完成通知，progress fence 19602|83174.785|

旧包完成→新包提交 **9.949 ms**；其中旧包完成→Sleep 返回 **1.237 ms**，再准备 **8.682 ms**，进入 Vulkan 后约 **0.030 ms** 出现新包提交通知。该包提交到完成仅 0.139 ms；它是该批次的首个小包，不能代表整帧渲染耗时。Reflex 的本帧 GPU start 为 83174.720 ms，前帧 end 为 83164.650 ms，包络间隔 **10.070 ms**，与队列通知相互印证。

其他工作负载和辅助队列可在此期间运行；结论是本场景主图形队列缺少下一帧新工作，**不是“整卡空闲 9.949 ms”**。HAGS 提交/完成通知也不是硬件 shader 执行起止时间。

## 4. 首次提交之前的工作

frame 1851 的现有 CPU scope（包含关系，不能相加）：

|阶段|ms|
|---|---:|
|Record RenderGraph|8.151|
|Streamer prepare|**5.039**|
|Stream Begin|4.528|
|Consume requests|**4.304**|
|其中 Deduplicate loads|0.674|
|其中 Update resident demand|1.213|
|其中 Admit request batch|2.415|
|Admit 子项 Refresh request priorities|1.277|
|Visibility prepare|0.994|
|Update GPU Profiler|0.562|

frame 2090 同样为 Streamer prepare 4.758 ms、Consume requests 4.180 ms、Refresh request priorities 1.277 ms。CPU 准备消耗落在提交前，慢 pacing 又减少了其与前帧 GPU 工作重叠的机会。

下一步优先顺序：

1. **先做相位实验**：在保持输入采样位于 Sleep 之后的前提下，从 StreamerSubsystem 分离只依赖已完成帧的反馈消费、退休回收和异步结果整理，验证能否在 pacing 前或独立 CPU 作业中完成。先限定生命周期、代际和完成信号；本帧相机需求、cut 与资源发布仍在正确同步点处理，render pass 不接回资源加载职责。测首次提交时间及渲染延迟，不能只看某个 scope 缩短。
2. **继续减少请求维护本身的工作量**：优先处理 resident demand 与 Refresh request priorities 的增量更新，避免对未变化需求全量刷新；同视角/cut/驻留条件验证质量与队列积压。
3. **驱动 pacing 单独验证**：同进程捕获慢状态后 On→Off→On，观察队列空档及 SDK 渲染延迟是否跟随切换。当前跨进程 Off 比较和模块栈不足以断言驱动缺陷。不要以固定阈值跳过 Sleep，也不据此调整系统线程优先级。

本轮仅新增诊断，没有改变 Reflex 默认模式、VSync 或提交策略，没有宣称取得性能优化收益。

## 5. 工具、验证与原始证据

- `METALLIC_PACING_TRACE=1` 启用 TraceLogging provider `Metallic.Pacing`，GUID `{69d58154-7152-4112-a81b-69f8598bb378}`；默认关闭。Sleep / PCL 使用真实 Streamline frame ID；队列事件使用时间、TID、queue family，避免给异步上传伪造当前帧 ID。
- [采集器](E:/metallic/Tools/CapturePacingTimeline.ps1) 用唯一 WPR instance，只停止自己创建的会话；超时自动保存，保存失败只取消自己的会话。最新版本快照 WPRP 及其 SHA256。工作负载不提升权限。
- [WPRP](E:/metallic/Tools/PacingTrace.wprp) 包含 CSwitch / ReadyThread 栈、DxgKrnl 与应用标记；`wpr -profiles` 验证通过。GPU provider 使用 GUID / NonPagedMemory；初版 focused 配置没有生成 DxgKrnl，故慢 focused ETL 不包含 GPU 包，不能声称有。
- [分析器](E:/metallic/Tools/AnalyzePacingTimeline.py) 支持按 frame ID 匹配报告、逐份时钟锚定、CPU 阻塞/Ready 拆分及 HAGS 通知导出；已在两组实测 ETW 导出上运行，105 次长 Sleep 统计与独立分析一致，frame 2090 精确事件断言通过。
- Release `MetallicGPUDrivenSample` 构建成功，PowerShell 解析、WPR profile 解析和 `git diff --check` 通过。5 轮 Full 应用采样完成；不是可见编辑器交互验收，也未增加 Vulkan validation 验收。
- focused On 和 queues On 的 ETL header 均为 0 lost buffers / events；**这不等于保留了全程**：Memory 环形缓冲覆盖旧记录。focused CPU 只分析最后 7.67 秒；queues On GPU 记录约 77.34–88.77 s，覆盖 frame 2090。queues repeat 在 150 s 超时保存，缺少 CaptureEnd，不作为全程 ETW 证据。所有本轮采集控制器均已结束。

主要文件：

- [慢状态 ETL](E:/metallic/build-release/pacing-etw-focused-0925/trace1/Pacing.etl)、[慢状态分析](E:/metallic/build-release/pacing-etw-focused-0925/trace1/Timeline.json)、[Full 帧数据](E:/metallic/build-release/full-pacing-etw-focused-on-0925/run1/Frames.jsonl)。
- [GPU 队列 ETL](E:/metallic/build-release/pacing-etw-queues-0925/trace1/Pacing.etl)、[队列与 CPU 分析](E:/metallic/build-release/pacing-etw-queues-0925/trace1/Timeline.json)、[70 ms CPU 导出](E:/metallic/build-release/pacing-etw-queues-0925/trace1/ExampleCpu.csv)。
- [可随文档保留的结构化证据](E:/metallic/Documentation/ZorahFullPacingTimeline20260925.json)、[时间线图片](E:/metallic/build-release/pacing-etw-focused-0925/Timeline.png)。

重采示例（管理员 PowerShell 仅运行采集器；看到 `trace1/Started.json` 后立即在普通终端启动工作负载）：

```powershell
pwsh -NoProfile -File Tools/CapturePacingTimeline.ps1 -OutputRoot E:/metallic/build-release/<new-etw> -Sessions 1 -SessionTimeoutSeconds 300
```

```powershell
$env:METALLIC_PACING_TRACE='1'
$env:METALLIC_REFLEX_MODE='on'
try {
    pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-run> -Runs 1 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 240
} finally {
    'save' | Set-Content build-release/<new-etw>/trace1/Stop.request
    Remove-Item Env:METALLIC_PACING_TRACE
    Remove-Item Env:METALLIC_REFLEX_MODE
}
```

导出与分析（替换路径；CPU 短窗口必须覆盖完整 Sleep，queue family 参数依据设备配置）：

```powershell
xperf -i <trace>/Pacing.etl -o <trace>/Markers.csv -a dumper -provider '{69d58154-7152-4112-a81b-69f8598bb378}'
xperf -i <trace>/Pacing.etl -o <trace>/Gpu.csv -a dumper -provider '{802ec45a-1e99-4b83-9920-87c98277ba9d}'
xperf -i <trace>/Pacing.etl -o <trace>/Cpu.csv -a dumper -range <start-us> <end-us> -stacktimeshifting
python -B Tools/AnalyzePacingTimeline.py --run <run>/run1 --markers <trace>/Markers.csv --cpu <trace>/Cpu.csv --gpu <trace>/Gpu.csv --graphics-family 0 --output <trace>/Timeline.json
```

