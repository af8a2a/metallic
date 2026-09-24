# WorkloadCase：固定帧上下文与 A/A

`WorkloadCase.py` 将现有 raster comparison harness 限定为一个显式工作负载。它运行注册的生产 shader，不使用 coverage diagnostic shader 代替生产内核。当前范围是同步 graphics queue 上的 early/late 帧内执行；不支持 isolated replay 或任意运行中场景的快照恢复。

```powershell
python -B Tools/Perf/WorkloadCase.py run --case Tools/Perf/WorkloadCase.MiniZorahWorkControl.json --output build/workload-case-new --runs 3 --timeout 180
python -B Tools/Perf/WorkloadCase.py verify build/workload-case-new
```

输出目录必须不存在。默认使用 `build-release/Source/MetallicGPUDrivenSample.exe`，可用 `--exe` 显式指定新构建。每次启动独立进程，进程间串行运行。任何采集、身份或输出检查失败立即停止，不将剩余缺失运行补成成功。

## Case 与生产入口

配置文件显式声明 sample、variant、scope、输出/内部渲染尺寸、预热与恢复帧、采样帧数、轮数，以及事先确定的 A/A 波动阈值。内部尺寸独立校验，不能把上采样后的输出尺寸当作 shader 执行尺寸。MiniZorah 与 Full Zorah 是不同 case；前者通过不代表后者通过。

| variant | 生产 shader 入口 | 内部 raster mode |
| --- | --- | --- |
| swLegacy | streamClusterRasterLegacyMain | 1 |
| swPrepared | streamClusterRasterMain | 0 |
| swPlane | streamClusterRasterPlaneMain | 2 |
| swCooperative | streamClusterRasterCooperativeMain | 3 |
| swWorkBins | streamClusterRasterWorkBinsMain | 4 |
| swWorkControl | streamClusterRasterWorkControlMain | 5 |

这些 mode 与旧 comparison suite 的 10–15 选择值不同。当前自动实测以 WorkControl 为试点，其余入口仍需各自验证。配置不认识的字段、sample、variant 或 scope 会失败。注册 case 固定 SW 阈值 8 pixels、禁用 temporal jitter、禁止异步软件光栅，并使用相同的恢复帧。

## 证据与检查

- 预热达到 streamer ready 后，冻结 geometry/CLAS/cut/TLAS 与 texture publication；保留真实帧上下文和 HZB 历史，不声称每帧所有可写资源均恢复到相同初值。
- 诊断帧读回 active cut、page mappings、early/late bins、软件 bin 的有效列表指纹、indirect dispatch 参数、depth/visibility。读回完成并恢复后才开始计时。
- 在实际绑定生产 compute pipeline 后记录 module、entry、SPIR-V 指纹、early/late 和所走队列分支。它是引擎侧绑定证据，不是 Nsight module hash、原生 queue handle 或 submit ID。
- 每个 measured frame 核对 shader/state、驻留、纹理发布与 GPU timestamps。诊断 scope 不得进入测量帧。非空软件 bin 必须对应非零 dispatch；两阶段合计零工作量直接失败。
- before/after 和各轮之间要求固定输入、binding 与 depth/visibility 字节一致；三个独立进程之间再比较同一身份。某个阶段为空时保留零计数，不能据此宣称该阶段的有效性能。
- 保存 case、实际 engine config、源码/Shader/Pipeline 文件副本与 SHA256、可执行文件和同目录 DLL 的 SHA256、GPU/driver、日志、逐帧数据、输出读回和 GPU telemetry。`verify` 重新核对归档 hash 并重做语义检查。

`AA.json` 使用独立进程的运行中位数作为统计单位；比较 graph GPU 和同步 early+late SW 总时间的跨运行相对极差。阈值来自 case，默认 10%，只是初始资格检查，不是收益显著性阈值。`stable` 表示本次 A/A 满足该阈值，`inconclusive` 表示噪声过大；两者均不接受任何优化候选。depth/visibility 自身重复一致也不等于与参考渲染器等价。

发生 GPU 错误、报告失败或超时只终止本次启动的进程。若完整 `Capture.json` 已写出而应用 20 秒内仍未退出，runner 回收该进程，记录 `teardownReclaimed`；离线检查仍要求报告和全部输出完整。不能把这种运行描述为正常退出。

## 诊断采集边界

`config` 子命令输出 harness 配置，不启动 GPU 工作。`profileHoldSeconds` 会在显式选中的 variant 就绪后写 `ProfileReady.json`，不再只能选择旧版 shader。它与正常计时资格分离。

`nsightTraceFrames` 可设为 1–3，要求 rounds=1 且没有 hold。须由 `ngfx` GPU Trace 活动注入，并启用 `--start-with-ngfx-sdk` / `--stop-with-ngfx-sdk`；应用不会自行注入 GPU Trace。Graphics Capture 与 GPU Trace 不能混用。启动调用发生在冻结、诊断读回和恢复帧之后；停止调用前 drain GPU 工作。SDK 不可用或未注入会失败，不能忽略错误继续出具正常 timing 结论。

这里实现的是 **SDK 采集边界**。`sdkTrace.complete` 只表示 start/stop 成功；独立 trace 文件、Nsight shader/range 对齐与 Source/IL 导出必须另验，`artifactVerified` 不会因 SDK 调用成功自动变为 true。

## 尚未补齐的可信度

本版对超大 StreamAsset 记录路径、长度和修改时间，**没有内容 hash 和完整纹理/依赖闭包**。Manifest 明确写 `portableInputSnapshotComplete=false`。同样，GPU telemetry 不是逐进程竞争证明，`externalProfilerAbsenceVerified=false`。不能把局部 A/A `stable` 当作完整 M2 或自动接受性能优化的资格。

后续需要补内容寻址的资产/依赖身份、准确的竞争进程记录、SDK 实际 trace 验证及 Nsight 范围对齐，并处理 Full Zorah 资源预算与退出清理。异步队列和 isolated 生产内核必须分别建 case 和验收，不能直接套用同步帧内结果。
