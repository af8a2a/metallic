# MiniZorah CPU Stream Begin 细分

已将 `Stream Begin` 下的 CPU 工作接入现有 profiler 树。重启更新后的 `build-release/Source/MetallicGPUDrivenSample.exe`，在 **Profiler → Table → GPUDriven → Stream Begin** 展开即可查看；新子项的 Queue 为 CPU，GPU 列为 `--`，可按 CPU avg 排序。VBuffer 与独立 GPUDrivenStreamAsset 两条调用路径都接入了相同的统计。

本次只添加观测，不改变页面请求、冷页回收、CLAS 分配或上传调度行为。计时使用 render thread 的 steady-clock elapsed time，不是 CPU cycle，也不包含后台 I/O/解码线程的完整执行时间。

**计时覆盖范围**

| 一级 scope | 进一步细分 |
| --- | --- |
| Residency completion | reset/latency、update/storage/unload 完成任务、排队请求 |
| CLAS completion / expiry | 完成的 CLAS 批次、退休 CLAS 到期扫描 |
| Retire unloaded CLAS | 几何卸载对应的 CLAS 退休 |
| GPU request feedback | 映射、请求消费、解除映射；请求消费再分加载/卸载去重、驻留需求更新、请求入队 |
| Joint cold page reclaim | 待释放内存抵扣、候选准备、卸载调度；候选准备再分驻留扫描与排序 |
| Discard obsolete CLAS plans | 清理失效上传计划 |
| Prepare page uploads | 排队加载排序、I/O 调度、已解码页面收集、待上传排序、传输批次准备、payload/CLAS plan、上传跟踪提交 |
| Queue resident CLAS | 新驻留页加入 CLAS 队列 |
| Publish resident pages | 当前驻留页列表写入 host buffer |

CPU 子项不分配 GPU timestamp；完成后挂接到当前 `Stream Begin` 下，保留原 GPU scope 的查询索引与异步结果对应关系。每个阶段按批次计时，没有逐页生成 profiler 节点。CPU-only 标志也写入 Frames.jsonl；分析 CSV 中 CPU-only 行的 GPU 数据为空，而非伪造 0 ms。条件执行 scope 的 cpuAvgMs 按整段帧数摊销，cpuRecordedAvgMs 仅统计实际出现的样本。

**同轨迹实测**

沿用 [cfg 回放基准](E:/metallic/Documentation/MiniZorahCfgRoam.md) 的 3,000 帧相机输入：前进 12 个场景单位、停留、返回。最终采样目录为 `build-release/stream-begin-profile/replay-v2`；m1/m2 为关闭 validation 的性能回放，quality 单独开启 validation 与检查点读回。下表均为每段 300 帧的 CPU 均值，单位 ms。

| scope | 起点 m1 / m2 | forward_2 m1 / m2 | 返回停留 m1 / m2 |
| --- | ---: | ---: | ---: |
| Stream Begin 总计 | 0.900 / 0.613 | 2.357 / 2.310 | 0.905 / 0.858 |
| CLAS completion / expiry | 0.408 / 0.186 | 0.458 / 0.438 | 0.210 / 0.209 |
| GPU request feedback | 0.228 / 0.184 | 0.559 / 0.563 | 0.241 / 0.232 |
| Joint cold page reclaim | 0.250 / 0.232 | 1.190 / 1.160 | 0.435 / 0.398 |
| Prepare page uploads | 0.0009 / 0.0008 | 0.116 / 0.114 | 0.0015 / 0.0014 |

移动段 `forward_2` 的冷页回收进一步分解为：

| 子项 | m1 ms | m2 ms |
| --- | ---: | ---: |
| 抵扣待释放内存 | 0.159 | 0.155 |
| 扫描驻留候选 | 0.204 | 0.204 |
| **排序冷页候选** | **0.700** | **0.680** |
| 调度冷页卸载 | 0.126 | 0.120 |

同期 CLAS 退休页扫描为 **0.439 / 0.420 ms**，驻留需求更新为 **0.378 / 0.387 ms**。冷页排序约占 Stream Begin 的 29–30%；联合回收整体约占 50%。起点几乎没有排序和卸载操作，但仍逐帧扫描 CLAS 页、活跃页和驻留需求，所以总成本不会自动降为零。

![CPU Stream Begin 分阶段组成](E:/metallic/build-release/stream-begin-profile/analysis-v2/CpuStreamBegin.png)

代码与数据共同指向三项后续工作：

1. `prepareEvictionCandidates()` 每帧重建候选并排序，比较器反复通过 pages_.at 读取 lastUsedFrame。优先考虑保持既有 age/page-id 次序的增量候选结构，或缓存排序键、按实际回收需求减少排序工作。先保持回收语义，再验证相同轨迹的装卸与质量。
2. `MeshletStreamCompactClasPool::beginFrame()` 即使没有退休页也扫描整个 CLAS page map。可单独维护到期退休队列，避免扫描所有活跃 CLAS。
3. `consumeGpuRequests()` 的驻留需求更新和回收阶段的 pending-free 抵扣都扫描集合。后续可用增量计数/标记减少重复扫描，但必须保留延迟反馈与新上传页面的保护条件。

这次没有把计时变化称为性能优化。两轮起点存在明显波动，CPU elapsed time 会受系统调度、缓存和内存访问影响；移动段候选排序的重复测量较一致。父项减去九个直接子项，起点与 forward_2 的未归属时间约为 7–11 μs，包含观测发布与调用开销，不能当作整个 instrumentation 开销的精确 A/B 测量。嵌套项不可重复相加。

**验证和数据**

- Release 构建通过：MetallicRhiTests、MetallicGPUDrivenSample。
- 最终版本 m1/m2/quality 各 3,000 帧通过；检查 CPU-only 子项无 GPU timing、parent 在子项之前、时长非负、直接子项耗时不超过父项，且后续 GPU scope 均能正确解析。
- 两轮末尾均为 11,598 驻留页，CLAS pending=0；最终 visibleOverTargetRefinements=0，保持原先 1.5 px 收敛条件。quality 的诊断读回不进入性能对比。
- [结构化摘要](E:/metallic/Documentation/MiniZorahStreamBeginResults.json)、[完整 scope CSV](E:/metallic/build-release/stream-begin-profile/analysis-v2/Scopes.csv)、[完整分析](E:/metallic/build-release/stream-begin-profile/analysis-v2/Evidence.json)、[运行 manifest](E:/metallic/build-release/stream-begin-profile/replay-v2/Manifest.json)。manifest 保存运行二进制与相机输入的 SHA-256；原始基准产物未覆盖。

复测命令（使用新目录）：

```powershell
& E:/metallic/Tools/RunMetallicCfgReplay.ps1 -Replay E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Replay.json -OutputRoot E:/metallic/build-release/stream-begin-profile/replay-next
python E:/metallic/Tools/AnalyzeMiniZorahCfgRoam.py E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1 --metallic E:/metallic/build-release/stream-begin-profile/replay-next --output E:/metallic/build-release/stream-begin-profile/analysis-next --plots
```

关键实现：[CPU recorder](E:/metallic/Source/Runtime/Render/Profiling/CpuProfile.h)、[Stream Begin](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2297)、[回收候选](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:1332)、[CLAS 退休扫描](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamCompactClasPool.cpp:297)。
