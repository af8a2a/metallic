# ZorahFull 预算不足时的请求准入优化

日期：2026-09-23。实现与三轮 Full 固定漫游验证已完成。

## 行为变化

- 本帧回收候选耗尽、256 页回收额度耗尽或卸载任务无法调度后，记录预算准入已受阻。后续装不下的请求直接延后，不再反复创建/删除驻留哈希节点、进入分配失败和回收准备路径。
- 仍逐请求检查页数预算及实际可分配尺寸；大页失败不能阻止可装入的小页，已释放容量可立即使用。I/O 窗口检查也提前到创建驻留记录之前。
- 不增加按 page ID 的固定退避时间，不保留过时优先级。每帧重评冷页年龄；同帧新增明确的冷页反馈使候选快照失效并恢复准入。最新请求仍按原有屏幕收益排序。
- 根页锁定、热页保护、每帧 256 页回收上限和 GPU 完成后的延迟释放协议继续生效。
- Profiler 的 CPU Request / Reclaim Work 与 Full 导出增加 allocationAttempts、budgetRetrySuppressed。allocationFailures 现在统计实际进入分配路径的失败；提前延后的请求单列，不能把该计数下降解释为缺页消失。

## 同条件三轮

与 `full-roam-recheck-0923` 核对：资产、shader 摘要、路线、显示/渲染尺寸、画质、VSync 和验证配置均一致。RTX 5070 Ti，1797×660 输出，DLSS Quality（1198×440），LOD 1.5 px，10 秒预热、30 秒漫游。

| 轮次 | 之前均值 ms | 之后均值 ms | 之前 P95 ms | 之后 P95 ms | 之前超预算 | 之后超预算 |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 35.64 | 28.38 | 47.02 | 39.56 | 59.4% | 16.7% |
| run2 | 37.83 | 30.71 | 48.37 | 40.31 | 75.3% | 26.2% |
| run3 | 30.63 | 29.31 | 40.58 | 37.75 | 23.5% | 16.6% |

| 指标（三轮均值范围） | 之前 | 之后 |
|---|---:|---:|
| Consume requests CPU | 7.81–9.33 ms | 6.19–7.07 ms |
| Admit demand / prefetch CPU | 2.39–3.21 ms | 1.32–1.59 ms |
| 实际分配失败 / 帧 | 13313–13371 | 1 |
| 实际分配尝试 / 帧 | 旧导出无此计数 | 55–60 |
| 提前抑制预算重试 / 帧 | 未实现 | 13318–13353 |
| 上传吞吐 | 955–974 页/秒 | 968–975 页/秒 |

三轮共 3059 帧，602 帧超过 33.33 ms。全部 HZB 有效，RenderGraph drainReasonMask 为 0，无缺失 GPU 计时、I/O 加载失败或请求溢出。几何容量仍为 3.5 GiB，保持接近满驻留。实际分配尝试包含成功分配和触发延迟回收的尝试，因此并不等于本帧完成上传数。

这是顺序运行的同路线对比，后台负载未完全受控，且时间驱动路线在不同帧率下不会产生逐帧相同的 cut/驻留状态。整帧收益不能全部归因于本改动；重复请求不再分配节点的回归测试、新增工作量计数和维持的上传吞吐是更直接的证据。

## 验证

- Release 的 MetallicGPUDrivenSample、MetallicRhiTests 构建通过。
- 15 项 meshlet / joint cold reclaim RHI 测试全部通过，启用默认 Vulkan validation，日志无 VUID / validation error。
- 强化年龄延迟回收测试：10000 次重复请求仅增加抑制计数；实际分配失败和候选扫描各一次；年龄到期恢复回收、延迟释放后重新加载。
- 新增 budget_admission：高优先级大页失败后低优先级小页仍能加载；根页不被回收；下一帧重新评估；reset/initialize 清除受阻状态。
- 强化 demand_cache：同帧受阻后收到明确冷页反馈即可恢复回收，截断反馈仍保护未明确标记 unused 的页面。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*streamer_meshlet*:*streamer_joint_cold_reclaim*" --output-dir build-release/budget-admission-tests
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
python Tools/AnalyzeZorahFullSlowFrames.py build-release/<new-directory>
```

## 后续边界

本项已消除预算耗尽后的重复分配重试，没有改变去重与全量排序。剩余 Validate / merge loads 为 2.30–2.83 ms、Sort admission priority 为 0.82–0.86 ms，可继续复用请求索引/容器以减少每批哈希节点分配。Preflight 仍需按原计划细分。尚未稳定达到 30 fps；既有 BLAS overflow 聚合计数仍最高约 12425，需另行拆分原因和验收回退质量。

证据：[汇总 JSON](ZorahFullBudgetAdmissionResult.json)；原始三轮 `build-release/full-budget-admission-0923/run1..3`，每轮包含 Capture.json、Frames.jsonl、Summary.json 与 GPU 记录；根目录包含 Comparison 与 SlowFrames 分析。
