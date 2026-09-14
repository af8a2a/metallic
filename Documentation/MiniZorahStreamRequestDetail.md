# MiniZorah 请求消费与冷页调度细分

本次继续细分 CPU Stream Begin 的剩余热点，未改变请求顺序、页面年龄条件、回收预算或 CLAS 生命周期。新增批次级 CPU scope 和逐帧工作量计数；逐页循环只增加整数计数，没有逐页读时钟或 GPU timestamp。

更新后的程序中，可在 **Profiler → Table → GPUDriven → Stream Begin** 展开；CPU 子项的 GPU 列仍为 `--`。在 **Profiler → Streaming → CPU Request / Reclaim Work** 展开工作量面板，默认折叠。该面板显示当前采样帧的计数，不是 Table 中的历史时间均值。

## 新增观察项

| 上层 scope | 新子项 / 对应计数 |
| --- | --- |
| Deduplicate loads | Prepare load containers、Validate / merge loads |
| Deduplicate unloads | Prepare unused-page set、Validate / insert unused pages |
| Admit request batch | Queue request task、Consume ready request tasks |
| Consume ready request tasks / Consume queued requests | Select latest request task、Apply explicit unloads、Refresh request priorities、Sort admission priority、Cancel stale queued loads、Admit demand / prefetch、Release request task |
| Joint cold page reclaim | 独立的 Evaluate budget pressure，不再混在 Credit pending frees 中 |
| Update resident demand | 访问页数、新于反馈的页、unused 页、刷新页、不完整反馈保护页 |
| Schedule cold evictions | 访问数、状态拒绝、CLAS 大小查询、年龄拒绝、调度失败、预算回收、保留期回收 |

冷页计数表示工作次数，包括同一帧重复调用时的重复访问。`coldCandidates` 是候选构建次数之和；`coldVisited` 是实际进入联合调度循环的访问次数。二者不保证在所有场景下相等。

## 同轨迹结果

仍使用 `20260914-idle-v1/Replay.json`：每轮 3,000 帧，1920×1080、1.5 px、几何 1 GiB、CLAS 512 MiB、开启异步计算。m1/m2 为关闭 validation 的性能轮；quality 单独开启 validation，检查点读回不参与性能均值。校验了输入哈希、执行相机、帧索引与阶段。

forward_2 每轮 300 帧，以下为 CPU 均值，单位 ms。子项嵌套，不应将父项与子项相加。

| scope | m1 | m2 |
| --- | ---: | ---: |
| Stream Begin | 0.8160 | 0.8119 |
| GPU request feedback | 0.3593 | 0.3550 |
| 加载请求容器准备 | 0.0018 | 0.0019 |
| 加载请求校验/合并 | 0.0063 | 0.0061 |
| unused 集合准备（清空、reserve） | 0.0343 | 0.0337 |
| unused 页校验/插入 | 0.0875 | 0.0856 |
| 驻留需求更新 | 0.1868 | 0.1840 |
| 请求优先级刷新 | 0.0016 | 0.0015 |
| admission 排序 | 0.0050 | 0.0050 |
| demand / prefetch admission | 0.0326 | 0.0338 |
| Joint cold page reclaim | 0.2898 | 0.2909 |
| 候选扫描 | 0.0546 | 0.0565 |
| 候选排序 | 0.1104 | 0.1117 |
| 冷页筛选/调度循环 | 0.1238 | 0.1217 |

Map / Unmap feedback、预算压力判断、请求任务选择和过期队列清理均低于 0.001 ms/帧。请求的 admission 排序约 0.005 ms，不是当前主要热点。

工作量解释了耗时来源。forward_2 每帧两轮均值：

- 驻留需求访问 **13,078 页**，其中 **11,112 页刷新使用时间**、**1,953 页标记 unused**、约 **13 页因新于反馈而跳过**。该循环仍对绝大多数热页逐帧更新。
- 冷页调度访问 **1,953 页**，执行同样数量的 CLAS 大小查询；约 **1,942 页年龄不达标**，实际保留期回收 **11.645 页**。年龄拒绝占 **99.40%**。
- 两轮整条轨迹都没有预算压力触发的回收或调度失败。此路径主要在为尚未达到保留期的 unused 页反复排序、查找与检查。

[各阶段工作量图](E:/metallic/build-release/stream-request-detail/analysis/CpuWork.png) 与 [CPU 时间分布](E:/metallic/build-release/stream-request-detail/analysis/CpuStreamBegin.png) 可对照查看。上述结论来自 CPU 耗时和工作量，不能将 0.123 ms 全部归因于 CLAS 查询；该 scope 也包含状态/年龄判断和成功卸载的操作。

## 优化顺序建议

1. **先减少无效冷页筛选。** 在预算压力关闭时，把低成本年龄判定放到 CLAS 大小查询之前。候选已按最后使用帧排序，可进一步评估保留期截止位置，避免遍历/排序整个尚未到期的后缀。最近上传保护、预算压力路径和同帧预算变化必须单独验证，不能遇到任意年龄拒绝就直接退出。
2. **再降低 unused 集合维护和热页刷新成本。** 集合准备/插入合计约 0.121 ms，驻留需求更新约 0.185 ms。比较复用存储的页标记或 epoch 方案，同时保持新上传页与迟到/截断反馈语义，并核算额外 CPU 内存。
3. demand / prefetch admission 约 0.033 ms，放在上述两项之后。当前证据不支持优先优化 admission 排序或 map/unmap。

这次是观测改动，尚未执行以上算法优化。新增计数和 scope 有观测成本；同阶段总计约 0.814 ms，上一轮约 0.807 ms，两轮间差异混合了计数成本和运行噪声，不应据此估算精确开销或宣称加速。

## 验证与证据

- Release sample 与 RHI tests 构建通过。
- 3 项 focused tests 通过：需求缓存、联合冷页回收、延迟回收/年龄保护；补充完整反馈、截断反馈与逐帧计数清零断言。
- 9,000 帧回放通过，包括 3,000 帧 Vulkan validation。逐帧检查需求/冷页分类计数覆盖全部访问，以及 CPU 子 scope 总时间不超过父 scope。
- 最终几何/CLAS 均为 11,598 页，分别占 251.967 / 255.635 MiB；CLAS pending/retiring 为 0，可见超 1.5 px refinement 为 0。运动检查点 forward_1 / forward_2 仍各有 2 / 1 个暂未收敛项，停留与最终检查点收敛。
- [汇总 JSON 与源码/二进制哈希](E:/metallic/Documentation/MiniZorahStreamRequestDetailResults.json)。原始数据位于 `build-release/stream-request-detail/replay`，分析位于 `analysis`；新增 `CpuWork.csv` 提供每阶段 mean/min/max/total，原有 `Scopes.csv` 包含全部 CPU 子项。

```powershell
python Tools/AnalyzeMiniZorahCfgRoam.py build-release/vk-minizorah-roam/20260914-idle-v1 --metallic build-release/stream-request-detail/replay --output build-release/stream-request-detail/analysis --plots
```
