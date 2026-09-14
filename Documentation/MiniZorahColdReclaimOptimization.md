# MiniZorah 到期冷页与 unused 集合优化

已按 [细分报告](E:/metallic/Documentation/MiniZorahStreamRequestDetail.md) 完成两项优化。相同轨迹两轮均值：forward_2 的 CPU Stream Begin 从 **0.882 ms 降到 0.340 ms（下降 61.4%）**；整个非预热区间从 **0.634 ms 降到 0.268 ms（下降 57.7%）**。这里测量的是 CPU Stream Begin，不是 GPU 或整帧加速比例。

## 实现与语义

**冷页只排序当前可能到期的前缀。** 每帧仍收集完整候选，按最小可能年龄分区，只排序到期部分。缓存保留未到期后缀；同帧预算压力出现或分配器需要更年轻的候选时，补排后缀并合并到已有排序。保留原来的 `lastUsedFrame → page ID` 顺序、一次驻留扫描、256 页上限和延迟释放规则。

联合回收只访问到期前缀。在 CLAS 大小查询之前先检查最低可能年龄和最近上传保护。每成功回收一页后重新判断压力；若压力解除，利用有序的候选快照停止处理未达到保留期的后缀。实际页年龄仍单独复核，避免同帧请求刷新后错误回收。CLAS 压力下零 CLAS 字节的页仍使用保留期规则；压力年龄与保留期大小关系不作额外假设。

**unused 哈希集合替换为位图。** 每个逻辑页一位，首次设置某个字时记录其索引，下一批只清空这些字。去重保留原输入顺序，先检查 page ID 边界再访问位图。完整、空、迟到和截断反馈的需求判定不变。MiniZorah 有 1,356,959 个逻辑页，位图占 **169,624 字节（约 166 KiB）**；全部触及字的索引有效数据最多 84,812 字节，另外保留向量容量。它占用 CPU 内存，不增加 GPU 几何或 CLAS 池。

## 同条件实测

先保留优化前的细分版程序并重新采集 m1/m2，随后构建优化版再跑 m1/m2/quality。每轮 3,000 帧，沿用 `20260914-idle-v1/Replay.json`，1920×1080、1.5 px、几何 1 GiB、CLAS 512 MiB、异步计算。每轮新进程，保留 cook、OS 文件缓存和 PSO 缓存；质量轮单独开启 validation 与读回。两种程序的相机输入和基准测试源码哈希一致。

下表是 forward_2 每段 300 帧的 CPU 均值，单位 ms；父子 scope 有嵌套，不能相加。

| scope | 优化前 m1 / m2 | 优化后 m1 / m2 |
| --- | ---: | ---: |
| Stream Begin | 0.9791 / 0.7857 | 0.3670 / 0.3136 |
| GPU request feedback | 0.4111 / 0.3473 | 0.1418 / 0.1229 |
| unused 集合准备 | 0.0395 / 0.0333 | 0.0010 / 0.0008 |
| unused 页校验/插入 | 0.0894 / 0.0865 | 0.0096 / 0.0069 |
| 驻留需求更新 | 0.2267 / 0.1760 | 0.0806 / 0.0704 |
| Joint cold page reclaim | 0.3750 / 0.2751 | 0.0612 / 0.0480 |
| 候选扫描 | 0.0661 / 0.0538 | 0.0535 / 0.0418 |
| 候选分区/排序 | 0.1100 / 0.1099 | 0.0014 / 0.0011 |
| 冷页筛选/调度 | 0.1977 / 0.1105 | 0.0053 / 0.0045 |

其中 unused 集合准备与插入合计均值从 0.1244 ms 降到 0.0092 ms。[耗时对比图](E:/metallic/build-release/stream-cold-opt/analysis/CpuComparison.png) 的横条是两轮均值，刻度是各轮均值。CPU elapsed time 包含抢占和调度影响，两轮之间存在波动，不能将单个数值视为固定性能。

forward_2 每帧工作量：

| 计数 | 优化前 m1 / m2 | 优化后 m1 / m2 |
| --- | ---: | ---: |
| 收集的冷页候选 | 1953.503 / 1953.530 | 1953.383 / 1953.110 |
| 调度循环访问 | 1953.503 / 1953.530 | 11.643 / 11.643 |
| CLAS 大小查询 | 1953.503 / 1953.530 | 11.643 / 11.643 |
| 调度循环中的年龄拒绝 | 1941.860 / 1941.887 | 0 / 0 |
| 实际保留期回收 | 11.643 / 11.643 | 11.643 / 11.643 |

CLAS 查询与调度循环访问均减少 **99.4%**。未到期页在连续数组分区时已排除，不再进入逐页调度；这不表示候选收集或所有年龄比较都消失了。驻留需求仍约 13,078 页/帧，候选收集仍约 1,953 页/帧，这些扫描留待后续根据收益决定是否增量化。

## 正确性与验证

- Release sample 和 RHI tests 构建通过。
- 6 项 validation 测试通过：需求缓存、联合回收、延迟回收/年龄保护、CLAS 回收重载、CLAS runtime 生命周期、紧凑 CLAS 生命周期。
- 新增/加强检查：重复位去重、所有资产页字及尾字、越界 ID、触及字清空复用；空/非空到期前缀扩展；同帧出现 CLAS 压力；一次回收后压力解除；待释放空间抵扣；年龄与 ID 排序。
- 优化后 **9,000 帧**回放通过，包括 **3,000 帧 Vulkan validation**。逐帧需求/回收计数分类和 CPU scope 层级检查通过。
- 四轮性能回放最终一致：几何/CLAS 驻留页均 11,598，几何 **251.967 MiB**、CLAS **255.635 MiB**，CLAS pending/retiring 为 0，可见超 1.5 px refinement 为 0。
- 全程上传页：优化前 36,383 / 36,385，优化后 36,385 / 36,385；回收页：24,785 / 24,787 → 24,787 / 24,787。无无效请求或页加载失败。异步完成时序允许少量事件差异。
- 质量轮运动检查点 forward_1 / forward_2 仍各有 2 / 1 个暂未收敛项，停留与最终检查点收敛。该轨迹没有预算压力回收；压力分支的正确性由针对性测试覆盖，未据此宣称压力路径的同等性能收益。

## 复测与证据

[汇总 JSON、源码哈希和二进制哈希](E:/metallic/Documentation/MiniZorahColdReclaimOptimizationResults.json)。原始回放位于 `build-release/stream-cold-opt/before` 和 `after`；修改前程序保留在 `before-runtime`。`analysis/Evidence.json` 是完整前后 CPU 对比；`detail-analysis` 包含当前 CPU scope 与工作量 CSV/图表。运行日志和构建日志也保留在同一根目录。

```powershell
Tools/RunMetallicCfgReplay.ps1 -Replay E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Replay.json -OutputRoot E:/metallic/build-release/stream-cold-opt/recheck
python Tools/CompareMiniZorahStreamBegin.py build-release/stream-cold-opt/before build-release/stream-cold-opt/after --replay build-release/vk-minizorah-roam/20260914-idle-v1/Replay.json --output build-release/stream-cold-opt/analysis --plots
python Tools/AnalyzeMiniZorahCfgRoam.py build-release/vk-minizorah-roam/20260914-idle-v1 --metallic build-release/stream-cold-opt/after --output build-release/stream-cold-opt/detail-analysis --plots
```
