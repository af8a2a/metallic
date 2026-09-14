# MiniZorah CPU Stream Begin 优化结果

已完成 [CPU Stream Begin 细分报告](E:/metallic/Documentation/MiniZorahStreamBegin.md) 中的首轮优化。在相同相机输入与设置下，去掉预热后的 Stream Begin CPU 均值从 **1.742 ms 降到 0.660 ms（下降 62.1%）**；forward_2 阶段从 **2.447 ms 降到 0.807 ms（下降 67.0%）**。这是该 CPU scope 的耗时变化，不代表整帧或 GPU 加速比例。

## 实现

- 冷页候选收集时保存 `lastUsedFrame` 与 page ID，比较器直接比较值，消除排序中的重复哈希查找。仍按最后使用帧升序、同龄页按 ID 升序排序，保留每帧一次扫描、256 页卸载上限及现有年龄保护。
- CLAS 退休改为到期队列。每帧只处理已到期记录，通过页状态和当前退休期限排除复用后的旧记录；再次退休必须等待新的期限。固定 queued-frame 延迟保证队列期限单调。
- 联合回收的待释放空间抵扣改为遍历现有卸载任务页列表，核对状态与 task index，避免扫描所有 active pages。
- 驻留表维护对应的 `PageEntry*` 列表，在需求更新和冷页候选扫描时直接访问节点。随驻留表的 swap-remove 同步更新，擦除 map 节点前移除指针；rehash 不会使 unordered_map 节点地址失效。需求判定仍逐页执行，延迟反馈、新上传页、截断反馈的保护条件保留。

未改变 1.5 px 阈值、显存预算、预取策略或回收年龄。到期释放的遍历顺序从哈希表顺序变为队列顺序，物理分配地址不要求与旧版相同。

## 同条件测量

基准版本为 `379600d42e128297df44ff9c87f6feea144aee15`。修改前复制可执行文件与运行库，完成 m1/m2；随后构建修改版并完成 m1/m2/quality。每轮新进程、新 GPU 驻留状态，保留已有 cook、OS 文件缓存和持久化 PSO 缓存。

沿用 vk_lod_clusters cfg 基准生成的 Replay.json：1920×1080、1.5 px、CLAS 开启、异步计算开启，几何预算 1 GiB、CLAS 预算 512 MiB。每轮 3,000 帧，预热 300 帧后前进 12 个场景单位、停留、返回；路径总长 24 单位。每段相机位置按固定帧推进，无限帧运行。分析脚本校验 Replay SHA、测试源码 SHA、全部执行相机、帧索引和阶段。

下表是每段 300 帧的 CPU 均值；“整体”合计每轮预热后 2,700 帧。m1/m2 均关闭 validation，quality 单独开启 validation 与检查点读回，不参与性能均值。

| 阶段 | 优化前 m1 / m2 (ms) | 优化后 m1 / m2 (ms) | 两轮均值下降 |
| --- | ---: | ---: | ---: |
| 起点停留 | 0.789 / 0.593 | 0.125 / 0.125 | 81.9% |
| forward_2 | 2.865 / 2.029 | 0.772 / 0.842 | 67.0% |
| 返回停留 | 0.987 / 1.001 | 0.276 / 0.224 | 74.9% |
| 整体，去掉预热 | 1.870 / 1.614 | 0.663 / 0.657 | 62.1% |

forward_2 的热点拆分如下，均为两轮均值。子项嵌套在总计内，不能将这些行相加。

| CPU scope | 优化前 (ms) | 优化后 (ms) |
| --- | ---: | ---: |
| Stream Begin 总计 | 2.447 | 0.807 |
| Expire retired CLAS | 0.480 | 0.014 |
| Credit pending frees | 0.162 | <0.001 |
| Scan resident candidates | 0.209 | 0.055 |
| Sort cold candidates | 0.689 | 0.107 |
| Update resident demand | 0.407 | 0.179 |

[耗时对比图](E:/metallic/build-release/stream-begin-opt/analysis/CpuComparison.png) 中横条为两轮均值，短刻度为各轮均值。计时使用 steady-clock elapsed time，包含线程抢占；两个串行样本的范围不是置信区间。优化前两轮存在明显波动，但优化后两轮均低于优化前两轮。其余 8 个非预热阶段也均下降，详见结果 JSON。

## 回收与质量验证

四轮性能回放的最终结果完全相同：几何 **251.967 MiB**、CLAS **255.635 MiB**，驻留几何/CLAS 页均 **11,598**，CLAS clusters **143,820**，CLAS pending/retiring 均为 0。最终 cut 字段一致，可见超出 1.5 px 的 refinement 为 0，没有 capacity fallback。

| 全程累计 | 优化前 m1 / m2 | 优化后 m1 / m2 |
| --- | ---: | ---: |
| 上传页 | 36,381 / 36,385 | 36,387 / 36,385 |
| 回收页 | 24,783 / 24,787 | 24,789 / 24,787 |
| 上传字节 | 1,216,542,544 / 1,216,733,904 | 1,216,808,976 / 1,216,698,224 |
| 无效请求 / 页加载失败 | 0 / 0 | 0 / 0 |

优化后最大上传量相对优化前最小值增加约 0.022%。相机输入一致，但异步 I/O 的完成时间可以改变少量页面事件，因此不要求逐帧请求量完全相同。

- Release 的 `MetallicGPUDrivenSample` 与 `MetallicRhiTests` 构建成功。
- 6 项聚焦测试通过：`streamer_joint_cold_reclaim`、`streamer_meshlet_residency_eviction_delay_age`、`streamer_meshlet_demand_cache`、`stream_clas_eviction_reupload`、`stream_clas_runtime_lifecycle`、`clas_compact_lifecycle`。其中新增了反向插入、同龄 ID 排序、年龄优先、重复卸载/重载检查，并加强旧退休记录不能提前释放 CLAS 的断言。
- 修改后 9,000 帧 MiniZorah 回放通过，包含 3,000 帧 Vulkan validation 质量回放、15 个检查点。初始加载检查点仍有未收敛项；运动中的 forward_1 / forward_2 检查点分别有 2 / 1 个可见超阈值项，停留与最终检查点均为 0。这次未承诺流送过程每一帧都完全收敛。
- 原有 vk/Metallic 分析器校验通过，导出 3,674 条 scope 和 810 条内存/计数器记录。本次收益判断采用 Metallic 自身前后对比。

## 证据与复测

- 原始前后回放：`build-release/stream-begin-opt/before`、`after`。
- 保留的修改前程序：`build-release/stream-begin-opt/before-runtime/MetallicRhiTests.exe`。
- [汇总结果与源码哈希](E:/metallic/Documentation/MiniZorahStreamBeginOptimizationResults.json)。
- 完整均值/P95/最大值：`build-release/stream-begin-opt/analysis/Evidence.json`；参考程序对齐数据：`build-release/stream-begin-opt/reference-analysis`。
- 聚焦验证日志：`build-release/stream-begin-opt/focused.log` 中 5 项通过；新增排序测试修正了 fixture 对 `requestPage()` 返回值的误用后，最终通过记录见 `order-final.log`。

```powershell
Tools/RunMetallicCfgReplay.ps1 -Replay E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Replay.json -OutputRoot E:/metallic/build-release/stream-begin-opt/recheck
python Tools/CompareMiniZorahStreamBegin.py build-release/stream-begin-opt/before build-release/stream-begin-opt/after --replay build-release/vk-minizorah-roam/20260914-idle-v1/Replay.json --output build-release/stream-begin-opt/analysis --plots
```

下一步应针对剩余的 GPU 请求消费与冷页调度继续细分：forward_2 中 `GPU request feedback` 约 0.344 ms，`Joint cold page reclaim` 约 0.294 ms。驻留需求更新仍是 O(驻留页数)，候选排序仍是 O(n log n)；本轮只是降低访问成本。若再改为增量 hot/cold 集合或有界候选选择，需要继续验证迟到反馈、年龄阈值跨越及旧候选复用，不能只凭 CPU 耗时降低验收。
