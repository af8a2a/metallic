# ZorahFull：请求合并复用与按需优先选择

日期：2026-09-23。

## 实现

- 请求合并索引改为按需分配的 4 KiB 数组块（每页 4 字节），不再逐批 clear/reserve unordered_map、创建和销毁哈希节点。槽位只有在当前 scratch 向量的对应位置仍为该页时才有效；不依赖帧 epoch，避免回绕与旧批次误命中。reset 销毁索引块。scratch 与空闲任务向量交换并复用容量。
- 重复请求继续合并最大 screen benefit，真实需求覆盖 prefetch；异常 ID、NaN、短优先级列表语义保持。latency 跟踪移至合并后，每个唯一页处理一次。
- 在刷新完整批次的需求/收益后，已驻留、上传中、已排队页面跳过准入；待卸载页保持原失败统计。全部需求先刷新，再取消过期队列与选择冷页，避免误取消仍被请求的工作。
- 以优先堆替代整批 stable_sort；比较顺序仍为 demand 优先、prefetch LOD 优先、收益/字节及既有 aging、page ID 决胜。只弹出实际进入准入的条目；无优先级旧路径保持逆序消费。
- I/O、预取保留容量或预算耗尽时，线性过滤暂不可执行的剩余请求；保留碎片空间能容纳的小页，并重建剩余堆。容量 gate 仍由真实失败建立，不会提前跳过必要的冷回收；下一帧及已有容量/冷页事件继续触发重新评估。
- 本次未改变 GPU 请求协议、LOD、页面预算、冷回收年龄或跨帧需求丢弃规则。没有固定 Top-K 截断，没有把未执行条目假装为已上传；需求仍由既有 feedback/latency 机制保活与重试。
- Full JSON 导出增加 requestDuplicatesMerged、admissionCandidates、admissionPriorityPops、admissionCalls、admissionBypassed。admissionBypassed 包括已分配/在途页面和被容量 gate 延后的页面，不能当作成功准入数。

## 正确性验证

Release MetallicGPUDrivenSample / MetallicRhiTests 构建通过。

20 项相关测试通过，验证日志没有 FAILED、SKIPPED、VUID、Validation Error 或 error。覆盖原有优先级、预取、延迟反馈、最新批次、冷回收、碎片、小页补位、预算抑制、上传完成、取消/有序发布及 CLAS/BLAS 生命周期。

新增 streamer_meshlet_request_selection：与完整排序的独立优先级 oracle 对照；prefetch/demand 重复及 max benefit；无效 ID；容量仅容纳三个候选时只进行三次成功分配和一次预算失败，不对剩余条目继续堆弹出/准入；重复批次无新增准入调用；下一帧重新检查容量；多轮输入顺序变化与 reset 不复用旧索引。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*streamer_meshlet*:*streamer_joint_cold_reclaim:*streamer_ordered_publication_retry:*stream_clas_runtime_lifecycle:*stream_clas_eviction_reupload:*stream_blas_cut_cache" --output-dir build-release/request-selection-tests
```

构建日志：build-release/request-selection-build.log；回归日志：build-release/request-selection-tests.log。

## 三轮 Full 结果

三轮固定路线、输出 1797×660 / 内部 1198×440、DLSS Quality、LOD 1.5 px，10 秒预热、30 秒采样。已逐项验证与 readiness-cache 基准的 config、absoluteKeyframes、graph、输出/内部尺寸、hidden、vsync、validation 设置一致。GPU/shader/资产保持同一配置，后台环境不视为完全相同。

| 轮次 | Consume 旧均值 ms | 新均值 ms | 降幅 | 新 P95 ms | 准入调用/帧 | 上传页/秒 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.654 | 4.694 | 17.0% | 6.772 | 46.55 | 988.3 |
| 2 | 5.584 | 4.568 | 18.2% | 6.887 | 44.11 | 983.6 |
| 3 | 5.984 | 4.739 | 20.8% | 6.690 | 44.59 | 989.9 |

每帧约 1.35 万请求中只对约 44–47 个条目执行优先堆弹出与 requestPage；预算阻塞尾部不再逐条重复准入。该场景 GPU 输入的重复计数为零，CPU 仍保留廉价合并/验证以支持其他入口与预取/demand 合并。第一轮 Validate / merge loads 从 1.975 ms 降到 0.257 ms，但旧 scope 包含 latency，因此不能把这一差额全部归因于去重：新 Track merged demand 仍为 1.155 ms，Deduplicate loads 父区间约 1.413 ms。Prepare admission heap 约 0.120 ms，另有 Filter blocked admission 约 0.314 ms，不把它们隐藏在排序收益之外。

未达到之前设定的 Consume 均值 <2.5 ms / P95 <4 ms 目标。剩余成本包括 merged latency tracking、resident demand 全表刷新，以及候选刷新/回收调度；本次没有将这些工作移出计时冒充消失。

原有吞吐约 986–993 页/秒，本轮 984–990 页/秒，基本保持；无加载失败或请求溢出。所有采样帧 HZB 有效、drainReasonMask=0、GPU timing 完整。geometry 仍为 3.5 GiB 预算，BLAS overflow 最高仍为既有的 12425。未专门导出新的逐页延迟分布，所以不能声称请求 P99 尾延迟已改善。

**整帧未改善，不能宣称达到稳定 30 fps。** 新三轮整帧均值 23.40 / 22.25 / 22.36 ms，P95 33.87 / 32.90 / 29.11 ms；3974 帧中 149 帧超过 33.33 ms（3.75%）。旧三轮为 21.22 / 21.03 / 22.02 ms、超预算比例 0.62%。分项 CPU 改善是真实观测，但整帧历史对照不是完全隔离的 A/B，不能将整帧变化全部归因于本次改动。

## 未覆盖的帧起始等待

核对第一轮 Frame 子区间发现约 3.8 ms 未归属，其路径包含 StreamlineFrameScope 构造。随后仅给 Full benchmark 的该调用加上 Streamline frame begin / Reflex pacing CPU scope，重新构建最终样例并补跑一次同路线 30 秒验证；未改变 Reflex 模式、sleep、frame-slot 或 GPU 同步。

补测该区间均值 **3.856 ms**，P95 **18.066 ms**，最大 **21.196 ms**；Consume requests 仍为 **4.682 ms**。VulkanStreamline.cpp 构造路径包括 slReflexSleep、标记及偶发状态更新，因此这个计时是整个帧起始调用，不能当成精确的单独 sleep API 时间。它解释了大部分此前无子计时覆盖的 Frame 开销；需要后续区分 pacing 策略/标记与其它工作，不能直接删除等待。

补测整帧均值 23.18 ms、P95 34.36 ms，1295 帧中 96 帧超预算。它用于归因，与前三轮分开列示。最终二进制相比前三轮仅增加此 CPU scope，请以对应 Manifest 的 hash 区分。

## 证据与复现

- [结构化结果](ZorahFullRequestSelectionResult.json)：三轮条件核对、前后 Consume 子项、计数、吞吐、补测等待统计及各自 Manifest。
- 正式数据：build-release/full-request-selection-0923/run1..3，Comparison 与 SlowFrames 汇总。
- 补测：build-release/full-request-selection-pacing-0923/run1。
- 新计时构建：build-release/request-selection-pacing-build.log。最终样例已构建；此前 20 项回归对应相同的 runtime/request-selection 代码，之后只加了 benchmark CPU scope。

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```
