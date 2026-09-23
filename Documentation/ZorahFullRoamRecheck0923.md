# ZorahFull 复测与下一步优先级

2026-09-23。已重新构建当前提交 `c4f78f70e`，连续完成三轮 Full 固定路线。结论：下一步先优化预算耗尽后的 CPU 请求处理，并拆解 Preflight/其他 CPU 热点；暂不优先继续重排 SW 光栅工作。

## 条件与结果

RTX 5070 Ti，驱动 616.92；编辑器输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px、8 px 软硬分流。完整相同 graph、相机关键帧、VSync、隐藏编辑器；每轮 10 秒预热后采样 30 秒，关闭 validation 和诊断工作量重放。三轮条件及运行期间 executable/shader hash 一致。

| 轮次 | 帧数 | 均值 ms | 平均 fps | P95 ms | P99 ms | 最大 ms | >33.33 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 842 | 35.64 | 28.1 | 47.02 | 51.83 | 85.13 | 59.4% |
| 2 | 793 | 37.83 | 26.4 | 48.37 | 55.59 | 79.35 | 75.3% |
| 3 | 980 | 30.63 | 32.6 | 40.58 | 48.98 | 66.46 | 23.5% |

总计 2615 帧，1327 帧超过 33.33 ms，三轮均未满足持续至少 30 fps。不能再以此前单轮 28.30 ms 作为稳定性能结论。

本轮仍有桌面/浏览器等后台活动，CPU 负载没有锁定。采样结束后的 CPU 快照也显示后台占用；该快照不是采样期间完整 CPU 时间线，不能定量归因到某个进程。GPU per-engine 监控存在一次超过 100% 的异常读数，不能据此声称有超量 GPU 竞争。原始监控已保留。本轮应作为当前环境下的复测范围，不能将与 9 月 22 日的差异直接称为代码退化。

## 已有修复没有回退

- HZB：2615/2615 帧有效。
- `drainReasonMask`：2615 帧均为 0；外部 completion 数均为 1。
- 输入前帧槽等待均值 0.036–0.069 ms，Graph submission slot wait 0.025–0.027 ms。
- GPU timing 无缺失；运行完整结束，日志无 DeviceLost。
- 文件加载失败、GPU 页面请求队列 overflow 均为 0。

## 慢帧更偏向 CPU

| Scope 均值 ms | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|
| Graph CPU 录制/提交 | 24.96 | 26.82 | 21.79 |
| Graph GPU envelope | 20.44 | 19.88 | 21.03 |
| 请求消费 CPU | 9.13 | 9.33 | 7.81 |
| Validate / merge loads CPU | 3.14 | 3.01 | 2.71 |
| Admit demand / prefetch CPU | 3.07 | 3.21 | 2.39 |
| 排准入优先级 CPU | 0.85 | 0.91 | 0.87 |
| 更新 resident demand CPU | 0.83 | 0.84 | 0.71 |
| Joint cold page reclaim CPU | 0.006 | 0.007 | 0.005 |
| Preflight CPU | 4.84 | 4.63 | 3.58 |
| Shadows pass CPU | 4.25 | 4.68 | 3.71 |
| Shadows pass GPU | 0.045 | 0.043 | 0.045 |
| Viewport Panel CPU | 4.18 | 4.56 | 3.76 |
| Texture streaming CPU | 1.15 | 1.20 | 1.14 |

这是 inclusive scope 表，父子项不能累加，CPU/GPU 也不能相加为整帧。自动分析只对 CPU 叶子 scope 排名，不对不同来源的嵌套时钟强行相减估算 exclusive 时间。

超预算帧中请求消费均值为 10.23/10.22/9.99 ms，而这些帧的 GPU envelope 均值为 20.29/19.93/21.48 ms。GPU 时间没有随慢帧等比例上升，支持先处理 CPU 的方向；这些是同帧统计相关性，不是 Nsight critical-path 证明。

转向、近墙和返回阶段尤其容易超预算。冷页调度本身已很小，继续主要投入 cold-candidate 筛选的收益有限。独立上传均值仅 0.029/0.032/0.029 ms，也不是当前首选。

## 第一优先：预算耗尽时的请求背压与重试抑制

三轮几何池容量都为 3.5 GiB，使用量持续接近上限。每帧请求均值约 1.35–1.36 万，`allocationFailures` 均值约 1.33–1.34 万，最高 16245；实际上传均值只有约 30–36 页/帧。这个计数在上次 28.30 ms 的捕获中也已存在。

源码 [MeshletStreamResidency.cpp](../Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp) 表明，这不等于 Vulkan OOM：当没有可用冷页/容量、卸载调度失败或分配器无空间时都会增加该计数。其中 `frameAllocationDeferredCount` 和 `frameResidentBudgetFailureCount` 能进一步区分原因，但目前普通漫游导出没有这些细项。

`consumeGpuRequests()` 每批清空 `requestMarks_`（unordered_map），遍历并合并 load 请求；`consumeReadyRequestTasks()` 排序后逐条调用 `requestPage()`。尽管耗尽后的冷页扫描已是常数时间，依然存在大量不可能成功的逐页处理。根据计数与代码，这很可能是持续重试造成的重复开销；当前数据未导出逐页 ID，尚不能量化跨帧相同请求比例。

建议按以下顺序实施：

1. 导出预算拒绝、碎片/最大空闲块、延迟释放、逐帧唯一/重复请求、实际准入数。先确认拒绝原因，不把所有失败都归为显存 OOM。
2. 对已知因容量失败的请求建立延后集合；空闲容量/可回收页状态变化、冷页到期或需求优先级提升时再重试，并保留有界最长等待，防止饥饿。
3. 限制每帧准入尝试工作量，按屏幕收益和实际可容纳大小选择候选。不能在第一个大页失败后直接跳过后面可放入的小页。
4. 改用复用容器、分块 stamp/稀疏索引去重，减少 hash 节点反复分配；缓存未变化的请求优先级，避免每帧完整重排。

建议阶段验收目标：请求消费均值降至约 3–4 ms、P95 降至约 5–6 ms，同时保持 1.5 px 配置、页预算和根页覆盖；额外验证无请求饥饿、请求到驻留尾延迟不恶化。这里是工程目标，不是已测得或保证可获得的收益。

## 第二优先：拆解 Preflight 与非渲染 CPU 成本

Preflight 均值 3.58–4.84 ms、慢帧内 4.68–5.54 ms，已经与部分 GPU 子阶段同量级。

[RenderGraphExecutor.cpp](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp) 的区间包含队列/合约检查、外部 completion 的 `isComplete()` 查询、scene stamp 获取和容器维护。外部完成点仅一个，不能因为该区间慢就认定存在大量对象累积，也不能直接将它全部解释为 GPU 等待。

先对这些操作加子 scope，确认 timeline 查询是否成为驱动端开销，再考虑合并查询或利用已知提交依赖清理。仍须保留 resize、场景切换、取消和销毁的生命周期测试。

Shadows CPU 3.71–4.68 ms，而 GPU 约 0.045 ms，另应拆解灯光记录构建、资源/descriptor 准备和 `shadows_.record()`。Viewport Panel CPU 3.76–4.56 ms 应单独拆 UI 和资源登记。现有证据不能把这些 CPU 成本归因到阴影射线数量或 SW 原子写入。

## 与性能并行的质量门槛：BLAS overflow 分因

三轮 `blasOverflowCount` 最大均为 12425，非零帧分别为 658/842、610/793、786/980。该字段来自 GPU BLAS header，与页面请求队列 overflow 是不同计数。

[GPUDrivenStreamAsset.slang](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang) 会在无效 group、引用容量不足、CLAS 状态不满足或插入越界等多个分支累计此字段；部分分支标记 fallback。当前 aggregate 计数不能确定每种原因，也不能仅凭它认定有内存破坏或画面漏物体。

下一轮需拆开原因，并统计真正动态 BLAS 覆盖的实例比例、fallback 比例；对阴影/遮蔽作图像检查后，才能把性能与目标画质一起验收。暂不以增加全部容量作为解决办法。

## 交付与复现

本次未修改运行时渲染算法，只增加分析工具和报告。

- 原始三轮：`build-release/full-roam-recheck-0923/run1`、`run2`、`run3`。
- 根目录 `Comparison.json/md`、`SlowFrames.json/md`：每轮结果、分阶段慢帧、scope 和最慢帧。
- [精简数据](ZorahFullRoamRecheck0923Result.json)。
- [分析工具](../Tools/AnalyzeZorahFullSlowFrames.py)：读取既有捕获，不引入 GPU 诊断工作。

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
python Tools/AnalyzeZorahFullSlowFrames.py build-release/<new-directory>
```

下一项建议明确为：**先做预算拒绝分因计数和请求背压，再以相同三轮路线验证慢帧改善**。后续持续 30 fps 验收需扩展到更长路线、保持显示与画质条件，并控制后台 CPU/GPU 负载。
