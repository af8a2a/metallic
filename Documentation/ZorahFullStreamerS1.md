# ZorahFull Streamer S1：延迟跟踪降本

2026-09-24，基于 e84464e01（profile marker 修正后）。S1 核心实现完成并通过回归，**合计均值 ≤0.5 ms 的性能验收目标尚未达到**。没有关闭 measurePageLatency，也没有省略预算阻塞需求。

## 实现

- 合并反馈批次只读一次时钟；保留 sourceFrame→源时间环和第一次 demand 时间。反馈时间统一为批次开始，相比原来的逐请求取时有批内精度变化，不代表 GPU 产生请求的精确时刻。
- pending 哈希表改为按需分配的 1024 项页索引块和可复用记录槽；准入、I/O、上传阶段直接查槽。索引块与槽容量保留至场景生命周期结束，按触及页面块和历史 pending 峰值增长，不为全部 Full 页面分配 Request 对象。
- 256 桶时间轮替代 beginFrame 每帧全表扫描。重复需求只更新 lastSeenFrame；旧事件到期时续期。真正过期时才查页面是否在途，在途记录下一帧重查，保留原来的取消和尾延迟行为。
- 每条记录只有一个可直接移除的事件；完成/放弃先摘链再复用槽。因此采用同步取消保证生命周期安全，没有另外保留可能过期的事件或 generation tag。
- 诊断 snapshot 仍遍历有效槽以统计 pending 和 oldest；没有把该扫描加入每帧路径。每个合并请求的 pages_ 驻留/fallback 检查仍保留，是下一步可复用查询的剩余成本。

实现入口：[tracker](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamLatency.h:90)、[到期处理](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:386)、[批量时间](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:936)。本次只修改 Streamer 与回归测试，未改变预算、页面准入、shader 或 RenderPass 职责。

## 验证

Release MetallicGPUDrivenSample、MetallicRhiTests 构建成功；9/9 项聚焦回归通过，Vulkan validation 关闭。覆盖请求选择、prefetch、预算、上传完成、发布重试、demand cache 和联合冷回收。

新增 [生命周期回归](E:/metallic/tests/rhi/StreamerTests.cpp:2216) 使用确定时间逐帧对照原全表扫描算法，检查 pending、abandoned、首次需求/反馈时间、完成直方图和帧延迟；覆盖预取升级、重复需求、稀疏页 ID、槽复用、在途保护、时间轮回绕、超过轮长的保留期限和跳帧。持续刷新 14000 条预算阻塞需求的 300 帧中，不调用 residency 过期判定；停止刷新后到期一次性回收，统计不丢失。

Full Frames 导出未包含页面 demand→drawable 直方图，故本次真实 Full 尾延迟分布尚未完成前后实测验收；统计语义由确定时间的旧算法对照和已有生命周期测试验证。没有通过每帧导出完整 snapshot 引入额外扫描。

## 同配置 Full 漫游

RTX 5070 Ti 16 GB，616.92；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、SW 8 px。每轮预热 10 秒、采样 30 秒；Reflex On，隐藏窗口、VSync、2 frame slots。核对 graph、相机关键点、配置、分辨率和 shader SHA256 均与四轮 marker 修正后基线相同。流送和 cut 未冻结，因此逐帧请求与驻留并非强制重放一致。

下表保留所有轮次；延迟合计先逐帧加 Reset / latency tracking 与 Track merged demand，再计算分位数。CPU-only 区间没有 GPU 计时。

|版本/轮次|Reset 均值 ms|Track 均值 ms|合计均值 / P95 ms|Stream Begin 均值 ms|上传页/秒|整帧 P95 ms|>33.33 ms|Sleep 均值 ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|基线 1|0.895|1.354|2.250 / 3.460|6.432|981.3|34.490|112|13.282|
|基线 2|1.079|1.722|2.801 / 4.111|7.837|989.6|29.125|8|4.694|
|基线 3|0.988|1.609|2.598 / 4.073|7.416|985.5|28.397|10|3.893|
|基线 4|0.810|1.352|2.162 / 3.217|6.250|981.1|34.517|123|13.425|
|中间版（每帧重挂） 1|0.010|0.888|0.898 / 1.631|5.117|992.9|26.689|0|6.159|
|中间版（每帧重挂） 2|0.008|0.735|0.743 / 1.254|4.564|993.7|26.224|0|6.885|
|中间版（每帧重挂） 3|0.010|0.864|0.875 / 1.545|4.887|981.1|33.770|80|10.341|
|最终版（惰性续期） 1|0.132|1.164|1.297 / 2.034|6.974|974.6|35.282|133|8.872|
|最终版（惰性续期） 2|0.124|0.913|1.037 / 1.604|5.748|978.7|34.629|119|12.545|
|最终版（惰性续期） 3|0.119|0.858|0.976 / 1.537|5.388|978.4|34.467|119|13.896|

最终版相关 CPU 区间为 **0.98–1.30 ms/帧**，此前 **2.16–2.80 ms/帧**，但仍未达到 0.5 ms。最终版请求约 1.36 万/帧、驻留约 6.21 万页；上传约 **975–979 页/秒**，此前约 981–990，差约 0.2%–1.5%，没有出现吞吐量级下降。三轮 loadFailures / requestOverflows 都为 0，GPU timing 无缺失、HZB 全部有效、snapshot 未冻结。已有 BLAS overflow 最大 12425 仍存在，未在本次处理。

本轮不是严格隔离的交错 A/B：最终版采样期间观察到两个 Edge 进程各约 81% 的单核等效 CPU 使用率，未修改的 Update resident demand、Visibility prepare 等也变慢；GPU 进程遥测为空，不能证明没有后台 GPU 竞争。中间版与最终版不能仅凭均值判断惰性续期的性能收益，最终版的确定收益是减少持续请求的每帧重挂操作；应在隔离负载后再交错验证实际收益。

最终版三轮整帧 P95 为 34.47–35.28 ms，仍有超预算帧。Reflex Sleep 均值 8.87–13.90 ms，不能把 CPU 跟踪节省换算成等额 FPS 提升；本次不宣称达到持续 30 fps，也不以中间版两轮无超预算帧作为最终验收。

## 后续

优先复用合并需求的 pages_ 驻留/fallback 查询，把 Track merged demand 剩余 0.86–1.16 ms 继续降下来；先明确 resident、fallback、取消和重试的缓存失效边界，不能直接以“已有 pending”跳过驻留检查。然后按 S2/S3 推进 resident 冷热增量维护和跨帧候选缓存。正式达标前仍需空闲环境下的交错复测，以及采样区间边界的 Full 延迟快照对照。

## 证据与复现

- [结构化结果](E:/metallic/Documentation/ZorahFullStreamerS1Result.json)，包括四轮基线、三轮中间版、三轮最终版、条件逐项比对及安全计数。
- 最终构建日志：`build-release/streamer-s1-lazy-build-0924.log`；回归：`build-release/streamer-s1-lazy-tests-0924.log`。
- 最终采样：`build-release/full-streamer-s1-lazy-0924/run1..3`；中间采样：`build-release/full-streamer-s1-0924/run1..3`。
- 汇总脚本：`build-release/AnalyzeStreamerS1.py`，直接读取 Capture / Frames / Summary / Manifest，可重算同帧合计。

```powershell
$env:METALLIC_REFLEX_MODE='on'
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-dir> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```
