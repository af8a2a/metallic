# ZorahFull 合并需求驻留查询优化

2026-09-25。消除 `Track merged demand` 对每个合并请求的 `pages_.find`；改为读取由页面生命周期维护的资格位。两轮 Full 测得该区间均值从 **0.731–0.750 ms 降至 0.276–0.295 ms**，延迟跟踪合计从 **0.814–0.843 ms 降至 0.339–0.363 ms**，达到 S1 合计均值 ≤0.5 ms 的局部目标。整帧验收未通过，见下文。

## 实现与生命周期

- 每个逻辑页一个排除位，等价于原判定 `residentState(page.state) || page.lockedFallback`。只在启用延迟测量时初始化，大小为 `ceil(pageCount/64)*8` 字节，不逐帧重建；vector 容量可保留至 manager 销毁。
- `setPageState` 统一更新该位；fallback 锁定单独更新，因为锁定在途页面时状态可能不变。释放存储先转 Unloaded，再删除节点；场景重置清空位图。
- 未分配、预算阻塞、排队、待上传和 PendingUnload 页面仍可跟踪；fallback 即使未完成上传也排除。保留合并后 prefetch→demand 语义和原批次时间戳。
- 不向异步请求保存 PageEntry 指针或跨批次快照。`Refresh request priorities` 中用于修改页面和准入的查询仍保留；此次没有宣称移除全部驻留查询。
- 仅修改 Streamer 和回归测试，未改变预算、加载量、shader、RenderPass 或 Reflex 策略。

## 验证

Release 样例与 MetallicRhiTests 构建成功，10/10 项聚焦测试通过，Vulkan validation 关闭。新增 `streamer_meshlet_latency_eligibility` 覆盖立即/延迟准入、排队时 fallback 锁定、重复/非法请求、预算阻塞、上传完成、PendingUnload、删除后重新请求、场景 ID 复用及关闭测量。真实 GPU 上传完成/取消路径由已有 upload-completion 回归覆盖。

## Full 同配置漫游

基线为当前 HEAD `d0861a4a4` 修改前新采的两轮，不与 9/24 的旧构建混用。RTX 5070 Ti、驱动 616.92，输出 1797×660、内部 1198×440，DLSS Quality、LOD 1.5 px、SW 8 px，Reflex On，隐藏窗口、VSync、2 frame slots；每轮预热 10 秒、采样 30 秒。四轮 graph、相机路线、配置、分辨率及 shader SHA256 一致。流送/cut 未冻结。

|版本/轮次|Track 均值 / P95 ms|Reset 均值 ms|合计均值 / P95 ms|Stream Begin 均值 ms|上传页/秒|整帧 P95 ms|>33.33 ms|Sleep 均值 ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|before run1|0.731 / 1.274|0.084|0.814 / 1.402|5.402|997.4|26.242|0|6.099|
|before run2|0.750 / 1.208|0.093|0.843 / 1.349|5.605|992.1|26.838|0|5.619|
|after run1|0.276 / 0.514|0.064|0.339 / 0.645|3.866|971.3|34.205|108|15.824|
|after run2|0.295 / 0.517|0.068|0.363 / 0.648|3.993|979.7|34.215|103|15.409|

合计分位数先逐帧相加 Reset / latency tracking 与 Track merged demand，再计算。CPU-only 区间没有 GPU 数据。

两轮优化后的 Consume requests 均值为 3.38–3.50 ms（此前 4.87–5.00）；Update resident demand 为 0.72–0.76 ms、Refresh request priorities 为 0.81–0.83 ms。未观察到相邻 CPU 区间出现抵消收益的增长。但未修改区间也变快，且帧率改变，不能把所有下降归功于本次缓存。

请求约 1.35 万/帧、驻留约 6.21 万页，上传约 971–980 页/秒（此前 992–997，下降约 1%–3%）。四轮 loadFailures / requestOverflows 为 0，GPU timing 无缺失、HZB 有效且 snapshot 未冻结。既有 BLAS overflow 最大 12425 仍存在。Full 导出不包含 demand→drawable 延迟直方图，本次没有声称完成真实 Full 尾延迟分布验收；统计语义由生命周期回归保护。

**整帧没有改善**：优化后 P95 34.21 ms，分别有 108/103 帧超过预算；此前 P95 26.24–26.84 ms、无超预算帧。Reflex Sleep 均值从 5.62–6.10 ms 增至 15.41–15.82 ms，帧槽等待只从 1.98–2.14 ms 降至 0.54–0.58 ms；因此不能把 CPU 节省换算为 FPS 提升。四轮有效 Reflex mode 都为 On，策略没有改动。当前证据定位到 pacing 等待增长，尚不能证明其原因。

本轮为连续 A1/A2/B1/B2，非交错锁频实验；桌面进程仍在运行。局部目标达成，持续 30 fps 的验收仍需单独排查 Reflex pacing 并交错复测，不通过关闭测量或忽略超预算帧宣称达标。

## 证据与复现

- 结构化结果：`Documentation/ZorahFullStreamerResidencyLookupResult.json`，保留每轮条件、耗时和安全计数。
- 构建：`build-release/streamer-residency-cache-build-0925.log`；回归：`build-release/streamer-residency-cache-tests-0925.log`。
- 原始采样：`build-release/full-streamer-residency-before-0925/run1..2` 与 `build-release/full-streamer-residency-after-0925/run1..2`。
- 汇总：`python -B build-release/AnalyzeStreamerResidencyLookup.py`。

```powershell
$env:METALLIC_REFLEX_MODE='on'
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-dir> -Runs 2 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```
