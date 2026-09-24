# Profile marker 修正后的 ZorahFull 复测

2026-09-24，HEAD e84464e01。已重新构建 Release MetallicGPUDrivenSample，完成四轮 Reflex On 和一轮 Off 诊断，共 6297 帧。未修改运行时实现或产品默认设置。

## 计时边界核对

本次修正把 profileScope 的层级导出为配对的 Vulkan debug labels，并新增 Visibility prepare、Visibility raster、Upload flush。VBuffer 父范围仍包含 Streamer、LOD、CLAS/BLAS/TLAS。CPU-only publishCpuProfile 区间仍然只统计 CPU，没有 GPU 时间。

本次采用修正后应用的 GPU timestamp / CPU profiler 导出；没有新采集 Nsight GPU Trace，也不声称获得了 shader stall/occupancy 数据。对全部采样帧检查了新层级时间覆盖、CPU-only 无 GPU 值、每帧一条流送样本、HZB 有效和实际光栅路径。

## 条件与结果

RTX 5070 Ti 16 GB，driver 616.92；输出 1797×660、内部 1198×440，DLSS Quality、LOD 1.5 px；10 秒就绪后预热、30 秒固定漫游。隐藏窗口、VSync=true、2 frame slots。流送未冻结，SW threshold=8 px，async=false，metadataFastClassification=true，cullHardwareClassification=false，SW workload replay 关闭。

五轮 executable SHA256、shader SHA256 相同。与 9 月 23 日相机关键点、分辨率、帧槽、VSync 相同；当前导出新增 sample、routeFrames=0、temporalJitter=true 和分类开关字段。源码和 shader digest 已变化，不能将旧/新计时差异归因于 marker 修正本身。

| 轮次 | 帧数 | 整帧均值 ms | P95 ms | P99 ms | >33.33 ms 帧数 | Sleep 均值 ms |
|---|---:|---:|---:|---:|---:|---:|
| On 1 | 1098 | 27.329 | 34.490 | 37.225 | 112 | 13.282 |
| On 2 | 1323 | 22.681 | 29.125 | 31.645 | 8 | 4.694 |
| On 3 | 1378 | 21.772 | 28.397 | 32.593 | 10 | 3.893 |
| On 补测 | 1105 | 27.168 | 34.517 | 36.858 | 123 | 13.425 |
| Off 诊断 | 1393 | 21.540 | 27.584 | 30.735 | 4 | 0.048 |

**仍未达到持续漫游每帧至少 30 fps。** 没有删除慢轮或把 On/Off 混合成平均收益。

On 1 的 GPU Engine 进程遥测因 Get-Counter 错误为空，因此不能证明该轮无后台竞争；GPU 总利用率遥测和应用 profile 完整。补测有进程遥测并再次出现慢 On 状态，故不能把该现象仅归给首轮遥测失败。其他轮仍有浏览器/桌面后台活动，未锁时钟，不作为严格微秒级 A/B。

## 新层级给出的工作成本

下列为四轮 On 的轮次均值范围。early+late 先在同一帧相加，再统计；父子区间不重复相加。

| 区间 | CPU 均值 ms | GPU 均值 ms | 解释 |
|---|---:|---:|---|
| 几何 Stream Begin | 6.25–7.84 | 约 0.03 | 主要 CPU 调度热点 |
| Consume requests | 约 4.9–6.2 | 无 | 嵌套于 Stream Begin |
| Reset/latency + Track merged demand | 2.16–2.80 | 无 | 不重叠的两个区间，仍支持优先降低跟踪成本 |
| Visibility prepare | 1.16–1.71 | 接近 0 | 同步 GPUScene、视图、绑定与参数；独立于 Streamer |
| Stream traversal | 0.69–0.78 | 2.48–2.71 | 包含 LOD 与加速结构构建 |
| Visibility raster | 约 0.26–0.33 | 约 10.2–11.4 | 纯 visibility 执行范围 |
| Software raster，early+late | 很小的录制时间 | **6.71–7.47** | GPU 侧最大单项之一 |
| Soft/hard classification，early+late | 很小的录制时间 | **0.66–0.80** | 远小于 SW 光栅，不宜继续作为唯一优化重点 |
| Deferred 整个节点 | 含纹理 Streamer prepare | **5.48–6.56** | 不能全部当作材质 shader 时间，须按子 scope 检查 |

Stream traversal 中 Detail demand 约 0.44 ms、LOD frontier 约 0.43 ms；另有 mask/prefix/emit、CLAS/BLAS/TLAS。因此不能与 vk_lod_clusters 的单独 Traversal Run 直接比较。新标签进一步明确了归属，没有推翻此前 CPU Streamer 的热点判断。

约 1.35 万请求/帧、6.2 万驻留页的规模仍在；几何上传维持约 **981–992 页/秒**，各轮 loadFailures/requestOverflows 为零。已有 BLAS overflow 最高 12425 仍存在，本次没有解决或掩盖该容量问题，不能据此宣称完整 RTAS 质量验收通过。

## 节奏波动需要单独处理

较慢的两轮 On：Sleep 均值约 13.3 ms，P95 21.0 / 21.8 ms；GPU envelope 反而只有约 19.2–19.3 ms。较快 On：Sleep 均值约 3.9–4.7 ms，GPU envelope 约 21.2–22.1 ms。慢轮增加的墙钟时间主要落在 Sleep，而不是更重的 GPU 工作。

Off 的 Sleep P95 为 0.088 ms，帧槽等待均值升至 6.36 ms。它比慢 On 快，但与快 On 接近。这一轮诊断不足以决定永久关闭 Reflex；默认仍保持 On。frameLimitUs=0 也不能排除 VSync、隐藏窗口、呈现状态及驱动策略的影响。

下一次需要归因整帧收益时，应先固定并记录可见编辑器/隐藏窗口状态、呈现模式、Reflex 设置和 GPU 时钟，再交错重复采样；必要时使用 CPU 调度/提交时间线看驱动等待。不要把约 13 ms Sleep 当作可直接消除的 CPU 计算，也不要据单轮低 P95 宣称优化已稳定。

## 下一步顺序

1. **Streamer 先做 S1。** 将默认开启的延迟跟踪改为每批取时、复用查询、到期桶清理；保留预算阻塞请求的尾延迟统计。当前相关区间仍有 2.16–2.80 ms/帧，优先级不变。再做 S2 驻留冷热增量维护、S3 跨帧候选缓存。
2. **单独细分 Visibility prepare。** 新暴露的 1.16–1.71 ms 不应归给 Streamer。优先拆 syncRuntimeGeometry、syncGPUSceneRasterState、prepareGPUSceneView、ensureGPUSceneBindings 与 buffer/descriptor 更新，依据代际确定可缓存部分。
3. **GPU 主线转向 SW raster，再看 Deferred 子项。** SW 成本约为分类的 9–10 倍；保留相同 camera/cut/residency 的对照，依据已有/新增工作量计数决定解码投影、无效三角形 setup、扫描或原子路径，避免仅凭大 marker 就选择重排方案。当前计时没有证明某一类 shader stall。
4. **整帧验收先控制节奏差异。** 新旧都使用 On，保留所有轮次，记录 Sleep+帧槽等待以及 GPU envelope；优化实际 CPU/GPU 工作与减少帧墙钟波动分别报告。

CPU 降本仍可能表现为更多 Reflex 等待，因此 S1 的验收同时检查工作量、需求尾延迟、上传吞吐与整帧，而不是承诺节省 2 ms CPU 就一定节省 2 ms 整帧。

## 证据

- [结构化结果](E:/metallic/Documentation/ZorahFullMarkerRecheck20260924.json)：五轮 Manifest、帧统计、同帧聚合 scope、Reflex 子计时、计数和遥测完整性。
- `build-release/full-marker-recheck-0924/run1..3`
- `build-release/full-marker-recheck-repeat-0924/run1`
- `build-release/full-marker-recheck-off-0924/run1`
- 构建日志：`build-release/profile-marker-recheck-build-0924.log`。五轮完整 GPU timing，性能跑测关闭 validation；这是性能/数据完整性复测，不是新的 validation-clean 或图像逐像素验收。

复现命令与前轮一致，OutputRoot 需使用新目录：

```powershell
$env:METALLIC_REFLEX_MODE='on'
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-dir> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```
