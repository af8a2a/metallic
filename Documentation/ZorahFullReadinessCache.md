# ZorahFull：单次 overlap 检查与生命周期就绪缓存

日期：2026-09-23。实现、8 项相关回归及三轮 Full 漫游验证完成。

## 实现

- RenderGraphExecutor 每帧对每个 pass 只调用一次 supportsFrameOverlap，同时构建阻塞名称与 drain 标志；保留 Check frame overlap contracts 子计时，删除重复的 Collect overlap blockers 扫描。
- MeshletStreamRuntime::sceneReadiness 缓存布尔结果和加载进度；sceneReady 提供布尔查询，VisibilityBufferPass 使用该入口。加载期每帧使未就绪结果失效，由下次查询刷新；完整根覆盖就绪后跨帧复用，不再遍历全部根页。
- 缓存以 runtime 资源代为生命周期。reset/initialize 创建独立共享状态；取消回调只捕获原代状态和原代页表标志，不捕获 runtime 指针。页表初始化或有序发布取消会清除就绪结果。
- fallback BLAS 从“录制即 built”改为跟踪提交事务：未提交不发布 ready，取消后可重新构建，只有队列接受的构建才计入就绪。仍依赖既有 GPU 提交依赖保证后续读取顺序，不增加 CPU 等待。
- 几何根页保持锁定。CLAS 池的破坏性操作通过 observer 通知根覆盖状态；普通流送页退休不清除缓存。主动退休根 CLAS 或清空池时立即标记根覆盖失效；此时 fallback BLAS 的根引用也可能失效，必须重建 runtime，不能仅因 CLAS 再次出现便恢复旧 ready。
- Debug snapshot 增加 sceneReadinessScans、sceneRootsInvalidated、fallbackBlasRecorded、fallbackBlasSubmitted，便于检查生命周期。

## 三轮 Full 固定漫游

与预算准入优化后的基准核对资产、shader、路线、输出/渲染尺寸和质量配置：1797×660 输出，DLSS Quality 1198×440，LOD 1.5 px；每轮 10 秒预热、30 秒采样。

| 轮次 | 均值 ms | P95 ms | 超过 33.33 ms | Preflight 均值 ms | overlap 检查均值 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 21.22 | 28.22 | 6/1414（0.42%） | 0.0042 | 0.0013 |
| 2 | 21.03 | 27.26 | 5/1427（0.35%） | 0.0039 | 0.0010 |
| 3 | 22.02 | 30.12 | 15/1363（1.10%） | 0.0043 | 0.0012 |

4204 帧中 26 帧超预算（0.62%），最长连续超预算为 2/1/3 帧。全部 HZB 有效、drainReasonMask 为 0、无缺失 GPU 计时、无加载失败或请求溢出。上传吞吐约 986–993 页/秒，几何预算仍为 3.5 GiB，LOD 与分流参数未调整。

此前正常三轮 Preflight 为 3.33–4.04 ms；细分归属轮受后台 CPU 负载影响达到 7.04–8.93 ms。现在降至约 0.004 ms，且回归证明就绪后的连续查询和帧推进不再全量扫描。整帧从此前预算准入轮的 28.38/30.71/29.31 ms 降到本轮约 21–22 ms，但两组后台负载不同，不能把整帧差值全部归因于本次代码。用户在本轮前已暂停大型任务；采样中仍有浏览器等活动。

本轮三次 P95 均低于 33.33 ms，但仍有慢帧，尚未达到严格的全程最低 30 fps。既有 BLAS overflow 聚合计数最高仍为 12425，此项未改变其预算或回退行为。

## 验证

- Release MetallicGPUDrivenSample、MetallicRhiTests 构建通过；最终源码仅在采样后做了缩进整理，并重新构建。
- 6 项核心回归通过：streamer_ordered_publication_retry、stream_blas_cut_cache、frame_output_consumer_gpu_dependencies，以及 3 项 editor_profiler 测试。
- 2 项补充 CLAS 回归通过：stream_clas_runtime_lifecycle、stream_clas_eviction_reupload。
- 覆盖：首次 fallback BLAS 取消后重建并就绪；未提交/已取消时不发布 ready；1000 次稳定查询与帧推进不增加扫描；普通 CLAS 页退休不破坏根缓存；根 CLAS 退休立即失效；非 RT 初始发布取消/重试；reset 后及重新初始化时不复用旧就绪/进度；每 pass overlap 检查次数为一次；同/跨队列输出消费者与重建生命周期继续有效。
- Vulkan validation 开启的回归日志无 VUID/validation error。git diff --check 通过。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*streamer_ordered_publication_retry:*stream_blas_cut_cache:*frame_output_consumer_gpu_dependencies:*editor_profiler*" --output-dir build-release/readiness-cache-tests-final
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*stream_clas_runtime_lifecycle:*stream_clas_eviction_reupload" --output-dir build-release/readiness-cache-clas-tests
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

原始数据：build-release/full-readiness-cache-0923/run1..3；Comparison.json/md 与 SlowFrames.json/md。汇总：[ZorahFullReadinessCacheResult.json](ZorahFullReadinessCacheResult.json)。
