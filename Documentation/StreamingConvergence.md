# 流送需求、首屏就绪与同帧几何发布

2026-09-15。落实 [MiniZorah 流送对比](MiniZorahStreamingComparison.md) 中的前三项改进；实时管线、全局 ViewConstant 和现有保守覆盖规则继续使用。

## 行为变化

### 需求独立于驻留绘制集合

CPU 参考实现及 GPU 的线性、BVH、cooperative 遍历都按误差和可见性产生正式页请求，不再等待父级先变为可绘制。所有父级 active 的条件仍用于建立安全绘制集合。细页提前到达时，只有满足原有多父级 DAG 覆盖条件才替换粗页。

因此，缺少中间级几何时可以一次发出多个级别的需求；中间级尚未到达的细页也不会成为孤立的绘制片段。容量不足时仍退回完整 terminal cut。预取继续承担视锥和误差范围之外的前瞻，不再替正式需求承担绕过父级驻留等待的职责。

涉及 `MeshletLod.cpp`、`GPUDrivenStreamAsset.slang`。

### Streamer 提供首屏就绪状态

`MeshletStreamRuntime::sceneReadiness()` 区分资源初始化完成与基础覆盖完成。它统计已确认驻留的根几何页；启用 cluster RT 时，还要求根页 CLAS 和 fallback BLAS 齐备。`StreamerSubsystem` 汇总在用会话的状态。

编辑器在此阶段显示场景准备进度并遮住尚不完整的图像，持续执行渲染图、反馈和流送，不阻塞 UI 等待全部高精度数据。就绪之后细节仍然按需加载。这是完整基础覆盖的首屏条件，不表示全场景加载完成，也不把隐藏加载过程算作吞吐提升。

视口尺寸变化继续复用同一会话，已就绪状态不应因正常尺寸变化退回加载。

### 上传可在当前命令录制中发布给 GPU

页面上传现在在遍历之前记录拷贝及 transfer → shader read 屏障。正式实时路径通过 StreamerSubsystem flush，保留上传统计。

`StreamUploadCompletion::isRecordedBefore()` 只认可拷贝所在的同一个、仍有效的命令录制。相同 frame context、已经入队的另一个命令段、取消的尾段和重新录制的 command buffer 都不能冒用这份证明。

Residency 据此生成仅用于 GPU 的 drawable 页表 patch；CPU 页面仍为 PendingUpload，直到既有 completion 协议确认完成，随后才更新 CPU 驻留、CLAS 入队和回收状态。每页只保留一份最终 patch，避免 PendingUpload 与 drawable patch 并发写同一表项。

页表更新缓冲按帧槽分开，避免 CPU 覆盖仍在途的更新数据。初始化或更新录制被取消后，下次从 CPU 状态重建 GPU 页表。回滚回调只持有独立的有效性标志，不捕获可能已销毁的 runtime 指针。

本次缩短的是几何 upload → GPU 可绘制链路。CLAS 的尺寸回读、CPU 分配和搬移完成确认仍然保留，尚未迁移为参考程序的 GPU 分配器。原有 `UploadToDrawable` 等 CPU 延迟统计仍按完成确认计时，不能直接当成新路径的 GPU 首次使用时间；调试快照增加 `orderedUploadPages`、`sceneReady` 和 `scenePreparationFraction`。

## 验证

Release，RTX 5060，当前配置未启用 NRD。

- `meshlet_lod_stream_reference_frontier`：根页之外的多个层级同时缺失时，一轮产生全部所需请求；细页先到仍保持完整粗级覆盖。
- `meshlet_lod_stream_bvh_reference` 和 `meshlet_lod_stream_gpu_matches_reference`：365 组 CPU/GPU 对照，包括多父级、容量回退、视锥、预取配额和稀疏状态清理。
- `meshlet_lod_stream_scene_runtime_cut`：16 组 Bunny 收敛结果匹配完整驻留参考，覆盖硬件/混合光栅。新增首批 GPU 发布发生在 CPU 驻留确认之前的检查。
- `streamer_meshlet_upload_completion`：验证同录制可见性，以及未 flush、取消、前段已提交但尾段取消、错误 frame context、完成前不回收等路径。
- `streamer_ordered_publication_retry`：取消初始页表和上传录制后，在另一帧槽重试；回读 GPU active header，确认根页已参与选择，而 CPU 驻留仍待确认，再验证后续首屏就绪。该项单独开启验证层通过，VUID 计数为 0。
- `streamed_realtime_pipeline` 和 `minizorah_realtime_pipeline`：渲染断言通过。基础覆盖分别在第 5、17 个测试帧就绪；MiniZorah 为 3163 根几何页 + 3163 根 CLAS 页。后续至第 182 帧的相机/尺寸变化保持会话和就绪状态。
- 默认 MiniZorah Sample（`METALLIC_SMOKE_TEST_FRAMES=120`，`--smoke-test --debug-control`）完成 120 帧提交和呈现；该运行用于桌面启动检查，不用于吞吐比较。

第 17 帧是离屏测试的基础覆盖里程碑，包含该测试的预算和 CPU/GPU 等待方式；不是桌面首屏耗时，也不是与旧版或参考程序的速度倍数。

复现命令：

```powershell
cmake --build build-release --target MetallicRhiTests MetallicGPUDrivenSample
build-release/tests/MetallicRhiTests.exe --rhi-no-validation '--gtest_filter=*meshlet_lod_stream*:*streamer_meshlet_upload_completion*:*streamer_ordered_publication_retry*'
$env:METALLIC_TEST_MINIZORAH = '1'
build-release/tests/MetallicRhiTests.exe --rhi-realtime --rhi-no-validation '--gtest_filter=*streamed_realtime_pipeline*:*minizorah_realtime_pipeline*'
```

本机需在 MSVC 开发环境中构建。日志保留在 `.cache/streaming-convergence/`。

验证限制：启用 Vulkan 验证层的场景测试在 NVIDIA driver 的 descriptor heap 绑定路径发生访问异常；临时换回旧 shader 和旧运行时流送协议仍能复现，关闭验证层后通过。场景级验证层检查尚未通过。带 Streamline 的两项实时测试输出通过后，进程仍停在既有的全局 teardown 问题处；Sample 完成 120 帧后也未正常退出。这些进程由外部终止，不能记作正常退出。
