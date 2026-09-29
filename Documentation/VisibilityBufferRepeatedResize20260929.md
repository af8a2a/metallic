# VisibilityBuffer 同尺寸重复重建

用户日志：2026-09-29 15:53:02.523–15:53:44.608，42.085 秒内出现 330 次 `1198x440 -> 1198x440`；总 resize 日志 333 次。

## 原因和实际工作

`VisibilityBufferPass::syncRuntimeGeometry()` 在任何材质纹理快照发布后，将 `bindingViewAllocationId_` 置零。`ensureFrameResources()` 把绑定分配 ID 不匹配和分辨率变化合并判断，因此普通纹理 mip 流送也进入完整 resize 路径。

这不仅是日志误报：会重新创建 pass-owned 剔除 depth/visibility 目标、binding registry、材质 remap、owner mask 和 SPD 辅助资源，并退休旧 bundle；同时重置 frame index、HZB、前帧相机和冻结相机状态。GPUScene 视图自身有尺寸/容量复用检查，hybrid cluster pool 也有复用条件，所以不能把每条日志描述成所有 GPU 缓冲均重建。

## 修改

- `ensureFrameResources()` 只在实际宽高变化时重建屏幕目标。
- 材质绑定 dirty 单独维护。实际 GPUScene allocation 变化交给绑定更新路径，仍使 HZB 失效，但不重建尺寸未变的剔除目标。
- 纹理快照更新后比较光栅实际引用的 `TextureView` 和逻辑索引映射。Full 流送中未用于 alpha/displacement 的贴图已映射到 fallback；这些着色纹理换 mip 时，保留光栅描述符及 HZB。
- 真正的材质编辑、alpha/displacement 视图或索引变化仍刷新绑定并保守使遮挡历史失效。保留 dirty 至新 bundle 成功安装，避免失败后漏掉重试。
- 不改动 Streamer 的资源加载职责，不改 LOD、光栅质量或流送预算。真实绑定变化仍使用现有 bundle 更新路径，本轮没有把所有绑定资源进一步拆成独立缓存。

## 验证

Release sample 和 RHI tests 构建通过；`git diff --check` 通过。

开启 Vulkan validation：

1. `RhiRendering.zorah_full_first_frame` 通过：MiniZorah→Full，材质覆盖、后续帧及输出检查；无同尺寸 resize。
2. `RhiRendering.visibility_buffer_abeautiful_game_transmission` 通过：运行时材质编辑、透射/体积对照、binned/unbinned 图像对照和真实尺寸切换。日志只记录实际宽高变化。
3. `RhiRendering.streamed_realtime_pipeline` 首次缺少 `--rhi-realtime` 被跳过；补齐该设备配置后重新运行通过，覆盖流送、resize/recompile 连续性和阴影输出。

日志无 Validation Error、VUID 或 DeviceLost。Full settled 输出已检查，建筑、植被和人物覆盖正常；原有单样本噪声仍在。验证输出为 960×540、DLSS 关闭，不能替代长期编辑器漫游图像验证。

证据：`build-scheduling-release/vbuffer-refresh-validation.log`、`build-scheduling-release/vbuffer-refresh-streamed.log` 及同名目录。

## Full 编辑器短漫游

同现有绝对相机路线：180 帧、逻辑 30 秒、预热 3 秒；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、8M candidates。已有磁盘缓存，新进程，普通计时，关闭工作量读回和验证层。使用 `Tools/RunZorahFullRoam.ps1` 及 `build-release/blas-selected-normal-route.json`。

- 纹理升级累计 609→852（+243），降级 0→1；refined images 376→543。确认覆盖了持续纹理发布。
- 同尺寸 resize **0 次**。仅启动时发生 `2048x1036→1365x691→1198x440` 两次真实尺寸切换。
- BLAS overflow、页面 IO 失败和请求溢出均为 0；无 DeviceLost。
- 单轮帧均值 24.024 ms，P95 34.132 ms，12/180 帧超过 33.33 ms。Visibility prepare CPU 均值 1.223 ms，P95 1.964 ms。

本轮是行为验证和单次成本观察，没有新做交错三次 A/B；不能由历史运行推算严格提速比例，也尚未通过持续 30 fps 验收。没有用“少打印日志”代替资源生命周期修复。

证据：`build-release/vbuffer-refresh-full/run1/{Capture.json,Frames.jsonl,Summary.json,stdout.log,Gpu.csv,GpuProcesses.csv}`，配置及二进制/shader 身份见上级 `Manifest.json`。
