# VBuffer 空 resident 光栅开销

2026-09-16，基线 `e118f5a`，RTX 5060 / 8 GiB / 驱动 610.47。

## 诊断与修复

用户截图的 `Resident early/late` 分别为 4.103 / 3.955 ms。默认 MiniZorah 场景只有流送几何，GPUScene 仍有实例表，但 resident 候选 meshlet 范围为空。原来的 `drawVisibility()` 每帧仍执行两轮空分类、软件像素缓冲清除、异步 compute/graphics 分支、join 和全屏 merge。

同路线基线 m1 的这两个父 scope 中位数为 4.314 / 3.980 ms；实际空 software/hardware raster 子 scope 仅约 0.005 ms。父 scope 横跨多个提交，包含跨队列等待和 GPU 调度延迟，不能把其耗时全算作 shader 执行时间。这条空路径在本轮之前、也在上一轮遍历改动之前就已存在；本次数据不支持将截图异常归因于新增 tile 层级。

修复以 CPU 已知的 `activeMeshletCount_` 判断 resident 生产者是否存在：

- 范围为空且要求 Load 时直接返回，取消空 late 阶段。
- 范围为空但要求 Clear 时，仅执行 visibility/depth attachment clear，记录为 `Clear visibility`；冻结相机的独立剔除目标同样清除。
- 范围非空保留原有 resident 光栅路径，混合场景仍提交两套生产者。

纯流送每帧异步分支从 4 个降至 2 个。没有读取 GPU 计数后阻塞 CPU；也没有因为某帧 GPU 剔除结果为空而跳过后续潜在可见几何。保留 resident LOD header 初始化和公共实例剔除，避免改变调试资源及 GPUScene 状态契约。shader、LOD 误差、流送预算和 stream 两轮遮挡剔除均未改动。

## 同路线复测

使用 `Tools/RunMetallicCfgReplay.ps1` 与 `.cache/gpudriven-four/Replay.json`，前后各两次完整 3000 帧实时管线回放，另跑一次修改后的质量回放。输出 1920×1080，DLSS Quality 内部分辨率 1280×720，LOD 目标 1.5 render px。原始文件在 `.cache/vbuffer-empty-resident/{before,after}`，摘要及可执行文件、shader、回放 SHA256 在 [VBufferEmptyResidentResults.json](VBufferEmptyResidentResults.json)。

以下为 `return_hold` 全部 300 帧的 P50，单位 ms，未移除异常帧：

| 阶段 | 修改前 m1 | 修改前 m2 | 修改后 m1 | 修改后 m2 |
|---|---:|---:|---:|---:|
| VBuffer | 19.998 | 4.913 | 4.014 | 4.109 |
| Resident early | 4.314 | 0.177 | 无提交 | 无提交 |
| Resident late | 3.980 | 0.150 | 无提交 | 无提交 |
| Clear visibility | — | — | 0.0034 | 0.0034 |
| Stream early | 4.528 | 1.759 | 1.751 | 1.767 |
| Stream late | 3.665 | 0.215 | 0.220 | 0.229 |
| Host frame | 31.147 | 9.168 | 8.061 | 8.266 |

**这些数字不能解释为隔离环境下的固定提速比例。** 修改前 m1 后台 Unity 的非空 3D 引擎采样 P50 为 72.6%；修改前 m2 中途负载下降，修改后也存在短时竞争。即使不改代码，两次基线 VBuffer 已从 20.0 ms 变为 4.9 ms，说明截图量级的等待很容易被后台 GPU 负载放大。确定的改进是删除两个空队列分支与合并；当前场景测得约 4.0–4.1 ms，不能承诺所有机器或后台负载下都有相同比例。

四次计时回放和修改后的质量回放最终 `cut` 全部字段完全一致：19,394 active groups、162,989 selected clusters、early 26,806 hardware / 19,172 software clusters、late 200 hardware / 361 software clusters，容量回退实例为 0。质量回放第 29 帧仍有未完成流送的细节；第 59 帧及之后所有检查点可见超目标细化数为 0，最终最大可见细化误差 1.499928 px。

## 验证

- Release `MetallicRhiTests` 和 `MetallicGPUDrivenSample` 构建通过，sample 可执行文件已更新。
- `meshlet_lod_stream_scene_runtime_cut` 开启 Vulkan validation 通过。改为真实纯流送元数据与全局 RenderView；断言仅两个异步分支，并比较 CPU/GPU cut、软硬光栅覆盖/ID、透视/正交、两种 Z、冻结相机。新增转身后空背景 ID 为 0、深度为对应 clear value 的检查。
- `hybrid_raster_scene_equivalence`、`render_graph_gpu_driven_preview_pass_render` 开启 validation 通过，覆盖常驻生产者。
- `render_graph_gpu_driven_mixed_producer_render`、`visibility_buffer_deferred_openpbr`、`streamed_realtime_pipeline` 在 `--rhi-realtime --rhi-no-validation` 下通过。混合测试仍要求四个异步分支，验证常驻与流送生产者同时绘制。
- 首轮混合/延迟测试开启 validation 时出现 `0xC0000005`，不是断言或 VUID 错误；关闭 validation 后同项通过。本轮未重新定位其异常栈，不将这些项标记为 validation 通过。前一轮同类 descriptor-heap validation 异常的调查见 `MiniZorahTraversalBudget.md`。
- 完整实时回放使用 `--rhi-no-validation`；Streamline 在测试报告完成后的退出阶段停滞，由脚本清理自己启动的进程，记录于各 `Process.json`，不计入帧耗时。

原始验证日志为 `.cache/vbuffer-empty-resident/{build,stream-test-final,tests,integration}.log`；Bunny 的可视化输出在 `stream-test-final/StreamLodBunny-*.png`。完整回放汇总可重跑：

```powershell
python Tools/SummarizeMetallicCfgReplay.py --before .cache/vbuffer-empty-resident/before --after .cache/vbuffer-empty-resident/after --quality .cache/vbuffer-empty-resident/after/quality --output .cache/vbuffer-empty-resident/Comparison.json
```
