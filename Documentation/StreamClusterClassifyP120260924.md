# P1：cull 中的确定 HW 分流与 MiniZorah 漫游验收

2026-09-24，RTX 5060 8 GB / driver 616.92，Release 优化构建。

结论：P1 在真实 MiniZorah 漫游中降低精确分类约 **10.46%**，`cull + classify + stable bins` 合计降低 **7.14%**。完整 VisibilityBufferPass 的均值仅降低 **0.18%**，RenderGraph GPU envelope 均值增加 **0.19%**，两者的轮次均值范围均重叠，不能认定整帧收益。P1 保留为实验，`VBuffer.cullHardwareClassification=true` 显式启用；默认 false 继续使用 P0。

## 实现

- [GPUDrivenStreamAsset.slang](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang)：`cluster.forceHardware` 在新 cull 中独立于 metadata sphere 快速分类开关。MASK/BLEND/coverage 或 tessellation 确定需要 HW 的可见条目直接写原 candidate 的 HW tag，不进入精确分类队列。无效球体触发的 metadata fallback 同样不阻止这种语义确定的分流。
- [StreamClusterClassify.slang](../Shaders/Features/GPUDriven/StreamClusterClassify.slang)：P1 精确队列保证不存在这些强制 HW 条目，删除其消费者中的 raster bindings → group instance index → instance flags 依赖加载。投影、阈值、三角解码、128 threads、三处同步和四 word 队列 ABI 不变。
- P0 的 producer/consumer 成对保留为 `streamClusterCullP0Main` / `streamClusterBinP0Main`。运行时开关同时选择两个 pipeline，避免旧 producer 配新 consumer。P0 和 P1 复用同一份投影及分桶逻辑。
- [VisibilityBufferPass.cpp](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp) 的软件光栅 identity 额外导出该开关，分析器检查每帧实际选择与请求的 A/B 模式一致。

驱动 pipeline executable statistics：classifier 的寄存器数均为 40，共享内存 P0 3264 B → P1 2752 B，二进制 P0 15616 B → P1 13824 B；cull 寄存器数均为 115、共享内存均为 0。普通模式与 capture-symbols 模式结果一致。驱动报告的 Local Memory Size 异常，不据此推断 spill；本次没有测量 occupancy 或 stall 分布。

## MiniZorah 实时光照漫游 profile

使用真实 `MetallicGPUDrivenSample.exe --sample gpu-driven-sample`，保留 live geometry/CLAS streaming、LOD、HZB、实时光照和 DLSS。没有冻结 cut/residency。四轮顺序 **P0 → P1 → P1 → P0**，每轮预热 5 秒、测量 360 帧，共 1,440 个性能帧。输出 1280×720，DLSS 内部渲染 **853×480**，LOD pixel error=1.5，SW threshold=8px，async SW=false。

相机按帧序号固定推进，包含静止、前进、近墙、左右转向、返回及恢复；距离 6 个场景单位，yaw ±65°。配置中的 30 秒是轨迹插值坐标，每轮采样以 360 帧结束，实际漫游采样约 6 秒，不能把它表述为 30 秒壁钟采样。两种模式的 sample index、轨迹时间、camera keyframes、图配置与分辨率均由分析器核对。关闭 temporal jitter，排除不同加载帧数造成的 jitter 相位差；未锁 GPU 时钟。

以下均为每帧 early+late 合计的 GPU 时间，再平均两轮均值；父子 scope 不重复相加。

| GPU scope | P0 均值 ms | P1 均值 ms | 耗时降低 |
|---|---:|---:|---:|
| 精确分类 | 0.339438 | 0.303933 | **10.46%** |
| Cluster cull | 0.078112 | 0.080840 | −3.49% |
| Stable bins | 0.049714 | 0.049126 | 1.18% |
| 上述三阶段合计 | 0.467264 | 0.433899 | **7.14%** |
| 完整 VisibilityBufferPass | 3.907464 | 3.900514 | 0.18% |
| RenderGraph GPU envelope | 9.199972 | 9.217667 | −0.19% |

分类链净节省约 **33.4 μs/帧**，已经计入 cull 的增加，不能只展示分类自身的 35.5 μs 节省。完整 VBuffer 的约 7 μs 差异小于轮次波动：P0 两轮为 3.918880 / 3.896048 ms，P1 为 3.892656 / 3.908372 ms。未建立统计显著的完整 pass 收益；GPU envelope 同样不能认定改善。editor loop 约 16.7 ms，受 present/Reflex 节奏影响，不用于宣称 FPS 收益。

性能测量使用应用现有 GPU timestamp profiler，所有性能帧都有完整 GPU timing，未启用 validation、Nsight 注入或 readback 诊断。旧 Nsight GPU Trace 的 metric-set 阻塞未在本实验中解决；这里不声称取得了新的 Nsight Shader Profiler stall 分布。

## 独立诊断：实际发生了什么

另跑 P0/P1 各 360 帧，仅每 30 帧读回 cull/bin header；这两轮 **不进入性能统计**，也不运行 SW workload replay。12 个相机点 × early/late=24 组计数，相机和 phase 一致：

| 诊断累计 | P0 | P1 |
|---|---:|---:|
| exact clusters | 328,198 | 328,168 |
| fast SW | 34,446 | 34,452 |
| fast HW | **0** | **0** |
| candidate overflow | 0 | 0 |

这条 MiniZorah 路线没有触发语义确定的 HW 分流，性能改善主要来自删除 classifier 的重复检查，而不是少处理大量 HW cluster。强制 HW 分流本身由专项测试中的 MASK、BLEND、tessellation、metadata-disabled 和无效球体输入验收。

Live streaming 仍存在小幅工作量差异：exact 总量相差 30（约 0.009%）；24 组 cull 计数有 16 组完全一致，bin header 有 11 组一致。不同进程的平均 resident pages 约 9,135–9,248。因此这些诊断不是同一冻结帧的逐位证明，也不应将所有整帧波动归因于 P1。逐位正确性使用下述独立集成对照。

## 正确性验收

[StreamClusterClassificationTests.cpp](../tests/rhi/StreamClusterClassificationTests.cpp) 扩为 24 个用例，逐一运行独立参考、P1 成对路径、P0 成对路径，覆盖 early/late。除了原有 candidate tags、stable bins、records、retry 与 dispatch 参数比较，增加：

1. 在 stable scatter 之前读回 P1 精确队列，检查每项都不是强制 HW。
2. P0 减少的 exact 数量必须等于 P1 增加的 cull HW 数量，fast SW 数量不变。
3. 显式加入 BLEND flags、metadata 关闭的全 tessellation 场景，以及 metadata 关闭、jitter、异常球体混合场景。

[MiniZorahStreamAssetTests.cpp](../tests/rhi/MiniZorahStreamAssetTests.cpp) 为真实 MiniZorah 和 Bunny 的 VisibilityBufferPass 增加 P0/P1 图像对照。对比 terminal roots、settled original/far/near 相机 cut，分别开启/关闭 metadata 快速分类，检查原始 visibility ID 和 D32 depth 位模式完全一致；相机 pose 对照期间冻结 streaming cut，之后恢复 live streaming。MiniZorah 每次比较 1920×1080=2,073,600 像素。

扩展检查最初在 far + metadata enabled 处出现过一次 depth 不一致，visibility 一致（保留在 `integration-final.log`）。增加 P0/P0、P1/P1 连续帧稳定性探针后，`depth-probe` 中八组对照均为零差异；严格断言没有改为容差比较。最终使用 `--gtest_repeat=2` 连续运行三个测试，两轮均为 3/3 通过。MiniZorah 八组、Bunny 四组对照的 visibility、depth 和重复 depth 均为零差异。该初次差异的具体根因尚未确认，不能宣称已经修复 HZB/异步光栅问题，这也是保留实验开关的原因之一。

另在 capture-symbols + pipeline statistics 模式运行分类专项测试并通过。完整场景集成使用 `--rhi-no-validation`：本机旧 validation layer（header 341）不识别 `VK_KHR_device_address_commands`，启用后完整渲染此前在 shader 执行前崩溃。分类专项在 validation 开启下通过，但保留了设备初始化警告，不能表述为 validation-clean。

## 复现与数据

在 x64 VS Developer PowerShell 构建 `MetallicRhiTests` 和 `MetallicGPUDrivenSample`，然后运行：

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
& build-release/tests/MetallicRhiTests.exe --rhi-no-validation '--gtest_filter=*stream_cluster_cull_classify_equivalence:*stream_metadata_vbuffer:*minizorah_vbuffer' --gtest_repeat=2 --output-dir build/p1-acceptance

& Tools/RunMiniZorahClassifyComparison.ps1 -OutputRoot E:/metallic/build/p1-roam-new -Frames 360 -WarmupSeconds 5 -IncludeDiagnostics
python Tools/AnalyzeMiniZorahClassifyComparison.py E:/metallic/build/p1-roam-new
```

脚本拒绝覆盖旧目录，按顺序执行进程并保存 executable/shader hash、运行环境、GPU 时钟、逐帧 JSONL、scope 定义与独立诊断计数。分析器拒绝缺帧、缺 GPU scope、不同轨迹或超出分类开关的配置差异。

- [本次原始 profile、CSV 与日志](../build/classify-p1-20260924/roam/)
- [完整聚合 Summary.json](../build/classify-p1-20260924/roam/Summary.json)
- [验收与实验摘要](StreamClusterClassifyP120260924.json)
- [验收日志](../build/classify-p1-20260924/acceptance.log)
- [首次远景差异日志](../build/classify-p1-20260924/integration-final.log)

采样时两种模式都由配置显式指定。采样后仅将普通运行的默认开关设回 P0；被测 P0/P1 shader 和显式 A/B 选择逻辑未改变。后续若要默认启用 P1，需继续排查首次远景差异，并在更多独立漫游重复中确认完整 pass 的净收益。
