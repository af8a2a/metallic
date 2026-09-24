# GPUDrivenSample VisibilityBuffer：2026-09-24 Capture 分析

本轮读取指定 Capture、验证回放、检查对应渲染路径；没有修改渲染实现。**现有证据足以提出具体 A/B 优化候选，但不足以确认哪个子阶段最慢或给出收益毫秒数。** GPU Trace 在本机失败，不能把仓库旧 RTX 5070 Ti 测量套到本次 RTX 5060。

后续用户提供了 `streamClusterBinMain` Shader Profiler CSV 和截图，已形成[针对该 kernel 的源码归因与改进方案](StreamClusterBinProfile20260924.md)。新数据支持优先优化分类初始化/同步和重复材质判断，但仍未提供完整 GPU Trace 的 pass 毫秒分布；下文关于 CLI 失败的记录保留为历史证据。

## Capture 与测量有效性

- 输入：[MetallicGPUDrivenSample_2026_09_24_13_16_12.ngfx-capture](../Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_24_13_16_12.ngfx-capture)，UUID `d9ecd06a-e50e-4516-898e-4c07312b20b4`，第 55 帧；Vulkan，RTX 5060，驱动 616.92，Nsight Graphics 2026.3.1 / build 38722833。
- 元数据中的 1600×900 是编辑器交换链。截图包含侧栏、底部面板及 DLSS-SR 设置，不能把这个尺寸当作 VBuffer 内部分辨率；本次未获得其准确 render extent。
- 官方 `ngfx-replay` 隐藏回放 3 次。`capture.png`、`replay_0.png`、`replay_2.png` 的 SHA-256 完全一致，首帧 MAE=0。退出阶段进程仍驻留，导出完成后只终止本轮创建的回放进程。
- `iteration_times.csv` 的 `msGpuTime=-1`；8.741–10.957 ms 的 `msFrameTime` 包含 CPU 提交和等待，**不是 GPU 帧耗时，更不是 VisibilityBufferPass 耗时**。
- GPU Trace 先后尝试指标集名称、ID、各架构 JSON、项目配置和 present-count 触发，均返回 `No single-pass metric set selected. GPU Trace can't proceed.`。没有可用的 D3DPERF_EVENTS / GPUTRACE_REGIMES 数据，因此本报告不列 pass 时间、occupancy、barrier 占比、原子瓶颈或带宽瓶颈数值。
- 同 build 的版本字符串告警、`VK_NV_low_latency` 扩展 revision 告警及 NvAPI profile 注册失败均存在。成功的图像回放支持分析该帧工作内容；它不验证原程序的 Reflex 或 CPU 帧节奏。

[NVIDIA 文档](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-ui.html#gpu-trace)区分 Graphics Capture 与从回放采集的 GPU Trace；后者测量的是回放工作，资源 reset、呈现及预热条件仍须核对。

## 已确认的内容，以及不该重复建议的优化

当前源码 HEAD 为 `19348327b072283bf819d2761285698f0a2c5376`。Capture 的 shader 对象包含以下入口，名称与当前源码对应；**没有导出并比对捕获 SPIR-V，不宣称捕获二进制与 HEAD 完全相同。对象列表也不是逐事件 pipeline 绑定列表。**

| Capture 中的 shader 对象 | 当前源码对应实现 |
|---|---|
| `streamClusterPrepareMain` | setup/count/prefix/scatter；count/scatter 已按 active groups 并行，只有 block prefix 单组 |
| `streamClusterCullMain` | 每 lane 一个候选，独立 cull；元数据快速分流，压缩精确分类工作 |
| `streamClusterBinMain` | 只对需要精确判断的 cluster 投影顶点、检测三角形 |
| `streamClusterRasterWorkControlMain` | 首 wave 协作加载、cluster 内共享屏幕顶点、不做局部工作量重排 |
| `gpuDrivenStreamAssetMeshMain` | prebinned 路径每 cluster 一次 indexed mesh 输出、唯一顶点复用 |
| `hzbSpdMain` | SPD HZB 入口；仅凭名字不能识别 wave-ops 编译变体 |

所以“并行候选展开”“把 cull 从 128-thread 分类中拆出”“共享顶点”“改成 indexed mesh”已经不是新的优化建议。元数据快速分类是否在此帧启用、HZB 是否有效、实际 HW/SW 数量，还需要 buffer/参数或应用诊断快照，不能仅由入口名得出。

API 流中有 10 次 `vkQueueSubmit2`、285 次 `vkCmdPipelineBarrier2`、56 次直接 dispatch、19 次 `vkCmdDispatchIndirect2KHR`、118 次 `vkCmdWriteTimestamp2`。这些是整个捕获帧的调用数量，包含其他 pass；既不证明 285 个 GPU 气泡，也不能把它们全归于 VBuffer。

## 优化候选顺序

下面按验证成本与实现关联排列，不是已测得的耗时排名。

### 1. 先做分流阈值与异步 A/B，找出实际昂贵的工作

当前 [默认 MiniZorah 管线](../Pipelines/Samples/gpu_driven_realtime.metallic_graph.json:49) 显式 `asyncSoftwareRaster=false`；[执行路径](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:3010) 已支持 early HW/SW 并行。软件尺寸默认 8 px，分类判断是三角形屏幕 bbox 宽高限制，并非覆盖像素数。该默认配置不等于已读取 Capture 的运行时属性。

第一组实验保持 LOD、camera、cut、驻留、render extent、jitter 不变：阈值 4/8/12 px × early async off/on，再加真正全 HW 的参照。记录候选、cull、精确 classify、stable bins、SW、HW、merge 及整段 Stream early/late 时间。降低阈值会把更多 cluster 交给 HW，增大会加重单 lane 扫描尾部；最佳值必须在 RTX 5060 上重测。开启异步的上限取决于两分支耗时比例及 SM/访存竞争，不能直接用 SW+HW 相加估算收益。

先检查 `paramsHzbValid`，并导出 early/late 的候选数、存活数、快速 SW/HW 与精确回退数。当前源码已经使用 `reprojectionInvalidationRevision()` 区分普通运动与 camera cut；不要把旧文档中的相机历史失效缺陷当成本次已证实的问题。当前 Capture 是第 55 帧，也不能默认代表充分预热后的稳定驻留。

### 2. 如果 SW 主导，优先压缩 cluster setup，再决定扫描算法

[loadStreamSoftwareWave](../Shaders/Features/GPUDriven/StreamRasterCooperative.slang:17) 仍需顺序追踪 params/header → active group → instance → page table → payload header → cluster。虽然首 wave 协作加载已实现，另外三个 wave 要等共享数据就绪。[WorkControl 路径](../Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang:6) 随后再等待顶点投影完成，才分配一 lane 一个三角形。

可比较两个独立方案：

- 在 cull 阶段写紧凑 raster descriptor（几何地址/count、instance/record 引用），让 SW/HW 消费已校验的描述，减少后续重复指针追踪和 header 校验。将额外写读带宽、descriptor 寿命/页面更新安全一起计入；不建议未经测量缓存所有投影顶点。
- 对低三角形填充率的 cluster 比较 32/64/128-thread 工作组，并通过 stride loop 完整处理顶点/三角形；32-thread 路径可研究 wave 内协作。需保留页面校验、uniform 退出和原始 triangle ID，不可只改 numthreads 造成漏面。

[hybridRasterPreparedTriangle](../Shaders/Modules/GPUDriven/HybridRasterTriangle.slang:91) 的扫描已经使用整数 edge stepping，但每 lane 仍串行跑 bbox 的双层循环，每个覆盖样本执行一次 64-bit `InterlockedMax`。利用已有 `softwareRasterWorkload` 诊断，分别看 triangle-slot fill、empty triangles、bbox visits、covered samples / atomic attempts：

- 顶点/三角形多、bbox visits 少：优先削减解码、投影和无效三角形 setup；不要先写复杂 tile rasterizer。
- bbox visits 多但覆盖率低、扫描尾部明显：再尝试微小 bbox 专用路径或让较大 bbox 多 lane 协作。
- 只有地址竞争/原子 stall 的证据成立，再评估 tile 内深度裁决；`InterlockedMax` 的存在本身不证明 atomic 饱和。

已有 [局部分桶对照](ZorahFullLocalWorkBins.md) 在另一台 GPU/另一场景状态上比同等屏幕顶点复用路径更慢。因此 `softwareRasterWorkBins=true` 只应作为本机实验项，不能作为无条件建议。深度增量算法也不能只看指令数，必须检查 depth/ID 等深语义。

### 3. 如果 HW 占比高，拆分 opaque 与 alpha-tested、单双面管线

这是当前源码中一个仍然存在的具体额外成本。[流式 graphics pipeline](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:1723) 使用 `CullMode::None`；[fragment](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4660) 先用 `SV_IsFrontFace`/doubleSided 执行 discard，再调用 `streamAlphaAccepts()`。Opaque 也走到 bindings/instance flags 查询，mesh 输出仍携带 UV、instanceIndex 等字段。

按 opaque/masked 与 single/double-sided 分 HW bins：opaque 单面使用固定功能背面剔除及只输出 position+primitive visibility ID 的 shader；masked 保留 alpha coverage 逻辑。resident 路径已有 alpha 编译变体可参考。这有机会减少 fragment 工作、属性导出及动态分支，但要以 HW 区间/fragment workload 验证收益。必须保留反射实例绕序、正反面定义和双面/alpha 语义，不能全局打开背面剔除。

### 4. 如果 late 或 stable bins 仍显著，优化控制路径

- [late 候选准备](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3942) 已利用 retry mask，但 count/scatter 仍覆盖 active groups。若 late 候选占比低，可由 early 建紧凑 retry-group 队列，并并入 late 新可见实例；保持稳定 group/cluster 顺序及 overflow fallback。不能因为一个静态帧 late 输出少就删除恢复路径。
- [stable histogram](../Shaders/Features/VisibilityBuffer/VisibilityHybridRaster.slang:119) 每有效 lane 扫描前面 lane 的 tag 计算 rank；满组同桶时有 8128 次源级 tag 比较。可用每 wave 的 ballot/prefix count，加 4 个 wave 的桶计数前缀代替，保持稳定 scatter 顺序。这是算法工作量分析，尚不是机器指令计数或时间测量。
- [candidate scan](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3921) 当前是共享内存 Hillis–Steele scan，可评估 wave scan + 跨 wave prefix。它已并行化；不要再把整个候选阶段描述为单组串行遍历。
- 空 late 仍有 clear、若干 dispatch、merge/HZB 的固定成本。只有确认其比例值得处理，再用合法的 GPU 条件执行或零工作间接调度缩短路径；避免 CPU readback 决策引入新的等待。

## 验收与缺失数据

在相同状态下分别计时与诊断；诊断重放、buffer readback 不混入计时帧。至少覆盖空候选、late 解除遮挡、相机切换、近裁面、单双面、alpha、反射实例与容量溢出。保持 depth/visibility 与对应参考一致；改变 HW/SW 分流时，需单独说明已有 HW/SW 浮点差异。验收看完整 VBuffer 和 GPU graph 的改进，不能只看局部 kernel。

要把以上候选排成有把握的收益优先级，仍缺本 Capture 对应的 GPU Trace：early/late 子阶段时间、SW/HW 时间交叠、active lanes、occupancy、barrier/long-scoreboard、SM/L2/DRAM 吞吐，以及实际候选/分桶/几何工作量。当前无法声称“SW 是本帧最大项”“显存带宽不足”或“预计提升 X%”。

## 本轮证据

### 使用 cli-anything-nsight-graphics skill 复查

按用户要求，随后通过已安装 skill 的私有 `.runtime/Scripts/cli-anything-nsight-graphics.exe` 继续验证，harness 0.2.0：

- `doctor info` 返回 `ok=true`，确认 unified+split 模式和 GPU Trace Profiler 活动支持；`doctor versions` 只发现 Nsight Graphics 2026.3.1。
- `replay analyze --metadata --logs` 返回 `ok=true`，读取 602 个对象、1555 个 API 事件，捕获日志中 severity ≥2 的错误数为 0。这不代表后续 GPU Trace 初始化成功。
- 第一轮 `gpu-trace capture --auto-export --summarize` 失败。harness 抛出“refusing to summarize stale export tables”，只转述空 stderr，省略了底层 stdout。
- 第二轮去掉 `--summarize`，让原始执行结果完整返回；同时不传 `--platform`，避免上一轮遇到的 Qt platform 参数解析问题。实际官方命令包含 `--architecture "Blackwell GB20x" --metric-set-id 0`，30 presents 后抓 1 帧、最长 1000 ms，产物放入全新目录。
- **第二轮结果：`ok=false`、`returncode=1`、`artifact_count=0`。** stdout 明确显示目标进入 Presenting，随后 `Session established. Starting activity...`，再出现 `ERROR: No single-pass metric set selected. GPU Trace can't proceed.`。这把阻塞定位到官方 ngfx 的 profiling 初始化阶段；还不能进一步确认是设置、驱动或权限问题。没有因该报错更改驱动、全局设置或安装版本。
- 在 ngfx 已退出、无 Trace 产物后，清理本轮创建而仍驻留的回放进程，使继承的 stdout 管道关闭并让 harness 返回。未终止用户程序。

因此 skill 复查没有增加有效的 GPU 时间/计数器证据，上述优化建议仍是待验证候选。继续定量分析所需的是同一 Capture 回放的有效 `.ngfx-gputrace` **及导出目录**（`FRAME.xls`、`GPUTRACE_FRAME.xls`、`D3DPERF_EVENTS.xls`，最好包含 `GPUTRACE_REGIMES.xls` 和 `REPRO_INFO.xls`）。现有 skill 可直接 `gpu-trace summarize --input-dir <exact-export-directory>`；它不能从失败的采集或仅有的 `.ngfx-capture` 推算出这些时间。

证据：[Skill 原始 GPU Trace 执行结果](../build/nsight-visibility-20260924/skill-trace-2/result.json)、[Skill replay analyze 结果](../build/nsight-visibility-20260924/skill-replay-analysis-result.json)、[复现脚本](../build/nsight-visibility-20260924/RunSkillTrace2.ps1)。

### 原始回放与源码分析

- [结构化摘要](VisibilityBufferNsight20260924.json)（不包含捕获的用户环境变量）。
- [原始 Capture 图](../build/nsight-visibility-20260924/capture.png)、[API 列表](../build/nsight-visibility-20260924/functions.txt)、[对象列表](../build/nsight-visibility-20260924/objects.json)。
- [普通回放输出](../build/nsight-visibility-20260924/replay)、[回放日志](../build/nsight-visibility-20260924/replay.log)。
- [首轮 Trace 失败日志](../build/nsight-visibility-20260924/trace-1.log)、[最终 Trace 失败日志](../build/nsight-visibility-20260924/trace-5.log)、[最后尝试的采集脚本](../build/nsight-visibility-20260924/RunTrace.ps1)。
- [摘要生成脚本](../build/nsight-visibility-20260924/AnalyzeCapture.py)。
