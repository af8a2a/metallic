# VSM CPU 开销细分

2026-10-28，基于包含 resident caster cache 的 `6631ee04a`。本轮添加默认关闭的 CPU 诊断和可复核报告，没有改变 VSM 分配策略、shader、灯光、页预算或默认图。结果用于选择下一轮优化目标；上一轮普通 A/B 仍见 [CPU 优化记录](VirtualShadowMapsCPU.md)。

后续优化与测量：[资源复用与池化](VirtualShadowMapsResourcePool.md)、[紧凑页变更上传](VirtualShadowMapsCompactUpdates.md)。下文耗时保留原基线口径。

## 当前开销与优先级

固定相机、temporal jitter 开启、输出 768×512、DLSS Quality 实际内部 512×341。每场景 3 个诊断开启进程，每进程 128 帧，排除前 16 帧。表中为相同保留帧的**排他耗时均值**和占 Shadows node CPU 的份额；父区间扣除直接子区间，不相加重复计时，也不相加阶段中位数。所有进程等权，计时包括诊断本身的成本。

| 阶段 | LookDev ms / 占比 | Sponza ms / 占比 |
| --- | ---: | ---: |
| 旧录制状态替换 / 相关资源释放 | 5.5808 / 54.05% | 0.7793 / 48.88% |
| Buffer 创建 | 2.9104 / 28.19% | 0.4041 / 25.35% |
| CPU 页表重映射 | 0.5488 / 5.31% | 0.1599 / 10.03% |
| Demand / build / raster 命令录制 | 0.4169 / 4.04% | 0.0589 / 3.69% |
| Buffer map / memcpy / flush / unmap | 0.2394 / 2.32% | 0.0323 / 2.03% |
| Buffer descriptor acquire | 0.1006 / 0.97% | 0.0122 / 0.77% |
| 投影准备与命令录制 | 0.1041 / 1.01% | 0.0103 / 0.65% |
| MaterialValueProgramSet 创建 | 0.0948 / 0.92% | 0.0194 / 1.22% |
| 其余 VSM 工作与父区间残差 | 0.3010 / 2.92% | 0.1069 / 6.70% |
| Shadows node 内、VSM execute scope 外 | 0.0291 / 0.28% | 0.0108 / 0.68% |
| **Shadows CPU 总量** | **10.3257** | **1.5942** |

前两项合计 **82.23% / 74.23%**。这是已测成本所在的位置，不是可以全部消除的收益上限或已实现加速。GPU 页缓存稳态仍然命中：每帧 reused faces 为 8 / 1，dirty bounds 和 pipeline cache miss 均为零，但临时 CPU/RHI 资源仍重复创建。Caster textures / geometry 阶段仅 0.00343 / 0.00498 ms，继续优化 caster 已不是这两个静态 resident 场景的主要方向。

建议按以下顺序提出和验证候选：

1. **在 VSM 内复用每面临时 buffer，先减少 native 创建和释放。** 每帧重新创建的工作缓冲、统计、间接参数等是首要目标；按 GPU completion 管理可复用 slot / 容量，保留在途录制和上一帧 page table 的寿命。先在 VSM 范围验证，当前数据不足以支持全局 RHI 分配器改造。验收需要缓存/重绘一致、取消与多帧在途寿命、resize / page budget 变化、CPU 普通 ABBA 和 live allocation 峰值。
2. **减少相同页映射的 CPU 构造与上传。** 静止局部灯、未跨页移动的方向光且没有 dirty bounds 时，有可复用内容；必须把 origin、projection、caster invalidation、forceRedraw 和 budget/settings 变化纳入正确性条件。当前重映射加全部 buffer 上传占 7.63% / 12.06%；其中上传还包含 view / material 等内容，不能全部算为可节省的 remap 成本。
3. **再评估命令录制和小型共享数据。** Demand/raster CPU 约 0.417 / 0.059 ms；material program、descriptor 和点光投影表不是当前首要项。不要从这些 CPU 区间推断 GPU shader 性能。

## 资源和调用链证据

| 稳态每帧工作量 | LookDev | Sponza |
| --- | ---: | ---: |
| 录制灯面 | 8：方向光 1、spot 1、point 6 | 1：方向光 |
| 基础 face buffers | 80 | 10 |
| Device buffers | 40 | 5 |
| HostUpload buffers | 40 | 5 |
| 基础 face buffer 逻辑字节 | 89,169,824 | 11,164,056 |
| 其中 HostUpload 数据 | 6,300,608 | 791,032 |

这些计数对应 `record()` 的十个 buffer 槽；不包含 LookDev filtered point 的六个额外 192 B face table，也不是 Vulkan 分配 padding、实际 VRAM 或峰值。在采样区间，每个 face 的大表尺寸为 98,305 项。

[VirtualShadowMaps.cpp](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp) 中 `device.createBuffer()` 的五个 Device 槽是 pages/work/statistics/indirect/owners。[VulkanRHI.cpp](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp) 的 `allocationInfoForMemory(Device)` 使用 `VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT`；创建包含 memory-budget mutex、requirements/admission 和 `vmaCreateBuffer`。

`state_ = state` 可以释放旧 State 的最后引用；旧 buffers、views、leases 和 cache 引用随之释放。Buffer wrapper / allocation 最后引用进入 `BufferImpl::~BufferImpl()`，在 memory-budget mutex 内调用 `vmaDestroyBuffer`；dedicated allocation 的 VMA 路径进一步调用 `vkFreeMemory`。该链没有 `Device::waitIdle`。这里测到的是**整个状态替换边界**，尚未把 mutex 等待、驱动 free 和其他析构分别计时，不能把 5.58 ms 全标成 `vkFreeMemory`。

Compact output 的 atlas 在 reuse 时复用，output texture view 创建是 CPU wrapper；texture/view/image/depth/sampler lease 仍合在一个小阶段内。Point face table 的 24 个跨面 lease、buffer 和 barrier 也保持合并计时。描述符数字仅明确指独立测量的 buffer acquire，不代表全部 registry 成本。

单面数据保存在报告中。Point 第 5 面的 inclusive record 均值约 1.776 ms，其他 point 面约 1.115–1.248 ms。每个旧 State 的 point-face leases 都持有六面的 projection/statistics；按 0→5 替换时，前五面的 allocation 可持续存活到旧 face 5 被替换。因此 face 5 可以多承担五个 statistics 和五个 projection allocation 的释放，不能直接把耗时差异当作该面几何或 GPU 负载差异。稳态 page table / atlas 仍由 cache 持有，不属于这批最后释放；比较优化时应累计整盏点光源或所有 faces。

UE 对照与下一轮候选拆分见 [Unreal VSM CPU 工作研究](VirtualShadowMapsUnrealCPU.md)：资源池/子分配、compact 页变更、跨灯面批处理及增量失效的源码锚点均已核实。

资源复用已实现，稳态创建计数、寿命回归及普通 ABBA 结果见 [VSM 资源池](VirtualShadowMapsResourcePool.md)。该轮确认 CPU 收益，同时保留 LookDev Shadows GPU 增加和 Sponza 整图 GPU guard 未通过的边界。

## 计时和复现合同

两个 VSM Pass 支持 graph property `benchmarkCPUProfile: true`；默认 false。`VirtualShadowMaps::record/project` 接受可选 `CPUProfileRecorder*`，沿用纯 CPU scope，不增加 GPU timestamp 查询。十个 buffer 的 create/upload/descriptor 子区间分别累计发布，LookDev 共 210 sections，低于既有每 node 256 上限。默认不启用 `benchmarkCPUThreadDiagnostics`；此结果是主机 elapsed time，包含抢占/驱动等待，不能称为纯线程执行时间。

Acceptance JSON 增加 `cpuOnly`、`cpuWork`、可选 `cpuThreadProfile`，并导出 `stats.preparation`。开启 CPU attribution 必须使用 diagnostic 模式，`performanceEligible=false`。默认 runtime 行为和普通 timing 验收规则保持原样。

录制区间外的共享 preparation 单独列出：LookDev / Sponza 的 command pool reset 约 0.00361 / 0.00330 ms、frame begin 约 0.00037 / 0.00037 ms。`RenderFrameContext::begin()` 的 wait/resources clear 和提交槽准备不在 Shadows CPU 内，也不能全部归入 VSM。Node 与 VSM 根区间的差值、各父子差值全部保留；最大逐帧闭合误差为浮点舍入量级 `1.78e-15 ms`。

固定采样顺序为每场景 OFF、ON、ON、OFF、OFF、ON，共 12 个独立进程；同一 Release / Streamline ON / NRD OFF 可执行文件。磁盘缓存保留，每进程创建新 scene/device，GPU 实验串行、各进程有超时，没有保证系统 GPU 独占或改变时钟。不开 core/sync validation，保留验收 harness 的 diagnostic metadata readbacks，第 127 帧捕获最终颜色。帧只作为回放分布，不作为独立实验样本；本轮没有优化收益 CI。

诊断开关的进程中位数有交叠和波动，不能证明零测量开销。此表的诊断均值不能与上一轮普通计时中位数直接相减、解释为回退，或推算编辑器 FPS。大型 cooked assets 延续路径身份限制；源码、二进制、小型 graph/case 配置和各证据文件 SHA256 已核对。

原始包及输入快照：[Plan / Runs](../.cache/vsm-cpu-breakdown/Plan.json)、[Inputs](../.cache/vsm-cpu-breakdown/Inputs)、[采集脚本](../.cache/vsm-cpu-breakdown/RunPlan.py)。报告工具检查完整 manifest、运行状态、跨进程 source/binary 身份、场景配置仅 profiling 开关变化、实际尺寸、排他闭合及输出 hash；输出不同会明确标记，不删除该进程。

```powershell
python -B Tools/Perf/VirtualShadowMapCPUProfile.py .cache/vsm-cpu-breakdown <new-report-directory>
```

[完整报告](../.cache/vsm-cpu-breakdown-report-final/report.md)包含阶段、单面、工作量和 preparation 表；[Summary.json](../.cache/vsm-cpu-breakdown-report-final/Summary.json)、[逐进程统计](../.cache/vsm-cpu-breakdown-report-final/Processes.json)、[逐帧 CSV](../.cache/vsm-cpu-breakdown-report-final/Frames.csv)、[复现脚本](../Tools/Perf/VirtualShadowMapCPUProfile.py)均保留。已运行 Python / Matplotlib 绘图并检查图例、坐标与裁切。

![CPU 排他开销](../.cache/vsm-cpu-breakdown-report-final/CPUAttribution.png)

## 正确性与限制

12 次采集均完成、无 SDK/runtime error，源码/输入/二进制在采集期间未变；每场景跨进程源码和二进制一致，graph 除 profiling 开关外相同。LookDev 六次最终输出 raw SHA256 全相同。**Sponza 六次中一次（第 3 进程、ON）与其余五次不同，原始输出控制未通过。** 全部帧和进程保留在归因统计，没有择优重跑。

不同的 Sponza 图像有 10,681 / 393,216 像素变化（2.716%），12,470 个 RGBA 分量不同，其中 12,026 项仅差 1/255，最大差 24/255，RMSE 0.10665/255。最终 shadow header/statistics、拓扑、材质纹理语义与 resident pages 相同；但前 2–5 帧异步 geometry residency 不同。与早期驻留差异遗留 temporal history 的解释相符，仍未定位到具体 temporal pass，不能证明 DLSS 是唯一原因。

另预先固定一次 OFF / ON 补充检查：48 帧、关闭 jitter，第 16 帧驻留稳定后 camera cut，第 47 帧最终捕获和 cache/force-redraw 比较；core validation 开、sync validation 因既有 Streamline 边界关闭。两个进程通过，最终输出 raw hash 相同，depth/shadow 跨模式 hash 相同，各自 cache/force-redraw shadow 和 HDR 逐字节相同且 HDR 无非有限分量。见[补充检查计划](../.cache/vsm-cpu-breakdown-history-check/Plan.json)和[比较结果](../.cache/vsm-cpu-breakdown-history-check/Comparison.json)。这支持稳定历史下 profiling 开关不改变结果，但不覆盖或解除原先 jittered 控制失败。

`MetallicRHITests` 与编辑器 `Metallic` 已构建成功；VSM content 回归 50 组 cached/uncached GPU 几何和阴影比较、14 个拒绝边界及 gated lifetime 检查通过，core/synchronization validation 零消息，见[回归 JSON](../.cache/vsm-cpu-breakdown-content/VirtualShadowMapContent/VirtualShadowMapContent.json)。没有 shader 修改。本轮没有编辑器交互、动态几何、stream-only VSM 或 Full Zorah 的性能验证，也未改变 P5 的默认提升状态。
