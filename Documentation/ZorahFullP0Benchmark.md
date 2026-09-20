# ZorahFull P0：编辑器漫游与分项计时

2026-09-20。P0 提供测量入口，不改变 LOD、阴影、材质质量或流送调度策略。

## 使用入口

GPUDrivenSample 的 **File → Benchmark ZorahFull (180s)**：加载 Full 内置预设，锁定当前视口尺寸，根页就绪并在起点暖机 5 秒后开始采样。路线为静止、前进 6 个场景单位、左右转向观察、返回、静止收敛。位移和转向按经过时间插值，与帧率无关；按 Escape 可中止。默认输出到 `Captures/FullRoam/<时间戳>/`。

批量运行使用 PowerShell 7，从仓库根目录执行：

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-roam-new `
  -Runs 3 -DurationSeconds 180 -WarmupSeconds 5 `
  -Width 1797 -Height 660
```

不传 Width/Height 时使用新编辑器窗口读取的布局，捕获后锁定实际视口；它不代表另一个已打开窗口的精确尺寸。要严格复现当前前台视口，使用该窗口内的菜单入口，或者明确传入它的渲染尺寸。CLI 使用隐藏窗口，不操作主桌面鼠标/键盘，也不覆盖用户的 `imgui.ini`。隐藏窗口仍执行完整编辑器渲染、swapchain 提交与 Present，但 DWM/显示调度可能不同于前台窗口，结果必须标明此条件。

`-Validation` 用于正确性检查，其时间不进入性能结论。`-RouteConfig` 可传入 JSON，包含 `distance` 和 `keyframes`；每点有 `t`（严格递增且首尾为 0/1）、`forward`（相对总距离）、`yaw`（度）和 `stage`。实际绝对 camera keyframes 会写入报告。CLI 对每轮启动独立进程，保存 GPU 频率、温度、功耗、整卡使用量及各进程引擎占用。

## 输出和归属

- `Manifest.json`：可执行文件/shader 摘要、Git HEAD/dirty 列表、GPU/驱动、路线配置、资产路径/大小/修改时间。大型 meshstream 的身份是文件元数据，未声称做全文件内容哈希。
- `Capture.json`：实际输出/内部渲染尺寸、完整 graph 参数、绝对路线、加载耗时、scope 字典、GPU 缺失帧数及采样完成状态。`capture_complete` 表示采样完成，**不是通过 30 fps**。
- `Frames.jsonl`：逐帧 start-to-start 时长、路线阶段、CPU/GPU scope、GPU execution ID、几何/CLAS/纹理驻留与请求计数、预算可用量、BLAS 构建数量/cluster references/overflow 和反馈源帧。
- `Uploads.jsonl`：已完成发布的独立纹理上传，包含提交/完成观察帧、GPU timestamp 时长、上传量和 CPU 观察的完成延迟。后者包含排队及轮询延迟，不是 GPU 执行时间。该文件可能带入起点暖机末尾的一次已完成记录；汇总器按采样 execution ID 筛选。
- `Summary.json` / `Summary.md`：整体及阶段 p50/p95/p99/max、超过 33.33 ms 的帧数、最长连续超预算段、各 scope 分布、慢帧及对应流送状态。

Scope 平均值只使用实际出现的有效样本；条件执行 scope 的平均值是每次出现的成本，并非用全帧数归一化。父子 scope 不能相加。`allocationFailures` 是本帧页面准入/池分配尝试失败计数，可重复包含同一页，不等同于 Vulkan OutOfMemory。BLAS overflow 的具体原因仍需结合容量/回退路径细分。

单独重新汇总：

```powershell
python Tools/AnalyzeZorahFullRoam.py build-release/full-roam-new/run1
```

CPU/GPU 用 graph generation + execution ID 关联；GPU 完成后回填原 CPU 样本。UI 的 500 帧历史不再截断批量记录；捕获有 60000 帧上限，越界、graph/视口变化、GPU timing 缺失会使捕获失败。结束采样之后再等待 outstanding graph queries，不在每帧插入同步图像读回。CPU 帧时间包括采样维护开销。

## 新增计时

Stream traversal 中增加 **Fallback BLAS、BLAS reset/cut compare/count/setup/insert/build、TLAS input/build**。BLAS header 的 32 字节统计附加到既有完成反馈，没有额外 CPU 等待。构建数量是实际 GPU 反馈，不能把提交了 build 命令等价为全部 BLAS 重建，也不能用总实例数减 build 数伪造缓存命中率。

Deferred 中增加 CPU-only 的 **Texture streaming** 子树：反馈消费、迁移完成轮询、上传查询解析、generation 发布、image/staging 准备、worker 释放、上传命令准备/录制/提交、旧 generation 退休、候选/尺寸查询、排序准入、worker 启动及反馈缓冲准备。

独立纹理迁移提交拥有自己的 query pool，在自身 graphics command buffer 的首尾记录 timestamp，只在该提交已完成后读取和销毁。Profiler → Streaming → Texture Residency 显示最近已完成上传的 GPU ms、提交帧和完成观察帧。不能把它挂在当前 Deferred 帧下冒充同帧 GPU scope，也不能将它与嵌套 RenderGraph scopes 直接求和。

RenderGraph GPU envelope 不含编辑器合成/Present，也不包含另行提交的纹理上传；start-to-start editor frame 则覆盖整个交互循环。纹理现有请求延迟仍是入队至发布，不包括候选等预算的时间，尚不能称为端到端画质收敛延迟。

## 验证

- Release Sample / RHI tests 构建通过。
- `ktx2_texture_streaming` validation 回归通过，新增检查 CPU profile 树、独立上传样本数量、帧序关系与 GPU timestamp 可用性；原有细化、冷回收、取消反馈、共享预算与旧/新资源生命周期检查继续通过。
- Full 隐藏编辑器短程 validation 路线完成，BLAS/TLAS 每个采样帧都有 GPU 查询，未发现 validation/device 错误。validation 耗时仅用于诊断。
- 12 秒非 validation 检查完成，计时导出和汇总通过。它经过时间缩放且曾与测试构建并行，不作为正式性能基准。

正式三轮结果和推进建议见下方实测补充。


## 三轮正式基准

RTX 5070 Ti，输出 **1797×660**，DLSS Quality 内部 **1198×440**，Full 默认完整材质/阴影、LOD 1.5 render px、普通纹理按需细化上限 512。每轮独立进程，根页就绪后起点暖机 5 秒，固定路线 180 秒。隐藏编辑器；validation、逐帧截图和详细 smoke 日志关闭。磁盘/shader 缓存保留，不宣称冷缓存性能。

| 轮次 | 采样帧 | p50 ms | p95 ms | p99 ms | 最大 ms | >33.33 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1530 | 119.126 | 134.431 | 146.538 | 172.135 | 1530 |
| 2 | 1518 | 119.764 | 134.308 | 152.797 | 295.189 | 1518 |
| 3 | 1609 | 113.021 | 128.330 | 135.686 | 176.079 | 1609 |

合计 **4657 帧**，GPU frame/scope 归属检查通过，没有缺失的 graph GPU 样本。当前同条件均未达到 30 fps，三轮平均吞吐约 8.4–8.9 fps。软件光栅/分类的成本在三轮间稳定，而 CPU 开销及长尾仍有波动。

| 项目 | 三轮均值范围 | 解释 |
| --- | ---: | --- |
| RenderGraph GPU envelope | 约 85–86 ms | 不是完整显示帧，也不能与以下子项相加 |
| Early software raster GPU | 50.6–51.1 ms | 最大已测 GPU 项 |
| Early soft/hard classification GPU | 21.8–21.9 ms | 第二大已测 GPU 项 |
| TLAS build GPU | 0.340–0.347 ms | 完整 Build 存在，但目前不是主要成本 |
| BLAS build GPU | 约 0.19–0.20 ms | 需要结合实际 build/overflow 看，不能推断为高缓存命中率 |
| CPU Stream Begin | 11.94–15.75 ms | request feedback/deduplicate/admission 是主要子项 |
| CPU Texture streaming | 约 1.3 ms | 条件执行的 image 准备和 generation 发布成本值得后续收敛 |
| 独立纹理上传 GPU | 0.0231–0.0238 ms/批 | 1293 个采样 execution 中已完成的上传，GPU timestamp 全部有效 |

几何池使用率始终高于 **99.2%**；每帧页面准入/池分配失败尝试约 **8451–16219**，BLAS header overflow 峰值 **12430**。这不是设备内存分配错误日志，但证实应检查容量饥饿、重复准入和 RT fallback。当前数据没有证明固定 1.5 px 已经在所有可见区域收敛，也没有做全场景画质验收。

各轮保存了整卡与各进程 GPU 负载，未观测到 Unity/vk_lod_clusters 等其他大型渲染器；DWM、聊天及桌面应用仍有小幅引擎负载。Windows Get-Counter 偶有无效样本，错误保存在 `GpuProcesses.stderr.txt`，有效样本继续记录；不能据此声称严格独占 GPU。第三轮 GPU 主项与前两轮一致，CPU 时间更低，保留分轮结果而不只汇总一个平均 FPS。

证据：

- [三轮对照](../build-release/full-roam-p0-baseline/Comparison.md)、[启动与二进制记录](../build-release/full-roam-p0-baseline/Manifest.json)。
- [第一轮](../build-release/full-roam-p0-baseline/run1/Summary.md)、[第二轮](../build-release/full-roam-p0-baseline/run2/Summary.md)、[第三轮](../build-release/full-roam-p0-baseline/run3/Summary.md)。同目录保留原始 Frames/Uploads、GPU 与进程 CSV。
- [仓库内精简结果](ZorahFullP0Results.json)，包含分轮 frame/scope/内存/准入/BLAS overflow 和其他进程的最高已报告引擎占用。
- [纹理流送 validation 回归](../build-release/full-roam-p0-unit.log)、[Full validation 捕获](../build-release/full-roam-p0-validation/run1/Capture.json)。
- [额外五项回归](../build-release/full-roam-p0-regression.log)：Editor profiler history、延迟 GPU 捕获归属（超过 UI 500 帧、停止捕获后回填、graph reload 重用 ID、重新捕获）、render graph GPU profiling、CLAS runtime lifecycle、BLAS cut cache，全部通过。

## 基于 P0 调整下一步

1. **先处理软件光栅和软硬分类工作量。** 在固定画质下记录实际 cluster/triangle 数、分类结果与软件线程/像素工作量，做阈值和硬件路径对照，再选择具体内核/调度优化。软件光栅单项已超过 33.33 ms，不能只优化偶发长帧。
2. **同步解决容量与准入问题。** 审计 Full 根页、可细化容量和 BLAS 每实例容量/回退原因；减少重复失败请求的 CPU 消费。将细化是否收敛作为性能验收约束。
3. **TLAS Update 和纹理上传优化降为后续项。** 它们仍有价值，但现有平均 GPU 时间不足以解释与 30 fps 的差距。纹理 CPU generation 发布/创建可以做长尾收敛，不应作为当前主要 GPU 提速方案。

P0 没有改这些调度/质量策略；这些方向留给下一阶段实施。
