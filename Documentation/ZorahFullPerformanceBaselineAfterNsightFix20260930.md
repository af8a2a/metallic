# ZorahFull 修复后的 2560×1440 生产漫游基线（2026-09-30）

当前最终生产构建重新运行三轮原始 live 漫游，固定 **2560×1440 输出 / 1707×960 DLSS Quality**。独立进程帧均值的中位数为 **16.779243 ms**（倒数约 **59.597 FPS**），三轮均值范围 16.635656–16.905329 ms。所有长帧保留；三轮 `sampledRouteMeets30Fps` 均为 false，不能称为严格稳定 30 FPS。

这是最终生产路径的独立基线，未混入固定视角 candidate A/B、validation 移动漫游或 SDK capture/replay。每轮 ready 后预热 10 s，按墙钟测量 30 s；6 m 往返与 ±65° 转向、时间抖动开启、隐藏编辑器、VSync 关闭、Reflex 默认、两个 frame slots。默认 Device metadata、生效的 realtime Deferred、非冻结；无 Nsight 注入、shader debug、validation 或工作量 instrumentation。

| 运行 | 帧数 | 均值 ms | P50 ms | P95 ms | P99 ms | 最大 ms | >33.333 ms | 最长连续超预算 | readiness s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 | 1788 | 16.779243 | 15.7750 | 24.6787 | 27.8118 | 48.8032 | 2 (0.1119%) | 1 | 7.6640474 |
| run2 | 1775 | 16.905329 | 15.4445 | 25.1844 | 28.9656 | 59.5542 | 4 (0.2254%) | 1 | 5.6937630 |
| run3 | 1804 | 16.635656 | 15.7906 | 24.1163 | 27.8969 | 37.6342 | 2 (0.1109%) | 1 | 7.5609050 |

`frameMs` 为编辑器循环 start-to-start，包含等待、事件、录制、提交、present 和采集记账。没有池化三进程帧来推断显著性；中位数和范围只描述这三次运行。`loadingSeconds` 从加载 draw loop 开始到首次 ready 且 preview 有效，排除此前应用/设备/场景初始化和 ready 后预热，不能称为完整启动时间。OS/shader/pipeline 缓存未清空，冷热程度未单独测量。

## 主要 GPU 范围

以下均为 inclusive scope 的每轮均值，单位 ms；父子不能相加。Graph envelope 不含编辑器合成/present 与独立纹理上传提交。

| GPU 范围 | run1 | run2 | run3 | 进程均值中位数 |
|---|---:|---:|---:|---:|
| Graph GPU | 13.048930 | 13.164241 | 12.960612 | 13.048930 |
| VBuffer | 9.797483 | 9.871592 | 9.722201 | 9.797483 |
| Traversal | 3.447276 | 3.467390 | 3.432985 | 3.447276 |
| Deferred | 1.060768 | 1.057712 | 1.057434 | 1.057712 |
| Deferred shade | 0.941944 | 0.940664 | 0.938429 | 0.940664 |
| SW early | 2.078980 | 2.091219 | 2.062602 | 2.078980 |
| SW late | 0.135263 | 0.134272 | 0.131755 | 0.134272 |
| HW early | 1.257809 | 1.265336 | 1.260877 | 1.260877 |
| HW late | 0.072581 | 0.073627 | 0.068943 | 0.072581 |
| BLAS build | 0.374895 | 0.376284 | 0.378752 | 0.376284 |
| TLAS build | 0.301546 | 0.307160 | 0.298825 | 0.301546 |
| DLSS pass | 1.109255 | 1.122898 | 1.097770 | 1.109255 |
| AutoExposure | 0.576835 | 0.582649 | 0.574592 | 0.576835 |

每个列出的 GPU 范围覆盖全部测量帧，`missingGpuFrames=0`。精确完整路径、GPU P50/P95/P99、CPU pacing/maintenance/texture streaming、分阶段数据和 marker frame scope ID 已保留到机器可读报告。所有 marker 与逐帧 shader 身份为默认 group32/wave32、`streamClusterRasterGroup32Main`，SPIR-V 指纹跨帧/跨轮相同，且未冻结。

## Metadata、反馈与流送完整性

输入配置未覆盖 `deviceImmutableMetadata`；Capture 实际配置和所有逐帧状态均为 true。Immutable group+topology payload、allocation、submitted bytes 均为 **298,294,832 B**；ready 每帧成立，staging bytes 为零，upload batches 为 5。Editor/runtime/route sample 与 graph scope execution 顺序已核验；request/BLAS 反馈可用，没有未知或未来帧。

| 运行 | 页 uploads | 页 evictions | 页 upload bytes | allocation denial min–max / frame sum | Missing CLAS 峰值 | Publication invalidated 峰值 |
|---|---:|---:|---:|---|---:|---:|
| run1 | 104205 | 105033 | 6642787040 | 0–1 / 1321 | 586 | 29781 |
| run2 | 101636 | 102461 | 6477158688 | 0–1 / 1309 | 552 | 27991 |
| run3 | 104715 | 105542 | 6674822832 | 0–1 / 1329 | 563 | 28293 |

Allocation 字段在每轮 maintenance 开始清零，随后预算准入/存储分配拒绝会递增；frame sum 是重复尝试数，不是唯一失败页面，也不等于致命错误。Missing CLAS 与 publication invalidated 为反馈状态快照，未累计成唯一失败数。

| 运行 | accepted texture feedback first→last（差） | texture upgrades first→last（差） | texture upload bytes first→last（差） |
|---|---|---|---|
| run1 | 595→2383 (1788) | 1130→2699 (1569) | 201749728→543623344 (341873616) |
| run2 | 567→2342 (1775) | 1068→2687 (1619) | 186500320→541154256 (354653936) |
| run3 | 624→2427 (1803) | 1169→2699 (1530) | 209658800→544344240 (334685440) |

核心 load failure/request overflow/BLAS overflow/invalid group 与 BLAS 预算/存储拒绝最大值：`{"loadFailures":0,"requestOverflows":0,"blasOverflowCount":0,"blasInvalidGroups":0,"blasReferenceBudgetRejected":0,"blasBuildBudgetRejected":0,"blasStorageRejected":0}`。原始日志 VUID/device loss/error 模式为零。纹理反馈/升级/上传字段是累计值，上表使用末减首，未将重复快照相加；纹理请求帧延迟定义仍排除预算准入前等待。

取消数量并未由逐帧 schema 导出，因此不声称 baseline 中 cancellation=0。当前源码只在 completion submitted 且 readback transaction resolved/未取消时读取 staging；取消 tail 的契约由独立最终 RHI validation 记录覆盖，7 pass / 0 skip，普通三轮漫游本身没有故障注入取消测试。没有将当前 Graphics 生产者/消费者的既有契约外推为未来跨队列支持。

## 显存、背景活动与身份

| 运行 | 整卡采样峰值 MiB | 峰值首次时间 | GPU CSV 全区间 |
|---|---:|---|---|
| run1 | 13789 | 2026/09/30 20:43:04.155 | 2026/09/30 20:42:13.420–2026/09/30 20:43:06.176 |
| run2 | 13610 | 2026/09/30 20:43:41.997 | 2026/09/30 20:43:06.429–2026/09/30 20:43:57.215 |
| run3 | 13714 | 2026/09/30 20:44:27.963 | 2026/09/30 20:43:57.428–2026/09/30 20:44:50.228 |

这些约 1 Hz 记录覆盖整个进程，包含初始化、加载、预热、测量和 teardown；没有显式 UTC capture origin，无法精确裁成测量时窗。整卡显存包括后台进程，不能当 renderer 独占。GPU Engine 为 ≥0.1% 的单 engine 条件采样，不相加、不把缺失记录置零；完整背景进程 peaks/窗口、GPU clocks/temp/power 与 geometry/CLAS/scratch/texture/BLAS pool 范围均保留。相同参数不证明 live texture working set、各图片 mip 或 OS cache 内容完全一致。

- Git HEAD `d21e01b9729dd561a60e6a9c56b393d7d0d58124`，包含 manifest 记录的未提交永久修复；EXE/DLL/源码快照以当前证据包为准。
- 最终生产 EXE SHA-256 `1413B7CAD9072547426C8FBFC0935F97A86B01CB38B7B27D5A5E1A1E2E829E92`；shader tree SHA-256 `5E0AD05CF6D947A9AE880537509A6D6C24E80F08C65443C7FC16F680470E3A38`，与独立 final-default acceptance 一致。
- Meshstream 长度 **54,921,584,816 B**，mtime UTC **2026-09-30T01:29:26.1897670Z**。与历史 published identity 的长度/mtime 一致；本次没有重新读取 54.9 GB 内容计算 SHA，不能将历史 hash 宣称为本次独立内容验证。

## 历史比较边界与复现

[原 2026-09-30 基线](ZorahFullPerformanceBaseline20260930.md)的旧数值保持历史记录，git HEAD 为 `f91330...`，GPU 子范围为 `Path trace shading`。本记录为 `d21e01...` 加永久修复，当前 `supplementaryPathTracing` 默认 false、实时 `Deferred shading`。代码和默认着色路径已不同，因此不计算新旧精确加速比，也不把差值归因于 meshlet 烘焙或某一修复。

标准配置：[ZorahFullBaseline.Live.json](../Tools/Perf/ZorahFullBaseline.Live.json)。使用新目录复现：

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build/zorah-full-production-new -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -RouteConfig Tools/Perf/ZorahFullBaseline.Live.json -NoVSync -TimeoutSeconds 900
```

三轮的分辨率、相机绝对关键帧、所有有效配置、合并图属性、diagnostic/debug/validation 标签相同；严格 diff 为零。原始三轮独立保存，不与其它实验池化。没有本次 1440p live 路线的新图像对照或逐像素视觉/时间稳定性证明。

证据：当前普通三轮原始 `final-production-roam/run{1,2,3}`、`ProductionRoamBaseline.json/md`、`AssetIdentityAudit.json`、独立 `FinalAcceptance.json/md`、RHI 最终 validation 及 SDK capture/replay 在 `build/nsight-fix-implementation-20260930-04/`。各实验组分别保留，SDK/validation 不作为本基线帧数据。

