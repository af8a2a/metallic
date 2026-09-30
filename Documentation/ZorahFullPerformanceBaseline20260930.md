# ZorahFull 2560×1440 性能基线（2026-09-30）

这份记录保留 `f91330e` 版本的历史测量。此后 Deferred 默认功能及 Nsight
反馈／静态元数据资源路径已调整；最终版本的生产漫游复测见
[修复后的 1440p 基线](ZorahFullPerformanceBaselineAfterNsightFix20260930.md)。
两份记录保留各自条件与数值，不以版本间差异推断某一项改动的精确收益。

在当前 meshlet 全量重烘焙后，重新测量生产默认路径。以后这套基线固定 **2560×1440 输出**，不以旧的 1797×660 结果计算加速比。RTX 5070 Ti / 驱动 616.92，当前源码 `f91330ef567f310bdb9682f7c842c24307bcb076`；既有 MSVC/Ninja Release 构建检查通过，无需增量编译。

## 固定条件

- 完整 `gpu-driven-zorah-full`，正式缓存 cook revision 4，54,921,584,816 字节（51.1497 GiB）。缓存 SHA-256 为 `ddffaf7fee0364cb34215fba31d24a83da55206e95e776cbbb3a5b3a7f52ab3f`；在 GPU 测量结束后计算，未对采样造成磁盘读取竞争。
- 输出 2560×1440；DLSS Quality 实际内部渲染 **1707×960**。时间抖动开启，默认 `softwareGroupSize=0`；每个采样帧均绑定 `streamClusterRasterGroup32Main`，group/subgroup 32，SPIR-V FNV1a64 `1407237746718151905`。
- 三个独立进程串行运行，ready 后预热 10 秒，然后采样 30 秒。相同的 6 m 往返与 ±65° 转向、绝对相机关键帧和图属性。路线按墙钟推进，各轮经过的帧数不同。
- 隐藏编辑器窗口，VSync 关闭，Reflex 生产默认。流送保持活动，不冻结 cut/geometry/CLAS/纹理；无 validation、Nsight/NvPerf 注入、工作量重放或图像 readback。未清空 OS 文件缓存、shader 或 pipeline 缓存，没有额外执行 shader warmup target。

## 帧时间与加载

`frameMs` 是编辑器循环 start-to-start，包含等待、事件、录制、提交、present 和采集记账。它不是纯 GPU 时间。按独立进程报告，均值的中位数为 **25.313 ms**（其倒数约 **39.51 FPS**），三轮均值范围 24.447–26.916 ms；相对极差为 9.754%。此数值描述重复运行波动，不是 frozen A/A 资格判定。

| 运行 | 帧数 | 均值 ms | P50 ms | P95 ms | P99 ms | 最大 ms | >33.33 ms | 最长连续超预算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,186 | 25.313 | 25.205 | 31.754 | 36.022 | 46.489 | 32（2.70%） | 1 |
| 2 | 1,228 | 24.447 | 24.328 | 31.489 | 34.299 | 38.316 | 22（1.79%） | 1 |
| 3 | 1,115 | 26.916 | 26.854 | 33.990 | 38.113 | 69.762 | 73（6.55%） | 2 |

第三轮 P95 超过 33.33 ms；三轮都未达到严格持续 30 FPS。没有将第三轮或长帧排除，也没有混合不同进程的帧来推断显著性。

`loadingSeconds` 为进入 benchmark 的加载循环到首次 ready 且 preview 有效，分别为 **6.002 / 5.147 / 5.422 秒**。它排除此前程序/设备/场景初始化，也排除 ready 后的 10 秒预热，不能称为启动到首帧的总耗时。

## 主要 GPU 范围

以下是各运行的范围均值，单位 ms。RenderGraph envelope 不包含编辑器合成/present 或独立纹理上传提交；父子范围为 inclusive，不能相加。

| 范围 | 运行 1 | 运行 2 | 运行 3 |
|---|---:|---:|---:|
| RenderGraph envelope | 24.659 | 23.810 | 24.868 |
| VBuffer | 8.836 | 8.546 | 8.827 |
| Stream traversal | 3.292 | 3.224 | 3.306 |
| Deferred | 13.847 | 13.362 | 14.053 |
| Path trace shading（Deferred 子范围） | 13.726 | 13.239 | 13.929 |
| SW raster early | 1.756 | 1.695 | 1.755 |
| SW raster late | 0.121 | 0.115 | 0.123 |
| HW raster early | 1.074 | 1.027 | 1.048 |
| HW raster late | 0.071 | 0.064 | 0.069 |
| BLAS build | 0.374 | 0.374 | 0.387 |
| TLAS build | 0.300 | 0.295 | 0.311 |
| DLSS-SR pass | 1.029 | 1.012 | 1.028 |
| AutoExposure | 0.517 | 0.511 | 0.516 |

当前最大单个 GPU pass 为 Deferred。CPU pacing 前的流送维护均值为 1.566 / 1.502 / 1.593 ms。全范围分解与每个阶段的 P50/P95/P99 已保存在原始 `Summary.json` 和批次 `BaselineSummary.json` 中；不能把这些 pass 时间写成整帧加速。

## 流送、显存与验证边界

- 三轮 Capture 均完整，分辨率、相机、配置和图跨进程一致。每帧生产 shader 身份已核对，`missingGpuFrames=0`，BLAS/TLAS 必需时间范围覆盖全部采样帧。原始日志没有 VUID、DeviceLost 或加载错误；runner 以 0 退出，没有 teardown 回收记录。
- IO load failure、request overflow、BLAS overflow、预算/存储拒绝和 invalid group 为 0。每轮页面 allocation failure 合计 **10**；`blasMissingClasInstances` 峰值为 **515 / 501 / 587**，表示 CLAS 未就绪时的回退。该计数是重复采样的 header 快照，不累计成唯一失败数量。
- NVML 在整个进程生命周期内采样的整卡显存峰值为 **12,515 / 12,610 / 12,646 MiB**，最高 GPU 温度 65°C。这是整卡而非程序独占值，且不只覆盖 30 秒采样区间。
- 保留后台 GPU Engine 原始记录。其条目为单个 engine 的条件采样，工具过滤低于 0.1% 的条目；三轮均未记录到 benchmark 自身的 process 条目，因此进程归因覆盖不完整。不能据此证明 GPU 独占，也不能把多 engine 利用率相加来解释单帧。计时波动保留为实际环境下的观察值。
- 缓存 payload/完整源覆盖及 960×540 validation 首帧检查来自当前 revision 4 cook 的既有验证。本次只测量 1440p 普通漫游，没有另外采集该分辨率的图像或执行与参考渲染器的逐像素正确性比较。缓存文件内容已独立哈希；没有重新哈希全部 glTF/图片/buffer 依赖或保存完整资产副本。
- 旧基线使用 revision 2 缓存和 1797×660 输出；其间还改动了 initial loading、几何共享、压缩和 LOD 策略。这份结果建立当前基线，不是单独烘焙策略的 A/B 收益证明。历史 frozen WorkControl case 显式使用旧 128 线程入口，与本生产 live 基线分别保留。

## 复现与证据

固定配置：[ZorahFullBaseline.Live.json](../Tools/Perf/ZorahFullBaseline.Live.json)。使用新的输出目录：

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/zorah-full-baseline-1440p-new -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -RouteConfig Tools/Perf/ZorahFullBaseline.Live.json -NoVSync -TimeoutSeconds 900
python -B Tools/AnalyzeZorahFullRoam.py build-release/zorah-full-baseline-1440p-new
```

本次实际命令沿用旧 live route config，再显式指定 `-Width 2560 -Height 1440`；最终 `Config.json` 与新固定配置加上述时间参数一致。用户更改分辨率前启动的批次保留在 `build-release/zorah-full-baseline-20260930-live-01`，标记中断，不纳入本基线。

用于以后精确比较的数值记录：[ZorahFullPerformanceBaseline20260930.json](ZorahFullPerformanceBaseline20260930.json)。离线复核通过：65 个证据文件的 SHA-256、25 个当前 EXE/DLL、缓存长度/mtime、三轮条件与逐帧默认 shader 身份；复核日志保存在 `build/ZorahFullBaseline1440pVerify20260930.json`。该次复核没有再次读取全部 51 GiB 缓存，使用测量后计算的内容 hash 与未变化的文件 metadata，也没有重放 GPU 或解码图像。

证据目录：[zorah-full-baseline-20260930-1440p-01](../build-release/zorah-full-baseline-20260930-1440p-01/)。包括三个进程的 Capture、逐帧遥测、GPU/后台活动、日志、Summary，批次 Comparison/BaselineSummary、资产身份、EXE/DLL 哈希以及相关源码/工具/cook 记录快照；大文件和采集输出保留在本地 build 目录。
