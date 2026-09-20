# ZorahFull：SW 工作量与 RenderGraph 等待归因

2026-09-20。已接入可关闭的工作量诊断、实际 shader 标识、CPU preparation scopes 与等待原因，并完成 Full 对照。此次保留现有渲染和同步策略，建立后续修复基准。

## 主要结论

此前冻结 SW 7.3 ms 与漫游 33.1 ms 的差距，主要来自 HZB 历史失效后增加的提交量。实际选择的 shader 没有退回旧实现。CPU 侧还存在编辑器外部输出使用完成点触发的逐帧整图等待。

相机更新路径调用 `HistoryResourceManager::invalidateAll(CameraMotion)`，该操作递增通用 invalidation revision；VBuffer 读取这个通用 revision 并把变化当作 camera cut。因此即使只是重复设置同一个相机，也会丢弃上一帧 HZB。`RenderView::setCamera()` 对合法的相同参数仍返回成功，重复设置也会进入上述 history invalidation 路径。普通漫游每帧更新相机，冻结对照则不做这个操作。

代码依据：

- [EditorApplication.cpp](../Source/Editor/EditorApplication.cpp)：`applyViewportCameraProperties()`。
- [VisibilityBufferPass.cpp](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)：`execute()` 中 `observedHistoryInvalidationRevision_` 与 `cameraCut`。
- [HistoryResources.cpp](../Source/Runtime/Render/HistoryResources.cpp)：通用 revision 与排除 CameraMotion 的 `reprojectionInvalidationRevision()` 已分开维护。

## 同 camera、cut、驻留的因果对照

新增 `-HistoryComparison`：冻结几何、CLAS、cut、TLAS 与纹理发布；使用完全相同的相机、输出 1797×660、DLSS Quality 内部 1198×440、1.5 px、8 px 分流。两轮换序，每种模式各 32 个计时帧；诊断快照和恢复帧不计入耗时。

两种 SW 模式仅区别于是否每帧调用相机设置函数，并传入当前相机的原值。

| 指标，early 阶段工作量 | 保持相机 | 重复写入相同相机 | 比值 |
|---|---:|---:|---:|
| 参数中的 HZB history valid | 1 | 0 | — |
| SW clusters | 1,041,657 | 4,598,812 | 4.41× |
| 提交三角形 | 42,565,256 | 188,582,571 | 4.43× |
| cluster 内唯一顶点处理量 | 103,089,190 | 451,470,079 | 4.38× |
| bbox 像素访问 | 3,203,542 | 7,245,368 | 2.26× |
| 覆盖样本 / 像素原子尝试 | 638,149 | 1,420,953 | 2.23× |
| SW GPU 均值，early+late | **9.906 ms** | **43.015 ms** | **4.34×** |
| RenderGraph GPU 均值 | 17.881 ms | 51.439 ms | — |

两个模式的最终 **visibility 和 depth 逐位一致**，两轮计数也完全一致。增加的工作没有改变输出。

- 同一 cut：`3018165815145797720`，222528 active groups。
- 同一页映射：`14115650577649230408`；计时帧的几何、CLAS、纹理驻留相等。
- 同一入口：`Features/GPUDriven/GPUDrivenStreamWorkRaster.streamClusterRasterWorkControlMain`。
- 同一 SPIR-V FNV1a64：`14699073418322314354`，mode 5，共享屏幕顶点、不分桶。
- 该 hash 标识编译后的 SPIR-V，不代表驱动最终机器码 hash。

本轮另一份开启 Vulkan validation 的冻结五路径对照中，共享顶点 SW 为 7.086 ms，与上轮 7.317 ms 接近。不同运行的时钟/环境和驻留不完全相同，不能把 7.086→9.906 解释为代码退化；应使用同一运行的换序 A/B 比较。原始 7.3/33.1 两份历史数据无法事后变成严格同状态对照，新实验验证了造成该数量级差距的具体机制。

## 无诊断漫游中的等待

关闭工作量重放和读回，10 秒预热后采样 30 秒，共 454 帧，GPU timing 缺失 0：

| 指标 | 均值 |
|---|---:|
| 整帧 | 66.132 ms，P95 75.905 ms |
| RenderGraph GPU | 42.176 ms |
| Early SW GPU | 28.540 ms |
| Refresh scene bindings CPU | 0.132 ms |
| Preflight CPU | 3.395 ms |
| Prior frame drain CPU | **33.768 ms** |
| Submission slot wait CPU | 0.000115 ms |
| Command pool reset CPU | 0.00676 ms |
| Frame setup CPU | 0.00634 ms |

454/454 帧 `drainReasonMask=4`、没有阻止 overlap 的 pass；本次由 external completion 触发 drain，非 scene revision 或 pass overlap 合约。454/454 帧的 `paramsHzbValid=0`。

`RenderGraphExecutor::transitionOutput()` 将使用 graph 输出的编辑器提交 completion 登记为外部依赖；下一次 `execute(RenderGraphSubmitDesc)` 因列表非空调用 `waitForSubmittedWork()`。这解释了外层 CPU 区间的大部分空白。`Prior frame drain` 是等待前序工作，不应再与同一流水线 GPU 耗时相加当作独立工作量。普通 submission slot wait 几乎为零。

下一步应检查外部输出访问是否能仅通过 GPU completion 依赖排序，保留资源 generation、resize 和真正需要 CPU 修改的资源保护；不能简单删除所有等待。

## 统计接口与语义

普通 `Frames.jsonl` 现在包含：

- `streaming[].softwareRaster`：解析缺省属性后的 module、entryPoint、mode、SPIR-V hash、阈值、是否强制 HW、是否请求 async、冻结状态与 HZB 状态。
- `graphPreparation`：executionId、drainReasonMask、externalCompletionCount、overlapBlockingPasses。
- CPU profiler 中与 GPU envelope 平级的 `Graph preparation / ...` scopes，避免把 GPU 等待伪装成 GPU pass 内部 CPU 工作。
- mask 位：1=pass 不支持 overlap，2=scene stamp 变化，4=external completion；可同时置位。

使用 `-WorkloadCounters` 时，普通漫游每 60 帧采样一次，`Capture.json.workloads` 按 editor frame 和 early/late 分开保存。冻结对照仅在原有 before/after 诊断快照执行计数。`WorkloadSummary.json` 验证 cluster 总数与对应 GPU bin 头一致，并汇总覆盖比例。

计数通过独立 shader 对相同 SW 列表重放 snapped 整数覆盖计算，不写 visibility/depth，也不修改正式光栅内核。每个 wave 归约后累加到 64 位统计缓冲区；使用 32 位分段归约，不要求 shaderSubgroupExtendedTypes。原子尝试数等于该光栅路径每次覆盖通过后调用 InterlockedMax 的次数，**不是获胜写入次数、原子冲突次数或原子停顿时间**。

`uniqueVertices` 是每 cluster 内唯一顶点处理量，不是全场景去重后的顶点总数。`emptyTriangles` 包含背面、退化或空 bbox，尚未拆分具体原因；`nonemptyTriangles` 表示进入 bbox 扫描的三角形，并不保证覆盖像素。当前精确覆盖支持标记针对 mode 4/5；其他入口的重放结果不应冒充该入口逐指令测量。

`hzbValid` 反映导出时状态：pass 结束后当前帧 HZB 已生成，可能为 true；判断本帧是否使用历史时应看 `paramsHzbValid`。诊断快照在光栅前采集，两者状态一致。

计数与读回会明显扰动 GPU，漫游报告显式标记 `diagnosticRun` 和诊断帧；其帧率不用于性能验收。普通默认路径不执行计数 dispatch 或读回。

## 验证与下一步

- Release 样例与 RHI 测试构建成功。
- 开启 Vulkan validation 的覆盖回归通过：逐三角形运行原始光栅，独立计数其覆盖像素，对照诊断的 covered/atomic attempts；包含单双面、两种深度方向、两种 subpixel 精度，原有精确深度/ID 回归继续通过。
- 开启 validation 的 22 组流式回归通过：实际 reset/count shader、early/late、空列表、异常页、2D dispatch；核对 bin 总量和三角形/覆盖记账。
- Full validation 五路径 40 个计时帧通过；共享顶点与分桶路径相对原 SW 的图像检查通过，未报告 VUID/device 错误。
- 同状态 history 因果对照 96 个计时帧通过，新增自动断言：camera A/B 的 shader 相同、HZB 状态符合预期、两轮 cut/驻留稳定、最终图像相同。
- 初次诊断触发的 subgroup 功能校验已修正；两次显存预算不足的运行和初次校验失败运行不作为最终验收证据。

优化优先级据此调整为：**先区分普通相机运动和真正 camera cut，恢复 HZB 历史；再处理编辑器外部 completion 导致的 CPU 串行等待。** 修复后重新建立同条件漫游基准，再决定 scanline 或进一步工作量重排。三角形/顶点处理量增长约 4.4×，而像素扫描和原子尝试只增长约 2.2×，现阶段不应继续把差距主要归因于扫描或原子写入。

## 复现

```powershell
# 正常漫游：关闭工作量计数，导出等待与实际 shader 标识
pwsh -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-waits-new -Runs 1 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660

# 工作量：单独的诊断运行
pwsh -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-counts-new -Runs 1 -DurationSeconds 15 -WarmupSeconds 10 -Width 1797 -Height 660 -WorkloadCounters
python Tools/AnalyzeZorahFullWorkload.py build-release/full-counts-new/run1

# 相同状态，仅重复写入相同相机；两轮换序
pwsh -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-history-new -Runs 1 -HistoryComparison -Rounds 2 -SampleFrames 16 -SettleFrames 8 -WarmupSeconds 10 -Width 1797 -Height 660 -WorkloadCounters
python Tools/AnalyzeZorahFullWorkload.py build-release/full-history-new/run1
```

证据：[结构化结论](ZorahFullWorkloadAndWaitsResult.json)、[同状态因果对照](../build-release/full-sw-history-causal/run1/Summary.json)、[Full validation 对照](../build-release/full-sw-workload-frozen2/run1/Summary.json)、[无计数漫游](../build-release/full-sw-workload-roam/run1/Summary.json)、[实时工作量](../build-release/full-sw-workload-counts/run1/WorkloadSummary.json)。
