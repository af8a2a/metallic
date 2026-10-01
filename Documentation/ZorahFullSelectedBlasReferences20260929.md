# BLAS 引用按实际选中数量分配（2026-09-29）

## 改动

移除初始化时按 primitive 全部 LOD cluster 总量向场景前部实例分配固定引用区间的策略。现在由 Streamer 的 GPU BLAS 输入流程按本次 cut 分配：

1. 统计每个实例的实际选中 cluster 数，标记未就绪页面的完整实例回退。
2. 每 64 个实例做局部排他前缀和；第二级扫描 block 总量，再形成全局引用偏移与构建索引。
3. 按稳定实例顺序准入完整实例，填充 CLAS 引用并间接构建。分别检查总引用、构建槽和单 BLAS cluster 上限。

初始化只查询全局引用/构建/单次构建的驱动存储上限。BLAS 字节预算不足时缩减引用容量，保留实例构建槽；仅在引用数少于构建槽时同步限制槽数。查询和实际提交使用相同上限，原有 `maxBlasBytes` 不变。primitive 全 LOD 总量仍用于保守的**单 BLAS 查询上界**，不再作为每个实例的预留开销。

block scratch 放在 BLAS header 的 cut cache 之后，Full 43,068 个实例约需 10.5 KiB。无 CPU 同步读回，没有新增 renderpass 资源加载。已有全 cut 精确缓存、提交取消失效和 CLAS 发布代际失效继续保留。

超预算时接受符合容量的稳定前缀，其余实例使用原有完整 fallback BLAS，不发布截断几何。这个版本不在前缀之后寻找较小实例填尾部空隙，也没有改成按屏幕收益排序；这使准入结果确定且可验证。逐实例 BLAS 缓存不在本次范围内。

## 统计与布局

BLAS header 从 32 B 扩展到 64 B，cut cache 起点从第 2 个 uint4 调整为第 4 个；CPU、Slang、读回、间接构建计数偏移和测试保持一致。帧参数复用原有 padding 传递单 BLAS 上限，结构总大小仍为 416 B。

新增 Full 漫游导出：`blasRequestedClusterReferences`、`blasRequestedInstances`、`blasReferenceBudgetRejected`、`blasBuildBudgetRejected`、`blasOversizedInstances`、`blasMissingClasInstances`、`blasInvalidGroups`。debug snapshot 增加构建容量和单 BLAS 上限。

需求统计来自通过 group/页面就绪检查的计数；一个实例若有部分 CLAS 未就绪，即使其他 group 已计数，也整体回退。因此请求引用数不必等于获准引用数。非法 group 会使可识别的所属实例回退，并计入非法 group 计数。`missingClasInstances` 统计输入阶段已标记 fallback 的实例，异常输入时可与非法 group 计数重叠。这些是本次输入重建计数；精确 cut 缓存命中时计数清零，不能解释为场景没有动态 BLAS。反馈源帧与当前 CPU 帧分开记录。

## 验证

沿用 MSVC Release 配置，构建成功：

```powershell
cmake --build build-release --target MetallicGPUDrivenSample -j 6
cmake --build build-scheduling-release --target MetallicRHITests -j 6
```

以下 **6 项 GPU 测试通过，无跳过**，均启用验证层：

- `RHIRendering.stream_blas_selected_allocation`：新增测试直接执行生产 shader，137 个实例跨 3 个线程组，初始实例容量全零；覆盖稀疏 mask、部分尾组、反向实例映射、精确容量、引用/构建槽/单 BLAS 超限和未就绪 CLAS。逐项核对连续区间、完整回退、实际引用内容、地址低位进位、间接参数和尾部未被写入。
- `RHIRendering.stream_blas_cut_cache`：稳定 cut 复用、LOD 改变、取消录制和 CLAS 退休后的失效。
- `RHIRendering.minizorah_clas_in_flight`。
- `RHIRendering.stream_clas_runtime_lifecycle`。
- `RHIRendering.stream_clas_eviction_reupload`。
- `RHIRendering.zorah_full_first_frame`：一次 MiniZorah→Full 切换、完整准备、后续渲染及材质分桶检查。

日志未发现 Vulkan validation error、VUID 或 DeviceLost。检查了 Full settled/base-color 输出，建筑、人物、植被和材质覆盖正常；保留既有单样本噪声。这是 960×540 原生、DLSS 关闭的测试图，不代表编辑器 DLSS 漫游的逐像素或长期稳定性证明。

原始结果：

- `build-scheduling-release/blas-selected-validation/`，日志 `build-scheduling-release/blas-selected-test.log`。
- `build-scheduling-release/blas-selected-lifecycle/`，日志 `build-scheduling-release/blas-selected-lifecycle.log`。
- `build-scheduling-release/blas-selected-full-visual/`，包括 `ZorahFullFirstFrame.json`、`ZorahFull-settled-0.png` 和 `ZorahFull-base-color-0.png`。

## Full 固定漫游：容量问题已消除，构建工作量显著增加

新进程、已有磁盘 cook/shader/纹理缓存；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、8M raster candidates。沿用 180 帧绝对相机路线，逻辑时长 30 秒、预热 3 秒。几何和 CLAS 预算未改变。

第一轮每 30 帧进行工作量读回，是诊断运行；与最近主线 CLAS 分块记录比较：

| 指标 | 历史固定区间 | 实际 cut 分配 |
|---|---:|---:|
| 完成路线帧数 | 180 | 180 |
| 动态 BLAS 构建数峰值 | 219 | 12,561 |
| BLAS 引用数峰值 | 18,781 | 5,046,530 |
| 聚合 overflow 峰值 | 12,425 | **0** |
| 引用 / 构建槽 / 单 BLAS 拒绝 | 未细分 | **全部为 0** |
| 非法 group 峰值 | 未细分 | **0** |
| 未就绪 CLAS 实例峰值 | 未细分 | 1,310 |
| 页面 IO 失败 / 请求溢出 | 0 / 0 | 0 / 0 |
| 几何 used 峰值 MiB | 3,582.268 | 3,582.268 |
| CLAS backing 峰值 MiB | 1,792 | 1,792 |
| NVML 整卡峰值 MiB | 13,368 | 13,653 |

不同指标的峰值不一定来自同一反馈帧。几何准入 allocationFailures 两轮单帧峰值均为 1，不属于 BLAS overflow。未就绪 CLAS 的实例仍使用 fallback；不能宣称整个场景始终零回退。

这轮证明了容量利用和实例覆盖改善，不证明整帧提速或显存下降。诊断 BLAS build 平均约 7.515 ms，原来因固定区间不足而跳过的大量细化实例现在实际进入构建。整卡峰值还受背景和驱动分配影响，不能只归因于这项改动。

证据：`build-release/blas-selected-full-roam/`；历史对照 `build-release/clas-demand-final-roam/`；汇总 `build-release/blas-selected-full-comparison.json`。runner manifest 记录程序/shader 摘要、配置与工作区状态。

## 关闭工作量读回的单轮计时

另跑同路线，`workloadEvery=0`、分类/软件工作量计数关闭，`diagnosticRun=false`，验证层关闭。保留正常异步 profiler/流送反馈：

| 指标 | 结果 |
|---|---:|
| 帧平均 / P50 | 34.204 / 32.162 ms |
| 帧 P95 / P99 | 51.420 / 79.568 ms |
| 超过 33.33 ms | 87 / 180 帧 |
| GPU envelope 平均 | 31.073 ms |
| BLAS build 平均 | **6.517 ms** |
| TLAS build 平均 | 0.298 ms |
| Stream early software raster 平均 | 11.399 ms |
| Deferred 平均 | 5.285 ms |
| BLAS overflow / IO 失败 / 请求溢出 | 0 / 0 / 0 |

范围时间是 inclusive，不累加父子 scope。这是一轮新实现的成本观察，没有新做同条件旧实现的普通计时 A/B，也没有三轮重复，因此不作为性能收益验收；**本轮尚未达到漫游 P95 30 fps 目标**。180 帧固定路线也不代表长时间驻留压力测试。

证据：`build-release/blas-selected-full-normal/run1/{Capture.json,Frames.jsonl,Summary.md,Gpu.csv}`。

下一步应先做 **逐实例 dirty 与 BLAS 复用**，让构建量由变化实例决定，再考虑相同物体空间 cut 的跨实例共享。当前全场景 dirty 会把这批已恢复覆盖的实例一起重新构建，已经成为明确的 GPU 成本。
