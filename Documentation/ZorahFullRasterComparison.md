# ZorahFull：同状态全 HW 与 1/2/4/8 px 分流对照

## 复现

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-raster-compare-new `
  -Runs 1 -RasterComparison -Rounds 3 -SampleFrames 64 -SettleFrames 8 `
  -WarmupSeconds 10 -Width 1797 -Height 660
```

`-Validation` 仅用于短程正确性检查；正式计时关闭 validation。`Rounds` 在同一个进程、同一冻结状态内改变模式顺序；`Runs` 启动独立进程，可能得到不同的 cut，汇总器不会跨进程合并。

## 测量约束

Full 内置预设，RTX 5070 Ti；输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 render px，完整材质与阴影。固定起点相机，显式关闭共享 RenderView jitter，HW/SW 串行；保留 HZB 两阶段剔除。根页就绪并暖机后，等待已提交 GPU 工作，再冻结几何页面发布/淘汰、active cut、CLAS/BLAS/TLAS 更新和纹理调度/发布。CPU 后台任务可能完成，但不发布到被冻结资源。

全 HW 仍执行候选展开、元数据剔除、稳定分桶及 mesh shader 绘制；跳过逐三角形软硬分类、SW 队列/像素清空、SW dispatch 和 SW merge。阈值模式沿用原分类策略；masked/tessellation 约束保持原语义。该入口仅供基准使用，未更改正常场景的默认分流策略。

每组计时前后在独立诊断帧读回 active groups、页面物理映射、分桶计数、VBuffer depth/visibility。FNV1a64 哈希比较 live cut 内容和页面映射，排除页面表中允许变化的 lastRequestFrame。逐计时帧检查几何/CLAS/纹理驻留量、相机、graph generation 与输出尺寸。分析器要求同一模式前后和跨轮输出哈希完全相同。跨模式的覆盖、ID 和深度差异单独报告，不将其误标为位精确。

诊断读回、GPU drain、模式切换及恢复帧均不计入测量。每个模式每轮 64 帧，3 轮共 192 帧；顺序为 HW→1→2→4→8、8→4→2→1→HW、2→HW→8→1→4。rasterTotal 只加 early/late 父 scope，不重复叠加子 scope。

**这是 RHI GPU timestamp 的固定状态对照，不是新的 Nsight Trace，也不是实时漫游 FPS 验收。** 冻结后的整图时间排除了遍历/流送/RTAS 更新，不能与旧 P0 的持续漫游绝对耗时直接相减。当前只验证一个起点视角；更换路线或视角应重新冻结采样。

初次短测发现共享 RenderView jitter 未关闭，已丢弃 `full-raster-compare-validation` 的性能/图像结论。修正后 `full-raster-compare-validation-fixed` 通过，五种模式各自前后 depth/visibility 完全一致。

## 2026-09-20 正式结果

正式目录：[full-raster-compare-formal](E:/metallic/build-release/full-raster-compare-formal/Manifest.json)。Release，validation 关闭，shader debug 关闭。15 组全部完成，计时及图像检查通过。

| 模式 | 光栅总均值 ms | p50 | p95 | 分类均值 | SW 均值 | HW 均值 | GPU graph 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全 HW | 19.640 | 19.664 | 20.357 | 0.000 | 0.000 | 17.814 | 22.524 |
| 1 px | 22.232 | 22.180 | 23.068 | 5.744 | 11.223 | 3.318 | 25.198 |
| 2 px | 21.794 | 21.757 | 22.548 | 5.669 | 12.037 | 2.164 | 24.727 |
| 4 px | 21.929 | 21.760 | 23.366 | 5.760 | 12.807 | 1.403 | 24.859 |
| 8 px | 21.416 | 21.396 | 22.123 | 5.651 | 13.078 | 0.746 | 24.258 |

每模式 192 帧。全 HW 三轮均值 19.595 / 19.739 / 19.585 ms，8 px 为 21.438 / 21.365 / 21.445 ms，全 HW 每轮均胜出；4 px 存在 21.540–22.243 ms 波动，因此不夸大 2/4 px 的细小排名差异。全 HW 相对 8 px 减少 **1.776 ms（8.3%）**，不是把原有 0.746 ms HW 分支直接当作全硬件耗时：全量转 HW 后该分支增加到 **17.814 ms**。

### 固定状态证据

- 221,815 active groups，cut 哈希 `16023908386971210489`；页面映射哈希 `5258174110348134690`，30 个前后快照全部相同。
- 每计时帧均为 60,277 驻留页；geometry 3,758,094,592 B，CLAS 1,786,832,640 B，texture 176,894,464 B。纹理累计升级/降级计数为 394/3，期间不变。
- 同一模式前后、跨三轮的 visibility/depth 均逐位一致；不存在相机抖动混入。
- Early 展开候选均为 4,546,026；全 HW early 1,076,673 个 cluster，SW 为 0。8 px 为 HW 37,157 / SW 1,039,513；1 px 为 HW 189,441 / SW 887,233。减少阈值只把约 15.2 万个 cluster 从 SW 移到 HW，绝大多数仍落在 SW。
- HZB 历史没有强制冻结；跨模式深度微差使 early 总存活数量相差最多 4 个，late 重试候选为约 313 万，各模式最终 late HW/SW 均为 0。每个模式的数量跨轮稳定。这是保留真实两阶段剔除的结果，不能声称跨模式 survivor 集合逐位相同。
- 测量期间 GPU 时钟约 2655–2917 MHz，整卡显存约 13.0–14.5 GiB；进程监控未发现另一大型渲染器，但仍有 DWM、浏览器视频解码、壁纸等轻负载。没有将“非完全空闲桌面”包装为实验室独占 GPU。三轮换序用于检查漂移，原始监控随报告保留。

### 图像差异

527,120 个内部渲染像素，HW 有效覆盖 404,352 个。所有阈值相对 HW 的覆盖增减均为 **0**。

| 模式 | ID 不同像素 | 占全图比例 | 最大 depth 绝对差 | 相同 ID 像素的最大 depth 差 |
|---|---:|---:|---:|---:|
| 1 px | 26 | 0.00493% | 2.0818785e-05 | 2.3585744e-07 |
| 2 px | 38 | 0.00721% | 1.1336757e-05 | 2.3585744e-07 |
| 4 px | 53 | 0.01005% | 0.00076033175 | 4.9127266e-07 |
| 8 px | 67 | 0.01271% | 0.0012487839 | 1.1529773e-06 |

这些是 reversed-Z buffer 的数值差，不是世界空间距离。8 px 有极少数不同 ID 像素出现较大深度差，不能全部归因于浮点舍入，也不能仅凭零覆盖差声明画质完全等价；后续应检查 subpixel snap、边界规则、遮挡层选择及深度插值。原始逐像素 depth/visibility 二进制和最大误差像素索引均已保存，未用 DLSS 输出掩盖原始 VBuffer 差异。

## 下一步判断

1. **先优化分类的几何访问。** 当前 1/2/4/8 px 分类均约 5.7 ms，对阈值基本不敏感。它足以抵消混合光栅的收益。优先增加保守元数据快速分流，让容易判断的 cluster 不再读取/变换全部顶点并遍历三角形；模糊、近裁面、MASK 等情况保留精确路径。5.7 ms 只是完全消除此阶段的理论上限，实际快速路径也有成本。
2. **继续优化 SW setup 和扫描，不先改默认阈值。** 8 px 在本视角的混合组最快；1 px 降低 SW 约 1.855 ms，却增加 HW 约 2.572 ms，分类成本仍在。SW 13.078 ms 仍是混合组最大单项，共享屏幕顶点 setup、增量边方程/深度平面具有更直接的目标。变更时用本入口持续核验 HW/SW 覆盖和深度差异。
3. **保留真正全 HW 作为性能基线。** 本次没有将默认模式切成全 HW；仅一个固定起点视角，不能证明漫游全程最优。优化分类后，以同一对照重新评估；至少补近景、远景、密集植被视角后，再决定全 HW/混合的默认或动态选择。

尚未完成 30 fps 漫游验收，也没有用冻结整图 22.5–25.2 ms 推算实际交互帧率。

## 验证与实现

- Release `MetallicGPUDrivenSample`、`MetallicRhiTests` 构建通过。
- `stream_cluster_cull_classify_equivalence`：validation + bindless，通过 11 组 early/late 夹具，覆盖全 HW、HZB、空候选及超过 65535 的调度；检查稳定分桶和可见性 ID。
- `ktx2_texture_streaming`：validation + bindless，通过；增加 120 帧冻结后 mip/驻留/升级降级计数不变检查，并继续原冷回收测试。
- `stream_blas_cut_cache` validation 回归通过。
- Full 短程 validation 和正式 960 帧均完成，无 Validation Error / VUID / DeviceLost；正式运行前后可执行文件与 shader hash 一致。
- 分析器现在拒绝同模式前后/跨轮图像变化；`git diff --check` 与 Python 语法检查通过。

入口：[EditorRasterComparison.cpp](E:/metallic/Source/Editor/EditorRasterComparison.cpp)、[RunZorahFullRoam.ps1](E:/metallic/Tools/RunZorahFullRoam.ps1)、[分析器](E:/metallic/Tools/AnalyzeZorahFullRasterComparison.py)。

可携带结果：[ZorahFullRasterComparisonResult.json](E:/metallic/Documentation/ZorahFullRasterComparisonResult.json)。本机原始证据：[Capture](E:/metallic/build-release/full-raster-compare-formal/run1/Capture.json)、[Summary](E:/metallic/build-release/full-raster-compare-formal/run1/Summary.json)、[GPU 监控](E:/metallic/build-release/full-raster-compare-formal/run1/Gpu.csv)、[进程监控](E:/metallic/build-release/full-raster-compare-formal/run1/GpuProcesses.csv)。
