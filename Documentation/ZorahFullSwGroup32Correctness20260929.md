# ZorahFull：wave32 / 32 线程正确性验收（2026-09-29）

32 线程通过本轮边界簇、非零 late、漫游停点逐字节对照，以及正常配置下连续 180 秒运行检查。生产默认仍是 WorkControl 128；本轮没有改 shader 或自动提升候选。

## 验收条件与结果

RTX 5070 Ti，输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px，SW/HW 分流 8 px。使用 `build-scheduling-release/Source/MetallicGPUDrivenSample.exe`。资产沿用已有 Full meshstream；15 秒 ready 后预热，shader cache 命中。所有 GPU 测试串行。

| 检查 | 实际覆盖 | 结果 |
|---|---|---|
| GPU 边界簇 | 313 个输入 × 4 个实际生产入口，共 1,252 次 dispatch | 打包深度/可见性一致；保护区、尾部 ID 和预期覆盖检查通过 |
| 固定视角、非零 late | early 1,348,353 / late 159,696 SW 簇 | 生产 128 与实验 32/64/128 的输出、列表、间接参数一致 |
| 漫游停点对照 | 3×60 秒，8,308 移动帧，3 个不同 cut | 每个停点内冻结 cut/驻留，所有 before/after 图像逐字节一致 |
| 连续正常漫游 | 180.002 秒，8,088 帧，时间抖动开启，流送保持活动 | 所有帧绑定 32 线程及同一 SPIR-V；无加载失败、请求溢出或 BLAS overflow |

边界 fixture 直接编译 `streamClusterRasterWorkControlMain` 和三个 group 入口，未复制 raster 内核。顶点数覆盖 3/31/32/33/63/64/65/127/128，三角形数覆盖 0/1/31/32/33/63/64/65/127/128；单独隔离每个三角形 ID 0–127，使其他三角形无法遮住该 ID 的遗漏。另覆盖 float3/float4、正向/反向 Z、反射变换、非法顶点/三角形数量、未驻留页、未选中簇、列表外工作组及尾部保护区。所有合法非空用例必须实际写出像素，不能以“两边都是空图”通过。

Full 精确对照关闭 jitter、串行 HW/SW；每个目标帧前从偏移 `[0,0,3]` 的相机预热 4 帧，制造非零 late。三个漫游段之间恢复流送，段末冻结并各测四种入口；**不同停点不要求同 cut，仅同一停点内比较**。

| 停点 | active groups | early SW 簇 | late SW 簇 | 深度/可见性差异 |
|---|---:|---:|---:|---:|
| 前进并转向 | 185,814 | 1,229,626 | 281,285 | 0 字节 |
| 反向转头 | 152,215 | 703,679 | 134,745 | 0 字节 |
| 回程 | 208,958 | 1,363,858 | 154,964 | 0 字节 |

分段漫游驻留页范围 61,082–62,480，上传 16,862 页、回收 16,512 页。逐帧身份确认没有冻结流送，也没有退回 128。三个停点的深度预览已人工检查；逐字节验证使用原始 `.bin`，不使用预览图判断误差。

## 连续漫游与尚存问题

正常连续路线使用 Full 默认起点，向前 6 米、左右转向 65 度并返回。时间抖动开启，按当前 Full 管线使用串行 HW/SW（`asyncSoftwareRaster=false`）。上传 31,359 页、回收 30,377 页，驻留页 61,351–62,887。加载失败、请求溢出、BLAS overflow 均为零。

该单次正常运行的编辑器 start-to-start 帧时间：均值 22.26 ms，P50 19.42 ms，P95 33.16 ms，P99 35.80 ms，最大 57.45 ms；357/8,088 帧超过 33.33 ms，最长连续超预算 2 帧。**尚未达到“每帧至少 30 fps”；本轮也不是进程配对的性能 A/B，不能据此声称新的加速比例。**

页面准入/分配失败计数仍然存在：正常运行逐帧计数合计 8,088（每帧 1）；该计数包括驻留预算耗尽时的准入失败，不能等同 DeviceLost/加载失败，也不能忽略。相应代码是 `MeshletStreamResidencyManager::allocatePageStorage` 附近的预算/存储失败分支。精确输出对照证明的是相同实际 cut 下的 raster 等价，不证明目标 LOD 在预算下完全收敛。这一问题应留给后续流送预算调查，不应归因或归功于线程组大小。

另保留一轮开启 debug control/validation 的连续 180 秒数据：481 帧，471 次逐帧准入/分配失败，CPU Stream traversal 平均 279.09 ms，GPU RenderGraph 平均 17.82 ms，整卡显存接近 15 GiB。没有 validation error、VUID 或 DeviceLost。该诊断运行与正常运行的 CPU 开销差距很大；未做单因素归因，不能用它评价 32 线程的运行性能。

32 线程实际 SPIR-V FNV1a64 为 `1407237746718151905`，与先前线程组性能实验及本轮所有正常/诊断样本一致。异步 SW 队列、所有可能相机下的时间重建画质不在本轮逐像素验收范围；精确图像验收是关闭抖动后的深度/可见性。

## 可重复执行

```powershell
# 在已有 MSVC 配置中构建，再运行 GPU 边界测试。
cmake --build build-scheduling-release --target MetallicGPUDrivenSample MetallicRHITests
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-validation --rhi-bindless --filter stream_group_raster_boundaries

# 三段活动流送 + 同停点四入口精确对照。使用新的输出目录。
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-scheduling-release/sw-group-checkpoints-new -SwGroupComparison -Executable E:/metallic/build-scheduling-release/Source/MetallicGPUDrivenSample.exe -RouteConfig Tools/Perf/SwGroupCorrectness.Checkpoints.json -Runs 1 -Rounds 3 -SampleFrames 8 -SettleFrames 8 -WarmupSeconds 15 -Validation -NoVSync -TimeoutSeconds 900

# 正常、不中断、带时间抖动的连续漫游。
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-scheduling-release/sw-group-live-new -SoftwareGroupSize 32 -Executable E:/metallic/build-scheduling-release/Source/MetallicGPUDrivenSample.exe -RouteConfig Tools/Perf/SwGroupCorrectness.Live.json -Runs 1 -DurationSeconds 180 -WarmupSeconds 15 -NoVSync -TimeoutSeconds 700

python -B -m unittest discover -s tests/perf -p TestRasterComparisonScopes.py
python -B -m unittest discover -s tests/perf -p TestSwGroupCorrectness.py
```

构建通过，边界 GPU 测试通过，8 个分析器测试通过。反例覆盖空参考图、空 late、停点内 cut 变化、误用其他停点参考图；旧嵌套 scope 计时测试也保持通过。默认参数仍保留原运行方式，实验 group 大小必须显式 opt-in。

## 本地证据

- [固定非零 late](../build-scheduling-release/sw-group-late-acceptance-01/run1/Summary.json)
- [三个漫游停点](../build-scheduling-release/sw-group-roam-acceptance-01/run1/Summary.json)
- [停点深度预览](../build-scheduling-release/sw-group-roam-acceptance-01/run1/roam-depth-previews.png)
- [正常连续 180 秒](../build-scheduling-release/sw-group32-live-normal-180s-01/run1/Summary.json)
- [诊断连续 180 秒](../build-scheduling-release/sw-group32-live-180s-01/run1/Summary.json)
- [边界 GPU 测试日志](../build/SwGroupBoundary.log)
- [证据哈希清单](../build-scheduling-release/sw-group32-correctness-evidence-20260929/EvidenceHashes.json)

各运行目录保留 Config/Manifest/Capture、逐帧数据、GPU/进程监控及原始日志；精确对照额外保留原始深度/可见性数据。生成证据不纳入源码控制。
