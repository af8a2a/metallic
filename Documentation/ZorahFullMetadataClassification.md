# ZorahFull：保守元数据快速分类

## 实现

复用已有 meshstream cluster 包围球，无需重新 cook，不增加页面格式、GPU buffer 或投影缓存。`streamClusterCullMain` 在存活 cluster 上先做元数据判断，直接写原候选的分类标签；稳定分桶和 visibility record ID 沿用现有路径。只有未确定的 cluster 写入精确分类工作列表。

- MASK/BLEND 或 tessellation 对象直接进入 HW，与旧分类规则一致。
- 用实例变换的保守尺度（含 shear）、实际 render camera、透视区间投影或正交投影、jitter 计算整个包围球的屏幕上界。
- 仅当球完整处于近/远裁面内、整个投影范围都满足 SW 屏幕边界和 `maxPixels` 限制时，直接进入 SW。外扩世界空间误差余量和每边 1/16 px 屏幕余量；非法半径/非有限值回退。
- 大球、裁面相交、屏幕边缘或阈值不确定情况仍执行原逐顶点/三角形分类。大球不能证明其中存在大三角形，因此没有“大球强制 HW”的近似。

流式路径默认启用；`VBuffer.metadataFastClassification=false` 保留精确路径，用于同状态 A/B。非流式路径不改动。GPU bindings 复用原 padding word，结构大小保持 96 B。

剔除后 header[0/1/2] 分别记录精确回退 / 快速 SW / 强制 HW 数量，随后由稳定分桶覆盖；诊断帧在 AfterStreamEarlyClusterCull / AfterStreamLateClusterCull 读取这三个计数。没有在计时帧加入同步读回。

## 验证方法

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-metadata-classify-new `
  -Runs 1 -MetadataComparison -Rounds 3 -SampleFrames 64 -SettleFrames 8 `
  -WarmupSeconds 10 -Width 1797 -Height 660
```

三个模式为真正全 HW、8 px 精确分类、8 px 元数据快速分类；三轮换序，在同一个进程内冻结 camera、cut、几何/CLAS/纹理驻留。每模式每轮 64 帧，共 576 帧。DLSS Quality 内部 1198×440，jitter 关闭、LOD 1.5 px，保留完整材质、阴影和两阶段 HZB。分析器要求快速开关两侧的分桶、depth 和 visibility 完全一致，同模式跨轮也必须逐位一致。具体冻结机制沿用 [固定状态对照](E:/metallic/Documentation/ZorahFullRasterComparison.md)。

GPU 等价回归与独立精确参考比较稳定分桶、候选标签、visible records、HZB retry 和实际 indirect 调度数；14 组 early/late 夹具包括透视/正交、标准/reversed Z、1/2/4/8/32 px、不同 render/cull camera、jitter、反射/nonuniform/shear、近远裁面、屏幕外、空候选、超过 65535 的工作、非法索引/半径、MASK 和 tessellation。夹具显式要求快速 SW/HW 和回退都被执行。

Full 短程 validation 已通过，无 VUID / DeviceLost，快速开关图像和分桶一致。最终非法半径防御分支另经 GPU validation 单测通过。

这是一处分类优化，不改变 SW 光栅算法，也没有修复旧 HW/SW 之间的少量深度/ID 差异。以下实测为 RHI timestamp 的单视角固定状态对照，不是新的 Nsight Trace，也不等于 30 fps 漫游验收。

## 正式实测（2026-09-20）

Release，shader debug/validation 关闭，三轮每模式 192 帧。实测收益均在本次同一冻结状态内计算，未拿上一次不同 cut 的绝对时间相减。

| 模式 | 分类 ms | Cluster cull ms | SW ms | HW ms | 光栅总均值 ms | 光栅 p95 ms | GPU graph 均值 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 真正全 HW | 0.000 | 1.341 | 0.000 | 18.971 | 21.031 | 21.801 | 23.937 |
| 8 px 精确 | 5.794 | 1.428 | 15.604 | 0.704 | 24.322 | 25.051 | 27.252 |
| 8 px 元数据快速 | 0.577 | 1.443 | 15.652 | 0.729 | 19.246 | 19.949 | 22.180 |

分类下降 **90.0%**；光栅总耗时减少 **5.077 ms（20.9%）**，已包含新增元数据判断和工作表维护成本。三轮精确路径分别为 24.351 / 24.343 / 24.273 ms，快速路径为 19.320 / 19.246 / 19.171 ms，换序后收益稳定。当前快速混合路径比同轮全 HW 快约 8.5%。

### 工作量与一致性

- Early 存活 1,074,681 个 cluster：快速 SW **951,115**，快速 HW **8,733**，精确回退 **114,833**。跳过 **89.3%** 的精确分类 workgroup；late 当前固定视角没有最终存活 cluster。
- 最终分桶两侧完全相同：HW 37,019 / SW 1,037,662；候选 overflow 为 0。每个模式的计数跨三轮不变。
- 18 个前后快照的 cut 哈希为 `7230532556492933287`，页面映射哈希为 `6928581180108922903`；active groups **221,402**。
- 每计时帧驻留保持相同：59,914 页，geometry 3,758,094,592 B、CLAS 1,775,049,344 B、texture 176,566,784 B；纹理累计升级/降级 392/3 不变。
- 精确与快速的 **VBuffer depth、visibility 均逐位一致**；各自前后和跨轮也一致。相对于全 HW 的旧光栅差异没有被此优化放大或掩盖。

源文件：[GPUDrivenStreamAsset.slang](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang)、[VisibilityBufferPass.cpp](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)、[GPU 等价测试](E:/metallic/tests/rhi/StreamClusterClassificationTests.cpp)。

证据：[正式 Manifest](E:/metallic/build-release/full-metadata-classify-formal/Manifest.json)、[Capture](E:/metallic/build-release/full-metadata-classify-formal/run1/Capture.json)、[Summary](E:/metallic/build-release/full-metadata-classify-formal/run1/Summary.json)、[可携带结果](E:/metallic/Documentation/ZorahFullMetadataClassificationResult.json)、[短程 validation](E:/metallic/build-release/full-metadata-classify-validation/run1/Capture.json)、[最终 GPU 回归日志](E:/metallic/build-release/metadata-classify-test4.log)。正式运行前后可执行文件和 shader 摘要校验一致，无 DeviceLost / validation 报错。

## 下一步

本次保留 8 px 分流默认值并启用快速分类。剩余精确分类约 0.58 ms，继续增加复杂元数据或改变 cook 的收益空间已经明显缩小。当前 SW 约 15.65 ms 成为下一优先项：先做共享屏幕顶点 setup，再做边方程和深度平面的增量计算，并沿用此开关对照检查覆盖、深度及可见性。

未改变持续漫游的流送、RTAS 更新和 CPU 成本；冻结 GPU graph 22.18 ms 不能当作完整漫游帧时间。本次只验证当前 Full 起点视角的性能收益，后续仍需在近景/远景/植被与正式路线中复测。
