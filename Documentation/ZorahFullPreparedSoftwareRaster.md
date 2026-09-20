# ZorahFull：SW 顶点预计算与增量扫描

## 结果与默认策略

实现、GPU 正确性回归与两轮 Full 固定状态对照已完成，**尚未测得稳定的性能收益**。保留独立的新 SW pipeline 供后续实验，默认继续使用 legacy SW；浮点深度增量默认关闭。保守元数据快速分类仍启用，分流阈值保持 8 px。

第二轮正式采样（每模式 192 帧、三轮换序）的 GPU 平均耗时：

| 路径 | SW ms | 分类 ms | 光栅合计 ms | Graph ms |
|---|---:|---:|---:|---:|
| 全 HW | 0 | 0 | 18.660 | 21.012 |
| legacy SW | 13.528 | 0.583 | 16.840 | 19.180 |
| 预计算顶点＋整数边增量＋cluster 相机参数 | 13.606 | 0.579 | 16.926 | 19.254 |
| 以上＋浮点深度增量实验 | 13.384 | 0.577 | 16.695 | 19.038 |

新精确路径 SW +0.58%，光栅合计 +0.51%。逐轮新旧 SW 差值为 +0.339、-0.076、-0.029 ms，方向不一致；不认定提速或稳定退化。深度增量 SW -1.06%，但有图像 ID 差异，也不作为默认优化上线。

第一轮未缓存 cluster 相机参数时，legacy/prepared 的 SW 为 14.389/14.345 ms，仅 -0.31%；深度增量为 14.415 ms。两轮冻结的 cut 不同，**只能各自内部比较，不能用两轮绝对耗时之差宣称收益**。

## 实现范围

[GPUDrivenStreamAsset.slang](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang) 增加三套独立 compute shader 入口，由 CPU 选择 pipeline。各入口向共用函数传入常量模式，供编译器专门化；未增加 GPU 运行时模式开关。旧 SW 留作参考。非流式与三角形队列路径沿用原实现。

新路径先按唯一顶点完成透视除法、viewport 变换、subpixel snap 和 z/w，结果存入每工作组共享的 128×uint4 缓冲，三角形通过局部索引复用。原 clip 与新 prepared 数据复用同一块 2048 B raw-word 顶点 scratch，避免把负整数坐标当作浮点 NaN；未增加全局投影缓存、页面格式或 cook 工作。另由 lane 0 按 cluster 计算相机基向量、投影与 jitter 参数，写入一份小型共享结构；保持逐顶点 world/view/projection 运算顺序，不合并矩阵。

[HybridRasterTriangle.slang](../Shaders/Modules/GPUDriven/HybridRasterTriangle.slang) 按三角形在 bbox 首个像素建立带 top-left bias 的整数边方程，随后按列加 stepX、按行加 stepY；空 bbox 立即返回。精确路径保留原深度表达式，覆盖、绕序、double-sided、subpixel 与 64-bit depth/ID atomic 规则不变。

深度平面实验预计算梯度，每行重新定位起始深度，行内浮点增量步进；改变了浮点运算顺序，不能套用精确路径的逐位等价结论。

运行时属性（VBuffer 节点）：

| 属性 | 默认 | 行为 |
|---|---|---|
| `softwareRasterPreparedVertices` | `false` | 开启唯一顶点/相机参数预计算和整数边增量 |
| `softwareRasterIncrementalDepth` | `false` | 仅 prepared 开启时生效，使用实验浮点深度步进 |

## 验证与证据

- GPU 夹具：共享边、退化/亚像素三角形、透视深度、标准/reversed Z、绕序/双面、4/8 位 subpixel。精确路径 packed depth/ID 逐位相同；深度平面夹具 ID/覆盖相同、深度误差不超过 2e-6。原 HW 对照与 queue overflow 检查继续通过。
- 缓存投影 GPU 夹具：透视/正交、不同 render/cull camera、jitter、反射/非均匀变换等，cached/reference clip 逐位一致；流式分类等价检查通过。
- Full：每个 case 前后及跨轮同模式图像稳定；camera、cut、页面映射和统计驻留相同。prepared 与 legacy 的原始 depth、visibility、early/late 分桶逐位相同。
- 最新 Full 深度平面实验：527120 像素中覆盖差异 0，ID 差异 30，最大深度差 4 ULP（1.12e-8）；这说明微小深度舍入仍可能改变等深竞争结果。
- 最初短测后半程受到额外 GPU 负载影响，耗时已剔除。用户暂停其他 GPU 任务后完成两轮正式采样；第二轮退出后整卡约 2–5% 占用。

条件：RTX 5070 Ti，编辑器输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px、完整材质/阴影、关闭 jitter，串行 HW/SW 便于归因。每轮包含全 HW、legacy、prepared、depth-plane，三轮换序，共 768 个计时帧。诊断读回与恢复帧排除在计时之外。

这是 RHI timestamp 固定状态内核对照，**不是 Nsight Trace，也不是持续漫游 30 fps 验收**。

- [结构化结果（两轮）](ZorahFullPreparedSoftwareRasterResult.json)
- [最新采样清单和代码摘要](../build-release/full-sw-cached-projection-formal/Manifest.json)
- [最新 Capture](../build-release/full-sw-cached-projection-formal/run1/Capture.json)
- [最新完整统计与图像差异](../build-release/full-sw-cached-projection-formal/run1/Summary.json)
- [首轮完整统计](../build-release/full-sw-prepared-formal/run1/Summary.json)

复现：

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-sw-prepared-new `
  -Runs 1 -SwComparison -Rounds 3 -SampleFrames 64 -SettleFrames 8 `
  -WarmupSeconds 10 -Width 1797 -Height 660
```

`-SwComparison` 显式选择三种 SW pipeline，不依赖生产默认值。构建最后仅修改默认选择与其他对照模式的默认选择；被测 shader 与正式采样一致。

## 后续方向

现有结果只排除了“减少这些重复算术就能显著提速”的假设，尚不能确定瓶颈。下一步应针对同一固定状态获取 shader profiler 的寄存器/occupancy、停顿原因、每 warp 有效 lane、64-bit atomic 与内存流量证据，再决定是否推进按三角形像素工作量重排或 tile 内减少重复原子写入。避免继续仅凭运算数调整深度表达式；浮点舍入已被证明会影响可见性 ID。
