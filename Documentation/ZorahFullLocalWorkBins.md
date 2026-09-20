# ZorahFull：局部光栅工作量分桶与稳定 ID

## 结论与默认设置

已实现每 cluster 的 bbox 面积/行数分桶，并保留原始可见性 ID。Full 固定 camera、cut、驻留的 3 轮换序对照显示：**局部分桶还没有超过同等顶点复用、不分桶的路径**，因此作为可切换选项保留，默认关闭。默认开启本轮验证有效的共享屏幕顶点和精确整数边步进。

RTX 5070 Ti，输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px、完整材质/阴影、8 px 软硬分流；每模式 192 帧，共 960 帧：

| 路径 | SW 均值 ms | 光栅合计 ms | Graph ms |
|---|---:|---:|---:|
| 全 HW | 0 | 19.215 | 21.730 |
| 旧串行装载 SW | 15.004 | 18.549 | 21.269 |
| 原协作装载默认 | 8.647 | 12.120 | 14.694 |
| 协作装载＋屏幕顶点复用，不分桶 | **7.317** | **10.760** | **13.339** |
| 相同复用＋局部分桶 | 8.215 | 11.670 | 14.281 |

复用路径相对原默认 SW **降低 15.38%（1.330 ms）**。局部分桶相对相同复用路径 **增加 12.27%（0.898 ms）**；不能把它相对旧默认的改善全部归因于分桶。逐轮 SW：

| 轮次 | 原协作装载 | 复用不分桶 | 复用分桶 |
|---|---:|---:|---:|
| 1 | 8.470 | 7.061 | 8.202 |
| 2 | 8.429 | 7.246 | 8.138 |
| 3 | 9.042 | 7.643 | 8.306 |

这是 RHI timestamp 固定状态内核对照，冻结了遍历、流送发布和 TLAS，关闭 jitter，HW/SW 串行；诊断读回与恢复帧不计时。**不代表持续漫游已达到 30 fps，也没有测得新的 lane utilization 或 occupancy。**

## 实现与 ID 约束

[StreamRasterWorkBins.slang](../Shaders/Features/GPUDriven/StreamRasterWorkBins.slang) 在 128 线程组内使用四个桶：

| 桶 | bbox 条件 |
|---|---|
| 0 | 面积 ≤4 像素且行数 ≤2 |
| 1 | 其余面积 ≤16 像素且行数 ≤4 |
| 2 | 其余非空 bbox |
| 3 | 背面、退化、空 bbox 或无效尾 lane |

bbox 来自与光栅一致的 snapped 顶点和屏幕裁剪范围。比较面积时用宽度与阈值/行数比较，避免整数面积乘法溢出。每个 wave 计算桶内 rank 和桶计数，结合前序 wave 计数形成稳定偏移，经共享内存 scatter 后分配执行 lane。新增 16 个计数和 128 个映射，共 576 B，以及两次组同步；没有全局排序或三角形队列上传。

排序 payload 保存 `bucket + originalLocalTriangleIndex`。最终写入使用原 `recordIndex` 和该原始三角形序号，**不使用重排后的 lane 作为 ID**。页、cluster、材质、光栅 record 及软硬分类列表均不重编号，64 位 packed depth/ID 的 `InterlockedMax` 规则保留。

[GPUDrivenStreamWorkRaster.slang](../Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang) 为独立模块，避免新共享内存声明改变旧 shader 的资源占用。唯一顶点只做一次投影除法、viewport 变换、snap 和 z/w；使用已有 uint4 shared scratch 保存屏幕坐标和深度原始位，避免浮点 NaN canonicalization 改写整数坐标。bbox 和光栅共同复用这些结果，深度仍使用精确表达式；浮点深度增量关闭。原版逐顶点投影表达式保留，未引入此前 prepared-camera 实验的相机合并运算。

驱动编译统计（普通与优化符号模式一致）：

| 路径 | 寄存器/线程 | 共享内存 B/组 |
|---|---:|---:|
| 原协作装载 | 44 | 4984 |
| 复用不分桶 | 37 | 3512 |
| 复用分桶 | 37 | 3512 |

两个复用入口位于同一模块，驱动对它们报告相同的共享内存分配；不能从相同大小推断两者具有相同的同步或 LDS 访问成本。

## 开关

VBuffer 的 Runtime Settings 新增：

- **Shared SW Screen Vertices**：`softwareRasterSharedScreenVertices=true`，本轮新默认。关闭且分桶也关闭时回到上一轮协作装载路径。
- **Local SW Work Bins**：`softwareRasterWorkBins=false`，开启采用本轮已通过严格图像验收的局部分桶路径，并使用共享屏幕顶点。

以上适用于流式 cluster SW、`softwareRasterCooperativeLoad=true` 且旧 `softwareRasterPreparedVertices=false` 的路径。旧实验入口仍可显式选择。小于 32 lane 的 subgroup 使用原始 SW 回退；实际 GPU 测试使用 32 lane，未声称实测 64 lane 设备。

## 验证与证据

- 96 组分桶 GPU 参数对照：标准/reversed Z、单/双面、4/8 位 subpixel，以及 0/1/127/128/129/完整夹具的三角形数量。检查桶序、原始三角形序号的严格单调性和完整排列，并逐位比较覆盖、深度及 ID；旧 prepared 与 HW/SW 光栅回归继续通过。
- 22 组 early/late 页面装载、分类、异常页、float3、jitter 等 GPU 回归通过。
- 正式 Full 三轮：所有新入口相对 legacy 的 depth、visibility 逐位一致，前后及跨轮图像稳定，early/late 分桶和驻留一致。cut `16081482197411645748`，page mappings `11467514091117935817`，222953 个 active group。
- 最终 shader 的 12 组入口/符号模式 SPIR-V hash 与这次成功对照一致。
- 最终构建开启 validation 再测 Full 五路径、80 帧通过；不同的冻结 cut 下 depth/visibility 仍逐位一致，无 VUID/device 错误。
- 后续尝试 wave histogram 前缀和、单桶/少量三角形旁路时，Full 出现 13 个 visibility 像素差异、435 个深度像素超过 8 ULP。该组合已撤回，不放宽精确性断言；不把失败运行用于优化结论。尚未把差异归因到特定编译器变换。

证据：

- [结构化结果](ZorahFullLocalWorkBinsResult.json)
- [正式完整统计](../build-release/full-sw-work-reuse-formal/run1/Summary.json)
- [正式运行 Manifest](../build-release/full-sw-work-reuse-formal/Manifest.json)
- [最终 GPU 回归及编译统计](../build-release/sw-work-final-test.log)
- [最终 Full validation 对照](../build-release/full-sw-work-final-validation/run1/Summary.json)

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-sw-work-new `
  -Runs 1 -SwWorkComparison -Rounds 3 -SampleFrames 64 -SettleFrames 8 `
  -WarmupSeconds 10 -Width 1797 -Height 660
```

该套件显式选择五条路径，不依赖生产默认值；旧装载/阈值/元数据对照继续可用。下一步如继续推进分桶，应先量化每桶三角形数量、跨 wave 混合比例和额外同步成本，确认 128 三角形的局部范围是否足以带来收益，再扩大重排范围。
