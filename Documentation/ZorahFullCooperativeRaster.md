# ZorahFull：精简 SW 描述与首 wave 协作装载

## 结果与默认策略

协作装载已设为流式 cluster SW 光栅的默认路径。RTX 5070 Ti 上三轮换序、每模式 192 帧的固定状态对照：

| 路径 | SW ms | 分类 ms | 光栅合计 ms | Graph ms |
|---|---:|---:|---:|---:|
| 全 HW | 0 | 0 | 19.326 | 21.791 |
| legacy SW | 14.092 | 0.584 | 17.516 | 19.984 |
| 首 wave 协作装载 | **8.507** | 0.583 | **11.920** | **14.408** |

相对 legacy，SW 降低 **39.63% / 5.585 ms**，光栅合计降低 **31.95% / 5.596 ms**。三轮 legacy/cooperative 分别为 14.040/8.447、14.078/8.610、14.160/8.465 ms，收益方向一致。

条件：编辑器输出 1797×660，DLSS Quality 内部 1198×440，LOD 1.5 px，完整材质和阴影，8 px 分流、保守元数据分类。冻结相机、cut、几何/CLAS/纹理驻留及 TLAS，关闭 jitter，串行 HW/SW 以便归因。共 576 个计时帧，诊断读回和恢复帧不计时。这是 RHI timestamp 内核对照，不能据此宣称持续漫游达到 30 fps。

## 实现

[StreamRasterCooperative.slang](../Shaders/Features/GPUDriven/StreamRasterCooperative.slang) 使用独立 SW 描述，只保留所选 render/cull camera、实例变换、顶点/三角形地址和计数、可见性 ID 与实例标志。描述名义大小 **592 → 176 B**；不携带遍历、LOD、材质、请求、包围球和锥体状态。

首个完整 wave 分摊参数、group 字段和页面头读取，通过 wave shuffle 广播后逐阶段校验；页面头每 lane 读取一个 word。验证范围后，17 个 lane 分别写相机、变换和最终光栅字段。阶段临时标量复用并在最后使用后结束存活，避免全工作组保留大型结构。其余 wave 在原有组屏障等待，不增加中间组屏障。

之后仍按唯一顶点投影、每 lane 处理一个三角形，调用原 `hybridRasterTriangle`。保持投影表达式顺序、jitter、反射绕序、双面、subpixel、深度/可见性打包和 64 位原子竞争规则。没有合并此前未证实提速的 prepared-vertex/浮点深度增量实验。

入口支持 32/64 lane wave；小于 32 lane 时回退 legacy。非流式和三角形队列路径不受此选择影响。页面/cluster/instance 边界验证保留；异常页面仍统一拒绝，所有 wave 能到达组屏障。

VBuffer 属性 `softwareRasterCooperativeLoad` 默认 `true`；显式设为 `false` 可回到 legacy。`softwareRasterPreparedVertices` 默认 `false`，显式开启时仍优先使用独立 prepared 实验路径；`softwareRasterIncrementalDepth` 继续默认关闭。

## 编译资源与正确性

驱动 pipeline executable statistics（普通模式及优化符号模式结果一致）：

| 资源 | legacy | cooperative |
|---|---:|---:|
| 每线程寄存器 | 96 | **44** |
| 每工作组共享内存 B | 9408 | **4984** |
| 机器码 B | 30208 | 24064 |

资源下降支持了减少装载状态的方向，但没有重新采集 Nsight 的 active-lane/occupancy 指标，不能把寄存器减少直接换算成 occupancy。驱动返回异常大的 Local Memory Size 原值，保留在结构化证据中，不能用来断言无 spill。

- GPU early/late 分类与装载对照 22 组通过：render/cull camera、jitter、反射/非均匀变换、float4/float3、错误格式/页长/偏移、超界 cluster、空和复用场景等。新旧 clip 逐位比较，局部索引和 validity 一致。
- 既有混合光栅覆盖、深度和 overflow GPU 回归通过。
- 开 validation 的默认 Full 配置、持续流送路线回归通过：30 秒、73 帧，完整 GPU 计时且无 validation/device 错误。validation 下 CPU 帧耗时均值 414.5 ms，不作为性能数据。
- Full 开 validation 的固定状态短测通过。正式三轮中 cut `16081482197411645748`、page mappings `11467514091117935817`、222953 个 active group 一致；每模式前后及跨轮图像稳定。
- **cooperative 与 legacy：527120 像素的覆盖、visibility ID、深度逐位相同，early/late 分桶相同。** 与全 HW 的已有差异仍存在，不能把新旧 SW 一致解读为 SW/HW 全部一致。
- 正式采样后整卡回落至 4–5% 占用。原先一次设备枚举失败的运行未产生 Capture，不纳入性能结果；后续 GPU 恢复后重新运行并通过。

## 正常漫游回归

关闭 validation 后，默认 Full 配置沿固定路线运行 30 秒，399 帧，GPU 计时全部有效，无 device/validation 错误。平均帧耗时 **75.317 ms**、p95 **84.140 ms**，尚未满足 30 fps。该 Full 配置的 `asyncSoftwareRaster=false`，本次未验证异步 SW 路径。未采集同条件 legacy 漫游对照，不能用该数值量化整体漫游提速；上表的收益只适用于同 cut 的内核对照。

- [正常漫游 Capture](../build-release/full-sw-cooperative-roam-release/run1/Capture.json)
- [正常漫游逐 scope 统计](../build-release/full-sw-cooperative-roam-release/run1/Summary.json)
- [validation 漫游 Capture](../build-release/full-sw-cooperative-roam/run1/Capture.json)

## 证据与复现

- [结构化结果和原始编译统计](ZorahFullCooperativeRasterResult.json)
- [正式采样 Manifest](../build-release/full-sw-cooperative-formal/Manifest.json)
- [正式完整统计](../build-release/full-sw-cooperative-formal/run1/Summary.json)
- [validation 短测](../build-release/full-sw-cooperative-validation2/run1/Summary.json)
- [22 组与光栅 GPU 回归日志](../build-release/sw-cooperative-boundary-test.log)

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 `
  -OutputRoot build-release/full-sw-load-new `
  -Runs 1 -SwLoadComparison -Rounds 3 -SampleFrames 64 -SettleFrames 8 `
  -WarmupSeconds 10 -Width 1797 -Height 660
```

三种模式均显式选择被测 pipeline，不依赖默认值。正式采样结束后仅调整生产默认值和对照 harness 的选择/恢复；被测 shader 没有再修改。旧 `-SwComparison` 继续显式选择 legacy/prepared/plane，阈值及元数据对照默认使用新的生产路径。
