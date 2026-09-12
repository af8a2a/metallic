# 常驻自适应 Meshlet LOD

普通 GPUScene 场景在 `VisibilityBufferPass` 中按共享 group 误差选择 LOD。选择结果接入现有 cluster 预分箱、Mesh Shader、软件光栅和异步 compute 分支，随后通过原有 visibility 解码进入 Deferred。

## 运行时设置

| 属性 | 默认值 | 含义 |
| --- | --- | --- |
| `autoLod` | `true` | 按当前裁剪相机选择自适应 cut |
| `lodPixelError` | `1.5` | 内部渲染分辨率的像素误差；减小会细化 |
| `lodBias` | `0` | 有效阈值为 `lodPixelError * exp2(lodBias)`；正值变粗 |
| `lodLevel` | `0` | 关闭 Auto 后的手动 cut；0 为最细，超出深度时保留完整终止分支 |
| `visualization` | 继承管线设置 | `lod` 为 LOD 层级着色；观察模式不参与几何选择 |

冻结裁剪相机也会冻结 LOD 所用相机。误差基于内部 render height，DLSS 改变内部渲染尺寸时会相应改变选择。视锥和 HZB 裁剪仍在选择之后执行；早、晚 HZB 和软硬光栅消费同一帧的同一份 cut。

## 数据契约

- `MeshletLodGroupRecord`：32 字节，包含共享 LOD sphere、替换误差（对象空间长度）、层级和 terminal 标记。终止组由 DAG 引用关系确定，可在不同深度结束。
- `GPUSceneGpuMeshletRecord::lod`：`{level, globalOwnerGroup, globalRefinedGroup, reserved}`，无 LOD 引用使用 `UINT32_MAX`。group 索引在 GPUScene 内全局化。
- `GPUSceneRasterDrawLayout::adaptiveRange`：所有有效 LOD cluster 的常驻候选。没有有效层级的几何使用 base range。原有 base/整档范围保留，其他 GPUScene 使用者无需改变。
- `MeshletLodSelection`：16 字节，`{instanceIndex, clusterIndex, recordIndex, geometryIndex}`。前两项是 GPU 选择结果，后两项保留既有 visibility 和几何解码关系。
- 选择缓冲前 16 字节为 `{count, capacity, candidateCount, overflow}`，后续前 `count` 项有效。容量按全部候选预留，最细 cut 也不会因输出容量不足丢失几何。
- 间接参数缓冲为两组 `uint3`：偏移 0 用于 cluster 分类，偏移 12 用于每组 32 项的 task shader。二维 dispatch 使用 65535 的行宽。

`buildMeshletLodMetadata()` 校验 group/level 范围、payload 索引、owner/refined 引用、无环和误差单调。无效层级回退 base range。当前 clod 构建器合并前代 LOD sphere，保证共享范围的层级保守性；不能用单个 cluster 的紧包围球替换它。

## 选择与调度

CPU 参考实现为 `selectMeshletLodReference()`，GPU 使用相同投影公式和判断：

```text
needsFine(group) = terminal(group) OR projectedError(group) > targetPixels
emit(cluster) = needsFine(owner) AND
                (refined 无效 OR NOT needsFine(refined))
```

投影包含球范围、偏轴位置、近裁剪面、正交投影和实例变换。实例尺度使用 `sqrt(||AᵀA||∞)` 上界，覆盖非均匀缩放、负缩放和 shear。球与近面相交时请求精细表示。手动模式用 group 层级替换误差条件，同样保留不同深度终止的分支。

GPU 按“重置 → 选择并计算块内前缀 → 块间前缀及间接参数 → 散射”稳定压缩，结果保持候选原顺序。VBuffer 编码使用原始 `recordIndex`，不会把压缩后的数组位置误当作跨帧几何身份。

选择、参数和 scratch 缓冲按 frame slot 分配；绑定更换和 resize 走现有 GPU 资源延迟回收。分箱读取 GPU 有效计数，软硬分支经原有队列同步消费同一份分箱输出。

## 调试与验证

`AfterResidentLod` 提供以下资源供 Debug Control 抓取：

- `lod.<pass>.header`：计数、容量和溢出。
- `lod.<pass>.selections`：实际 `(instance, cluster)` 和原始 record 映射。
- `lod.<pass>.arguments`：GPU 间接调度参数。
- 常规 GPUScene 抓取点另提供 `gpuScene.<pass>.lodGroups`，与 meshlets、instances、meshletDraws 联合检查。

对应回归：

```powershell
build/tests/MetallicRhiTests.exe --filter meshlet_lod
build/tests/MetallicRhiTests.exe --filter hybrid_
build/tests/MetallicRhiTests.exe --filter gpu_scene_global_gpu_resources
```

覆盖 CPU 合法 cut、混合深度终止组、GPU 稳定顺序、空输出、多块前缀、相机/变换、手动/自动切换，以及真实 Bunny 的 CPU/GPU 对照和 VBuffer ID 有效性。混合光栅测试对覆盖和 cluster ID 作严格比较；同一 cluster 中近重合三角形允许固定功能插值与 compute 深度计算产生最多 8 ULP 的舍入差异。

2026-09-12 在 RTX 5070 Ti 上验证：

- 主程序与 RHI 测试构建通过；LOD/元数据/混合光栅 7 项回归通过，其中包含 24 组稳定 GPU/CPU 列表对照、16 组真实 Bunny cut 对照和 108 帧软硬光栅检查。
- Bunny 在 193×157 透视视图中，最细 cut 为 550 个 cluster；1.5 px 自动 cut 为 73 个，eye Z 从 0.22 拉远至 0.8 后为 5 个。正交测试也覆盖了不同 LOD 混合的 cut。测试生成 `MeshletLodReport.json` 与对照 PNG。
- GPUScene、VBuffer 材质/场景切换、帧槽复用、共享相机和 Raytrace/SIGMA 回归通过。ray/raster 属性比较测试显式使用手动 LOD 0，保持几何基准一致。
- Sponza 完整实时编辑器通过 120 帧烟测，验证了 VBuffer 观察模式、串行/异步一致性和 Off 恢复，进程正常退出。
- `--rhi-realtime --filter realtime_clustered_dlss_pipeline` 的渲染断言通过（含 DLSS-SR、可选 NR 和 resize），但测试进程在 Streamline teardown 中抛出异常、退出码为 1；关闭验证层仍可复现。该测试的进程退出问题尚未解决，不能记为完整通过。

## 当前成本与后续工作

本版遍历全部常驻候选，选择成本为 O(候选 cluster × 实例)，并为最坏情况预留列表。已经能减少进入光栅的 cluster；大场景的下一步是加入空间 BVH 与 group 级裁剪，使选择成本随实际访问节点增长。尚未给出整帧性能收益结论。

StreamAsset 已接入相同误差模型和设置，按实际驻留状态生成完整 frontier，详见 [StreamingMeshletLod.md](StreamingMeshletLod.md)。当前 RT 阴影和 OMM 使用原有全场景 RT 几何。
