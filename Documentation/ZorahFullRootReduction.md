# ZorahFull 合法保底集缩小

2026-09-29：已实现新的 LOD cook 策略，并完成三个真实大型 primitive 的离线和 GPU 验证。本页记录探针阶段的结果；用户随后要求取消旧资源兼容并执行 Full 全量重建。下列收益只代表三个独立探针，不能当作 Full 总量或加载时间的改善。

## 策略与覆盖约束

原策略对同位置顶点的任意法线分量差都设置 `meshopt_SimplifyVertex_Protect`。Dome 的 4,153,573 处法线差异中，4,052,900 处最大原始分量差不超过 `1e-4`；其 LOD0 的 2,181 个 group 有 2,173 个因简化输出仍超过输入的 85% 而终止。Window 的 1,240 个 group 全部在 LOD0 终止。仅按完整属性逐位合并顶点，三个探针均无收益：它们引用的完整属性 tuple 全部不同。

最终策略：

- LOD 构建前，仅在本地索引副本中合并 P3/N3/UV2/T4 全部逐位相同的引用，比较时排除结构体 padding。
- 法线与同位置代表顶点比较，按**归一化方向**的最大分量差判断，`<= 1e-3` 不设置硬接缝 Protect，但法线继续参加已有的加权属性简化误差。较大的法线接缝、UV 差异和切线符号差异继续受到保护；有属性时仍禁用 sloppy fallback。
- 不修改源顶点、法线、UV、切线或原始 indices；粗层引用仍来自原始属性 tuple。LOD0 保持完整三角形多重集和绕序。
- 简化结果为空，或输出仍超过输入的 85%（减面不足 15%）时，保留原 group 作为 terminal，避免空结果丢失分支。完整保底集仍取**所有层级的全部 terminal 分支**，不截取最高 LOD，也不从停滞 group 中任取一个 cluster。

方向比较使用 double 长度，避免非单位短法线误判：例如长度 `1e-4`、相差 90° 的两条法线必须保留接缝。同向不同长度不会被错误视作方向硬边。`1e-3` 是接缝分类阈值，并非最终粗 LOD 的几何或着色误差上限；粗根画面仍会损失细节，正常运行继续按已有 LOD 误差和流送逻辑细化。

实现入口：[scene.cpp](../Source/Runtime/Scene/scene.cpp)、[GeometryAttributes.h](../Source/Runtime/Scene/GeometryAttributes.h)。

## 实测结果

三项均保留源属性、材质、纹理引用与实例变换，分别有 1,733,192 / 3,678,446 / 986,422 个源三角形，每项一个实例。基准和候选使用相同源、4 workers、8 GiB process commit 上限、ByteRle 磁盘编码。字节为 compact shading 的设备 payload 加 256 字节分配对齐，**不是总显存**。

| 探针 | 根页：原 → 新 | 根 cluster：原 → 新 | 根 payload MiB：原 → 新 | 减少 |
|---|---:|---:|---:|---:|
| Dome / primitive 5152 | 2,181 → 167 | 54,120 → 4,097 | 146.71 → 12.07 | 91.77% |
| Cylinder / primitive 4108 | 2,662 → 2,282 | 66,232 → 56,739 | 110.54 → 96.72 | 12.51% |
| Window / primitive 820 | 1,240 → 480 | 30,424 → 11,798 | 84.39 → 34.83 | 58.72% |
| 合计 | 6,083 → 2,929 | 150,776 → 72,634 | 341.64 → 143.62 | 57.96% |

Cylinder 仍有约 368 万个受保护顶点，收益明显小于 Dome，不能把穹顶的比例推广到整个场景。新策略增加了可用的中间 LOD，三项磁盘缓存合计从 653.78 MiB 增至 987.61 MiB（+51.06%）。未清空 OS 文件缓存，没有进行加载耗时 A/B，因此不报告端到端加速。

汇总与原始报告：[可复核摘要](ZorahFullRootReduction20260929.json)、[完整本地报告目录](../build-release/root-shrink-probes-20260929/Summary.json)。`Baseline` 是原 revision 2，`Exact` 是仅逐位合并实验，`NormalTolerance` 是原始分量实验，**`Normalized` 才是最终策略**；中间实验文件只保留作证据。

## 验证

- 三项最终 `Normalized` cook：逐页 payload 校验、所有 LOD 顶点完整属性 tuple 精确匹配、LOD0 有向三角形多重集检查全部通过。运行时打开还校验层级引用、所属 primitive、LOD/误差顺序和 terminal 标记。
- CPU：19 项通过；可选的旧 `ZorahProbesMatchIndependentResidentImport` 因未配置其专用 manifest 跳过。新增用例包含重复顶点 soup、微小方向差、短法线大角度接缝、同向不同长度、UV/切线保护、源属性与绕序、缓存兼容和 checkpoint 重启。[日志](../build-scheduling-release/root-reduction-scene-normalized.log)。
- 6 项 RHI cut/metadata/frontier/BVH/tile/GPU-reference 回归通过。[日志](../build-scheduling-release/root-reduction-gpu.log)。
- 三项真实 GPU 探针通过，RTX 5070 Ti / 驱动 616.92，开启 Vulkan validation，未发现 `VUID-`。首帧完成全部根 geometry 与 CLAS：Dome 334/334、Cylinder 4,564/4,564、Window 960/960 个 readiness 资源步骤。CLAS 启用时每根页计两步。[GPU 日志](../build-scheduling-release/root-reduction-gpu-fixed.log)。
- 首帧固定粗根、关闭视锥裁剪，保存整体 `stream-root-*` 与独立 resident LOD0 参考图并检查图像；随后恢复 LOD0 近景，四种视图共 12 项像素对照通过：mappedNormal、baseColor、normalTexture、final。RGB 8-bit 平均绝对误差最大约 0.267，最大异常像素比例约 0.10%。[逐项结果](../build-scheduling-release/root-reduction-gpu-fixed/ZorahZ4Probes.json)。

整体根图用于验证可见结果和观察粗化，不宣称与 LOD0 逐像素等价；像素误差对照的范围是最大源三角形附近的 LOD0 近景，不代表所有视角或长时间漫游。首轮 GPU 测试误读静态图属性，按未启用 CLAS 计算期望步骤而失败；实际 readiness 已完成。测试改为合并运行时属性后重跑通过，原日志保留。

测试 manifest 为 [GpuProbes.json](../build-release/root-shrink-probes-20260929/GpuProbes.json)，临时容量仅用于容纳探针全部页。没有修改生产图的几何、CLAS、普通流送预算或 readiness 条件。

## 缓存与使用

`kGeometryCookRevision` 从 2 升至 3，resident meshlet cache 从 3 升至 4。按后续全量重建要求，运行时只接受当前 cook revision，已移除 revision 2 和 legacy position-only 的兼容例外；旧缓存需要重新 cook。partial checkpoint 记录并核对 cook revision，防止新旧策略混合续建，同版本正常续建保留。

新离线 cook 默认采用该策略；已有 Full 正式缓存不会因此自动变小。以后重建时必须预留完整输出及临时 decode/checkpoint 空间，再验证完整场景。诊断可用：

```powershell
build-scheduling-release/Source/MetallicMeshletCook.exe --source <probe.gltf> --output <new.meshstream.bin> --report <report.json> --workers 4 --memory-mib 8192 --compression byte-rle --validate-attributes --lod-diagnostics
build-scheduling-release/Source/MetallicMeshletCook.exe --inspect --compact-shading --output <new.meshstream.bin> --report <compact.json>
```

`--lod-diagnostics` 仅记录本次实际构建的 primitive，包含每层输入/目标/实际三角形、终止原因计数、精确 tuple 数与原始属性差异分布。`normalSeamNormalizedComponentTolerance` 是策略阈值；`normalDifferences` 仍统计**原始**分量差，二者口径不同。
