# MiniZorah M1：离线 cook

**M1 已完成（2026-09-12）。** 全部 3,163 个源 primitive、19,144 个 primitive instance 的流式缓存已经生成，所有页及源覆盖核对通过。运行配置为 RelWithDebInfo（优化编译）、4 个简化工作线程、6 GiB 进程提交内存硬上限，每 8 个几何保存检查点。

## 工具

`MetallicMeshletCook` 是只依赖 Scene/Task 的 CPU 工具，不启动编辑器、Vulkan、普通全量 Scene 或 RTAS。已有 `Metallic --build-meshstream` 入口仍保留。

```powershell
cmake --build build-relwithdebinfo --target MetallicMeshletCook MetallicSceneTests --parallel 2
python Tools/PrepareMiniZorahCook.py --directory build-relwithdebinfo/minizorah-m1
```

Windows 上需要在已配置 MSVC 的终端中构建。探针工具保留原始 bufferView/accessor 引用和所有选中实例，直接引用原始 `.bin`；已有且内容未变的薄 glTF 不会重写，因此不会破坏缓存身份。

全量命令：

```powershell
build-relwithdebinfo/Source/MetallicMeshletCook.exe --source E:/metallic/Asset/MiniZorah/zorah_main_public.v2.gltf --output E:/metallic/Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin --report E:/metallic/Asset/MeshletCache/MiniZorahCook/MiniZorah.manifest.json --workers 4 --memory-mib 6144 --checkpoint-interval 8 --validate-payloads
```

`--max-geometries N` 可以在本次完成 N 个几何后保存并暂停，退出码为 2；重复相同源和输出路径继续。不要把 `.partial` 当作完成的缓存。异常退出时恢复到最近已发布的检查点，未提交的几何会重新构建。

`--memory-mib` 使用 Windows Job Object 的 **process committed memory** 限制，覆盖构建主线程、并行任务和工具自身分配；它不是显存或系统总内存限额，也不是进程 RSS 限额。分配失败不会自动降低几何精度；修正资源条件后可以恢复构建。`--workers` 限制简化阶段线程数，外层按几何顺序构建和提交。平台不能安装显式内存上限时命令返回错误。

检查已完成缓存而不重新 cook：

```powershell
build-relwithdebinfo/Source/MetallicMeshletCook.exe --inspect --source E:/metallic/Asset/MiniZorah/zorah_main_public.v2.gltf --output E:/metallic/Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin --report E:/metallic/Asset/MeshletCache/MiniZorahCook/MiniZorah.validation.json --memory-mib 6144 --validate-payloads
```

检查结果使用单独文件，避免把独立验证过程的时间/峰值误当作原始 cook 的统计。

完成后，与源 glTF 独立核对覆盖范围：

```powershell
python Tools/VerifyMiniZorahCook.py --source Asset/MiniZorah/zorah_main_public.v2.gltf --manifest Asset/MeshletCache/MiniZorahCook/MiniZorah.manifest.json --events Asset/MeshletCache/MiniZorahCook/MiniZorah.manifest.json.events.jsonl --output build-relwithdebinfo/minizorah-m1/FullCook.verified.json
```

该检查核对每个源 primitive 的唯一映射、逐几何实例数、叶级三角形总数、terminal 集合及构建事件中的源 mesh/primitive/顶点/三角形计数。它要求 cooker 已完成逐页验证，不能取代二进制验证。报告记录源 glTF 和 manifest 的 SHA-256；这些哈希不是完整缓存或外部 `.bin` 的内容哈希。

## 本轮实现

- 为 offline builder 增加逐几何 decode/build/encode/complete 回调、阶段耗时、源索引与生成计数。工具追加 `.events.jsonl`，报告累计进程峰值；encode 时间包含该几何之后实际执行的检查点开销。
- 增加 `MeshletBuildOptions::maxWorkers`，默认值保持普通导入器原先的并发策略。
- 已输出的旧 LOD cluster 索引立即释放，包括 terminal 分支；原先 `clear()` 保留了旧级别的容量。
- 并行简化任务捕获异常，等待已启动的线程结束后向调用者报告，避免内存分配失败直接触发 `std::terminate`。
- 页目录记录原有 4 字节尾部 padding 改为显式清零的保留字段，仍为 104 字节，未改变 v9 布局；旧缓存仍可读。对旧 Small 基线逐字节检查，差异全部位于这些无语义的 padding，几何和 LOD 数据相同。
- 完成后生成 manifest：各 LOD 页/cluster/三角形与存储字节、全部 terminal 页集合、256 B 对齐根驻留下界、实例化 terminal group/cluster 数、目录字节和当前 frontier 状态大小。已完成缓存通过现有目录/拓扑验证，`--validate-payloads` 进一步验证每页 header、cluster 范围、指标及局部三角形索引。

## 子资产实测

2026-09-12，本机约 32 GiB RAM，优化编译；三个子资产都在 4 GiB 进程提交上限内完成，并逐页验证。Small 使用 1 个工作线程，另外两项使用 4 个。

| 子资产 | 源三角形 | 实例 | 全流程时间（含页验证） | 峰值提交内存 | 缓存字节 | 页数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small / mesh 0 | 479,502 | 1 | 1.13 s | 83.4 MiB | 25,120,949 | 622 |
| Largest / mesh 2018 | 32,054,609 | 1 | 99.19 s | 3,666.9 MiB | 1,011,703,525 | 22,533 |
| RepeatedFloor / mesh 1949 | 19,461,216 | 365 | 47.63 s | 2,236.2 MiB | 604,038,981 | 13,223 |

三个子资产各有一个 terminal 页；这不是完整 MiniZorah 根页数量的估计。Full 场景的结果以最终 manifest 为准。

三个子资产也均通过 `VerifyMiniZorahCook.py` 的源覆盖核对，复用地板的 365 个实例完整保留。

## 回归

在 `build-relwithdebinfo/minizorah-m1` 独立工作目录运行，避免复用旧任务拥有的根目录测试输出：

```powershell
../tests/MetallicSceneTests.exe --gtest_filter=SceneImport.MeshletLod*:SceneImport.MeshletStream*:SceneImport.MeshoptCompressedMeshletStreamAsset
../tests/MetallicSceneTests.exe --gtest_filter=SceneImport.MeshletPersistence
```

5 项通过。新增 `MeshletStreamCookWorkersProgressAndRecovery` 用具有多组 LOD 的两几何网格，在第一个 durable checkpoint 后注入中断，以 4 线程恢复，并与 1 线程直接构建的整个缓存逐字节比较。既有范围读取、meshopt 解码、26 GiB 稀疏源文件、拓扑、压缩和恢复测试继续通过。

## 全量结果

2026-09-12，全量一次完成，退出码 **0**。没有通过删减 primitive、实例或源三角形满足预算；恢复能力由上述中断测试独立验证。

| 项目 | 实测结果 |
| --- | ---: |
| 源 primitive / primitive instance | 3,163 / 19,144 |
| 叶级三角形 | 1,627,207,159，与源完全一致 |
| 实例化源三角形 | 18,937,042,387 |
| 叶级 cluster / 全部 LOD cluster | 16,086,545 / 33,020,491 |
| 全部 LOD 三角形 | 3,248,358,625 |
| LOD 范围 | 0–19，各几何深度不同 |
| group / page | 1,356,959 / 1,356,959 |
| 缓存文件 | 60,916,791,801 B（56.73 GiB），v9，None 编码 |
| 全流程耗时，含全部页验证 | 3,077.58 s（51 分 17.58 秒） |
| 峰值进程提交内存 / 配置上限 | 4.15 / 6 GiB |
| 全流程峰值 RSS | 9.34 GiB |
| 已发布检查点 | 396 |

所有 **1,356,959 页**均经过解码和 payload 校验。源覆盖脚本逐个核对了全部 primitive 的唯一映射和实例数量，并将构建事件的 mesh、primitive、顶点数及三角形数与 glTF metadata 对应。正式资产可重新打开且源依赖身份匹配；`.partial` 和 `.meshopt-cache` 均已清理。

内存上限限制的是 committed memory。几何构建完成前记录到的 RSS 峰值为 3.55 GiB；后续完整缓存映射读取和逐页验证使全流程 RSS 峰值达到 9.34 GiB，因此不能把本次结果描述为“RSS 限制在 6 GiB”。

阶段累计时间：decode 149.22 s、meshlet/LOD 构建与简化 2,560.50 s、encode/检查点 250.01 s；最后一个几何完成于 2,963.79 s，后续目录收尾、缓存打开和全部页验证共 113.79 s。阶段累计不包含所有初始化与回调开销。临时目录按约一分钟间隔采样，观测到 31,562,322,674 B 的峰值下界；该采样值不是精确临时磁盘峰值。

### 交给 M2 的预算数据

| 项目 | 实测结果 |
| --- | ---: |
| 完整 terminal 页集合 | 3,163 页，每个几何 1 页 |
| terminal payload，256 B 对齐 | 5,731,584 B（5.47 MiB） |
| 最大设备 payload / 对齐槽位 | 74,688 / 74,752 B |
| terminal 加一个流式页的字节下界 | 5,806,336 B（5.54 MiB） |
| 实例化 terminal group / cluster | 19,144 / 19,144 |
| 资产主目录数组，CPU 布局 | 421,920,120 B（402.37 MiB） |
| 当前每实例 frontier 状态合计 | 162,274,640 B（154.76 MiB） |

根页 payload 很小，位置压缩暂未成为完整粗表示首帧的硬门槛。M2 必须将现有 1,024 根页限制提高到至少 3,163（例如预留 4,096），并另外分配可细化页面的数量和字节预算。原路线的 2 GiB page pool 可以作为待测起点；上表不是总 GPU 内存预算，还需要计算目录转换/副本、候选输出、帧槽、附件等。全量缓存约 56.73 GiB，后续仍需测量真实流式工作集和 I/O 收敛成本。

产物与证据：

- [正式缓存](../Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin)
- [完整 manifest，含每个几何和 terminal 页 ID](../Asset/MeshletCache/MiniZorahCook/MiniZorah.manifest.json)
- [可纳入版本管理的结果摘要](MiniZorahCookResult.json)
- [全量运行日志](../build-relwithdebinfo/minizorah-m1/FullCook.log)、[退出码](../build-relwithdebinfo/minizorah-m1/FullCook.exit.txt)
- [独立源覆盖核对结果](../build-relwithdebinfo/minizorah-m1/FullCook.verified.json)
