# MiniZorah M4：遍历长尾与驻留复用优化

2026-09-13，RTX 5070 Ti / 驱动 616.64，RelWithDebInfo。完整 MiniZorah cook、1920×1080、1 GiB 页池、自动 LOD 1.5 px、异步 HW/SW；与 [M4 初始验收](MiniZorahRoaming.md) 使用相同路线和预算。

## 实现与依据

新增 `BeforeStreamUpdates`、`AfterStreamUpdates`、`AfterStreamFrontier`、`AfterStreamPrefix`、`AfterStreamEmit` 检查点。修改前第 15 秒近景的 frontier 区间为 40.17 ms，emit 为 4.50 ms；prefix 约 0.064 ms，页更新约 0.019 ms。frontier 区间包含 reset 与阶段屏障；计时检查帧与普通性能帧分别统计。

### 每实例工作组协作

运行时从现有 group 元数据派生不跨 LOD 层级、最多 64 个 group 的 tile，沿用 BVH 的保守球和误差上界。tile 按 group ID 从大到小排列，终止页始终参与；不改变磁盘 cook 格式。原 BVH 与线性 shader 保留为独立对照。

新的 `streamCooperativeLodMain` 每实例使用 64 线程。每个 tile 内并行计算父组状态、需求和 drawable；父组的 LOD 严格更高，tile 间使用工作组内的全内存屏障。稳定前缀将 active IDs 按原顺序写入稀疏列表，再并行计算 cluster mask。emit 同样按升序 group ID 稳定写出原始记录；全局 prefix、完整 terminal 回退和 VBuffer 身份保持原契约。被完全替换的 ancestor 也保留在稀疏清理列表中，避免转向或重载后残留 active 状态。

### 未使用页留作缓存

原 GPU unload 列表表示“本帧未使用的 resident 页”，此前 CPU 将其直接转为卸载任务。现在完整 StreamRuntime 将这份列表作为需求反馈：显式未使用页留在预算内缓存，完整列表中未出现的 resident 页刷新使用时间。空反馈表示所有驻留页仍被使用；截断反馈仅允许显式列出的冷页成为淘汰候选。

预算不足时，原批量淘汰流程只选择 GPU 确认未使用、满足年龄条件的非根 resident 页，仍遵守每帧 256 页与延迟释放。`unloadPage()` 及非反馈批次的显式卸载接口保留原行为。根页、PendingUpload 和在途 I/O 不因缓存策略提前释放。新增 `frameCachedUnusedPageCount` 与 `frameResidentDemandCount` 便于观察复用。

### 避免反复扫描碎片

保留页池后，分配压力持续存在，暴露出 `MeshletStreamStorage::canAllocate()` 对每个失败请求重复扫描空闲块的成本。现在缓存最大空闲块和考虑对齐后的最大可分配区间；释放/合并或消耗最大块时使缓存失效，其他分配保留有效界限。重复失败检查为常数成本，实际分配仍沿用原 first-fit 地址顺序。顺带拒绝对齐大小溢出的请求。

## 60 秒路线对照

本轮先测当前基线，再测协作遍历和缓存策略，最后加入空闲区间缓存。统计来自各运行的完整计时帧；累计装卸量截至第 55 秒检查点。

| 指标 | 修改前 | 协作遍历 + 驻留缓存 | 加入分配检查缓存 |
| --- | ---: | ---: | ---: |
| GPU P95 | 59.90 ms | 23.70 ms | 24.06 ms |
| 同步帧 P95 | 67.21 ms | 48.60 ms | 31.66 ms |
| CPU 记录 P95 | 7.72 ms | 27.06 ms | 8.89 ms |
| 第 15 秒 frontier / emit | 40.17 / 4.50 ms | 1.59 / 0.161 ms | 1.44 / 0.157 ms |
| 第 15 秒实际 cluster 候选 | 342,448 | 390,511 | 380,815 |
| 候选采样峰值 | 346,943 | 457,842 | 458,477 |
| 累计上传字节 | 6,419,642,112 | 3,116,817,312 | 3,156,006,240 |
| 累计完成卸载 | 155,725 | 54,448 | 55,712 |

最终 60 秒对照中，GPU P95 下降约 60%，同步帧 P95 下降约 53%，上传字节下降约 51%，卸载次数下降约 64%。候选与缓存细节增加；没有降低分辨率、放宽 LOD 阈值或固定最粗 LOD。保留更多细节后，P50 不再被大量近乎仅含根页的帧拉低，因此中位数不能单独用来判断这项优化。

路线按累计渲染时间推进，流式加载具有异步性；各运行并非逐帧完全相同的 residency/cut。阶段时间也含屏障，不能将上述比值解释为严格固定输入的单内核加速比。测试每帧同步等待，计时帧不读回输出/debug，离屏图省略 FinalBlit，启用 Vulkan 验证；数据不是编辑器呈现 FPS。

单独将静态拓扑改成 `MemoryLocation::Device` 的对照没有收益（GPU P95 61.23 ms），已撤回该实验。`HostUpload` 枚举本身不能证明实际分配位于系统内存或经 PCIe 读取；最终优化来自工作组织和驻留复用。

## 最终验收

最终版本构建 `MetallicRhiTests`、`MetallicGPUDrivenSample` 和 `Metallic` 成功。33 项回归全部通过（163.809 秒），覆盖流式 residency、元数据、混合 producer、场景切换、统一 VBuffer、异步 HW/SW 与 MiniZorah。线性、BVH、协作三个实现共 219 组 GPU/reference cut 对照通过；新增测试覆盖冷页缓存、空/截断反馈、热页保护、延迟释放以及碎片池连续失败、合并与对齐溢出。回归和长测日志均无 Vulkan validation error 或 device loss。

64 MiB、60 秒压力路线通过 12 个检查点和 5,805 个计时帧：GPU P95 9.99 ms，同步帧 P95 13.93 ms，第 55 秒累计上传 258,321,440 B、完成卸载 13,763 次。此档依赖较粗 cut，不能据此宣称 1.5 px 质量收敛。

1 GiB、660 秒巡航通过 33,786 个计时帧和全部 132 个检查点。每 5 秒核对根覆盖、共享父组依赖、cluster mask、候选数量和预算；无容量回退、页面加载失败或无效 GPU 请求。与原 M4 的同预算长测对照如下，累计装卸量均截至第 655 秒：

| 指标 | 原 M4 长测 | 本轮最终长测 |
| --- | ---: | ---: |
| GPU P95 / P99 | 61.36 / 68.41 ms | 23.86 / 24.91 ms |
| 同步帧 P95 / P99 | 68.99 / 76.45 ms | 32.68 / 40.61 ms |
| 同步帧最大值 | 247.90 ms | 250.04 ms |
| 累计上传字节 | 77,479,478,304 | 23,891,273,792 |
| 累计完成卸载 | 1,845,629 | 614,481 |
| cluster 候选采样峰值 | 415,386 | 458,814 |

GPU P95 下降 61.1%，同步帧 P95 下降 52.6%，累计上传下降 69.2%，卸载下降 66.7%。同步帧 P95 达到路线的 33.3 ms 观察目标；P99 和最大值仍有尖峰，不是每帧截止时间保证。最终 CPU 记录 P95 / P99 为 9.94 / 15.82 ms，GPU 最大值为 28.61 ms；原 M4 长测 CPU 采样为无效零值，不作比较。

页池采样峰值 1,068,755,456 B，低于 1 GiB。进程本地显存热身周期峰值为 2,401,689,600 B，最终周期为 2,399,920,128 B，通过允许增长 64 MiB 的平台检查；进程提交内存采样峰值 4,003,262,464 B，工作集采样峰值 3,943,645,184 B。保守 refinement 超标数采样峰值为 8,640，近裁面无界项峰值为 6,092，质量条件仍开放。

完整数值、原始报告路径与 SHA-256、最终源文件 SHA-256 见 [结构化结果](MiniZorahRoamingOptimizationResult.json)。原始日志、每 5 秒 JSON 与图像位于 `build-relwithdebinfo/minizorah-m4-opt/`；可查看 [长测原始报告](../build-relwithdebinfo/minizorah-m4-opt/cruise/MiniZorahRoamingReport.json) 和 [近景输出](../build-relwithdebinfo/minizorah-m4-opt/cache/roam-15.png)。

在已构建的仓库根目录复现长测：

```powershell
$env:METALLIC_TEST_MINIZORAH = '1'
$env:METALLIC_MINIZORAH_ROAM_SECONDS = '660'
$env:METALLIC_MINIZORAH_ROAM_MIB = '1024'
& .\build-relwithdebinfo\tests\MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_roaming --rhi-validation --rhi-async-compute --output-dir E:/metallic/build-relwithdebinfo/minizorah-m4-opt/recheck
```

将秒数和预算分别设为 `60`、`64`，使用下面的 filter 可复现本轮 33 项回归；将输出目录另设以保留长测结果。

```text
*streamer_meshlet*:*meshlet_lod_stream*:*hybrid_*:*gpu_scene*:RhiRendering.stream_metadata_*:RhiRendering.render_graph_gpu_driven_mixed_producer_render:RhiRendering.render_graph_scene_binding_contract:RhiRendering.visibility_buffer_async_scene_handoff:RhiRendering.minizorah_vbuffer:RhiRendering.minizorah_roaming
```

## 后续边界

1.5 px 质量收敛仍需要单独验收。保守 refinement 超标数包含不可见部分、共享父依赖和近裁面无界投影，不是超标像素数；预算满时保留完整粗 cut 并报告推迟请求。缓存减少了重复装卸，有限预算下仍可能需要重新加载。后续重点是屏幕收益优先级、层级可见性需求和驻留编码；GPU 长尾已从 frontier 转向分类/分箱与光栅阶段。
