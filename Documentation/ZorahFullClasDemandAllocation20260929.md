# CLAS 按需物理分配（2026-09-29）

## 实现范围

`MeshletStreamCompactClasPool` 的持久 CLAS 存储从一次性申请 `maxStorageBytes` 改为独立 Buffer 分块分配。Full 的上限仍为 2 GiB；默认增长粒度为 64 MiB，通过 `MeshletStreamClasPoolDesc::storageChunkBytes` 配置。初始化不申请持久 CLAS 数据块，已有暂存构建池、MOVE scratch、地址表和页表仍预分配。

先完成暂存 CLAS 构建并读回实际尺寸，再在已有块中分配；没有连续空闲区域时才增长。每页完整落在一个块内，超出粒度的页面申请能容纳该页的块，尾块受剩余预算约束。MOVE 使用对应块的目标 Buffer，地址表发布绝对设备地址。增长不搬移活页、不修改稳定 cluster ID。

`storageBuffer()` 仅为旧的连续池/暂存 builder 提供单 Buffer；compact 池返回空指针。需要检查某页底层存储时使用 `pageStorageBuffer(pageIndex)`；运行时仍使用地址表与页表，没有新增 renderpass 加载逻辑。

## 回收与预算

- 页退休同时保留帧宽限期和退休时在途提交的完成条件。未提交命令也不能被误判为完成；取消后可继续回收。
- 每个帧完成点每次维护只查询一次，供块和退休页面共享，避免逐页重复查询 GPU。
- 完全空闲且没有在途使用的块释放底层 Buffer。命令保留一个帧资源租约；完成后清空租约里的 Buffer 引用，防止已完成但尚未重置的命令对象继续占用显存。
- 全局预算拒绝增长时，保留已构建的暂存源供重试；30 帧内抑制新的物理增长尝试，仍允许利用现有空闲块。空块释放后可以提前恢复增长。
- 非预算类分配错误仍返回失败。MOVE 取消、待构建页放弃和地址槽不足都回收目标分配。

统计明确分开：`storageBudgetBytes` / profiler `clasCapacityBytes` 是预算；`storageBytes` / `clasAllocatedBytes` 是池持有的 Buffer 字节；`usedStorageBytes` / `clasUsedBytes` 是页子分配字节。增加 `clasStorageChunks`。冷页回收阈值继续使用预算，避免把刚增长的第一块当成整个容量。Profiler 显示实际 backing/块数，Full 漫游 JSON 同步导出。

Buffer 字节不等于整卡显存：VMA 块保留、驱动开销、其他资源和其他进程需要另看设备预算/NVML。本次只释放全空块，未实现跨块活页整理，长期碎片可能仍使 backing 接近预算。

## 验证

沿用现有 MSVC Release 构建，无 SDK/编译器/默认质量变化：

```powershell
cmake --build build-release --target MetallicGPUDrivenSample -j 6
cmake --build build-scheduling-release --target MetallicRhiTests -j 6
$env:METALLIC_TEST_MINIZORAH='1'
.\build-scheduling-release\tests\MetallicRhiTests.exe --gtest_filter=RhiResource.clas_actual_sizes_and_move:RhiResource.clas_compact_lifecycle:RhiRendering.minizorah_clas_in_flight:RhiRendering.stream_clas_runtime_lifecycle:RhiRendering.stream_clas_eviction_reupload --rhi-validation --output-dir build-scheduling-release/clas-demand-verified
```

5 项通过，无跳过。生命周期测试覆盖零初始 backing、实际尺寸 MOVE、两块增长、每页地址归属、活页地址稳定、过期/复活、未提交帧阻止回收、取消构建/搬移与重试、预算不足不发布地址、空块实际从设备 Clas 预算域释放。MiniZorah 运行 1,200 帧，其中 1,198 帧观测到在途重叠，并逐帧检查 used ≤ allocated ≤ budget。另有旧池流送生命周期及预算耗尽/重上传回归。日志含机器上已存在的 Vulkan overlay manifest 缺失警告，不是 GPU 验证错误。

## 同条件 Full 漫游

比较改前二进制与改后二进制；shader 摘要和全量 cook 文件相同。输出 1797×660，渲染 1198×440，DLSS Quality，1.5 px LOD，8M raster candidates。沿用 `work-capacity-route-8m.json`：固定 180 帧绝对相机路线，逻辑时长 30 秒、预热 3 秒，每 30 帧采集工作量。每轮从新进程进入 Full，已有磁盘 cook/shader/纹理缓存；这不是冷磁盘测试。

该脚本运行隐藏的编辑器渲染路径；工作量读回会扰动诊断帧耗时，内存单轮对比不能作为 30 fps 或纯性能验收。

| 指标 | 整池预分配 | 最终版按需块分配 |
|---|---:|---:|
| CLAS 预算 MiB | 2,048 | 2,048 |
| 初始化持久 CLAS backing MiB | 2,048 | 0 |
| 漫游采样 backing MiB | 2,048 | 1,536–1,792 |
| 漫游采样 backing 块数 | 1 | 24–28 |
| CLAS 子分配峰值 MiB | 1,733.585 | 1,733.616 |
| CLAS scratch 统计字节 | 167,679,945 | 167,679,945 |
| 完成漫游帧数 | 180 | 180 |
| NVML 首个样本 MiB | 3,539 | 3,652 |
| NVML 整卡峰值 MiB | 13,305 | 13,368 |
| 全量准备时间（秒） | 103.72 | 99.26 |

同路线中持久 CLAS backing 的采样峰值减少 **256 MiB**，初始分配减少 **2 GiB**；暂存、scratch、几何和纹理策略未改。两组实际 CLAS 子分配峰值几乎相同，未通过降低几何质量取得此差值。两组 loadFailures/requestOverflows 均为 0；几何 allocationFailures 单帧最多为 1，两组均出现，它是流送准入计数，不能表述成“所有分配从未被拒绝”。

**尚未证明整卡峰值稳定下降。** 首轮分块实现记录过 12,906 MiB，但增加完成后资源租约释放后的最终版为 13,368 MiB，改前为 13,305 MiB。整卡观测有明显轮间波动，初始后台占用也不同；不能挑选首轮的 399 MiB 降幅作为最终收益，也不能仅凭初始占用差值归因。已确认的收益是 CLAS 自身的分配字节及空块实际释放。本次未分解剩余驱动/其他资源峰值，不据此宣称整卡显存或帧时间改善。

### Full 图像与切换验证

最终二进制另通过 `RhiRendering.zorah_full_first_frame`，启用验证层，设置 `METALLIC_TEST_ZORAH_FULL=1`、`METALLIC_ZORAH_FULL_CYCLES=1`。完成一次 MiniZorah→Full 切换、全量 terminal cut 准备、120 帧后续渲染、材质分桶/非分桶检查、场景释放。检查了 settled 与 base-color 输出，建筑、植被、人物和材质均有正常几何覆盖。该检查为 960×540 原生分辨率、关闭 DLSS、背景关闭；图像有单样本着色噪声，不能等同于编辑器 DLSS Quality 图像逐像素等价或长时间漫游稳定性。合计 6 项 GPU 测试通过。

Full 图像证据：`build-scheduling-release/clas-demand-full-visual/{results.json,ZorahFullFirstFrame.json,ZorahFull-settled-0.png,ZorahFull-base-color-0.png}`。


本地证据（构建目录，不纳入源码）：

- `build-scheduling-release/clas-demand-verified/results.json`、`CompactClasLifecycle.txt`、`MiniZorahClasInFlight.jsonl`。
- `build-release/clas-demand-baseline/run1/{Capture.json,Frames.jsonl,Gpu.csv,Summary.md}`。
- `build-release/clas-demand-final-roam/run1/{Capture.json,Frames.jsonl,Gpu.csv,Summary.md}`。
- `build-release/clas-demand-comparison.json`：设置、哈希、峰值及逐帧统计。
