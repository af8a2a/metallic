# MiniZorah：CLAS 实际尺寸、搬移与联合冷页回收

2026-09-13。已在 MiniZorah VBuffer 默认启用，不需要重新 cook。`build-release/Source/MetallicGPUDrivenSample.exe` 和 `Metallic.exe` 已构建成功。

当晚修复了切换后 CLAS 搬移 scratch 大小取错字段导致的 GPU 越界写入，详见 [崩溃复现、修复和验证](MiniZorahClasMoveCrash.md)。原有内存对照未重跑；新增移动 scratch 约 192 KiB，计入工作区，不改变几何/CLAS 驻留分配统计。

## 实现

- RHI 支持 CLAS 构建时输出每个对象的实际字节数，以及 `MOVE_OBJECTS_NV` 重定位。检查范围、对齐与搬移目标重叠，使用驱动搬移命令修复内部地址。
- 新的 `MeshletStreamCompactClasPool` 使用已有固定槽池作为有界临时构建区。构建完成后异步读取尺寸，由已有 CPU 分配器按每个 CLAS 的实际尺寸及硬件对齐打包，再搬移到持久池。
- 构建和搬移分别跟踪 `RenderFrameContext` 完成点与提交事务。只有搬移完成才发布共享页面/cluster 地址；运行时没有新增 fence/timeline 阻塞等待。没有新页时仍推进未完成批次，正在构建的页不重复消耗构建预算。
- 处理构建或搬移录制取消、构建中卸载、快速重载和完成后的延迟释放。暂时无法分配的页留在队列等待空间；仍可由 VBuffer 绘制几何。
- 联合回收从 GPU 明确报告未使用的页中选取，几何和该页 CLAS 使用同一个页面生命周期。保护锁定回退页、近期上传和近期使用页。CLAS 在几何卸载完成后退役，再经过 queued-frame 延迟释放。
- 复用每帧一次的候选扫描，每帧最多驱逐 256 页。先扣除已排队卸载/退役的字节，避免 GPU 延迟释放期间重复扩大驱逐。

当前尺寸分配仍由 CPU 完成，实际尺寸经过异步读回；参考实现的全 GPU 分配/描述生成尚未移植。这会增加构建到可用的完成轮次，但不阻塞渲染线程。当前搬移用于将新 CLAS 从临时区放入紧凑持久区，不主动重新整理仍被引用的持久对象。

## 默认参数

```json
"enableClas": true,
"enableClusterRtx": false,
"compactClas": true,
"coldPageRetentionFrames": 120,
"maxClasBytes": 536870912,
"maxClasBuildClusters": 8192
```

两个池任意一个达到 85% 时进入回收压力状态，目标降到 70%；压力下最低闲置年龄 16 帧，普通闲置页保留 120 帧。设 `coldPageRetentionFrames=0` 可保留原来仅分配失败时驱逐的策略。`compactClas=false` 保留原有固定槽路径；紧凑路径要求命令使用可跟踪的帧提交。

几何池仍预留 1 GiB。CLAS 持久池预留从 1,536 MiB 降至 512 MiB。构建/搬移工作区约 156.1 MiB，含临时 CLAS、scratch、构建描述和临时地址/尺寸表，独立显示，不混入图表中的驻留 CLAS。持久地址表、几何池预留、其他渲染缓冲也不属于图表的有效驻留字节，因此下表不是整机显存占用。

## 同轨迹对照

RTX 5070 Ti，1920×1080，1.5 px，Vulkan validation 与异步 Compute；前 60 帧静止，随后 120 帧小角度转向，最后 180 帧停留以验证异步构建和冷页回收收敛。两种配置均使用原有 streamasset。结果关注字节数，不作 GPU 性能承诺。

| 最终指标 | 原固定槽、无主动冷页回收 | 实际尺寸 + 联合冷页回收 |
|---|---:|---:|
| 几何有效驻留 | 363.5 MiB | 250.6 MiB |
| CLAS 分配占用 | 1,140.9 MiB | 256.1 MiB |
| 几何与 CLAS 合计 | 1,504.5 MiB | 506.7 MiB |
| 驻留页面 | 14,732 | 11,530 |
| CLAS cluster 数 | 207,700 | 143,127 |
| 待构建 / 容量推迟页 | 0 / 0 | 0 / 0 |

CLAS 总占用下降 77.6%，包含两项效果：冷页回收减少驻留 cluster；对新配置最终相同的 143,127 个 cluster，固定槽需 786.2 MiB，实际分配 256.1 MiB，单独的尺寸分配收益为 67.4%。几何占用下降 31.1%。

Profiler Streaming 新增 encoded、相同 cluster 固定槽估算、build/move workspace 和每帧搬移数量，继续使用可折叠的图表 scope。

## 验证

- 最终功能回归 11/11 通过：GPU 实际尺寸/两次搬移及非法重叠拒绝；紧凑池延迟发布、构建/搬移取消、重新激活、退役；联合回收与既有需求缓存/年龄限制；CLAS 低预算及卸载重载；Profiler；MiniZorah 收敛；StreamAsset RTAS 光线查询。
- 最终 2,400 帧压力漫游通过：1920×1080、1.5 px，几何池 128 MiB、CLAS 池 64 MiB，持续移动与大角度转向。累计驱逐 246,192 页次，CLAS 容量推迟 274,931 页次（包含重复尝试）；未退出，无 Vulkan validation 错误。小池压力轨迹持续产生需求，不要求积压归零。
- Release 的 Metallic 和 MetallicGPUDrivenSample 构建成功；Streaming 面板截图已检查。无需重新导出 MiniZorah。

[完整结构化对照](E:/metallic/Documentation/MiniZorahClasCompactionResults.json) · [最终功能日志](E:/metallic/build-relwithdebinfo/clas-compact/final.log) · [压力数据](E:/metallic/build-relwithdebinfo/clas-compact/final-stress/MiniZorahProfiler.json) · [Streaming 面板](E:/metallic/build-relwithdebinfo/clas-compact/final/Profiler-Streaming.png)

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
.\build-relwithdebinfo\tests\MetallicRhiTests.exe '--gtest_filter=*minizorah_profiler_streaming:*clas_compact_lifecycle:*streamer_joint_cold_reclaim:*clas_actual_sizes_and_move' --rhi-validation --rhi-async-compute --output-dir E:/metallic/build-relwithdebinfo/clas-compact/verify
```

同轨迹旧实现对照设置 `METALLIC_TEST_CLAS_LEGACY=1`；压力测试设置 `METALLIC_TEST_CLAS_ROAM_STRESS=1`，并清除 legacy/off 开关。正常验证不要设置这些开关。
