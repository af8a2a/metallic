# MiniZorah 分类、光栅与页面收益调度

日期：2026-09-13。完整资产与 cook 保持不变，继续使用 `gpu-driven-minizorah-vbuffer`。本轮实现并行 mesh 输出、early/late 候选过滤和屏幕收益页面排序；1.5 px 质量收敛仍开放。

## 60 秒实测

RTX 5070 Ti、驱动 616.64、RelWithDebInfo、1920×1080、1.5 px、异步 HW/SW、Vulkan validation。相机通过共享 `RenderView` 驱动，与编辑器视口使用同一链路。计时是同步离屏 GPUDriven + MaterialResolve，排除呈现 blit、计时帧的图像/调试回读及检查点验证，不代表编辑器呈现 FPS。

| 1 GiB 页面预算 | 本轮基线 | 仅分类/光栅改动 | 最终版本（含收益排序） |
| --- | ---: | ---: | ---: |
| GPU P50 / P95 / P99（ms） | 17.44 / 21.82 / 22.86 | 10.75 / 12.82 / 13.15 | 10.99 / 12.98 / 13.45 |
| 同步帧 P95 / P99（ms） | 27.30 / 28.88 | 18.21 / 19.55 | 19.20 / 21.37 |
| CPU 记录 P95（ms） | 6.19 | 5.85 | 6.67 |
| 计时帧数 | 3,572 | 4,573 | 4,441 |
| 55 秒检查点累计上传（GB，十进制） | 3.225 | 3.321 | 3.913 |
| 55 秒检查点累计淘汰页 | 57,406 | 59,781 | 73,160 |

最终版本相对本轮基线的 GPU P95 降低 40.5%，同步帧 P95 降低 29.7%。收益排序在 1 GiB 档没有进一步降低整体帧时：相对仅光栅改动，CPU P95 增加约 0.83 ms、上传增加 17.8%。15 秒近景的超标 refinement 记录由 2,037 变为 1,478，近裁面无界记录由 1,422 变为 878，但这些数量依赖具体 cut，不能当成已达到 1.5 px 的证明。

路线按渲染墙钟推进，异步页到达、实际 cut 和帧数会不同。以上是同配置完整工作负载的单次对照，不是锁定同一 cut 的纯 kernel 微基准。第一版排序草稿保存在 `priority/`；其中较低的 GPU P95 不作为最终版本成绩。最终版本将准入排序键预先计算，CPU 记录最大值从该草稿的 158.18 ms 降为 17.93 ms。

## 成本拆分和实现

新增 candidates / classify / stable bins / raster join / resolve 检查点，区分此前合在 “bins” 区间内的工作。15 秒近景：

| 阶段（ms） | 基线 | 仅分类/光栅改动 | 最终版本 |
| --- | ---: | ---: | ---: |
| frontier | 1.422 | 1.426 | 1.442 |
| early candidates | 0.663 | 0.777 | 0.782 |
| early classify | 4.605 | 4.954 | 5.020 |
| early stable bins | 0.039 | 0.025 | 0.045 |
| early raster join | 7.463 | 1.468 | 1.223 |
| late candidates | 0.626 | 0.866 | 1.106 |
| late classify | 4.638 | 1.206 | 1.031 |
| late raster join | 0.461 | 0.277 | 0.108 |

光栅区间在 HW/SW 汇合处结束，不能将它解释为两个重叠队列各自的独立耗时。稳定前缀/散射本身只有几十微秒；early 分类和候选准备是下一步主要目标。

`gpuDrivenStreamAssetMeshMain` 从单线程逐三角形输出改为每工作组 64 线程。lane 0 加载一次 cluster，组内共享，线程并行输出原来的 64 三角形 chunk；两个 chunk/cluster、顶点格式、几何 ID 和等深绘制顺序保持一致。

early 候选只展开可见实例。early 分类将上一帧 HZB 拒绝的 cluster 写入每 active group 的重试 mask；late 只展开这些 cluster 和本帧恢复的实例。当前帧视锥、锥剔除失败不进入重试列表；每帧 early 准备清理 mask。中间的 resident producer 不写这块缓存，保留混合 producer 的顺序与原 record 命名空间。15 秒检查点的 late 候选从完整 cut 缩到约 6.6 万。

## 页面收益规则

协作 frontier 为缺失页面计算以下启发式，使用与 LOD 选择一致的剔除相机：

1. 将 group 的包围球和几何误差变换到世界空间，以投影球面积估计屏幕覆盖，限制在视口面积内。视锥外依赖的收益为零，仍保留请求。
2. 收益为覆盖面积乘误差超额权重；权重 `clamp(projectedError / target - 1, 1, 64)`，总收益上限 `1e9`，避免近裁面无界误差垄断浮点范围。该估计不使用 HZB 可见像素数。
3. 同一页面跨实例以 `InterlockedMax` 合并正浮点收益，执行在请求去重之外，避免第一个请求者决定整个共享页的优先级。
4. CPU 按收益 / 解压后页面字节排序准入；已跟踪请求的等待年龄带来有界 1–5 倍权重。I/O 和已经准备好的上传继续根页优先、近期需求优先，再比较收益和年龄。已提交的 I/O / 上传保留原完成生命周期。

年龄只作用于已跟踪请求，零收益依赖没有无限等待保证。本轮没有实现祖先页面撤销、跨页依赖收益传播、精确可见像素收益或压缩格式变化。完整 terminal cut、共享父组关系和 PendingUpload 保护继续由原有选择/驻留逻辑保证。

请求头保持 64 B，使用两个原 padding 字段描述新增区域：页 ID、紧凑 float 收益列表、GPU 专用逐页 max 表。每帧 GPU 清零 max 表，帧末按请求 ID 收集收益，只回读紧凑前缀。MiniZorah 增加约 5.18 MiB 的 GPU 逐页表和 256 KiB 的收益列表；每帧回读额外 256 KiB。CPU `PageEntry` 利用原 padding，仍为 48 B；请求任务另带预计算排序键。early/late mask 在当前 4,194,304 cluster 容量下另占 16 MiB/光栅器。

两个流式入口默认开启 `screenSpacePagePriority`；设置为 `false` 可对照旧请求顺序。零偏移的旧请求头继续有效。调试绑定增加 `loadPriorities`，快照显示开关与请求/回读缓冲大小；漫游检查点验证收益为有限非负数。

## 64 MiB 压力对照

使用同一最终二进制，仅切换收益开关：

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| GPU P95（ms） | 13.51 | 5.38 |
| 同步帧 P95（ms） | 19.89 | 8.90 |
| CPU 记录 P95（ms） | 4.29 | 3.81 |
| 55 秒累计上传（MB，十进制） | 261.57 | 211.08 |
| 累计淘汰页 | 13,901 | 12,580 |
| 15 秒细组数 | 6,001 | 6,947 |
| 15 秒超标 refinement 记录 | 2,521 | 3,049 |

收益排序在此压力档减少了装卸并装入更多细组，但超标记录仍多，不能以低帧时替代质量验收。开启时 15 秒的 666 个请求中有 635 个获得正屏幕收益。

![64 MiB 收益排序近景](../build-relwithdebinfo/minizorah-screen-benefit/pressure-on/roam-15.png)

## 660 秒循环漫游

最终版本完成 660.008 秒、49,156 个计时帧和全部 132 个检查点。GPU P95 / P99 为 12.92 / 13.31 ms，同步帧 P95 / P99 为 18.38 / 20.26 ms，CPU 记录 P95 为 6.09 ms。无无效页请求、页面加载失败、Vulkan validation error 或 device loss。

暖机周期（120–180 秒）进程本地显存峰值为 2,424,102,912 B，末轮（600–660 秒）为 2,422,333,440 B，未持续增长；页面池始终不超过 1 GiB。655 秒检查点累计上传 34.677 GB、淘汰 863,106 页，因此本轮也没有消除 1 GiB 下的工作集切换成本。

## 验证与复现

`Metallic`、`MetallicGPUDrivenSample`、`MetallicRhiTests` 的 RelWithDebInfo 构建成功。最终 34 项回归全部通过（253.289 秒），覆盖流式准入/上传/退避、219 组 GPU/reference cut、收益清零/收集、混合光栅、GPUScene、完整 MiniZorah VBuffer、resize/reload 和共享相机漫游；日志无 VUID 或 Vulkan validation error。已有 pipeline cache 写入警告仍可见，shader 编译不计入计时路线。

新驻留测试验证共享页面收益取最大值、每字节收益、根页保护、NaN 和短反馈；GPU cut 测试以 NaN 污染逐页表，验证每帧清零、紧凑收集、未请求页面无残留和可见请求产生正收益。两档 60 秒路线均通过所有 12 个完整 cut、覆盖、相机和页池检查点。

```powershell
$env:METALLIC_TEST_MINIZORAH = '1'
$env:METALLIC_MINIZORAH_ROAM_SECONDS = '60' # 长测为 660
$env:METALLIC_MINIZORAH_ROAM_MIB = '1024'   # 压力对照为 64
$env:METALLIC_MINIZORAH_PAGE_PRIORITY = '1' # 旧顺序对照为 0
& .\build-relwithdebinfo\tests\MetallicRhiTests.exe `
  --gtest_filter=RhiRendering.minizorah_roaming --rhi-validation --rhi-async-compute `
  --output-dir E:/metallic/build-relwithdebinfo/minizorah-screen-benefit/reproduce
```

[结构化结果](MiniZorahScreenBenefitResult.json)、[最终回归日志](../build-relwithdebinfo/minizorah-screen-benefit/final-tests.log)、[最终 60 秒报告](../build-relwithdebinfo/minizorah-screen-benefit/final/MiniZorahRoamingReport.json)、[660 秒报告](../build-relwithdebinfo/minizorah-screen-benefit/long/MiniZorahRoamingReport.json)。

下一步优先减少 early 分类重复加载/投影和候选准备串行扫描；质量侧测量每次细化的可见收益和 request→drawable 延迟，再决定层级需求可见性、依赖收益传播与驻留压缩的投入顺序。继续同时约束画面误差、CPU 成本与装卸量。
