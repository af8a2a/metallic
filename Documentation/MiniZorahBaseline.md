# MiniZorah 修复后同条件基准

2026-09-14，协议 `minizorah-fixed-v1`。运行时提交 `cd3d7076953eb1fbecebf341c0b9fc49eafe8a0f`，包含 CLAS MOVE 使用 `updateScratchSize` 的修复。本次仅新增测试和分析工具，未修改运行时算法。

**正式完成 4 次性能测试和 2 次质量测试，共 50,400 帧。** 六轮均通过，另有一次 600 帧预热。无 DeviceLost、VUID、加载失败、请求溢出或分配失败；返回初始视角后，几何与 CLAS 队列均清空。静止视角按现有 cook 误差度量收敛至 1.5 px；运动样本仍有少量超目标细化，详见质量部分。

可用于后续提交对照的[结构化结果](E:/metallic/Documentation/MiniZorahBaselineResults.json)、[逐帧数据与原始日志](E:/metallic/build-release/minizorah-baseline/20260914-fixed-v1/Manifest.json)、[完整 scope 汇总](E:/metallic/build-release/minizorah-baseline/20260914-fixed-v1/Summary.json)已保留。此基准比较 Metallic 的 A/B 配置；未运行 Nanite 或 vk_lod_clusters，也不与此前不同分辨率的截图直接比较。

## 固定条件

| 项目 | 设置 |
| --- | --- |
| GPU / 驱动 | RTX 5070 Ti 16 GB / 616.64 |
| CPU / 系统 | Ryzen 7 7800X3D，16 逻辑处理器 / Windows build 26200.9445 |
| 构建 | Release，MSVC 14.51，Ninja，`MetallicRhiTests.exe` |
| 场景 | 同一 `MiniZorah.meshstream.bin`，60,916,791,801 B，1,356,959 页，19,144 实例 |
| 渲染 | 1920 × 1080，1.5 render px；VBuffer + MaterialResolve；混合软硬光栅，异步 compute，HZB |
| 几何预算 | 1,024 MiB |
| CLAS | A 关闭；B 开启实际尺寸分配/搬移，512 MiB，最多 8,192 clusters/frame |
| 流送 | 冷页保留 120 帧；屏幕收益优先级、预取、低延迟请求、完成驱动上传均开启 |
| 测试顺序 | 600 帧 B 预热 → A1 → B1 → B2 → A2 → quality-A → quality-B；各自新进程/设备 |

A/B 启用相同 Vulkan 设备能力；归一化图配置后仅 `enableClas` 不同。六轮相机文件 SHA-256 完全相同：`0a5a4b4acafb08caf89da3e3e58f8a455781d0ad06ee0e707ef7a84a0b715d8d`。二进制及关键源文件的运行前后哈希一致，见 [IntegrityAfter.json](E:/metallic/build-release/minizorah-baseline/20260914-fixed-v1/IntegrityAfter.json)。Cook 的路径、大小和修改时间见 [HostAndAsset.json](E:/metallic/build-release/minizorah-baseline/20260914-fixed-v1/HostAndAsset.json)；未计算整个 60.9 GB 文件的哈希。

相机逐帧固定步进，不按实际耗时推进；不限制帧率。每轮共 8,400 帧，其中两次完全相同的 3,600 帧路线包含平移、转向和两次远景切换。`1/60 s` 只定义轨迹参数，不表示以真实 60 FPS 运行。

| 阶段 | 帧索引（从 0 开始） | 用途 |
| --- | --- | --- |
| cold_start | 0–299 | 新设备空驻留池的启动流送 |
| static_warm | 300–599 | 初始视角驻留后的静止段 |
| roam_first | 600–4199 | 首次路线 |
| roam_repeat | 4200–7799 | 原路线重放，保留正常回收策略 |
| settle | 7800–8099 | 回到初始视角并等待收敛 |
| static_return | 8100–8399 | 返回后的静止段 |

冷启动保留已有 cook、OS 文件缓存、shader/PSO 持久缓存；没有清空磁盘缓存。重复路线也不是强制全驻留。四轮性能回放实际用时 53.52–58.14 秒。

计时使用 `RenderGraphExecutor` 的多帧提交接口及 execution ID 对齐的 GPU timestamps，包含异步队列汇合。每轮 8,398–8,399 帧提交前仍有上一帧在途。性能阶段关闭 validation、分离调试观察器，不插入截图、完整 cut 读回或逐帧 GPU drain。调试资源仅在 compile 前登记，以便计时结束后检查；每轮末尾额外提交一个不计时的诊断帧。质量轮另开 validation，并在检查点等待及读回，其耗时不参与 A/B 表。

这是离屏渲染基准，不包含编辑器 UI、交换链呈现、Streamline 或 Aftermath。CPU host 指标包含相机更新、execute、帧槽等待和完成统计收集；GPU 指标是执行区间，不可直接换算为编辑器帧率。测试未锁频或独占桌面 GPU，四轮性能记录的核心频率中位数均为 2,910 MHz、温度中位数 59–60°C；整卡显存随其他应用变化。quality-A 还记录到整卡 98% 利用率和约 15.1 GiB 占用，因此质量轮更不能用于性能比较。两次复测只能提供初始波动范围，不能据此认定百分之一级的优化收益。

## 帧时

下表为同配置两次性能运行合并后的 **P50 / P95 / P99，单位 ms**。分位数按排序后 `floor(p × (N−1))` 取样，未删除长帧；原始数据保留每次运行的统计。

| 阶段 | A：GPU | B：GPU | A：CPU host | B：CPU host |
| --- | --- | --- | --- | --- |
| 新池启动 | 3.508 / 4.313 / 4.802 | 3.540 / 4.280 / 4.742 | 5.304 / 7.024 / 8.453 | 5.630 / 9.178 / 10.740 |
| 首次静止 | 3.613 / 4.399 / 4.929 | 3.555 / 4.188 / 4.707 | 5.470 / 6.392 / 7.453 | 5.678 / 6.566 / 7.703 |
| 首次漫游 | 3.822 / 4.696 / 5.246 | 3.884 / 4.592 / 5.034 | 7.082 / 9.339 / 10.286 | 7.459 / 9.624 / 10.950 |
| 重复漫游 | 3.695 / 4.353 / 4.721 | 3.834 / 4.589 / 5.055 | 6.264 / 8.512 / 9.684 | 6.675 / 9.317 / 11.057 |
| 返回收敛 | 3.485 / 4.379 / 4.716 | 3.519 / 4.327 / 4.776 | 5.443 / 7.030 / 8.017 | 5.606 / 7.228 / 9.840 |
| 返回静止 | 3.436 / 4.251 / 4.529 | 3.538 / 4.134 / 4.410 | 5.106 / 6.237 / 6.892 | 5.186 / 5.976 / 6.937 |

首次漫游 GPU P50：A1/A2 为 3.862/3.793 ms，B1/B2 为 3.894/3.875 ms；CPU host P50 分别为 7.117/7.031 和 7.394/7.543 ms。CLAS 开启后的 CPU host 增量比 GPU 整帧增量更明显，但其中包含流送策略带来的工作量变化，不能解释为纯驱动 build 时间。

所有性能轮的最大 CPU host 长帧均为第 0 帧，63.60–66.24 ms，其中 `Initialize / light grid` 的 CPU scope 为 42.41–45.80 ms。这与 GPU 第 0 帧的 2.92–3.93 ms 是不同问题。设备创建另耗时 131.8–140.9 ms，图编译 3.35–4.10 秒，均在逐帧计时之外。

![固定路线的 GPU 耗时与实际分配量](E:/metallic/build-release/minizorah-baseline/20260914-fixed-v1/BaselineTimeline.png)

图中为 60 帧中位数，用于观察路线趋势；P99 和峰值请读取表格/原始逐帧数据。

## 分项热点

以下为两次运行、两段漫游共 14,400 帧的均值。scope 保留父子路径与 queue，**嵌套区间和并行软硬光栅耗时不可相加作为整帧耗时**。

| scope | A GPU ms | B GPU ms | 补充 |
| --- | ---: | ---: | --- |
| LOD frontier | 0.885 | 0.890 | 当前主要遍历成本 |
| Prefetch | 0.285 | 0.285 | 接近 early 分类成本 |
| Stream early / 分类 | 0.237 | 0.237 | 存活 cluster 的软硬分类 |
| Stream early / 软件光栅 | 0.651 | 0.654 | compute queue |
| Stream early / 硬件光栅 | 0.458 | 0.455 | graphics queue，与软件光栅有重叠 |
| CLAS build 所在 scope | — | 0.098 | B 的 CPU 均值另为 0.140 ms；仅代表此标记范围 |
| MaterialResolve | 0.079 | 0.078 | 不是本路线主要 GPU 热点 |

`Stream Begin` 的 CPU 均值 A/B 为 **1.605 / 1.808 ms**，GPU 均值接近零。该范围覆盖 `cmdBeginFrame` 的 CPU 流送处理；不能把其 CPU 成本归入 GPU build，也不能用当前粗 scope 区分请求整理、回收、完成处理等子项。后续可先细分此范围，再决定实现改动。

## 内存、流送和 CLAS

均为 MiB。实际分配、池容量、构建临时区及进程显存分别列出。

| 指标 | A | B |
| --- | ---: | ---: |
| 几何实际分配峰值 | 442.41 | 431.78 |
| 返回后几何实际分配 | 251.97 | 251.97 |
| CLAS 实际分配峰值 | 0 | 444.41 |
| 返回后 CLAS 实际分配 / 编码尺寸 | 0 | 255.64 / 255.64 |
| 几何池容量 | 1,024 | 1,024 |
| CLAS 持久池容量 | 0 | 512 |
| CLAS scratch / 构建临时区统计 | 0 | 156.25 |
| 回放结束时进程 DXGI local usage | 2,142.9 | 2,884.8 |

CLAS 的 156.25 MiB 是当前统计口径的多批次临时构建存储、scratch 和辅助缓冲，不是单个 MOVE scratch。持久池仍整块预分配；空闲容量仍有资源成本。DXGI 进程值包含测试进程中的设备、图资源、拓扑和分配器，不能当作 Geometry + CLAS 之和；整卡 `nvidia-smi` 值还包含其他进程。

返回后两种配置均驻留 **11,598 页**；B 已发布 **143,820 个 CLAS**，pending/retiring 均为零。全程没有 CLAS 分配拒绝，B 的待构建页数峰值为 512。

| 全路线流送指标 | A1 / A2 | B1 / B2 |
| --- | ---: | ---: |
| 累计上传 MiB | 4,853.74 / 4,853.27 | 4,945.14 / 4,944.64 |
| 回收次数 | 138,512 / 138,495 | 140,836 / 140,832 |
| 重复路线上传 MiB | 2,270.00 / 2,269.57 | 2,315.79 / 2,315.37 |
| CLAS 构建数量（累计，含重建） | 0 | 2,656,744 / 2,656,313 |
| CLAS 搬移数量（累计，含重复） | 0 | 2,656,744 / 2,656,313 |
| demand → drawable P95 / P99 ms | 48 / 68；45 / 65 | 61 / 88；61 / 94 |

延迟为运行时累计直方图的近似分位数，drawable 表示几何可用于 VBuffer，不表示 CLAS 已可供光追使用。当前 VBuffer 样例的 `clusterRtxEnabled=false`，这次只测 CLAS 生产/回收成本，没有 RT 渲染收益。

B 的几何峰值略低、累计上传/回收略高，与 CLAS 预算触发联合回收的设计一致：当前策略在占用达到 85% 时进入压力回收，目标降到 70%。本路线上的重复上传是后续保留策略/页面复用优化应跟踪的量；由于路线包含远景切换和正常 120 帧到期回收，不能仅凭累计计数判定所有重载都是错误。

## 质量和稳定性

quality-A/B 各检查 33 个快照：第 29、59、119、179 帧，之后每 300 帧，及末尾诊断帧。每个快照验证 DAG cut 完整覆盖、共享父级一致性、无重复/不可达 group、候选数量约束和相机确实到达 VBuffer。未出现 active/candidate 容量回退。

- 第 29 帧仍在加载，A/B 可见超目标细化数为 2,665/2,756；第 59 帧的采样已为零。这里只能说“到第 59 帧检查点已收敛”，不是精确的首次达标时间或真实 1 秒内达标承诺。
- 第 1,199 和 4,799 帧，A/B 各存在 1 个近面误差无界的细化节点。`FLT_MAX` 是误差度量的哨兵，不表示测得对应数量的像素偏差。
- 第 2,999 和 6,599 帧，A 有 1 个超目标细化，最大约 1.50154 px；B 有 3 个，最大约 1.61656 px。两次路线重复出现，作为后续请求/收敛优化的固定检查点保留。
- 最终 A/B 均为 22,891 个 active groups、248,855 个 cut clusters，可见超目标细化为零，最大可见细化误差约 1.49992 px。所有性能轮的末尾诊断也通过。

这些检查基于 cook 的误差包围体和参考 cut 规则，不是与全精度真值图像逐像素比较，也没有逐帧穷举整个路线的质量。仍需将“加载瞬态完整回退”与“每时刻满足 1.5 px”区分开。

## 复测

在配置好 MSVC 的 Developer PowerShell 中执行：

```powershell
Set-Location E:\metallic
cmake -S . -B build-release -DMETALLIC_BUILD_TESTS=ON
cmake --build build-release --target MetallicRhiTests -j 6
& .\Tools\RunMiniZorahBaseline.ps1 -OutputRoot build-release/minizorah-baseline/next-change
python .\Tools\AnalyzeMiniZorahBaseline.py build-release/minizorah-baseline/next-change --plots
```

[测试入口](E:/metallic/tests/rhi/MiniZorahRoamingTests.cpp:612)、[运行脚本](E:/metallic/Tools/RunMiniZorahBaseline.ps1)、[分析脚本](E:/metallic/Tools/AnalyzeMiniZorahBaseline.py)。默认测试保持 opt-in；脚本为每次实验建立新目录，拒绝覆盖既有结果。

后续改动至少对照：首次/重复漫游 CPU host 与 GPU P50/P95/P99、首帧最大值、geometry/CLAS 峰值与返回值、累计上传和回收、demand → drawable 尾延迟，以及上述固定质量检查点。优先进一步拆分 CPU `Stream Begin` 和首帧 light grid 初始化，同时用 LOD frontier 与重复路线重载量衡量优化收益。
