# MiniZorah cfg 相机回放与 profiler 对比证据

2026-09-14，RTX 5070 Ti，驱动 616.64。已完成参考程序四轮性能采样、独立截图验证，以及 Metallic 两轮性能回放和一轮 Vulkan validation / cut 验证。脚本直接调用参考程序已有的相机路径和 profiler sequencer，不需要操作桌面。

这份基准固定了输入相机、移动距离和帧步进，并保留原始 cfg 与显式对齐配置两组数据。两端的渲染任务、cook 和内存统计仍有差异，因此用分阶段数据确定优化方向，不能把整帧 GPU 时间解释成渲染器加速比。

**脚本与证据**

- [RunVkMiniZorahRoam.py](E:/metallic/Tools/RunVkMiniZorahRoam.py)：读取原始 cfg，生成相机轨迹，隔离运行参考程序，导出分阶段 GPU/CPU scope、内存、驻留/遍历计数及 GPU 监控。
- [RunMetallicCfgReplay.ps1](E:/metallic/Tools/RunMetallicCfgReplay.ps1)：让 Metallic 基准测试读取同一份 Replay.json，分开运行性能与质量验证。
- [AnalyzeMiniZorahCfgRoam.py](E:/metallic/Tools/AnalyzeMiniZorahCfgRoam.py)：校验轨迹、样本完整性，生成 CSV、Evidence.json 和对比图。
- [持久化结果摘要](E:/metallic/Documentation/MiniZorahCfgRoamResults.json)：配置、二进制/轨迹哈希、各阶段结果、scope 均值、质量检查、流送尾延迟。
- [完整原始证据](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Evidence.json)、[scope CSV](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Scopes.csv)、[内存/计数 CSV](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Memory.csv)。CSV 中时间单位为 ms；Memory 分组为 bytes，Resident/Traversal 的计数条目为数量，Cached BLAS memory 为 bytes。

本轮输出位于 `E:\metallic\build-release\vk-minizorah-roam\20260914-idle-v1` 和 `20260914-idle-metallic`。旧批次 `20260914-cfg-v1` 因参考渲染结束后整卡仍持续约 99% 占用而弃用，原因保存在其 Discarded.json 中。用户暂停其他 GPU 渲染后补采本轮；参考各轮末尾六次整卡采样回落至 0–12%。这排除了先前持续饱和的现象，但不是 GPU 完全独占或锁频证明。

**相机与配置**

原始配置为 [zorah_main_public.v2.cfg](E:/metallic/Asset/MiniZorah/zorah_main_public.v2.cfg)。起点 Eye 为 `(55.34291, 6.4527273, 0.32432523)`，Center 为 `(46.38051, 5.7994967, 0.5308178)`，FOV 为 `60.000008°`，near/far 为 `0.020000003 / 29999.998`。沿初始朝向的水平投影前进 12 个场景单位，终点 Eye 约为 `(43.346094, 6.4527273, 0.600730)`；保持高度、朝向与俯仰，然后原路返回，总路程 24 个场景单位。

| 阶段 | 帧号（从 0 开始） | 位移 |
| --- | --- | --- |
| warmup | 0–299 | 起点，剔出汇总性能均值 |
| start_hold | 300–599 | 起点停留 |
| forward_1 / 2 / 3 | 600–1499 | 0→4→8→12，每段 300 帧 |
| far_hold | 1500–1799 | 12 处停留 |
| return_1 / 2 / 3 | 1800–2699 | 12→8→4→0，每段 300 帧 |
| return_hold | 2700–2999 | 返回后停留 |

每段线性插值，关闭 spline smoothing 和 loop。固定的是每帧位移，运行不锁 60 Hz；`nominalStepSeconds=1/60` 是轨迹参数，不代表实际墙钟时长。cfg 的 cameraspeed=2 原样保留，但固定路径回放不依赖键盘漫游速度。参考程序额外运行末尾 `capture_flush` 段刷新日志，它不计入对比；截图验证每段 120 帧，端点相同，也不计入性能结果。

| 项目 | vk cfg1 / cfg2 | vk aligned1 / aligned2 | Metallic m1 / m2 |
| --- | --- | --- | --- |
| 场景 | 原始 cfg + 已有 nvsngeo | 同左 | 同源 MiniZorah stream cook |
| 相机 | cfg 相机、同一条路径 | 同左 | 同一份逐帧 Replay.json |
| 窗口/最终输出 | 1920×1080 | 1920×1080 | 1920×1080 offscreen |
| 渲染 | RT + BLAS/TLAS + DLSS-RR | RT + BLAS/TLAS，无 DLSS | VBuffer + MaterialResolve，生产 CLAS |
| 内部分辨率 | DLSS 输入 2560×1440，输出 3840×2160，再缩至窗口 | 1920×1080 | 1920×1080 |
| LOD 设置 | 默认 1.0 px | 显式 1.5 px | 1.5 render px |
| Geometry / CLAS 预算 | 默认 2048 / 2048 MiB | 显式 1024 / 512 MiB | 1024 / 512 MiB |
| RT 消费 CLAS | 是 | 是 | 否，clusterRtxEnabled=false |
| 原生场景统计 | 2,068 geometries / 16,383 instances | 同左 | 3,163 primitives / 19,144 instances |

参考程序先用 `--configfile` 解析原 cfg，再应用 headless/采样参数；aligned 仅额外覆盖 supersample、DLSS、LOD error 和两个预算。CLAS position bits=8、BLAS sharing/caching/merging、AO、阴影等 cfg/default 行为继续保留。原始 cfg 与参考 checkout 没有改写；exe、DLL、shader 和 ini 在 Metallic 输出目录内隔离运行。

参考的 `Renderer::getTraversalErrorThreshold()` 会把 LOD error 乘 window→render 比例，再除 traversal view height；所以 cfg 的 1.0 是窗口像素口径，不能直接称为 1.0 DLSS 输入像素。aligned 的缩放为 1，消除了这一分辨率口径差异，但两个 cook 的误差定义、层级组织与剔除策略仍未统一。

参考代码证据：[相机播放](E:/vk_lod_clusters/src/lodclusters.cpp:1403)、[误差换算](E:/vk_lod_clusters/src/renderer.cpp:647)、[原生 profiler 内存导出](E:/vk_lod_clusters/src/lodclusters.cpp:939)。本轮参考 HEAD 为 `1febfa7694ebdc8f97005a07897029272ca9e85a`；实际运行二进制另有 SHA-256，不把源码版本号当成二进制可复现构建证明。

**采样结果**

下表为排除 warmup 后九个等长阶段的 GPU 均值；显示两轮各自结果，不合成一个缺乏误差范围的单值。

| 配置 | 第 1 轮 GPU ms | 第 2 轮 GPU ms | 说明 |
| --- | ---: | ---: | --- |
| vk 原始 cfg | 16.359 | 16.829 | 其中 Render 8.687 / 8.931，DLSS 7.062 / 7.269 |
| vk 对齐配置 | 5.636 | 5.357 | 仍包含完整 RT、BLAS/TLAS |
| Metallic | 3.793 | 4.084 | GPU graphics envelope，包含 async join |

Metallic 对应 host frame 均值为 **7.290 / 7.439 ms**，CPU record 均值为 **3.503 / 3.326 ms**。GPU scope 时间不是实际交互帧率，也不能忽略 CPU 提交、等待和编辑器开销。两轮都有 2,999 帧在上一提交仍在飞行时进入下一次提交，未为了计时逐帧插入 waitIdle。

![分阶段 GPU 时间和内存端点](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/Comparison.png)

| 阶段 / scope | vk aligned 两轮 ms | Metallic 两轮 ms |
| --- | ---: | ---: |
| 起点 Stream Begin CPU | 0.004 / 0.004 | 0.629 / 0.675 |
| forward_2 Stream Begin CPU | 0.058 / 0.068 | 2.137 / 1.897 |
| 返回停留 Stream Begin CPU | 0.004 / 0.004 | 0.996 / 0.928 |
| 起点 Traversal Run / LOD frontier GPU | 0.118 / 0.105 | 1.068 / 1.126 |
| forward_2 Traversal Run / LOD frontier GPU | 0.111 / 0.098 | 0.600 / 0.616 |
| 远端停留 Traversal Run / LOD frontier GPU | 0.122 / 0.099 | 0.583 / 0.582 |
| forward_2 Clas Build New / CLAS build GPU | 0.057 / 0.054 | 0.113 / 0.112 |

这些 scope 是功能对应项，内部工作量并不完全相同：参考另有 Clas Prep Allocation、Clas Allocate New、Clas Append New 和 BLAS 构建，不能只用 Clas Build New 代表它的全部 CLAS 成本。Metallic 的 Stream early/Software raster 在起点约 0.857 / 0.859 ms，与 LOD frontier 一起是静止画面的主要 GPU 优化线索。嵌套 scope 与异步队列存在包含、重叠，不能把所有行相加。

**流送、驻留与质量**

| 阶段 | vk aligned1 Geometry / CLAS MiB | Metallic m1 Geometry / CLAS MiB |
| --- | ---: | ---: |
| 起点停留 | 121.92 / 127.06 | 251.97 / 255.64 |
| forward_2 末尾 | 118.23 / 113.98 | 317.29 / 324.28 |
| 远端停留 | 115.68 / 101.74 | 195.27 / 199.84 |
| 返回停留 | 122.01 / 129.56 | 251.97 / 255.64 |

Metallic 两轮完整轨迹中的 Geometry 峰值为 **325.68 / 325.76 MiB**，CLAS 峰值为 **333.67 / 333.75 MiB**，CLAS scratch 为 **156.25 MiB**。返回停留后两轮都恢复为 11,598 驻留页、143,820 CLAS clusters，pending / retiring 均为 0；CLAS 实际分配字节与 encoded 字节均为 268,052,864。两轮都累计卸载 24,784 页，上传约 1.13 GiB。该路线已覆盖真实装卸和联合回收，没有复现旧的持续累积到显存耗尽现象；它不代表任意漫游时长都不会增长。

参考 aligned1 返回时驻留 94,225 clusters，CLAS 共 129.56 MiB；Metallic 为 143,820 / 255.64 MiB。按各自驻留计数粗算，差异同时来自约 1.53 倍的 cluster 数量和约 1.29 倍的平均记账字节/cluster，不能把总量差异全部归因于分配器碎片。两个 cook、CLAS 精度、驻留策略不同，需要继续按相同可见几何覆盖量核对。

**Geometry 表中口径尤其需要区分**：参考 `getGeometrySize(false)` 是 persistentGeometry + usedDataBytes，CLAS 也是 coarsest 常驻 CLAS + usedClasBytes；参考 UI 的 Streaming 小图只使用后两项，排除了 persistent 部分。因此本表中的 122 MiB 不能直接与用户截图中小图的 14 MiB 相比。Metallic 本表记录实际页池占用，不包含独立的 LOD topology/state（本轮分别约 164.58 / 154.76 MiB）、请求缓冲、scratch 等，也不等于整进程显存。参考 MemoryReport 没有导出足够字段恢复小图的精确拆分，本次没有臆造这部分数据。来源：[参考内存汇总](E:/vk_lod_clusters/src/scene_streaming.cpp:1675)、[参考 Streaming 图](E:/vk_lod_clusters/src/lodclusters_ui.cpp:1507)。

Metallic m1/m2 最终 demand→drawable 统计的 P99 分别为 141 / 95 ms；这是其流送直方图报告的尾延迟，受 OS 缓存与调度影响。参考仅导出阶段 profiler 与端点驻留，没有对应请求时延样本，所以不能声称参考 P99 更低。下一轮应补齐相同定义的 demand、admission、upload、drawable 时间戳。

独立 quality 回放共 3,000 帧、15 个 cut 检查点，通过相机到 VBuffer 的读回检查、预算/请求/加载不变量和最终 1.5 px 收敛检查。最终 visibleOverTargetRefinements=0，最大可见 refinement error=1.499920 px；移动段帧 899/1199 分别曾有 2/1 个可见待细化组，最后停留均收敛。性能 m1/m2 各自的末尾诊断也通过。不能把“最终收敛通过”解释成漫游每一帧都已达到目标误差。

参考截图验证了端点位移和返回：[起点](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/verify/screenshot_1_start_hold.jpg)、[前进 12 单位](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/verify/screenshot_5_far_hold.jpg)、[返回](E:/metallic/build-release/vk-minizorah-roam/20260914-idle-v1/verify/screenshot_9_return_hold.jpg)。分析器逐项校验了 Metallic 的全部 3,000 个输入相机与 Replay.json 一致；参考端验证依赖原生路径实现、日志与截图，没有声称已读回其每帧 GPU 相机矩阵。

**据此安排的优化顺序**

1. 先细分 CPU Stream Begin：完成队列、需求准备、冷页扫描、CLAS 回读/回收分别计时，定位静止时仍有 0.6–1.0 ms、移动时约 2 ms 的来源；随后减少全表扫描和无变化帧处理。
2. 优化 LOD frontier 的每帧工作量，同时补 visited nodes、selected cut、unique geometry clusters 与 raster candidates 的计数对应。起点约 1.1 ms 的重复遍历比 CLAS 构建更值得优先处理；需要按实际访问量归一化，不能直接宣称比参考慢十倍。
3. 分解 Geometry/CLAS 的常驻根数据、可回收页、编码字节、对齐/碎片、待退休与 scratch，再核对 cook 覆盖量和 CLAS 精度。当前往返回收有效，主要剩余差距已不宜用“未释放”一概解释。
4. 建立完整 CLAS→BLAS→TLAS→ray consumer 后，再用相同光照、RT 特性与分辨率做端到端比较；当前 Metallic 整帧数据用于自身回归。

**复现**

在已经初始化 MSVC 构建环境的终端执行。每次选择新的输出目录；参考脚本要求已有 glTF 和 `.gltf.nvsngeo` 缓存，避免误触全场景 cook。图表依赖 Python matplotlib。

```powershell
cmake --build E:/metallic/build-release --target MetallicRhiTests --config Release
python E:/metallic/Tools/RunVkMiniZorahRoam.py --output E:/metallic/build-release/vk-minizorah-roam/repeat-v1
& E:/metallic/Tools/RunMetallicCfgReplay.ps1 -Replay E:/metallic/build-release/vk-minizorah-roam/repeat-v1/Replay.json -OutputRoot E:/metallic/build-release/vk-minizorah-roam/repeat-metallic
python E:/metallic/Tools/AnalyzeMiniZorahCfgRoam.py E:/metallic/build-release/vk-minizorah-roam/repeat-v1 --metallic E:/metallic/build-release/vk-minizorah-roam/repeat-metallic --plots
```

只生成路径而不运行 GPU 可加 `--prepare-only`；可用 `--distance` 改距离，用 `--phase-frames` 改帧数。当前 Metallic 基准固定 1920×1080，允许总帧数 600–8400；若修改参考分辨率或超出帧数范围，需要同步调整 Metallic 基准，不能继续标记为对齐结果。

参考 profiler 每阶段原始 300 帧，首段得到 287 个有效样本，后续各段 288 个；它剔除了 reset/延迟查询帧，导出整数微秒 avg/min/max/last。Metallic 保留完整每阶段 300 帧。CSV 的 gpuAvgMs/cpuAvgMs 按该端有效阶段帧数摊销，gpuRecordedAvgMs/cpuRecordedAvgMs 保留实际出现 scope 时的平均值。两端是同段均值比较，不是精确时间戳逐帧配对；参考也没有足够数据计算 P99。新进程会重建 GPU 驻留，但保留 OS 文件缓存、已有 cook 和各自持久化缓存，不能称为磁盘冷启动基准。

参考二进制五轮均在所需报告完整生成后，于退出阶段停滞。脚本在收齐并验证全部十段报告后等待 8 秒，仅回收自己启动的子进程；Profile.json 明确记录 `capture_complete`、`forcedCleanupAfterCapture=true` 与终止退出码。这些是采样完整性通过，**不是参考程序正常退出或稳定性测试通过**。实际强制回收发生在被比较的帧之后；末尾刷新段、截图验证以及清理等待均不进入性能表。
