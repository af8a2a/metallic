# MiniZorah 启动长帧定位

日期：2026-09-13。基线提交 `48298747a2ef2b5a2e109b77187c21141d86becc`，完整 MiniZorah、1920×1080、1 GiB 页面池、1.5 px。

后续已接通本文发现的内部 LOD 持久化管线缓存缺口：命中缓存后的首个离屏渲染调用降至 1.91～2.16 秒，两条 LOD 管线创建降至合计 2.01～11.17 ms。见 [内部 LOD 缓存接入](MiniZorahLodPipelineCache.md)；以下保留当时的定位与扩容对照数据。

## 结论

冷启动细节加载期间，Streamer 按 64 KiB 对齐、仅扩到当次请求所需容量。一个上传批次中每跨过容量边界便新建整块暂存缓冲，旧缓冲必须留到 GPU 拷贝完成。因此几十页上传会形成几十块几乎同样大的缓冲，造成提交后、GPU 首个时间戳前的长等待，以及两帧后集中释放资源的尖峰。

已改为倍增扩容，保留旧拷贝地址及原完成点生命周期。64 KiB 仍用于对齐，首次容量按调用者配置分配；单次大请求直接满足所需大小，倍增及排队帧数乘法有溢出检查。不会改变页面池预算、页面优先级、完成发布条件或几何误差阈值。代价是最终暂存容量可能比刚好够用的容量更大，后续小请求会复用它；暂存缓冲不计入几何页面池。

## 直接证据

追踪记录前 128 次 preview render，覆盖根页建立及冷启动细节请求，运行顺序为线性 → 倍增 → 倍增复测 → 线性反向复测。

| 指标 | 线性两轮 | 倍增两轮 |
| --- | ---: | ---: |
| 暂存分配次数 | 119 / 117 | 3 / 3 |
| 累计新建缓冲容量 / MiB | 1829.625 / 1784.250 | 56 / 56 |
| 追踪范围内最大计时帧 / ms | 199.066 / 141.984 | 32.741 / 35.137 |
| 最大资源释放阶段 / ms | 10.901 / 16.150 | 0.012 / 0.013 |

累计容量是每次新建缓冲的完整容量之和，包含两个帧槽，**不是同时驻留峰值、实际上传字节或页面池使用量**。倍增的三块缓冲为 8、16、32 MiB。第一对 GPU 负载存在变化，不能把 GPU 执行耗时下降全部算成扩容策略收益；反向复测仍复现相同的提交后长等待及大量分配。

线性第一轮第 30 帧的时间分解：

| 阶段 | 时间 / ms |
| --- | ---: |
| 完整同步帧 | 199.066 |
| CPU execute | 10.996 |
| 其中实际 queue submit 调用 | 0.323 |
| submit 全部返回 → GPU 帧起点 | 169.136 |
| GPU 帧起点 → 终点 | 18.353 |
| GPU 终点 → CPU wait 返回 | 0.321 |

此帧发生 21 次扩容，容量合计 259.875 MiB；第 33 帧扩容 27 次、486 MiB，又产生 127.528 ms 的 GPU 开始前延迟。反向对照第 35 帧扩容 36 次、645.750 MiB，对应 114.383 ms 的 GPU 开始前延迟。等待热点在 `RenderGraphExecutor::waitForSubmittedWork` → `GpuCompletionPoint::wait` → `vkWaitSemaphores`，其等待时间不等于 GPU shader 执行时间。

通过 `Queue::calibrateTimestamps` 将 GPU 时钟映射到 CPU steady clock。长帧的映射不确定度约 0.008～0.013 ms；GPU 完成后 CPU 返回约 0.17～0.44 ms，足以排除“GPU 已完成但 CPU 晚醒上百毫秒”作为这些帧的主因。GPU 起点在图的初始 view 准备后，因此该间隔也包含首个时间戳前的少量 GPU 设置，不能细分为纯 OS 排队时间。

源码、时钟对齐与扩容策略的干预实验共同确认了暂存分配放大的影响。驱动/WDDM 内部的驻留或分页操作仍是解释，尚无内核级事件证明其具体分项。尝试的 Nsight Graphics 2026.3.1 离屏 GPU Trace 没有生成有效捕获；未用空捕获作为证据。

## 关闭追踪的 60 秒路线

两个进程均启用 GPU 完成后发布、同帧准入和预取，只比较暂存扩容策略。根页准备和首次建图不计入下面的漫游帧分位数，过程中不做输出 readback，59 秒检查点单独验收。

| 指标 | 线性 | 倍增，已核对二进制 |
| --- | ---: | ---: |
| 最大同步帧 / ms | 237.636 | 30.998 |
| 同步帧 P95 / ms | 22.651 | 13.278 |
| 实际请求到可绘制 P95 / ms | 255 | 101 |
| 实际请求到可绘制 P99 / ms | 261 | 132 |
| 实际请求到可绘制平均 / ms | 93.491 | 46.845 |
| 实际请求完成数 | 8799 | 8696 |
| 请求 P95 / P99 / 帧 | 10 / 13 | 10 / 13 |
| 页面池末次使用 / MiB | 670.908 | 671.352 |
| 累计上传 / MB，十进制 | 700.668 | 701.131 |

两组实际需求未完成数、丢弃数、无效 GPU 请求、加载失败及意外取消均为 0，末次可见超标 refinement 为 0。请求样本构成随帧时及预取归类变化，不能当作逐页完全配对的数据。最终工作集相差约 0.07%。GPU P95 为 16.585 / 9.433 ms，运行环境并未独占 GPU，不据此认定纯光栅吞吐收益；提交前后定位与分配次数的变化是本轮主要证据。

审计时发现一个构建陷阱：恢复实验源码保留了文件时间戳，Ninja 复用了线性策略的 `Streamer.cpp.obj`。目录 `latency-geometric-60` 和 `final-startup` 实际仍是线性实现，**不作为倍增结果**。原始数据保留，并在结构化结果中标注。强制更新源码时间戳、确认 Streamer 重新编译，再通过 `frame_upload_growth_burst` 后，使用 `latency-geometric-verified-60` 完成上表复测；`final-startup-verified` 再次确认仅 3 次扩容。

## 首次建图仍有独立成本

最终版本首个 preview render 为 5.013 秒，其中 `preview.compile` 为 4.933 秒，首次 execute 为 75.739 ms。扩容优化不解决首次建图成本。

| 初始化阶段 | 时间 / ms |
| --- | ---: |
| 打开并核对 cook 资产 | 196.236 |
| 场景 GPU 元数据创建与填充 | 1057.089 |
| ActiveBuild 计算管线创建 | 1300.423 |
| Cooperative LOD 计算管线创建 | 549.065 |
| 两条管线的 shader 模块准备合计 | 49.214 |
| 1 GiB 几何页面池创建 | 2.835 |

管线创建仍耗时，即使日志显示 SPIR-V cache 命中。源码还确认了一个接入缺口：`MeshletStreamRuntime::ActiveBuildPass::initialize` 的这两条计算管线没有传入 `ComputePipelineDesc::pipelineCache`，Vulkan 后端因此将 `VK_NULL_HANDLE` 传给 `vkCreateComputePipelines`；外层 VisibilityBufferPass 已有持久化管线缓存能力，但这两条内部管线尚未接入。下一步优先接通现有缓存、测量命中/创建耗时并安排预热，同时优化元数据准备。元数据阶段包含 CPU 构建、HostUpload 缓冲创建及填充，需要进一步分开测量后决定是否随 cook 预打包。随后再考虑可复用分块上传 arena 或初始化预留，压低倍增后偶发的 20～30 ms 首次使用尖峰。当前没有把“首次建图 5 秒”或所有设备上的最大帧视为已解决。

## 验证与复现

新增 `frame_upload_growth_burst` 用 64 个 20 KiB 页面验证分配放大上限，并在第 1 个帧槽中跨多次扩容记录拷贝，延迟 GPU 完成、推进 CPU 帧、销毁 Streamer 后再逐字节核对输出；同时验证容量溢出拒绝。该回归在线性对照下失败，倍增版本通过；既有 `frame_upload_lifetime` 也通过。

- `Metallic` 和 `MetallicRhiTests` 构建通过；64 项相关 RHI 回归全部通过，包含 Streamer、帧/提交生命周期、GPU 时钟校准、RenderGraph GPU profiling、多队列、混合光栅和完整 MiniZorah VBuffer。
- 固定初始视角 10 秒质量检查：首个可见超标归零点为累计渲染 0.511 秒，末次最大可见 refinement 误差 1.499921 px。稳定 `roam-5.png` 与前序 `minizorah-cold-start/fixed-0/roam-5.png` 完全一致，SHA-256 为 `9824bc17b155a4f4466603fd488502c53c4e742cae5f8dca8b29669ace09aaf4`。
- 64 MiB、10 秒压力检查：末次使用 62.79 MiB，驱逐 718 页，无无效请求、加载失败或意外取消；完整回退通过。末次仍有 3012 个可见超标 refinement，**仅验收预算及回退，不算 1.5 px 质量收敛**。

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
$env:METALLIC_MINIZORAH_ROAM_SECONDS='10'
$env:METALLIC_MINIZORAH_ROAM_MIB='1024'
$env:METALLIC_MINIZORAH_PREFETCH='1'
$env:METALLIC_MINIZORAH_LOW_LATENCY='1'
$env:METALLIC_MINIZORAH_COMPLETION_UPLOADS='1'
$env:METALLIC_MINIZORAH_LATENCY_ONLY='1'
$env:METALLIC_MINIZORAH_STARTUP_TRACE_FRAMES='128'
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_roaming --rhi-validation --rhi-async-compute --output-dir build-relwithdebinfo/minizorah-startup-stalls/repro
```

`MiniZorahStartupTrace.json` 包含 CPU 阶段、扩容字节、帧与 execution ID、GPU 时间跨度及校准不确定度。最大追踪 1024 帧 / 65536 个 CPU 事件，溢出计入 `droppedEvents`；本轮 128 帧结果均无事件丢失。默认关闭，不读取时钟或写日志；启用时在渲染线程内存中收集，检查点/结束时输出文件。关闭 `STARTUP_TRACE_FRAMES`、时长设为 60 可复测无追踪路线。

测试使用既有 cook、shader 缓存和系统文件缓存，冷启动指新进程及 GPU 工作集重新建立，不代表物理磁盘冷缓存。测试为无窗口同步离屏图，不包含编辑器输入、展示拷贝和 swapchain；定位涉及的 Streamer、帧生命周期及 RenderGraph 提交路径由编辑器共享。

原始输出在 `build-relwithdebinfo/minizorah-startup-stalls/`。数据与哈希见 [结构化结果](MiniZorahStartupStallsResult.json)，前序问题见 [冷启动尾部](MiniZorahColdStart.md)。
