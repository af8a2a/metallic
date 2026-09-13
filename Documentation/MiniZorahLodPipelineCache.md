# MiniZorah 内部 LOD 持久化管线缓存

日期：2026-09-13。基线提交 `462efba506b1814de3fc8974d76e99d1cd1adb19`。

## 接入结果

`MeshletStreamRuntime::initialize` 新增可选的调用方缓存参数，只在初始化期间使用，不保留指针，不改变 `MeshletStreamRuntimeDesc` 的配置比较。页表初始化/更新、遍历、ActiveBuild、Cooperative LOD 共五条内部计算管线均接入；启用 CLAS 时，BLAS/TLAS 输入准备管线也传入相同缓存。不传缓存的旧调用方式继续可用。

VBuffer 将原有 `.cache/pso/VisibilityBufferPass.pso` 的创建提前到流式 runtime 初始化之前，后续 `createPipelines` 复用同一对象，编译结束统一保存。这样内部新增条目不会被后续重建缓存覆盖。独立 `GPUDrivenStreamAssetPass` 使用已有 RHI 持久化机制，在 `.cache/pso/GPUDrivenStreamAssetPass.pso` 保存内部管线，并沿用编译成功后保存、保存失败告警及析构保存的行为。独立入口的外层光栅管线未纳入本次修改。

没有修改 PSO 哈希或缓存格式。设备/backend 兼容性、SPIR-V 和管线状态变化后的失效仍由现有 RHI 处理。`LOD PSO cache enabled=… activeHit=… cooperativeHit=…` 日志用于核对实际接入，两项 hit 是 PSO 哈希命中，不等同于驱动内部反馈；下列实际创建耗时同时用于验证收益。

## 独立进程启动实测

完整 MiniZorah、1920×1080、1 GiB 页面池、1.5 px；沿用上一轮倍增上传暂存缓冲。每组运行 10 秒，追踪前 128 次离屏渲染。表中首帧指首个 `RenderGraphPreviewRenderer::render` 调用，包含建图和首次 GPU 完成等待，**不是进程启动到窗口显示的时间**。

| 运行 | 首帧 / ms | 建图 / ms | ActiveBuild PSO / ms | Cooperative LOD PSO / ms |
| --- | ---: | ---: | ---: | ---: |
| 修改前初测 | 5638.604 | 5536.971 | 1458.619 | 605.487 |
| 接入后首次填充 | 5519.396 | 5478.634 | 1417.861 | 664.946 |
| 缓存已保存，新进程 | 1910.722 | 1876.985 | 1.089 | 10.079 |
| 缓存已保存，再次复测 | 2164.222 | 2125.416 | 1.300 | 0.710 |
| 修改前反向复测 | 4868.301 | 4762.225 | 1345.230 | 541.576 |

首次填充确实仍有创建成本：外层缓存从 142 增至 147 个 PSO，24 次命中、5 次未命中，写出 9,447,492 字节驱动数据。随后两次新进程均为 **29 次命中、0 次未命中**，两条 LOD hit 均为 true。两条主要 LOD 管线合计从约 1.89～2.06 秒降至 2.01～11.17 ms。整体首帧降至 1.91～2.16 秒；实际缓存复用还覆盖另外三条内部管线，不把整体差值全部归到这两条管线。

反向对照使用相同可写执行环境及已经更新的外层缓存，但修改前的 runtime 仍不向内部管线传入缓存，首帧依然为 4.87 秒。首次初始化的 GPU 元数据准备尚需 1039.680 / 1211.140 ms，是后续主要优化对象；本次没有异步化或预打包元数据，也不承诺首次运行在空缓存下达到上述热缓存时间。

初次 `populate` / `warm` 两轮受到受限执行环境的目录写权限影响，日志报告 `cannot write temporary .pso file`，磁盘缓存保持旧版本。这两轮仅作为诊断记录，**不是已持久化后的性能数据**。没有修改目录 ACL；获得授权后在可写环境中重新执行 `populate-writable`、`warm-writable`、`warm-repeat` 和反向对照。原始失败记录及缓存前后快照均保留。

## 验证

新增 `stream_lod_pipeline_cache_persistence` 使用独立 bindless 设备和独立测试缓存，验证五条内部管线首次全部 miss，保存原生驱动数据、销毁缓存及 runtime 后全部 hit；随后不传缓存仍可初始化。缓存对象可先于 runtime 释放，runtime 不持有悬空缓存指针。既有 `pipeline_cache_persistence_and_shader_invalidation` 同时通过，覆盖管线状态/着色器变更及无效、不兼容缓存处理。

`Metallic`、`MetallicRhiTests` 构建通过。2 项缓存回归及 14 项集成回归全部通过，后者包含 VBuffer、完整 MiniZorah 的独立 StreamAsset 首帧、GPU LOD/reference cut、双帧槽、resize 与 shader reload。独立入口实测首次为 5 次 miss，重建后 5 次 hit / 0 miss；CLAS 输入管线首次新增 2 个 PSO 也通过集成路径。具体记录与哈希见 [结构化结果](MiniZorahLodPipelineCacheResult.json)。

固定初始视角 10 秒质量检查通过，末次可见超标 refinement 为 0。稳定 `roam-5.png` 与前序 `minizorah-startup-stalls/quality-fixed-0/roam-5.png` 逐字节相同，SHA-256 为 `9824bc17b155a4f4466603fd488502c53c4e742cae5f8dca8b29669ace09aaf4`。本轮不重复上一轮上传缓冲修复的 60 秒性能验收，不将初始化收益解释为稳定漫游 GPU 吞吐提升。

## 复现

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
$env:METALLIC_MINIZORAH_ROAM_SECONDS='10'
$env:METALLIC_MINIZORAH_ROAM_MIB='1024'
$env:METALLIC_MINIZORAH_PREFETCH='1'
$env:METALLIC_MINIZORAH_LOW_LATENCY='1'
$env:METALLIC_MINIZORAH_COMPLETION_UPLOADS='1'
$env:METALLIC_MINIZORAH_LATENCY_ONLY='1'
$env:METALLIC_MINIZORAH_STARTUP_TRACE_FRAMES='128'
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_roaming --rhi-validation --rhi-async-compute --output-dir build-relwithdebinfo/minizorah-lod-cache/repro
```

从能够写入项目 `.cache/pso` 的环境运行两次，第二次使用不同输出目录。先核对第一轮无保存失败、第二轮 LOD hit 均为 true，再比较 `MiniZorahStartupTrace.json` 中的 `streamInit.activePipeline`、`streamInit.cooperativePipeline` 和 `preview.compile`。已有完整缓存时，两次都可能命中；无需清空用户缓存。

测试使用已有 cook、SPIR-V 和系统文件缓存，未清空驱动内部缓存，不代表物理磁盘冷启动。全部场景测试均为无窗口离屏运行。原始输出及 PSO 快照在 `build-relwithdebinfo/minizorah-lod-cache/`，源码、二进制和日志哈希见结构化结果。前序定位见 [启动长帧](MiniZorahStartupStalls.md)。
