# VBuffer GPU 标记归属修正

2026-09-24。

原有 `RenderGraphPass: VBuffer (VisibilityBufferPass)` 包含场景 streaming、LOD 遍历和加速结构构建，但只有部分 Hybrid raster 命令手写了 Nsight labels。应用 `profileScope()` 的 GPU timestamp 层级没有导出为 debug labels，导致父标记前半段存在大量 GPU 工作，却没有对应的子标记。

现在 `profileScope()` 同时生成嵌套的 Vulkan debug labels，使用与应用 profiler 相同的阶段名称。另增加 `Visibility prepare`、`Visibility raster` 和 `Upload flush`，将纯 visibility 执行与节点公共工作分开。父标记继续保留整个节点的范围，不通过缩短父范围隐藏真实开销。

```text
RenderGraphPass: VBuffer (VisibilityBufferPass)
├─ Streamer prepare
│  └─ Stream Begin / Upload preflight …
├─ Visibility prepare
├─ Stream traversal
│  ├─ Page updates / Detail demand
│  ├─ LOD frontier / mask / prefix / emit
│  ├─ Prefetch / CLAS build
│  └─ BLAS … / TLAS build
├─ Visibility raster
│  ├─ Initialize / light grid / Resident LOD
│  ├─ Early instance cull
│  ├─ Stream early
│  │  ├─ Candidates / Cluster cull / Soft/hard classification / Stable bins
│  │  └─ Software raster / Hardware raster / Raster merge
│  ├─ Early HZB / Late instance cull
│  ├─ Stream late …
│  └─ Late HZB / Debug composite
├─ Stream End
└─ Upload flush
```

这是逻辑层级，实际可用阶段取决于场景和路径。旧的、更底层的 `Hybrid raster: …` 标记保留在相应子阶段内。通过 `publishCpuProfile()` 发布的 CPU-only 阶段不会伪装成 GPU labels 或 GPU 时间。

异步 raster 会将一个 pass 拆成 graphics producer、compute branch、graphics branch 和 graphics join。每个 command buffer 都有独立且配对的 label stack，恢复相同的父层级以便识别归属；分支失败时只清理逻辑 scope，不向已经结束的 command buffer 写 label/timestamp。应用原有 graphics 起点到 join 的 timestamp 仍表示 elapsed envelope。跨队列的同名 marker 可能重叠，不能把它们直接相加当成总耗时。

涉及代码：

- [标记生成及分支恢复](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp)
- [每个 recording 的标记栈](../Source/Runtime/Render/RenderGraph/RenderGraphGpuLabels.h)
- [纯 visibility 范围](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)
- [标记协议与 scope budget 回归](../tests/rhi/GpuProfilingTests.cpp)
- [MiniZorah 实际 GPU 队列和 scope 层级验收](../tests/rhi/EditorProfilerTests.cpp)

验收覆盖标记嵌套与配对、分支失败、query budget、取消提交、跨队列依赖，以及 MiniZorah 360 帧漫游。场景测试前 180 帧串行，后 180 帧 early/late 均启用 async raster，逐帧检查 GPU 时间可用性、父子层级和 software raster 的实际队列。

结果：Release 构建通过；7 项 profiling/分支回归通过，跨队列依赖测试通过，MiniZorah 漫游通过。360 个串行 software raster scope 均归属 graphics queue，360 个异步 scope 均归属 compute queue（每帧 early/late 各一个）。另在 validation 开启下运行标记协议、scope budget、取消计时、异步分支四项测试，全部通过，但存在下述设备初始化警告。

- [7 项回归日志](../build/nsight-labels-tests-final.log)
- [validation 子集日志](../build/nsight-labels-validation.log)
- [MiniZorah 完整验收 JSON](../build/nsight-labels-minizorah/MiniZorahProfiler.json)
- [MiniZorah 验收日志](../build/nsight-labels-minizorah-final.log)

复现：

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
& build-release/tests/MetallicRhiTests.exe --rhi-no-validation '--gtest_filter=*gpu_profiling*:*frame_parallel_compute_join_and_cancellation:*frame_cross_queue_graph_dependencies:*editor_profiler_history:*editor_profiler_capture_attribution:*minizorah_profiler_streaming' --output-dir build/nsight-labels-acceptance
```

本机旧 Vulkan validation layer 不认识 `VK_KHR_device_address_commands`，完整渲染在 validation 开启时发生异常，因此完整 GPU 场景使用 `--rhi-no-validation`。这不是 validation-clean 验收，也没有取得新的 Nsight GPU Trace。标记协议测试直接检查提交给 debug label 接口的名称层级和配对关系；MiniZorah 验证实际 GPU 执行及 timestamp 归属。

运行新构建的 `build-release/Source/MetallicGPUDrivenSample.exe` 后重新采集，展开 Nsight 的 Markers 层级即可使用新标记；旧 capture 保存的是旧命令流，不会随源码更新。
