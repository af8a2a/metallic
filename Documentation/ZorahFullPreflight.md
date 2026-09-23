# ZorahFull Preflight 细分（2026-09-23）

## 实现

保留 Preflight 总计时，增加七个 CPU 子 scope：Validate queue contracts、Append incoming waits、Poll / retire external completions、Copy external dependencies、Read scene revisions、Check frame overlap contracts、Collect overlap blockers。没有增加 GPU 查询、改变同步、跳过检查或修改驻留策略。

EditorProfiler 现在按 preparation 的 parent 索引构建树；子 scope 可在 Preflight 下折叠，Full 捕获导出保留相同层级。原有顶层 Graph preparation / Preflight 路径保持不变。

## 实测归属

三轮均为固定 Full 路线，1797×660 输出、DLSS Quality 1198×440、LOD 1.5 px，预热 10 秒、采样 30 秒。原始输出：build-release/full-preflight-0923/run1..3。

| CPU scope 平均 ms | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|
| Preflight | 8.9263 | 8.4405 | 7.0351 |
| Validate queue contracts | 0.0011 | 0.0009 | 0.0017 |
| Append incoming waits | 0.0001 | 0.0001 | 0.0001 |
| Poll / retire external completions | 0.0018 | 0.0016 | 0.0017 |
| Copy external dependencies | 0.0002 | 0.0002 | 0.0003 |
| Read scene revisions | 0.0002 | 0.0001 | 0.0002 |
| Check frame overlap contracts | 5.0822 | 4.7331 | 3.9263 |
| Collect overlap blockers | 3.8306 | 3.6999 | 3.1010 |

两遍帧重叠检查占 Preflight 的 **99.85% / 99.91% / 99.89%**。完成点查询与清理约 0.0016–0.0018 ms，场景版本读取和依赖复制也很小。三轮共 1608 帧，全部七个子计时均存在、子项合计不超过父项、GPU 计时无缺失，drainReasonMask 均为 0。

采样结束后观察到 7zFM 约 1207% 的多核 CPU 占用，存在明显后台 CPU 负载。本轮整帧均值 62.98 / 54.06 / 52.27 ms 不能与上轮作为受控性能对比，也不能认定细分计时导致这些变化。上述毫秒数是本轮负载条件下的墙钟耗时；稳定重复出现的开销归属可用于决定下一步。

## 代码对应与下一步

1. RenderGraphExecutor::execute 先用 any_of 调用 supportsFrameOverlap()，随后为收集 overlapBlockingPasses 再调用一遍；Full 没有阻塞 pass，因而两遍都遍历完整执行列表。
2. VisibilityBufferPass::supportsFrameOverlap 调用 MeshletStreamRuntime::sceneReadiness。该函数逐个遍历 lockedFallbackPages，查询 residency_.pageResident 和 clasPool_->pageHasClas，并检查 fallbackBlasPrimitives 的 built 状态。其他当前 Full pass 的重叠判断主要是布尔值或属性读取。结合两个子 scope 的耗时，重复的就绪遍历是主要优化对象；当前计时没有继续区分该函数内部几何、CLAS 和 BLAS 子项。
3. 第一阶段：每帧只评估一次各 pass 的 overlap 契约，同时得到 drain 标志和阻塞名称，消除第二遍完整扫描。
4. 第二阶段：把完整的进度统计与布尔 ready 查询分开。按资源生命周期维护轻量就绪状态；在初始化、重建、取消和资源失效时清除，在根几何、根 CLAS 与 fallback BLAS 均完成后发布。不能让旧场景的 ready 状态跨代复用。
5. 后续在 CPU/GPU 空闲条件下重新跑同路线，以单项计时和整帧慢帧比例验收。无需优先优化 timeline 查询或依赖容器。

## 验证

- Release MetallicGPUDrivenSample 与 MetallicRhiTests 构建成功。
- editor_profiler_history、editor_profiler_capture_attribution、editor_profiler_column_sorting、frame_output_consumer_gpu_dependencies 共 4 项通过。
- 增强 profiler history 回归，验证 preparation 父子关系、同级 slot scope、CPU 数值以及延迟 GPU 回填不会污染 CPU 子项。
- 输出消费者测试继续验证同/跨队列依赖、无 CPU drain、双槽复用与重建生命周期。
- git diff --check 通过。

```powershell
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*editor_profiler*:*frame_output_consumer_gpu_dependencies" --output-dir build-release/preflight-tests
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

详细数据：[ZorahFullPreflightResult.json](ZorahFullPreflightResult.json)。
