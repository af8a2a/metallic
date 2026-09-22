# ZorahFull：编辑器输出依赖的 CPU 等待优化

2026-09-22。已将正常帧的输出读取依赖交给 GPU timeline 排序，保留资源重建和销毁所需的 CPU 等待。同条件 30 秒 Full 漫游整帧均值从 44.15 ms 降至 28.30 ms，平均约 35.3 fps；P95 37.91 ms，仍未满足全程至少 30 fps。

## 实现

位置：[RenderGraphExecutor.cpp](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp)。

编辑器调用 `transitionOutput()` 时已执行两件事：让编辑器提交等待 graph 完成；记录编辑器 frame completion，保护共享输出的生命周期。下一帧 graph 原本既将这些 completion 加入 `slot.frame` 的 GPU dependencies，又因它们非空执行 `waitForSubmittedWork()`，造成重复的 CPU 串行等待。

本次调整：

1. 正常 `execute(RenderGraphSubmitDesc)` 不再因存在 external completion 设置 drain bit 4。所有队列 segment 仍通过 `QueueSubmissionTracker::submitSegment()` 等待这些 GPU timeline completion。
2. 每帧移除已完成的外部 completion，避免历史提交状态不断积累。未完成 completion 保留在 executor 内，并由 frame dependency 保活其 semaphore。
3. 编译/resize、shader reload、场景绑定更新、场景 revision 变化、非 overlap pass 和析构仍保留原有等待。两个 submission slots 仍在复用前等待各自完成。
4. 不在成功挂接 GPU 依赖后立即清空 external completion：录制失败或重建时仍需保护外部消费者。`drainReasonMask` 保留 1/2 的含义，历史 bit 4 在新捕获中不再出现；`externalCompletionCount` 表示尚未完成的外部消费者数。

没有改动 shader、LOD、流送预算或帧槽数量。外部命令缓冲录制接口 `execute(CommandBuffer&)` 的原有同步路径保持不变。

## 受控依赖测试

新增 [frame_output_consumer_gpu_dependencies](../tests/rhi/FrameContextTests.cpp)：

- 在 graphics 和独立 compute 队列上分别消费 graph 输出。
- 用未放行的 timeline gate 阻塞消费者，下一次 graph execute 使用零 CPU 等待超时，必须仍能成功提交。
- 新 graph 的 GPU completion 必须保持未完成；放行后读回必须是前一帧数据，验证 GPU 不会提前覆盖正在读取的输出。
- 连续四轮复用两个帧槽；重复 `transitionOutput` 不得增加重复依赖，已完成的历史 completion 不得累积。
- 待消费的输出触发重建时，重建仍必须等待，验证资源替换的生命周期保护。

该测试在原始 executor 上失败、修复后通过。Release 构建成功；开启 Vulkan validation 的 14 项相关回归通过，包括 descriptor snapshot、历史依赖、双槽复用、跨队列提交、fork/join、提交取消和上传生命周期。

测试集存在一个已确认的基线失败：`frame_self_submit_two_slots` 在原始 executor 和修复版上均报告 `independent copy branch was blocked by the graphics branch`，发生在尚未添加外部输出消费者之前。当前 graph 计时 start/join 边界与该旧独立 copy 断言不一致，本次没有改动这项行为。最终 14 项通过的命令排除了该已复核的失败；不能表述为整个 `frame_` 集合全绿。

日志：

- `build-release/output-wait-frame-tests.log`：首次完整集合，14 通过、1 基线失败。
- `build-release/output-wait-negative-tests.log`：原始 executor 对照，新增测试预期失败，旧 copy 测试同样失败。
- `build-release/output-wait-restored-build.log`、`output-wait-restored-tests.log`：恢复优化代码并重新编译后的最终构建和 14 项通过记录。

## Full 验证与性能

重新采集本次修改前基准，避免直接使用 9 月 20 日的旧结果。两次性能运行均使用 RTX 5070 Ti、驱动 616.92，输出 1797×660、DLSS Quality 内部 1198×440、完整相同 graph 和绝对相机关键帧、10 秒预热、30 秒路线、VSync 开启、隐藏编辑器、关闭 validation 和诊断工作量重放。

两次运行的配置、graph、相机路线、输出与内部尺寸逐项相等。时间采样路线下帧数不同，cut/驻留和 GPU 时钟并未逐帧锁定，因此性能数据是同配置运行对比；正确性和无 CPU 阻塞由上述受控 gate 测试独立验证。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 采样帧数 | 680 | 1061 |
| 整帧均值 ms | 44.152 | **28.301** |
| 平均速率 fps（1000/均值） | 22.6 | **35.3** |
| 整帧 P95 ms | 52.626 | 37.910 |
| 整帧 P99 ms | 58.911 | 46.825 |
| 整帧最大 ms | 63.014 | 64.325 |
| >33.33 ms 帧数 | 680/680 | **137/1061（12.9%）** |
| 最长连续超预算帧数 | 680 | 28 |
| Prior frame drain CPU ms | 13.064 | **未触发** |
| 输入前帧槽等待 CPU ms | 0.016 | 0.155 |
| Render Frame/Wait Slot Completion CPU ms | 0.011 | 0.010 |
| Graph submission slot wait CPU ms | 0.000066 | 0.023 |
| Graph 录制/提交 CPU ms | 22.462 | 20.067 |
| Graph GPU envelope ms | 21.392 | 22.071 |
| GPU 请求反馈 CPU ms | 7.695 | 6.986 |
| 外部 completion 数 | 每帧 1 | 每帧 1 |
| drain reason | 每帧 4 | **每帧 0** |
| HZB 有效帧 | 680/680 | 1061/1061 |
| 缺失 GPU 计时帧 | 0 | 0 |

整帧均值下降 35.9%。GPU 时间没有降低；收益来自 CPU 录制与前序 GPU 工作重叠。等待没有等量转移到帧槽入口。表中 scope 存在父子关系，不能把父子耗时相加解释为独立成本。

另跑 15 秒开启 Vulkan validation 的 Full 漫游，32 帧完整结束，无 VUID/DeviceLost，HZB 32/32 有效，drain mask 全为 0，外部 completion 数始终为 1。debug/validation 有显著开销，该运行仅用于正确性检查，不作为性能成绩。

## 剩余方向

当前平均已超过 30 fps，但尾部仍不达标：137 帧超过 33.33 ms，最大帧未改善。下一步应按慢帧拆解 CPU 请求反馈（目前均值约 7 ms）、graph 录制与流送更新；同时检查对应 GPU workload，不能根据均值宣称持续 30 fps。此次短测也不能替代长期漫游/频繁切场景的稳定性验收。

## 证据与复现

- 修改前：`build-release/full-output-wait-before/run1/`。
- 修改后：`build-release/full-output-wait-after/run1/`。
- Full validation：`build-release/full-output-wait-validation/run1/`。
- 每个目录保留 Capture、Frames、Summary、GPU 占用/竞争监控、日志；根目录 Manifest 保存 executable/shader hash、驱动和资产标识。
- [机器可读对比](ZorahFullOutputDependenciesResult.json)。

性能命令（使用新的输出目录）：

```powershell
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-directory> -Runs 1 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

定向回归：

```powershell
build-release/tests/MetallicRhiTests.exe --filter frame_output_consumer_gpu_dependencies --rhi-validation
build-release/tests/MetallicRhiTests.exe '--gtest_filter=*frame_*-*frame_self_submit_two_slots' --rhi-validation
```
