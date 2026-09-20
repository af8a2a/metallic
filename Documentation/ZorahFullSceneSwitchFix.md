# ZorahFull 场景切换分配失败修复（2026-09-20）

## 原因与修复

用户 14:33 日志在 Full 创建 `MeshletStreamRuntime visible clusters` 的约 512 MiB 缓冲时被统一预算拒绝：申请 536870400 B，当时可用 285802054 B。失败发生在完整纹理上传之前，纹理域仅 512 B；本次不是贴图解码失败。

`RenderGraphExecutor::waitForSubmittedWork()` 过去只等待 GPU、释放命令缓冲，帧槽内 `RenderFrameContext` 仍保留旧 stream owner。旧 pass 销毁后，MiniZorah 的池因这些引用继续存在；下一次 `begin()` 才释放引用，但 Full 在此之前就因预算不足无法编译。日志中的 geometry 峰值 4 GiB = Full 3.5 GiB + Mini 512 MiB，CLAS 峰值 2.25 GiB = Full 2 GiB + Mini 256 MiB。

现在在全部提交完成、命令缓冲释放、时间戳解析后 reset 帧槽，释放已完成帧的资源引用。Streamer 随后收集失去外部引用的旧 session，在新 stream 分配前释放旧池。继续保留预算、安全余量和 GPU 完成等待。

编辑器编译失败时消费当前 dirty 请求，停止同尺寸/同输出的逐帧重试。视口增加 **Retry scene load** 按钮；显式重试、图设置/输出/尺寸变化、重新加载图或场景允许再次尝试。

## 验证

- Release `MetallicGPUDrivenSample.exe` 和 `MetallicRhiTests.exe` 构建通过。
- 新增 `render_graph_scene_switch_retirement`：两个队列提交帧持有旧场景 GPU buffer，连续替换三次，要求新 pass 分配前旧 owner 已释放。旧实现稳定失败，修复后通过。
- 19 项定向回归通过，覆盖资源回收、图重编译、预算拒绝、上传生命周期、跨队列完成点、取消/DeviceLost 清理、历史依赖和 GPU 时间戳。
- `frame_self_submit_two_slots` 的独立 copy 分支断言在修复版和仅撤回本次 frame reset 的对照版本均失败，属于现存限制，未修改该测试来掩盖失败。
- 960×540 原生、Vulkan validation 开启：两轮 MiniZorah 队列渲染 → Full，全量根级在两轮第 397 帧就绪；asset/world 两入口通过、卸载后 stream session 为零。材质域 391697408 B（约 373.55 MiB，包含 fallback），没有重复 Full 纹理 owner。耗时受当时负载影响，不作为性能比较。
- **真实 Editor 路径、隐藏窗口、1404×674、完整 DLSS 图**：两轮 MiniZorah → Full 均通过；各在第 397 帧达到 103528/103528 根页就绪，再继续 60 帧。就绪时 geometry 为 3758096384 B / 1 allocation，CLAS 为 2147483648 B / 1 allocation；两轮均未出现预算拒绝。随后故意把 graph reservation 设为 UINT64_MAX，验证编译失败暂停；恢复预算不会自动重试，显式 retry 可恢复渲染。日志末尾的这一次 OutOfMemory 是测试主动注入。

隐藏窗口测试通过应用自身 CLI 执行，不操作主桌面的鼠标、键盘或前台窗口。此次覆盖切换、根级首帧与短时继续渲染，不等同于长时间自由漫游验收。

## 证据与复现

- [最终 19 项回归](../build-release/full-switch-fix/final-regression.json)
- [旧实现对照](../build-release/full-switch-fix/baseline.log)
- [原生两轮 Full 报告](../build-release/full-switch-fix/full/ZorahFullFirstFrame.json)
- [Editor / DLSS 两轮切换及 retry 日志](../build-release/full-switch-fix/editor-full.log)

```powershell
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
$env:METALLIC_SMOKE_TEST_ZORAH_FULL_SWITCH = '1'
& E:/metallic/build-release/Source/MetallicGPUDrivenSample.exe --smoke-test
```

```powershell
$env:METALLIC_TEST_ZORAH_FULL = '1'
$env:METALLIC_ZORAH_FULL_CYCLES = '2'
& E:/metallic/build-release/tests/MetallicRhiTests.exe --rhi-validation '--gtest_filter=*zorah_full_first_frame' --output-dir E:/metallic/build-release/full-switch-fix/full
```

## 几何覆盖验收纠正

上述测试验证了资源回收、页就绪、DLSS 提交和失败重试；当时环境背景可见，未证明 Full 表面成功着色。用户 15:01 的空场景截图暴露了独立的几何解码问题。现已加入隐藏环境后的真实输出像素检查，详见 [Full 几何解码修复](ZorahFullGeometryDecodeFix.md)。
