# MiniZorah 切换后 CLAS 搬移崩溃

2026-09-13，RTX 5070 Ti，驱动 616.64。已修复，不需要重新 cook。

## 根因与证据

`MOVE_OBJECTS_NV` 的 scratch 需求由 `VkAccelerationStructureBuildSizesInfoKHR::updateScratchSize`
返回。紧凑池分配和 RHI 参数检查都误用了 `buildScratchSize`。本机移动查询的该字段为 0，
于是实际只分配了 256 字节对齐余量，检查也错误地放行。

[NVIDIA 参考实现](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/src/scene_streaming.cpp)
使用 `m_clasScratchMoveSize = buildSizesInfo.updateScratchSize`；本地参考代码亦如此。

用户日志中的 Sponza → MiniZorah 切换在 22:30:57 出现 `DeviceLost`。用真实编辑器的
1564×708 视口、多队列和多帧提交复现两次，均在切入 MiniZorah 后的前几帧失败。
Aftermath 都报告驱动内部 `AS Build or Refit / ray_tracing_02 @ 0xad0` 的 GPU 写入页故障。

第二次复现的地址记录明确显示越界：

| 项目 | 数值 |
| --- | --- |
| 搬移 scratch 起点 | `0x175fd000` |
| 当时可用 scratch | 256 字节 |
| GPU 故障写入地址 | `0x175fe000` |
| 相对 scratch 偏移 | 4096 字节 |

故障地址关联的旧资源分别为 3296、9888、412 字节，并非固定同一个被提前销毁的对象。
越界写入能否触发页故障取决于相邻地址的分配状态，因此单次加载或较大的共享 scratch
可能掩盖问题。初期只依据 Aftermath 的 `Destroyed` 标志怀疑资源过早释放，不是最终根因。

原始与复现文件：

- `.cache/aftermath/Metallic_Engine_Editor-48556-1.json`：用户崩溃。
- `.cache/aftermath/Metallic_Engine_Editor-1852-1.json`：第 4 次切换复现。
- `.cache/aftermath/Metallic_Engine_Editor-64956-1.json`：带 scratch 地址记录的复现。
- `build-relwithdebinfo/clas-device-lost/editor-trace-repeat.log`：复现命令和地址。

## 修复

- `MeshletStreamCompactClasPool` 按移动查询的 `updateScratchSize` 加对齐余量分配。
  MiniZorah 默认批次的缓冲从 256 字节变为 196,992 字节，约 192.4 KiB。
- Vulkan RHI 对每次实际搬移按同一字段检查 scratch 的有效剩余范围，在记录 GPU 命令前拒绝不足的缓冲。
- RHI 接口补充移动查询字段说明；实际尺寸分配、搬移、联合冷页回收继续启用。
- `METALLIC_TRACE_CLAS=1` 可记录 build/move 地址、计数和 scratch 容量/需求；默认不输出。
- 增加编辑器隐藏窗口切换回归，以及不逐帧等待全部 GPU 工作的 MiniZorah 回归。

## 验证

修复前，关闭 CLAS 和使用旧固定槽 CLAS 分别通过 4 次切换、4800 帧；紧凑搬移路径复现上述故障。

修复后：

- Release 编辑器完成 **12 次 Sponza → MiniZorah 切换**，每次先渲染 Sponza 60 帧，再渲染 MiniZorah 64 帧，
  每轮结束验证编辑器与渲染图提交都已完成。无 DeviceLost。
- **12/12 RHI 回归通过**，无跳过、无 Vulkan validation 错误。包含实际尺寸/两次搬移、
  不足 scratch 拒绝、紧凑池生命周期、流送驱逐重载、联合冷页回收、StreamAsset 光线查询和 Profiler。
- 其中 MiniZorah 多帧回归完成 **1200 帧**，1199 帧观察到提交重叠，包含大角度持续转向。
- 两对象移动查询需要 128 字节 scratch；新增回归用不足该尺寸的缓冲验证 RHI 返回 `InvalidArgument`。
- `MetallicGPUDrivenSample.exe` 和 `Metallic.exe` Release 重建成功。

日志：`build-relwithdebinfo/clas-device-lost/editor-fixed.log`、`final-tests.log`、
`final-tests/MiniZorahClasInFlight.jsonl`、`final-tests/ClasSizeMove.txt`。

验证边界：RHI 验证层较旧时自动回退 KHR OMM 为 shader alpha traversal，日志有相应提示；
Release 编辑器切换验证使用实际 KHR OMM 路径。编辑器各轮 GPU 完成后已输出通过标记；
测试进程在 Streamline/NGX telemetry 退出阶段等待，诊断确认后终止了测试进程，未把该退出等待计作 GPU 故障。

隐藏编辑器回归（环境变量仅用于 smoke test）：

```powershell
$env:METALLIC_SMOKE_TEST_MINIZORAH_SWITCH = '1'
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
$env:METALLIC_SMOKE_TEST_SWITCH_CYCLES = '12'
$env:METALLIC_SMOKE_TEST_ROAM_FRAMES = '64'
.\build-release\Source\MetallicGPUDrivenSample.exe --smoke-test
```

省略后两个参数时运行 4 轮、每轮 MiniZorah 1200 帧，180 帧后开始自动转向。
