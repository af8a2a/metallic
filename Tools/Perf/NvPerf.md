# NvPerf Vulkan backend

直接在 Metallic 进程内通过 Nsight Perf SDK 采集，不启动 Nsight UI，也不依赖
`.ngfx-gputrace` 或 CSV 导出。当前仅支持 Windows x64、NVIDIA GPU，以及经过
availability 检查的单 pass 指标集。

## 构建

SDK 不进入仓库。解压完整 SDK 后配置 Release 构建，例如：

```powershell
cmake -S . -B build-release -DMETALLIC_ENABLE_NVPERF=ON -DMETALLIC_NVPERF_SDK_ROOT=E:/metallic/build/nvperf-sdk-2025.1
cmake --build build-release --target MetallicGPUDrivenSample --config Release
```

使用已初始化的 MSVC x64 环境。默认 `METALLIC_ENABLE_NVPERF=OFF`；启用后也只在
`METALLIC_NVPERF=1` 时加载指定 SDK 的运行库、添加所需 Vulkan 扩展并插入范围。
未编译 backend 却请求采集时明确报错，不生成成功占位结果。

## 三次独立采集与复核

```powershell
python -B Tools/Perf/NvPerf.py run --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --assets build/m2-assets-20260925/mini.json --exe build-release/Source/MetallicGPUDrivenSample.exe --build-dir build-release --output build/nvperf-new
python -B Tools/Perf/NvPerf.py verify build/nvperf-new
```

输出目录必须不存在。runner 使用现有 shader experiment 互斥锁，串行启动三个独立
隐藏窗口进程，每个默认限时 240 秒，仅清理自己启动的进程。SDK 采集仍需要真实
GPU 和可用图形会话；“headless”在这里指无人操作、无分析器 UI，不承诺无桌面服务。

默认指标：`gpu__time_duration.sum`、`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`。
`--metrics <json>` 可指定含 1–16 个完整 NvPerf metric 名称的 JSON 数组。
名称、硬件计数 availability、计划 pass 数都在开始采集前验证。
多 pass 返回 `multipass_requires_restored_workload`；不会把不同历史状态的帧拼成
一个结果。后续应先补齐状态恢复协议再启用多 pass。

## 证据契约

- 只测 primed WorkControl 的单个目标帧；在 graphics queue 实际
  `dispatchIndirect` 周围放置 `WorkControl/early` 与 `WorkControl/late` 两个 range。
  不是从宽范围 debug marker 或 ProfileMark 推测 shader 归因。
- 异步软件光栅关闭；采集前后 GPU drain，profiling 期间 present 前 queue idle。
  禁止 Graphics Capture、GPU Trace、RenderDoc、validation 与 pipeline statistics 混用。
- 不改变系统 GPU 时钟；显式标记为 diagnostic。M2/M3 普通性能验收拒绝这些运行。
- 必须 single pass 完整 decode、零 range/trace bytes 丢失、范围恰好两项、指标有限非负。
  time duration 必须非零。值的单位名与量纲来自 SDK evaluator。
- 保存 availability、config image、counter prefix、counter data image、JSON、stdout/stderr、
  进程生命周期、源代码、SDK 头文件/运行库哈希、实际绑定 SPIR-V、工作列表和最终图像 readback。
  workload 绑定依据相同 priming 策略与采集前后检查点，未在计数区间内插入 readback。
- 三次验证比较完整 workload 身份与 depth/visibility；逐指标保留全部原始值，报告
  `(max-min)/median`，超过 10% 标记不稳定。不稳定不伪装成 backend 采集失败或候选加速。
- `verify` 重新验证归档文件哈希、工作负载 readback 与 JSON 统计。
  **不会再次用 SDK 解码 CounterDataImage**，返回 `counterImageReevaluated=false`。
  归档哈希用于检测意外改动，不是外部签名。

范围指标不提供源码行、PC sampling、SASS 依赖或孤立回放归因；计数器本身的硬件
范围语义仍由 NVIDIA 定义。Full Zorah 驻留资格和 M3 性能验收是独立事项。

2025.1 SDK 将 Metallic 的 Vulkan 1.4 版本标记为未正式支持。backend 保存
`vulkanApiVersion` 和 `vulkanVersionOfficiallySupported`，stderr 原样保留警告；
实际成功采集不等于 NVIDIA 官方认证该组合。升级配套 SDK 后需重跑验证。

参考：[Nsight Perf SDK](https://developer.nvidia.com/nsight-perf-sdk)、
[SDK 已知限制](https://developer.nvidia.com/nsight-perfsdk/getting-started/release-note-v2026.3)。
