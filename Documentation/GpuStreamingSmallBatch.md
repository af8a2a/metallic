# 同队列小批次解压与流送速度指标

2026-09-18，基于 `56e5da2`。本轮在保留同队列 EXT 解压的基础上，为零散需求增加后台 CPU 解码分流，并拆出 copy / decompress / barrier 的 GPU 计时。完整数据、输入与二进制 hash、原始日志路径见 [结构化证据](GpuStreamingSmallBatchEvidence.json)。前一轮强制 GPU 解压结果保留在 [阶段一报告](GpuStreamingDecompressionImplementation.md)。

## 策略和使用

`enableGpuDecompression=true` 时，新增 `gpuDecompressionMinBatchBytes`，默认 **1 MiB**，单位为合格 GpuTiles 页的最终 device payload 字节。设为 `0` 可复现原来的全 GPU 策略。GPU 解压总开关默认仍为 false；本轮没有证明自适应策略稳定优于纯 CPU 解码，尚不调整总开关默认值。

Residency 在提交 worker 前，按既有优先级检查当前可接纳的请求：受页数预算、剩余解码后上传字节预算、in-flight 空位和 prefetch 配额约束。达到阈值则允许该次调度使用 GPU；不足则让现有后台 worker 直接解码。不等凑批，不把解码搬到渲染线程。单个超预算页的进展规则保持不变。

这是**请求接纳阶段的策略**，不是每次 GPU 命令的最小输出量保证。异步 worker 完成顺序和后续上传预算仍可能把一个大请求批次拆成多次小 flush。原有 synchronous 兼容路径也采用相同阈值，但其解码仍在调用线程进行。

两条 GPUDriven / VisibilityBuffer pass 都接受该属性，例如：

```json
{
  "enableGpuDecompression": true,
  "gpuDecompressionMinBatchBytes": 1048576
}
```

GPU 路径继续使用同一上传 recording 内的 copy → EXT 解压 → publish barrier，沿用原 completion receipt、取消和 frame-slot 复用机制。CPU / GPU 结果可以混合进入既有发布链路，不改变 CLAS sideband 和最终驻留几何格式。

## 新指标的含义

Profiler → Streaming 显示数值和默认展开的 **Loading Speed** 曲线；`debugSnapshot().throughput`、`Baseline.json` 的 `finalStream.throughput` 与 `Frames.jsonl` 的 `stream.throughput` 提供相同数据。

| 字段 / UI | 统计终点 | 单位 |
| --- | --- | --- |
| `loadedStoredMiBPerSecond` / Load | 有效 worker 结果被 Residency 接收；按源页 payload 大小计数 | MiB/s |
| `loadedPagesPerSecond` | 同上，包含重复加载 | pages/s |
| `preparedMiBPerSecond` / Prepared | 已准备页对应的最终 device payload 逻辑大小；GPU 分支此时尚未在 CPU 解压 | MiB/s |
| `transferMiBPerSecond` / Copy payload | 接纳进 staging 的数据；CPU 页为解码后数据，GPU 页为 tile 存储数据 | MiB/s |
| `geometryReadyMiBPerSecond` / Geometry ready | Residency 观察到上传 completion receipt 完成 | MiB/s |
| `geometryReadyPagesPerSecond` | 同上，以完成页数计 | pages/s |
| `smallBatchCpuPages` | 开启 GPU 路径后，经小批次策略分流且成功准备的 CPU 页累计数 | pages |

速率使用实际墙钟时间，窗口约 1 秒，最多每 50 ms 存一次历史样本；初始化不足 50 ms 时不显示瞬时尖峰，空闲帧继续老化历史，重载清空计数。`windowSeconds` 给出实际分母，长帧时可能大于 1 秒。

Load 是映射文件的逻辑 payload 处理量，**不是物理 SSD 吞吐**。Copy 不含 GPU 封装 sideband，也不是 PCIe 硬件计数器；已接纳后取消的工作可能已计入 Copy。Geometry ready 不代表 CLAS/RT 构建或首次 draw 消费完成。加载、准备和完成计数分开，不能把 staging 接纳误记为加载完成。

`Stream Begin` 下新增 `Upload preflight`、`Upload copies`、`Decompression input barrier`、`GPU decompression`、`Decompression publish barrier`。计时使用既有延迟回读机制，不增加 CPU 等 GPU 的同步点。

## MiniZorah 同路线对照

RTX 5060，驱动 616.92。相同 v10 GDeflate 资产、参考相机路线、LOD/驻留/上传预算、实时 DLSS-SR；1920×1080 输出、1280×720 内部渲染。每轮 3000 帧，帧时间统计排除前 300 帧。七轮使用完全相同的可执行文件、源代码指纹、Shader 和 Replay hash；关闭 validation，OS 文件缓存已暖。

| 运行顺序 | 策略 | GPU P50 ms | GPU P95 ms | 全程有解压命令的帧 | 活跃窗口 Geometry-ready 中位数 MiB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| forced1 | 全 GPU | 8.540 | 9.510 | 1980 | 30.08 |
| adaptive1 | 1 MiB 自适应 | 7.856 | 8.609 | 95 | 33.62 |
| adaptive2 | 1 MiB 自适应，有竞争 | 8.047 | 11.618 | 98 | 31.11 |
| forced2 | 全 GPU，有竞争 | 10.386 | 12.216 | 1981 | 24.32 |
| cpu | 纯 CPU | 7.950 | 8.758 | 0 | 33.09 |
| forced3 | 全 GPU，补测 | 8.500 | 9.668 | 1972 | 30.05 |
| adaptive3 | 1 MiB 自适应，补测 | 7.953 | 8.964 | 96 | 32.76 |

保留中间两轮，不把背景竞争当成代码收益：进程引擎采样中，adaptive2 / forced2 的 Unity 3D 平均占用约 **7.71% / 19.34%**，峰值 **36.03% / 36.93%**；其余轮 Unity 3D 平均低于 0.04%。采样覆盖启动、预热、回放和退出阶段，缺省行按该时间点零值处理；它不是整个 GPU 利用率，也不证明其他轮完全无干扰。

两组低干扰配对的 GPU P95 分别降低 **0.901 ms（9.48%）** 和 **0.704 ms（7.28%）**，有解压命令的帧数减少约 **95.2% / 95.1%**。活跃窗口 Geometry-ready 速率由约 30.1 增至 32.8–33.6 MiB/s。该速率分布只统计稳态中大于零的窗口，不是整段墙钟平均或磁盘带宽；固定路线仍会因异步完成时序产生少量页序列差异。

第一组稳态分项揭示收益来源：

| GPU 分项 | 全 GPU | 自适应 |
| --- | ---: | ---: |
| Stream Begin，全部 2700 个稳态帧平均 | 0.546 ms | 0.024 ms |
| Upload copies，出现该分项的帧平均 | 0.0065 ms | 0.0062 ms |
| GPU decompression，执行帧平均 | 0.753 ms | 0.798 ms |
| Decompression publish barrier，执行帧平均 | 0.0016 ms | 0.0015 ms |
| 稳态解压执行帧数 | 1936 | 66 |

减少的是支付 GPU 固定解压成本的次数，单次 GPU 解压没有变快。自适应第一轮中，29,609 个完成页包含 9,898 个 GPU 安装页和 19,711 个 CPU 分流页；接纳的 copy payload 为 698,432,656 字节，最终完成几何为 768,692,320 字节。相比全 GPU 策略多传一部分解码后数据，换取避免零散帧的同队列阻塞。纯 CPU 对照 P95 为 8.758 ms，落在两轮自适应结果之间，因此不宣称相对纯 CPU 的稳定整帧收益。

## 正确性与剩余限制

- 四个目标构建通过：`MetallicGpuPageTests`、`MetallicRhiTests`、`MetallicMeshletTranscode`、`MetallicGPUDrivenSample`。
- 6 项独立测试通过：包含同一 PageLoader 混合 CPU/GPU 请求、重初始化、原生解压字节 oracle，以及墙钟吞吐、空闲归零和重置。
- 3 项解压 RHI 测试通过：阈值为零、恰好一页大小和一页大小加一的边界；CPU/GPU 最终字节一致；未完成上传不计 Geometry ready；既有取消与 slot 复用用例仍通过。
- 3000 帧质量回放通过，15 个安全 cut/预算检查点通过；第 29 帧仍在细化，第 59 帧起各采样点无可见超目标细化。最终 29,371 页、762,443,168 字节完成，19,076 页走小批次 CPU，10,295 页走 GPU；加载失败/非法请求为零。最终闲置窗口速率归零。
- 7 轮性能数据的速率均有限且非负，累计计数单调，最终 Geometry-ready 页/字节与上传完成统计一致。
- 额外 7 项 Streamer 回归中 **6 通过、1 失败**：上传字节预算、上传完成、有序发布重试、buffer、constant 和 render graph flush 通过；`streamer_texture_upload` 报纹理像素不匹配，单独执行也失败。2026-09-17 构建的 `build-dev/tests/MetallicRhiTests.exe` 在同驱动下出现同一失败，说明现象早于本次改动，但该旧二进制没有可核对的精确 source hash，尚未定位根因。没有将它计为通过或改动纹理路径掩盖结果。

正确性检查使用进程级 `VK_LAYER_PATH` 指向本地 **VVL 1.4.357 + f6ff981**，开启同步校验，日志无 Vulkan validation error；系统 SDK 和 External 未修改。完整回放仍在结果落盘、GoogleTest 完成后遇到既有 Streamline teardown 停留，由脚本在 8 秒宽限后清理自己启动的进程，`Process.json` 有记录。检查点 oracle 不替代逐像素验证，也不证明实际首屏绘制延迟改善。

复测命令（每次指定新的输出目录）：

```powershell
Tools/RunMetallicCfgReplay.ps1 `
  -Replay .cache/gpudriven-four/Replay.json `
  -OutputRoot .cache/fast-streaming/recheck-adaptive `
  -Cases m1 -Realtime `
  -StreamAsset .cache/fast-streaming/MiniZorah.gdeflate.meshstream.bin `
  -GpuDecompression On -GpuDecompressionMinBatchBytes 1048576
```

全 GPU 对照把阈值改为 `0`；纯 CPU 对照使用 `-GpuDecompression Off`。质量检查使用 `-Cases quality` 并设置上述 `VK_LAYER_PATH` 和 `VK_LAYER_VALIDATE_SYNC=1`。
