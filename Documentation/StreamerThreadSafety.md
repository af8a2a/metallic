# Streamer 线程安全与 CPU 吞吐量基线

测量日期：2026-10-07（Asia/Shanghai）。范围为 `UploadStreamer` 的 CPU staging、请求归属、命令录制和上传资源寿命；吞吐量不代表 GPU 传输速度、加载时间或帧率。

## 并发契约

参考本机 `E:\NRI\Include\Extensions\NRIStreamer.h` 的设计：一个 Streamer 服务多个线程，每个独立录制任务使用自己的 copy batch，填充结束后消费一次，帧结束前汇合全部使用者。

- `beginCopyBatch()` 返回非零、不会复用的进程内 ID。不同 batch 可以并发 staging/录制；每个录制线程需要独立 command pool/context。同一 batch 的填充、查询凭据、取消和消费由调用方排序。
- `StreamBufferDataDesc::copyBatch`、`StreamTextureDataDesc::copyBatch` 和解压上传的 batch 参数指定请求归属。`copyStreamedData(commands, batch)` 只取走该 batch，成功或失败后均失效。取消/过期/跨 Streamer 的 ID 不能继续写入或消费。
- 零 ID 保留原有默认 batch，供协调线程处理目的资源上传。默认 flush 不会取走显式 batch；不带目的资源的 staging 和常量上传可以在工作线程调用。
- `pendingCopyCompletion(batch)` 的凭据覆盖该 batch 的提交事务和所属帧 GPU completion。错误帧、录制失败、回调异常、取消、丢弃或 Streamer 销毁不会发布未提交上传。无 `beginFrame(frame)` 的兼容用法不产生 completion 凭据。
- `stats().pendingCopies` 汇总尚未取走的全部 batch，`pendingCopyStats(batch)` 只统计指定 batch。正在录制的批次已从 pending 统计移除。`StreamingUploads::flush()` 只计入默认 batch。
- 创建、移动、销毁、`beginFrame/endFrame` 和关联帧的生命周期操作必须在全部使用者汇合后由协调线程执行。目标资源至少存活至录制结束；GPU barrier、队列依赖和 GPU 资源复用仍由调用方负责。
- 使用 `beginFrame(frame)` 时，arena 保留到 GPU completion；未使用该接口时，调用方必须防止 GPU 尚在读取的 ring 被覆盖，并保持 Streamer/源资源存活。内部锁不会把未等待 GPU 的 ring 复用变成安全操作。

接口见 [UploadStreamer.h](../Source/Runtime/Render/Streamer/UploadStreamer.h)。这不改变 `MeshletStreamRuntime`、`MeshletStreamResidency`、`StreamingUploads` 等上层状态机各自的线程约束。

## 实现取舍

请求、解压 barrier 和完成凭据归入各自的 batch。全局 mutex 只保护 staging arena、batch 容器和统计；录制时在锁内取走请求，随后在锁外调用 RHI 和 profiling callback，允许其他 batch 继续上传/录制。复用最多 32 份请求容器，减少重复 vector 分配。

staging 的分配、map、memcpy、flush、unmap 仍在锁内，避免共享映射状态的数据竞争。本次没有引入无锁分配器或并发 memcpy。`ScreenSpaceShadowPass` 的参数发布迁移到显式 batch，避免取走其他任务待录制的上传；Meshlet 协调上传继续使用兼容入口。

`beginFrame` 由协调线程一次性把 arena 持有者登记到 `RenderFrameContext`。工作线程只在 Streamer 锁内向持有者添加扩容后的 buffer，不再调用 `frame.retain()`。同一帧中扩容替换的 buffer 和压缩输入 arena 都保留到 GPU 完成，包括 Streamer 提前销毁的情况。

独立 host 写入按 `Buffer::hostWriteAlignment()` 隔开非 coherent flush atom；常量 ring 的帧 stride 也满足该对齐。coherent memory 使用直接返回路径，避免每次 staging 的两次无用取模。动态分配和输入 chunk 总大小增加溢出检查。

## 测量条件

| 项目 | 条件 |
| --- | --- |
| CPU | AMD Ryzen 7 7800X3D，8 核 / 16 逻辑处理器 |
| GPU / 驱动 | NVIDIA GeForce RTX 5070 Ti / 617.42 |
| 构建 | `build-pass-stages-nrd`，Release，Ninja / MSVC 14.51.36231 |
| 系统 | Windows 平衡电源方案；未固定频率和线程亲和性 |
| 原实现 | `d079a20ca1889cd4f12ef2697f3692216c3de110` |
| 负载 | 64 B 或 4096 B 请求，1/2/4/8 worker；全部 worker 共用一个 Streamer |
| 每 worker 操作量 | 64 B：32,768 请求；4096 B：8,192 请求 |
| 预热 | 每组合先完整执行 3 轮，随后采样 7 轮；预留足够 arena，测量轮不扩容 |
| 计时范围 | CPU staging/录制循环和起止 barrier；不含设备、目标资源、线程和 command pool 创建，也不含 join、清理或 GPU 提交/执行 |
| 模式 | 验证层关闭；`queuedFrameCount=1`，未调用 `beginFrame(frame)`；shader/PSO 不参与测试 |

`stage_only` 只做数据 staging，不带目的 buffer，两版均直接调用内部同步 API。`stage_record_16` 每 16 个请求录制一次 buffer copy，每个 worker 使用独立 command buffer 和不重叠的目的范围。旧实现使用外部 mutex 包住完整的“16 次 staging + flush”，保证旧共享待录制列表的请求归属；新实现每组创建独立 batch，无此外部锁。吞吐量单位是请求/s，**不是所有公共 API 调用的次数/s**。

原实现与最终版使用同一份加长后的 benchmark 源码；原实现定义 `METALLIC_STREAMER_BENCHMARK_BASELINE`，只排除它没有的 batch API 和新增正确性测试。该宏只供基线构建使用。

最初的短采样每 worker 仅 2048 次、1 轮预热，64 B 单线程样本不足 1 ms，多进程结果受调度影响明显。这些数据用于发现开销和改进测试，保存在本机诊断目录；**不用于下面的最终对比**。最终测量保留每轮全部样本，不按结果挑选或删除离群值。

## 最终结果

最终采样顺序为 A→B→B→A、B→A→A→B（A=旧实现，B=最终版），每实现每组合 28 个样本，共 896 行。以下为总吞吐量中位数，单位 M requests/s。范围列保留该组合的全部最小/最大值。

| 负载 | 字节/请求 | worker | 原实现 | 最终版 | 变化 | 原实现范围 | 最终版范围 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage_only | 64 | 1 | 3.240 | 3.248 | +0.2% | 2.356–3.402 | 2.802–3.373 |
| stage_only | 64 | 2 | 2.961 | 2.852 | -3.7% | 2.267–3.165 | 2.452–3.123 |
| stage_only | 64 | 4 | 2.709 | 2.657 | -1.9% | 2.193–2.812 | 1.972–2.923 |
| stage_only | 64 | 8 | 2.746 | 2.566 | -6.5% | 2.494–2.838 | 2.062–2.701 |
| stage_only | 4096 | 1 | 2.892 | 2.726 | -5.7% | 2.757–3.299 | 0.479–3.235 |
| stage_only | 4096 | 2 | 2.614 | 2.253 | -13.8% | 2.390–2.866 | 1.056–2.807 |
| stage_only | 4096 | 4 | 2.438 | 2.263 | -7.2% | 1.798–2.613 | 1.211–2.561 |
| stage_only | 4096 | 8 | 2.447 | 2.341 | -4.4% | 1.029–2.520 | 1.412–2.564 |
| stage_record_16 | 64 | 1 | 2.386 | 1.881 | -21.1% | 0.822–2.650 | 0.405–2.407 |
| stage_record_16 | 64 | 2 | 1.883 | 1.934 | +2.7% | 1.509–2.279 | 0.850–2.492 |
| stage_record_16 | 64 | 4 | 1.779 | 2.170 | +22.0% | 1.225–2.225 | 1.774–2.251 |
| stage_record_16 | 64 | 8 | 1.742 | 2.045 | +17.4% | 1.562–1.966 | 1.886–2.172 |
| stage_record_16 | 4096 | 1 | 1.865 | 2.054 | +10.1% | 0.704–2.201 | 1.941–2.370 |
| stage_record_16 | 4096 | 2 | 1.786 | 2.174 | +21.7% | 1.232–1.893 | 1.974–2.352 |
| stage_record_16 | 4096 | 4 | 1.747 | 2.089 | +19.6% | 1.606–1.891 | 1.942–2.166 |
| stage_record_16 | 4096 | 8 | 1.664 | 1.933 | +16.1% | 1.448–1.791 | 1.447–2.026 |


8 worker 的 staging+record 中位数分别由 1.742→2.045 M requests/s（64 B，+17.4%）、1.664→1.933 M requests/s（4096 B，+16.1%）。单线程 64 B 显式 batch 则由 2.386→1.881 M requests/s（-21.1%），归一化成本约 419→532 ns/请求；这个差值包含测试循环、batch 管理及录制，不能单独归因于 mutex。纯 staging 的多线程结果有回退，4096 B / 2 worker 中位数下降 13.8%。

取舍：保留 batch 的请求隔离与并发录制，避免全局录制锁阻塞其他任务；不宣称所有负载提速。单协调线程可继续用默认 batch，不需要为每个上传创建显式 batch。当前测试明确接受上述显式 batch 单线程及部分 staging 成本，并保留这组数据作为后续优化的回归基线。即使加长采样，桌面调度仍产生明显离群值；表中是本机测得的中位数和完整范围，不是跨硬件保证，也不是频率受控的 mutex 微基准。


该基准没有测量 completion-tracked 路径的 `frame.retain()` 消除收益，也没有测量并发纹理上传/解压吞吐量。正确性测试覆盖这些路径中的资源寿命与请求归属，不能据此推断其性能提升。

## 正确性验证

最终构建成功，开启 Vulkan validation 的 39 项测试通过；1 项 opt-in CPU benchmark 在正确性运行中按设计跳过，性能采样时单独执行。

- 8 worker 各自上传 128 KiB buffer、4×4 RGBA8 texture 和常量，检查常量偏移唯一、完成凭据归属，并经 GPU 回读逐字节校验全部数据。
- 8 个录制回调同时进入，验证不持有 Streamer 全局锁；回调内查询统计并触发额外 2 MiB 扩容。GPU timeline gate 阻止完成，验证同槽重用被拒绝、Streamer 销毁后旧 arena 仍可被 GPU 正确读取。
- 默认/显式 batch 隔离、取消、跨 Streamer、跨帧、重复消费、endFrame 丢弃、销毁、回调异常及 chunk 总大小溢出。
- GPU GDeflate 与 raw tile 混合上传交替使用默认/显式 batch，验证默认 flush 不偷取解压请求、CPU/GPU 字节一致、取消和 slot 重用。
- 既有 Meshlet 上传完成/发布重试、预算/延迟/回收、buffer/texture/constant 上传、帧内扩容和 GPU 寿命、RenderGraph 并行录制/像素输出回归。
- 生产阴影 pass 多帧测试覆盖 raw/SIGMA、运行时参数修改和分辨率变化；查看了 `ShadowSigma.png` 与 `ShadowDisabled.png` 的输出，阴影开启后出现预期的遮挡区域。没有进行与旧版本逐像素图像 A/B。

最终日志无 VUID。Vulkan loader 仍提示本机两个失效 layer JSON 路径。未运行 ThreadSanitizer、长时间压力测试、其他 GPU/平台；本机没有强制非 coherent 分配，相关对齐分支不是已验证的跨硬件结论。

## 证据与复现

- [采样 CSV](Benchmarks/StreamerThreadSafety-2026-10-07.csv)
- [基线提交、源码/二进制 SHA-256、采样顺序和环境](Benchmarks/StreamerThreadSafety-2026-10-07.json)
- [测试与 benchmark](../tests/rhi/StreamerConcurrencyTests.cpp)
- 本机日志、HTML 报告、早期诊断样本及图片：`.cache/benchmarks/streamer-thread-safety/`；最终正确性记录为 `long-regression.log`。

在兼容的 Visual Studio x64 开发环境运行：

```powershell
cmake --build build-pass-stages-nrd --target MetallicRHITests -j 8
$env:METALLIC_STREAMER_BENCHMARK = 'batches'
& .\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-no-validation `
  '--gtest_filter=*streamer_cpu_throughput*' --output-dir E:/metallic/.cache/streamer-benchmark
Remove-Item Env:METALLIC_STREAMER_BENCHMARK
& .\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-validation `
  '--gtest_filter=*streamer*:*frame_upload*:RHIEmptyHandle/22.*:*parallel_recording_builtin_pixels*:*parallel_recording_context_lifetime*:*realtime_ray_traced_sigma_shadows*' `
  --output-dir E:/metallic/.cache/streamer-regression
```

重建旧基线时，在上述基线提交上登记同一测试文件，并为该源定义 `METALLIC_STREAMER_BENCHMARK_BASELINE=1`，使用 `METALLIC_STREAMER_BENCHMARK=baseline`。最终实现也接受 `baseline` 模式，可用于观察调用方仍保留外部锁时的成本。不要在构建或 GPU 回归仍运行时同时采样。
