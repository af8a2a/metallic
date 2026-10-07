# BindlessHeap 线程安全与 CPU 吞吐量基线

测量日期：2026-10-07（Asia/Shanghai）。本报告记录内部同步改造的正确性和 CPU API 开销，不代表 GPU pass 或端到端帧时间变化。

## 实现与并发边界

- sampler、image、buffer 三个槽位分配器分别加锁；SampledImage/StorageImage 共用 image 槽位，Buffer/AccelerationStructure 共用 buffer 槽位。
- 独立分配器的 mutex、计数和空闲列表按 64 字节边界分隔，减少本次 Windows x64 平台的缓存行争用。
- sampler/resource 写入另用两把锁，覆盖完整的 descriptor write、dirty range 和 flush；分配操作不占用写入锁，所有操作只持有一把 heap 内部锁。
- 单元素 sampler 更新使用栈上 scratch，避免原路径的三个临时 vector 分配。批量分支保持动态存储。
- 命令绑定只读取不可变布局与地址，不增加锁。ResourceRegistry 继续锁住登记/复用/回收的组合操作。
- 同一槽位的写入/释放、heap 移动/销毁由调用方协调。输入资源保持存活且不变；descriptor 覆盖/回收前仍须等待 GPU 使用结束。

接口契约见 [RHI.h](../Source/Runtime/Render/GAPI/RHI.h)，实现见 [VulkanRHI.cpp](../Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp)。

## 测量条件

| 项目 | 条件 |
| --- | --- |
| CPU | AMD Ryzen 7 7800X3D，8 核 / 16 逻辑处理器 |
| GPU / 驱动 | NVIDIA GeForce RTX 5070 Ti / 617.42 |
| 构建 | 现有 build-pass-stages-nrd，Ninja / MSVC 14.51.36231，Release |
| 电源与调度 | Windows 平衡方案；未固定频率或线程亲和性，桌面系统仍可能引入波动 |
| 吞吐量验证层 | 关闭；正确性测试单独开启 Vulkan validation |
| 缓存/预热 | 每次采样、每个 worker 先执行 10,000 次相同操作；无 shader/PSO 编译或 GPU 命令提交 |
| 计时范围 | CPU RHI 调用和循环、起止 barrier；不含 Device/资源/线程创建、预热和线程 join |
| 采样 | 原实现 → 最终版 → 最终版 → 原实现；每进程每组合 7 次，共 14 次/组合 |

原实现以提交 `ded951bd27b459eadbba3482a6b2ae2ac9601821` 的 VulkanRHI.cpp 构建，搭配相同的新测试程序。单线程基线直接调用；多线程基线在每次公共 API 调用外围加同一把 mutex，确保不存在旧代码数据竞争。最终版直接调用内部同步路径。也记录了最终版仍使用外部锁的控制样本，供分析上层重复同步成本。

操作量固定，原实现和最终版一致：

- `allocate_release`：每 worker 1,000,000 个 allocate/release 对，计为 2,000,000 次 API 操作。worker 按编号轮转 sampler/image/buffer 三类。**该结果不等同于所有线程只分配 buffer 的争用情况。**
- `buffer_write`：每 worker 1,000,000 次 writeBufferView；各 worker 写不同槽位，引用相同只读 view。
- `sampler_write`：每 worker 200,000 次 writeSampler，各 worker 写不同槽位。
- `mixed_write`：每 worker 300,000 次写入；偶数 worker 写 buffer、奇数 worker 写 sampler。线程数为 1 时仅包含 buffer 写入，不是两类操作的混合。

## 最终配对结果

以下为总吞吐量中位数，M ops/s = 每秒百万次 API 操作。最后一列为最终版 14 个样本的最小值/最大值；原实现全部样本也保留在 CSV 中。

| 负载 | 线程 | 原实现 M ops/s | 最终版 M ops/s | 变化 | 最终版样本范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 混合槽位分配/释放 | 1 | 133.997 | 76.423 | -43.0% | 68.350–82.808 |
| 混合槽位分配/释放 | 2 | 83.928 | 141.764 | +68.9% | 86.700–149.582 |
| 混合槽位分配/释放 | 4 | 69.765 | 153.161 | +119.5% | 116.666–158.725 |
| 混合槽位分配/释放 | 8 | 35.445 | 111.246 | +213.9% | 102.225–119.464 |
| Buffer view 写入 | 1 | 50.184 | 39.090 | -22.1% | 31.456–41.816 |
| Buffer view 写入 | 2 | 34.733 | 32.350 | -6.9% | 27.938–36.418 |
| Buffer view 写入 | 4 | 31.653 | 28.202 | -10.9% | 24.171–31.470 |
| Buffer view 写入 | 8 | 18.006 | 15.903 | -11.7% | 15.350–17.063 |
| Sampler 写入 | 1 | 8.569 | 20.685 | +141.4% | 15.888–22.102 |
| Sampler 写入 | 2 | 6.744 | 16.715 | +147.9% | 13.723–18.706 |
| Sampler 写入 | 4 | 5.812 | 12.986 | +123.4% | 12.460–13.744 |
| Sampler 写入 | 8 | 5.180 | 9.531 | +84.0% | 7.291–9.742 |
| Buffer/Sampler 混合写入 | 1 | 55.276 | 29.741 | -46.2% | 18.259–42.280 |
| Buffer/Sampler 混合写入 | 2 | 11.166 | 22.174 | +98.6% | 17.916–27.783 |
| Buffer/Sampler 混合写入 | 4 | 10.166 | 19.043 | +87.3% | 17.754–20.454 |
| Buffer/Sampler 混合写入 | 8 | 8.602 | 16.691 | +94.0% | 16.180–16.996 |

分配/释放从独立锁和缓存行隔离中获得多线程收益，但单线程从约 7.46 ns/API 增至 13.09 ns/API（一个 allocate/release 对约增加 11.3 ns）。Buffer view 单线程约从 19.93 ns 增至 25.58 ns；8 线程吞吐量回退约 11.7%。这些是包含测试循环的归一化 CPU 成本，不能当成独立测得的 mutex 耗时。

Sampler 和混合写入的收益同时来自内部锁划分与单元素临时分配消除，不能全部归因于加锁。前两轮候选分别验证了仅加两把锁、移除 sampler 临时分配、分开槽位锁；最后验证缓存行隔离后才选定当前实现。早期短于 1 ms 的试测未用于本表。

当前取舍：保留线程安全保证与低争用分配路径，明确接受以上单线程及纯 buffer 写入成本。ResourceRegistry 仍有自己的组合操作锁，因此本表也不能直接推导生产场景的吞吐量或帧率提升。

## 正确性验证

最终构建成功。开启 Vulkan validation 的 16 项回归全部通过，无跳过：

- 8 worker、64 轮满容量分配/耗尽/交错回收再分配，覆盖全部 handle kind，检查唯一性及容量恢复。
- 8 worker 并发更新 384 个 descriptor，每槽 32 轮；GPU 验证 constant buffer、完整 backing slice、带偏移 buffer view，以及 storage image 写入后经 sampled image/sampler 读取的值。不同寻址模式和每项唯一数据用于检测槽位串写。
- 既有 BindlessBuffer、native descriptor heap、ResourceRegistry、RenderGraph 并行录制和像素结果回归。

日志无 VUID 报错。已有的错误 stride / 注入录制失败负向测试会输出预期 error 日志；Vulkan loader 仍报告两个失效的本机 layer JSON 路径。这些并未导致测试失败。

未运行 ThreadSanitizer、跨平台或长时间 soak；没有强制非 HOST_COHERENT heap，也没有新增并发 AS descriptor GPU 消费测试。AS 槽位分配已覆盖，其描述符写入与其他 resource 写入共用锁。GPU 读回证明本机路径正确，不涵盖上述未测配置。

## 证据与复现

- [原始 728 行采样数据](Benchmarks/BindlessHeapThreadSafety-2026-10-07.csv)
- [基线提交、源码与可执行文件 SHA-256](Benchmarks/BindlessHeapThreadSafety-2026-10-07.json)
- [测试与 opt-in benchmark](../tests/rhi/BindlessHeapConcurrencyTests.cpp)
- 本机各轮完整日志、试测数据和 HTML 报告：`.cache/benchmarks/bindless-thread-safety/`；最终正确性日志为 `final-regression.log`。

从仓库根目录、x64 Visual Studio developer shell 运行：

```powershell
cmake --build build-pass-stages-nrd --target MetallicRHITests -j 8
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-validation '--gtest_filter=*bindless_heap_concurrent*' --output-dir .cache/benchmarks/bindless-thread-safety/recheck

$env:METALLIC_BINDLESS_BENCHMARK = 'internal'
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-no-validation '--gtest_filter=*bindless_heap_cpu_throughput' --output-dir .cache/benchmarks/bindless-thread-safety/remeasure
Remove-Item Env:METALLIC_BINDLESS_BENCHMARK
```

原实现基线使用保留在 `build-pass-stages-nrd/tests/MetallicBindlessBaseline.exe` 的二进制，并设置 `METALLIC_BINDLESS_BENCHMARK=baseline`。该模式只允许运行单线程 direct 和多线程 external 同步测试；不要用原实现运行新增的无外部锁并发正确性测试。二进制丢失时，在独立 checkout 中保留新测试并使用 manifest 指定提交的原 VulkanRHI.cpp 重建。
