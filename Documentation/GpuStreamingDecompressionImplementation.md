# GPU-ready 几何页流送：第一阶段实现

2026-09-18，基于 `5b5a3c3`，对应 [接入设计](GpuStreamingDecompressionResearch.md) 的阶段 1。EXT direct 解压已通过新驱动下的集成测试和完整实时回放。默认仍使用 CPU 路径：GPU 解压降低 CPU 加载处理和 staging 字节，但当前同队列实现增加了 GPU 帧时间，尚未达到默认启用门槛。

## 已实现

- v10 流文件读取及独立离线转码工具 `MetallicMeshletTranscode`。保留 v8/v9 读取，转码支持 v9/v10 输入；不重新生成 LOD，不改变 page/group/cluster 编号。禁止覆盖已有输出，失败清理自己的临时文件，目录校验通过后再发布输出文件。离线转换以最多 8 个后台批次并行编码，按原页序写出。
- `GpuTiles` 传输封装：最终 float3 紧凑 payload、每 tile 最多 64 KiB、Raw/GDeflate 独立选择、范围检查、metadata/tile CRC32C。解压后仍是已有 shader 使用的 v4 payload。编码器固定 NVIDIA libdeflate 提交 `8ba9502fb30d2bf728592d121f0d402e40c8cb05`，下载内容另有 SHA-256 校验。
- CPU GDeflate 回退；GPU worker 保留压缩数据并提取 CLAS sideband。GPU 路径无需先把完整页在 CPU 上解压，CLAS 的 vertex/index 地址基于最终驻留页计算。
- RHI 查询/启用 `VK_EXT_memory_decompression` 与 `GDEFLATE_1_0`，增加 64 位原生 buffer usage、VMA usage2 支持和批量 direct 解压命令。预检查拒绝非法范围、重叠目标和不支持的命令队列。
- Streamer 在同一安装批次中完成压缩 tile 上传、EXT 解压和 Raw tile 复制。仅 payload tile 经 GPU copy，sideband 留在 CPU；输入缓冲区按 frame slot 保留至完成。每 slot 最多接纳 64 MiB 压缩封装输入，仍沿用原有解压后字节预算与页数预算。输入缓冲区容量按需增长，增长时旧分配也保留至完成，因此瞬时分配量可能超过当前容量。
- 安装记录包含解压和写后 barrier，再允许同 recording 的有序发布；CPU 驻留完成仍等待原有 submission/completion receipt。失败预检查发生在任何 copy 写入之前。取消未提交批次沿用原有回收机制。
- 两条 GPUDriven/VBuffer 路径都接受 `enableGpuDecompression`，回放脚本支持独立选择资产和 CPU/GPU 解码模式。

## 驱动升级后的完整验证与性能结果

本机现为 RTX 5060、驱动 **616.92**、Vulkan **1.4.351**。Loader 返回 291 个设备扩展，`VK_KHR_device_address_commands` 已列出且 feature 为 true。原先的引擎设备创建阻碍已解除。完整数据、协议、原始日志 hash 见 [运行时证据](GpuStreamingDecompressionRuntimeEvidence.json)。

本轮修正了 RHI 合成测试的混合 tile 输入：先前在 Bunny 有效几何区内写随机字节以制造 Raw tile，破坏了索引；现在使用独立合法三角形，随机数据只放在第二 tile 的 padding。该输入同时用于原生 GPU oracle 和引擎 Streamer 测试。另把 staging 字节与 GPU 安装页计数接入逐帧回放报告，并修复失败测试在 Streamline teardown 中等待超时的问题。

- **5 项独立测试通过**：含 1,024 页 CPU oracle、65 页 GPU oracle、混合 tile/短尾块和固定页输出比较。
- **2 项 RHI 集成测试通过**：实际 Streamer 上传/EXT 解压、逐字节比较、取消 recording、完成 receipt 和重复 frame-slot 复用；不再跳过。
- **MiniZorah 3000 帧实时质量回放通过**：DLSS-SR、CLAS 和 streaming 开启，1920×1080 输出、1280×720 内部渲染。15 个检查点的安全 cut/预算断言通过；第 29 帧仍在细化，第 59 帧起各采样点无可见超目标细化。最终 29,367 个 GPU 安装页完成，加载失败/非法请求为 0，pending 为 0。
- 上述正确性测试开启 Vulkan validation 与同步校验，日志中无 validation error。质量回放另有关闭 validation 的独立通过记录。它们是检查点 oracle，不能等同于逐帧所有像素或实际 draw 消费时刻的证明。

系统原 SDK 1.4.341 验证层不认识新的地址命令，因此本轮只在 `.cache/fast-streaming/vvl` 构建验证工具，通过进程级 `VK_LAYER_PATH` 使用。SDK 1.4.357 原版在质量检查点发生 heap 保留区间跟踪的悬空指针崩溃，CPU/GPU 两种模式均可复现。回移 Khronos 的 [f6ff981 修复](https://github.com/KhronosGroup/Vulkan-ValidationLayers/commit/f6ff981ec58c04e1f71ae06f818488d933452ac7) 后质量回放通过；未关闭该检查，也未修改系统 SDK 或工程 `External/`。完整回放的验证工具版本明确为 **1.4.357 + f6ff981**；两项 RHI 与五项独立测试在未回移修复的 1.4.357 上也已通过。

性能采用 **CPU → GPU → GPU → CPU** 串行四轮，每轮相同 v10 压缩资产和参考相机脚本、相同 LOD/驻留/上传预算、3000 帧，不开 validation。每组两次结果如下；GPU 帧分布排除前 300 帧预热。

| 指标 | CPU GDeflate 解码 | EXT GPU 解码 |
| --- | ---: | ---: |
| GPU 整帧 P50，ms | 8.042 / 8.117 | 8.478 / 8.634 |
| GPU 整帧 P95，ms | 9.181 / 9.230 | 9.774 / 9.870 |
| GPU 整帧 P99，ms | 9.762 / 9.694 | 10.284 / 10.374 |
| 每页 CPU 加载处理平均耗时，ms | 0.1441 / 0.1434 | 0.0192 / 0.0211 |
| 累计 host staging 字节 | 768,705,648 / 768,586,240 | 590,091,684 / 589,985,992 |
| 完成安装页数 | 29,610 / 29,608 | 29,615 / 29,611 |
| 有上传帧的 Stream Begin GPU 平均耗时，ms | 0.0075 / 0.0079 | 0.7635 / 0.7705 |

两轮平均后，CPU 加载处理耗时约降 **86.0%**，host staging 字节约降 **23.2%**，但 GPU 帧 P95 约增 **6.7%**。Stream Begin 额外 GPU 开销约 0.76 ms/上传帧，符合小批次同队列解压成本，仍需细分 copy/decode/barrier 才能归因到具体阶段。固定相机不等于固定页序列，因此没有用不同页数推导每字节吞吐。

这些是 OS 文件缓存已暖的交替回放；重启进程没有清空文件缓存，不能作为 SSD 吞吐结果。保留每轮 GPU/竞争进程记录和二进制/回放路线/Shader hash。桌面上的 Unity、DWM 等其他应用仍在运行；证据中保留各轮后台引擎采样，表中差异是本次观测，不能作为 GPU 独占条件下的性能保证。测试都在结果落盘和 GoogleTest 结束后遇到既有 Streamline teardown 停留，由回放工具 8 秒后清理自己的进程，`Process.json` 明确记录，不宣称正常进程退出。首次 DLSS 初始化约十余秒，不纳入稳态帧分布；旧 latency 的 `*Drawable` 终点仍为 CPU 驻留完成，不能宣称实际首次绘制或首屏墙钟延迟改善。

## 初始驱动 610.47 的历史验证

GPU 为 RTX 5060，驱动 610.47，Vulkan 1.4.341。独立 `MetallicGpuPageTests` 直接建立只要求 EXT 解压、BDA 和同步能力的 Vulkan 设备；这不替代引擎集成测试。

已完成的测试：

- Raw/GDeflate 离线转换、CPU 逐字节 round trip、完整测试资产拓扑保持、CLAS sideband 与 CPU 计划一致、损坏/截断包拒绝、已有输出拒绝覆盖。
- PageLoader 两个 worker 的 CPU 回退/GPU sideband 分支、无效页请求传播和重初始化。
- 真正的 EXT GPU 解码：三 tile 混合 Raw/GDeflate、64 KiB 边界与短尾块、重复复用缓冲区、精确同步阶段和访问掩码。GPU 与 CPU 逐字节一致。
- MiniZorah：全部 page 目录的逻辑字段与紧凑 device size 比较；完整 group/node/refined-group 表比较；均匀选取含首尾的 1,024 页进行 CPU/CLAS oracle，其中 65 页实际执行 GPU 安装（Raw copy 或 EXT 解码）。
- Vulkan Validation 与 synchronization validation 开启，以上 GPU 测试没有 validation error。

五项独立测试（含下面的固定页基准）和先前三项既有 SceneImport 流文件/压缩/恢复测试通过。原始结果：[独立测试复测日志](../.cache/fast-streaming/gpu-page-tests-recheck-final.log)、[SceneImport 回归日志](../.cache/fast-streaming/scene-tests.log)、[RHI 集成复测日志](../.cache/fast-streaming/rhi-recheck-final.log)。集成测试用例还覆盖上传 receipt、取消 recording 和 frame-slot 复用，但本机执行被设备创建阻断，两项均跳过，不能计作通过；GoogleTest 在全部跳过时仍返回 0，不能只检查进程返回码。

当前分支的 `87735db` 已将 `VK_KHR_device_address_commands` 设为引擎硬要求。2026-09-18 驱动升级前复核 610.47：禁用所有隐式层的 Loader 查询，以及绕过 Loader 直连 `nvoglv64.dll` 的 ICD 查询，都返回 280 个设备扩展，目标扩展未列出，`deviceAddressCommands=false`。这描述的是当时驱动暴露的能力，不是 RTX 5060 硬件不支持。两份原始结果为 [Loader 查询](../.cache/fast-streaming/address-loader.log)、[原生 ICD 查询](../.cache/fast-streaming/address-direct-icd.log)；路径与 hash 见 [复核证据](GpuStreamingDecompressionFollowupEvidence.json)。[NVIDIA 官方记录](https://developer.nvidia.com/vulkan-driver) 已在 Windows 595.92 Vulkan beta（2026-03-13）中列出该扩展，不能仅凭驱动版本数字大小判断当前分支包含该能力。

因此当时 `MetallicRhiTests` 在创建设备时跳过用例，尚未验证 MiniZorah 完整实时回放。上述历史结果没有被记作集成通过；616.92 的新结果单独记录在前节。

## 固定页安装基准与校验优化

新增可选 `GpuPageCodec.FixedPageUploadBenchmark`：从同一 MiniZorah v10 文件均匀选择 128 页（含首尾），固定页 ID、输出位置和最终字节。Raw 控制组在计时前生成相同布局的 Raw 封装；三组都保留运行时校验。输入预读到 CPU 内存，每组 4 次预热、32 次采样，完整重复 3 轮；数据包含全部页 ID、P50/P95 以及原始采样文件 hash。性能测量关闭验证层，另行开启同步校验运行并核对三组 GPU 输出。

单批最终输出 4,346,336 字节；CPU 路径 GPU copy 同量，EXT 路径 GPU copy 的 tile 数据为 3,072,976 字节（减少 29.30%）。这来自命令记录的 payload 字节，并非 PCIe 硬件计数器；Host staging 还包含 sideband 和对齐。复用相同的预录制命令，GPU query 分别测量复制、解压加 barrier 和总安装时间，正确性 readback 排除在计时之外。

该基准发现 CPU 校验串行逐字节 CRC32C 的开销，现改为可移植的 slicing-by-eight，保持相同多项式、初值、结尾和存储 checksum。独立 bitwise oracle 验证新编码 checksum；先前生成的 MiniZorah 文件继续通过 CPU/GPU 抽样测试，无需重新转码。

以下是关闭验证层的 3 轮 P50 中位数，单位 ms：

| 路径 | 优化前 CPU 准备 | 优化后 CPU 准备 | GPU 安装 | CPU 准备至提交完成等待 |
| --- | ---: | ---: | ---: | ---: |
| Raw 封装、CPU 校验并上传 | 6.514 | 2.256 | 0.336 | 2.716 |
| GDeflate、CPU 解码并上传 | 24.292 | 20.776 | 0.335 | 21.324 |
| GDeflate、EXT GPU 解码 | 4.433 | 1.384 | 1.532 | 3.052 |

GPU 路径的 CPU 准备时间降低约 68.8%；其 GPU 复制约 0.262 ms、解压及 barrier 约 1.270 ms。这也表明解压有实际 GPU 成本，不能只看传输字节减少。该基准是暖内存、单 CPU 线程、同 GPU 队列的隔离测量，不含磁盘缺页、多个 worker、命令录制、CLAS、驻留发布或渲染，不代表完整 Streamer/实时回放性能。仍需完整回放决定默认策略。

复测命令（输出目录需已存在）：

```powershell
$env:METALLIC_GPU_PAGE_BENCHMARK = 'E:/metallic/.cache/fast-streaming/fixed-pages.json'
$env:METALLIC_GPU_PAGE_NO_VALIDATION = '1'
build-release/tests/MetallicGpuPageTests.exe --gtest_filter=GpuPageCodec.FixedPageUploadBenchmark
# 正确性复测移除 METALLIC_GPU_PAGE_NO_VALIDATION，并设置 VK_LAYER_VALIDATE_SYNC=1。
```

## MiniZorah 转码结果

完整结果及目录 hash 见 [结构化证据](GpuStreamingDecompressionImplementationEvidence.json)。原资产未修改，新文件位于本地 `.cache/fast-streaming/MiniZorah.gdeflate.meshstream.bin`，未纳入版本控制。

| 指标 | 原 v9 | 新 v10 |
| --- | ---: | ---: |
| 文件字节 | 60,916,791,801 | 35,709,843,144 |
| 存储 payload 字节 | 60,494,871,392 | 35,279,798,868 |
| 最终紧凑 GPU payload 字节 | 48,681,578,880 | 48,681,578,880 |
| 页数 | 1,356,959 | 1,356,959 |
| cluster 数 | 33,020,491 | 33,020,491 |

文件缩小 **41.38%**；新 payload 含传输封装和 sideband，相对现有 float3 紧凑 device payload 缩小 **27.53%**。这是容量结果，最终几何驻留占用保持不变。

## 使用与复测

在已设置 MSVC 环境的终端构建：

```powershell
cmake -S . -B build-release -DMETALLIC_BUILD_TESTS=ON
cmake --build build-release --target MetallicGpuPageTests MetallicMeshletTranscode MetallicGPUDrivenSample MetallicRhiTests
build-release/Source/MetallicMeshletTranscode.exe input.meshstream.bin output.meshstream.bin
# --raw 写相同的 GPU-ready 封装，但所有 tile 使用 Raw。

$env:METALLIC_TEST_MINIZORAH = '1'
$env:VK_LAYER_VALIDATE_SYNC = '1'
# 本轮完整回放使用 1.4.357 + f6ff981；此路径仅用于本地测试工具。
$env:VK_LAYER_PATH = 'E:/metallic/.cache/fast-streaming/vvl/install/bin'
build-release/tests/MetallicGpuPageTests.exe
build-release/tests/MetallicRhiTests.exe --filter streamer_gpu_ --rhi-validation
```

离线依赖可通过 `-DFETCHCONTENT_SOURCE_DIR_METALLIC_GDEFLATE=<解压后的固定版本目录>` 提供。不设置 `METALLIC_TEST_MINIZORAH` 时，大资产抽样用例跳过；CPU 与合成 GPU 测试不依赖 MiniZorah。

运行时将 VBuffer/GPUDriven pass 的 `streamAssetPath` 指向新文件，设置 `enableGpuDecompression: true`。不支持 EXT/GDeflate 的设备使用 CPU 回退；此回退仍需满足引擎本身的设备要求。`completionDrivenUploads` 必须启用。设置 `enableGpuDecompression: false` 可在同一文件上测试 CPU 解码。

```powershell
tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json `
  -OutputRoot .cache/fast-streaming/replay-gpu -Realtime `
  -StreamAsset .cache/fast-streaming/MiniZorah.gdeflate.meshstream.bin -GpuDecompression On
# 使用新的 OutputRoot 和 -GpuDecompression Off 得到 CPU 对照。
```

调试报告增加 `gpuDecompressionEnabled`、`frameStoredUploadBytes`/`totalStoredUploadBytes`、`frameGpuDecompressedPages`/`totalGpuDecompressedPages`。原 `frameUploadBytes`/`totalUploadBytes` 继续表示解压后字节预算；Stored 计数表示 Host staging 的封装字节，含 CPU sideband，不能直接当作 PCIe 计数；GPU 页计数表示被接纳的 GPU 安装页，包含全 Raw 页，不表示已完成或已绘制。

## 后续边界

第一阶段的集成正确性和固定相机 A/B 已完成，当前结果决定保留 opt-in。下一阶段应首先细分传输/解压 GPU timestamp，并针对小批次解压成本比较合批、CPU/GPU 分流及异步队列；同时补充真实传输字节和 CLAS/实际 drawable 端点，避免仅以后台 CPU 时间决定默认策略。`VK_KHR_copy_memory_indirect` 尚未接入，继续遵守阶段 3 的性能验收门槛，不因硬件可用就扩大本阶段范围。

API 依据：[buffer usage2](https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferUsageFlags2CreateInfo.html)、[EXT 解压批次与 64 KiB 限制](https://docs.vulkan.org/refpages/latest/refpages/source/VkDecompressMemoryInfoEXT.html)。
