# ZorahFull 纹理有界并行验收

2026-09-20。在 U0＋U1 基础上完成并行 header 探测、KTX2 mip tail 预取与顺序上传。默认 4 个工作线程；`materialTextureLoadWorkers` 可设置为 1–8，仅影响加载调度，不改变资源缓存身份或材质质量。GPUDrivenSample Release 已重新构建。

## 实现与边界

- Header 按引用 image 去重后并行读取；主线程汇总错误，再执行原有 GPU allocation 查询和 mip 预算选择。
- 每个预取线程持有独立 reader/Zstd context，一张纹理只打开一次 payload，按选定 mip 直接解码到该任务独占的普通内存。
- 按逻辑纹理顺序领取任务，最多保留 `2 × workers` 个运行中或已完成任务。领取前扣除字节额度，额度覆盖完整 decoded tail 加最大单 mip compressed scratch，总限额 64 MiB。主线程复制到 staging 并释放结果后才归还额度，避免后续已完成任务阻塞更早任务的调度。
- 压缩 scratch 预留到所需最大值，任务结束时释放，Zstd context 保留复用。额度不包括线程栈、文件流、解码器和资源元数据，不能解释为进程 RSS 上限。超过 64 MiB 的单张尾链沿用原有串行路径，不进入预取；本次 Full 512 cap 没有触发该路径。
- GPU image/view 创建、staging 分配/flush、逻辑描述符发布、copy/acquire 提交均留在主线程。保持 64 MiB/128 regions/3 批的上传限制；没有将共享 arena 或 command pool 交给工作线程。
- 异步 pump 遇到尚未就绪的结果会返回；同步 prepare 可以阻塞等待条件变量。取消先停止任务领取，在 mip 边界检查停止请求，join 后释放任务数据，再按原规则等待已提交 GPU 工作并清理资源。正在执行的文件读取不能被即时中断。

本次保留 vector 解码加 staging memcpy；直接解码到 staging、批次参数调整及命令对象复用仍是独立候选。Header 探测仍在 begin 阶段等待完成，尚未完成 U4 的全程交互式加载改造。

## Full 基准

固定 Full glTF、4418 张纹理、44140 mip、512 cap、2 GiB GPU texture budget、Vulkan validation。按 1/4/2/8/8/2/4/1 和 1/4/2/8/1/4 顺序启动独立测试进程。首轮 1 线程保留为预热记录（5568.80 ms），以下汇总后续 13 轮。没有清空系统文件缓存或锁定 GPU 时钟；这是缓存预热后的本机纹理资源阶段测量，不是严格冷启动或完整场景首帧。

| 工作线程 | 测量轮数 | wall 中位数 / 范围 ms | Header 中位数 ms | 等待解码中位数 ms | 预取峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 2346.40 / 2304.11–2373.81 | 186.61 | 1189.29 | 1.17 |
| 2 | 3 | 1325.10 / 1297.19–1505.54 | 102.05 | 263.37 | 2.30 |
| 4 | 4 | 1095.12 / 1000.48–1461.73 | 57.80 | 11.27 | 4.60 |
| 8 | 3 | 1028.13 / 1007.45–1272.21 | 33.75 | 0.19 | 9.17 |

4 线程相对同版本 1 线程中位数缩短约 **53.3%**。8 线程相对 4 线程约再缩短 6.1%，但增加线程和预取存量，因此默认保持 4。1 线程也使用流水线，不能将它称作原 U1 串行实现。

所有轮次的计数、GPU allocation、payload 及末尾描述符 GPU 采样一致：

- 文件打开 4418 次，解码 44140 mip，Zstd context 数等于相应线程数。
- 存储数据读取 963761960 B，解码 1458267384 B；加 fallback 后 resident payload 1458267388 B。
- GPU image allocation 1469821440 B；上传 369 批，copy/acquire 完成后发布。
- staging 峰值 192 MiB，达到已有 3 页容量；默认 4 线程额外预取峰值约 4.6 MiB，未靠放大上传容量取得收益。

完整回归单次 Full wall 为 1745.31 ms、header 486.71 ms；首次开发验证为 1797.41 ms。缓存状态对 header/读取影响明显，因此保留这些记录，不将旧用户日志的 61.53 秒或 U1 的 6379 ms 直接用于计算本轮加速比。

原始各轮 JSON、日志及汇总：[comparison.json](E:/metallic/build-release/zorah-upload-parallel/comparison.json)。4 线程的 `imageCreateMs` 中位数 627.00 ms、`recordMs` 162.97 ms；读取/解码等待已大幅减少，后续优先量化 image 创建长尾和 U2 的合并 mip copy/批次对象复用，而不是继续扩充工作线程。

## 验证

Release 构建 `MetallicGPUDrivenSample` 与 `MetallicRhiTests` 成功。4 项测试全部通过，日志未见 Vulkan validation error：

- `ktx2_texture_resources`：BC4/5/7 各 mip GPU 采样、NPOT、swizzle/sRGB、描述符顺序；1/2/4/8 线程逐 mip CPU 字节对照、字节及任务限额、生产者等待时销毁、坏帧后恢复；异步取消重载；加载末尾 payload 损坏时失败且不发布，清理后成功重载。
- `bc_texture_padded_upload`：BC 上传对齐和压力预算路径。
- `scene_upload_pipeline`：非 KTX 上传、取消及同步回归。
- `zorah_texture_resources`：真实 Full 的全部尾链计数、GPU 预算、描述符与最后 GPU 采样。

证据：[final-tests.json](E:/metallic/build-release/zorah-upload-parallel/final-tests.json)、[final.log](E:/metallic/build-release/zorah-upload-parallel/final.log)、[Full 资源结果](E:/metallic/build-release/zorah-upload-parallel/final/zorah-textures.json)。

复测单轮（测试环境变量只控制 Full 测试输入的 graph property）：

```powershell
$env:METALLIC_ZORAH_Z3_FULL = '1'
$env:METALLIC_KTX_LOAD_WORKERS = '4'
& build-release/tests/MetallicRhiTests.exe --rhi-bindless --rhi-validation `
  '--gtest_filter=*zorah_texture_resources' `
  --output-dir build-release/zorah-upload-parallel/recheck
```

新增 `prefetch` 峰值、worker 数和 `decodeWaitMs` 输出。`textureWallMs` 是资源阶段墙钟时间；`build/open/read/decode` 是累计工作耗时，工作线程之间及与 GPU 上传存在重叠，不能相加为加载墙钟时间。copy/acquire 仍为主机观察到的完成延迟，不是 GPU timestamp。

本轮没有重 cook 几何，也未验收 DLSS 开启的 Mini→Full 编辑器首帧；先前 `eWarnOutOfVRAM` 错误分级问题仍按交互加载计划处理。
