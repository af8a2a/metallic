# ZorahFull U0＋U1：上传观测与读取器复用

2026-09-20。已实现并构建 Release `MetallicGPUDrivenSample`，完成含 Vulkan validation 的 4 项纹理/上传回归。保持 512 cap、BC 格式、mip 内容、128-region/64 MiB 批次限制与最多 3 批在途不变。

## 实现

- `Ktx2MipReader` 每张纹理打开一次 payload 文件，跨 mip/纹理复用 Zstd context 和压缩输入 scratch。上传路径另复用解码输出 vector；保留原来的单 mip API 作为兼容包装器。reader 为单 worker 所有，不可并发共享。
- 记录 header 探测、预算规划、文件打开/读取、解码、image/view 创建、staging 分配/拷贝/flush、command 对象准备、录制、copy/acquire 提交、在途回压与最后等待。输出累计 payload 字节、mip/reader/context 数和最慢 5 张纹理。
- copy/acquire 记录从提交到首次观察完成的主机时延。**这不是 GPU timestamp 耗时**，包括调度、轮询延迟；其累计值可能重叠，不能与 CPU 阶段直接相加。完成样本和回收采用同一次 timeline 观察，避免漏样本。
- Info 日志每秒输出上传进度，完成时输出汇总；逐 batch 的原始日志降为 Debug。`SceneUploadStats` 和测试 JSON 同时提供计数与阶段数据。
- pending bytes/regions 改为增量维护；录制、barrier 和 staging 引用释放只访问该批 pending texture 索引，不再反复扫描全部已上传纹理。沿用原先完成后回收 staging 的生命周期。

代码：[KTX reader](E:/metallic/Source/Runtime/Render/Streamer/Ktx2Texture.cpp)、[上传管线](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp)、[统计接口](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.h)。`buildMs` 及 image/staging 子阶段当前统计 KTX 分支；header/plan、提交和完成统计覆盖该次资源准备。`wallMs` 从材质准备开始到完成发布，普通非流式场景可能包含其他资源准备/等待，不应当作纯纹理 DMA 时间。

## Full 实测

最终记录：[zorah-textures.json](E:/metallic/build-release/zorah-upload-u01/final/zorah-textures.json)、[tests.json](E:/metallic/build-release/zorah-upload-u01/final/tests.json)、[final.log](E:/metallic/build-release/zorah-upload-u01/final.log)。纹理专项读取 metadata 和材质资源，不加载 Full geometry/CLAS，也不启用 DLSS。

| 项目 | 最终运行 |
| --- | ---: |
| KTX 图像 / 含兜底 image | 4418 / 4419 |
| payload 文件打开 / mip 解码 / Zstd context 创建 | 4418 / 44140 / 1 |
| 所选压缩 payload 读取 | 963761960 B |
| 解码 payload，不含兜底 4 B | 1458267384 B |
| image allocation | 1469821440 B，约 1.369 GiB |
| staging 峰值 | 128 MiB |
| 上传批次 / copy 完成样本 / acquire 完成样本 | 369 / 369 / 369 |
| 材质资源准备 wall | 6379.48 ms |
| header / plan | 1998.80 / 12.32 ms |
| payload open / read / decode | 128.23 / 2144.31 / 1141.77 ms |
| image/view 创建 | 603.02 ms |
| staging allocate / memcpy / flush | 0.94 / 54.45 / 0.74 ms |
| command setup / record | 13.82 / 145.53 ms |
| copy / acquire submit API wall | 20.97 / 10.96 ms |
| 在途回压 / 最后等待 | 0 / 4.16 ms |

U1 已将旧代码每 mip 打开/context 创建的 44140 次结构性工作量，分别降到 4418 和 1。mip 数、上传字节、材质索引、实际 image 分配和批数保持一致，没有通过降低纹理质量取得结果。

修改前同机纹理测试主体为 10.179 s；最终测试主体为 8.959 s，其中资源准备为上述 6.379 s，其余包括 metadata 与 GPU 采样等。初次修改后另两次资源准备为约 6.806 s。文件缓存/系统负载未受控，前后不能作为严格性能倍率；用户先前约 61.5 s 的交互运行也不与本次条件等价。

新计时表明，本次 header＋payload read＋decode 约占资源准备 83%，在途批次无观测回压。下一步优先评估有界并行 header/读取/解码；U2 合批仍可减少提交工作，但不应把 369→44 批直接换算成整体 8 倍加速。

## 验证与边界

通过 `ktx2_texture_resources`、`bc_texture_padded_upload`、`scene_upload_pipeline`、`zorah_texture_resources`，均未跳过。覆盖 BC4/5/7、全部合成 mip 字节、NPOT/mip rebasing、swizzle/sRGB、>255 描述符、共享资源/预算失效、读取器 reopen、越界 mip、关闭/失败打开、header 检查后文件截断、损坏 Zstd 帧及恢复、raw payload，以及异步上传/取消与 GPU barrier。Full 核对全部 44140 mip 计数、字节总量、4418 个纹理索引和末尾 descriptor 的 GPU 采样；不宣称 Full 每个像素或每 mip 全量 GPU 回读校验。

```powershell
$env:METALLIC_ZORAH_Z3_FULL='1'
.\build-release\tests\MetallicRhiTests.exe --rhi-bindless --rhi-validation '--gtest_filter=*ktx2_texture_resources:*bc_texture_padded_upload:*scene_upload_pipeline:*zorah_texture_resources' --output-dir build-release/zorah-upload-u01/final '--gtest_output=json:build-release/zorah-upload-u01/final/tests.json'
```

证据目录中的 `baseline` 为修改前记录；`after` 首轮报告文件因输出目录不存在而未写出，但日志及测试成功结果保留。已修正 Full 测试创建输出目录并检查写入状态；`verified` 用于核对字段时发现完成样本计数窗口问题，最终修复后的验收以 `final` 为准。

本轮没有实施 U2 改批大小、U3 多 worker/直接写 staging，也没有修复 DLSS 的显存警告映射；Full 交互加载与漫游仍需后续独立验收。
