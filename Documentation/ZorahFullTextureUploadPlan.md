# ZorahFull 材质上传链优化方案

2026-09-20 后续：**有界并行准备已落地**，默认 4 worker、8 个在途/ready 任务、64 MiB CPU payload/scratch 限额。预热后同版本 1→4 worker 的 Full 纹理 wall 中位数 2346→1095 ms，验证和边界见 [有界并行验收](ZorahFullBoundedTextureUpload.md)。U2 批次组织及直接 staging 解码仍单独评估。

U0＋U1 已于 2026-09-20 完成：Full 实测 4418 次 payload 打开、44140 mip、1 次 Zstd context，4 项回归通过。实现、计时口径与证据见 [U0＋U1 验收](ZorahFullU01Upload.md)。本次主要时间在 header/读取/解码，后续优先评估有界并行准备；以下保留原始调研与候选方案。

2026-09-20。按用户最新优先级，先处理材质上传长尾。依据为用户原始日志、Metallic `4c46a2a20`、本地参考 `E:/vk_lod_clusters` 的 `1febfa7`，以及本轮读取全部 4418 张 KTX2 的 header/level index。没有重新解码全部纹理或执行 GPU 测速，没有修改运行时代码。

## 1. 固定 4 MiB / 120 regions 的来源

当前上传器同时限制每批 64 MiB、128 regions、最多 3 批在途。4418 张纹理中，4404 张在 512 cap 下保留 10 个 mip，其余 14 张更少，总共 44140 个 mip。每 mip 计一个 region，12 张常见纹理为 120 regions，再加一张会到 130，触发 `shouldFlushBefore()` 提前提交。

512×512、16 B/4×4 block 的 BC5/BC7 完整尾链是 349552 B；12 张恰好 **4194624 B**，与用户日志一致。128 是应用层阈值，不是这一 Vulkan 操作固定只能提交的 region 数。

staging arena 每页 64 MiB，提交时整页转给 batch，完成后才回收。常见批次只用约 4 MiB，却占用一页，使用率约 6.25%。这是预留容量利用率，不表示 DMA 会传输全部 64 MiB；也不表示每批都会重新分配页，现有 arena 会复用完成的页。

位置：[合批限制/计数](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1298)、[提前提交](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1410)、[staging 页](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:176)。

## 2. 上传日志能证明什么

| 指标 | 当前记录 |
| --- | ---: |
| 完成上传 | 369 批，1,459,358,444 B，平均 3.77 MiB/批 |
| 首末批提交跨度 | 59.886 s |
| 相邻提交间隔中位数 / 最大 | 32 ms / 1218 ms |
| ≥250 ms 的间隔 | 65 段，合计 44.536 s |
| 上述长间隔中，上一条日志 inFlight 少于 3 | 43 段 |
| 提交时 inFlight 为 1 / 2 / 3 | 222 / 97 / 50 批 |
| 所选 mip 的磁盘压缩数据 | 963,761,960 B，约 919 MiB |

`Submitted` 在 CPU 准备、录制和提交之后打印；`inFlight` 统计上传及 graphics acquire 尚未被回收的 batch，并非 copy 引擎利用率。提交间隔包含文件读取、解码、image 分配、拷贝、驱动调用、GPU/调度等待。以上数据不能分离每项耗时，也不能将约 60 秒直接当作 1.36 GiB 的 PCIe 传输耗时。大量长间隔发生在上一批计数未满时，说明仅增加在途批次数不能作为有证据的首选。

## 3. 源码确定的重复工作

**逐 mip 重开文件与创建解码器。** `buildMaterialTextureStep()` 一次处理一张纹理，`createKtxMaterialTexture()` 串行遍历 mip；每次 `decodeKtx2Mip()` 都打开 `ifstream`、创建 compressed vector、创建/释放 `ZSTD_DCtx`。本资产全部所选 44140 mip 为 Zstd：当前对应约 44140 次 payload 文件打开与 44140 次 decoder 创建，另有 header 探测。改为每张纹理一次 reader 后，payload 打开次数可降到 4418；decoder 可缩减为每个 worker 一份。[decode](E:/metallic/Source/Runtime/Render/Streamer/Ktx2Texture.cpp:204)、[调用](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:747)

**临时解码内存与第二次拷贝。** 当前先解压到普通 vector，再 memcpy 到已分配的 staging。可以给 decoder 一个有容量的目标 span，按已校验的 mip offset 直接写入 staging；同时复用压缩输入 scratch。须实测 HostUpload 的 CPU 写入属性：直接解码到映射内存未必在所有设备上快于缓存内存解码加 memcpy，因此保留两种路径作 A/B。[Zstd API](https://github.com/facebook/zstd/blob/dev/lib/zstd.h)

**每 mip 一条 copy，每批新建同步对象。** 当前 RHI `copyBufferToTexture()` 生成一个 region 的 `vkCmdCopyMemoryToImageKHR`。整场景约 44140 条纹理 copy；同一 image 的多 mip 可组成一次 multi-region copy，降到约 4418 条。每个独立 copy batch 还创建 upload/acquire 两套 pool+command buffer、一个 semaphore，分别提交 copy 与 graphics acquire。369 批意味着 738 次 queue submit，以及对应对象创建/销毁。它们均为结构性工作量，不是实测耗时占比。[当前 RHI](E:/metallic/Source/Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp:5758)、[Vulkan copy 规范](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyMemoryToImageKHR.html)

**重复扫描已完成纹理。** pending bytes/regions 在计数和 flush 判定时扫描累积的 `materialTextures`，录制与 barrier 同样扫全表。改成 pending texture 索引列表和增量计数，录制只访问该批资源；失败/取消路径统一撤销 pending，避免重复上传或漏上传。

## 4. 参考实现可借鉴的边界

本地 `scene_textures.cpp` 每个 worker 打开一次文件映射、读取选定尾链，使用 `parallel_batches` 分发纹理，并将 mip 直接解码到 uploader mapping。`AsyncUploader` 管理 staging 和提交；完成 copy 后集中处理 graphics acquire。它没有实现基于视图的纹理 mip streaming。

采用“每图一次 reader”即可先降低打开次数；不必首先改成映射整文件。映射不等于将全部文件读入 RAM，但热页/缺页代价仍要计入。只读取 KTX level index 指定的低 mip 范围，避免为 512 cap 读取 47 GiB 的完整纹理源数据。

参考：[每纹理 reader](E:/vk_lod_clusters/src/scene_textures.cpp:323)、[直接写 staging](E:/vk_lod_clusters/src/scene_textures.cpp:558)、[并行上传](E:/vk_lod_clusters/src/scene_textures.cpp:742)。

## 5. 建议实施顺序

| 阶段 | 改动 | 验证目标 |
| --- | --- | --- |
| U0：可归因基准 | 将 `alpha-test resources` 拆成 header/plan、file open/read、Zstd、image/view allocation、staging allocate/copy/flush、record/submit、copy completion、graphics acquire wait；逐图累计与最慢 Top N，按秒聚合日志 | 明确 61.5 s 的主导项；CPU wall 与 worker 累计时间分开，GPU 区间不能与重叠 CPU 区间直接相加 |
| U1：减少串行重复开销 | 每图一次 reader、每 worker 一个 Zstd context、复用 compressed scratch；pending 列表和计数；暂不改变 mip 选择和解码内容 | payload 打开约 44140→4418；每个 mip 解码字节哈希完全一致，截断/坏数据仍明确失败 |
| U2：批次组织和对象复用 | 在途仍为 3；先 A/B 512/1024 regions 与 16/32/64 MiB；复用完成后的 pool/command buffer/timeline；按 image 合并 mip copy 和 image barrier | 提交批次/CPU 录制时间显著下降；staging 总占用有硬上限；等待完成后才能 reset pool，不误复用在途资源 |
| U3：有界并行与流水线 | 2/4/8 worker 扫描；worker 持有独立 reader/context、独占目标 span；主提交线程管理共享资源表和队列；ready queue 及 staging 使用 byte credits；A/B 直接 staging 与 vector+memcpy | IO/解码与 GPU copy 重叠；CPU scratch 和 staging 总量有界；取消/失败时排空 worker、等待已提交 GPU 工作，再释放资源 |
| U4：交互加载与可选缓存 | 图准备调用已有异步资源接口，避免主线程无限 pump；进度/取消；若 U1–U3 后仍由文件读取主导，再做带内容签名的 mip-tail pack/cache | 加载期间 UI 持续响应；性能与响应性分别验收；不为优化纹理而重 cook 几何 |

建议第一批实施 **U0＋U1**，随后以独立开关测试 U2，避免把全部改动混在一起失去归因。若 U0 已证明 GPU 等待/驱动分配占主导，应调整 U2/U3 顺序，不强行扩大 CPU 并行。

U2 的 header-only 模拟如下。保持 512 cap、原纹理顺序、64 MiB 字节上限；按常见日志推断的 16 B offset 对齐。只包含纹理，排除 fallback 和材质 buffer，不是 GPU 测速：

| region 上限 | 模拟纹理批数 | 平均每批 |
| --- | ---: | ---: |
| 128（当前） | 368 | 3.78 MiB |
| 512 | 87 | 15.99 MiB |
| 1024 | 44 | 31.61 MiB |
| 2048 | 23 | 60.47 MiB |

当前真实日志 369 批与纹理模拟 368 批口径不同，真实日志含兜底与材质 buffer。**推荐先测 1024 regions / 32 MiB**，预期工作量约 45 批级别；批数约 8 倍下降不等于加载速度提高 8 倍。增加 region/byte 上限后应增加短时间 flush 条件，避免慢 IO 或最后一批迟迟不提交。时间预算只约束合批等待，不能打断未完成的单 mip 解码。

初版可以继续让 staging 跟随 graphics acquire 完成后回收，保持现有保守生命周期。将 staging 回收提前到 copy completion、集中 graphics acquire 属于后续优化，须分别追踪 copy 完成和纹理可供采样两个状态，不能只删等待。并行时也不能直接让多个 worker 无锁访问现有 arena、materialTextures 或同一个 command pool。

## 6. 验收矩阵

优先使用已有 `ktx2_texture_resources` 与 Full 材质资源测试，隔离 geometry/CLAS/DLSS 开销。固定 4418 张纹理、512 cap、相同格式/swizzle/sRGB 和 GPU image 分配；不靠降低 mip 或扩大显存预算取得加载加速。

每个候选配置测多轮热缓存和单独标注的冷条件：报告阶段 wall time、文件数/读取字节、解码数、批次/regions、copy/acquire 时延 P50/P95/max、CPU 与 staging 峰值。若没有控制系统文件缓存，称“首次运行”，不称严格冷启动。并行 worker 累计耗时与整阶段 wall time分别报告。随后再跑真实 Mini→Full 交互切换验证生命周期。

质量检查覆盖 BC4/5/7、NPOT/小 mip、全部 mip 内容、swizzle/sRGB、MASK/玻璃；错误检查覆盖文件截断、解码失败、取消、resize、加载后立刻切场景。最终首帧耗时与 UI 最长无响应间隔分别记录；DLSS 的预算警告问题仍需独立修复。

本轮分析工具及输出：[analyze_upload.py](E:/metallic/build-release/zorah-full-load-analysis/analyze_upload.py)、[upload-analysis.json](E:/metallic/build-release/zorah-full-load-analysis/upload-analysis.json)。读取 header/level index 约 3 秒，无 GPU 负载；不能将此时间解释成纹理解码或上传性能。
