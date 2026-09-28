# ZorahFull NTC 显存可行性调研（2026-09-28）

## 结论

NTC 可以进一步减少材质纹理驻留，但不能压缩几何、CLAS/BLAS/TLAS、渲染目标或 DLSS 工作区。当前仓库 Full 默认纹理预算仅 512 MiB，因此不能将约 14 GB 整卡峰值乘以 NTC 的纹理压缩率。短期优先补齐峰值同帧内存归因；NTC 更适合以相同预算提高纹理清晰度，或在实际纹理工作集已达数 GiB 时降低总显存。

本次完成代码检查、官方 SDK 调研和本地 KTX2 header/level-index 探针；未修改渲染实现，未训练 Zorah NTC、未进行 NTC GPU 性能或画质验收，也未重新复现用户报告的 14 GB 峰值。

## 当前数据与实现

本次直接读取 `Asset/ZorahFull/zorah_textured_public.v1.gltf` 与引用的 KTX2 文件头，结果见 `build-release/ZorahFullNtcProbe20260928.json`：

| 指标 | 结果 |
|---|---:|
| 材质 / 引用纹理 | 1514 / 4418 |
| NTC 图片 | 0 |
| BC7 sRGB / BC5 / BC4 | 1583 / 2774 / 61 |
| MASK 基色图片 / texture transforms | 110 / 501 |
| KTX2 磁盘文件合计 | 47.258 GiB |
| 全部 BC mip payload | 85.024 GiB |
| 普通图 128、MASK 512 的基础尾链 payload | 106.560 MiB |
| 全部图片裁到最长边 512 的尾链 payload | 1390.712 MiB |

这里的 payload 是 KTX2 Zstd 解压后的 BC 数据，不是 Vulkan 实际 allocation，也不是当前运行时工作集。全部 512 的结果不能解释为当前已驻留 1.39 GiB。

当前 `Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json` 的三个消费者采用同一策略：初始 128、细化上限 512、MASK 512、冷却 180 帧、普通材质 image allocation 预算 512 MiB。`ScenePathTraceResources.cpp` 还检查共享设备预算并计算旧新 image 共存；驱动/VMA 保留块与 payload 必须分开。运行时编辑器覆盖可能不同，尚不能用预设证明用户峰值时确实是 512 MiB。

已有 NTC 基础：

- `Source/Runtime/Render/NeuralTextureResources.cpp/.h`：LibNTC 元数据读取、latent array、FP8/INT8 权重和 GPU 权重转换；目前按场景准备，最多 64 sets。
- `Shaders/Interop/NeuralTextures.slang`：CoopVec FP8 / Generic INT8 推理；一次推理生成整组输出，但现有 helper 只返回指定四通道，不能直接为每个材质贴图重复调用，否则可能重复解码同一组。
- `Documentation/NeuralTextureCompression.md`：已接通材质可视化、path tracing、RTXDI；其 FlightHelmet 的 95.8% 历史节省是相对 RGBA8，不能迁移为 Full 相对 BC 的收益。GPUDriven 独立采样路径尚未接入。
- 当前 `build-release/CMakeCache.txt` 的 `METALLIC_ENABLE_NTC=FALSE`。只改变 glTF 引用无法启用这一构建的 NTC。
- 本地 `E:/RTXNTC` 与本次官方 main 显示 SDK 0.10.0 BETA；实际接入必须锁定库、shader 和 cook 版本。官方 changelog 记录了早期格式破坏性变更，旧部分 latent 装载接口也已移除，不能假设 KTX2 尾链替换可直接套用到 NTC。[官方变更记录](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/ChangeLog.md)

## 三种路径

| 路径 | 最终主要驻留 | 对当前目标的价值 |
|---|---|---|
| Inference on Load | 解码/转码后的 BCn | 相同 mip/格式通常不减少相对 BC 的稳态显存；加载阶段还有临时资源。适合减小包体与传输数据 |
| Inference on Sample | latent + 权重，着色时解码 | 真正降低纹理驻留；增加 shader 计算与过滤改造成本；建议先试点 |
| Inference on Feedback | NTC 数据 + 按需 BC tile cache + scratch | 以虚拟纹理缩小可见工作集，适合后续高分辨率；工程量更大 |

官方示例中同尺寸材质 bundle 从 24 bpp BC 降到约 5 bpp NTC，约 4.8 倍；这是特定材质质量条件下的例子，不能视为 Full 的保证。on-load 回到等价 BC 后显存仍是 BC 的大小。[官方模式与占用示例](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/README.md)

on-sample 一次只产生一个未过滤 texel。直接模拟多次双线性/三线性/各向异性采样会增加推理次数，SDK 建议 STF 与时域滤波配合；已有 DLSS Quality 只是有利基础，不等于遮挡显露、镜头移动和高光已经合格。5070 Ti 的现有 CoopVec 基础可复用，仍须实际确认 capability、权重布局和 Slang 编译路径。[官方 on-sample 集成](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/integration/InferenceOnSample.md)

官方 on-feedback 示例当前依赖 DX12 Sampler Feedback/Tiled Resources，没有可直接搬入 Metallic Vulkan 的完整实现。可以自行做 shader tile feedback 与 sparse image/软件 atlas，但现有“每图片 mip 需求”反馈还缺 tile 坐标；它是独立虚拟纹理项目。示例还会将 NTC 源数据驻留显存，不能忽略这部分成本。[官方 Renderer 与 on-feedback 说明](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/Renderer.md)

## 收益范围与主要风险

估算使用：`净节省 = 被替换 BC 实际 allocation - latent/weights 实际 allocation - 保留 BC - 新增常驻/迁移开销`。总显存还受分配器保留块影响。

假设被替换纹理全部达到 4 倍压缩，暂忽略固定开销：

| 实际可替换纹理 | 理想化净节省 |
|---|---:|
| 512 MiB | 384 MiB |
| 2 GiB | 1.5 GiB |
| 4 GiB | 3 GiB |

这些是假设算例，不是实测预测。若当前确实使用 512 MiB 预算，即便将它全部替换，也不能解释数 GiB 的整卡下降。反之，把全量 85 GiB BC 转成 4 倍压缩后一次性加载，仍约 21 GiB，远超过 16 GiB 显卡；NTC 仍然需要预算、按需装载和回收。不要拿全分辨率 NTC 与当前 128/512 BC 混作等画质对照。

关键风险：

1. **吞吐目标。** 当前追求 Full 漫游至少 30 fps；NTC 增加 Deferred 中的推理、寄存器和数据读取成本。与 DLSS 并存后的净开销必须测量，不能用 SDK 简单场景帧率推导。
2. **一次 bundle 推理服务多个通道。** 按同一 set、UV、mip、随机过滤采样点共享输出；不同 texture transform/分辨率/采样器不能强行共享一次结果。501 个 transform 必须逐材质检查，源图复用也要避免被 bundle cook 复制放大。
3. **覆盖与阴影。** MASK/BLEND opacity 和 displacement 保持传统 BC；避免每次阴影 alpha 测试解码全套 PBR。独立 alpha BC4 方案须验收覆盖率与 mip；首版可直接保留现有固定 BC 图。SDK 也建议对 alpha 特殊处理，神经误差经过 cutoff 会放大。[质量与 alpha 指南](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/SettingsAndQuality.md)
4. **资产保真。** 当前来源已经 BC 有损；再训练只能逼近该来源，不能恢复原始细节。有无损源优先用无损源；没有时保留 BC 解码图作为比较基准。保留一次 sRGB 转换、KTX swizzle、BC5 法线重建、normal scale、specular 语义和稳定 TBN。
5. **预算遗漏。** 当前神经资源独立统计，不能仅跳过 conventional image 然后声称预算满足。latent、权重、转换 scratch、staging、在途新版本和延迟退休旧版本都应纳入 Streamer 的统一预算和实际 allocation 统计。

## 推荐推进顺序

### N0：峰值归因与代表材质 cook

先在用户实际配置下记录同帧的 texture resident/pending/retired、geometry used/capacity、CLAS used/reserved/retiring、BLAS/TLAS/scratch、RenderGraph transient/persistent、DLSS/NRD、upload/staging，以及设备 heap usage 与 NVML 整卡占用。记录峰值所在阶段：加载、细化、漫游或切场景。不要把各项独立最大值相加。

选择 16–32 个按实际驻留字节排序的材质，覆盖石墙、金属、复杂法线、叶片及不同 UV transform。保留 512 BC 的等分辨率对照，再测 2K/4K 的等质量预算潜力；试几个 SDK 支持的 bpp/质量档。输出逐通道误差、法线角误差、粗糙度/高光对照、cook 时长和完整 GPU 分配估算。此阶段不做全量 Full 转码。

### N1：Deferred opaque on-sample 小试点

在独立兼容构建中启用既有 NTC，而非更改现有性能基准的 SDK 配置。复用 FP8 解码基础，以 bundle 为单位共享推理输出，接入正确 mip/UV 足迹及 STF。只替换选定 opaque 材质；保留 BC fallback 与固定覆盖图。固定相机、cut、驻留及 shader 功能对照 BC/NTC，再运行自然漫游；分别记录 Deferred、DLSS、整帧 P95/P99、峰值 heap 与画质。

建议项目门槛（待测，不是 SDK 保证）：代表集等质量净纹理 allocation 至少减少 60%，Deferred+DLSS 额外 GPU 时间不超过约 1 ms，且漫游 P95 仍不超 33.3 ms；出现明显噪点、法线/alpha 失真或时域拖影即保留 BC。最终阈值按项目实际余量调整。

### N2：Streamer 管理 NTC set 驻留

突破 64-set 固定上限，建立稳定逻辑 set ID 与可回收物理槽位；由 StreamerSubsystem 负责 IO、准入、冷回收、权重转换及 completion 后代际发布。RenderPass 只消费已发布资源和记录采样需求。先做按 set 的需求驻留；多分辨率 set 或 latent 分片必须另行验证 SDK 坐标/网络依赖，不宣称现成支持任意 sparse latent mip。

统一 BC/NTC 预算；切换期间计算旧新共存，消除正式运行时为了对比而双份常驻。unsupported/失败只保留或恢复 BC 路径，取消提交不得发布新 set。

### N3：高精度虚拟纹理（按结果决定）

如果等质量 NTC 的算力成本不合适，但高分辨率纹理工作集确实是主导，再评估 BC tile cache + NTC on-feedback。先用现有 BC 资产验证 Vulkan tile feedback/cache，分离“虚拟纹理工作集缩小”与“NTC 编码”各自的收益，再接入按 tile 转码。

短期决策：可以推进 N0/N1；若同帧统计确认纹理仍只占几百 MiB，要降低当前 14 GB 峰值，应优先处理最大 allocation owner 的容量、在途副本和退休生命周期，而非立即全场景 NTC 化。
