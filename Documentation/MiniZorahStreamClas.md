# MiniZorah 流式 CLAS

2026-09-13。MiniZorah VBuffer 样例现在默认构建驻留几何页的 CLAS，继续使用原有 VBuffer 绘制。CLAS 可以独立启用；只有 `enableClusterRtx` 才分配和构建动态 BLAS、fallback BLAS 与 TLAS。

## 使用与观测

重新启动 `build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe`，选择 MiniZorah VBuffer。样例配置：

```json
"enableClas": true,
"enableClusterRtx": false,
"maxClasBytes": 1610612736,
"maxClasBuildClusters": 8192
```

`enableClas` 同样适用于独立 GPUDrivenStreamAssetPass。设备未启用或不支持 cluster acceleration structure 时，独立 CLAS 自动关闭，光栅路径继续运行；显式 RTX 请求仍严格检查支持。构建预算至少容纳资源中最大的一页，否则初始化报错。

Profiler 的 Table → Stream traversal → **CLAS build** 显示 CPU/GPU 构建阶段时间。Streaming 显示几何与 CLAS 堆叠占用，以及 CLAS 驻留页/cluster、本帧构建量、等待量、退役量、因容量不足推迟的页数；**CLAS Streaming** 图表默认折叠。CLAS 数值是池内已分配空间，包含退役等待释放的块，不是驱动编码后的紧凑尺寸。

## 生命周期

1. 上传成功入队时，从已经解码的 payload 提取紧凑构建参数；不在渲染遍历阶段重新读取或解压 streamasset。保留参数直到构建完成或对应几何分配被撤销。
2. 只在上传完成、页面进入驻留集合之后将其加入 CLAS 待构建队列。每帧按 cluster 上限取一批，复用现有 RHI 的多对象间接 CLAS 构建命令。
3. CLAS 地址按页面和 cluster 索引关联，驻留期间持续复用。队列只处理有变化或等待构建的页，不扫描完整场景/全部驻留页来寻找缺失 CLAS；不新增 GPU 等待或运行时读回。
4. 几何页卸载完成后，CLAS 进入退役阶段；经过 queued-frame 延迟再回收池空间。退役期间重新载入同一不可变页时可重新激活已有 CLAS。若上传完成前旧 CLAS 已释放，则使用保留的上传参数重新构建。
5. 保留现有按需求和内存压力管理几何缓存的行为：仅转出视野不会立即清空缓存，因此转回视角时已有 CLAS 不会全量重建。CLAS 容量不足时页面留在队列，VBuffer 渲染继续；Profiler 明确显示积压。

## 与参考实现的关系

参考本机 [scene_streaming.hpp](E:/vk_lod_clusters/src/scene_streaming.hpp) 的四阶段生命周期，以及 [scene_streaming.cpp](E:/vk_lod_clusters/src/scene_streaming.cpp) 的上传完成后批量构建逻辑。已实现按驻留页增量构建、稳定复用、分帧预算、延迟回收和对应的可观测数据。

当前复用 Metallic 已有的 CPU 持久分配器，按驱动最坏尺寸为每个 CLAS 预留固定槽位。参考实现还具有 GPU 生成构建描述、获取实际尺寸、构建后移动到紧凑持久分配器的路径；这部分尚未移植。因而不能将当前 CLAS 池占用与参考程序的紧凑 CLAS 字节数直接对比。下一步应优先实现实际尺寸分配/搬移，降低显存预留，再接入当前 LOD cut 的 BLAS/TLAS 和射线消费。

CLAS 是 Cluster BLAS 的组成部分，不能直接作为 TLAS 实例；参见 [Khronos 的扩展说明](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cluster_acceleration_structure.html)。本次完成流式 CLAS 构建层，不代表 MiniZorah 已切换到光追渲染。

## 实测与验证

RTX 5070 Ti，1920×1080，1.5 px，180 帧（前 60 帧固定，后 120 帧转视角），开启 Vulkan validation 和异步 Compute。开关对照使用相同路径、设备功能和采样方式，GPU 与后台应用共享；数值是本机单次验证，不是稳定性能承诺。

| 指标 | 关闭 CLAS | 开启 CLAS |
|---|---:|---:|
| 漫游段 RenderGraph GPU 平均 | 4.713 ms | 4.780 ms |
| CLAS build GPU 平均 | — | 0.085 ms |
| 最终驻留几何页 | 14,732 | 14,732 |
| 最终 CLAS 页 / cluster | 0 | 14,732 / 207,700 |
| CLAS 池分配占用 | 0 | 1,140.9 MiB |
| CLAS 池容量 | 0 | 1,536 MiB |
| 最终待构建 / 容量推迟页 | 0 | 0 / 0 |

1 GiB 的试跑在约 18.6 万 cluster 处耗尽，留下 1,052 页积压。2 GiB 预留在本机同时运行的后台 GPU 工作下出现较大帧时间波动，因此最终收紧为满足该轨迹需求的 1.5 GiB；更长漫游仍应观察池容量和积压。

三个目标 Metallic、MetallicGPUDrivenSample、MetallicRhiTests 构建成功。最终功能组 9/9 通过：Profiler 历史/排序、上传完成/取消、压缩页驻留上传、MiniZorah 180 帧、独立 StreamAsset RTX 回归、CLAS 页计划、CLAS 池构建/退役/重新激活、低构建预算运行时。

运行时专项测试共 420 帧，每帧预算仅一页最大 cluster 数，验证逐层加载产生的积压收敛、稳定视角不重复构建、转出转回持续复用、无 BLAS/TLAS 分配路径。CLAS 池专项测试显式覆盖退役后重新激活与延迟释放；几何缓存卸载另由既有 eviction / batched unload 测试覆盖。

[结构化结果](E:/metallic/Documentation/MiniZorahStreamClasResults.json) · [最终日志](E:/metallic/build-relwithdebinfo/stream-clas/final.log) · [卸载回归](E:/metallic/build-relwithdebinfo/stream-clas/eviction.log) · [Streaming 截图](E:/metallic/build-relwithdebinfo/stream-clas/final/Profiler-Streaming.png) · [排序计时表](E:/metallic/build-relwithdebinfo/stream-clas/final/Profiler-Table-Sorted.png)

复现命令（PowerShell，先构建）：

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
.\build-relwithdebinfo\tests\MetallicRhiTests.exe '--gtest_filter=*minizorah_profiler_streaming:*stream_clas_runtime_lifecycle:*meshlet_stream_clas_pool_build' --rhi-validation --rhi-async-compute --output-dir E:\metallic\build-relwithdebinfo\stream-clas\verify
```

设置 `METALLIC_TEST_CLAS_OFF=1` 可运行同一 MiniZorah 测试的关闭 CLAS 对照；正常验证保持此变量未设置。
