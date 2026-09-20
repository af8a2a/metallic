# ZorahFull 交互加载失败与优化计划

2026-09-20。Metallic 基线 `4c46a2a20`；本地参考 `E:/vk_lod_clusters` 基线 `1febfa7`。本轮分析用户 11:29–11:31 日志并核对双方源码，没有重跑 GPU 场景或修改运行时代码。

用户后续调整优先级：先优化中间的材质上传长链，按 [U0–U4 上传专项方案](ZorahFullTextureUploadPlan.md) 推进。本文件保留整体问题依赖；DLSS 错误分级和总预算仍是交互验收所需的独立工作。

## 已确认的失败链

1. 默认 MiniZorah 启动后，11:29:56 切换到 Full，源路径与 Full 预设正确。
2. Full metadata 成功，43068 primitive 实例、1514 材质、4418 纹理。纹理上传完成 369 批、1,459,358,444 字节；预算规划的 image allocation 为 1,469,821,440 字节（约 1.369 GiB）。
3. Full 图于 11:31:03 编译成功，输出 1404×674；DLSS 渲染分辨率为 936×449。
4. 11:31:07，`slEvaluateFeature(kFeatureDLSS)` 返回 `eWarnOutOfVRAM`。Metallic 的 `errorFromSl()` 将其转换为 `Error::OutOfMemory`，随后 RenderGraph 执行失败并停止预览。日志没有 Vulkan 分配失败或 DeviceLost 证据；退出码 0 也不能代表渲染成功。

Streamline 的实现先运行 begin/end evaluate，仅在结果为 `eOk` 且预算余额为零时附加此警告。它意味着预算压力，不能直接等同于 DLSS 求值失败或输出未生成。应在 evaluate 边界单独处理警告，保留真正错误的失败路径；不能只屏蔽日志而不控制内存。定位：[Metallic 映射](E:/metallic/Source/Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp:192)、[本地 SDK](E:/metallic/External/streamline/source/plugins/sl.common/commonInterface.cpp:549)、[官方源码](https://github.com/NVIDIA-RTX/Streamline/blob/main/source/plugins/sl.common/commonInterface.cpp)。发布 DLL 的实现版本仍应在修复验证中记录。

本次日志没有各资源域、进程与系统预算快照，因此尚不能归因于其他应用、旧 Mini 场景泄漏、DLSS 内存泄漏或某一特定池。需要测量这些假设。

此前 Z5 验收为 960×540 原生、DLSS 关闭的离屏 asset/world 加载及释放。它证明数据和着色链路成立，没有覆盖这次 Mini→Full 的交互式 DLSS 场景切换。**Z5 数据首帧保留通过，交互式首帧验收仍开放。**

## 耗时证据

| 阶段 | 本次日志 |
| --- | ---: |
| Full metadata 导入 | 0.949 s |
| Full RenderGraph 编译总计 | 66.741 s |
| `alpha-test resources` | 61.531 s，约占图编译 92.2% |
| shader / compute pipelines | 54.87 ms |
| 纹理上传 | 369 批，平均 3.77 MiB/批，最多 3 批在途 |

`alpha-test resources` 实际调用共享材质/全部纹理资源准备，不只处理 MASK。`SceneResourceManager::acquire()` 用无限时间片循环 pump 异步准备，使图编译同步等待全部完成。当前日志不足以把 61.5 s 再分成 IO、Zstd、image 创建、CPU 拷贝和 GPU 等待。上传配置上限为 128 regions，日志中许多批次为 120 regions；小 mip 的合批策略值得优化，但不能据此宣称它就是全部耗时原因。

定位：[同步等待](E:/metallic/Source/Runtime/Render/Streamer/SceneResourceManager.cpp:200)、[材质准备调用](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:3400)、[上传上限](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1298)。整理的日志和数值：[summary.json](E:/metallic/build-release/zorah-full-load-analysis/summary.json)、[normalized.log](E:/metallic/build-release/zorah-full-load-analysis/normalized.log)。

## 与参考实现的关键差距

| 项目 | Metallic 当前 | 本地 vk_lod_clusters | 优先改进 |
| --- | --- | --- | --- |
| CLAS 池 | 实际对象尺寸分配/MOVE 已有，但外层 buffer 初始化即分配 2 GiB；另有多批 build/MOVE scratch | 动态 CLAS 从 128 MiB 起，以 128 MiB 增长，最高预算独立；粗级 CLAS 单独常驻 | 区分对象有效大小、物理池容量和最大预算，增加按需扩容 |
| 几何池 | 一次分配 3.5 GiB；紧凑根页已占 2.956 GiB，细化余量约 0.544 GiB | 动态 storage 使用最高 128 MiB 的分配块；persistent 与 active 分开 | 先审计根页，再支持按块分配；只切块无法减少根页本身 |
| 保底 LOD | 51764 终止页、512130 实例 root groups；保真约束导致多终止分支 | 每个 active geometry 的最低 LOD 为单 cluster group | 统计根页贡献与简化停止原因，改进 coarse representation；不能直接删根页或破坏 UV/alpha/材质 |
| 纹理 | 已有 BC 尾链直传、实际分配预算、3 批上传；同步等待完整材质集 | 多线程加载、直接解码到异步 uploader staging；预算内预加载 mip 尾链 | 并行 IO/解码、合批、进度与可取消加载 |
| 场景切换 | graph/frame retention 与共享资源管理器保护生命周期 | teardown 显式等待并销毁旧 render scene，再加载新场景 | 在安全完成点记录旧资源释放量，预算不足时避免两场景资源峰值重叠 |

参考位置：[CLAS 起始/增长配置](E:/vk_lod_clusters/src/scene_streaming_utils.hpp:41)、[CLAS 扩容](E:/vk_lod_clusters/src/scene_streaming.cpp:1958)、[几何块分配](E:/vk_lod_clusters/src/scene_streaming_utils.cpp:1011)、[单 cluster 保底](E:/vk_lod_clusters/src/scene_streaming.cpp:288)、[纹理并行](E:/vk_lod_clusters/src/scene_textures.cpp:656)、[切换释放](E:/vk_lod_clusters/src/lodclusters.cpp:438)。

比较时必须保留三个边界：

- 参考的 streaming budget 不包含全部 persistent 数据，不能与 Metallic 整池数值或整卡占用直接比较。参考 RT 路径可在 CLAS 建好后丢弃 position，而 Metallic 的 VBuffer 光栅仍要读 position，不能照搬。[官方 streaming 文档](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/streaming.md)
- 本地参考的 `SceneTextures` 明确是预算内预加载，尚无视图驱动 mip streaming；不要将其解释成完整虚拟纹理系统。其 DLSS 使用 NGX，与 Metallic 的 Streamline 返回值路径不同。
- Full cfg 开启压缩、位置/UV/CLAS 位截断、自适应误差和 skipmeshes；Metallic 当前保留全部实例、固定 1.5 px 目标，属性有自己的保真约束。现有截图不能建立等质量的内存或帧时倍率。

## 推进顺序与验收

| 阶段 | 具体交付 | 验收门槛 |
| --- | --- | --- |
| F0：交互首帧修复 | evaluate 警告与真错误分级；成功输出继续提交并报告压力；真实失败保留诊断/恢复入口，自动验收不能只看退出码 | 注入 warning / 真错误分别覆盖；Full 直启与 Mini→Full 在 DLSS 开/关下有有效材质画面；不将真 OOM 转成功 |
| F1：总预算和切换峰值 | 采集 Vulkan heap budget/usage；分域记录 geometry、CLAS、scratch、BLAS/TLAS、texture、帧资源、DLSS 及在途旧资源；给后处理和增长留余量；按压力暂停细化、回收冷页或选择更低 mip 尾链 | 10 次 Mini↔Full 往返，无持续资源增长；直启/切换/resize 均记录峰值；低预算能受控回压或明确说明最小工作集不足 |
| F2：CLAS 与工作缓冲按需分配 | 分离 start/max/grow 容量，优先分段地址稳定方案；需 MOVE 时计入双池峰值并在完成后发布新地址；build scratch 按批量和实际并发配置 | 固定 root cut 下实际 CLAS 分配显著低于当前 2 GiB（具体值由根 CLAS 测量确定）；增长/回收/多帧在途验证无悬空地址，画面覆盖不变 |
| F3：材质加载长尾 | 将大 scope 分成 header/预算、IO、Zstd、image、staging、submit、GPU wait；有界并行解码和 staging 直写；按 bytes/regions/time 合批；使用已有异步准备接口使加载进度可显示、可取消 | 同机同场景分别测冷/热加载，多轮报告 P50/P95；369 批降幅和端到端时间均记录；不扩大 staging 峰值来伪造收益，4418 张材质资源完整 |
| F4：根页与几何表示 | 离线列出 root bytes 排名前列、终止原因、跨材质几何重复；先做小探针 coarse LOD、几何/材质解绑与压缩；再决定是否重 cook Full | 根页显著下降且 MASK、玻璃、法线/UV/TBN 保真通过；固定 1.5 px 时明确收敛比例和最大误差；未过探针不重跑全量 cook |
| F5：持续漫游/纹理细化 | 在共享预算下按屏幕收益提升 mip；与 geometry/CLAS 冷页调度协调，加入滞后避免反复装卸 | 固定往返/急转/近景路线，报告帧时 P95/P99、请求尾延迟、重载量、各池峰值与质量，不依靠隐藏内容或动态降低质量混淆比较 |

执行依赖：**F0 → F1 → F2** 先恢复可用并压低峰值；F3 随后处理已有明确证据的加载长尾；F4 解决 2.956 GiB 根页和长期质量空间；F5 完成 Z6。下一次最小实施批次建议是 F0 + F1 的观测/首帧矩阵，不以关闭 DLSS或继续增大 Full 固定池预算作为最终验收。
