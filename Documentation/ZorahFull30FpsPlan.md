# ZorahFull 漫游 30 fps 推进计划

2026-09-20。目标条件：RTX 5070 Ti，用户当前编辑器视口，DLSS Quality，完整材质与现有阴影，固定 LOD 1.5 render px。本文为代码及已有验证数据分析，没有新增性能采样，也没有修改运行时。

P0 已完成：[三轮实测及实现](ZorahFullP0Benchmark.md)。实测软件光栅约 51 ms、软硬分类约 22 ms，TLAS Build 约 0.34 ms；实际下一步改为先处理光栅/分类工作量与容量准入。以下是采样前的候选路线，不再将 TLAS Update 视为首要性能优化。

## 结论与证据边界

下一步先建立 Full 编辑器漫游的帧时间基准，补齐 RTAS 与纹理流送的分项计时；随后优先验证 TLAS 更新与流送时间预算。具体 GPU 热点排序由新基准决定。当前证据不足以承诺优化幅度或认定某个 pass 已经是最大瓶颈。

- [T2/T3 验证](ZorahFullTextureStreaming.md)证明纹理按需细化和冷回收可用：原生测试前进后 image allocation 167.86 MiB，离开视野后 116.26 MiB。它有 validation、逐帧读回和人工朝天阶段，不能作为真实漫游 FPS 基准。
- 真实编辑器烟测验证过两轮完整 DLSS/HDR 切换，但视口被测试强制为 1404×674、相机基本静止、逐帧日志很多。它不代表用户当前视口，也不能证明稳定 30 fps。
- MiniZorah 的旧 Nsight capture 和固定路线没有覆盖 Full 的完整材质、阴影及当前 RTAS 路径，不能移用其毫秒数据给 Full 排序。
- 纹理细化目前最高 512，普通基础尾链为 128，MASK/位移固定保底。此次目标沿用该画质条件；原始 4K/8K 纹理保真应作为另一个目标记录。

## 当前实现中值得优先测量的路径

| 路径 | 源码中确认的行为 | 下一步判断需要的数据 |
| --- | --- | --- |
| RTAS | 流式路径每帧准备 BLAS 输入、提交流式 BLAS 构建命令并完整 Build TLAS。BLAS 已比较完整 cut 和 CLAS 发布版本，存在跨帧缓存，不能将提交命令理解为每帧全量重建 BLAS | BLAS 输入各阶段、实际 build 数/cluster 数、缓存命中率、TLAS Build 的 CPU/GPU ms |
| RTAS profiler | `AfterStreamClasBuild` 结束细分 scope；后续 fallback BLAS、BLAS input/build、TLAS input/build 只落在外层 Stream traversal 内 | 分开这些 scope，消除外层未归属时间 |
| 几何/CLAS 流送 | Full 上限为每帧 256 页 / 8 MiB 上传、8192 clusters CLAS 构建。这些是工作量限制，不能保证每帧执行时间 | 上传、解压、CLAS build/MOVE、回收耗时，以及与慢帧的相关性 |
| 纹理流送 CPU | 完成反馈逐图消费；可调度时逐图筛选，对候选查询 allocation size；每批创建解码器和上传 command pool/tracker；image 准备阶段的 1 ms 是软限制 | 反馈、调度/查询、解码等待、创建 image/view、提交、发布/退休分别计时 |
| 纹理流送 GPU | 迁移另行提交到 graphics queue；现有请求延迟统计仅含入队至发布 | 独立上传提交的 GPU 时间、队列等待、首次需求到实际可采样的延迟，包含等待预算 |
| 驻留结构 | [根页审计记录](ZorahFullInteractiveLoadPlan.md)：3.5 GiB 几何池中根页约 2.956 GiB，细化余量约 0.544 GiB；51764 终止页、512130 实例 root groups。CLAS 外层池配置 2 GiB | 当前实际 root/active bytes、可用细化量、重复装卸率、CLAS 有效/物理/scratch 容量、heap headroom |

源码入口：

- [流送与 RTAS 顺序](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2492)、[BLAS 缓存输入](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:3540)、[TLAS Build](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:3782)。
- [现有 traversal scope](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:1125)。
- [纹理迁移](../Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1478)、[纹理调度](../Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1552)。
- [Full 运行配置](../Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json)。

## P0：建立真正的 Full 同条件基准

复用现有 replay/数据汇总机制，但必须接到 Full 的真实编辑器渲染图，保留 Shadows、Deferred、DLSS、输出与编辑器开销。现有 `RunMetallicCfgReplay.ps1` 固定为 MiniZorah 协议和测试入口，不能仅替换资产路径便称为 Full 基准。

1. 启动记录实际 output extent、DLSS 返回的 render extent、HDR/Present/VSync 模式、帧数在途、驱动、二进制/shader/资产摘要、所有质量与预算配置。采样期间固定视口，不用烟测分辨率替代当前视口。
2. 从 Full cfg 相机开始保存一条约 180 秒路线：静止、沿主通道前进、近墙观察、连续转向进入新区域、返回原路、静止收敛。使用时间驱动相机曲线，确保优化前后速度和路径一致；首次访问与回访分段统计。路线先检查穿模和代表性。
3. 性能模式关闭 validation、逐帧截图/同步读回与详细文本日志。保留异步 timestamp 和轻量计数，结束后导出。正确性另跑带 validation 的测试；分阶段截图用于核对几何覆盖、材质、阴影和细化状态。
4. 独立运行至少三次，区分进程冷启动、首访、热回访；启动/首次 PSO 准备单列。自然漫游产生的页面缺失、上传和细化必须计入，不得借预热整个路线隐藏流送长帧。记录其他 GPU 负载。
5. CPU：frame slot 等待、录制、反馈/调度、资源创建、提交/Present。GPU：traversal、分类、软/硬光栅、RTAS 各项、Shadows/Sigma、Deferred/binning、DLSS/输出，以及独立上传提交。多队列重叠和嵌套 scope 不直接求和，分析完整关键路径。
6. 逐帧导出 frame time、各 scope、几何/CLAS/纹理有效与物理驻留、待退休/在途字节、请求积压和年龄、上传/回收/重复装卸、BLAS 命中率、画面细化状态。按慢帧对齐指标，而非比较独立平均值。

验收不是平均 FPS：记录 p50/p95/p99/max、超过 33.33 ms 的帧数与最长连续超预算段。p99 ≤33.33 ms 是阶段门槛，不能称为“始终 ≥30 fps”；最终固定路线的采样帧应无超预算，且多轮成立。建议稳态关键路径争取 25–28 ms，为流送留余量，这是设计目标而非实测预测。CPU/GPU 可以重叠，不能将各自预算简单相加。固定路线通过也不保证任意视角都达到目标。

## P1：降低 RTAS 的重复工作

这是最明确的 GPU 优化候选，但应在 P0 得到耗时后再定投入规模。

- 先拆分 fallback BLAS、BLAS reset/cut compare/count/setup/insert/build、TLAS input/build；记录 actual build count 与命中率。当前 BLAS 输入部分按 `maxActiveGroups=1048575` 发起 dispatch，评估是否能改为按实际 active/dirty 数量的间接调度。
- 保留现有 BLAS cut 缓存，识别 CLAS 发布、搬移、退役引发的必要失效；评估同一几何且相同有效 cut 的实例共享，避免重复构建相同 BLAS。
- 对比 TLAS Build 与满足后端约束的 Update。参考程序在首帧后调用 Update；不能只改枚举，必须同时检查创建标志、scratch、有效实例集合变化和多帧生命周期，并验证后续 ray traversal 总成本。
- 只有 TLAS 输入、BLAS 内容/包围盒和地址版本都未变化时才研究跳过更新。不能仅凭相机静止或 BLAS 地址相同判定安全，也不能将延迟 CPU 读回当成本帧一致性依据。

参考：[vk TLAS 调用](E:/vk_lod_clusters/src/renderer_raytrace_clusters_lod.cpp:1069)、[Build/Update 实现](E:/vk_lod_clusters/src/renderer_raytrace_clusters_lod.cpp:1468)、[BLAS sharing/caching 开关](E:/vk_lod_clusters/src/lodclusters_ui.cpp:840)。这些是方法参考，不代表两个 Full workload 等价。

## P2：把内存预算扩展为流送时间预算

目标是减少首次转向和进入新区的长帧，同时保持细化能收敛。

- 在现有字节/页数硬上限外，结合已完成帧的实测成本，给几何上传/解压、CLAS build/MOVE、纹理创建/上传设置可调的每帧工作额度。GPU 时间只能反馈估计，不能假定本帧提交后仍可精确硬截断。
- 保底和真实可见缺页优先，其次细化，再次预测性预取；保留最低推进额度和请求 aging，避免压力下一直不细化。新旧资源共存继续受统一内存预算保护。
- 纹理侧缓存 `(格式、尺寸、mip 尾链)` allocation 查询结果，复用 worker/command pool/staging；维护待细化集合与到期冷集合，减少全表筛选。只在分项 CPU 数据支持时投入较大重构。
- 评估纹理迁移走 copy queue 的收益，完整计入所有权转换、完成依赖与同步成本；“异步提交”本身不等于与渲染重叠。
- 对请求等待加入毫秒统计；评估用时间而非帧数表达冷却/保留，避免 FPS 改变导致相同移动速度下缓存寿命变化。回访重复装卸与细化等待共同决定保留策略。

通过标准：首访/快速转向 p99 和超预算帧数下降，返回路线重复装卸下降或不退化，几何/纹理仍在约定时间内收敛，无覆盖缺口或 old/new 峰值越界。禁止靠长期停留在粗 LOD 获得表面 30 fps。

## P3：按实测 GPU 热点优化稳态渲染

| 如果基准显示 | 推进内容 | 验证重点 |
| --- | --- | --- |
| Shadows/RT traversal 占主导 | 分离 shadow ray 与 Sigma，分析 MASK/透明 any-hit、RT cut/BLAS 共享和射线工作量 | 不能用主视锥剔除直接丢失离屏遮挡者；保持阴影覆盖 |
| Deferred 占主导 | 分离 binning、材质求值与 texture feedback；按材质特征专门化，减少重复解码/采样，测反馈原子热点 | 完整材质、法线/TBN、透明及 DLSS guides 保真 |
| Traversal/光栅占主导 | 按实际工作量调度，测软硬阈值和 asyncSoftwareRaster A/B，再决定队列并行 | Full 当前 asyncSoftwareRaster=false；并行可能争抢资源，只认整帧收益 |
| DLSS/输出占主导 | 检查 guides 重复生成、无用中间图、格式/转换和输出尺寸 | 保持用户选定的 Quality 与视口；降低分辨率不能算同条件优化 |

诊断可以临时关掉某个 pass 估算成本，但最终验收恢复全部既定功能。此前候选展开并行、分类重组、硬件唯一顶点复用、Mini Stream Begin 冷集合优化已有实现，不作为未经测量的“新优化”。

## P4：根页与物理池的结构性收敛

P0 同时记录内存；若出现预算反复拒绝、细化容量不足或频繁重载，应将本项提前到 P2。

- 按 primitive/material 排序根页字节、终止原因、UV/法线 seam 与重复属性贡献，先在占比最大的少量探针上改进 coarse representation，再决定全量 recook。
- 几何/材质属性解绑、压缩和粗 LOD 策略必须通过 UV、alpha、法线/TBN、透射与固定 1.5 px 验证，不能用删除根页或漏对象换内存。
- CLAS 物理池、scratch 和工作缓冲按有效容量增长；记录有效字节、保留容量和预算三种口径。单纯把几何池分块不能消除近 3 GiB 根页。
- 参考的 RT 渲染可采用与光栅不同的几何保留方式；Metallic VBuffer 仍需要位置和属性，不能直接承诺复制其内存数字。

建议下一次实现交付：**P0 Full 编辑器 replay + RTAS/纹理分项 profiler**。有基准后优先完成 **P1 TLAS 更新 A/B** 与 **P2 流送长帧控制**，再用实际 scope 占比决定 P3 的首个目标。
