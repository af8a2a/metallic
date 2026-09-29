# ZorahFull 加载路径对比：Metallic 与 vk_lod_clusters

日期：2026-09-29。Metallic HEAD `92698b62e780970590423fc4d4dacdbb1e385182`；本机参考 `E:/vk_lod_clusters` HEAD `1febfa7694ebdc8f97005a07897029272ca9e85a`。本轮读取源码、磁盘缓存头和已有原始运行记录，没有修改运行时、重 cook、启动 GPU 基准或重现用户本次一分钟加载。工作区存在其他进行中修改，本文以读取时实现为准。

用户提供的观测是 Metallic 超过一分钟、参考程序三秒以内。本轮确认了能解释差距的结构性工作量与调度差异，但尚未取得这两次运行的同条件阶段日志，不能将其当作已复现的 20 倍性能对照。

## 结论

优先参考 **很小的常驻最粗 LOD + 专门的初始化批量上传/CLAS 构建路径**。Metallic 已经有 cooked cache、Windows 内存映射、metadata-only 导入、并行纹理解码和按需 CLAS 存储；重复增加这些已有机制不能解释当前全部差距。

当前 Full 保底集仍包含 51,764 页、1,154,901 clusters、50,326,944 个唯一三角形，紧凑属性后的对齐设备 payload 约 2.956 GiB。运行时通过常规逐帧流送加载它，并要求全部根页、CLAS 和 fallback BLAS 达到就绪条件。参考程序为每个 active geometry 常驻一个最粗 group，源码要求该 group 只有一个 cluster；初始化直接批量上传和构建，然后才由正常渲染帧请求更细数据。

## 1. 已有实测先区分计时边界

最近一份完整编辑器运行：[stdout.log](E:/metallic/build-release/sw-group32-default-live-01/run1/stdout.log)、[Capture.json](E:/metallic/build-release/sw-group32-default-live-01/run1/Capture.json)、[Manifest.json](E:/metallic/build-release/sw-group32-default-live-01/Manifest.json)。该记录来自当天更早的 `0a35a479...` 加工作区修改，RTX 5070 Ti、驱动 616.92，隐藏窗口、VSync 关闭、Reflex 开启；不能称为本次 HEAD 新测。文件缓存未经清空，shader 日志显示缓存命中。

| 阶段/口径 | 已有记录 |
|---|---:|
| 编辑器初始化，包含设备/交换链 | 2.380 s |
| Full metadata 导入 | 0.956 s |
| 全部 4418 张初始材质纹理准备 | 0.810 s |
| RenderGraph 编译总计，包含上述 metadata/纹理和 streaming 初始化 | 4.844 s |
| 绘制循环开始到根资源及预览就绪，包含图编译 | 18.348 s |
| 上两项之差，包含逐帧加载、构建、渲染和等待 | 约 13.504 s |

`loadingSeconds` 是 `readyAt - loadingStart`，**不包含就绪后的 10 秒预热**，也不包含更早的编辑器初始化。`loadingFrames=782` 则包含预热期间的绘制次数，不能将两者相除求加载帧率。实现见 [EditorFullRoamBenchmark.cpp](E:/metallic/Source/Editor/EditorFullRoamBenchmark.cpp:201)。上述 13.504 秒不是单独测得的磁盘、DMA 或 CLAS 时间。

另几份今天的历史记录：`clas-capacity-roam` 为 16.366 s，`clas-lifetime-policy-roam` 为 18.112 s，`incremental-residency-epoch-roam` 为 18.712 s，`compact-record-full-roam` 为 32.329 s。各自代码、配置、窗口节奏和缓存状态不同，仅表明端到端就绪时间存在波动，不用于计算改动收益。

旧 [交互加载记录](ZorahFullInteractiveLoadPlan.md) 的 66.741 s 图编译、61.531 s 材质准备已不能代表当前路径。后续 [有界并行纹理加载](ZorahFullBoundedTextureUpload.md) 已落地；当前预设也改为普通纹理 128、MASK 512、后台细化到 512、512 MiB 预算。旧 512-cap 测试与当前首帧纹理工作量不同。

## 2. 实现差异

| 项目 | Metallic 当前 Full | 本机 vk_lod_clusters | 对加载速度的含义 |
|---|---|---|---|
| 离线缓存 | V9 `.meshstream.bin`，205,327,470,912 B，约 191.23 GiB | 同目录 `.nvsngeo`，53,319,001,672 B，约 49.66 GiB | 文件尺寸相差约 3.85 倍，但双方都按需访问，不能换算为加载倍率 |
| 映射与导入 | Windows `CreateFileMappingW/MapViewOfFile`；stream metadata 不加载原始几何 buffer；JSON 先经 nlohmann 解析、投影和序列化，再交 TinyGLTF | 映射几何缓存；cgltf 仍解析场景 metadata，缓存有效时跳过原始几何 buffer，GPU instancing 所需数据例外 | Metallic 已经具备缓存加载基础；重复 metadata 解析仍有约秒级优化空间 |
| 缓存打开 | 拷贝 group 目录，遍历 primitive/group/page/node/refinement topology 做验证；当前 V9 的 payload 按需解码 | `GeometryView` 指向映射中的各数组，按 geometry 建立视图 | Metallic 还有全局目录扫描/复制成本；没有证据表明它读取了全部 191 GiB payload |
| 初始保底集 | 所有实例涉及 primitive 的全部 terminal branches，包括隐藏实例和提前停止简化的分支 | 每个 active geometry 的最后一级 group，且 `clusterCount == 1` | 这是首帧必须读取、转换、上传和建 CLAS 的工作量差异 |
| 初始化调度 | 根页进入普通上传队列，Full 256 页/帧、8 MiB/帧、4 个加载 worker、1024 在途任务；CLAS 每轮最多 8192 clusters | 初始化用 `BatchedUploader` 上传 hierarchy 和最粗 group；专门建立低精度 CLAS/BLAS；细化再受流送限额控制 | Metallic 的加载吞吐与渲染/Present/帧同步节奏耦合 |
| 纹理 CPU/GPU 管线 | 4 worker 有界预取，单图 reader、复用 Zstd context；普通 vector 到 staging；主线程创建 image/view、录制提交；64 MiB/128 regions/3 批在途 | worker 映射文件、创建 image、直接解码到 uploader mapping；异步 transfer 上传、最后集中 graphics ownership acquire | 参考减少额外拷贝和主线程串行工作；当前 Metallic 这段已不到一秒，优先级低于根加载 |
| 就绪与交互 | `sceneReadiness()` 等全部 locked roots resident、对应 CLAS 就绪、fallback BLAS 已提交；未就绪时禁止视口交互 | 后台场景/纹理加载线程，主线程接手几何初始化；粗几何完成后渐进细化 | 后台线程提升响应性；首帧实际工作量仍取决于保底表示，不能只移线程就宣称加速 |

代码定位：

- Metallic 映射/目录验证：[MeshletStreamAsset.cpp](E:/metallic/Source/Runtime/Scene/MeshletStreamAsset.cpp:4210)；metadata 投影与二次解析：[scene.cpp](E:/metallic/Source/Runtime/Scene/scene.cpp:2628)。参考缓存：[scene_cache.cpp](E:/vk_lod_clusters/src/scene_cache.cpp:258)，导入：[scene_gltf.cpp](E:/vk_lod_clusters/src/scene_gltf.cpp:440)。[官方缓存说明](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/README.md) 同样说明大型缓存使用内存映射。
- Metallic 根集合：[MeshletStreamRuntime.cpp](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:919)，每帧上传：[MeshletStreamRuntime.cpp](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2454)，字节限流：[MeshletStreamResidency.cpp](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamResidency.cpp:1360)，CLAS 批量：[MeshletStreamRuntime.cpp](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2544)。
- 参考初始化：[scene_streaming.cpp](E:/vk_lod_clusters/src/scene_streaming.cpp:216)，单 cluster 保底约束：[scene_streaming.cpp](E:/vk_lod_clusters/src/scene_streaming.cpp:288)，初始 CLAS size/build 同步提交：[scene_streaming.cpp](E:/vk_lod_clusters/src/scene_streaming.cpp:2348)。参考也有同步等待，优势不能描述成“完全没有 GPU wait”。
- Metallic 纹理批次：[ScenePathTraceResources.cpp](E:/metallic/Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp:1353)，参考直接写 staging：[scene_textures.cpp](E:/vk_lod_clusters/src/scene_textures.cpp:558)，参考后台加载：[lodclusters.cpp](E:/vk_lod_clusters/src/lodclusters.cpp:300)。
- Metallic 就绪：[MeshletStreamRuntime.cpp](E:/metallic/Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp:2299)，视口交互门槛：[EditorApplication.cpp](E:/metallic/Source/Editor/EditorApplication.cpp:6309)。

## 3. 首帧工作量和限额的具体影响

根集数据来自 [完整审计](E:/metallic/build-release/root-cut-audit-full.json) 和 [摘要](ZorahFullRootCutAudit20260929.json)。完整审计明确 `compactShadingAttributes=true`，因此 2.956 GiB 已计入当前 N/T 紧凑表示，且包含 256-byte 页分配对齐；它不是 GPU 实测 VRAM，也不是磁盘压缩数据读取量。

- 2.956 GiB / 8 MiB 约为 379 个上传迭代量级。由于这里用对齐容量估算，不作为精确最少帧数；即使完全没有磁盘等待，也已是数百次迭代。
- 51,764 页 / 256 页约 203 轮；1,154,901 clusters / 8192 约 141 个 CLAS build 批量。这几种阶段可以重叠，**不能相加为加载帧数**。
- 若有效上传迭代只能随 60/30 Hz 帧推进，约 380 轮对应约 6.3/12.7 秒；这是模型估算，尚未包括 I/O、CPU 转换、CLAS size/build/MOVE、fallback BLAS 和队列尾部延迟。
- 初始队列按 fallback、请求年龄、屏幕收益及 page 大小排序，未按文件 offset 合并根读取。页任务通过映射读取并生成独立 payload，再进入 staging。冷文件缓存下，分散页访问及页面转换可能增加长尾；本轮没有 page-fault/disk trace，不能把一分钟的剩余部分直接归因于它。

根集包含约 1.99 GB 的 LOD0 terminal payload。OPAQUE 根占全部根字节约 99.4%；仅限制 MASK 或减少透明纹理不能解决这个几何保底集。审计没有保存简化停止原因，因此目前不能确定有多少源于 UV seam、法线、材质边界或误差约束。

参考还按 accessor 组合与材质结构去重，支持实例独立材质集：[scene_gltf.cpp](E:/vk_lod_clusters/src/scene_gltf.cpp:602)。这是可参考的离线组织方式，但现有 Full 审计的 exact-accessor sharing 潜力只有约 48.2 MiB 根字节；不能把材质解绑单独宣称为数 GiB 的直接收益。

## 4. 可以继续借鉴，但目前不是主要解释

本轮日志的初始纹理实际读取 67,932,477 B，解码 111,736,568 B，35,539 mip；含其他材质上传共 280 批。128-region 阈值使 64 MiB 字节上限往往达不到。合并同 image 的多 mip copy、复用命令对象、调整 regions/time flush 有优化空间。

不过这轮 staging memcpy 仅 4.99 ms，record 90.82 ms，image 创建 48.14 ms，header 499.25 ms；直接 staging 解码没有证据能省出几十秒。多 worker 的 open/read/decode 是累计工作时间，不能与 wall 相加。将 tail/header 聚合进带签名的资源包可能有助于冷启动，仍需要实测。

`StreamerSubsystem::prepareScene()` 当前先同步 `resources_.acquire()`，再初始化 geometry；`acquire()` 用无限预算 pump 完成资源准备。因此图编译期间仍有串行等待。可把其改为有界、可取消的 prepare 状态机，并在预算内重叠纹理和几何准备，但收益上限须按重叠前实际占比判断。见 [StreamerSubsystem.cpp](E:/metallic/Source/Runtime/Render/Streamer/StreamerSubsystem.cpp:57)、[SceneResourceManager.cpp](E:/metallic/Source/Runtime/Render/Streamer/SceneResourceManager.cpp:197)。

## 5. 建议实施顺序

1. **先建立当前一分钟运行的阶段基线。** 分开记录用户请求、metadata 完成、cache open/validate、纹理准备、stream 初始化、根 I/O/CPU 转换/上传、CLAS size/build/MOVE、fallback BLAS、首次有效 present、可交互。加载阶段持续记录 pending roots、实际上传 bytes、批量未满原因、CPU/GPU 时间和磁盘读取；不要只采就绪后的漫游。
2. **做专门的初始加载调度。** 在 Streamer 内引入 loading 状态，按字节/内存/GPU 完成点驱动根上传和初始 CLAS 队列，避免只能每次完整渲染一帧推进 8 MiB。使用有界 staging 和 scratch，保留完成点与预算检查；可减少不必要的细化/完整着色工作。就绪后恢复现有逐帧预算。先 A/B 加载模式，再决定批量大小，避免仅放大所有常驻池。
3. **缩小合法的保底表示。** 以根贡献最大的 primitive 做离线探针，记录简化停止原因，生成更小且覆盖完整的 coarse proxy/terminal representation。把 root payload 连续布局或生成可验证的 bootstrap pack，减少冷加载散读。先证明材质、法线/UV、MASK、阴影/反射覆盖正确，再重 cook Full。不能直接丢弃 terminal branches，也不能只移除 readiness 检查。
4. **随后减少 metadata/纹理小文件开销。** 缓存版本化场景 metadata，避免 nlohmann → dump → TinyGLTF 重复解析；考虑受内容签名约束的 header/tail 包和多 mip copy。将完整离线校验与运行时必需范围/依赖检查合理分层，不能无条件删除边界检查。

若目标是接近三秒，第二步可以减少帧节奏造成的等待；第三步才会根本降低首帧必需工作量。当前数据不足以承诺达到三秒。

## 6. 比较时必须对齐

本地参考 Full cfg 开启 compressed、multi-material、mapped cache、位置/UV/CLAS 位截断、adaptive error，并有 `skipmeshes`；Metallic 当前保留自己的材质/法线保真规则、固定 LOD 目标及全部实例。参考纹理是预算内预加载，不能等同于 Metallic 的反馈驱动 mip 流送。相同 glTF 不代表相同初始几何、纹理精度或加载完成条件。

后续 A/B 固定场景源与 cook 版本、相机、实例筛选、纹理/LOD 质量、分辨率、渲染路径、shader/driver cache、VSync/Reflex、显存压力。分别报告首次运行和重复运行；未控制 OS 文件缓存时不称严格冷启动。至少同时报告首次有效画面、可交互粗场景、目标视图细节收敛三种时点。本轮没有重新验证参考的三秒，也没有复现用户当前一分钟。
