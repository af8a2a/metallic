# MiniZorah：分离需求遍历与安全绘制集合

2026-09-16；基线 `d35bd59`，RTX 5060 / 8 GiB / 驱动 610.47。

## 实现范围

将独立的细节请求分摊到跨实例的 GPU 工作队列，再由原有父级优先遍历生成安全 cut。借鉴 WorkDistribution 的 wave 内可变长度任务分配思路，自行实现 Slang 代码；没有复制 Unreal 源码，也没有将包含多父依赖的安全 cut 改造成 Nanite 的完整持久化层级队列。

```text
Page updates / priority clear
  → reset demand bits + sparse state clear
  → visible-instance / conservative root tests → compact bounded task queue
  → demand workers + wave child distribution → group bits / tile bits / page requests
  → dispatch barrier
  → parent-first safe frontier → mask → prefix budget → stable emit
  → prefetch / finalize / existing CLAS and raster paths
```

- **任务划分**：运行时将每个 primitive 的有序 tile 森林划分为不相交的子树，每项最多 8 个节点，覆盖全部叶子。不可见实例不入队；每实例的 64 线程组并行测试子树根的保守误差界，压缩实际需要访问的根。复用现有 16 字节 traversal work item 缓冲。
- **全局负载分配**：固定 worker 从完整发布的队列中按 wave 聚合领取任务，lane 持有自己的有界子树游标。每轮最多进行 8 次空源补充，避免一个 lane 扫描空任务尾部拖住已经就绪的 wave。默认 realtime 图及两个 MiniZorah 专用图使用 4096 个 worker 线程，通用默认值仍为 1024。
- **wave 内分配**：对每个 lane 产生的 group 数做 prefix/sum，二分寻找扁平 group 的源 lane，再用 shuffle 取得 owner 与索引。使用实际 wave lane ID/宽度，不依赖 workgroup lane 到 subgroup 的隐式映射；尾部空 lane 也参与 shuffle。
- **需求传播**：准确的误差与视图需求判断写入 per-instance group 位图，原子标记叶子到根的 tile 路径；已经被其他 lane 标记的祖先可提前退出。winning lane 在 dispatch 结束前完成传播。缺页请求继续独立于父页可绘制状态，保留页面请求去重、屏幕收益优先级及 PendingUpload 规则。
- **安全 cut**：frontier 读取需求位图，跳过无需求子树，保留所有父组 active、页可绘制、终端集合齐备等条件。按 LOD 层发布状态，mask/prefix/emit 保持稳定顺序及整帧容量回退。没有通过修改误差阈值、关闭剔除或减少输出几何来换取性能。
- **发布与容量**：入队和消费为不同 dispatch，以显式设备屏障连接，不在一个 dispatch 中轮询未发布 payload。以全部静态任务模板的数量检查队列容量；超限时完整回退到 ordered demand，不截断。关闭分布式需求时也保留原来的合并 clear/frontier/mask 调度，避免给兼容路径增加拆分成本。
- **自适应调度**：固定分布式调度在密集视角受益，但中远景会因额外的清理、入队和 mask 变慢。因此，沿用已有请求反馈的完成时序，额外拷回 4 字节 group 测试计数，决定后续帧是否使用分布式需求。默认低于 65,536 次测试时切回 ordered，重新进入需要达到阈值的 1.125 倍，避免临界值反复切换；首次没有反馈时使用分布式路径。该策略不增加 CPU 等待，反馈滞后仅影响调度成本，不影响当前帧的精确选择或完整覆盖。

需求位图和父节点索引均由运行时元数据派生，不修改资产 cook 格式。MiniZorah 有 145,245 个静态任务模板，需求缓冲含统计头共 1,755,468 字节；另外增加每 tile 的一个父索引、每实例四个 metadata word，以及每任务一个根索引。任务 payload 使用已有队列，不新增同尺寸队列。

## 可观测性与开关

Profiler 现在区分 `Priority clear`、`LOD clear / demand seed`、`Detail demand`、`LOD frontier`、`LOD mask`、`LOD prefix`、`LOD emit`。新增 checkpoint 为 `AfterStreamPriorityClear`、`AfterStreamStateClear`、`AfterStreamDemand`、`AfterStreamMask`。ordered 路径的 clear/mask 仍计在 frontier，空拆分区间只含 checkpoint 开销。

`streaming.<pass>.demandStats` 仅导出 32-word 统计头：

| word | 含义 |
| --- | --- |
| 0 | worker claim 游标，允许领取尾部空区间，不能当作实际任务数 |
| 1–3 | 完成任务数、worker 访问节点数、测试 group 数 |
| 4–7 | 单任务最大节点数、最大 group 数、非空任务数、扁平分配 wave 槽位 |
| 8–15 | 测试 group 数直方图：0、1–8、9–16、17–32、33–64、65–128、129–256、>256 |
| 16 | 本帧发布的实际任务数，应与完成任务数一致 |
| 17 | 当前帧是否实际运行分布式需求 |
| 18 | 本帧完整需求的 group 测试数，两个路径均记录，供调度反馈使用 |
| 19–31 | 保留 |

`lodState` 的每实例 sparse header 第四个 word 记录入队前的根测试次数；质量报告的 `cut.demand.seedRootTests` 汇总该字段。该工作不计入 worker visited，分析时必须同时考虑，不能声称节点总工作只剩 worker 的部分。

`distributedPageDemand` 是两个 stream pass 的初始化配置，默认 true；`distributedDemandMinGroups` 默认 65,536，设为 0 强制使用分布式路径。`maxTraversalWorkItems` 限制队列容量，`maxTraversalWorkers` 控制 worker 线程数。更改初始化配置需重新编译图。回放脚本支持 `-DemandTraversal Default|Ordered|Distributed` 与 `-DemandWorkers`；Default 使用自适应策略，Distributed 将阈值设为 0，Ordered 禁用分布式需求。脚本记录请求配置并检查显式指定的需求模式与实际 worker 预算，避免对照参数被 harness 覆盖。

debug snapshot 的 `distributedPageDemand` 表示功能可用，`distributedDemandActive` 表示当前帧实际选择，`recentDemandGroupTests` 来自最近完成的请求反馈。阈值是本机 MiniZorah 测量得到的成本启发式，不是所有 GPU、资产的通用最优值。

实际 worker 数沿用 runtime 的实例数上限并按 64 向上取整，通过 `demandWorkers` 导出。MiniZorah 实例数足以使用全部 4096 线程；只有很少实例但单 primitive 特别大的资产仍受这一旧上限约束，后续应单独解耦该预算。

## 测量协议

沿用仓库 `.cache/gpudriven-four/Replay.json`，SHA-256 `feb79872154850af32db25a54ba3d22b48b9a04a10f7f2e8dadaf19f98f2f2d2`。每轮 3,000 帧、10 阶段；完整实时管线，输出 1920×1080、DLSS Quality 内部 1280×720、LOD 1.5 render px、几何/CLAS/动态 BLAS 预算 1024/512/256 MiB。CLAS、阴影、光栅队列和照明配置保持一致。

旧的 `LOD frontier` 同时包含 priority clear、稀疏状态清理和 mask，必须与新路径这些阶段加上 detail demand 的**逐帧总和**比较。汇总器输出 `frontierEnvelopeMs`，不能拿新的 frontier 单项与旧 scope 比较。

原始数据在 `.cache/stream-distribution/`。所有指定重复轮次的全部帧保留，没有剔除慢轮次。计时回放不插入诊断回读，独立 quality 回放检查完整 cut、可见误差、任务统计和中间绘制集合。每次记录路线、shader、可执行文件、harness 哈希，确认回放中没有修改它们。

## 最终测量结果

下面为同等 frontier 范围的 GPU P50，单位 ms，保留每阶段全部帧；m1/m2 为完整独立重复。最终两轮都启用自适应调度、4096 workers。

| 阶段 | 基线 m1 | 基线 m2 | 最终 m1 | 最终 m2 |
| --- | ---: | ---: | ---: | ---: |
| warmup | 1.112 | 1.111 | 0.938 | 0.945 |
| start_hold | 1.111 | 1.111 | 0.941 | 0.948 |
| forward_1 | 1.119 | 1.120 | 0.933 | 0.944 |
| forward_2 | 0.640 | 0.636 | 0.716 | 0.728 |
| forward_3 | 0.458 | 0.457 | 0.523 | 0.527 |
| far_hold | 0.612 | 0.611 | 0.675 | 0.677 |
| return_1 | 0.458 | 0.458 | 0.523 | 0.526 |
| return_2 | 0.636 | 0.634 | 0.708 | 0.731 |
| return_3 | 1.120 | 1.121 | 0.934 | 0.943 |
| return_hold | 1.112 | 1.116 | 0.940 | 0.947 |

密集视角观察到约 15–16% 的同范围 P50 降低。中远景相对基线仍增加 **0.06–0.10 ms**，没有完全消除轻负载回归；自适应避免了固定分布式调度更大的 0.15–0.29 ms 增量，但新 shader 路径、阶段记录及统计仍有成本。本轮不宣称所有视角都更快。

`return_hold` 的完整 Stream traversal：基线 1.772/1.789 ms，最终 1.612/1.636 ms；VBuffer 为 3.945/8.060 → 4.288/8.630 ms。相同最终构建的强制 ordered 对照为 frontier envelope 1.189/1.188 ms、Stream traversal 1.875/1.874 ms。ordered shader 已恢复合并调度，但没有恢复到基线的完全相同指令与测量成本。

后台 Unity 等进程的 GPU 负载在各轮之间变化，最终 m2 明显受竞争影响。全路线逐帧 frontier **均值**为基线 0.880/0.929 ms、最终 0.825/1.106 ms，尾部延迟也保留。因此，这些是本机测量观察，不能据此归因整帧收益或宣称已经证明隔离环境下的全路线提速。完整均值、P50/P95/P99/max、每进程引擎采样和监控覆盖均保存在 JSON 中。

最终近景检查 105,483 个候选子树根，发布 17,144 个任务，其中 17,107 个非空；worker 访问 67,178 个节点、测试 80,813 个 group。两轮分配 96,864/96,032 个 wave 槽位，单任务最多 7 个节点、320 个 group，只有 4 个任务超过 256 groups。safe frontier 则访问 128,095 个节点、测试 39,784 个 group。相比旧遍历，需求工作没有凭空消失：根筛选和 worker 的节点测试也必须计入，总收益来自并行分配及 cheap demand-bit 查询。

所有对照的最终 cut 一致：19,394 active groups、162,989 selected clusters，early 26,806 HW / 19,172 SW，late 200 HW / 361 SW，容量回退实例数为 0。独立 3,000 帧质量回放包含 15 个检查点；第 29 帧仍有待加载细节，第 59 帧及以后可见超目标细化数量均为 0，最终最大可见误差为 1.499928 px。899–2399 帧的检查点实际使用 ordered，2699 帧后恢复 distributed，切换没有破坏覆盖或误差目标。

## 开发对照与排除项

完整分布和 manifest 见 [StreamWorkDistributionResults.json](StreamWorkDistributionResults.json)。保留全部中间试验，避免只选最终近景的有利样本：

- `prototype` 将全部模板直接投入 worker，detail demand 约 1.03 ms，比原 frontier 整段更慢；未保留。
- `bits4096` 尝试平扫 tile 位图，frontier 约 0.90 ms；改为沿有序层级跳过无需求子树。
- `seeded4096`、`refill4096`、`seed-filter4096` 依次引入可见实例入队、wave 空源补充、保守根筛选，近景 detail demand 约 0.448 → 0.356 → 0.256 ms。
- `final-ordered` 是拆分 clear/mask 的 ordered 对照，额外 dispatch 使 traversal 约 2.02 ms；当前 ordered 路径恢复合并调度。`release-ordered` 保存合并调度后的中间版本。
- `final` 实际使用 1024 workers，原因是默认 sample 加载 `gpu_driven_realtime`，最初仅更新了两个 MiniZorah 专用图。保留这两轮及其独立质量回放，实际参数见报告，不能标成 4096。
- `final4096` 是固定分布式调度，两轮近景 envelope 约 0.937/0.940 ms，但中远景增加 0.15–0.29 ms，因此继续实现自适应调度。最终结果使用 `adaptive`，不以该中间构建替代最终测量。
- 更早的 `ordered` 目录不作为 ordered 对照：当时 harness 覆盖了请求的环境变量，实际仍是 distributed=true。该无效对照在 JSON 中明确记录。

## 正确性验证

Release 的 `MetallicRhiTests` 和 `MetallicGPUDrivenSample` 构建通过。最终 `adaptive-tests.log` 的 8 项测试在 Vulkan validation 开启时全部通过：GPU/CPU oracle 共 872 个用例，涵盖共享父级、511/8191-group 大拓扑、缺页、PendingUpload、请求优先级、预取、隐藏恢复、稀疏状态清理、输出预算和逐帧切换调度；任务统计与实际测试数量也做一致性检查。任务根的 CPU 检查覆盖不同上限下叶子不重叠、不缺失。

真实 Bunny 测试核对 8 组视图/LOD × 4 种光栅配置的 cut、有效 ID、深度和冻结相机，确认自适应的两种调度都被执行，并验证队列容量为 1 时退回完整 ordered cut。混合 resident/stream 生产者和独立软硬光栅等价测试也通过。最终渲染图像和统计位于 `.cache/stream-distribution/adaptive-tests/`。

保留一个未定位的测试异常：中间构建的 `release-tests.log` 在 Bunny 容量回退引发图重编译后捕获过一次 CPU SEH `0xc0000005`。同一构建随后两次单项、两轮完整测试，以及最终自适应测试均未复现；没有放宽断言或删除失败日志，目前不能宣称其根因已经修复。

完整实时回放关闭 Vulkan validation，专项测试开启。Streamline 在报告和测试结束标记写完后可能停滞于退出清理，脚本只回收自身启动的进程，并记录在 `Process.json`；这段清理不计入帧时间。进程 GPU 监控对无效计数器样本继续采集后续有效样本，报告同时记录监控错误字节数和覆盖情况。

## 验证命令

```powershell
cmake --build build-release --target MetallicRhiTests MetallicGPUDrivenSample --parallel 6
./build-release/tests/MetallicRhiTests.exe '--gtest_filter=*meshlet_lod_stream*:*hybrid_raster_scene_equivalence:*render_graph_gpu_driven_mixed_producer_render' --rhi-validation --output-dir .cache/stream-distribution/adaptive-tests
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/stream-distribution/adaptive-ordered -Cases m1,m2 -Realtime -DemandTraversal Ordered
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/stream-distribution/adaptive -Realtime -QualityWithoutValidation
```

## 保留的限制

安全 frontier 仍为每实例一个线程组，父组依赖与同层 tile 调度尚未跨线程组分摊。根筛选仍扫描可见实例的静态模板；预算检查采用最坏情况下的全部模板数量，小预算即使当帧可见任务不多也会回退。未来可以在这些统计基础上继续拆分同层工作与动态任务预算，而无需冒险放松完整覆盖条件。

本次没有重写 CLAS 发布、BLAS 复用、压缩位置格式、光栅或 DLSS。vk_lod_clusters 的截图包含不同 renderer、cut、任务和 BLAS 复用条件，不能据本轮数据宣称达到其 0.084 ms 遍历耗时。
