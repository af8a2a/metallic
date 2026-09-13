# MiniZorah GPU Driven 里程碑路线图

日期：2026-09-13。目标资产：`E:/metallic/Asset/MiniZorah/zorah_main_public.v2.gltf`。

**当前进度：M1、M2、M3 已完成；M4 已接入同帧准入、自适应预取、GPU 完成后发布页面，并定位修正启动时上传暂存缓冲的分配放大。** 新增 CPU 阶段与 GPU 时钟对齐追踪，确认百毫秒长帧主要落在提交后、GPU 开始前；暂存倍增扩容将前 128 帧分配从 117～119 次降到 3 次。无追踪 60 秒路线最大同步帧从 237.636 降到 30.998 ms，请求墙钟 P95/P99 从 255/261 降到 101/132 ms，工作集差约 0.07%。首次建图仍约 5 秒，下一步优先处理 LOD 管线创建和场景 GPU 元数据准备；不把本轮结果当作全部设备或十分钟漫游的最大帧保证。见 [最新启动长帧定位](MiniZorahStartupStalls.md)、[冷启动发布优化](MiniZorahColdStart.md)、[预取与尾延迟](MiniZorahPrefetch.md)、[质量收敛](MiniZorahQuality.md)、[成本与收益调度](MiniZorahScreenBenefit.md)、[遍历与页面复用](MiniZorahRoamingOptimization.md)、[M4 初始基线](MiniZorahRoaming.md)、[M3 统一 VBuffer](MiniZorahVBuffer.md)。

上一轮需求策略曾通过 660 秒循环漫游：49,156 帧、132 个检查点，GPU P95 12.92 ms、同步帧 P95 18.38 ms；末轮进程本地显存峰值低于暖机周期。新需求策略本轮持续测试为 180 秒，不沿用旧版十分钟验收结论。

路线最初依据实际 glTF 元数据、Metallic 实现、本机 Unreal Engine 5.7.4 Nanite 源码和已有回归日志制定。资产统计见 [MiniZorahAssetAudit.json](MiniZorahAssetAudit.json)，全量 cook 和完整场景首帧实测见 [MiniZorahCook.md](MiniZorahCook.md)、[MiniZorahFirstFrame.md](MiniZorahFirstFrame.md)。当前优先级已根据 M3 统一入口与 M4 长测更新。

**结论：以 MiniZorah 为约束，主线应改为“受预算约束的离线构建 → 独立流式首帧 → 统一 VBuffer 的纯流式入口 → 持续漫游”。常驻路径 BVH、复杂材质和 RT 扩展不再是最近的前置工作。**

## 1. 实际资产决定的约束

| 项目 | 本地核对结果 | 对路线的影响 |
| --- | --- | --- |
| 源 `.bin` | 10,001,629,940 B，约 10.00 GB / 9.31 GiB | 超过 4 GiB，读取偏移必须保持 64 位 |
| meshopt 解码后的 bufferView 总量 | 31,690,629,036 B，约 31.69 GB / 29.51 GiB | 不能把全部解码结果及其复制品同时保存在内存 |
| 源几何 | 2,068 mesh、3,163 primitive、1,627,207,159 三角形 | 约 16.27 亿源三角形；构建按几何复用 |
| 实例 | 16,988 个 mesh node、19,144 个 primitive instance | 不能按实例复制几何与全部 LOD 列表 |
| 实例化三角形 | 18,937,042,387 | 约 189.37 亿，必须依赖 LOD 和可见性 |
| 最大 primitive | 32,054,609 三角形、16,005,261 顶点 | 单个 primitive 的构建峰值也需要实测和限制 |
| 高复用几何示例 | `SM_StoneFloor_6m_A1_*`：每份 19,461,216 三角形，365 个实例 | 这一种几何贡献约 71 亿实例化三角形，适合验证共享和实例裁剪 |
| 顶点属性 | 全部只有 Float32 VEC3 POSITION | 优先位置、拓扑和几何法线；UV、切线质量不是这个资产的前置需求 |
| 纹理、动画、蒙皮 | 全部为 0 | 纹理流式、WPO、蒙皮、动态细分不进入首个里程碑 |
| 材质 | 声明 3,283 个，活动场景引用 2,882 个 | 必须保留实例到材质的映射，但无需完整纹理材质系统 |

三角形统计按 glTF primitive/index accessor 计数，不是焊接后的唯一三角形数量，也不是每帧绘制量。第二个无 URI 的 buffer 声明了解码地址空间，不能因此判定缺少一个 31.69 GB 文件；此资产要求 `EXT_meshopt_compression`。最大的单个解码 bufferView 约 366.8 MiB，风险主要来自构建中多份数组、邻接关系、简化任务和累积元数据的叠加。

本机物理内存约 31.60 GiB，GPU 为 RTX 5070 Ti，显存报告 16,303 MiB。审计时可用内存约 2.2 GiB、GPU 已用约 14,083 MiB，因此这些即时空闲量不适合用来跑完整性能基线；本轮没有尝试全量加载。后续验收需记录运行时其他进程占用，并使用可重复的资源条件。

原始文件是几何导出。其 README 明确说明用于 NVIDIA `vk_lod_clusters`，并建议按 mesh 拆分处理。里程碑图像应以本次几何导出为准，原 RTX 演示中的纹理与动画无法从这份资产还原。[本地 README](../Asset/MiniZorah/README.md)、[NVIDIA 参考程序](https://github.com/nvpro-samples/vk_lod_clusters)。

## 2. 已有能力和真正的前置缺口

已有能力应直接复用：clod 构建、完整 terminal cut、共享父组约束、PendingUpload 保护、容量溢出回退、BVH 误差剪枝、早晚 HZB、稳定分箱和异步软硬光栅。之前“尚未接入自适应 LOD / 流式 VBuffer”的分析已经过时；当前说明见 [ResidentMeshletLod.md](ResidentMeshletLod.md)、[StreamingMeshletLod.md](StreamingMeshletLod.md)、[HybridRasterizer.md](HybridRasterizer.md)。

| 问题 | 代码证据与影响 | 本里程碑优先级 |
| --- | --- | --- |
| 全量常驻记录无法表示场景 | 128 三角形/cluster 下，实例化叶 cluster 至少 147,955,116；当前 ID 只能寻址 33,554,431 个 record。GPUScene 还预展开 base、各档 LOD、adaptive ranges | 必须使用按帧选择结果寻址的流式路径 |
| 统一 VBuffer 的纯流式入口 | M3 已加入 metadata Scene 与明确的 StreamAsset geometry ownership；空 resident layout、全局 ID、独立标量材质 resolve 均已验收 | 已完成 |
| 默认根页上限太小 | 每 primitive 至少需要一个 terminal group/page；完整保留 3,163 个 primitive 时，下界已超过示例的 `maxLockedFallbackPages=1024` | cook 后自动生成根集合及 byte/page/candidate 准入报告 |
| 流式分类按容量发起工作 | M4 已将默认混合路径改为稳定紧凑候选和 GPU indirect dispatch；缓冲预留容量不变 | 已完成实际数量派发，继续测分箱成本 |
| BVH 遍历长尾 | 已实现每实例 64 线程、按 LOD 分块协作；近景 frontier / emit 从 40.17 / 4.50 ms 降到 1.44 / 0.157 ms | 继续减少不可见细节的需求和分类/光栅成本 |
| 驻留几何尚未采用紧凑编码 | 源文件 meshopt 压缩不等于运行时页压缩；现有页仅 None/ByteRle，CPU 解压为设备格式，位置使用 float4 | 用 cook 数据判断是否将位置压缩提前为硬门槛 |

源码入口：[GPUScene](../Source/Runtime/Render/Subsystem/GPUSceneSubsystem.cpp)、[ID 上限](../Source/Runtime/Render/GPUDrivenRaster.h)、[VBuffer 与间接派发](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp)、[根集合预算验证](../Source/Runtime/Render/MeshletStreamRuntime.cpp)、[frontier、并行 prefix 与候选展开](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang)。

**不要先把 visibility ID 加宽来容纳 1.48 亿候选。** 当前编码足以容纳合理预算内的每帧可见集合；问题是把全场景实例几何预展开到它的地址空间。应保留真实的 `(asset/geometry, instance, cluster)` 身份，使用本帧紧凑 record 间接引用，且 early/late 与软硬分支共享本帧映射。跨帧历史不能把压缩序号当作稳定身份。

## 3. 按依赖推进的里程碑

### M0：建立资产与验收基线——本轮已完成静态审计

保留本轮统计，另外记录源文件身份、builder 配置、设备/驱动、图配置和相机脚本。首个固定视点使用资产 `.cfg` 中的相机：

```text
eye    = (55.34291, 6.4527273, 0.32432523)
target = (46.38051, 5.7994967, 0.5308178)
up     = (0, 1, 0)
fovY   = 60 degrees
near   = 0.020000003
far    = 29999.998
```

生成小型、可独立完成 cook 的测试子场景：普通几何、最大楼梯 mesh 2018、高复用地板 mesh 1949。薄 glTF 可保留原 accessor/bufferView 引用并引用原 `.bin`，无需复制整份二进制。小子场景用于隔离问题，完整场景是最终验收对象。

建议验收档位：1920×1080 内部渲染分辨率、1.5 render px，关闭 DLSS 建立清晰的几何基线；稳定漫游先争取 P95 帧时不超过 33.3 ms，再评估 60 FPS。静态场景、固定路线与原相机视点均保留。该帧时是待验证的工程目标，不是对当前实现的性能承诺。

### M1：全量离线 cook 可完成、可恢复、不会耗尽内存——已完成

2026-09-12 验收通过：新增 CPU-only `MetallicMeshletCook`，全部 3,163 个源 primitive 和 19,144 个实例映射通过核对，叶级保留全部 1,627,207,159 个源三角形；1,356,959 页全部验证通过，退出码 0。缓存 56.73 GiB，全流程 51 分 17.58 秒，峰值提交内存 4.15 GiB（配置上限 6 GiB），包含映射读取验证的峰值 RSS 为 9.34 GiB。三个子资产及 5 项相关回归通过，包括 1/4 线程与中断恢复产物的逐字节一致性。最终数据及复现命令见 [MiniZorahCook.md](MiniZorahCook.md)、[结果摘要](MiniZorahCookResult.json)。

现有 builder 已具备外部 glTF metadata/range read、`EXT_meshopt_compression` 解码磁盘缓存、逐 primitive 输出和断点恢复，应在此基础上加强预算和观测。[范围读取](../Source/Runtime/Scene/MeshletStreamAsset.cpp:2144)、[metadata 入口](../Source/Runtime/Scene/MeshletStreamAsset.cpp:3276)、[逐几何构建](../Source/Runtime/Scene/MeshletStreamAsset.cpp:3953)。

1. 先完成三个代表性子资产，记录每项 decode / meshlet / simplify / encode 时间、CPU peak commit、临时磁盘峰值、输出字节、group/cluster 数、terminal 集合大小。
2. 以最大 primitive 验证简化任务和临时数组峰值；增加按字节的构建并发预算。现有“每次最多多少 geometry”仅控制暂停位置，不限制单个巨型 geometry 的峰值，也不是硬内存预算。
3. 完整 cook 按 geometry 复用，并保留中断恢复。完成后只发布通过元数据、引用、DAG、payload 边界和根覆盖验证的正式 v9 资产。`.partial` 是恢复文件，不能视作可以渲染的正式子资产。
4. 在完成前持续输出预测值，完成后输出准确 manifest：各 LOD 字节、全部 terminal 页、最大 payload、实例 terminal 记录数、拓扑字节、每实例状态字节、建议驻留预算。

最初建议 CPU 峰值上限 24 GiB；实际执行根据本机同时运行的进程收紧为 6 GiB committed memory，并在最大 primitive 子资产上测得 3.58 GiB 峰值。该限制覆盖整个进程，不是 RSS 限额；线程数另外限制为 4。若单个 primitive 在控制并发后仍超预算，才将可分块/外存构建提升为本阶段必要工作；拆开原始压缩三角形流本身不能替代正确的 meshopt 解码。

早期分批构建命令示意（当前全量运行使用上文独立 cooker）：

```powershell
build\Source\Metallic.exe --build-meshstream Asset/MiniZorah/zorah_main_public.v2.gltf --output Asset/MeshletCache/MiniZorah.meshstream.bin --meshstream-compression none --meshstream-checkpoint-interval 1 --meshstream-max-geometries 1
```

可执行文件位置以实际构建目录为准；重复同一命令可分批恢复。`none` 是编码成本基线，不能预估为最终磁盘占用。通过峰值验证后逐步扩大每次 geometry 数量；最终全量完成后再做整个缓存的打开和渲染验收。

**完成条件：** 全部源 primitive/实例映射可核对；正常及中断恢复均可产出可打开的资产；运行内存不随已完成几何的完整 payload 累积。磁盘输出大小必须实测，不能把源 10 GB 当作所有 LOD 页的大小。

### M2：在现有独立 StreamAsset pass 中看到完整场景——已完成

2026-09-12 验收通过：新增 `GPUDriven / MiniZorah` sample 和 `--minizorah` 入口。3,163 个 primitive、19,144 个实例及全部 GPU 根记录逐项核对通过；两次独立进程启动的首像素为 5.624 / 5.586 s，全部 terminal 页就绪为 7.782 / 7.724 s。1920×1080 原始、远景、近景均出图，GPUScene resident 几何、实例、材质均为空，CLAS 关闭；9 项相关回归通过。系统文件缓存未清空，计时包含元数据核对、GPU 初始化和 debug readback，不能解释为断电冷盘时间或实时 FPS。[验收报告](MiniZorahFirstFrame.md)、[结构化结果](MiniZorahFirstFrameResult.json)。

实际接入发现，仅 `loadSceneInEditor=false` 还不够：旧 pass 的 World 依赖会让 RenderGraph 再次导入完整 Scene。新增 `streamAssetOnly=true`，同时取消 pass 的 Scene 依赖、内部 Scene 解析及 source lease；直接使用缓存实例和私有可见性槽，并显式初始化空 DrawSet 的视图版本。既有普通加载路径继续保留。[Sample](../Source/Runtime/Render/RenderSample.cpp)、[独立 pass](../Source/Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetPass.cpp)、[编辑器处理](../Source/Editor/EditorApplication.cpp:6848)。

新增专用 MiniZorah sample/profile，显式指定源、已完成缓存、原始相机、`autoBuildStreamAsset=false`、`enableClusterRtx=false`。先用现有硬件光栅和几何/LOD 可视化确认完整覆盖，随后使用已有简单光照查看结构。

从 M1 manifest 推导预算，替换 1,024 根页、4,096 驻留页等示例默认值。先为全部 terminal 页和安全输出预留空间，根页完整可绘制后逐级细化。不能通过少锁一部分根页或静默隐藏实例来绕过容量错误。

M1 实测：完整 terminal 集合为 **3,163 页、5.47 MiB 对齐 payload**，实例化 terminal group/cluster 均为 **19,144**；根集合加一个最大流式页的下界为 5.54 MiB。根页数量是现有默认配置的硬缺口，根 payload 字节数尚未要求先做压缩。主目录数组 402.37 MiB、当前 frontier 状态合计 154.76 MiB，仍需计入实际运行时副本与其他 GPU 资源。

最终首帧配置使用 **1 GiB 页池、`maxResidentPages=0`（纯字节预算）、4,096 根页容量、131,072 active groups**。profile 从完成的 manifest 生成。不能使用“页池 / 最大页大小”作为变长页数量上限，实测会先耗尽 slot 而闲置大量字节。512 MiB 压力测试保住全部根覆盖，但近景出现大量分配失败；1 GiB 三个采样的 active overflow 和分配失败均为零，近景仍有约 8,800 页等待上传，尚未收敛。峰值进程提交内存 3.81 GiB、工作集 1.95 GiB；页池以外还有 metadata、frontier、候选、帧槽和附件，不应把页池预算当成总 GPU 内存。后续仍可比较 1/2/4 GiB 档，初始 GPU/CPU 各 12 GiB 的观察目标保留。

**完成条件：** 全部 19,144 个 primitive instance 进入场景和可见性流程；从原视点、远景和近景看到完整结构，缺页时有粗表示；冷启动记录首像素与 terminal-ready 的时间，热启动不读取/解码整份原始 `.bin`；运行时不创建全场景 resident 顶点和普通 BLAS。

此阶段允许粗 LOD 尚未收敛，不能据此宣称“交互里程碑完成”。`no_mountains.cfg` 仅作为诊断对照；本资产匹配 50 个山体实例，共约 200 万实例化三角形，删除它们并不能解决主要几何规模问题。

### M3：统一 VBuffer 支持纯流式场景（已完成）

目标是将 M2 的加载优势接入已有 `VisibilityBufferPass`、混合光栅及后续着色，而不是为 MiniZorah 长期维护另一条功能分叉。

已交付 `--minizorah-vbuffer`，15 项 RHI 与 11 项 SceneGraph 回归通过。全场景保留 19,144 个实例及全部 606 个有效不透明 BLEND 实例；HW/异步混合覆盖一致，42 个边界像素的三角形 ID 差异已核对。标量材质使用独立 `VisibilityBufferMaterialPass`；原有 RTAS Deferred 对 metadata 输入提前报错。实际结果与 M4 边界见 [M3 验收](MiniZorahVBuffer.md)。

1. 提供只含节点、变换、材质、bounds 和几何句柄的运行时场景描述。流式 source 在 GPUScene 资源构建前就明确 ownership，避免生成其 resident 顶点、全部 LOD `MeshletDraws` 和普通 RTAS。
2. 允许 resident 范围为空，并保留混合 resident/stream 场景支持；完善纯流式场景的 instance/material identity、scene switch、resize 和 Deferred 输入契约。
3. MiniZorah 没有 NORMAL，使用已有基于三角形重建的几何法线；不需要为了首帧给十亿级顶点生成/储存整套法线、切线和 UV。
4. 它虽然声明 MASK/BLEND，但所有实际引用材质的常量 alpha 都是 1，没有纹理；MASK 均通过阈值。可以在此资产配置中按已证明的有效不透明性归类，保留原 source material ID、颜色参数和 double-sided。不能静默丢弃 606 个 BLEND primitive instance，也不需要为这些恒定 alpha=1 的表面先实现通用透明排序。
5. 第一版着色使用几何法线和源标量材质，确认所选管线不隐式拉起全场景 RTAS。需要几何阴影、反射的 pass 后接。

**完成条件：** M2/M3 相同 camera/LOD/frontier 下的覆盖和几何 ID 可核对；纯流式启动不回到普通 glTF 全量导入；硬件与异步混合输出一致，resize、场景释放和重新打开正常。旧混合 producer 回归仍通过。

### M4：在固定预算下持续漫游

已完成实际候选派发、视锥前移、并行 prefix、协作 frontier/emit、CPU 批量退避、有界 I/O、驻留复用、并行硬件 mesh 输出、late 重试 mask、收益排序和层级需求可见性。逐页延迟、同帧准入与受限预取之后，本轮改为核对拷贝提交及 GPU 完成后发布页面，消除固定发布等待并及时归还批次槽位。[最新结果](MiniZorahColdStart.md)；接下来分解仍然存在的启动长帧、预取转实际需求的尾部与 early 分类成本。

1. **实际候选数量驱动光栅。** 将 active group mask 展开为紧凑 cluster 工作列表，生成实际 count 和 indirect arguments。分类、前缀、散射、清零应随有效数量而不是预留容量增长；原始记录映射和容量回退继续正确。
2. **把实例剔除前移。** 全部 16,988 个 node 都是场景根节点，不能依赖源层级天然提供场景空间树。先得到可见/待复测实例，再进行重的 LOD frontier。上一帧 HZB 拒绝者必须留入本帧 late 阶段，不能永久跳过。
3. **并行 prefix 与协作遍历。** 实例 prefix 已并行化；frontier/emit 改为每实例 64 线程，按 LOD 层级的保守 tile 协作处理。稳定前缀保留原始 group ID 顺序，层间屏障保留共享父组依赖，完整 terminal 回退不变。继续从实际分类/光栅耗时判断下一步工作组织。
4. **将层级可见性与流式需求关联——视锥需求裁剪已完成。** group 范围包含自身误差扩展范围和全部细化后代，并对所有共享父组保守传播。屏外细节停止占用当前需求，完整 cut 与根页回退保持有效。三个固定视角在约 1.5 秒累计渲染时间的检查点达到目标；暂未用 HZB 剪掉质量需求。
5. **按收益调度页面——已接入。** GPU 以投影面积乘误差超额估计收益，共享页取实例最大值；CPU 以解压后字节和有界等待年龄排序准入、I/O 与就绪上传。根页与完整 fallback 继续保留，旧反馈格式和开关对照可用。接下来测 request→drawable 延迟和精确可见收益，联动每帧上传、驻留命中和反复装卸；只有 ancestor payload 阻碍目标收敛时才实现依赖安装/撤销。

M2 的 CPU 重复淘汰扫描和碎片空闲列表重复失败扫描已修复。GPU 未使用页留在预算内缓存，仅在压力下回收已确认的冷页，根页和近期需求得到保护。需求裁剪版本曾把 55 秒上传从 3.99 GB 降至 0.60 GB、驱逐从 74,740 页降至 0；新增预取后三轮工作集约 672 MiB，累计上传 701.92 MB，记录中仍无驱逐。60～175 秒的 24 个检查点无可见超标；64 MiB 档仍只满足完整回退，不满足 1.5 px 质量。

当前 frontier 虽然稀疏清理/emit，仍为实例预留 `32 + 12 × groupCount` 字节状态；共享 BVH 每节点 40 B。必须以 M1 的实际 group 数与实例分布核算，不能把“稀疏访问”当成“稀疏分配”。如果常驻 topology/state 本身越界，状态压缩或工作集化也成为本阶段前置条件。

**完成条件：** 固定 60 秒路线与近/远瞬移均无孔洞，10 分钟巡航无显存持续增长或 device loss；预算不足时保留完整粗 cut，不能假装已达到 1.5 px。计入未收敛页面和像素误差超标情况，避免用始终最粗 LOD 换取 FPS。达到约定帧时后再进行异步 HW/SW 阈值调优。

### M5：按实测瓶颈补压缩，完成验收

位置、索引和 cluster metadata 压缩是本场景与 Nanite 的重要差距，优先级高于 UV/材质扩展。若 M1/M2 已显示 terminal bytes、热工作集或 I/O 吞吐无法满足预算，本阶段应提前，而不是等到所有性能优化完成。

先测只存 float3 位置与更紧 metadata 的收益，再评估共享位置量化网格、局部索引/三角形编码、磁盘页 codec。量化误差要与简化误差一起约束，跨 group 共享边界必须解码到相同坐标；不是各 cluster 独立取 min/max 就能保证无缝。

首个里程碑可保留一 group 一 page 和 CPU 页解码，只要预算及收敛时间通过。GPU 转码、固定大小页、跨页 group/fixup 等复杂度由测得的瓶颈驱动。源 `EXT_meshopt_compression`、cook 磁盘压缩和 GPU 驻留编码应分别统计。

验收报告至少记录：cold/warm 启动时间、cook 时间和峰值、全量缓存大小、terminal 常驻下界、实际 GPU/CPU 总量、visited nodes/groups、候选/实际 cluster、HW/SW 分布、各阶段 GPU 时间、上传字节/延迟、误差未达标状态、P50/P95/P99 帧时。异步 GPU 阶段有重叠，不能简单相加当作整帧耗时。

## 4. 以 MiniZorah 为目标看 Nanite 差距

| Nanite 架构方向 | Metallic 当前状态 | 本次路线 |
| --- | --- | --- |
| 层级几何与视角 LOD | 已有 clod、合法 cut、BVH 误差剪枝 | 保留核心算法，验证十亿级构建成本 |
| 工作量随可见细节增长 | 纯流式入口、实际候选派发与按 LOD 层级的工作组协作已落地 | M4，继续推进层级可见性需求与分类/光栅效率 |
| 虚拟几何与资源预算 | 统一入口、完整 terminal、固定预算持续运行已验证，页面反复装卸仍较多 | M4，改善驻留收益与需求调度 |
| 紧凑编码与流式优先级 | 浮点 payload、CPU 解压；已接入屏幕收益排序、层级需求裁剪和有界预取 | 1 GiB 可容纳当前路线工作集，继续压低冷启动尾部，再根据紧预算收益决定压缩投入 |
| 小三角形光栅 | 已有稳定分箱、软件光栅及异步 HW/SW；两套深度再合并 | M4 后测瓶颈，暂不重做统一原子目标 |
| 材质、形变、阴影生态 | MiniZorah 没有对应完整源数据，RT 还有独立几何需求 | 首个几何加载里程碑之后 |

本机 UE 5.7.4 的 Nanite 在节点层执行 Frustum/HZB，并支持 persistent node/cluster traversal；其像素写入使用 depth/visibility 原子更新。这里用于理解减少无效访问和合并成本的方向，不预设复制这些实现就会更快。[层级裁剪](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:519)、[遍历](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:978)、[像素写入](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteWritePixel.ush:21)、[位置量化](E:/UnrealEngine/Engine/Source/Developer/NaniteBuilder/Private/Encode/NaniteEncode.cpp:191)。公开架构概述见 [Epic Nanite 文档](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine?application_version=5.7)。

延后的工作：常驻场景 BVH 统一、WPO/蒙皮、动态 tessellation、复杂材质/纹理流式、完整 VSM、CLAS RT/OMM 一体化。它们各有价值，但不应阻塞这份 POSITION-only 静态场景的光栅里程碑。后续 RT 也不能直接只使用主相机可见 cut，屏幕外遮挡物仍可能投影或出现在反射中。

## 5. 下一批具体改动

| 批次 | 可独立评审的结果 | 验收 |
| --- | --- | --- |
| PR 1（实现与验收已完成，未创建 PR） | MiniZorah 子资产构建/审计工具，cook 资源统计、字节预算与完整 terminal manifest | 全量完成、全部页和源覆盖验证通过；三个子资产及 5 项回归通过 |
| PR 2（实现与验收已完成，未创建 PR） | MiniZorah 独立 stream sample，manifest 驱动预算和固定相机，明确关闭普通 Scene/RTAS | 完整 GPU 根覆盖、两个独立进程首帧、三个视角、512 MiB 压力测试及 9 项相关回归通过 |
| PR 3（实现与验收已完成，未创建 PR） | 统一 VBuffer 的纯流式场景/GPUScene ownership 入口 | resident 几何零展开，实例/材质映射及 HW/SW 对照正确 |
| PR 4（持续运行基线已验证，未创建 PR） | 实际候选派发、并行 prefix、视锥前移、CPU 批量退避、漫游验证与阶段计时 | 660 秒预算/cut 完整性通过，交互帧时与质量条件仍开放 |
| PR 5（协作遍历与驻留复用已实现，未创建 PR） | 协作 frontier/emit、GPU 冷页反馈与碎片分配检查缓存 | 219 组 GPU/reference cut 对照、33 项回归、660 秒长测通过；同步帧 P95 32.68 ms，累计上传下降 69% |
| PR 6（分类/光栅与收益排序已实现，未创建 PR） | 并行 mesh 输出、late 重试候选、逐页屏幕收益与准入/I/O/上传排序 | 34 项回归、1 GiB 和 64 MiB 漫游；质量与 CPU/装卸权衡见最新报告 |
| PR 7（固定视角质量收敛已实现，未创建 PR） | 后代范围需求裁剪、可见误差度量和收敛期限回归 | 三个固定视角达到 1.5 px；180 秒三轮漫游，暖机后 23 个检查点无可见超标，近景收敛图像与旧版逐像素一致 |
| PR 8（预取与延迟处理已实现，未创建 PR） | 分段延迟统计、回读后同帧准入、受 CPU 剩余容量约束的视口边缘/细化预取 | 41 项回归、三个固定视角、180 秒漫游与 64 MiB 回退通过；转向恢复约 0.26 秒，上传增加 16.3%；剩余冷启动请求尾部单独报告 |
| PR 9（上传发布已实现，未创建 PR） | 基于实际拷贝提交和 GPU 完成的驻留发布、批次槽位及时回收、取消重试 | 冷启动请求帧 P95 14→11，密集路线首次质量归零 1.254→0.758 秒；保留墙钟 P95/P99 未改善的结果 |
| PR 10 | 启动长帧和请求最坏尾部、early 分类及候选准备成本 | 分离已就绪缓存命中与未就绪预取需求，定位录制/提交/等待/回读各阶段长帧；增加实际排队帧场景度量，继续约束质量、工作集和吞吐 |

M1–M3 的完整 cook、独立首帧和统一 VBuffer 已落地。M4 已接入屏幕收益排序、层级需求裁剪、自适应预取与 GPU 完成后发布，固定视角达到 cook 误差度量下的 1.5 px 收敛。下一步针对启动长帧、请求最坏尾部与 early 分类成本；64 MiB 压力回退仍不算质量验收，检查点结果也不等同于每个运动帧的质量保证。
