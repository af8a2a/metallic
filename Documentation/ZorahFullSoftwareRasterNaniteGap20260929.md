# ZorahFull SW Raster 与 Nanite 的实现差距

2026-09-29。Metallic 源码基线 `2c2acc47c9940466df3ce4e3b40df175e4b59d3b`；本地 Unreal `Engine/Build/Build.version` 为 **5.7.4**。本次为源码与已有证据调研，没有修改 shader、启动 GPU 对照或重新验收漫游性能。保留工作区中已有的 Profiler 修改。

用户明确允许按 NVIDIA 特性调优，主要验收目标为 RTX 5070 Ti。建议下一项先做 **wave32 下 32 / 64 / 128 线程组、分批处理顶点/三角形的独立对照**，同时重新采集当前 Full 视角的 cluster 填充率与三角形拒绝原因。后续按证据选择小 bbox 专门路径、自适应 scanline 或 cluster 构建改进；不先上全局三角形队列、重新打开局部分桶或重启位置压缩实验。

## 当前证据的含义

用户截图显示 Stream early GPU avg 9.730 ms，其中 Software raster 7.138 ms、HW 0.729 ms、分类 0.658 ms、cluster cull 0.768 ms、candidates 0.262 ms、stable bins 0.155 ms。SW/父 scope 的显示均值比约 **73.4%**，因此优先优化 SW 合理；这不是整帧占比。两个同名嵌套 Stream early 不能重复相加，500 帧历史里的条件 scope 也不保证样本集合完全一致。

截图相机 Eye=(6.456737,4.134918,-7.741189)，Center=(-2.162123,2.872261,-5.524320)，FOV=60，reversed Z。截图没有给出实际 shader 绑定、cut、驻留哈希及内部渲染尺寸，不能把它标为已完成的固定状态基准。

当前默认选择见 [VisibilityBufferPass.cpp](../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:1125)：协作装载、共享屏幕顶点开启，局部分桶关闭，选择 `streamClusterRasterWorkControlMain`。用户运行时可能改变设置，下一次采样仍须导出实际绑定，不能从 scope 名称推断 SPIR-V。

## 已经补齐的部分

- 首 wave 协作装载 SW 专用描述；共享唯一顶点投影、屏幕变换与 snap。
- 整数边函数按行/列增量步进；保留原始 triangle/visibility ID。
- cull 内 coverage/tessellation 强制 HW 与球体保守 fast-SW；不确定项才进入精确分类。
- HZB 相机运动失效问题已有修复，不能用历史异常工作量代表现在的实现。

历史 [局部分桶对照](ZorahFullLocalWorkBins.md) 中，相同顶点复用条件下，不分桶 SW 7.317 ms、分桶 8.215 ms，分桶增加约 12.27%。[M4](AgenticShaderOptimizationM4.md) 还记录独立 uint3 顶点数组候选的寄存器从 37 增到 41、shared 从 3512 B 增到 5560 B，候选已拒绝。这里引用已存报告，不是本次重新运行或重新校验原始包；不能承诺今天的编译结果相同。

## 与本地 Nanite 的具体差异

| 方面 | 当前 Metallic | UE 5.7.4 Nanite | 判断 |
|---|---|---|---|
| 普通 SW 工作组 | 每 cluster 128 线程；顶点和三角形直接用 lane 索引 | 常规配置 64 线程，按组大小循环处理顶点和三角形；其他 permutation 可为 32 | 最值得先做独立 A/B；减少空 triangle lane 的资源占用，但顶点阶段增加批次，不能预言 2 倍收益 |
| cluster 容量 | 128 vertices / 128 triangles | 上限 256 vertices / 128 triangles | Full 属性拆点可能先碰顶点上限，使三角形填充不足；需要 cook 直方图证实原因 |
| 变换与装载 | 首 wave 串联 active group、page table、payload 校验；描述就绪与顶点就绪各一次组同步；顶点代码重建相机基向量/投影参数 | 使用 view/instance 变换数据，普通共享顶点分支在顶点发布后同步 | 关注依赖装载链和重复 uniform 运算；不能把源码表达式直接当作驱动实际指令次数 |
| bbox 扫描 | 整个 bbox 的双循环，整数边增量 | 小范围矩形扫描；programmable 或 wave 中有宽 bbox 时进入 scanline | 真实尚未补齐的扫描差异；收益取决于宽 bbox 和细长三角形比例 |
| 深度计算 | 默认覆盖后保留原 barycentric 深度表达式；浮点增量关闭 | setup 预计算 DepthPlane，像素写入复用边值 | 有差距但精度敏感，已有失败/弱收益证据，优先级低于调度 |
| 输出原子 | 线性 buffer 上 64-bit packed depth/ID `InterlockedMax` | VBuffer 为 64-bit image atomic max；depth-only 有 32-bit 路径 | 两者都有原子；image/buffer 与局部性差异值得测量，不证明当前 atomic-bound |
| 分类 | 球体 fast-SW + 精确几何慢路径 | cull 内用 projected scale、EdgeLength、实例/变形缩放及 clipping 回退 | 后续可补 max-edge 元数据，但主要收益属于分类/上游，不能当作 SW 内核提速 |

源文件定位：

- Metallic [WorkControl](../Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang:28)、[128 线程入口](../Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang:135)、[装载/变换](../Shaders/Features/GPUDriven/StreamRasterCooperative.slang:15)、[bbox 与深度](../Shaders/Modules/GPUDriven/HybridRasterTriangle.slang:105)、[快速分类](../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:4293)。
- Nanite [线程组配置](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.usf:62)、[普通顶点/三角形循环](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.usf:682)、[cluster 上限](E:/UnrealEngine/Engine/Shaders/Shared/NaniteDefinitions.h:21)、[scanline](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush:230)、[自适应选择](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizer.ush:292)、[DepthPlane](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteRasterizationCommon.ush:313)、[原子写入](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteWritePixel.ush:20)、[分类](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteClusterCulling.usf:310)。

不能把 Nanite tessellation 的工作分配、voxel 队列或 programmable 材质路径当作普通 opaque SW 的实现。普通路径仍是一线程处理一个三角形的循环，并非全局像素任务队列。其 EarlyDepthTest 受 `NANITE_PIXEL_PROGRAMMABLE` 等条件控制，不能宣称普通 opaque SW 普遍依靠提前深度读取规避原子。

## 为什么先验证组大小与 setup，而非原子聚合

[9 月 20 日 Full 工作量记录](ZorahFullWorkloadAndWaitsResult.json) 的 HZB 有效、固定状态 WorkControl 样本：

| 指标 | 历史值 / 本次计算 |
|---|---:|
| SW cluster | 1,041,657 |
| 提交三角形 | 42,565,256，平均 **40.86 / cluster** |
| cluster 内唯一顶点处理量 | 103,089,190，平均 **98.97 / cluster** |
| triangle 数 / 128 槽总量 | **31.92%** |
| 背面、退化或空 bbox 等汇总拒绝 | 41,124,170，约 **96.61%** |
| 进入 bbox 的三角形 | 1,441,086 |
| bbox 样本访问 | 3,203,542 |
| 覆盖样本 / 原子尝试 | 638,149 |

31.92% 是逻辑 triangle 槽填充比，**不是硬件 active lanes 或 occupancy**；顶点阶段仍使用大量线程。96.61% 不是都能安全删除的小三角形，也没有区分背面与退化等原因。计数表示该历史视角的大量成本可能在装载、顶点变换和 triangle setup，而非最后的像素写入；仍只是优先级依据，不是当前 7.138 ms 的归因证明。

该记录约 91% 的非空 bbox 三角形面积不超过 4 像素（1,306,824 / 1,441,086）。因此，先量化单样本/小 bbox 路径，比直接将全部三角形改成 scanline 更合理。bbox 访问/覆盖约 5.02，但绝对访问量比顶点处理量小很多，不能把这比值等同于可获得的加速比。

Nanite 也会处理背面、空 bbox 和不可见像素；没有同资产、相机、质量、cut 的 UE 运行对照，不能声称 Nanite 具有某个固定倍数的工作量优势。Epic 也明确说明密集重叠几何会削弱 LOD/遮挡剔除效率：[官方内容性能说明](https://dev.epicgames.com/documentation/en-us/unreal-engine/working-with-naniteenabled-content)。

## 推进顺序与验收

1. **刷新当前 Full 视角基线，复用现有计数入口。** 保存截图相机与实际 render extent、DLSS/LOD/8 px 设置；固定 cut、驻留、jitter 和 HZB priming。导出实际 module/entry/device SPIR-V、early/late 工作列表和原始 depth/visibility。已有总计数继续使用，仅补 triangleCount/vertexCount 分布、各 setup 拒绝原因、非空但零覆盖比例、bbox 宽/高分布。细计数放在诊断 replay，正式计时关闭。
2. **SWR1：NVIDIA wave32 下的 32 / 64 / 128 线程组对照。** 保留一组一 cluster、相同记录与三角形 ID；顶点和三角形均按组大小循环。必须覆盖 0/1/31/32/33/63/64/65/127/128 个元素及所选 subgroup 的索引和同步契约，不按 lane 重新编号。分别测装载、顶点、triangle 阶段的资源/指令与总 SW 时间；不要同时叠加分桶、深度算术或 cook 修改。保留现有通用入口作回退，但不以其他厂商或 wave64 的性能作为本轮优化门槛。
3. **SWR2：缩减微三角形 setup。** 若零覆盖占比仍高，测试单样本 bbox 的精确覆盖快路径，覆盖失败时避开后续深度准备；审计编译后的相机基向量、tan、反射 winding 计算是否重复。若确有重复，再预计算 view/instance 常量，保持逐顶点运算顺序。先前 prepared-camera 组合没有稳定收益，因此必须拆成独立候选，不能直接重新启用旧方案。
4. **SWR3：矩形/scanline 双路径。** 仅在宽 bbox/细长三角形和循环长尾足够多时推进；沿用现有整数边判断与 top-left 规则，保守求行内区间，再做精确边测试。Nanite 的阈值与采样约定不能直接照搬。需要保留现有深度表达式，避免把扫描改进与浮点插值变化混在一起。
5. **SWR4：cluster 密度与可见工作量。** 统计完整属性导致的顶点拆分、顶点上限命中率、材质边界和 LOD 约束。如果确认大量 cluster 因 128 顶点上限而只有少量三角形，再考虑构建策略或位置/属性索引分离。256 顶点方案会跨 cook 格式、shader shared、HW mesh 输出、BLAS/CLAS 与驻留预算，不能只改一个常量；也不能牺牲法线/UV/材质保真或放宽 1.5 px 来报告同质量收益。

各候选采用至少三轮换序、相同状态的普通 timestamp 对照，报告 early/late SW、raster 合计和 Graph；保持原始 ID/覆盖/深度检查，再验证恢复漫游后的时序与高光。当前正确性门槛不因轻微深度差而自动放宽：[旧深度平面实验](ZorahFullPreparedSoftwareRaster.md) 已出现只有数 ULP 变化却改变 ID 的情况。

如需占用率、lane、shared conflict、原子或访存停顿归因，使用实际目标 dispatch 的硬件指标并记录精确名称/单位/范围。现有 [isolated replay](AgenticShaderOptimizationReplay.md) 已验证的是 MiniZorah，Full 不在验收范围，source snapshot 还有 1 GiB 上限；不要直接启动 Full 的全 buffer 克隆造成额外 VRAM 压力。先资格检查，必要时使用 Full in-frame 范围，并保留其非 isolated 的证据边界。SM active 不能当 occupancy，阶段级 shared conflict 不能直接归因到 WorkControl。

## NVIDIA 特调边界

NVIDIA warp 为 32 线程，组大小是 warp 的整数倍只是起点，不能保证较小组更快：[NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。这里借鉴硬件执行特征；Metallic 仍通过 Vulkan/Slang 执行，不假设 CUDA API 或资源配置可直接套用。

| 候选 | 目的 | 必须检查的代价 |
|---|---|---|
| 32 线程 / 1 warp / cluster | 所有线程参与描述装载，消除描述阶段等待的其他 warp；降低少三角形 cluster 的空槽 | 顶点最多四批、三角形最多四批；每 SM 驻留 block 数限制可能先于 warp 数限制 |
| 64 线程 / 2 warp / cluster | 减少空槽，同时保留两 warp 的顶点与 triangle 并行 | 组同步仍在，顶点最多两批，不能仅比较 triangle 阶段 |
| 128 线程 / 4 warp / cluster | 现有生产基线，一批完成最多 128 顶点 | 描述装载仅首 warp 执行，少三角形 cluster 的尾部利用率低 |

第一轮只改变组大小与必要的跨步索引。随后才对胜出候选分别尝试 warp shuffle、shared 布局、循环展开与 uniform 数据提升，并记录驱动实际寄存器分配、shared、spill、active warps/lanes、barrier 与总时间。避免对线程组、shared 数组、解码和深度同时改动而失去归因。

wave32 专用入口应显式验证 Vulkan subgroup 能力与实际 pipeline 配置，不能仅判断 vendor ID。即使只有一个 warp，也不能在存在共享内存交换时依靠隐式锁步直接删除同步；需要保留 Vulkan 内存可见性语义，验证独立线程调度下的正确性。NVIDIA 的官方说明同样要求对 warp 内共享数据交换处理同步：[Turing Tuning Guide](https://docs.nvidia.com/cuda/archive/12.0.1/turing-tuning-guide/index.html)。

具体资源门槛以 GB203 实测和驱动报告为准，不把数据中心 B200 的 SM/shared 参数当成 RTX 5070 Ti 参数，也不把 CUDA 理论 occupancy 当作 Vulkan 实测值。

## 暂不优先

- **全局三角形/pixel 队列与跨 cluster 排序**：增加前缀、scatter、临时驻留及流量，先证明当前局部分配无法解决瓶颈。
- **重开局部分桶或独立紧凑 shared 数组**：已有负面结果，需新假设与证据。
- **提前读深度或 tile 内原子合并**：先证明竞争/overdraw 占比；并发读写、等深 ID 胜出与同步成本必须正确。
- **浮点深度增量**：既有收益弱且改变 ID，继续默认关闭。
- **为 SW 性能重启有损几何压缩**：Nanite 确有压缩位置解码，但我方实验已归档；压缩不是当前已证明的首要解法。官方位置精度说明也强调边界一致性：[Nanite Technical Details](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-technical-details)。

本次仅新增研究文档，检查源码、已存统计和文档链接；没有新 Full GPU 计数、Nanite 同场景实测或性能收益声明。
