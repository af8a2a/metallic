# MiniZorah：遍历与预算控制

2026-09-16。基线 `f22b659`，RTX 5060 / 8 GiB / 驱动 610.47。

本轮保留 1.5 render px 误差定义和完整覆盖契约，改进 tile 遍历、输出容量分配和上传批次。没有加入路径追踪 pass，也没有通过降低目标画质取得性能数字。

## 实现

### 有序 tile 层级

`MeshletLod.cpp::buildMeshletLodTiles()` 在同级、最多 64 个 group 的 tile 之上建立 BVH4。包围球、最大误差、最大层级和 terminal 标志使用保守聚合；包围球半径向外舍入。GPU 可以通过 escape index 跳过整段不需要细化的 tile，正式需求和预取共用这套层级。

叶子仍按原来的 group 顺序排列，同一 tile 内不存在父子依赖。每个叶子执行完后保留组屏障，后续细级仍只在全部父级可绘制时进入安全 cut。没有引入会乱序处理共享父级的工作队列。小于等于 4 个 tile 的 primitive 保持平坦结构，避免额外内节点成本；旧 cook 不需要重建。

质量回放新增 `traversalVisitedNodes`、`traversalTestedGroups` 和 `traversalFlatTileBaseline`。后者按相同有效实例计算原平坦 tile 循环本应检查的数量；它不是 vk_lod_clusters 的节点数，也不是经过剔除的 group 数。

### 容量不足时按实例保留完整 cut

prefix 先为每个实例预留完整表示，再以稳定实例顺序分配剩余细节容量。可以容纳的实例保留原细节 cut，其余实例输出完整 terminal cut。细节表示比 terminal 更小的实例直接保留细节。

每实例 ready 字段的 bit 1 标记 terminal 回退，CPU/GPU active-header ABI 不变；`padding2` 记录发生回退的实例数。两轮并行 prefix 为混合结果生成连续输出偏移。计数先饱和再相加，避免极端输入溢出 uint32 后误判为容量充足。连完整基线表示都无法容纳时仍明确报告无效容量并停止输出，不能悄悄漏画部分实例。

这是确定性的实例级准入，不是最优背包分配：顺序中较大的细节 cut 可能让后续实例也回退，剩余空间不一定填满。尚未加入按屏幕收益重排、逐 group 容量分配或动态放宽 LOD 误差阈值。

### 上传同时受页数和字节限制

新增 `maxUploadBytesPerFrame`，默认 **8 MiB**，`0` 表示不限字节；实时 sample JSON 显式设置为 8,388,608。两个流送 pass 都读取此属性，Streamer 以转换后的 GPU payload 字节数计费，保留原来的页数上限。

同一帧多次调用共享已使用的字节额度。被延后的同步页和异步已解码页保留原队列和资源所有权。为防止大页永久饥饿，空帧允许一个超过字节额度的页独占上传；这条例外不允许同帧第二个页再次绕过额度。它约束上传数据量，不保证 CPU 毫秒数，也不替代几何、CLAS、BLAS 的驻留容量预算。

## 验证

- Release 构建 `MetallicRhiTests` 和 `MetallicGPUDrivenSample`。
- 23 项专项测试开启 Vulkan validation 通过，覆盖原有 365 组 CPU/GPU cut 对照、共享父级、缺页、视锥、预取、HW/SW 等价、页发布和回收。
- 新的多实例容量测试覆盖 257 个实例、24 组容量组合，包括混合回退、根容量不足、精确边界、更便宜的细节表示和 uint32 极值。
- 新的 tile 层级测试对 16,383 个 group 检查完整叶子顺序、同级约束和 32 组视图阈值下的需求一致性；粗视图访问量必须小于原 tile 数的四分之一。
- 新的上传预算测试覆盖同步／异步、无限额度、单页超过额度、同帧重复调用、延后重试和实际设备字节计费。
- 真实场景 `meshlet_lod_stream_scene_runtime_cut` 关闭 validation 连续三轮通过。

真实场景 validation 存在间歇性工具异常，不能报告为稳定通过：本机 `VkLayer_khronos_validation.dll` 在 `CoreChecks::ValidateReservedRangeOverlap` 内访问异常。临时异常处理器与本地 PDB 将调用链定位为 `PreCallValidateCmdBindResourceHeapEXT → DescriptorHeap::bind → TraversalPass::dispatch`，发生在本次 frontier dispatch 之前；也观察到相同配置通过的轮次。临时诊断代码已移除，没有禁用产品默认验证功能或修改外部依赖。原始栈保留在 `.cache/traversal-budget/scene-repeat.log`。该函数的[上游实现](https://chromium.googlesource.com/external/github.com/KhronosGroup/Vulkan-ValidationLayers/+/aaf283db8660069e0ea74f671737a104368d4a33/layers/core_checks/cc_descriptor.cpp#5716)用于交叉核对函数职责；当前证据不能代替更新验证层后的完整复测。

## Zorah 回放

使用仓库 `RunMetallicCfgReplay.ps1` 和原 `RunVkMiniZorahRoam.py::make_route()` 路线，Replay SHA-256：

`feb79872154850af32db25a54ba3d22b48b9a04a10f7f2e8dadaf19f98f2f2d2`

修改前后各两轮 3,000 帧计时，以及修改后独立 3,000 帧质量检查。完整实时图，输出 1920×1080、DLSS Quality 输入 1280×720、LOD 1.5 render px；几何／CLAS／动态 BLAS 预算仍为 1024／512／256 MiB，使用相同 60,916,791,801 字节 cook。相机每帧位置固定，不含 UI 或 present。

原始证据目录：`.cache/traversal-budget/before`、`.cache/traversal-budget/after`。测试日志：`specialized.log`、`scene-verified.log`。本机 Unity 进程仍持续占用 GPU，因此完整帧时保留在机器可读报告中，不用于宣称可归因的加速比。Streamline 退出阶段可能停住，沿用脚本在完整报告和测试终态落盘后回收自身进程的规则；`Process.json` 保留强制回收标记。

[机器可读报告](E:/metallic/Documentation/MiniZorahTraversalBudgetResults.json) 保留两轮完整帧时分布、GPU 竞争、上传分布和 15 个质量检查点。

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 最终视图遍历检查量 | 平坦循环应检查 289,427 个 tile | 实际访问 129,332 个层级节点（减少 55.31%） |
| 最终 active groups / selected clusters | 19,394 / 162,989 | 19,394 / 162,989 |
| 最终驻留 / CLAS 页 | 9,290 / 9,290 | 9,290 / 9,290 |
| 已用几何字节 | 150,612,224 | 150,612,224 |
| LOD topology 字节 | 172,573,172 | 173,553,812（增加约 0.94 MiB） |
| LOD state 字节 | 162,274,640 | 162,274,640 |
| 单帧上传峰值，m1 / m2 | 9,763,872 / 9,677,952 B | 8,387,856 / 8,377,888 B |
| return_hold Host P50，m1 / m2 | 34.283 / 29.536 ms | 30.503 / 30.163 ms |
| return_hold GPU P50，m1 / m2 | 34.336 / 29.342 ms | 29.510 / 30.296 ms |

遍历量的对照是同一视角和有效实例集合的平坦算法工作量推导，与实际新层级 GPU 计数相比；没有将不同相机或 vk_lod_clusters 的节点混在一起。15 个检查点的访问量减少范围为 **55.31%–56.03%**。该计数包含新内节点，不等于 GPU 时间减少同样比例。

质量轮启动第 29 帧有 2,612 个可见待细化项，第 59 帧及后续检查点均为 0。末端最大可见误差约 1.499928 px，CLAS backlog 为 0，BLAS 静止 cut 复用仍生效。回放的常规大容量配置没有触发 active-group fallback；实例预算压力行为由专项容量测试覆盖，不能把本路线当作低容量实景压力验证。

Unity 的非空闲 3D 引擎样本 P50：修改前 69.75% / 88.23%，修改后 71.77% / 72.14%。因此上述帧时是观测值，**不能归因于本次代码的净加速或回退**，也没有剔除较慢的一轮。


## 复现

```powershell
Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json `
  -OutputRoot .cache/traversal-budget/rerun -Realtime -QualityWithoutValidation

python Tools/SummarizeMetallicCfgReplay.py `
  --before .cache/traversal-budget/before --after .cache/traversal-budget/after `
  --quality .cache/traversal-budget/after/quality `
  --output Documentation/MiniZorahTraversalBudgetResults.json
```

后续仍需独立推进跨实例任务调度、稀疏 frontier 状态和基于实际压力的误差控制。本轮没有把有序层级跳跃称为 persistent traversal，也没有把上传字节额度称为总显存预算。
