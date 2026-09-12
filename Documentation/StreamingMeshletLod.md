# 统一流式 Meshlet LOD

`VisibilityBufferPass` 的常驻和 StreamAsset 几何，以及独立 `GPUDrivenStreamAssetPass`，共用 `MeshletLodMetric.slang` 的屏幕误差公式和四个运行时设置：`autoLod`、`lodPixelError`（默认 1.5 render px）、`lodBias`（默认 0）、`lodLevel`。旧 `enableGpuLodSelection`、`selectedLodLevel` 配置仍可读取，新键优先；可视化模式不再决定是否启用自动选择。

误差阈值范围 0.05–16，bias 范围 -4–4，有效阈值为 `error * exp2(bias)`。正交、近裁剪面、偏轴和非均匀实例变换与常驻路径一致。冻结相机只固定裁剪/LOD 相机，光栅及 Deferred 使用实时渲染相机；内部尺寸变化时重捕获冻结相机。两条光栅路径使用相同的 temporal jitter。

## 元数据与资产兼容

- StreamAsset v9 的 group 元数据为 44 字节：原 36 字节之后增加每 cluster 的 refinement offset 和 terminal flags。
- `refinedGroups()` 按 cluster 保存全局 group ID，独立于 payload 页驻留；运行时据此构造去重的父组 CSR。
- `primitiveTerminalGroups()` 返回完整终端集合，包括在较细 LOD 提前终止的分支。旧连续 fallback 字段仅作格式兼容；光栅和 CLAS fallback BLAS 使用新的完整集合。
- v9 打开时不加载几何 payload。v8 可读取，但首次打开必须解码各页以恢复拓扑；大资产建议重建为 v9。旧 partial checkpoint 自动失效，新 partial v8 保存拓扑。
- 检查引用范围、同 primitive、`refined < owner`、严格降低的 LOD、误差单调和 terminal 标记。当前 builder 合并前代 group sphere，保证全驻留条件下的投影误差单调。

## 有效 frontier

```text
desired(g) = terminal(g) OR projectedError(g) > targetPixels
reachable(g) = terminal(g) OR all(active(parent) for parent in parents(g))
active(g) = desired(g) AND reachable(g) AND drawable(page(g))
emit(cluster) = active(owner) AND (no refined OR NOT active(refined))
```

手动模式以 `group.level >= lodLevel` 替代误差比较。组可以有多个父组；所有父组必须生效，才能原子替换它们对应的粗表示。存在缺失祖先时，即使更细页仍驻留，也不能绕过祖先参与输出。页需求使用同一可达性规则逐层推进；PendingUpload 不可绘制，也不会重复请求加载。

每个 primitive 的所有终端页就绪前，不发布不完整 cut。初始化按全部实例（含隐藏实例）预留终端页和输出容量，预算不足返回明确错误，不跳过部分几何。页面预算还预留一个普通流式页（资产全为终端页时不需要）。

GPU 依次执行 reset → BVH frontier/count → prefix → sparse emit → indirect finalize。每个实例按拓扑逆序评估组，实例之间并行；输出按 instance/group 顺序稳定排列。压缩后的 active group 和 cluster mask 继续转换为现有 `(instanceId, clusterId)` 可见记录，供 cluster 分箱、早晚 HZB、Mesh Shader 和异步软件光栅消费。

精细输出超过容量时，prefix 在写出任何记录前，将整帧退回完整终端 cut；连终端容量也不足的非法 GPU 输入产生空输出和错误标记，禁止越界写入。正常运行时初始化已拒绝该预算。CLAS 动态 BLAS 使用选中 mask，备用 BLAS 使用同一终端集合，备用 BLAS 存储不足也明确失败。

## BVH 遍历加速

运行时从常驻 group 元数据构建每 primitive 的有序 BVH4，随拓扑上传一次，无需读取 payload 或修改 v8/v9 资产格式。每个节点 40 字节，包含包围球、最大误差、最大 LOD、终止标记 OR、逃逸索引及叶子 group 范围；叶子最多 4 个 group。

节点采用前序布局，叶子按 group ID 降序访问。聚合球用双精度构建并将半径向外舍入，节点的投影误差是子组误差的保守上界。透视判断还根据矩阵、坐标范围和平移量增加世界空间半径余量，避免大坐标浮点消减使变换后的父球漏掉子球；该余量只用于 BVH。节点误差满足阈值时直接跳到逃逸索引；手动模式按最大 LOD 剪枝。终止标记确保完整回退集合始终被访问。浮点比较留有保守余量，叶子仍使用原来的精确选择条件。

这个顺序保证所有父组先于子组生效，不需要有限容量的遍历栈或排序；共享父组、跨层终止、缺页与 PendingUpload 的规则均保留。此 BVH 按误差裁剪，视锥/HZB 继续由下游光栅流程处理，因而加速前后的完整 cut、请求和稳定输出顺序一致。未使用 BVH 的线性路径保留为差分测试对照。

frontier 同时记录所有 active group 的稀疏列表，包括已被细化替代的祖先。mask 计算及 emit 只遍历当前列表；每帧只清除前一帧列表中的状态，随后再处理隐藏或恢复，避免旧祖先错误放行后代。首次使用通过 phase 8 初始化设备状态缓冲，支持跨越单行 dispatch 上限的二维派发。

## 调试与验证

现有 `AfterTraversal` / `AfterPass` checkpoint 提供：

- `streaming.<pass>.activeHeader`：live count、capacity、overflow、terminalFallback、invalidCapacity。
- `streaming.<pass>.activeGroups`：page、instance、LOD 和逐 cluster 选择 mask。
- `streaming.<pass>.lodState`：所有实例的四 word header（fine count、output prefix、terminal count、ready）位于缓冲开头。随后每实例有 `2 * groupCount` 个 active/mask word、四个 sparse header word（all-active count、visited BVH nodes、tested groups、reserved），以及最多 `groupCount` 个降序 active group ID。
- `streaming.<pass>.pageTable`：关联请求和可绘制状态。列表容量中的空闲条目不是有效几何。

测试入口：

```powershell
build/tests/MetallicSceneTests.exe --gtest_filter=SceneImport.MeshletStreamAsset:SceneImport.MeshoptCompressedMeshletStreamAsset
build/tests/MetallicRhiTests.exe --gtest_filter="*meshlet_lod*"
build/tests/MetallicRhiTests.exe --gtest_filter="*gpu_driven_mixed_producer_render*"
```

合成参考测试检查原子几何覆盖、共享父组、跨层 terminal、缺页、PendingUpload、容量回退；GPU 差分测试对照选择、请求和间接参数。真实 Bunny 测试对照 CPU frontier 与 GPU group mask，并核对同帧 VBuffer 的每个有效 ID，同时切换透视/正交、标准/Reversed Z、误差、手动 LOD 和硬件/异步混合光栅。

## 当前成本

frontier 临时状态为每实例 32 字节，加该实例每组 12 字节；只扫描访问的 BVH 节点及叶子组，mask/emit 成本随 active 列表大小变化。由精细切换到粗糙 LOD 的首帧仍需清除前一帧的精细列表，后续帧不再清理所有组。BVH 在 primitive 实例之间共享，每节点 40 字节。

有序 BVH 保持拓扑及输出顺序，但空间聚合可能比重新排序的空间 BVH 更松；近景或高误差要求下可能访问全部节点。每实例仍由单个线程遍历，GPU 请求优先级、跨实例任务调度和自动驻留预算尚未优化。测试中的 group 检查数减少不等同于整帧耗时收益。

当前可达性方案保留已激活的祖先页。很小的预算可以维持完整粗 cut，但可能无法达到指定像素误差；进一步释放完全被替代的祖先 payload、合并请求和优先级调度仍是后续优化。

2026-09-12 的 RTX 5070 Ti 验证中，80 组 CPU 覆盖检查、37 组 GPU 差分与 16 组收敛后的真实 Bunny 对照通过。透视 0.05 px 达到 550 个 cluster，16 px 为 18 个；手动最粗为 1 个。硬件与异步光栅的有效 ID 完全一致，无需使用深度 tie 容差；透视和正交冻结选择相机后移动渲染相机，cut 保持不变且画面正确变化。结果保存在 `StreamMeshletLodSceneReport.json` 与对应 PNG 中。

独立 StreamAsset pass 在可见性、变换和尺寸更新时复用 Runtime 与驻留缓存；场景内容或资源配置变化才重新初始化。最终主程序、RHI 和 SceneTests 构建通过，CLAS/隐藏恢复/两次尺寸变化烟测，以及混合 resident/stream 生产者回归通过（开启 Vulkan validation）。

BVH 版本验证：主程序、RHI 和 SceneTests 构建通过，9 个相关回归测试全部通过，包括 146 组线性/BVH GPU 差分、16 组真实 Bunny 对照、冻结相机、CLAS 和混合生产者。CPU 额外覆盖分散包围球、剪切、镜像、阈值附近比较，以及 `1e8` 坐标消减导致近裁面漏选的回归；GPU 初始化检查跨越 65535 工作组的第二行末尾哨兵。

同一 Bunny 资产共有 50 个 group、19 个 BVH 节点。透视 16 px 阈值时访问 9 个节点、检查 7 个 group，输出仍为 18 个 cluster；正交 16 px 和手动最粗层级检查 4 个 group。0.05 px 近景仍检查全部 50 个 group。硬件与异步光栅选择及可见 ID 一致。对应结果在 `.tmp/bvh-lod/StreamMeshletLodSceneReport.json`，回归日志在 `.tmp/bvh-lod/regression2.log`；这些是选择工作量数据，没有据此推断整帧加速比例。
