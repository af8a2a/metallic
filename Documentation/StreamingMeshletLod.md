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

GPU 依次执行 reset → frontier/count → prefix → emit → indirect finalize。每个实例按拓扑逆序评估组，实例之间并行；输出按 instance/group 顺序稳定排列。压缩后的 active group 和 cluster mask 继续转换为现有 `(instanceId, clusterId)` 可见记录，供 cluster 分箱、早晚 HZB、Mesh Shader 和异步软件光栅消费。

精细输出超过容量时，prefix 在写出任何记录前，将整帧退回完整终端 cut；连终端容量也不足的非法 GPU 输入产生空输出和错误标记，禁止越界写入。正常运行时初始化已拒绝该预算。CLAS 动态 BLAS 使用选中 mask，备用 BLAS 使用同一终端集合，备用 BLAS 存储不足也明确失败。

## 调试与验证

现有 `AfterTraversal` / `AfterPass` checkpoint 提供：

- `streaming.<pass>.activeHeader`：live count、capacity、overflow、terminalFallback、invalidCapacity。
- `streaming.<pass>.activeGroups`：page、instance、LOD 和逐 cluster 选择 mask。
- `streaming.<pass>.lodState`：每实例四个 header word（fine count、output prefix、terminal count、ready），随后为紧凑的每组 active/mask 对。
- `streaming.<pass>.pageTable`：关联请求和可绘制状态。列表容量中的空闲条目不是有效几何。

测试入口：

```powershell
build/tests/MetallicSceneTests.exe --gtest_filter=SceneImport.MeshletStreamAsset:SceneImport.MeshoptCompressedMeshletStreamAsset
build/tests/MetallicRhiTests.exe --filter meshlet_lod
build/tests/MetallicRhiTests.exe --filter gpu_driven_mixed_producer_render
```

合成参考测试检查原子几何覆盖、共享父组、跨层 terminal、缺页、PendingUpload、容量回退；GPU 差分测试对照选择、请求和间接参数。真实 Bunny 测试对照 CPU frontier 与 GPU group mask，并核对同帧 VBuffer 的每个有效 ID，同时切换透视/正交、标准/Reversed Z、误差、手动 LOD 和硬件/异步混合光栅。

## 当前成本

本次统一选择的正确性与数据契约。frontier 临时状态为每实例 16 字节，加该实例每组 8 字节；计算仍扫描所有实例的 group，尚未实现工作量随可见节点缩减的 BVH 遍历、GPU 请求优先级或自动调整驻留预算。大场景还需测量并优化选择成本，不能仅凭减少光栅 cluster 数推断整帧性能收益。

当前可达性方案保留已激活的祖先页。很小的预算可以维持完整粗 cut，但可能无法达到指定像素误差；进一步释放完全被替代的祖先 payload、合并请求和优先级调度仍是后续优化。

2026-09-12 的 RTX 5070 Ti 验证中，80 组 CPU 覆盖检查、37 组 GPU 差分与 16 组收敛后的真实 Bunny 对照通过。透视 0.05 px 达到 550 个 cluster，16 px 为 18 个；手动最粗为 1 个。硬件与异步光栅的有效 ID 完全一致，无需使用深度 tie 容差；透视和正交冻结选择相机后移动渲染相机，cut 保持不变且画面正确变化。结果保存在 `StreamMeshletLodSceneReport.json` 与对应 PNG 中。

独立 StreamAsset pass 在可见性、变换和尺寸更新时复用 Runtime 与驻留缓存；场景内容或资源配置变化才重新初始化。最终主程序、RHI 和 SceneTests 构建通过，CLAS/隐藏恢复/两次尺寸变化烟测，以及混合 resident/stream 生产者回归通过（开启 Vulkan validation）。
