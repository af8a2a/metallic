# MiniZorah 多线程组候选展开

2026-09-13。候选展开已从单线程组遍历整个 active cut，改为 GPU 间接调度的分块计数、块前缀和、稳定散射。保留 early/late 选择规则、候选顺序、原始 visibility record ID、候选截断及溢出计数，LOD 质量仍为 1.5 px。

## 实现

[streamClusterPrepareMain](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3398) 使用独立于 early/late `traversalPhase` 的 `activeBuildPhase`：

1. Setup：从 GPU active header 生成实际 active group 数和 count/scatter 间接参数。
2. Count：每 128 个 active group 一个工作组，每 lane 处理一组，保存一次计算的 phase mask 和局部候选偏移。Early 在此清理 retry mask；不足一块的 lane 不访问越界数据。
3. Prefix：对块计数求前缀和，生成分类与稳定分箱的间接参数。这里只处理块总数，保留一个 128 线程组。
4. Scatter：多组并行展开 mask，按 `activeGroupIndex * maxActiveGroupClusters + clusterBit` 写出原 record ID；每组和每 bit 的相对次序保持不变。

入口固定视角有 22,891 个 active group，count/scatter 分别使用 179 个工作组；近景 20,265 个 active group 使用 159 个工作组。原实现分别让单组内每 lane 串行扫描约 179 / 159 组，且会重复读取 phase mask。

[prepareStreamClusterCandidates](E:/metallic/Source/Runtime/Render/VisibilityHybridRasterizer.cpp:207) 负责四次 dispatch 和阶段依赖。Setup/prefix 写参数后转为 indirect read；count 完成后转回写状态供 prefix 使用；scatter 完成后只发布候选写入，参数保持 indirect 状态。GPU 上的实际计数驱动调度，不读回 CPU，不按容量发空工作组。

临时 mask、局部偏移和块计数分别复用尚未被消费的 bin 0、rank 和 block 区域，后续稳定分箱覆盖这些数据；retry mask 位于独立区域。没有增加容量级显存，只将候选参数缓冲由 24 B 扩为 36 B。Shader entry 与 PSO 数量不增加，继续使用已有 pipeline cache；源码变化按现有机制使旧 shader 缓存失效。

## 固定视角验证

RTX 5070 Ti、1920×1080、1 GiB 页面预算、1.5 px、异步 HW/SW、Vulkan validation，RelWithDebInfo。使用同步离屏 GPUDriven + MaterialResolve；计时不含启动编译、输出读回和检查点验证。下面比较各自运行约 5 秒的收敛检查点。

| 固定视角 | 指标 | 基线（ms） | 并行展开（ms） |
| --- | --- | ---: | ---: |
| 入口 0 秒 | early candidates | 0.753 | 0.281 |
| 入口 0 秒 | late candidates | 0.802 | 0.245 |
| 入口 0 秒 | candidates 合计 | **1.555** | **0.525（降低 66.2%）** |
| 近景 15 秒 | early candidates | 0.681 | 0.261 |
| 近景 15 秒 | late candidates | 0.724 | 0.236 |
| 近景 15 秒 | candidates 合计 | **1.405** | **0.497（降低 64.6%）** |

两个视角的相机、active group/selected cluster 数、early/late 候选及软硬箱计数均相同；可见超标 refinement 为 0。最终 PNG 字节完全相同：

- 入口 `roam-5.png`：`9824bc17b155a4f4466603fd488502c53c4e742cae5f8dca8b29669ace09aaf4`
- 近景 `roam-5.png`：`b181ca6e019e5d95a28f7857054de6fdc80b00aa066d0fb871302122f3344f75`

入口固定 10 秒 GPU P95 为 10.840 → 9.090 ms，近景为 9.265 → 8.189 ms。检查点区间包含邻近准备步骤，不能当作单个 shader 的纯 kernel 时间；也不能与此前含编辑器 UI、不同视口尺寸的 Nsight capture 直接比较绝对值。

## 60 秒漫游

| 指标 | 基线 | 并行展开 |
| --- | ---: | ---: |
| GPU P50（ms） | 9.122 | 7.536 |
| GPU P95（ms） | 11.299 | 9.094 |
| GPU P99（ms） | 12.383 | 9.692 |
| 计时帧 | 4,485 | 5,748 |
| 完整检查点 | 60 | 60 |

GPU P95 降低 19.5%。全部检查点通过相机、参考 cut、覆盖、页池预算、请求合法性及流式错误检查；两轮均包含冷启动与转向后的临时未收敛阶段，不能解读成整条路线始终没有超标 refinement。

路线按渲染墙钟推进，帧数、页面到达和中间 cut 会不同；GPU 时钟未锁定，整轮收益并非固定输入的单 kernel 加速。基线还存在一次长帧离群值，不将其消失归因于本次候选改动。默认沙箱下基准测试的 PSO 保存受到限制，编译发生在路线计时前；可写缓存回归另行通过。

## 回归与复现

- `Metallic`、`MetallicGPUDrivenSample`、`MetallicRhiTests` 构建通过。
- 新增 [stream_cluster_candidates_stable_parallel](E:/metallic/tests/rhi/StreamClusterCandidateTests.cpp:16)：14 组 GPU 与独立 CPU 参考比较，覆盖 0/1/127/128/129、超过 128 个块的前缀、四种实例状态、恢复与 retry 选择、高位 cluster bit、空/缩小列表、候选超容量截断和二维分类参数。预先污染 scratch 与列表尾部，检查稳定 ID、未写尾部与 retry 内容，并复用真实 stable-bin 阶段。
- 另有 18 项相关回归全部通过（83 秒）：完整 MiniZorah 首帧/VBuffer/质量审计、GPU/reference LOD cut、StreamAsset、混合光栅的覆盖/等深/异步行为、两帧槽、resize/reload 和持久化 LOD PSO cache。日志没有 VUID 或 validation error。
- 两个固定视角各 10 秒和 60 秒漫游完成。没有改动 cook、1.5 px 误差判定或分类/光栅算法。

```powershell
& E:/metallic/build-relwithdebinfo/tests/MetallicRhiTests.exe `
  --gtest_filter=RhiRendering.stream_cluster_candidates_stable_parallel --rhi-validation `
  --output-dir E:/metallic/build-relwithdebinfo/minizorah-candidates/recheck
& E:/metallic/build-relwithdebinfo/minizorah-candidates/RunCase.ps1 `
  -Name reproduce -Seconds 60 -LatencyOnly 0 -Transitions 1
```

[结构化结果](E:/metallic/Documentation/MiniZorahParallelCandidatesResult.json)、[解析脚本](E:/metallic/build-relwithdebinfo/minizorah-candidates/Analyze.py)、[新增用例日志](E:/metallic/build-relwithdebinfo/minizorah-candidates/candidate-tests.log)、[18 项回归日志](E:/metallic/build-relwithdebinfo/minizorah-candidates/regression.log)、[最终漫游报告](E:/metallic/build-relwithdebinfo/minizorah-candidates/parallel-roam-60/MiniZorahRoamingReport.json)。

候选准备的串行长链已拆开；下一步仍是此前 profile 指出的分类工作组织：先批量剔除，再对存活 cluster 分类，减少 lane 0 加载造成的整组等待和重复投影。
