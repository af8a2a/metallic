# MiniZorah：硬件光栅复用唯一顶点

日期：2026-09-13。承接候选展开并行化、剔除与软硬分类重组。

## 实现

预分箱 StreamAsset 硬件路径改为每个 cluster 一个 64 线程 Mesh Shader 工作组，输出最多 128 个唯一顶点和 128 个索引三角形。每个顶点只加载、变换和输出一次。原来每个 cluster 使用两个 64 三角形工作组，逐三角形输出三个顶点；满 128 三角形 cluster 的顶点投影/输出次数由 384 次降至最多 128 次。这个数字是满 cluster 的上限对比，不代表场景平均顶点复用率。

逐三角形 visibility ID 改从 primitive output 的 `SV_PrimitiveID` 传给 fragment，避免共享顶点的 flat 属性混用不同三角形 ID。单双面标记在整个 cluster 内一致，仍由顶点属性传递。原始 record ID、箱内稳定顺序、等深覆盖规则与 MaterialResolve 解码保持兼容。

硬件 indirect arguments 同步改为每 cluster 一组。未预分箱的兼容路径仍保留每 cluster 两个 draw slot，继续沿用 `params.drawTaskCount / 2` 的记录容量换算；每个分块内部也使用索引顶点。旧软件队列用组共享投影坐标构造三角形，预分箱及纯硬件路径跳过这次共享坐标 barrier。

当前 Slang 2026.1.2 为 fragment `SV_PrimitiveID` 生成 SPIR-V `Geometry` capability。独立 `GPUDrivenStreamAssetPass` 补齐与 VBuffer 相同的 `geometryShader` 能力要求，对应测试设备启用该功能。Editor 与渲染图 preview 已启用，不需要改变全局 RHI 接口。

没有修改 cook、页面格式、LOD/cut、分类阈值及 visibility 命名空间，也没有增加全局缓冲容量或生产 PSO 数量。

主要代码：

- [StreamAsset Mesh Shader](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:3320)
- [逐图元 ID](E:/metallic/Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang:449)
- [硬件分箱 indirect 参数](E:/metallic/Shaders/Features/VisibilityBuffer/VisibilityHybridRaster.slang:177)
- [独立 pass 功能检查](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetPass.cpp:438)
- [GPU 光栅等价测试](E:/metallic/tests/rhi/StreamIndexedMeshTests.cpp:19)

## 正确性与工作量

新增测试保留旧的重复顶点 shader 作为参考，执行 8 组输入 × 3 种路径，共 24 对实际 GPU 绘制，逐位比较 R32Uint visibility 与 D32Sfloat depth。覆盖预分箱硬件、兼容纯硬件及旧软件队列，0/1/63/64/65/127/128 三角形、顶点索引 127、无效索引回退、近远裁剪、等深重叠、单/双面、透视/正交、正向/反向 Z、高位 ID、late 阶段与独立渲染相机 jitter。全部对照相同，且验证了非空覆盖、第二个三角形分块和软件队列实际参与。

MiniZorah 两个固定视角的最终 cut 与 PNG 完全相同，visible over-target 为 0，最大可见误差分别为 1.499921 px 和 1.499972 px。

| 固定视角 | 硬件 cluster | 原硬件工作组 | 新硬件工作组 | 软件 cluster |
| --- | ---: | ---: | ---: | ---: |
| 入口，路线 0 s | 42,879 | 85,758 | 42,879 | 24,354 |
| 近景，路线 15 s | 40,334 | 80,668 | 40,334 | 24,356 |

固定视角最终 PNG SHA-256：

- 入口：`9824bc17b155a4f4466603fd488502c53c4e742cae5f8dca8b29669ace09aaf4`
- 近景：`b181ca6e019e5d95a28f7857054de6fdc80b00aa066d0fb871302122f3344f75`

两次 60 秒漫游均通过 60 个检查点。完整构建与最终验证结果见下方验证记录。

## 性能观测

环境：RTX 5070 Ti，1920×1080，1 GiB 页面预算，1.5 px，预取、低延迟与 completion uploads 开启，软硬光栅异步。基线为本轮修改前已完成分类重组的代码。固定视角每次运行 10 秒，阶段计时取 2.5 秒后的 6 个稳定检查点中位数。

`AfterStreamEarlyRaster` 包含硬件、软件及汇合/屏障；以下数字是这个合并区间，不是隔离的硬件 kernel 时间。整图 GPU 计时覆盖 GPUDriven 与 MaterialResolve，排除编译/启动、检查点读回及图片输出，不含 presentation blit。

| 近景固定视角指标 | 原路径 | 唯一顶点 | 观测变化 |
| --- | ---: | ---: | ---: |
| Early 软硬光栅合并区间中位数 | 1.406176 ms | 0.888032 ms | −36.8% |
| 整图 GPU P50 | 4.019136 ms | 3.811040 ms | −5.2% |
| 整图 GPU P95 | 4.533408 ms | 4.435904 ms | −2.2% |
| 整图 GPU P99 | 4.839008 ms | 4.736256 ms | −2.1% |

入口基线出现明显外部负载波动（GPU P95 38.115 ms），因此仅用于正确性比对，不用于声称性能提升。唯一顶点入口运行的整图 GPU P50 为 3.598016 ms。

| 60 秒漫游指标 | 原路径 | 唯一顶点 |
| --- | ---: | ---: |
| 计时帧数 | 7,769 | 7,488 |
| GPU P50 | 4.387488 ms | 3.857920 ms |
| GPU P95 | 5.306304 ms | 4.616128 ms |
| GPU P99 | 5.771104 ms | 25.295136 ms |
| GPU 最大值 | 7.502848 ms | 75.364800 ms |

漫游 P50/P95 下降约 12.1%/13.0%，但 P99 明显变差，不能据此宣布长尾改善。GPU 未锁频且与后台应用共享；一次所有 Metallic 测试退出后的观测仍有约 73% GPU 占用。这个现象说明测量存在干扰，不能证明某个具体尖峰来自后台应用。漫游按墙钟推进，两次运行的中间 cut 与 IO 时序也不完全相同。

下一次独占 GPU 的同帧 Nsight 对照应确认硬件 mesh 阶段的顶点输出、寄存器/共享内存与占用率，并单独复核 P99。当前可确定的是硬件工作组减半、满 cluster 顶点投影次数降低，以及图像和质量保持一致。

## 验证记录与复现

- RelWithDebInfo 构建 `MetallicGPUDrivenSample`、`Metallic`、`MetallicRhiTests` 成功。
- 最终 21 项相关回归全部通过（104.882 秒，开启 Vulkan validation 与 async compute），没有跳过项，日志扫描未发现 Vulkan 验证告警/错误。包括独立 StreamAsset pass、VBuffer、全场景首帧、质量审计、候选/分类、稳定分箱与超过 65,535 组的间接调度、旧队列、shader reload、resize、双帧槽、LOD 持久化缓存。
- 新增 raster equivalence 测试的 24 对 visibility/depth 比较全部逐位相同。
- 最终日志：[final-regression.log](E:/metallic/build-relwithdebinfo/minizorah-indexed-hardware/final-regression.log)。早期日志中的 `SV_PrimitiveID` 功能验证告警已通过补齐设备功能配置和独立 pass 检查解决。

性能原始目录：[minizorah-indexed-hardware](E:/metallic/build-relwithdebinfo/minizorah-indexed-hardware)。其中保存原 shader 快照、每次运行的日志、PNG、MiniZorahRoamingReport.json 与 RunCase.ps1；结构化汇总：[MiniZorahIndexedHardwareResult.json](E:/metallic/Documentation/MiniZorahIndexedHardwareResult.json)。

重跑固定视角：

```powershell
& ./build-relwithdebinfo/minizorah-indexed-hardware/RunCase.ps1 -Name indexed-fixed-15-repeat -Seconds 10 -FixedView 15 -LatencyOnly 0 -Transitions 0
```

重跑 60 秒漫游：

```powershell
& ./build-relwithdebinfo/minizorah-indexed-hardware/RunCase.ps1 -Name indexed-roam-60-repeat -Seconds 60 -LatencyOnly 0 -Transitions 1
```
