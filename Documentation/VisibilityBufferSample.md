# Visibility Buffer Sample

`MetallicGPUDrivenSample` 无参数启动时默认加载 `gpu-driven-sample`，复用 [实时渲染 pipeline](RealtimePipeline.md)，默认场景为 Git 仓库常驻的 `Asset/Sponza/glTF/Sponza.gltf`。链路为 `VisibilityBuffer → Deferred → DLSS-SR → AutoExposure → DLSS-NR（默认关闭）→ FinalBlit`，场景在编辑器中加载并使用共享 ViewConstants。入口命令：

```powershell
cmake-build-release-visual-studio\Source\MetallicGPUDrivenSample.exe --smoke-test
```

原始可见性调试模式通过 `--visibility-buffer`（或兼容别名 `--legacy-preloaded`）加载，Sample ID 为 `gpu-driven-visibility-buffer`，同样默认使用仓库常驻 Sponza。`--scene <path>` 可覆盖所选模式的场景。下面描述该调试模式的可见性 pass；`--usd`、StreamAsset 和地形选项保留各自的专用图与资源。

## 帧内数据流

```text
GPUScene → instance cull → compute cluster cull / stable bins
                             ├─ HW bins → AS / MS + visibility PS
                             └─ SW bin  → async compute atomic depth/ID
                                                        ↓ merge
                                                 R32Uint ID + D32 depth
                                                        ↓
                          HZB → late cull/raster → final visibility/depth
                                                        ↓ (optional)
                                          ID / depth / coverage display
                                                        ↓
                                     GPUDriven.color → FinalBlit → Viewport
```

1. `GPUDrivenCulling.slang` 先用当前剔除相机执行实例视锥测试，再用上一帧相机与上一帧完整 HZB 判断早期绘制候选。历史 HZB 判为遮挡的实例只延后处理，不能直接丢弃。
2. `VisibilityBuffer.slang` 默认在 compute 中执行 producer ownership、meshlet 视锥、normal-cone 和历史 HZB 测试，并按整个 cluster 分类。GPU 稳定压缩输出四个硬件材质箱和一个软件箱。硬件箱由 amplification shader 每组消费 32 个 cluster；支持时请求完整 Wave32 并压缩 payload，不支持时保留 subgroup/groupshared fallback。
3. 默认启用 [Hybrid Rasterizer](HybridRasterizer.md)：所有三角形都满足尺寸阈值的 cluster 直接 indirect dispatch compute 软光栅，再把原子 depth/ID 合并到相同附件；包含大三角形、需要裁剪的三角形或 alpha mask 的 cluster 继续走硬件。Mesh shader 每组输出一个 meshlet 的共享顶点与索引（上限 128 vertices / 128 triangles），通过 per-primitive `SV_PrimitiveID` 输出可见性 ID。Opaque 变体仅输出 position；`VISIBILITY_BUFFER_ALPHA_MASKED=1` 变体另外输出 UV / material index 供 alpha test 使用。Fragment shader 只写入 `R32Uint` ID，不进行材质着色。
4. 深度附件使用 `D32Sfloat`。Opaque fragment 允许 early depth；masked fragment 先按 alpha cutoff discard，不能强制 early depth 写入。四个 raster bucket 分别覆盖 opaque/masked 与单面/双面材质，BLEND 暂不进入此路径。
5. Compute 将第一阶段深度归约成当前 HZB：Reversed-Z 用 min，普通 Z 用 max。第二阶段用当前剔除相机和当前 HZB 重测被延后的实例及 meshlet，将恢复可见的 meshlet 补绘到同一 visibility/depth；早期已经绘制的 meshlet 不再重复绘制。
6. 完整深度再生成下一帧 HZB，直接保留最终 `GPUDriven.visibility`（R32Uint）和 `GPUDriven.depth`（D32Sfloat），不执行属性重建或材质着色。
7. 可选的 `VisibilityBufferComposite.slang` 直接读取原始 ID / depth，输出 `GPUDriven.color`（Rgba8Unorm）供视口调试显示，不改变原始输出。关闭可视化时只清空 color，不绘制全屏三角形。内置图将 `GPUDriven.color` 连接到 `FinalBlit.source`，样例的 `previewOutput` 为自动呈现输出 `FinalBlit.color`，JSON 的 `outputs` 数组为空。原始 visibility / depth 仍是节点输出，可供后续 Pass 使用；整数 ID 应先经过可视化再呈现。

两阶段 meshlet 划分通过重算不变的早期遮挡条件完成，不需要有容量上限的延后候选追加队列。
默认由 compute 在两个阶段分类并生成稳定的硬件/软件列表，AS 仅间接消费对应硬件箱。`asyncSoftwareRaster=true` 时软件箱在独立 compute 队列执行，与硬件光栅重叠，在深度合并和 HZB 前通过 timeline semaphore 汇合；无独立队列时自动串行。`clusterPrebin=false` 保留 AS 扫描候选与 Mesh Shader 按三角形分流的对照路径。
StreamAsset 的 cluster 使用同一阶段划分和保守遮挡规则。历史无效时全部通过早期遮挡测试，
第二阶段始终读取本帧第一阶段构建的 HZB；相机切换、尺寸变化及场景修改沿用 GPUScene 的历史失效机制。

`ConservativeOcclusion.slang` 集中处理保守边界：以包围球的相机空间 AABB 投影两个 Z 端点，
避免离轴透视投影低估屏幕范围；额外扩大一个像素以覆盖光栅舍入与时域抖动。
与近裁剪面相交、无效数值或无法证明遮挡时保留候选。HZB 使用逐级向上取整的尺寸，
采样 mip 必须将屏幕矩形覆盖在每轴至多两个 texel 内，四个角全部满足严格深度分离才判遮挡。
深度比较包含保守偏移，Reversed-Z 投影避免远处深度相减消失；变换后的球半径也覆盖 shear。

### SPD HZB 生成

`VisibilityBufferPass` 默认使用 `HzbSpd.slang`，这是针对 FP32 深度与现有线性 HZB buffer 的
[FidelityFX SPD 算法](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/) Slang 实现：
每组 256 个线程处理 64×64 深度块，默认使用 wave operations：前两级在各线程的寄存器中归约，
Morton 布局让相邻 2×2/4×4 像素落在连续 lane 中，后两级用 `WaveReadLaneAt` shuffle 归约。
仅把 4×4 的结果经 LDS 交给首个 wave，再用 shuffle 完成块内剩余两级。
每块归约暂存从 256 个 float（1024 字节）降为 16 个 float（64 字节），完整六级归约的
workgroup 屏障从 9 次降为 1 次；最后工作组选举仍保留独立的 device-scope 屏障。
全局计数器选出最后完成的工作组，由它生成 mip 6 之后的尾部。
一次 dispatch 同时复制全分辨率 mip 0 并生成其余所有 mip。早期 HZB 与最终历史 HZB 各调用一次。

所有尺寸逐级向上取整，源边缘复制，完整保留奇数尺寸的最后一行/列。普通 Z 使用 max，
Reversed-Z 使用 min；不使用均值、线性采样或 FP16。无效深度按远平面清屏深度处理。
由于 native `DescriptorHandle` 不能携带 HLSL `globallycoherent` 修饰，mip 6 与计数器通过
device-scope release/acquire 原子操作发布/读取，并以工作组屏障同步；没有全局自旋等待。
每次生成前通过 GPU buffer copy 清零设备内存中的计数器，资源随视图绑定延迟回收。

Wave 版本通过 `SubgroupId * SubgroupSize + SubgroupLocalInvocationId` 分配像素，
不假设 `SV_GroupIndex` 与 wave lane 存在对应关系。框架使用 SPIR-V 1.6，256 个 X 方向线程
满足 [Vulkan 完整子组要求](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-full-subgroups)。
仅在 compute 支持 basic/shuffle 且可能的 subgroup size 全部位于 16～256 时启用；
shuffle 始终在有完整参与者的 16-lane 区块内取值，覆盖 Wave32/Wave64 等不同大小。
不满足能力要求的设备自动使用 LDS 版本。本机实际 GPU 验证的 subgroup size 为 32。

`hzbSpdWaveOps=false` 可切回 SPD LDS 版本；`hzbSpd=false` 可切回逐 mip dispatch 进行对照。任一输入维度超过 4096 时自动使用原有路径，
保持完整 mip 链而不截断。独立 `GPUDrivenStreamAssetPass` 暂保留原有生成路径；
`VisibilityBufferPass` 内的 resident/stream 混合绘制共用 SPD 生成的 HZB。

本机 RTX 5060 / Wave32 的 Sponza 对照（关闭 validation，预热 16 帧后取 32 帧中位数）如下。
时间是整个 VisibilityBuffer pass，包含光栅、两次 HZB 和计数器清零，不是孤立的 HZB kernel 时间。

| 分辨率 | 逐 mip | SPD LDS | SPD wave |
| --- | --- | --- | --- |
| 799×293 | 0.181 ms | 0.125 ms | 0.114 ms |
| 1920×1080 | 0.360 ms | 0.291 ms | 0.290 ms |

这是一次对照采样；1080p 的整个 pass 基本持平，不能据此声称 HZB kernel 有同等幅度的加速。
Wave 版本减少的 LDS 占用和块内同步次数可由生成的 SPIR-V 确认。

测试：`hzb_spd_conservative_reduction` 在支持 wave 的设备上验证 168 个完整金字塔（wave/LDS 各 84 个），
包括 1×1、单行/单列、16/32/64 像素边界、799×293、1920×1080、4096×4095、NaN 和重复生成；
普通 Z 与 Reversed-Z 的每一级均与 CPU min/max 参考比较。不同 tile 的深度范围不同，以检查尾部归约。
`hzb_spd_visibility_equivalence_timing` 验证 SPD wave/LDS/逐 mip 的可见三角形一致，记录 GPU 时间，
并覆盖 4097×65 自动回退。

本 Pass 不再创建 OpenPBR compute 管线、LUT 或 deferred color buffer，也不依赖环境光子系统。材质贴图只上传 MASK 几何所需的 base-color alpha 贴图；这属于可见性判定，不是着色。`VisibilityBufferShading.slang` 暂保留源码供后续独立着色阶段使用，当前 Pass 不编译、不调度它。

独立着色阶段现由 [VisibilityBufferDeferredPass](VisibilityBufferDeferred.md) 提供，
使用新的 `VisibilityBufferDeferred.slang` 和共享 OpenPBR 光照函数。
本 Pass 另发布 `rasterInfo` buffer，供延迟节点读取实际观察相机及 scene identity；
resident LookDev 比较通过 `LookDev.exe --sample lookdev-vbuffer` 启动。

RenderGraph 类型为 `VisibilityBufferPass`，内置图节点仍名为 `GPUDriven`。加载旧 JSON 时自动将 `GPUDrivenPreviewPass` 迁移到新类型；下次保存时写入新名称。

法线锥轴的余因子变换必须按方向归一化，不能套用相机向量的绝对长度阈值。仓库 Sponza 的节点缩放为 `0.008`，余因子向量的长度平方约为 `4.096e-9`；旧阈值 `1e-8` 会将有效轴替换成固定 `+Z`，导致地板随相机运动误剔除。现在先按最大分量缩放再归一化，保留余因子的朝向符号；无法确定方向的退化轴不执行法线锥拒绝。

GPU 回归 `gpu_driven_cone_scale_invariance` 验证 16 组缩放/旋转下的轴方向及正背面判定；`gpu_driven_sponza_culling_equivalence` 使用报告问题的三个相机位置，比较关闭全部剔除、开启全部剔除、逐项关闭剔除的可见性图，并覆盖连续 HZB 帧。修复前其中一个视角有 19,917 个像素不同，修复后全部组合逐像素一致。测试另输出 `SponzaShaded.png` 供同视角材质图检查；该图移除 SR/NR，以隔离几何剔除和时域重建。

当前路径不创建 RTAS、不发起 ray query，也不计算几何阴影或几何环境遮挡。双面材质跳过 normal-cone backface 剔除。

## Visibility 数据契约与 Nanite 参考

参考本地 Unreal Engine 的 `NaniteDataDecode.ush::UnpackVisPixel` 与
`NaniteWritePixel.ush::WritePixel`，沿用“可见 cluster record + cluster 内 triangle”的索引式 visibility 思路：

- `ID = ((recordIndex + 1) << 7) | triangleIndex`，低 7 位容纳 128 个三角形，编码值 0 为背景。
- Record 引用 `VisibleClusterRecord`，而不是直接引用 meshlet：record 同时确定实例、cluster、geometry/page 和 producer。Resident 与 stream 使用互不重叠的 record 地址区间。
- `GPUDrivenRasterCommon.slang` 与 C++ `GPUDrivenRaster.h` 定义 ID 范围和记录格式。
- `GPUDrivenSceneCommon.slang` 为剔除、光栅和材质重建共享同一 GPUScene / push / params 布局，避免重复声明发生 ABI 漂移。
- 默认启用 cluster 预分箱的软硬混合光栅：硬件保留 `R32Uint` ID + 固定功能深度附件，软件写入 `uint64(depth, ID)` 原子缓冲并合并到附件；详见 [Hybrid Rasterizer](HybridRasterizer.md)。

## 管线与固定剔除相机

Shader 职责按模块拆分：

| 模块 | 职责 |
| --- | --- |
| `GPUDrivenCulling.slang` | Reset、实例剔除、逐 mip HZB 回退 |
| `HzbSpd.slang` | 64×64 分块、单次 dispatch 的完整 HZB 归约 |
| `VisibilityBuffer.slang` | AS meshlet 剔除、opaque/masked MS、visibility PS |
| `VisibilityBufferComposite.slang` | 可选的 ID / device depth / coverage 全屏调试显示 |

PSO 共用 `.cache/pso/VisibilityBufferPass.pso`，shader 内容及变体宏参与缓存键。
首次加载新 shader 会创建对应缓存；日志中的 `PSO cache status/hits/misses/stored/bytes` 可用于检查复用情况。

启用固定剔除相机后，Pass 锁存相机 pose、投影和裁剪参数。实例视锥/HZB、
meshlet 包围球和 normal cone 使用这台相机，viewport 相机仅投影幸存的 meshlet。
固定模式使用独立内部 visibility/depth 生成两阶段 HZB，避免误用观察相机深度。
第一阶段 HZB 保持不变，直至固定视图和观察视图的晚期绘制都结束后才更新下一帧历史。
StreamAsset 目前不绘制固定剔除视图，且其光栅相机仅支持透视 Reversed-Z；固定模式或不匹配的投影模式下，
保守关闭 stream 实例/cluster 的 HZB 测试，不使用其他相机的深度证明遮挡。

## 可调开关

`VisibilityBufferPass` 暴露以下运行时设置；剔除默认开启，固定相机默认关闭：

- `visualization`：默认 `meshlet`，运行时切换无需重建图，不使 HZB 历史失效。
- `instanceFrustumCull`
- `instanceHzbCull`
- `meshletFrustumCull`
- `meshletNormalConeCull`
- `meshletHzbCull`：默认开启，控制 resident meshlet / stream cluster 的两阶段遮挡测试，与实例 HZB 开关独立。
- `hzbSpd`：默认开启，使用单次 dispatch 生成 HZB；超过 4096 的输入自动回退。
- `freezeCullingCamera`：勾选时捕获当前相机作为固定剔除相机；取消勾选后恢复使用实时 viewport 相机剔除。

| Visualization | JSON 值 | 显示内容 |
| --- | --- | --- |
| Meshlet ID | `meshlet` | 可见 cluster record ID 的哈希色（区分实例与 producer） |
| Triangle ID | `triangle` | 完整 packed ID 的哈希色，区分 meshlet 内三角形 |
| Depth | `depth` | 当前观察相机的原始 device Z 灰度，不做线性化 |
| Coverage | `coverage` | ID 非零为白色，背景为清屏色 |
| Off (VBuffer Only) | `none` | 仅生成原始 visibility / depth，color 保持清屏色 |

旧 `mode` 属性仍可读取；旧 Shaded / Base Color 回退为 Meshlet ID，不会重新启用材质着色。Depth 在固定剔除相机模式下仍显示观察相机深度，而非内部剔除深度。

## 验证

Shader 编译测试覆盖 Wave32 / atomic fallback 以及 opaque / masked 两种 MS，并检查 SPIR-V：opaque 不含用户 varying，masked 恰好导出 UV / material 两个 location。渲染测试覆盖 alpha 裁剪、固定相机、两阶段 HZB、奇数尺寸、五种可视化及环境光独立性，并读回验证 resident/stream 共用的原始 visibility/depth 不因可视化切换而改变。

`gpu_driven_two_pass_occlusion` 在 GPU 上直接验证延后恢复、持续遮挡、无重复绘制、历史失效、近裁剪面、
普通/Reversed-Z 深度，以及 32 组包围球的独立表面采样和 shear 边界。
`gpu_driven_temporal_occlusion_equivalence` 比较两套独立历史，在移动相机、抖动、尺寸变化和透视/正交投影下，
验证开启/关闭遮挡的 30 帧 Sponza 三角形可见性逐像素一致。

```powershell
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --filter render_graph_gpu_driven_preview_shader_compile
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_preview_pass_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_sponza_visibility_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_alpha_mask_render
cmake-build-release-visual-studio\tests\MetallicRhiTests.exe --rhi-validation --filter render_graph_gpu_driven_mixed_producer_render
```
