# Material System Phase 6 — 稀疏 Material Program Binning

Phase 6 新增 Program 驱动的 Visibility Buffer 分箱，保留 8×4 Wave32 空间 tile、lane mask 和 fused material/closure/lighting。生产 Deferred 默认使用新路径；`programBinning=false` 保留原固定五分类路径用于兼容与同负载 A/B。

## Program 表与生产调度

[MaterialBinningDesc](../Source/Runtime/Render/MaterialBinning.h) 接收两个新字段：

- `materialProgramBins`：以 source MaterialInstance 索引的 CPU 表，值为当前 active executable 的 dense bin slot。
- `programBinCount`：active table 的长度，含 slot 0 的背景 Program，最多 4096。零表示旧五分类路径。

GPU 沿用 resident/stream visibility 解码，执行 `Visibility → GPU scene instance → preview material → source MaterialInstance → program bin`。不再读取材质参数进行新路径的 GPU BSDF 分类；非法 visibility 仍归背景，越界 source material 仍遵循 Deferred 的 material-0 fallback。CPU 拒绝不完整、越界和占用背景 slot 的 instance 表。

[ScenePathTracePass](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) 将实际编译后的完整 `MaterialProgramKey` 与 `ComputeProgram` 句柄登记，按 key 去重建立 active table。当前生产 Surface compiler 仍选择已有 OpenPBR 的背景、dielectric、conductor、opaque 和 transmission 特化；CPU 将当前实例解析到这些具体 executable。只有当前实例需要的 Program 进入 active table，实例值和纹理句柄不会产生新 shader。

材质 generation 改变时重建 instance-to-program 表；shader reload 清除旧 key/句柄和表，再绑定新 executable。间接消费者按 active table 的具体管线执行，bin 的数字不再表示 BSDF 类型，也没有 full-screen `switch(programId)`。CPU push 和 dispatch 列表改为按 active bin count 分配。

该调度接口可接收任意已经编译且具有兼容资源布局的 Surface Program；48 个不同静态特化 Program 的 GPU 验收使用此生产分箱器。现有资产注册器仍只提供此前已有的内置模型；本阶段没有新增 48 种编辑器材质，也没有完成任意 Slang/graph 材质的资产导入。M2 Value Program 继续采用原有 general kernel fallback，避免用静态因素错误分类其动态输出。

## 稀疏任务生成

[VisibilityMaterialBinning.slang](../Shaders/Features/VisibilityBuffer/VisibilityMaterialBinning.slang) 的新路径有五阶段：

1. Reset 清零每个 active bin 的 offset/count。
2. Count 每个 wave 只枚举本 tile 中实际出现的 Program ID，每个 ID 产生一个 ballot mask。
3. Allocate 对 bin count 做 prefix sum，分配连续任务区间并重置 scatter cursor。
4. Scatter 再次解析同一 immutable visibility/instance 表，写入 `{tileIndex, laneMask}`。
5. Arguments 为每个 bin 写 GPU indirect groups，超过 65535 个 X workgroup 时延伸到 Y。

task 的 `programBin` 由所属 bin 的连续区间隐含提供，避免每条重复存储。每个有效 lane 恰好归一个 mask，边缘无效 lane 仍参与 wave 操作但不进入任务。背景和空 bin 也有合法间接参数。

任务缓冲是一个全局列表，预留容量为 `TileCount × min(ProgramBinCount, 32)`；实际任务由 prefix 分配，没有给每个 Program 单独预留一整屏区间。因一个 tile 最多只有 32 个非空 Program，容量上界独立于超过 32 的 Program 数量，即按补齐后的像素数有界。控制缓冲为 O(ProgramCount)，实例映射为 O(MaterialInstanceCount)。不创建 full-screen Closure Buffer。

`MaterialBinningParams` ABI 升至 `0x4d42494e00000005`，CPU/Slang 结构均为 144 字节。Count、Allocate、Scatter 之间的 buffer dependencies 使用现有 RenderGraph access plan；scratch 和实例映射通过 frame completion 保活，仅复用已完成帧的 allocation。映射使用 HostUpload buffer，提交前 flush。新增 shader 入口同步加入可选预热请求清单，预热仍不是默认构建依赖。

当前 prefix 采用单线程扫描最多 4096 个 bin，分类读取 visibility 两次；这是可验证的初版，不能据此宣称比五分类更快。

## GPU 验证

新增 `material_program_binning_sparse` 实例化 **48 个静态 Surface executable 加一个背景 Program**。它们是编译期参数不同的 procedural Lambert/Mirror 定义，各有不同 SPIR-V；共同调用 Phase 5 的 `shadeSurface` 和路径权重接口。CPU 选择具体管线，shader 输出自身静态 Program ID，防止仅把不同 bin 标签写回却始终执行同一程序。

- 同一屏幕混合 48 个 Program，包括一个 tile 中大量不同 Program 的情况。
- 每个像素读回 Program ID、原子 shading 次数和 fused lighting 值，与独立 CPU 预期比较；必须恰好写一次。
- 实例从 257 增到 4097、实例映射每帧变化，继续复用同一组 executable；执行过程不得增加 pipeline build count。
- 检查 prefix offset、非空任务数、mask、空 bin、非法 visibility、source material fallback、部分 tile、纯背景和跨 65535 的二维 indirect dispatch。
- 旧 typed/untyped 五分类覆盖测试继续保留；生产材质编辑刷新测试覆盖 generation 更新。

性能比较使用完全相同 fixture、材质参数、具体五个 executable、输出核验和分辨率。只切换旧固定五分类与稀疏路径；统计 GPU `Program classification` scope，排除 shader 编译、CPU 参数上传、后续 shading 和 readback。每组先运行两帧，再采六帧；不将分箱时间解释为端到端渲染时间。

构建与完整材质回归（x64 VS developer shell）：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests Metallic -j 8
.\build-scheduling-release\tests\MetallicRHITests.exe '--gtest_filter=*material_*:*visibility_buffer_deferred_openpbr*:*stream_material*:*texture_primary_ray_cone_pixel_footprint*:*scene_ray_tracing_position_fetch_authored_tangents*:*render_graph_scene_path_trace_material_textures_preview*' --rhi-validation --output-dir build/material-phase6-final '--gtest_output=xml:build/material-phase6-final/Tests.xml'
```

## 2026-10-04 实测验收

MSVC Release 的 `Metallic`、`MetallicRHITests` 和可选 `MetallicShaderCompiler` 构建通过。最终材质回归 **35 项通过、1 项跳过、0 失败**；跳过项仍为缺少 `METALLIC_ZORAH_Z4_PROBES` cooked 数据的 Zorah 专用探针。分箱的五个可选预热请求全部命中并成功读取 SPIR-V 缓存；没有运行或宣称完成全项目 shader warmup。

48 Surface Programs 的逐像素 ID、恰好一次 shading、CPU lighting reference、prefix 和 indirect 参数核验全部通过。覆盖 1×1、8×4 纯背景、17×9、512×256、513×257 和 4097×1025，各运行八帧；最后一个尺寸产生 131841 个同 Program tile task，跨越 65535 的 X 维限制。257/4097 instances 共享 49 个 shading executable 和一个测试 reset executable，改变实例映射没有增加管线数。保存的 `ProgramLighting.png` 已检查，它是极度混合的逐像素程序诊断图，不是最终画质展示。

性能设备为 NVIDIA GeForce RTX 5070 Ti、驱动 616.92，Vulkan validation 开启、shader debug symbols 关闭，使用预热后的相同负载。GPU 分箱 scope 六帧中位数：

| 五 Program（含背景）相同负载 | 旧五分类 ms | 稀疏 Program ms | 增加 ms |
| --- | ---: | ---: | ---: |
| 512×256，257 instances | 0.030656 | 0.042496 | 0.011840 |
| 513×257，4097 instances | 0.030544 | 0.044832 | 0.014288 |
| 4097×1025，4097 instances，密集单 Program | 0.182144 | 0.299056 | 0.116912 |

同屏 48 Surface Programs 加背景的新路径，在 512×256 / 513×257 的混合 fixture 分别为 0.149888 / 0.137072 ms。它们的任务数不同，不能与五 Program 表格当作等工作量加速比。当前新路径在旧五分类负载下更慢；收益是可扩展的 Program 调度与有界任务容量。没有锁定 GPU 时钟，原始样本保留供复核，不作端到端性能提升结论。

生产 HDR 每条场景渲染 256 帧，与冻结 Phase 0 `run-0` 比较：

| 执行上下文 | OpenPBR PT | OpenPBR Deferred | RTXCR Chiang |
| --- | --- | --- | --- |
| 完整 36 项测试运行 | RMSE 0.0005933642，max 0.05309039 | RGBA 逐位一致 | RGBA 逐位一致 |
| 独立 `material_asset_phase0_equivalence` 第一次 | RGBA 逐位一致 | RGBA 逐位一致 | RGBA 逐位一致 |
| 独立 `material_asset_phase0_equivalence` 第二次 | RGBA 逐位一致 | RGBA 逐位一致 | RGBA 逐位一致 |

整套运行的 PT RMSE 略高于此前冻结的 A/A 范围，不能把它标为通过旧误差阈值。两次独立进程复测均与基线逐位一致，说明差异依赖测试执行上下文；本次未追踪上下文差异的根因，也不声称整套测试的跨进程确定性。所有 HDR 有限，三张生产预览已检查；本阶段影响的 Deferred 在三次运行中均逐位一致。

最终回归未报告 Vulkan VUID 错误；仍有既有 layer manifest 缺失和旧 validation layer 的 OMM fallback 提示。未验证 Zorah 专用数据、大型场景内存压力、长时间 temporal/denoiser 稳定性、非 Wave32 GPU 或新的资产模型导入。

本地证据位于忽略的 `build/`：

- `material-phase6-build-final.log`、`material-phase6-warmup.log`：构建及五入口缓存检查。
- `material-phase6-final.log`、`material-phase6-final/Tests.xml`：完整回归与 skip 原因。
- `material-phase6-final/ProgramBinning.txt`、`ProgramLighting.png`：逐组 GPU 时间、任务数与诊断输出。
- `material-phase6-report/binning-timings.json`：原始时间样本和中位数。
- `material-phase6-final/MaterialBaseline.json` 和 `.rgba32f`：完整回归的原始 HDR。
- `material-phase6-repeat1/`、`material-phase6-repeat2/`：两次独立 256 帧 HDR 和测试结果。
- `material-phase6-report/comparison.json`、`pt-repeats.json`、`repeat-exact.json` 和 PNG：误差矩阵、逐位核验及预览。
