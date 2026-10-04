# 材质系统 Phase 0：当前行为基线

本阶段对应 `D:/Metallic_Material_System_Roadmap.md` 的 Phase 0。仓库已存在 M1/M2 的材质运行时工作，因此这里冻结的是 **2026-10-04 当前实现**，不声称恢复了重构前版本。既有 [仓库路线图](MaterialSystemRoadmap.md) 的 M 编号与外部文档的 Phase 编号不同。

## 固定工作负载

源配置为 [Cases.json](../LookDev/MaterialSystem/Cases.json)，三个独立图均只保留目标路径，避免比较图同时执行 PT 与 Deferred 污染时间。运行时不修改源图。

| 路径 | 场景和相机 | 尺寸 | 样本 / 深度 | 输出 |
| --- | --- | --- | --- | --- |
| OpenPBR PT | OpenPBRDefault glTF、scene sidecar；图内显式 eye/center/up/FOV/near/far | 768×768 | 4 spp × 256 帧 = 1024 spp，12 bounce | Reference.color |
| OpenPBR VBuffer Deferred | 同上，LOD 0，autoLod=false | 768×768 | 单次光栅表面采样；SH diffuse + prefiltered IBL；无渐进累计 | Deferred.color |
| RTXCR Chiang | Claire ponyTail_15vtx；固定 bounds-fit orbit yaw=10°、pitch=-4°、FOV=34°，资产 hash 固定 bounds | 768×432 | 4 spp × 256 帧 = 1024 spp，4 bounce | PathTrace.color |

OpenPBR 采用 scene sidecar 中的拆分太阳和 HDRI。Fiber 采用 studio_small_09_1k HDRI，强度 1.5，旋转 25°，显式空 punctual light list。环境和源资产在运行前后校验 SHA256。手动 EV100=0；原始输出是未曝光线性 RGBA16F/32F，预览统一 multiplier=1、sRGB transfer、无 tone mapping。原始 HDR 是 A/B 的权威数据，PNG 高光裁切仅供查看。Fiber 是现有 DOTS triangle BLAS / Chiang reference，不代表 strand visibility。

## 执行与复核

在已有 MSVC x64 开发环境、兼容 Release 配置执行：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests -j 8
python -B tools/Perf/MaterialBaseline.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/material-phase0-new
python -B tools/Perf/MaterialBaseline.py verify build/material-phase0-new
python -B tools/Perf/PlotMaterialBaseline.py build/material-phase0-new build/material-phase0-new-report
python -B tools/Perf/MaterialBaseline.py compare build/material-phase0-reference build/material-phase0-candidate
```

Python runner 依赖 NumPy；绘图额外使用 Matplotlib/Pillow，不进入默认 CMake 构建依赖。每次使用新目录，三次进程串行，默认每进程 600 秒超时。普通计时关闭 validation、Aftermath 和 Nsight injection；保留现有磁盘缓存，每个新 renderer 从第 0 帧开始，排除 0–31 帧，只统计 32–95 帧，最后在第 255 帧读回 HDR。GPU 图计时不包含最终 CPU 图像保存，不代表编辑器端到端帧延迟。独立进程是重复性统计单位，64 帧不能当作 64 次独立实验。

证据包保存 fixture、实际图、源代码 hash、工作区 diff、EXE/DLL hash、CMake cache、GPU UUID/driver、命令、日志、每帧 GPU 时间及 HDR。Manifest 检查文件集合和内容；任意输入在采集期间变化会拒绝出具合格基线。失败目录保留。A/B 要求 workload hash 与 GPU/driver 一致；不自动判定优化接受，也不把不交错的两个批次解释为统计显著加速。相同输入的三次图像误差随 verify 输出，跨版本误差由 compare 输出。

## 指标覆盖与缺口

| 路线图指标 | 当前采集范围 |
| --- | --- |
| Total deferred shading | Deferred 节点的 GPU timestamp，含分类、着色及该节点管理工作 |
| Material classification | 新增 GPU scope：reset/classify/indirect arguments 与相应 barrier |
| VBuffer resolve / Material evaluation / OpenPBR prepare / Direct lighting / Environment lighting | 均融合在 `Deferred shading` kernel；单独时间不可用，不能按整个 kernel 时间分摊 |
| VBuffer visibility | VBuffer 节点与已有子 scope，区别于 deferred 内的 resolve |
| VGPR/register pressure | 独立 pipeline executable 诊断已采集驱动 Register Count；它不是运行时 occupancy |
| Occupancy / texture latency / instruction count | 未采集；现有 WorkControl NvPerf runner 不支持本材质 workload，不能套用其数字 |
| Material bin occupancy / mixed-material tile ratio | 未读回；不能用源材质数量估算屏幕分布 |

父子 scope 有重叠，禁止相加。硬件诊断必须与普通计时分离。上述未采集项意味着外部路线图 Phase 0 的**完整性能指标验收尚未完成**；本批交付三条可重复 reference、GPU timestamp 基线、ABI 文档与可直接运行的 A/B 工具。Layered 参考等实际实现后再加入，不创建空白验收项冒充已有路径。

## 当前材质 ABI 与依赖

| 契约 | 生产者 | 消费者与约束 |
| --- | --- | --- |
| CPU payload | `ScenePathTraceResources.cpp` 从 scene 材质生成 `LegacyMaterialPayload`；`MaterialGeneration` 发布 | `SceneMaterial.slang::PathTraceMaterial`；720 bytes，不能把 C++ 对齐假设当作 Slang 自动匹配 |
| TextureInfo | `ScenePathTraceResources.cpp` 解析纹理索引、UV 与 NTC 映射 | `PathTraceTextureInfo`；48 bytes：4×uint（textureIndex/texCoord/ntcTextureSetIndex/ntcChannelMapping）+ 2×float4 UV transform；无效资源 UINT32_MAX |
| 参数布局 | 11×float4 = 176 bytes，随后 9×TextureInfo，offset 608 的 specular float4，再 2×TextureInfo | CPU static_assert 与 `material_runtime_gpu_abi` GPU probe 保持一致 |
| Program 身份 | `MaterialRuntime`：OpenPBR=1，RTXCR=2，payload `textureParams.z` | shader material runtime；0 为 legacy 推导。Program ID 不等于场景材质索引或 bin index |
| 材质 ID | 导入 scene 索引→GPUScene material；instance.identity.y 指向 GPUScene material；material.identity.z 指向 shading material | VBuffer 用 visibility record→instance→material→source material 查找；ray-hit 用命中几何的 materialIndex。resident/stream 解码及越界策略以对应 shader 为准 |
| 分箱 | `MaterialBinning.cpp` + `VisibilityMaterialBinning.slang` | 8×4 wave32 tile；背景/介电/导体/普通不透明/透射五类。bin={offset,count}，tile={tileIndex,laneMask}，各 8 bytes；间接参数 12 bytes/bin；count 为任务数，不是像素数 |
| 纹理绑定 | 场景上传与 `ParameterWriter`/资源 registry | typed resource handles、buffer spans 和 descriptor 表；shader 侧用 UV/显式 footprint。CPU encoded params 与 Slang 参数布局需一起变化 |
| OpenPBR | 材质纹理求值与法线映射生成 ResolvedInputs，`prepareOpenPBRMaterial` 调用 vendor prepare | prepared 是当前 wo 的临时状态；Projected Eval 已含 cosine，调用端不能重复乘；Sample 返回权重/PDF/event，独立 PDF helper 保留 |
| RTXCR | NV_materials_hair→payload；groom→DOTS triangle geometry；RTXCRMaterialAdapter | Chiang prepare/eval/sample；保留 authored normal/tangent，不套 Surface N·L；现有环境 NEE PDF 近似保持 |
| 生命周期 | MaterialBindingGeneration 保留 CPU snapshot 和 GPU 参数 buffer | RenderFrameContext/PreparedComputeDispatch 保留至提交完成；revision 变化重置相关历史；失败发布保留旧 generation |

关键源码：[LegacyMaterialPayload](../Source/Runtime/Render/Material/LegacyMaterialPayload.h)、[Slang payload](../Shaders/Modules/Material/SceneMaterial.slang)、[上传](../Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp)、[VBuffer resolve](../Shaders/Features/VisibilityBuffer/VisibilityBufferDeferred.slang)、[分类](../Shaders/Features/VisibilityBuffer/VisibilityMaterialBinning.slang)、[OpenPBR adapter](../Shaders/Interop/OpenPBRMaterialAdapter.hlsli)、[RTXCR adapter](../Shaders/Interop/RTXCRMaterialAdapter.hlsli)。

依赖边界：visibility/depth/rasterInfo/domain 必须来自同一 VBuffer；shading material generation 与 GPUScene source-index 映射必须一致；OpenPBR LUT、环境 SH/prefilter、light grid、阴影/RTAS 和纹理资源均影响结果。固定当前 TBN 规则：normal/geometryNormal 在法线贴图前不得按观察射线翻转，只能对最终 shading normal 单独 face-forward。

扩展材质、换编译器/驱动、改 scene/相机/纹理后必须建立新身份。此 shaderball 仅覆盖一个简单表面材料，不能宣称覆盖所有 OpenPBR lobe、复杂混合 tile、streaming 或全场景 VRAM 行为。

## 2026-10-04 实测与验收

硬件为 RTX 5070 Ti，驱动 616.92；复用 `build-scheduling-release` 的 MSVC Release 配置。普通证据位于 `build/material-phase0-20261004-c`；可视报告为 `build/material-phase0-20261004-final-report/report.md`，包括参考 PNG、GPU 时间序列、寄存器图及原始 JSON。`verify` 校验通过；首次输出目录和 `-b` 目录保留采集器初始化失败记录，不参与结果。

下表是三次**独立进程内 64 帧的中位数**，单位 ms：

| 范围 | 进程 1 | 进程 2 | 进程 3 |
| --- | ---: | ---: | ---: |
| OpenPBR PT 整图 | 9.52808 | 9.59707 | 8.74629 |
| OpenPBR Deferred 整图（含 VBuffer） | 0.588848 | 0.601136 | 0.574672 |
| VBuffer 节点 | 0.427040 | 0.437792 | 0.414640 |
| Deferred 节点 | 0.135744 | 0.136352 | 0.136096 |
| Material classification scope | 0.028512 | 0.028528 | 0.028480 |
| Deferred shading scope | 0.102976 | 0.103680 | 0.103552 |
| RTXCR Chiang 整图 | 0.746304 | 0.747888 | 0.736736 |

保留图中尖峰与进程漂移；这些数字不构成优化收益判定。PT A/A 不是逐位一致：后两次相对首个参考的 RGB RMSE 为 0.00045565 / 0.00052417，最大绝对通道误差 0.05389053。原因未定位，不能把此最大值直接设为未来回归容差。Deferred 与 Fiber 三次 HDR 完全一致。已目视检查三张参考；Deferred 与 PT 采用不同积分器，其差异不是本阶段修复目标。

独立 `METALLIC_VK_PIPELINE_STATISTICS=1` 诊断成功，日志为 `build/material-phase0-resources.log`，诊断次数 n=1：OpenPBR PT 的 Register Count=168，五个 Deferred executable 为 122/34/96/96/96，分类 reset/classify/arguments 为 16/24/16，Fiber=121。报告按 cache key 与 input/device SPIR-V 指纹保留身份，不按编译顺序猜五个变体对应的 feature class。驱动 Local Memory Size 含异常大值，保留原值，不解释为 spill。诊断帧时间没有混入上面的普通计时。

独立 Vulkan validation：`material_phase0_baseline`、`material_runtime_gpu_abi` 与两个 `material_binning_*_coverage`，**4 通过、0 失败、0 跳过**；日志未发现 VUID/Validation Error。证据验证器的 7 项测试通过，覆盖报告重复归档、文件篡改、帧缺失、诊断误混、scope 缺失、配置漂移和非有限 HDR。

- [x] 固定 OpenPBR VBuffer / PT 参考配置及图像。
- [x] 固定 RTXCR hair 参考配置及图像。
- [x] 三进程可复核 GPU timestamp 基线与独立寄存器诊断。
- [x] 材质 ABI、资源生产者/消费者与 A/B 工具。
- [ ] Occupancy、texture latency、instruction count 的硬件计数器。
- [ ] Material bin occupancy、mixed-material tile 比例的生产 workload 读回。
- [x] PT A/A 非逐位一致的原因及经过验证的回归判据：2026-10-04 [M1–M4 重新验收](MaterialSystemM1M4Acceptance.md) 确认初始异步环境占位帧导致采样序列变化。基线显式等待首次解码且逐帧断言 Ready 后，三个独立进程 A/A 全部逐位一致；三进程加最终完整序列共 12 张 HDR 对原冻结 run-0 逐位一致。上文保留修复前历史测量，不作为新容差。
