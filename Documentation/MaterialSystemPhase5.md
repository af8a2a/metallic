# Material System Phase 5 — 统一 Lighting Kernel

Phase 5 将 Surface 直接光累积收敛到一个 Slang generic，实现 CPU 选择具体 Program、结构化 Program Key 与设备内可执行管线缓存。OpenPBR、Debug Lambert 和新增的理想 Debug Mirror 在真实 GPU 上共用同一个 `shadeSurface`、灯光循环和路径延续调用。

## 统一接口与生产接入

[SurfaceLighting.slang](../Shaders/Modules/SurfaceLighting.slang) 提供：

- `integrateDirectLighting<TPrepared, TLights>`：只消费 `IPreparedSurfaceClosure`。光源枚举/采样和可见性由 `ISurfaceLightProvider` 提供；循环负责 projected BSDF、入射光、可见性和光源选择 PDF 的累积。
- `shadeSurface<TMaterial, TLights>`：每 hit 调用一次 evaluate、一次 prepare，再进入直接光循环。返回已解析材质、Prepared 和 radiance，供积分器继续散射。
- `shadeEvaluatedSurface`：接收已求值结果，供需要先处理 normal debug、unlit 或获取映射后的 hit/RNG 的生产入口使用，避免再次读取材质。
- `sampleSurfaceWeight`：通过同一个 Prepared 接口取得路径权重和统一事件标志。

`IPreparedSurfaceClosure` 新增 shading normal、默认 projected eval、PDF、weighted sample 和 emission 桥接。普通 BSDF 通过 `f * abs(cos)` 和 `f * abs(cos) / pdf` 默认实现接入；OpenPBR override 使用原生 projected value / weight，保留 Phase 4 的数值语义和背面发光抑制。`IWeightedPreparedSurfaceClosure` 保留为兼容标记。`SurfaceSamplingContext` 显式提供 throughput、波长和外部 IOR；无需额外状态的 Closure 默认忽略它。

三条生产路径已移除各自的直接光循环：

| 路径 | Provider 保留的原有行为 |
| --- | --- |
| OpenPBR PT | ReGIR/punctual 选择、RNG 推进、选择 PDF、阴影透射与介质堆栈；保留 throughput-first 乘法顺序 |
| VBuffer Deferred | cluster local/global 光源枚举、选中光源的 SIGMA 可见性 |
| ray-query realtime/direct | grid/non-grid 光源枚举、SIGMA 或 ray-query shadow fallback |

材料在循环外求值，循环中没有 `switch(materialType)`。Provider 可访问场景及遮挡物以追踪阴影，但不能重新 evaluate 当前表面的 Material Program。authored geometry normal/TBN、normal mapping 顺序和现有纹理上传 ABI 不变。

这里统一的是 Surface 直接光内核与材质到散射的入口。环境 SH/prefilter 近似、PT 环境 NEE/MIS、介质与 Fiber 积分器继续保留现有实现；不声称整个渲染器已成为一个模型无关积分器。OpenPBR 仍只支持 Radiance，Importance 的拒绝行为不变。

## CPU 特化与 Program Cache

[MaterialExecutable](../Source/Runtime/Render/Material/MaterialExecutable.h) 的 `specializeSurfaceMaterialProgram` 由 CPU 选择 OpenPBR、DebugLambert 或 DebugMirror 的具体 Slang 类型，通过编译定义实例化 generic。Shader 不按实例 ID 分支选择模型。第三种实现是有独立散射语义的理想镜面：连续方向 eval 为零，sample 返回 delta reflection、单位离散概率和染色反射权重；没有将 Lambert 改名充当 LayeredSlab。

`MaterialProgramKey` 包含 `definitionHash / irHash / specializationSignature / domain / qualityProfile / targetCapabilities`：

- definition 来自显式 Definition identity，缺省从模块、入口和参数 ABI 派生。
- IR identity 来自实际 SPIR-V 内容；specialization 包含 profile、debug mode、宏定义和完整常量/资源布局。
- domain、quality 和目标 capability 要求分别进入 key，缓存还按实际 Device identity 隔离。
- 实例编号、实例参数值和纹理句柄不进入 key；资源布局中的 descriptor capacity 等静态要求会进入 key。

编译先经过现有 Slang 磁盘缓存及依赖校验，再查设备内 executable cache。相同 key 且完整 SPIR-V/布局一致时，共享 `ComputeKernel` 的同一个 immutable implementation，不创建第二条 Vulkan pipeline。Artifact 另外持有不可变 `ComputeResourceEncoder`，资源契约校验与 executable 分离。缓存 lookup/build/publication 受锁保护，失败的编译或布局校验不会替换调用者现有 Kernel/Artifact。`ScenePathTracePass` 的现有编译路径接入该缓存。

成功的新 executable 分配新的 generation。缓存使用 weak ownership；pass/artifact 持有当前程序，已记录 dispatch 持有自己的 kernel 和参数附件，旧 generation 可在 GPU 工作完成后退役。`ComputeKernel` 的普通复制共享 executable；clear/reinitialize 仅替换该句柄，不修改其他持有者的 implementation。缓存不是永久驻留策略：所有 artifact owner 释放后，后续请求允许重新创建 executable。

三模型的 CPU 类型特化目前由共享诊断场景验证，生产场景继续使用既有 OpenPBR 入口。把场景 MaterialInstance 映射到异构 Program bins 并调度，是 Phase 6 的范围。

## 验证方法

新增 `material_surface_lighting_framework`：192×128 的解析球体/平面场景、两个参数实例、一张真实 GPU 纹理、最多三次路径散射。三种 Program 只替换 CPU 编译选择，场景、阴影查询、generic 光照与延续代码相同；各运行 1/8 灯并保存原始 RGBA32F 和 PNG。

- 同一模型第二次编译请求必须命中同一个 Artifact，pipeline build count 不增长。
- 同一个 Program 处理两个不同参数实例；增加灯数不能增加参数与材质纹理读取。
- 输出必须有限且非负，真实 sample 数与能量不能退化为零。
- Mirror 采样方向与独立 CPU reflection 公式比较，要求 pdf=1、Reflection|Delta；1/8 灯 HDR 必须完全相同，可见性查询为零。

`material_runtime_inflight_reload` 增加实际 executable cache hit、generation/IR key 变化和 weak artifact 退役断言。它将旧 dispatch 阻塞在 GPU semaphore 上，经历编译失败、布局失败和成功重载后，释放旧 artifact，再让旧工作完成；分别核验旧输出和新输出。现有 `material_runtime_generations` 验证 1000 instances 共享两个注册 Program。

逻辑原子计数不是硬件访存统计。诊断图采用低采样路径追踪，噪声用于检查数值/调度契约，不作为最终画质基准。Mirror 的 delta 反射验证也不代表 OpenPBR 已覆盖 delta 或 Importance 传输。

构建与回归命令（x64 VS developer shell）：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests -j 8
.\build-scheduling-release\tests\MetallicRHITests.exe '--gtest_filter=*material_*:*visibility_buffer_deferred_openpbr*:*stream_material*:*texture_primary_ray_cone_pixel_footprint*:*scene_ray_tracing_position_fetch_authored_tangents*:*render_graph_scene_path_trace_material_textures_preview*' --rhi-validation --output-dir build/material-phase5-final '--gtest_output=xml:build/material-phase5-final/Tests.xml'
```

完整渲染回归包括 Phase 0 的 OpenPBR PT / Deferred / RTXCR Chiang 各 256 帧原始 HDR，使用冻结基线比较；构建和 shader 编译结果单独记录，不等同于视觉或 temporal 验证。

## 2026-10-04 实测验收

MSVC Release `MetallicRHITests` 和 `Metallic` 编辑器目标构建成功；最终运行 35 项，**34 通过、1 跳过、0 失败**。跳过项为 `zorah_stream_material_probes`，未配置 `METALLIC_ZORAH_Z4_PROBES` 指向其 cooked `probes.json`。本次没有该专用场景的运行时结果。

三模型同一诊断场景的计数如下；1/8 灯的参数读取、纹理请求和纹理 Load 均完全相同：

| Program | 参数读取 | 材质纹理请求 | 材质纹理 Load | 1 灯枚举 | 8 灯枚举 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenPBR | 16287 | 130296 | 32574 | 16287 | 130296 |
| Lambert | 16947 | 16947 | 16947 | 16947 | 135576 |
| Mirror | 17802 | 17802 | 17802 | 17802 | 142416 |

模型间 hit 数不同，来自各自 BSDF 路径延续；不是工作量相同的性能比较。Mirror 实际产生 17802 个 delta reflection 事件，连续直接光不触发阴影查询；1/8 灯原始 HDR 相同。每个模型仅构建一个 executable，重复请求命中同一 Artifact；在途重载、失败保留与退役验证均通过。

完整生产 HDR 对比冻结的 `material-phase0-20261004-c/run-0`：

| 场景 | RGBA 逐位一致 | RGB RMSE | RGB max absolute |
| --- | --- | ---: | ---: |
| OpenPBR PT | 否 | 0.0003726577 | 0.05089766 |
| OpenPBR Deferred | 是 | 0 | 0 |
| RTXCR Chiang | 是 | 0 | 0 |

三条场景各 256 帧，所有输出有限，三张同曝光预览已检查。PT 差异低于冻结 Phase 0 的 run-0 对 run-1/2 A/A 差异（RMSE 0.000455647 / 0.000524175，max 0.05359751 / 0.05389053），符合本场景已有随机波动尺度；不宣称跨进程逐位确定性或其他场景的通用阈值。

最终回归未报告 Vulkan VUID 错误。环境存在已有 layer manifest 缺失和旧 validation layer 的 OMM fallback 提示。未进行大型场景压力、长时间 temporal/denoiser 稳定性或 OpenPBR Importance 验证；没有渲染性能提升结论。

本地证据保留在忽略的 `build/`：

- `material-phase5-build.log`、`material-phase5-editor-build.log`：RHI 测试与编辑器构建。
- `material-phase5-final.log`、`material-phase5-final/Tests.xml`：最终 35 项回归与 skip 原因。
- `material-phase5-final/SurfaceLighting.txt`、三个 `*MaterialProgram-{1,8}.rgba32f/.png`：共享光照的计数及六张输出。
- `material-phase5-final/MaterialBaseline.json` 和三张生产 `.rgba32f`：256 帧 HDR。
- `material-phase5-report/comparison.json`、`compare.py` 与 PNG：冻结基线比较和预览。
