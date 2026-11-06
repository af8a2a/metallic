# Virtual Shadow Maps P0 原型

2026-10-25。对应 [VSM 实施方案](VirtualShadowMapsResearch.md) 的 P0。

## 实现范围

`VirtualShadowMapPrototype` 是可复用的离屏 Vulkan 运行时原型。输入为世界空间三角形、独立正交光源投影、固定虚拟页表及接收点，不依赖主相机可见列表或 RTAS。它验证 D32 物理图集路线，尚未接入生产 GPUScene/meshlet 流送或默认实时图。

一次录制依次执行 GPU 计数清零、triangle/page 工作列表生成、indirect 参数生成、批量脏页清理、批量深度光栅和接收点查询。CPU 不逐页发 draw；清页与绘制各使用一次 `drawMeshTasksIndirect`。工作项、页表和投影通过 ResourceRegistry 绑定，shader 与 PSO 由 ShaderRegistry 获取。

Mesh shader 在映射物理图集前，对三角形做四个虚拟页 XY 平面的裁剪，最多输出 7 个顶点和 5 个三角形，保留 UV、深度和绕序。近远平面交给硬件裁剪；单面剔除与 MASK 在 fragment 中完成。无需 clip-distance、片元原子写入、task shader 或 RayQuery。

材质使用现有 `MaterialValueProgramSet` / `CoverageProgram`，包括 base alpha、显式提供的 alpha texture 和生成的 custom Coverage。P0 不执行完整 Surface/Closure。输入仅接受 OPAQUE/MASK；位移需已体现在传入顶点，未烘入的 displacement 设置明确拒绝。

## 输入和资源契约

- 投影为 row-major world→clip 的正交矩阵，最后一行必须为 `(0,0,0,1)`。深度固定 reversed-Z：near=1、far=0、clear=0、GreaterEqual。使用正高度 viewport，`uv = clip.xy * 0.5 + 0.5`，无额外 Y 翻转。
- 页表每项含 physical page 与 dirty bit；同一物理页只能有一个虚拟 owner。页大小为不超过 128 的二次幂，P0 虚拟/物理网格各轴最多 16；这些是原型范围限制。
- `clearAtlas=true` 初始化整个物理图集，并强制重画所有映射页；否则只清理、重画 dirty 映射页。未映射物理页不被工作列表修改。
- 调用方等待前次 GPU 使用完成后再更新输入。同一实例不支持并发更新。修改 geometry 后标记旧/新 footprint，修改 projection 后标记所有映射页，重用 physical page 后标记新映射；自动失效留在 P2。
- `record` 接受调用方已知的旧 atlas layout，完成录制后 atlas 为 ShaderRead。它不维护假定已经提交的 CPU layout 历史，也不内部等待 GPU。读回方显式建立 transfer 屏障。
- 接收结果区分 Valid、Unmapped、OutsideProjection 和 WorkOverflow；后三者返回 visibility=-1，不伪装成无阴影。工作列表有界并统计溢出，间接绘制数量被限制到分配容量，溢出时整个投影查询标为不完整。

## 源码入口

- [运行时 API](../Source/Runtime/Render/Shadow/VirtualShadowMapPrototype.h)
- [资源、管线和录制](../Source/Runtime/Render/Shadow/VirtualShadowMapPrototype.cpp)
- [Slang 工作生成、页裁剪、光栅和查询](../Shaders/Features/VirtualShadowMaps/VirtualShadowMapPrototype.slang)
- [GPU 验证与证据导出](../tests/rhi/VirtualShadowMapTests.cpp)

## 验证方法

复用 tests-enabled 的 `build-pass-stages-nrd`，保持 MSVC 14.51.36231 / Windows SDK 10.0.26100.0 配置：

```powershell
cmd.exe /d /s /c '"E:\VisualStudio\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=14.51.36231 -winsdk=10.0.26100.0 && cmake --build build-pass-stages-nrd --target MetallicRHITests -j 8'
python -B Tools/CheckShaderModules.py
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_prototype_pages --rhi-validation --output-dir .cache/vsm-p0
```

测试创建自己的非 RT 设备，明确关闭并断言 RayQuery、RT pipeline、RTAS 和 CLAS；打开 Vulkan validation 与 synchronization validation。以独立 CPU 三角形交点/重心深度为参考，不复制 GPU 的页裁剪和物理路由。比较 shadow texel center 处的深度与查询结果，单独记录精确几何边缘/alpha cutoff 上的排除像素。

覆盖打乱的物理页映射、跨页三角形、nearest depth、近远裁剪、单双面、标准/贴图/custom Coverage、clean 页保持、dirty 页清除、零工作、投影变化、128×128 页、空 caster 集、无效输入、缺页和工作列表溢出。PNG 来自实际 GPU 深度与 receiver query 读回；JSON 记录计数和验证状态。

## 当前验收

2026-10-25，在 NVIDIA GeForce RTX 5070 Ti / driver 621.01 上完成实际离屏 GPU 验收：

- `MetallicRHITests` Release 构建通过；`virtual_shadow_map_prototype_pages` 通过，Vulkan validation / synchronization validation 均开启，validation errors = 0。设备明确关闭并断言 RayQuery、RTAS、RT pipeline 和 CLAS。
- 128×128 页几何样例比较 114,688 个有效接收点，无深度/可见性差异；16,385 个缺页查询、4 个投影外查询分别返回明确状态。32 像素页、Coverage、单双面、空几何与投影更新样例合计再比较 36,014 个有效点；投影更新样例排除 19 个精确三角形边缘点。
- 零 dirty 时图集逐位不变；单页 dirty clear 清除 927 个原深度像素，其它页逐位不变。工作容量为 1 时记录 16 个请求、15 个 overflow，间接绘制严格限制为 1，投影内结果标记不完整。
- 三项相关回归通过：`hybrid_raster_depth_coverage_and_overflow`、`material_coverage_compiler`、`gpu_scene_coverage_classification`。
- Slang module ownership 检查通过：381 files / 269 modules / 112 implementation units。`git diff --check` 通过。
- 普通/custom Coverage 各 7 个实际 SPIR-V 产物，全部通过 `spirv-val --target-env vulkan1.4 --scalar-block-layout`（SPIRV-Tools v2026.4）；缓存 v3 校验和及读取依赖与当前源码匹配。全部不含 RayQuery/RayTracing capability，两份 fragment 均无 `EarlyFragmentTests`。

首次 GPU 验证发现并修复 CPU/Slang std430 布局不一致：`uint3` padding 会扩大 Triangle/Query stride。现在 shader 使用标量 padding，实际 SPIR-V 的 Triangle=112 B、Query=32 B，与 CPU 静态布局一致。

独立的 `Tools/CheckShaderRegistry.cmake` 检查仍失败在未修改的 `Source/Runtime/Render/Core/NativeShaderWarmup.cpp:560`；这是基线已存在的直接 `device.createShaderModule` 调用。本原型的 shader 和 PSO 均经 ShaderRegistry 创建。

本地可复查证据（不加入源代码版本控制）：

- [完整计数与状态 JSON](../.cache/vsm-p0/VirtualShadowMapPrototype.json)
- [128 页深度图集](../.cache/vsm-p0/VSMGeometry128-atlas.png) / [接收面可见性](../.cache/vsm-p0/VSMGeometry128-receivers.png)
- [Coverage 接收面](../.cache/vsm-p0/VSMCoverage-receivers.png) / [单双面接收面](../.cache/vsm-p0/VSMSidedness-receivers.png)
- [GPU 运行日志](../.cache/vsm-p0-run.log) / [相关回归日志](../.cache/vsm-p0-regressions.log)
- [实际 SPIR-V 验证与依赖身份](../.cache/vsm-p0/spirv/Validation.json)

接收面 PNG：白=可见，黑=被遮挡，红=缺页，橙=投影外，蓝=工作溢出。深度图集为 reversed-Z 灰度。

结论是 **P0 支持继续 D32 atlas + 页内几何裁剪路线**。原型没有分页分配器、clipmap、自动缓存失效、SMRT、生产场景流送或性能加速结论；生产实时图与默认阴影切换继续按后续阶段推进。
