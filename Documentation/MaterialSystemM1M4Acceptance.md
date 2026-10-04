# M1–M4 实现审查与重新验收（2026-10-04）

结论：**M1–M3 在已声明的实现范围内通过本次重新验收；M4 的 Value IR 与 Slab 原型通过各自测试，但尚未形成场景材质闭环，不能按完整里程碑关闭。建议先补齐该闭环，再推进 MaterialGraph。**

本文采用用户提供的 `D:/Metallic_Material_System_Roadmap.md` 编号：M1=Phase 0–1，M2=Phase 2–5，M3=Phase 6–8，M4=Phase 9–11。仓库早期 [MaterialSystemRoadmap](MaterialSystemRoadmap.md) 中的 M 编号不同，不能混用。审查基于提交 `a475da53949b58ee0f718f6b4a7f1105785b7d67` 加本次修复；未实施 MaterialGraph，也未更改 Slab 模型预算或默认 fused 调度。

## 实现审查

| 阶段 | 已确认的实现与验收范围 | 结论及边界 |
| --- | --- | --- |
| M1 Material Asset Foundation | Definition/Instance 语义资产、稀疏继承、版本迁移、资源 URI、事务性保存/发布；CPU/Slang 720-byte legacy ABI；实例共享程序；GPU 在途 generation 退休与热重载失败恢复 | 通过。资产定义目前注册 OpenPBR；场景自定义 Value Program 保存在 scene sidecar，不随 `.material` 导出。不能据此宣称任意模型资产已实现 |
| M2 Unified Surface Material API | Material→Closure→PreparedClosure；Debug Lambert/Mirror 与 OpenPBR 的统一 eval/sample/PDF/weighted consumers；SurfaceLighting、多灯和路径续追；resident/stream 材质与 authored TBN | 通过所声明的 Surface 能力。OpenPBR Importance 模式明确拒绝；Fiber 继续独立 transport，不套用 Surface cosine |
| M3 GPU-Driven Material Programs | 实际 ProgramKey 分箱与间接调度、feature 的 Program/Visibility/Pipeline 签名分离、独立 Coverage slicing、MASK 主可见性/阴影/RT 同代参数、动态 Coverage 绕过 OMM | 通过。Coverage 受限于声明的 IR 操作和 Surface MASK；program bin 优化依赖 wave32。未将单机测试外推为任意设备或大型场景可靠性 |
| M4 Material IR + Slab | Value IR 验证/折叠/DCE/CSE/稳定哈希/纹理 footprint；Closure IR 类型/拓扑/预算校验；Single/Mix/Layer GPU 数学与渲染；Program→Family 逻辑分类和 fused/split 原型 | **部分通过，完整里程碑未通过。** 两套 IR 仍未通过作者资产与生产场景连接，详见下方 P1 |

重点检查了作者资产解析与失败保留、shader 资源/参数 ABI、feature 与实例值的边界、coverage 与 shading 的依赖切片、frame completion 资源保留、Closure 预算及 production 调度调用方。没有进行全仓库安全审计。

## 发现与处理

### P1：M4 缺少 Value IR → Closure IR → 场景执行的闭环（未实现）

[MaterialClosureIR](../Source/Runtime/Material/MaterialClosureIR.h) 接收已解析的 `MaterialSlabRecord` 数值，尚无 Value IR 输出引用或统一 material lowering 入口。在 `Source/` 中，`lowerMaterialClosure` 只有定义；执行示例来自 `tests/rhi/MaterialSlabTests.cpp` 等独立 probe。

[MaterialRuntime](../Source/Runtime/Render/Material/MaterialRuntime.h) 的场景程序注册仍为 OpenPBRComposite/RTXCRChiang；[ScenePathTracePass](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) 生产 program bins 分配的 family 为 OpenPBRCompositeClosure。Slab 尚不能作为材质资产被场景选择，Value IR 仍生成 OpenPBR factors。独立 8 Programs→1 DualSlabFamily 实验不能证明生产 VBuffer/PT 已支持这些场景材质。

Slab 原型的限制本身是明确、合理的：单面漫反射、共享 normal basis 0、最多两个直接 Slab 加一个 Mix/Layer、48/96-byte resolved payload；更深拓扑、独立法线、镜面/折射/多次层间散射拒绝或未实现。测试通过证明该原型，不证明通用分层材质。

进入 MaterialGraph 前建议关闭以下验收门槛：

1. 建立可持久化的无 UI 作者定义，将动态参数/纹理的 Value IR 输出连接到 Slab 输入，并产生同一不可变 Program/generation。
2. 让同一 Single/Mix/Layer 材质进入实际场景 VBuffer 和 PT（含 secondary hit），复用 eval/sample 及 coverage 契约。
3. 验证保存/重载、编译失败保留、实例共享、动态参数更新、生产 Program→Family 多对一分类；保留 OpenPBR/Fiber 冻结参考。

无需先增加节点编辑器，也无需默认启用 packed closure buffer。Phase 11 的性能结论不在本次重测范围内。

### P2：基线把占位环境帧计入固定采样条件（已修复）

首轮完整序列虽然 `material_asset_phase0_equivalence` 返回通过，原始 PT HDR 对冻结参考的 RGB RMSE 为 `0.00045564657433930206`、最大通道差为 `0.05359751`；Deferred/Fiber 逐位相同。独立运行 PT 又逐位相同。同进程连续两个 baseline 用例即可复现，不能据独立通过忽略此差异。

新增环境时序记录确认：第二次 PT 捕获第 0 帧为 Loading、mapAvailable=false、revision=1，第 1 帧才 Ready、revision=2；另外两种捕获从第 0 帧即 Ready。异步解码受编译缓存与 CPU 时机影响，改变了累积历史和采样帧序列。

修复在 [EnvironmentLightingSubsystem](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h) 增加显式 `initialDecodeTimeoutMilliseconds`（默认 0），只在基线中设为 10000。初次解码未完成时等待，失败超时返回诊断；编辑器默认异步路径保持原行为。[MaterialBaselineTests](../tests/rhi/MaterialBaselineTests.cpp) 记录 `environmentTransitions`，并要求每个采样帧的真实环境均 Ready。最终完整序列三种 HDR 对原冻结 run-0 均逐位一致，且从第 0 帧 Ready。

注意：C++ baseline 测试负责捕获、有限值和输入条件检查，原始图像等价性仍须单独比较。本文的逐位一致结论来自实际 RGBA32F 比较，未增加容差或替换冻结参考。

### P2：native shader 测试仍要求旧物理指针 ABI（已修复）

首轮 `render_graph_openpbr_pathtracing_shader_compile` 失败。当前 [ParameterRoot](../Shaders/Modules/ParameterRoot.slang) 已使用 `BufferSpan<uint>` DR 根，旧 [RenderGraphTests](../tests/rhi/RenderGraphTests.cpp) 断言却要求 `PhysicalStorageBufferAddresses`。现在改为禁止该能力，保留 native heap capability/builtin、无 Binding/DescriptorSet、单一 push block 检查。OpenPBR、RTXDI、Guides 的全部调用方复测通过；另用 native buffer 嵌套布局和 position-fetch 实际 GPU 测试验证运行结果。

### P2：Inspector 向 device loader 请求 instance 命令（已修复）

实际 Roughness 拖动、分组 undo/redo、材质切换、dirty state、保存重载和比较渲染均能完成，但严格 CTest 因验证警告失败。[EditorDisplayRenderer](../Source/Editor/EditorDisplayRenderer.cpp) 原先对每个 ImGui 请求先调用 `vkGetDeviceProcAddr`，包含 `vkDestroySurfaceKHR` / `vkGetPhysicalDevice*` 等 instance 命令。改为使用能加载两类命令的实例 resolver，仍使用独立 ImGui 函数表。

另外，本机注册了失效的 EOS overlay / `E:/Validation.json` 清单。本次只在测试进程中设置 SDK `VK_LAYER_PATH` 及空 `VK_IMPLICIT_LAYER_PATH`，没有修改系统注册表、关闭 validation 或放宽 CTest 的零警告门槛。隔离环境下严格 Inspector 验收通过。

## 本次实际测试

Windows/MSVC Release，复用 `build-scheduling-release`；GPU 为本机 RTX 5070 Ti。顺序执行 GPU 测试，未同时运行多个 GPU workload。

| 项目 | 本次结果 | 本地证据 |
| --- | --- | --- |
| MetallicRHITests / MetallicSceneTests / Metallic / LookDev | 构建成功 | `build/material-m1-m4-build.log`、`material-m1-m4-final-build.log` |
| Scene 全部 127 项 | 118 通过，9 跳过，0 失败 | `build/material-m1-m4-scene.xml` |
| 最终材质与环境/缓存回归 60 项，validation 开启 | 57 通过，3 跳过，0 失败；无 Vulkan validation 消息 | `build/material-m1-m4-final/Tests.xml`、`material-m1-m4-final.log` |
| native 嵌套材质布局与 authored position-fetch | 2/2 通过，validation 开启 | `build/material-m1-m4-native/Tests.xml` |
| OMM ray query / partitioned | 2/2 通过，validation 关闭 | `build/material-m1-m4-omm/Tests.xml` |
| 严格 LookDev Inspector CTest | 最终二进制通过，真实 UI 操作与比较帧 | `build/material-m1-m4-editor-final.log`、`build/material-m1-m4-report/Editor.log` |
| Python 证据/性能工具 CTest | 10/10 测试组通过（包含 MaterialBaseline 和 ClosureScheduling） | `build/material-m1-m4-perf-tools.log` |
| Slab furnace/eval/sample/PDF/互易/能量与渲染 | 23 材质 × 3 角度 × 4096 样本通过；检查三张输出图片 | `build/material-m1-m4-final/SlabEnergy.txt`、`SingleSlab.png`、`DualSlabMix.png`、`DualSlabLayer.png` |
| OpenPBR PT / Deferred / RTXCR Chiang 冻结 HDR | 完整序列最终三图与 Phase 0 run-0 逐位一致，RGB RMSE/max=0 | `build/material-m1-m4-final/MaterialBaseline.json` 及 RGBA32F |
| 三个独立进程 HDR A/A | 三种输出全部逐位一致；三个进程均从第 0 帧使用就绪环境 | `build/material-m1-m4-repeatability/Manifest.json`、`build/material-m1-m4-repeatability.log` |

最终完整序列的 3 张 HDR 加上三进程的 9 张，共 12 张都与原冻结 `build/material-phase0-20261004-c/run-0` 逐位一致；比较保留双侧 SHA256、有限值检查、RGB RMSE/max 和环境就绪帧，见 `build/material-m1-m4-report/Comparison.json`，重算脚本为同目录 `Compare.py`。已查看 PT、Deferred、Fiber 显示图与三张 Slab 图；显示变换不参与 HDR 数值验收。

RHI 三个 skip 分别为缺少 cooked Zorah probes，以及当前 validation 路径不提供 OMM 的两个用例；OMM 已单独关闭 validation 执行，但不宣称其 validation 路径已通过。Scene 的 9 个 skip 涉及 Zorah、USD/SuperSponza 内容或可选依赖，不构成这些大型场景的运行验证。最终日志中的 `FrameEnvironmentProbePass` 一次 execute Failure 是 `frame_environment_submission_recovery` 主动故障注入，该测试随后恢复并通过。

本次未验证长期编辑会话、多设备/非 wave32、完整大场景 VRAM、NTC 专项视觉或 MaterialGraph 工作流；没有用 baseline 计时或 validation 下 fused/split 数字声称性能提升。

## 复现

从仓库根目录、x64 MSVC developer shell 执行。复用原 build 配置，先构建再运行测试。

```powershell
cmake --build build-scheduling-release --target MetallicRHITests MetallicSceneTests Metallic LookDev -j 8
ctest --test-dir build-scheduling-release -R '^MetallicSceneTests$' --output-on-failure
$env:METALLIC_NSIGHT_GRAPHICS_CAPTURE='0'
$env:METALLIC_SHADER_CAPTURE_SYMBOLS='0'
$env:METALLIC_VK_PIPELINE_STATISTICS='0'
# 下列 SDK 路径仅是本机配置；空目录只隔离无关的 implicit overlay。
$env:VK_LAYER_PATH='C:\VulkanSDK\1.4.350.0\Bin'
New-Item -ItemType Directory -Force build/material-m1-m4-empty-implicit-layers | Out-Null
$env:VK_IMPLICIT_LAYER_PATH=(Resolve-Path build/material-m1-m4-empty-implicit-layers).Path
$filter='*material_*:*visibility_buffer_deferred_openpbr*:*stream_material*:*texture_primary_ray_cone_pixel_footprint*:*scene_ray_tracing_position_fetch_authored_tangents*:*visibility_buffer_material_edit_refresh*:*visibility_buffer_deferred_shadow_history*:*opacity*:*render_graph_scene_path_trace_alpha_mask_preview*:*render_graph_openpbr_pathtracing_shader_compile*:*render_graph_pathtracing_guides_shader_compile*:*render_graph_rtxdi_shader_compile*:*slang_shader_*:*environment_subsystem_async_snapshot*:*frame_environment_submission_recovery*'
.\build-scheduling-release\tests\MetallicRHITests.exe "--gtest_filter=$filter" --rhi-validation --output-dir build/material-acceptance-new
ctest --test-dir build-scheduling-release -R '^MetallicLookDevMaterialInspectorSmoke$' --output-on-failure
ctest --test-dir build/perf-tests -C Debug --output-on-failure
python Tools/Perf/MaterialBaseline.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/material-repeatability-new
```

输出目录为本地证据，不纳入源代码。重跑应使用新目录；三进程工具保存源码、二进制、资产和输出哈希。复现 shell 的环境变量仅作用于该进程及其子进程。
