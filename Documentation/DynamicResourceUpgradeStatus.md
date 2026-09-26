# Dynamic resource 升级状态（2026-09-25）

**此前定位的 native 材质错误和 DeviceLost 已修复。全量资源 API 迁移仍未完成，默认模式保持 mapped；扩大回归后发现的 native 复杂流式光栅管线编译崩溃见下文。** Native buffer/image/sampler 继续使用 descriptor heaps；AS 使用显式 device-address handle。

## 已落地的工具链与后端

- Slang 升级到 2026.18.2，CMake 精确校验版本。`scripts/InstallSlang.ps1` 固定官方归档 SHA256；原安装备份在 `.cache/dynamic-resource-upgrade/slang-2026.1.2-backup`。
- `SlangShaderDesc::descriptorHeapMode` 支持 Default / Mapped / Native。Default 可用 `METALLIC_SLANG_DESCRIPTOR_MODE=native` 进入 native；显式模式不受环境变量影响。EditorDisplay 固定 Mapped，保持 ImGui 互操作。
- Native 启用 `spvDescriptorHeapEXT` 和设备支持的 `VK_KHR_shader_untyped_pointers`，没有 legacy Binding / DescriptorSet 的 shader 不附加 descriptor mappings。不支持 untyped pointers 的设备创建 native module 返回 Unsupported。
- 补齐 mapped storage image 非一致索引 feature。模式和修复版本进入 shader cache 身份，避免复用升级前字节码。

## DeviceLost 修复

环境：NVIDIA GB203-A，驱动 616.92，Slang 2026.18.2。以下是本机对照证据；后续独立 AS 调查已将数值读取故障缩小到驱动的 native heap load 路径，详见下文。

### Native buffer 嵌套结构布局

GPU 回读发现 `PathTraceMaterial` 的嵌套纹理索引读成了首个 float 的位模式；原先 material binning 的材质分类也错误。改用 raw byte-address load 得到正确数据，普通 SPIR-V 验证却无法检测这种 GPU 结果错误。

`NativeDescriptorHeapSpirv.h` 将 typed `OpBufferPointerEXT` 及其访问链转换为 KHR untyped pointers，每个访问链显式保留原始 pointee 布局类型，同时处理 runtime array length。保留资源索引、数组 stride、成员 offset 和 NonUniform 等装饰。mapped 输出保持字节一致。

转换在写入 shader cache 前执行，检查指令边界、能力和指针流；不支持的指针逃逸会产生编译诊断，不向 GPU 提交部分转换结果。不能以普通 spirv-val 通过代替 GPU 回读。

### AS handle 契约

Slang 默认 native AS 路径从 `ResourceHeapEXT` 读取 uint64，再转换为 AS。最初探针显示 CPU descriptor 内地址正确，但 GPU 原始 heap load 返回零；改用同一个 AS 的 device address 后，命中、UV、法线、BLAS compaction、TLAS refit 全部通过。尝试直接 AS descriptor load 和动态 stride 也未恢复正确结果，临时实验已移除。

后续 [AS handle 调查](AsHandleInvestigation.md) 使用独立 Vulkan + 手写 SPIR-V 复现 heap 数值 load 错误；同一字节经 physical pointer 读取正确，错误值也并非恒为零。已验证通过 storage-buffer descriptor 读取 AS 地址单元的方案可以统一为 heap index（index 3、真实 BLAS/TLAS 命中），但该 PoC 尚未集成。Slang 的官方 export hook 另有 native 编译崩溃，不能直接代替显式 resolver。

正式路径中，两种模式的 AS handle 都携带完整 64-bit device address。`ComputeProgram` 负责写入；`Metallic::resolveDescriptor` 负责转换。buffer/image/sampler 继续采用 Slang 当前模式的 heap 读取方式，AS 不增加固定 binding。

新的 typed params 如包含 AS，应调用 `resolveDescriptor(handle)`；直接使用 Slang 默认 AS `DescriptorHandle` 解引用会触发编译期保护，诊断要求使用 resolver。此方案没有宣称原生 AS heap load 已修复，也不依赖 opaque descriptor 字节布局。Partitioned AS 仍须单独 GPU 验证。

`ScenePathTrace.slang` 和 position-fetch probe 的原始光追、alpha traversal、TBN 逻辑已恢复，不通过跳过材质或 alpha 测试规避挂起。

## 回归与证据

- 新增 `native_descriptor_heap_nested_layout`：显式运行 mapped/native，GPU 写出含嵌套数组的完整结构，再分别用 raw 与 typed 读取，逐字比较 CPU 预期，并验证 GetDimensions。检查 malformed SPIR-V、转换幂等性和危险 AS lowering 的编译期拒绝。
- 新增 `scene_ray_tracing_position_fetch_native`：普通测试运行中也显式启用 native，覆盖 authored tangent、fetch/fallback、back face、miss、BLAS compaction 和 TLAS refit。
- `.cache/native-devicelost/resolve.json`：原先失败的五项关键用例全部通过，日志无 Vulkan validation 报告。
- `.cache/native-devicelost/regression.json`：新增两项用例通过；导出的 mapped/native 字节码均通过独立 `spirv-val --target-env vulkan1.3`。
- `cmake --build build --target Metallic MetallicRhiTests -j 12`：MSVC Debug 构建通过。
- `.cache/native-devicelost/final-native.json`、`final-mapped.json`：两种运行模式各 20/20 通过，日志均无 Vulkan validation 报告。包含 buffer/atomics、曝光、frame snapshots、材质分桶、OpenPBR VBuffer、RTXDI、guides 编译、光追和新增回归。
- `.cache/native-devicelost/final-smoke.log`：native 环境下 `Metallic.exe --smoke-test` 退出码 0，实际提交并呈现编辑器帧；这是启动 smoke，不是 DLSS 或长期交互验证。
- 测试生成的 `meet_mat.glb.meshlets.bin` 仅改变版本与 padding，逐字节确认几何一致后备份并恢复。原有 `External/microprofile` 工作区内容未修改。

原始失败日志保留在 `.cache/dynamic-resource-upgrade/`，本次隔离实验在 `.cache/native-devicelost/`，均为本地输出，不提交测试图片和临时 probe。

## 复现

在仓库根目录构建后运行：

```powershell
# 不修改全局模式，也会显式覆盖 native 修复
& .\build\tests\MetallicRhiTests.exe --rhi-validation '--gtest_filter=*native_descriptor_heap_nested_layout*:*scene_ray_tracing_position_fetch_native*'

# 完整渲染 native 回归
$env:METALLIC_SLANG_DESCRIPTOR_MODE = 'native'
& .\build\tests\MetallicRhiTests.exe --rhi-validation '--gtest_filter=*material_binning_indirect_coverage*:*visibility_buffer_deferred_openpbr*:*render_graph_rtxdi_preview*:*scene_ray_tracing_position_fetch*'
Remove-Item Env:METALLIC_SLANG_DESCRIPTOR_MODE
```

## CPU 最终 descriptor index（2026-09-25）

- CPU 上传 `BindlessHandle.shaderIndex`；`index` 保留为分配器局部槽位，只用于写 descriptor 和释放。
- GPU-driven、streaming、材质 texture remap、细分纹理、样例与 DLSS 辅助 pass 均传入所属 heap 的最终索引。图形采样不再隐式依赖 image slot 0。
- 移除 shader 的 `imageShaderIndexBase` / `bufferShaderIndexBase`，删除 RHI 公共 8 字节 push header 和公开 base getter。基址换算只留在 Vulkan heap allocator 内部；push payload 从字节 0 原样上传。
- 同步调整 NRD 和 ComputeResources 的指针字段位置、Resident LOD 的 float4 对齐，并将 shader cache request version 升至 22。
- 新增 `final_descriptor_indices_heap_switch`，显式验证 mapped/native、两个不同分区布局的 heaps、非零槽位、buffer 内嵌 descriptor index，以及 heap 切换后的 push payload。

这一轮不改变资源所有权：最终 index 仍属于具体 view/descriptor 和所属 heap。AS 的 buffer-address-cell 方案仍是独立 PoC，当前正式 AS handle 继续携带 device address。

扩大验证时还修复了 `DlssNrPass::reflect` 的失效引用：新增字段导致 `fields_` 扩容后，再读取先前的 input 引用可能生成非法输出 format。现在保存独立的 colorFormat 值，避免 resize 后进入驱动的无效格式路径。

### 本轮验证

- `Metallic` 与 `MetallicRhiTests` 的 MSVC Debug 构建通过。
- 默认 mapped：基础回归 47 项通过、1 项因未启用 Streamline 跳过；DLSS bypass、contract、DebugControl 状态恢复及 frame reuse 补充回归 6/6 通过。
- Native：最终回归 50 项通过、1 项因未启用 Streamline 跳过；三个复杂流式光栅/细分用例受下述已复现限制阻断。
- 单独启用 `--rhi-streamline` 后，两种模式的 `dlss_nr_runtime` 和 `dlss_nr_runtime_scene` 均为 2/2 通过，包括路径追踪、DLSS-RR/NR、GPU 回读及 slider 验证。
- 上述用例按名称去重：mapped 55 项通过，native 52 项通过。跳过的 DLSS runtime 已由实际 Streamline 运行补验；三个受阻 native 用例没有计入通过数。
- 两种模式的 `Metallic.exe --smoke-test` 均退出 0，实际提交并呈现编辑器帧。
- 最终回归无 Vulkan validation error 或 DeviceLost；mapped 细分管线仍有未使用的 mesh output Location 6 性能警告。DebugControl 的预期失败注入会记录 pass failure，不是新增回归失败。验证层版本导致 KHR OMM 关闭，因此此结果不覆盖该扩展。
- 本地逐项记录在 `.cache/final-resource-indices/verification.json` 和相邻 JSON/log；测试生成的 meshlet cache 仅修改头部版本/padding，已备份并逐字节恢复本轮开始的文件。

### Native 流式光栅的既有驱动限制（2026-09-25 记录）

后续独立重跑已将原因缩小到 **normalizer 转换后的混合 32/64 位原子操作指针**，三个实际用例和 8 行最小 shader 在相同驱动地址崩溃；mapped 三项重跑均通过。详见 [混合位宽原子指针调查](NativeAtomicPointerInvestigation.md)。

复杂 `streamClusterRasterMain` 在 NVIDIA 616.92 上会于 `vkCreateComputePipelines` 内发生 CPU 访问异常，尚未提交 GPU 工作。独立 Vulkan probe 分别编译本轮迁移前备份与迁移后源码；二者经相同 native normalization 后都通过 `spirv-val --target-env vulkan1.3`，但创建管线都返回进程异常 `0xc0000005`。因此这项阻断在本轮 CPU 索引迁移前已经存在。默认 mapped 模式中的 mixed-producer 与递归/位移细分渲染均通过。

证据位于 `.cache/final-resource-indices/pipeline-probe-results.json`、`pipeline-probe-before.log`、`pipeline-probe-after.log`、`mixed-producer-stack.log` 和独立运行的 `final-native-displacement.log`。最终 native 回归显式排除这三个受阻渲染用例，并单独保留失败及迁移前后对照记录；不将其计为通过。

## Typed uint64 指针链实施（2026-09-26）

按统一 shader 资源用法的决策，正式实现了 [typed 指针链方案](NativeAtomicPointerFixPlan.md)：

- normalizer 严格识别 `StorageBuffer -> Block -> 单一 uint64 runtime array`，要求 member offset 0、array stride 8，保留整个 typed 链；其余 descriptor buffer 保留显式布局转换。只包含已验证的 unsigned uint64，不把复杂结构或 signed int64 自动加入白名单。
- Copy/Select/Phi 传播表示策略，合流不兼容、部分 typed 链、指针逃逸或危险 mixed 32/64 untyped 原子在编译层明确报错；输出失败时保持原值，转换可重复且 mapped 字节不变。
- 原子 scope/semantics、descriptor index、CPU push ABI 和 AS resolver 保持现有契约。未修改生产 shader/RHI，也没有添加 BDA、按 shader 名称或按 pass 选择访问方式的特例。
- shader cache request version 更新为 23；预热工具和运行时使用同一 normalizer。

### 验证结果

环境：RTX 5070 Ti / NVIDIA 616.92 / Slang 2026.18.2 / MSVC Debug。

| 验证 | 结果 |
| --- | --- |
| normalizer 静态策略、Phi/Select、布局拒绝、AtomicStore/Load/CAS、边界与事务性 | 8/8 通过 |
| mapped RHI（按名称去重，含实际 Streamline DLSS） | 62/62 通过 |
| native RHI（按名称去重，含实际 Streamline DLSS） | 60 通过，2 项细分图像比对未通过 |
| 六种 stream raster 入口 | SPIR-V 校验、管线创建与真实 mixed-producer GPU 执行全部通过 |
| 混合原子 GPU 回读 | 显式 mapped/native 均通过；256 线程 Add/Min/Max/CAS、高位、返回序列、唯一 CAS 成功线程、guard、维度与 CPU-authored 嵌套 typed/raw 读取 |
| Shader warmup | 六入口冷编译 0 失败；再次运行 6/6 cache hits |
| 编辑器实际 DLSS camera smoke | mapped/native 均通过，无 Vulkan validation 消息 |

上述静态与 RHI 合并去重后，mapped 为 70 项通过，native 为 68 项通过；没有把两个失败用例计为通过。最终 GPU 测试导出的四个 mapped/native SPIR-V 模块也通过独立 `spirv-val --target-env vulkan1.3`。

原来三个用例的驱动编译崩溃已解除，`render_graph_gpu_driven_mixed_producer_render` 以及另外五种 raster 变体均通过。`tessellation_recursive_render`、`tessellation_displacement_render` 的 native resident 路径现在能运行，但图像仍不匹配；stream 组合没有报告差异。旧/新 normalizer 对受影响 resident mesh 的字节码逐字节相同，隔离结果指向 native 图像 handle lowering 相关问题，尚未形成最小根因证明。详见 [独立图像调查](NativeResidentImageInvestigation.md)。

验证边界：NRC 路径退出时 mapped/native 均报告相同的 74 个 Vulkan 子对象未释放警告；mapped 细分还存在已记录的未使用 mesh output Location 6 性能警告。不能将本轮描述为 validation 完全无告警；这些问题未在本轮改动 RHI/SDK 以掩盖。没有再观察到原混合原子管线 CPU 异常或 DeviceLost。

证据：`.cache/typed-atomic-implementation/verification.json`、各模式 JSON/log、`pipeline-results.json`、`warmup-*.log`、`editor-results.json`。所有诊断 shader cache 替换已逐字节恢复；测试改写的 meshlet cache 和 NRC 日志已保留生成副本并恢复原始文件。生产代码修改仅在 normalizer 与 shader cache request version。

## 迁移边界

已修复此前定位的 native GPU 正确性和挂起问题，并完成 CPU 最终索引及无 heap header 的 push ABI；原混合原子编译阻断已修复，但 native resident 细分仍有上述独立图像差异。`ComputeResources.slang` 的 slot API、program 私有 heaps 仍存在；共享 ResourceRegistry、typed params、frame arena，以及 NRD / GPU-driven 接入共享资源所有权仍须按 [迁移设计](DynamicResourceMigration.md) 推进。上述 DLSS 验证仅覆盖列出的 fixture 和 sample；长时间交互、其他驱动和全部场景仍未验证。

## 上游参考

- [Slang 2026.18.2](https://github.com/shader-slang/slang/releases/tag/v2026.18.2)
- [Slang descriptor handle 定制](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/03-convenience-features.html)
- [SPV_EXT_descriptor_heap](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_descriptor_heap.html)
- [SPV_KHR_untyped_pointers](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_untyped_pointers.html)
