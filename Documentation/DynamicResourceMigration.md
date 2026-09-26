# Metallic dynamic resource 迁移调研

> 实施进度、DeviceLost 修复及 CPU 最终索引迁移见 [升级状态](DynamicResourceUpgradeStatus.md)。以下现状、版本和源码行号为迁移前调研快照；全量迁移尚未完成。
日期：2026-09-25。本文是迁移设计，不表示迁移已经实现或通过 GPU 验证。

## 结论

建议迁移，但应同时解决三个独立问题：

1. shader 参数从“数字 slot → 通用资源表”改为按用途组织的参数结构，直接携带 typed handle。
2. Slang 从生成公共 descriptor arrays 改为直接生成 `SPV_EXT_descriptor_heap`。
3. descriptor 从 program 私有、dispatch 重写，改为 renderer/device 统一管理、资源 view 拥有稳定 handle。

只替换 `getResource<T>(slot)` 的名字，或者只把它改写成 `ResourceDescriptorHeap[index]`，均不能完成上述迁移。`DescriptorHandle<T>` 本身可以作为最终接口保留；它在 native heap 编译模式下同样可以直接访问 heap。

推荐最终接口：**typed handle + 每个 pass 的参数结构 + 共享资源注册表 + native heap SPIR-V**。普通 buffer 仍可使用 descriptor；本次不要求把所有 buffer 同时改为 BDA 指针。

## 已核实的现状

| 层次 | 当前行为 | 迁移含义 |
| --- | --- | --- |
| `Shaders/Modules/Core/ComputeResources.slang` | `uint2* resources`、`uint* constants`，`getResource<T>(slot)` 查表 | binding 已成为应用 slot，但仍保留旧参数组织方式 |
| `ComputeProgram.h` | 初始化声明 binding/kind/count；dispatch 再传 binding/resource；tableIndex 选择快照 | program 仍声明资源布局并负责资源提交 |
| `ComputeProgram.cpp` | 限制 256 slots；每个 program 有 heap；按 slot 写表和 constants packet | 资源句柄目前只对所属 heap 有效，不是全 renderer 通用索引 |
| `ComputeProgram::Impl::acquireTables` | 按 GPU completion 管理 heap/table 快照，同一提交重复使用 table 时另取快照 | 这是并发正确性机制，不能在删除 slot 层时一起删掉 |
| Vulkan compute pipeline | `bindingMappingCount == 0` 会自动生成 3 个公共映射，覆盖 sampler、image、buffer | “未提供 mappings”目前不等于“没有 mapping”，需增加明确 native 模式 |
| Vulkan heap | image/buffer 分区，各自按 descriptor size 换算 `shaderIndex`；RTAS 使用 buffer 槽 | 当前并不是所有资源共用一种 stride 的索引空间 |
| GPU-driven | 自己的 push ABI、局部 index + heap base、直接 `DescriptorHandle`；调用者传入 heap | 比 ComputeProgram 更接近目标，但也需要统一 handle/heap ABI |
| NRD adapter | 独立 heap、资源索引数组、自己的 push header | 不经过 ComputeResources，但仍需纳入底层 native 编译与共享 heap 迁移 |
| EditorDisplay | 存在明确的 `vk::binding`；由 `EditorDisplayRenderer` 编译并接入 ImGui | 这是实际使用的互操作路径，不能误认为所有自有 shader 都已无显式 binding |

当前扫描：38 个 `.slang` 文件包含 `getResource<` / `getResourceArray<`（包括 helper 本身）；22 个 C++ 调用文件使用 `ComputeProgramBindingDesc`（排除实现文件）。这不是全部受影响文件计数，GPU-driven、NRD、测试和 RHI 另计。

主要源码证据：

- `ComputeResources.slang:7`：公共 push 和 slot accessor。
- `ComputeProgram.h:23`、`:72`：program/dispatch 双重 binding 描述。
- `ComputeProgram.cpp:94`、`:160`、`:828`：私有 heap、快照、slot packet 写入。
- `VulkanRhi.cpp:2645`、`:2667`、`:9889`：heap 分区、索引换算、公共数组映射。
- `GPUDrivenSceneCommon.slang:9`：另一个携带 base/index 的 shader ABI。
- `NrdRuntime.cpp:294`、`Shaders/Interop/Denoising/NRD/Bindless.hlsli`：NRD 独立 heap/参数路径。
- `Shaders/Features/PostProcess/EditorDisplay.slang:19`、`EditorDisplayRenderer.cpp:79`：ImGui 显示互操作。

## 编译器与后端前提

本地 `External/slang/include/slang-tag-version.h` 和实际 `slangc -version` 均为 **2026.1.2**。执行：

```text
External/slang/bin/slangc.exe -capability spvDescriptorHeapEXT -help
error 14: unknown profile 'spvDescriptorHeapEXT'
```

因此 native SPIR-V 路径需要升级编译器。调研时官方 latest 指向 [v2026.18.2](https://github.com/shader-slang/slang/releases/tag/v2026.18.2)，可作为隔离验证候选；尚未下载或验证该二进制，不能将滚动文档中的全部行为直接视为该版本的实测结果。

官方描述了两条 lowering：默认模式生成公共数组；开启 `spvDescriptorHeapEXT` 后直接访问 heap。`ResourceDescriptorHeap[i]` / `SamplerDescriptorHeap[i]` 是另一种源码写法，仍走 descriptor-handle lowering；typed handle 并不比它“更不 dynamic”。见 [Slang bindless 文档](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/03-convenience-features.html)。

Vulkan 允许直接 heap built-in 访问，也允许将旧 set/binding 映射到 heap。两条路径必须在 RHI 中明确区分；native pipeline 仍需要 descriptor-heap create flag、heap 绑定和 push data。见 [Vulkan descriptor heaps](https://docs.vulkan.org/spec/latest/chapters/descriptorheaps.html)。

`SPV_EXT_descriptor_heap` 依赖 `SPV_KHR_untyped_pointers`。候选编译器、SPIR-V 工具和设备 feature 启用应按实际输出核对，不能只检查现有 `bindlessDescriptorHeap` 布尔值便宣告完成。见 [SPIR-V 扩展规范](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_descriptor_heap.html)。

建议引入编译/管线策略，例如 `MappedDescriptorArrays` 和 `NativeDescriptorHeap`，并与“slot 参数 ABI / typed 参数 ABI”分开。不要继续用 `usesResourceTable` 一个字段同时表达这些概念。EditorDisplay 和第三方互操作编译请求必须显式选择自己的模式。

## 最终参数模型

示意代码，须由候选编译器和 CPU layout 检查确认；它表达目标 ABI，不是可以直接替换现有 24 字节 push 的补丁：

```slang
struct BlitParams
{
    DescriptorHandle<Texture2D<float4>> source;
    DescriptorHandle<RWTexture2D<float4>> output;
    float exposure;
    uint width;
    uint height;
};

struct BlitRoot
{
    BlitParams* params;
};
[[vk::push_constant]] BlitRoot gBlit;

[shader("compute")]
[numthreads(8, 8, 1)]
void blitMain(uint3 tid : SV_DispatchThreadID)
{
    BlitParams p = *gBlit.params;
    if (tid.x >= p.width || tid.y >= p.height) { return; }
    Texture2D<float4> source = p.source;
    RWTexture2D<float4> output = p.output;
    output[tid.xy] = source.Load(int3(tid.xy, 0)) * p.exposure;
}
```

CPU 在资源 view 注册时获得 handle；dispatch 仅填充 `BlitParams` 并提交参数地址。program 负责 shader/pipeline 和参数 ABI，不再通过 binding 描述创建 heap 或替资源写 descriptors。

跨 C++/Slang 的 handle 采用明确 wire format，不能直接 memcpy 现有含 kind/index/shaderIndex 的 `BindlessHandle`。使用生成的布局或 reflection 校验字段 offset、alignment、大小，再用 `static_assert` 固定 CPU 侧契约。`DescriptorHandle<T>` 当前文档中的 SPIR-V 表示为 `uint2`，也仍应以锁定工具链的输出验证。

公共数学/颜色等 `Core` 模块不应附带 dispatch 专属全局 push。移除 `ComputeResources` 的副作用式导入，保留独立、唯一 owner 的 dispatch root；shader 库函数接收 typed 参数或上下文。可以保留通用“读取参数地址”的小 helper，但不能重新引入数字 slot。

材质纹理改用 material record 中的稳定 typed handles，或显式 handle 数组。连续 heap range 仅用于确实需要连续分配的批处理；不再让整个 texture collection 的正确性依赖 `base + index`。renderer 的资源依赖声明仍保留在 RenderGraph 中。

## Heap 与 handle ABI 决策

需要区分“直接访问 heap”和“统一 stride”：前者不强制后者。

| 方案 | 优点 | 代价/约束 |
| --- | --- | --- |
| 保留当前 image/buffer 分区和类型 stride | 普通 image/buffer 可以较小改动验证 native lowering | 句柄数值依赖资源类型的索引单位；RTAS 必须另行处理 |
| 统一 resource stride，sampler 独立 | 容易定义单一 resource index；资源注册与调试更直接 | descriptor 内存可能增加；全部 writer/index/copy/range/free-list 需要一致迁移 |
| 直接 descriptor 地址或自定义 fetch | 布局灵活 | 需要自定义 compiler fetch/后端 ABI；不建议作为首轮主线 |

**推荐最终使用统一 resource stride；首个 native PoC 可保留现有分区，以隔离 compiler lowering 风险。** 两阶段 ABI 必须显式区分，不能把旧 shaderIndex 交给新 heap。

最终 stride 由设备 descriptor size、alignment 和 RTAS 表示要求共同确定。建议用明确的 `SPIRVResourceHeapStride` 将 RHI 实际 stride 传入编译，并纳入 shader cache key。这样 image、buffer、AS 均有同一索引单位。sampler 仍使用独立 stride/heap。

特别注意：`SPIRVUnifiedDescriptorHeapStride` 按 image/buffer 最大 size 生成 stride，但官方当前说明它**不影响 acceleration-structure entries**。显式 resource stride 与 unified 选项也互斥，不能两者一起打开。见 [Slang 编译选项](https://docs.shader-slang.org/en/stable/external/slang/docs/command-line-slangc-reference.html#spirv-resource-heap-stride)。

### RTAS 是阻断性 ABI 差异

当前 `ComputeProgram.cpp` 把完整 AS device address 写进 slot，再作为 `DescriptorHandle<RTAS>` 解释。native 模式下，Slang 文档说明同一 handle 会先按索引从 heap 读取 64 位地址，然后转为 RTAS。把旧 address 原样喂给新 lowering 会产生错误 heap 访问。

最终建议普通 AS 使用真正的 AS descriptor handle，并采用上述显式 stride。若首个 PoC 保留按类型分区，则必须按其实际地址元素 stride 单独换算 AS index，或采用明确区分的 device-address 表示和经过编译验证的转换路径。**不得假定现在的 buffer shaderIndex 能直接用于 native AS 默认的 8 字节 stride。** Partitioned AS 单独验证编译与运行，不用普通 TLAS 的通过结果代替。

## 资源所有权与同步

推荐 renderer/device 级 `ResourceRegistry`，统一为渲染上下文提供一对当前可用的 resource/sampler heaps。注册的是具体 view，不只是 Texture/Buffer 对象。

- 长期 view 使用稳定 descriptor allocation；transient view 和参数分配由 frame arena 管理。
- image view key 包含资源身份/版本、format、mip/layer、view type、访问用途以及 descriptor 中的 layout；buffer key 包含 offset/range/用途。不能仅以对象地址或 Texture* 缓存。
- 更换底层 allocation、view 或描述内容时创建新版本；GPU 尚在使用的 descriptor 和参数不可原地覆盖。
- descriptor、view、底层资源、参数 packet、pipeline 都保留到实际提交完成。覆盖多 queue 和跨帧引用，不只按 CPU frame 编号回收。
- CPU handle 带 owner/registry 和 generation 校验；shader index 与 CPU 管理 handle 分开。generation 校验不会自动让 GPU 陈旧 index 变安全，仍必须延迟回收。
- heap 扩容优先预留容量；必须换 heap 时保持 live index 并复制 descriptors，保留旧 heap 到旧提交完成。禁止用重排所有 live indices 的方式在线扩容。
- indirect batch 共享只读资源/scene context；每项参数地址独立。permutation 兼容性改为参数 ABI/heap ABI，去除“各 program 必须按相同顺序分配同样 indices”的约束。
- RenderGraph 的访问声明、barrier、aliasing、queue ownership 保持独立。Shader 能动态取得资源不等于运行时能自动推断其读写依赖。

这部分是资源生命周期重构，不能只将 `frame->retain(tables)` 删除。NRD 临时资源、场景卸载、streaming eviction、resize 和热重载均需接入新管理者。

## 分阶段落地

| 阶段 | 具体改动 | 完成条件 |
| --- | --- | --- |
| 0：锁定工具链和基线 | 隔离安装候选 Slang；记录 compiler/DLL/工具版本；保存现有 shader 编译和渲染基线 | 原有功能在候选编译器 mapped 模式下可编译/运行；native 类型矩阵可编译并通过 SPIR-V 检查 |
| 1：native 后端 PoC | `SlangCompiler` 增加模式/stride；RHI compute/graphics/shader-object 区分 mappings 与 native；明确 RTAS 转换 | image+sampler、混合 buffer、AS 的 GPU 回读通过；native 用例无 set/binding 资源装饰与 mapping |
| 2：稳定资源管理 | 实现共享 registry、统一 stride、延迟回收、frame 参数 arena | 两个 program 共享同一 view/handle；重复 dispatch、跨帧、销毁/resize 后在途提交均正确 |
| 3：完整垂直切片 | FinalBlit 使用 typed params，移除该 pass 的 binding 描述和私有 heap；再迁 AutoExposure、SH/PDF | 两个不同 pass 验证共享资源；画面/回读正确；记录 descriptor 写入及 CPU 提交成本 |
| 4：批量迁移 | 后处理/调试 → lighting/ReGIR/RTXDI → path tracing/guides/SHaRC/NRC → material binning/GPU-driven/streaming | 每组原有编译、GPU、间接批处理、完整场景验证通过后继续 |
| 5：收尾 | 迁 NRD adapter 和样例；处理 EditorDisplay 边界；删除 slot API、旧 header/base 依赖和生产 mapping 分支 | 自有生产渲染路径满足最终 ABI 断言；例外有明确 allowlist；更新旧迁移文档 |

第 1 阶段可暂时保留 slot 参数，只用于降低后端切换的诊断成本；第 3～5 阶段必须真正消除它。保留临时兼容路径以支持阶段回退，但不要将“双架构长期维护”视为迁移完成。

EditorDisplay 当前依赖 ImGui layout；若要求连该路径都没有 binding，需要一并改造它与 ImGui texture/render-state 的互操作。闭源 Streamline/NRC SDK 的内部 ABI 不能由本项目全量改写，应保留为显式外部边界。进入/退出外部路径必须恢复实际 Vulkan heap/push 状态及 RHI 状态缓存，不能仅因“之前绑定过同一 heap 指针”就跳过恢复。

NRD 的算法输入编号可保留在 SDK adapter 边界，但应转换成 registry handles；它不应使生产 ComputeProgram 再次依赖通用 slot layout。

## 验收与性能判断

当前 `tests/rhi/RenderGraphTests.cpp::hasNativeComputeResourceInterface` 名称中的 native 指已有过渡接口。升级测试时应按选定 ABI 检查输出，不能只禁止 scalar/fixed-size bindings 然后继续接受公共 runtime arrays。

必要验收：

1. 使用 heap 的 native shader 有 `DescriptorHeapEXT`、对应 heap built-in；无资源 `DescriptorSet/Binding` 装饰。runtime array 本身合法，不应一概禁止；检查它从哪里取得资源。被优化掉全部 heap 使用的 shader 不应被错误要求保留 builtin。
2. native pipeline/shader-object 不再附加公共数组 mappings；继续检查 heap flags、push data 大小及 CPU/Slang 字段布局。
3. 类型矩阵包含 sampled/storage image、mip/layer、sampler、constant/structured/raw buffer、atomics、nonuniform 材质索引、普通/partitioned AS。需要支持相关扩展的 `spirv-val`。
4. 现有 `frame_descriptor_snapshots` 改写成资源/参数寿命断言，并增加跨 program 共享、释放后再分配、heap 扩容、多 queue、场景卸载和 resize 的真实用例。
5. 保留 `material_binning_indirect_coverage`、RTXDI/RELAX、OpenPBR/Standard path tracing、guides、position fetch、SH/PDF、SHaRC/NRC、GPU-driven/streaming 的回归。DLSS RR/NR 和编辑器 HDR/ImGui 需要实际交互负载验证。
6. compiler version、编译模式、实际 stride、参数/heap ABI 必须参与 cache 身份。已有 shader hash 含 Slang 版本/capabilities，但新增选项仍须加入；同步检查 PSO cache、热重载和旧模块失效。

直接 heap built-ins 的非一致访问规则不同于旧 descriptor-array 接口。迁移期继续明确标注可能分歧的材质索引，PoC 核对实际 SPIR-V；不要机械删除全部 `nonuniform`，也不要给所有 uniform 参数无差别加分歧标记。见 [Vulkan shader heap interface](https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources-descset)。

性能收益是待测假设：可能减少 descriptor 重写、program 私有 heap、slot 查表和 CPU 查 binding，但不能承诺 GPU 加速。应分别测 descriptor writes/bytes、参数上传量、heap bind 次数、分配高水位、CPU dispatch 准备耗时和 GPU pass/frame 耗时。统一 stride 的内存开销和参数结构的寄存器影响也应记录。使用固定场景、相同相机/功能、冷热缓存分开比较。

`Documentation/DescriptorHeapGpuFault.md` 已记录 shaderObject/驱动缓存组合问题；它是回归设计依据，不是本次重新验证的结果。保留现有 shaderObject 要求，默认缓存配置必须通过验证；禁用驱动缓存只用作诊断。

## 本次验证边界

完成源码/调用链调查、官方文档比对、本地编译器版本与 capability 拒绝验证。未升级依赖、未修改生产代码、未运行新的 GPU shader 或宣称候选编译器兼容。本文只新增迁移设计；工作区原有 Editor/Profiler/Streamer 修改未处理。
