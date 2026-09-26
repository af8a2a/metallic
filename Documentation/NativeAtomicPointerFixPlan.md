# Native 原子指针修复与重构方案

初版日期：2026-09-25；实施更新：2026-09-26。原分析与 PoC 见下文，根因见 [调查记录](NativeAtomicPointerInvestigation.md)。

## 实施状态

生产 normalizer 已采用布局驱动策略：严格匹配单成员 Block、成员 offset 0、uint64 runtime array、stride 8，保留完整 typed 链；其他 buffer 继续显式布局转换。初版只保留 unsigned uint64，signed/复杂结构未加入白名单。RHI、生产 shader、descriptor index 和 push ABI 均未改动；shader cache request version 已更新为 23。

增加了指针 Copy/Select/Phi 策略传播、冲突/逃逸/部分 typed 链诊断，以及混合 32/64 位 untyped 原子的编译前拒绝；失败时不发布部分输出。静态测试和 mapped/native 的 256 线程 Add/Min/Max/CAS、原子返回值、guard、GetDimensions、CPU-authored 嵌套结构 typed/raw 读取已通过。六种 raster 也已执行真实 mixed-producer GPU 回归。

原管线创建崩溃已解除，但两个 native 细分测试暴露了独立图像差异，因此完整 native 迁移验收仍未全绿。旧/新 normalizer 对受影响 resident mesh 产生逐字节相同的结果，见 [独立调查](NativeResidentImageInvestigation.md)。未采用任何 mapped/image/BDA 生产特例。最终验证记录见 [升级状态](DynamicResourceUpgradeStatus.md)。

## 建议

采用**小范围 normalizer 重构，以 buffer 类型/布局选择指针转换策略**：对经过严格识别的平坦 64 位整数 buffer 保留完整 typed 指针链，其他 buffer 继续使用显式布局的 untyped 转换。

这仍然是 native dynamic resource：资源继续通过最终 `shaderIndex`、`ResourceHeapEXT` 和 `OpBufferPointerEXT` 访问，不恢复 binding，也不改变 CPU push ABI。`OpBufferPointerEXT` 的结果允许使用 Uniform/StorageBuffer 的 pointer type；typed/untyped 是访问 buffer 数据的指针表示问题，与 heap 动态索引正交。[Khronos descriptor heap 规范](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_descriptor_heap.html)

## 本轮新增证据

两个隔离候选均保留原子操作本身、memory scope/semantics 和所有 descriptor index：

| 候选 | 最小混合原子 shader | 完整 streamClusterRasterMain | SPIR-V 校验 |
| --- | --- | --- | --- |
| 仅恢复 typed OpBufferPointerEXT 根指针，派生原子指针仍 untyped | 仍崩溃 | 仍崩溃 | 通过 |
| 按真实布局识别平坦 int64 buffer，保留其完整 typed 访问链 | 管线创建通过 | 管线创建通过 | 通过 |

第二种候选没有使用前轮探针中的类型 ID 3724 或 shader 名称。它识别 `StorageBuffer -> Block struct -> 单一 runtime array -> 64-bit integer`，要求首成员 Offset 为 0、ArrayStride 为 8。

六个生产入口均从当前 shader 源码重新编译，并在独立 Vulkan 程序中通过管线创建与 `spirv-val --target-env vulkan1.3`：

- `streamClusterRasterMain`
- `streamClusterRasterLegacyMain`
- `streamClusterRasterPlaneMain`
- `streamClusterRasterCooperativeMain`
- `streamClusterRasterWorkBinsMain`
- `streamClusterRasterWorkControlMain`

证据：`.cache/native-atomic-plan/layout-results.json`；候选 normalizer：`LayoutAwareNormalize.h`。独立程序没有 GPU 提交，因此这里不宣称三项渲染测试已修复，也不证明嵌套布局、SHaRC 或 DLSS 的 GPU 正确性。

## 方案比较

| 方案 | 范围与收益 | 限制 | 建议 |
| --- | --- | --- | --- |
| 按 shader 名称、入口或临时类型 ID 跳过转换 | 改动最少 | 新入口/编译器输出变化易再次触发，无法可靠覆盖共享 helper | 不采用 |
| 平坦 64 位整数 buffer 保留 typed，其余保留布局修复 | 只改 compiler compatibility 层；保持 native heap 和最终 index ABI；六入口编译验证通过 | 严格限定已证明的布局；复杂结构内原子另行处理 | 立即实施的核心策略 |
| normalizer 分成类型分析、策略分类、指针传播和重写 | 把上述规则集中在编译层，可检测未来不受支持的组合 | 有限增加代码与测试，需处理 Phi/Select 等合流 | 与核心修复一起做小范围重构 |
| 64 位原子单独改 BDA/地址参数 | 可以隔离逻辑指针表示，适合有明确地址所有权的专用数据 | 需修改 shader/CPU 参数、资源生命周期和所有调用点；后续已通过隔离 GPU 回读和六入口编译，完整渲染/性能尚未验证 | 私有原子资源的重构候选，见 [BDA 调研](AtomicBufferBdaAssessment.md) |
| 对受影响模块显式编译为 mapped | 已有三项完整渲染通过的基线，可作为应急模式 | native 迁移仍有例外，扩大映射分支 | 应急选择，不作为最终方案 |
| 通用字节偏移 lowering，再为每次原子生成 typed 标量 alias | 可处理复杂结构中的原子，保留同一 descriptor index | 需正确计算 Offset/ArrayStride/MatrixStride、动态索引、对齐和指针合流；接近一个编译后端 | 当前不做，出现实际需求再扩展 |

不建议全局关闭 normalizer：已知 typed 嵌套结构读取错误会恢复。也不能把 packed depth/visibility 的一次 64 位原子操作拆成两次 32 位操作，会失去整体原子更新语义。驱动已经发生 CPU 访问异常后，不应在同一进程中依赖捕获异常再重新编译 mapped；应急模式必须在创建管线前决定。

此前 `OpBitcast` 诊断变体在当前能力/寻址契约下被验证器拒绝，不能用“驱动恰好接受”代替合法性。规范中 untyped access chain 可以接受 typed Base，但其 Result Type 必须是 untyped；这也解释了为何仅保留根指针不是完整的 typed 原子访问策略。[Khronos untyped pointers 规范](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_untyped_pointers.html)

## 推荐实现边界

保留 `normalizeNativeDescriptorHeapSpirv` 对外接口和 compiler 插入点。将内部处理明确分为四步，不引入全局 ResourceRegistry 或 RHI 资源所有权改造。

### 1. 类型与布局分析

建立 type、decoration 和 descriptor root 信息。对白名单形状要求明确证据：

- storage class 为 StorageBuffer。
- root pointee 为带 Block decoration 的 struct，且恰好一个成员。
- 首成员 Offset 明确为 0，成员为 runtime array。
- 数组元素为 64 位整数标量，ArrayStride 明确为 8。
- 不把“结构中含 uint64”“元素总大小为 8”或 vector 等近似条件当成同一种布局。

初版只支持当前工具链的明确常量布局。缺失装饰、复杂结构或无法证明的布局不进入 typed 白名单。使用有边界检查的解析和命名 opcode/operand helper，避免把临时 probe 的简化扫描直接当作生产实现。

### 2. 为每个 descriptor root 指定策略

- `PreserveTypedScalar64`：保留原来的 OpBufferPointerEXT 及全部派生访问类型。
- `NormalizeExplicitLayout`：保持当前 untyped + 显式 Base Type 转换。

初版不必先全模块统计“是否恰好混合 32/64 原子”才决定是否保留平坦 64 位 buffer。以稳定布局选择策略，可避免某个 32 位统计原子的增删悄悄改变同一 buffer 的 lowering。

未经过验证，不扩展到所有标量/向量 typed buffer，也不恢复全部原始 typed 输出。signed int64 如纳入识别，必须增加对应 GPU 测试。

### 3. 指针数据流与兼容性检查

追踪 descriptor root、原始类型、layout type、storage class、策略和派生关系，而不只保存“待转成 untyped 的 ID 集合”。

- AccessChain/CopyObject/Phi/Select 传播一致策略。
- 合流涉及不同表示或不能证明兼容时给出明确编译诊断。
- 继续拒绝目前不支持的指针逃逸、函数传递和 memory copy，不默默保留部分错误转换。
- 完成传播后扫描原子消费者；若复杂结构仍产生本机已知危险的混合 32/64 untyped 原子组合，则在编译层明确拒绝并说明需要 typed 标量原子 buffer，避免再把该组合提交给驱动。
- 原子操作的判定须正确处理 Load/Store/CompareExchange 等不同 operand 布局，不可假设所有原子都有相同 result/value 位置。
- 保持 native AS 默认 heap load 拒绝逻辑与当前显式 AS resolver。

复杂结构内部的 64 位原子不能简单按整棵 root 跳过转换，否则会重新引入嵌套偏移错误。若后续有该需求，优先把并发计数/状态组织为独立、平坦、typed 原子 buffer；通用 alias/字节偏移 lowering 作为后续编译层扩展。

### 4. 确定性重写与缓存

保留原 descriptor index、布局装饰、NonUniform、原子 scope/semantics、数组 stride、push ABI 和 AS ABI。输出须满足：mapped 字节不变、native 转换幂等、失败时不发布部分输出。

同步递增 shader cache request version（本轮读取值为 22，实施时以当前值为准）。PSO 已使用 shader 内容身份，仍应验证预热与运行阶段采用相同 native/mapped 模式和最终字节码。若未来引入 driver-specific policy，policy 必须显式进入 cache key，不能让编译器隐式读取当前 GPU 后复用不区分设备策略的缓存。

## 验收条件

1. 静态转换：合法/非法布局、mapped byte identity、幂等、失败不改输出、指针合流诊断，以及 unsafe AS lowering 拒绝。
2. 新增真正 GPU 回读：同一 dispatch 混合 32/64 位 Add/Min/Max/CAS，非零索引、动态索引、多线程竞争；验证高 32 位不丢失、原子返回值/计数和 packed depth/visibility 结果。不能只检查管线创建。
3. 原子与嵌套结构同时出现的 shader：CPU 填充/校验数据并用 raw 读取作独立参照，确认保留 scalar64 不会破坏其余结构的 Offset/ArrayStride、数组/矩阵读取和 GetDimensions。
4. 原先三项 native 失败用例全部通过，六个 raster 入口均实际覆盖；对照 mapped 输出和既有容差。默认 mapped 的通过结果须保持。
5. 现有 nested-layout、final-index heap-switch、HybridRaster、流式 LOD、SHaRC、光追，以及实际启用 Streamline 的 DLSS 场景回归；两种模式的 editor smoke。
6. 冷编译与缓存命中路径一致，无 Vulkan validation error、DeviceLost 或原生 CPU 异常。启动 smoke 不替代完整渲染/GPU 回读。

本问题的修复完成标准是恢复上述 GPU 正确性；CPU 最终索引、共享资源所有权和 AS 统一方案可以保持各自独立的迁移节奏。
