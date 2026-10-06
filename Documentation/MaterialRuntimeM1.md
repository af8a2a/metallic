# M1 统一材质运行时

状态：M1 已完成（2026-10-01）。M0 继续使用现有 LookDev；本阶段交付现有 OpenPBR Surface 与 RTXCR Chiang DOTS Fiber 的运行时、编译发布和生命周期契约。

## 运行时与参数

[MaterialRuntime](../Source/Runtime/Render/Material/MaterialRuntime.h) 集中注册 Definition / Program / Instance / Schema / Capabilities。1000 个只改变颜色或贴图句柄的实例共享同一 Program；加入 Fiber 后是两个模型 Program。实例参数不参与语义 Program key，实际 executable key 由 SPIR-V、参数 ABI、资源 manifest、常量大小及 ray-query 要求构成。

- `MaterialGeneration` 是不可变实例/参数快照，保留 source material revision、唯一序号及源材质索引。
- Schema 用稳定参数 ID、类型、偏移和大小描述布局。校验拒绝重复 ID、重叠、越界、无效大小/对齐与未支持类型。
- `migrateMaterialParameters` 接受源布局、目标布局和目标默认值，按 ID 与类型迁移；重排不丢失值，新增或类型变化保留默认值并报告诊断。非法布局不覆盖上一份输出。
- `MaterialGeneration::create(schema, bytes, ...)` 将版本化参数布局降为内置执行 ABI。它与固定布局入口使用同一个生成和发布流程。
- [LegacyMaterialPayload](../Source/Runtime/Render/Material/LegacyMaterialPayload.h) 保留既有 720 字节 GPU ABI，作为当前内置模型的兼容布局；不会通过添加所有未来模型字段来扩展它。迁移 API 支持已注册类型的布局变化，当前类型是 Float、UInt、Float4 和 Texture。
- GPU `textureParams.z` 编码 Program ID：1 为 OpenPBR，2 为 RTXCR，0 按旧 hair 标志推导。CPU 拒绝冲突或非有限身份。
- 生产 pass 在编译和材质 revision 变化时校验目标能力。Fiber 只能进入支持 RTXCR 的 ray-hit 入口；不支持的 VBuffer / OpenPBR-only 目标返回明确诊断，避免静默套用 Surface 模型。

这里的布局迁移是数据契约。M1 内置消费者仍使用明确的兼容 ABI；任意自定义 Slang 模型、Value Program 与资源依赖分析属于 M2。

## 编译、发布与退休

[MaterialExecutable](../Source/Runtime/Render/Material/MaterialExecutable.h) 已接入 `ScenePathTracePass` 的普通、OpenPBR、VBuffer 分箱及缓存 permutation 编译。编译产物包含 SPIR-V、依赖、内容 key、参数 ABI 和资源 manifest。manifest 直接用于 `ComputeProgram` 的检查与参数编码；不是只记录在文档中的描述。当前 manifest 由受控内置程序声明，尚不代表能自动证明任意外部 Slang 的资源访问安全。

编译和 pipeline 创建先在候选对象完成，全部成功才替换 executable 与 artifact。编译或 manifest/pipeline 创建失败保留旧对象及诊断；没有成功版本时，实际 PT / Deferred pass 使用独立错误材质显示品红棋盘格。错误路径同时初始化已声明的辅助输出。

`RenderGraphCompileContext::shaderReload` 区分初次创建与整图替换。热重载候选失败必须使事务失败，不能用错误材质覆盖已有成功图。成功的图替换继续复用现有帧边界和历史重置契约。Slang 首次 `loadModule` 失败时，入口与诊断中的源码位置也进入追踪；修复报错 include 即可触发重试。

`MaterialBindingGeneration` 同时拥有 CPU 快照与 GPU 参数 buffer。resident、materials-only 上传和编辑均发布这个对象；材质编辑先创建/映射候选 buffer，成功后才替换发布对象并推进 material revision。可选分配策略用于预算拒绝及故障注入，默认仍调用 `Device::createBuffer`。

帧准备通过已有 `RenderFrameContext` 保留材质发布对象。ComputeProgram/PreparedComputeDispatch 继续保留 executable、编码参数和 descriptor 所用分配，纹理沿用现有不可变纹理快照。旧提交完成、命令记录和帧所有权释放后才回收；没有另建一套 fence 或资源回收器。材质 revision 变化使现有 PT / Deferred 累积历史和相关缓存失效。

这些入口要求沿用现有帧协调器，不能在并发录制中途修改发布状态。现有 `Buffer::flush` 为 void，驱动级 flush/device-loss 错误仍遵循 RHI 的设备错误处理；本批故障注入验证的是可观察的分配、映射、编译和 pipeline/manifest 拒绝。

## 散射与几何约定

[OpenPBR Slang closure](../Shaders/Modules/OpenPBRClosure.slang) 提供 Prepare、Projected Eval、Sample、Pdf。Vendor Eval 已含投影余弦，消费者不再乘一次 cosine；保留 diffuse/specular 分解、Sample 权重、eta 和积分器运算顺序。

[RTXCR adapter](../Shaders/Interop/RTXCRMaterialAdapter.hlsli) 提供 Prepare / Eval / Sample。FiberInteraction 只包含现有 DOTS 路径能提供的 authored normal、tangent、outgoing direction，不虚构 strand ID、半径或横截面坐标。法线/TBN 构建顺序保持原样，Fiber 不套用 Surface 的 N·L。现有环境 NEE 近似 PDF 明确列入模型近似，不宣称精确独立 Chiang PDF 或外部 Layer 能力。

第三方源码保持原样。view-dependent prepared BSDF 是求值临时状态，不存入实例代际。旧 standard Surface 着色入口继续作为兼容路径保留；其 executable 内容身份与 OpenPBR 入口分开，不能把两种估计器的结果当作同一路径比较。

## 验收证据

复用 `build-scheduling-release`（MSVC / Ninja Release），构建 `MetallicRHITests` 和 `LookDev`，不改变原有 SDK 配置。生成证据位于忽略目录 `build/material-runtime-m1/`，日志位于相邻 `build/material-runtime-m1-*.log`。

最终构建成功；完整回归 **23 项通过、0 失败、0 跳过**，命令启用 `--rhi-validation`，日志无 Vulkan VUID 报错。记录为 `build/material-runtime-m1-completion-build-final.log` 与 `build/material-runtime-m1-completion-final.log`。其中旧 `render_graph_rtxcr_material_preview` 自建 renderer 时关闭 validation；本次新增的 Claire 256 帧 HDR 捕获使用启用 validation 的设备，二者区别保留。

| 契约 | 验证 |
|---|---|
| 实例共享与身份 | 1000 个实例、两个模型、稳定 Program、非法 ID/NaN 拒绝、目标能力拒绝 |
| 布局迁移 | 字段重排、新增/类型变化默认值与诊断、畸形布局不覆盖旧值、实际内置 ABI 降低 |
| CPU/Slang ABI | GPU 逐个命名字段读回全部 720 字节，验证三条记录的偏移、数组 stride 和身份 |
| 场景发布 | resident / materials-only 编辑、无变化同步、撤销、分配失败、映射失败及恢复 |
| 在途生命周期 | timeline gate 暂停真实旧提交；期间编译失败/恢复并发布新参数；旧提交读回旧代码和全部旧参数，完成后旧对象释放；新提交读回新代码和新参数 |
| 初次错误材质 | 浮点棋盘格逐像素断言；另对真实 LookDev PT / Deferred 注入初次 OpenPBR 编译错误，验证完整输出及不受影响的 Fiber |
| 初次语法错误恢复 | 清空追踪状态，首次 include 语法错误后只修复 include，检测修改并成功重新编译 |
| 集成回归 | LookDev、材质编辑、场景交接、StreamAsset 着色/阴影/透射、帧 descriptor、图热重载与分箱 |

### 原始 HDR 对照

新增可选 `setRawReadbackEnabled` / `readbackBytes` / `readbackFormat`，保留预览转换前的原始数据，并支持 RGBA32Sfloat。默认不额外复制原始图像。

`material_runtime_hdr_capture` 使用固定 LookDev 的 Reference / Deferred 各 768×768、256 帧，以及 Claire DOTS 768×432、256 帧。PT 为 4 spp/frame，Deferred 为现有 64 次环境采样；每个输出从独立历史开始。

迁移前对照采用 HEAD 的四份 shader（两个 PT 入口、直接光照、Material 模块入口），CPU 端使用同一最终渲染器及兼容 GPU ABI。当前 shader 先备份，捕获结束逐字节恢复；这不是完整旧 CPU 二进制的 A/B。上一批完整旧二进制的 PNG 对照作为补充证据保留。

三个 RGBA32F 输出均无 NaN/Inf，迁移前后逐字节相同，最大绝对误差与 RMSE 均为 0。M1 此次适配迁移采用精确一致性验收；统计噪声预算和不同估计器之间的误差不由这个结论代替。

- `hdr-baseline/`、`hdr-current/`：原始浮点文件。
- `hdr-baseline-sources/manifest.json`：HEAD、修改前后源码 hash、备份和恢复确认。
- `hdr-comparison-final.json`：最终完整验收运行三个输出的误差、非有限值检查与 SHA-256；对应当前文件位于 `completion-final/`。
- `production-error/`、`production-error-check.json`：真实初次失败的 PT/Deferred 棋盘格与 Fiber 一致性。

Claire 验证现已覆盖 256 帧静态积累并启用 Vulkan validation；没有据此宣称已验证动画 groom、运动或 strand LOD，这些属于后续几何工作包。

### 材质分箱问题关闭

原失败用例把 `Probe.data` 直接标为图输出，RenderGraph 因而分配 HostReadback 内存；大尺寸测试在该内存执行数百万次原子写入。测试改为 Device 内存计算，并用独立 transfer pass 在计算结束后复制到 HostReadback。原有 5 秒限制、4097×1025 规模、三帧重复和全部覆盖断言不变。普通与 typed indirect 两种用例均已通过；这是测试读回方式修复，不作为生产分箱算法加速结论。

## M1 之后

M2 继续推进真正的自定义 Value Program、Coverage / TextureFootprint、稀疏 Program 调度及 RT 程序选择。M3 补编辑与持久化，M4 补可组合散射的独立数学验证。M0 的性能采样和完整基线维护仍独立记录；本次 HDR 与兼容性验收不代替性能结果。
