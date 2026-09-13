# MiniZorah M3：统一 VBuffer

日期：2026-09-12。沿用 M1 完整 cook、M2 的 1 GiB 页池和全部 3,163 根页。

**M3 已完成。** 全量场景、材质映射、硬件/异步混合光栅和生命周期验收通过。结构化摘要见 [MiniZorahVBufferResult.json](MiniZorahVBufferResult.json)。

## 入口与管线

```powershell
build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe --minizorah-vbuffer
```

编辑器入口为 **GPUDriven / MiniZorah VBuffer**，配置见 [gpu_driven_minizorah_vbuffer.metallic_graph.json](../Pipelines/Samples/gpu_driven_minizorah_vbuffer.metallic_graph.json)。M2 的 `--minizorah` 保留作独立 StreamAsset 对照。

管线为 `VisibilityBufferPass → VisibilityBufferMaterialPass → FinalBlitPass`。VBuffer 仍只产生可见性、深度及诊断图；材质着色由独立消费者完成。默认开启自动 LOD、异步混合光栅，目标误差 1.5 px。页池以外的元数据、frontier、候选和附件仍占用内存。

## 场景与几何归属

`streamAssetOnly=true` 配合 `sceneBinding=asset`，通过 `Scene::loadStreamMetadata()` 解析静态 glTF 的节点、变换、相机、材质、accessor 数量与 bounds。交给 tinygltf 的 JSON 投影移除了 buffers、bufferViews 和 accessor 存储引用；源文件保持原样，运行时不打开外部几何 `.bin`，不解码 meshopt，不运行普通 meshlet cache 或 LOD 构建。

`RenderPrimitive::storage=StreamAsset` 在 GPUScene 上传前确定 ownership。GPUScene 保留全局 geometry、instance、material 和 DrawSet identity，流式几何的 resident vertex/index/meshlet/LOD draw ranges 为空。VBuffer 允许空 resident layout，流式记录仍使用既有的统一 visibility ID 编码；旧 resident/stream 混合 producer 继续使用同一实现。

当前元数据加载范围明确限于外部 `.gltf`、静态三角形和标量材质。纹理、图像、蒙皮、动画、morph targets 不在此入口支持范围内，会明确报错。源 POSITION bounds 和 primitive 计数必须有效。此限制符合 MiniZorah 导出的实际内容。

## 材质与消费者契约

源 alphaMode、颜色、金属度、粗糙度、emissive、material ID 和 double-sided 保留。仅在元数据场景中，将无 baseColor 纹理且常量 alpha=1 的 BLEND 归入有效不透明桶；MASK 还要求阈值通过。真实半透明继续留在 BLEND 桶。流式光栅从 GPUScene 实例表读取双面标志，双面表面不做 normal cone 剔除，HW/SW 使用一致的背面规则。

`VisibilityBufferMaterialPass` 从 producer 的 `visibility`、`rasterInfo` 继承场景。消费者校验场景 identity、View generation、帧号和分辨率，再读取 producer 发布的借用 stream buffers。借用资源仅用于同次图执行，在 View 销毁时移除。它能解码 resident 与 stream 两种三角形，使用几何法线和源标量材质，提供 `shaded`、`baseColor`、`normal`、`instance` 显示。

这是一版固定方向光的材质预览，输出 LDR；不是完整 OpenPBR、阴影或反射。纹理与真实透射会被消费者明确拒绝。现有依赖 RTAS 的 `VisibilityBufferDeferredPass` / ray-traced pass 在 compile 阶段拒绝 metadata scene；SceneResourceManager 同样拒绝从 metadata 创建 resident/RTAS 资源，避免为了后续着色重新导入全量几何。

## 验证

```powershell
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.stream_metadata_* --rhi-validation --rhi-async-compute --output-dir E:/metallic/build-relwithdebinfo/minizorah-m3/bunny
$env:METALLIC_TEST_MINIZORAH='1'
build-relwithdebinfo/tests/MetallicRhiTests.exe --filter RhiRendering.minizorah_vbuffer --rhi-validation --rhi-async-compute --output-dir E:/metallic/build-relwithdebinfo/minizorah-m3/full
```

小场景覆盖缺失的 10 GB 外部 buffer、64 位 buffer offset、全局 ID 稳定性、材质分类变化、双面背面光栅、运行时材质修改、纯流式空 resident buffers、源 baseColor 核对、HW/异步混合 ID 对比、resize、释放及重新打开。全场景用例覆盖全部根实例、606 个有效不透明 BLEND 实例、三个视点和相同生命周期检查。结果与图片写到上述输出目录。

独立材质消费者另经同一三角形的 resident 与 mixed-stream 对照：使用非零 stream record base 时，baseColor 图像与 resident 路径逐像素一致；日志为 `build-relwithdebinfo/minizorah-m3/consumer-contract.log`。

最终结果在 `build-relwithdebinfo/minizorah-m3/verified/`，修复根节点检查前的初轮结果在 `full/`。启用 Vulkan validation 和独立 compute queue；两个全场景进程均通过。没有清空 OS 文件缓存，且运行日志存在 PSO 持久化失败警告，时间包含 shader/PSO 重建，不作为冷磁盘或稳定漫游帧时基准。

| 验收项 | 最终结果 |
| --- | ---: |
| 全局 geometry / instance / source material | 3,163 / 19,144 / 3,283 |
| GPU 根实例映射、变换及材质 ID | 19,144 全部核对通过 |
| 保留的有效不透明 BLEND 实例 | 606 |
| 根集合 baseColor 核对 | 2,023,008 像素，最大 8-bit 通道误差 0 |
| HW / async hybrid 覆盖差异 | 0 / 2,073,600 像素 |
| HW / async hybrid ID 差异 | 42 像素，均在三角形边界；内部差异 0 |
| 异步 compute 分支 | 4 |
| 首像素 / 全部根页可用 | 10.59 s / 13.13 s |
| 进程峰值提交 / 工作集 | 4.08 GiB / 2.15 GiB |
| resident vertex / meshlet draw buffer | 均未创建 |
| resize / 移除 producer / 重新打开 | 通过 |
| 配套回归 | 15 项 RHI + 11 项 SceneGraph，全部通过 |

节点构建中的既有 `setRoots()` 会对未变化的根集合重复做两两祖先检查。增加无变化快速返回后，本机 MiniZorah scene graph 阶段从 29,124 ms 降到 39 ms；没有更改层级或根节点选择语义。

三个 1920×1080 视点的可见像素分别为：原始 **2,032,100**，远景 **47,648**，近景 **2,024,363**。图像为资产原有的标量材质预览：3,283 个材质中 3,241 个 baseColor 为白色、40 个为灰色，不包含原演示的纹理。

同相机、同完整 terminal cut 的 M2/M3 对照：M2 覆盖 2,022,851 像素，M3 覆盖 2,023,008；M2 独有覆盖为 0。45,832 个 ID 差异全部落在 M3 的 double-sided 材质上（全场景共有 7,099 个双面实例），符合新增背面可见性的预期。独立的背面三角形 fixture 同时验证 HW/SW 双面可见、切回单面后消失。

## M4 前的限制

近景采样的页池使用 1,073,426,688 B，仍有 9,228 页排队、768 页 pending，单帧出现 335 次分配失败；全部 terminal 页仍保留，没有加载失败或无效请求。这证明预算下粗覆盖成立，**不代表已收敛到 1.5 px 或达到交互帧时**。M4 优先处理有效候选数量驱动的派发、分配失败退避和页面优先级，再测持续漫游。
