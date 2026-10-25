# Virtual Shadow Maps P1：单太阳硬阴影

对应 [实施方案](VirtualShadowMapsResearch.md) 的 P1，建立在 [P0 页内裁剪验证](VirtualShadowMapPrototype.md) 上。

## 当前实现

新增可选 `virtual-shadow-map-p1` sample 与 `VirtualShadowMapPass`。图从 VBuffer 的真实深度和匹配的 `rasterInfo` 重建接收点，生成按需页，光栅完整场景 caster，再把硬阴影交给 `VisibilityBufferDeferredPass`。固定使用 Sun 槽位 0，不申请 TLAS 或 RayQuery。默认图切换仍属于 P5。

P1 每帧重新标页、分配和绘制。固定 shader/pipeline 按 Device 和 Coverage 程序复用，材质输入和阴影资源仍逐帧更新。光源空间 clipmap 的半幅按层倍增，中心由未抖动相机位置按各层页网格对齐。所有投影使用相同的场景 Z 范围，shadow depth 为 reversed-Z；receiver 支持透视/正交和标准/反向深度，并使用输入的实际 jitter。

额外的全场景粗投影覆盖当前场景 bounds，独占 physical page 0。细页按层由细到粗确定性分配，其余页池耗尽时使用已经绘制的祖先页或粗投影。粗投影独立遍历所有 caster，不占细页工作列表容量；细页工作溢出会让该帧全部接收点改查粗级，避免查询部分完成的阴影。超出所有投影的接收点显示洋红并计数，不静默视为无遮挡。

深度图集在每帧初始化，CPU 不逐页绘制。细页工作由 GPU 对三角形投影包围盒内的已分配页生成，mesh shader 在映射物理图集前裁剪虚拟页四边；沿用 reversed-Z、MASK/custom Coverage 和单双面约定。间接绘制按 65,535 工作组分批。接收查询使用局部接收平面将比较深度移到 shadow texel center，再加世界单位 bias。

## 场景来源与资源合同

- resident Scene：从全部可见 authored 节点及其三角形生成独立的世界空间 caster，不使用主视图剔除结果。
- metadata-only StreamAsset：从所有实例及 `primitiveTerminalGroups()` 枚举常驻终端几何，GPU 从同帧页表解码位置/UV，并读取当前实例变换、显隐和材质绑定。仅用最粗 LOD 的 fallback range 会漏掉提前终止的分支，因此不采用该范围代替 terminal cut。
- VBuffer 发布同帧完整 geometry view，VSM 借用其 runtime/page pool，单独声明读取同步与 completion tracking。VSM 不再次运行主相机 traversal，不覆盖主视图 active cut。全部 terminal 页初始加载完成是前置条件；未 ready 的 stream 输入明确报错，sample 使用 `initialLoad=true`。
- StreamAsset 目前固定使用完整 terminal cut；尚未实现 shadow texel footprint 驱动的独立 refinement 或多视图 streaming demand 合并。终端 LOD 的几何近似可能与完整 resident 几何不同。
- resident 几何上传、VSM 页表、图集和参数为每次 recording 私有资源；stream 几何页池和页表借自 VBuffer runtime，两者都由提交生命周期保留。不提前发布跨帧有效缓存，Pass 当前不支持 frame overlap。
- 材质范围为 OPAQUE/MASK、UV0 base-alpha 纹理及 custom Coverage；支持 base-alpha 的 UV transform。BLEND、未应用的 displacement 和其它 UV 集明确不在 P1 内容范围，P4 再扩展。

## 图接口与配置

输入：`depth`（D32）及 `rasterInfo`（144 B，同 scene/view/frame）。输出：`shadow`（R8 硬可见度）、`parameters`（320 B 单灯过渡接口）、`debug`（RGBA8）、`statistics`（16×uint32）。同一太阳通过旧接口的 tagged identity 匹配；硬阴影 0/1 与既有 sqrt 编码兼容，不改变 Deferred 的解码约定。

| 属性 | 默认值 | 含义 |
| --- | ---: | --- |
| `pageSize` | 128 | 页边长，二次幂，最多 128 |
| `virtualPagesPerAxis` | 16 | 每个 clipmap 的页数/轴；默认 2048 虚拟 texels，最大 128 页/轴即 16K |
| `clipmapCount` | 4 | 细 clipmap 层数，另有一层全场景粗级 |
| `firstClipmapExtent` | 4 | 第一层半幅，单位米 |
| `physicalPageCount` | 256 | 含粗级页的池容量；默认 D32 图集裸内存 16 MiB |
| `maxWorkItems` | 65535 | 细页 triangle/page 工作容量；粗级绘制独立 |
| `shadowBias` | 0.002 | 向光源的世界单位 bias |

OpenPBR sample 为 88,264 authored triangles 单独设置 `maxWorkItems=1048576`（工作 buffer 8 MiB），并固定 VBuffer `autoLod=false, lodLevel=0`。主视图简化几何与完整 resident caster 不一致会产生自阴影斑块；固定 LOD 是 P1 演示的明确限制，尚未实现跨视图几何误差控制。

debug：细级为 `(levelCode, 0.5, 0)`，粗级为蓝色，背景黑色，投影外洋红。`VirtualShadowMapPages` 检查点另提供 atlas、page table、projection buffer。

statistics 顺序：requested fine pages、allocated fine pages、pool overflow、requested fine work、stored fine work、work overflow、非背景 receivers、coarse fallback receivers、background、outside-all、triangle slot capacity、clipmap count、physical capacity、ancestor fallback、disabled、reserved。stream triangle slot capacity 是 cluster count×128，尾 slot 不代表实际三角形。

## 验证

```powershell
cmd.exe /d /s /c '"E:\VisualStudio\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=14.51.36231 -winsdk=10.0.26100.0 && cmake --build build-pass-stages-nrd --target MetallicRHITests -j 8'
python -B Tools/CheckShaderModules.py
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_scene --rhi-validation --output-dir .cache/vsm-p1
```

`virtual_shadow_map_scene` 使用真实 glTF 地面和屏外 quad：移除 quad 不改变主视图深度，但应移除其投到可见地面的阴影。独立 CPU camera-ray / light-ray 参考验证硬可见性。标准深度由真实 Raster D32 经过 GPU `1-depth` 转换生成，并同步修改 metadata；主 renderer 本身固定 reversed-Z，测试没有依赖已忽略的 `camera.reversedZ` 属性。

测试组合覆盖 resident/metadata-only、透视/正交、标准/反向 receiver depth、页网格平移、clipmap 边界、只有一页的池溢出和只有一项的工作队列溢出。另以实际 Deferred 阴影开关对照验证单太阳消费链，并检查场景边缘的粗页空 texel、resident/stream 的 MASK alpha rejection。当前 fixture 关闭 jitter，不能将这些结果称为 temporal jitter 验收。

2026-10-25 本机 GPU 验收：`Metallic` 与 `MetallicRHITests` Release 构建通过，`virtual_shadow_map_scene` 通过。38 组主矩阵共检查 1,666,372 个非几何阴影边缘像素，可见度和 receiver depth 均零差异；4 组 MASK 测试各检查 49,601 个像素，也均零差异。16 组 resident/StreamAsset 对照的阴影逐像素一致；页池溢出与工作队列溢出得到相同粗页结果。透视/正交屏外 caster 分别改变 4,345/3,735 个阴影像素，主视图深度完全相同。

实际 Preview 设备开启同步验证，包含析构阶段的 validation 消息为 0；RayQuery 和 acceleration structure 功能关闭。Deferred 对照中 3,212/3,212 个阴影内部像素变暗，43,105 个远离阴影边缘的受光像素保持不变。该测试同时修复了 Deferred CPU 管线描述意外沿用 `requiresRayQuery=true` 默认值的问题。

8/8 个实际运行 SPIR-V 入口通过 `spirv-val --target-env vulkan1.4 --scalar-block-layout`，7 类 ABI 布局与当前源码一致，无 RT capability，fragment 无 `EarlyFragmentTests`。Slang module ownership audit 通过。P0 GPU 页原型回归亦通过。

相关回归：`gpu_scene_submission_recovery`、`gpu_scene_coverage_classification`、`stream_scene_cache_routing` 通过；后者需要运行器的 `--rhi-bindless`。`stream_metadata_contract` 仍在既有材质热编辑断言失败（`Material resolve ignored a live stream material edit`）。将本次修改过的 VBuffer publication 和 Streamer 缓存路由两文件恢复至 HEAD 后重新构建，该失败仍原样复现；本次未修改其材质解析实现。隔离记录在 [baseline run](../.cache/vsm-p1-baseline/run.log)，P1 文件已逐字节恢复并重新构建。这不计为通过的回归项。

本地证据（均在忽略的 `.cache/` 中，不纳入源码）：

- [主矩阵、MASK 与 Deferred 统计](../.cache/vsm-p1/VirtualShadowMapScene/VirtualShadowMapScene.json)
- [实际 Preview validation](../.cache/vsm-p1/VirtualShadowMapScene/PreviewValidation.json)
- [Deferred 阴影开启图](../.cache/vsm-p1/VirtualShadowMapScene/Deferred-ShadowOn.png)
- [独立几何参考误差图](../.cache/vsm-p1/VirtualShadowMapScene/Resident-Caster-PerspectiveReversed-Difference.png)（灰色为排除的几何边缘，红色才是误差）
- [实际 SPIR-V / ABI 检查](../.cache/vsm-p1/spirv/Validation.json)

实际 OpenPBR sample 另以 512×512 的同相机、LOD 0 阴影开关对照验图：[对照 PNG](../.cache/vsm-p1-openpbr-final/On-Off.png)、[比较结果](../.cache/vsm-p1-openpbr-final/Comparison.json)。两图深度逐字节相同，HDR 全部有限，5,159 个像素光照变化；60,933 个接收像素使用细级，粗级和投影外均为 0，球面未再出现先前的斑块。此既有 MaterialBaseline harness 开启设备 RayQuery；无 RT 结论来自上面的专用场景测试。

示例采用两次单变量对照：只把工作预算从 65,535 提到 1,048,576，细级恢复但斑块仍在；再仅将 receiver 固定 LOD 0，斑块消失。没有直接读取这份 sample 的工作溢出 counter，预算因果判断来自上述图像和层级对照。核心 fixture 则直接断言工作溢出 counter 与完整粗页结果一致。

没有性能加速结论、跨帧缓存、SMRT 或默认切换结论。P1 新入口尚未独立验收非恒定 alpha 纹理/custom Coverage 组合、动态物体/相机长程运动及 shadow LOD 质量；P0 的材质验证不能替代这些完整场景检查。

## 源码入口

- [运行时分页与录制](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp)
- [独立 caster 来源](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.cpp)
- [分页、光栅与查询 shader](../Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang)
- [统一几何解码](../Shaders/Modules/VirtualShadowMapGeometry.slang)
- [RenderGraph Pass](../Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp)
- [可选 sample 图](../Pipelines/Samples/virtual_shadow_map_p1.metallic_graph.json)
- [实际场景验收](../tests/rhi/VirtualShadowMapSceneTests.cpp)
