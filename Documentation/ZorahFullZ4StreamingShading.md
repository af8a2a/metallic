# ZorahFull Z4：流式着色

2026-09-19。Z4 接通了属性页到实时 OpenPBR、透明续追及 MASK 阴影的消费者。本阶段使用合成场景和 Z1/Z2 小探针验证；没有执行 Full 全量 cook，也没有把小探针结果当作完整场景首帧或性能基准。

## 实现

- 普通流式 VBuffer 从当前驻留页重建 normal、UV 和 authored tangent，不再只在细分路径解码属性。缺切线时保留 UV 导数回退；镜像手性、非均匀缩放及法线贴图使用稳定的世界空间 TBN。几何法线不随观察方向翻转，最终着色法线在法线贴图求值后 face-forward。
- 实际镜像探针还修复了常驻 VBuffer 的绕序缺口：普通 HW、SW 和细分回退路径对负 determinant 交换三角形顶点，法线锥保持 authored front 方向。原始三角形 ID/属性顺序保持不变，避免单面负缩放对象被当作背面剔除。
- 纹理 footprint 使用三角形 UV/世界面积、ray cone 和纹理变换。普通材质的 normal map 跟随 footprint；已有玻璃的确定性 mip 规则保留。KTX2 BC5 normal Z、BC4 specular alpha 和 sRGB 解码复用 Z3。
- 导入、文档保存、材质更新和 GPU 资源支持 `KHR_materials_specular` 的因子/颜色/两张贴图，以及 `KHR_materials_unlit`。OpenPBR 材质结构为 720 B，同步更新 deferred、材质可视化和 RTXDI 的布局。材质编号保留完整 uint。
- MASK 在写 VBuffer/depth 前执行 alpha 测试；MASK/BLEND cluster 进入硬件光栅，混合模式中的不透明簇仍可走软件光栅。常驻与流式光栅共享 SceneResourceManager 的 image/view 和 mip 尾链。
- 流式光栅有独立 bindless heap，不能复用常驻 heap 的 descriptor index。新增其自身的材质、纹理重映射 buffer 和 image descriptor；资源按在途帧退役。解除流式 image heap 的固定 4096 限制，按真实纹理数量及设备上限检查。
- 纯光栅设备只申请材质/贴图时，资源准备跳过常驻 GPU 几何与 RTAS，不因共享纹理 provider 引入光追能力要求；带 RT 的常驻图仍复用原有完整资源快照。
- CLAS 命中通过稳定的 `pageIndex * maxPageClusters + clusterIndex` 解码当前页；搬移后的地址不充当材质或几何身份。RT shadow 和透明续追能重建 UV、材质、TBN，并过滤 MASK。
- BLEND/玻璃主可见面仍由 VBuffer 确定，随后进入有界 OpenPBR 射线续追。BLEND 使用随机空交互，玻璃复用折射、界面栈和体吸收。该路径要求 `enableClas=true` 和 `enableClusterRtx=true`；只构建 CLAS 不会产生可追踪的 TLAS。

## 使用与复现

使用现有 **GPUDrivenSample** 实时图：`gpu_driven_realtime.metallic_graph.json` 已连接 VBuffer → Shadows → Deferred → 后处理，且开启 CLAS/TLAS。`GPUDriven / MiniZorah VBuffer` 的旧标量 material resolve 仍是诊断图，不是贴图着色入口。Full 本体需先完成 Z5 cook。

File/Open 切换新场景默认查找 `<源文件>.meshstream.bin`，例如 `StoneUdim.gltf.meshstream.bin`。当前 Z2 探针的自定义输出名为 `StoneUdim.meshstream.bin`；本测试通过图属性显式指定 `path` / `streamAssetPath`。编辑器中复现时同样填写实际路径，或 cook 到 File/Open 的默认位置。

在 MSVC 开发环境中：

```powershell
cmake --build build-release --target MetallicRhiTests MetallicSceneTests MetallicGPUDrivenSample -j 4
$env:METALLIC_ZORAH_Z4_PROBES='E:/metallic/build-release/zorah-z2/probes/probes.json'
New-Item -ItemType Directory -Force build-release/zorah-z4/probes | Out-Null
build-release/tests/MetallicRhiTests.exe `
  --gtest_filter=RhiRendering.stream_material_shading:RhiRendering.stream_material_transmission:RhiRendering.stream_material_shadow:RhiRendering.zorah_stream_material_probes `
  --output-dir build-release/zorah-z4/probes `
  --gtest_output=json:build-release/zorah-z4/results.json
```

探针测试独立解压 meshopt bufferView 成常驻参考，再与已有 cook 的流式 LOD0 比较。采用相同相机、256×256 分辨率和纹理资源设置；从非退化三角形选择近景，避免分散实例或细长植物使整幅画面几乎为空。输出每个探针的 baseColor、mappedNormal、normalTexture、final PNG 和 JSON 像素统计。玻璃/BLEND 的 final 使用不同于普通常驻 realtime 的续追，因此不把这两项 final 当作等价参考。

旧常驻 VBuffer 会跳过 BLEND draw，因此仅在生成的 BLEND 参考副本中将其改为 MASK、cutoff `1e-6`，匹配正 alpha 表面，用于属性对照。原始 glTF 与流式材质仍为 BLEND；透明合成由带背景的独立测试验收。不能将这个参考副本当作常驻 BLEND 合成实现。

Unlit 探针缺少源 NORMAL；常驻导入生成平滑法线，流式无 normal 属性时使用面法线。其 mappedNormal 不做等价断言，仍检查有效覆盖、baseColor、normalTexture 和最终 unlit 颜色。该差异不影响 unlit 辐射结果，也不作为其他有法线材质的误差豁免。

## 验证

验证使用 Release 构建与 Vulkan validation，图像及 JSON 保存在 `build-release/zorah-z4/`。合成测试包括：

| 测试 | 检查内容 |
| --- | --- |
| `stream_material_shading` | 257/258/259 材质编号、纹理变换、normal map、镜像/非均匀缩放、unlit；常驻/流式 8 个视图；HW/混合 MASK 孔洞 |
| `stream_material_transmission` | BLEND 保留前景与背景辐射；IOR=1 玻璃后的 MASK 孔洞与材质 260；材质分桶/不分桶输出一致 |
| `stream_material_shadow` | 斜向光将 MASK 棋盘投射到独立接收面；常驻 RT 与流式 CLAS 阴影覆盖对照 |
| `zorah_stream_material_probes` | 石材、缺切线实例、叶片、texture transform/BC4、玻璃、BLEND、unlit 和镜像实例的真实 KTX2 探针 |

合成属性对照的 RGB8 通道平均误差最大 **0.00320/255**，normal/TBN 视图误差为零。验收统计排除轮廓边界，另用显式孔洞计数检查覆盖，避免仅比较前景而漏掉多画的表面。真实探针以 mappedNormal 的几何覆盖作为比较区域，不把合法的黑色 baseColor 误判为空几何。

八个真实探针分批完成，32 条记录中 29 条作等价像素对照；玻璃/BLEND final 和无源法线的 unlit mappedNormal 按上述规则单独解释。汇总保留原始数据来源：[Z4ProbeComparison.json](E:/metallic/build-release/zorah-z4/Z4ProbeComparison.json)。

| 探针 | 可比较视图最大平均通道误差，RGB8 /255 |
| --- | ---: |
| StoneUdim | 0.003222 |
| InstancingNoTangent | 0.004805 |
| MaskedLeaves | 0.083717 |
| TextureTransformBc4 | 0.256380 |
| Glass | 0.005601 |
| Blend | 0 |
| Unlit | 0.000486 |
| MirroredInstances | 0.000605 |

最大离群像素比例为 0.2392%（离群定义：任一 RGB8 通道差大于 8）。Unlit 最终输出与 baseColor 对照误差一致；单面镜像模型修复后有 2755 个内部有效像素，最终颜色平均误差 0.000605/255。

兼容性回归通过：`stream_metadata_vbuffer`、`stream_metadata_contract`、常驻 RT alpha preview、无 RT 能力设备上的 GPUDriven alpha raster。Scene 的属性 cook、材质导入、文档/复合材质编辑和流式打开回归共 19 项通过，1 项可选完整 Zorah 几何对照未启用。构建及测试日志见 `build-release/zorah-z4/`；测试期间没有 VUID 错误，PSO 缓存写入警告不作为帧时数据使用。

最终单面镜像合成测试：[shading-final.json](E:/metallic/build-release/zorah-z4/shading-final.json)。镜像探针与旧 alpha 光栅通过记录：[mirror-rhi.json](E:/metallic/build-release/zorah-z4/mirror-rhi.json)；该批首次合成项曾因输出目录未创建而失败，随后已修复夹具目录创建并在 `shading-final.json` 复跑通过。其余分批记录保留了常驻 BLEND 参考、unlit 无法线参考和镜像绕序排查过程；八探针最终有效数据以 `Z4ProbeComparison.json` 为准，不将旧批次的汇总失败行改写为成功。

## 已知边界

- 这是受预算的 mip 尾链采样，不是按屏幕需求升降纹理 mip；持续纹理流送与跨几何/CLAS/纹理总预算属于 Z6。
- MASK 使用最细驻留 mip 的确定性双线性 repeat 覆盖。当前 Full 的 sampler 是 repeat/linear；通用 sampler 地址模式和 neural alpha 的统一不是本资产验收范围。
- OpenPBR 与 glTF 的金属/介质混合参数化不同；specular 控制保留纯介质和纯金属端点，部分金属度采用插值映射，不宣称与另一渲染器逐像素一致。
- 透明续追默认 2 samples、8 层 BSDF 深度，可调至 16；BLEND 空交互上限 16。有限采样可能有噪声；不会自动转成排序透明或 OIT。
- 独立 SIGMA shadow 通道目前支持 MASK 孔洞，BLEND/玻璃仍按二值阻挡处理；OpenPBR 续追中的 shadow transmittance 支持部分 alpha 和有色透射。完整彩色玻璃软阴影尚未统一。
- 当前 CLAS alpha query 强制 non-opaque 候选回调，包括不透明簇；后续可按实例 alpha 类型缩小回调范围。此轮不报告 Full 的帧时收益。
- 有 alpha 的流式 displacement 仍被显式拒绝；ZorahFull 本次不依赖该组合。静态变形、通用 WPO 和动态角色不在此次范围。

Z5 下一步是基于稳定的 Z2 属性格式生成 Full 全量 meshstream，以 Full cfg 相机验证 512 cap / 2 GiB 纹理预算下的首帧、根级几何、MASK/透明内容及场景切换生命周期。
