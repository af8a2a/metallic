# Virtual Shadow Maps P4：过滤与内容覆盖

对应 [实施方案](VirtualShadowMapsResearch.md) 的 P4，基于 [P3](VirtualShadowMapsP3.md) 的多灯、稳定身份与提交后页缓存。实现 Hard / PCF / SMRT、receiver footprint 选级，以及完整场景位移 caster。默认实时图切换和代表性大场景长期验收仍属于 P5。

## 光源与过滤合同

`VirtualShadowMapLightsPass.filter` 接受 `hard`、`pcf`、`smrt`，缺省为 `hard`。旧 `VirtualShadowMapPass` 继续发布兼容 SIGMA sqrt 解码的硬阴影；其非 hard 配置明确拒绝，软阴影使用多灯线性可见度接口。

Sun/Moon 直接使用已有 `angularRadius`，单位弧度、含义为半径。局部光新增可保存、可编辑的 `sourceRadius`，单位米、缺省零；经 `LightProperties`、原生/虚拟灯与 `GPUPunctualLight.spot[2]` 传递，原 64 B GPU 灯布局不变。这是 VSM 阴影发光尺寸，不改变现有解析点光/聚光的强度、BRDF 或 RT/PT 采样语义。旧场景未填写时保持零。

| 属性 | 缺省 | 含义 |
| --- | ---: | --- |
| pcfRadius | 1 | 3×3 PCF 的选定虚拟 texel 间距；零严格走 Hard |
| smrtRayCount | 8 | 每 receiver 的圆盘光源样本数，1–32 |
| smrtSamplesPerRay | 8 | 每条有界射线的步数，另含一次接收点查询 |
| smrtMaxRayLength | 10 m | 米制软阴影搜索上限；局部光另受光源视角展开限制，超出搜索距离的遮挡仍有效 |
| smrtRayLengthScale | 0 | 0 保留固定米制上限；大于 0 时，再以相机距离乘该比例限制搜索长度，范围 0–1000 |
| smrtMaxRayAngle | 0.03 rad | 局部光的视角展开上限，范围 0.001–0.35，不改变光源尺寸 |
| smrtTemporalEnabled | false | 多灯 pass 启用逐帧光源采样与独立可见度历史；Sponza Sample 显式开启 |
| smrtTemporalMaxFrames | 32 | 每灯累积帧数上限，1–64；1 可观察同一采样相位的原始结果 |
| receiverFootprint | true | Hard / PCF / SMRT 均按接收面像素投影到光空间的足迹选择起始级；false 显式从最细级开始 |
| tessellationFactor | 4 | 位移 caster 独立细分因子，1–8 |
| tessellationSubdivisionDepth | 0 | 位移 caster 额外均匀四分层级，0–2；每级增加 4 倍微三角 |

SMRT 参考本机 Unreal 5.7.4 的反向深度历史算法：从远端向 receiver 走二次分布采样，最后显式查询 receiver。首个有效样本做普通深度比较，以保留有界搜索之外的遮挡；后续按相邻参考深度的间隔判断交点。遇到当前单层深度无法表示的遮挡间隙时，沿前一表面的深度历史作有界斜率外推。有效清空深度也参与历史，未映射样本才跳过。每个查询沿用 receiver plane 的 texel-center 修正和米制 bias；零光源尺寸严格回到 Hard。

方向光在统一米制灯光深度中维护历史，外推斜率上限为 0.05 m。局部光限制光源视角展开（`smrtMaxRayAngle`，默认 0.03 rad）并最多行进 receiver-light 距离的 0.75，避免接近发光面的透视奇点；采样按起始面投影深度参数化。点光跨面使用统一逆径向深度，外推斜率上限为射线逆径向深度变化的 0.05，不混用六个面的轴向深度。保留本项目的 `smrtMaxRayLength` 米制上限，因此搜索距离仍限制可表示的半影宽度。

光源圆盘采用逐像素扰动的低差异序列，步长也使用像素种子；不再让全屏共享同一个光源样本。另在 receiver 选定的 texel 范围内抖动阴影图查询，接收平面修正仍锚定原射线点。固定世界偏移在点光跨面时连续。关闭时域时保留固定种子和确定输出；开启时使用 64 帧循环的低差异相位，同时驱动标页、光源圆盘、步长和 texel 抖动。1 条射线的原始可见度仍为二值，多灯 pass 的独立历史负责累积；不依赖 TAA、NRD 或 SIGMA。

单层深度不能恢复被第一层遮挡的其它几何。离散步长、遮挡层断点、极大光源和搜索距离限制仍会产生近似误差；SMRT 不是面积光几何射线的精确替代。

## 页寻址、需求与缓存

Hard / PCF / SMRT 共用 receiver 像素足迹选级：在原始接收点确定起始 clipmap/mip，Hard 中心查询、PCF 全部 tap 和 SMRT 整条射线均从该级开始，覆盖范围或驻留不足时再向更粗级查询。PCF 不再按偏移后的 tap 重新选级。标页和查询使用相同起始级；中心 debug 及回退计数也遵守该策略。PCF 半径为零、SMRT 光源尺寸为零时仍严格回到 Hard，且不改变基础分辨率。关闭 `receiverFootprint` 可显式恢复三种模式的最细级起点，单灯兼容 pass 也提供这个开关。

足迹由深度重建的接收面法线、相机像素微分和光投影共同计算，使用二维投影 Jacobian 的最小奇异值，保留掠射方向被压缩的细节；局部光包含透视除法，点光统一使用接收点所在的 canonical face。页预算不足时按粗层优先准入，再补细级；已准入 resident 保持物理槽，缺页仍先使用空闲槽再驱逐。这样细化需求超过预算时可使用相邻粗级，避免优先铺满细级后失去较粗层覆盖。

逐步趋零的光源采样足迹不能用于提升精细度，否则会请求大量不需要的细页、挤出有效 receiver 页面。每个 tap 独立重新投影、查页、定位物理 texel。细页未准入或 raster work 溢出时查询已完成的粗级。使用整数 texel load，不对物理 atlas 相邻页做无边界过滤。

Demand 覆盖同一 PCF tap / SMRT 反向采样位置，并共用像素种子、投影插值和 receiver 起始级。当前实现保守逐 tap 标记，尤其点光六面和高样本数时成本较高，不能由热缓存没有光栅工作推断过滤便宜。

过滤点光先录制六面的 atlas、页表和投影，再逐面 resolve。每面持有显式六面描述表，tap 根据世界位置重新选面；需求标记也处理其它 canonical face 的 receiver 射线。原始 receiver 最终仍只由自己的 canonical face 发布。六面资源由同一 recording 保活，继续禁止跨帧重叠写同一缓存域。

发光尺寸和过滤参数不是深度内容身份；它们改变需求和查询，已有完整页仍可复用。新增需求可分配/重画新页。灯位置、投影、caster、纹理发布和位移开关等真实内容变化继续进入失效边界。

CPU/Slang `ShadowPush` 扩为 192 B，`ShadowProjection` 保持 112 B，新增每面描述 `ShadowFace` 为 32 B。几何参数扩为 48 B，基础 triangle 保持 112 B。公开多灯 table 和 Deferred 参数布局保持 P3。

每面统计仍为 32 uint。新增 `[30]` 为过滤页查询次数，`[31]` 为其中粗级查询次数；`[9]` 包括无效粗页查询。射线段在完整 caster 包围投影外没有可表示遮挡层，不计作缺页。中心页 debug 沿用原显示，不代表全部过滤 tap 的级别；过滤回退应读计数。

## 位移、材质与纹理发布

位移开关取自匹配 receiver 的 `VisibilityBufferFrameInfo.reserved`，和生产主视图一致。关闭 tessellation 时两者都消费基础网格；开启时 VSM 为完整 resident 场景或 Stream terminal cut 生成独立细分 caster，不借用主相机可见 patch 列表。

几何求值复用 `VisibilityTessellation` 的 pattern/domain、边界规范化、插值法线和 `tessHeight`；高度来自当帧已发布物理纹理的 mip 0，使用相同 UV transform。额外层级使用同一 dyadic patch 四分规则，再在每个叶 patch 上应用固定细分因子。独立细分密度与主视图自适应拓扑不同，曲面近似误差取决于该密度，不宣称两者逐三角形一致。Bounds 扩张包含位移幅度，内容 key 包含位移参数、法线、因子和额外层级。48 B geometry 参数的原 reserved 字段现用于 subdivision depth，布局大小不变。

Alpha / height texture 使用不可变 material texture snapshot 并保活。`materialTexturePublicationRevision()` 跟随实际物理 image/mip 发布进入 dependency key，不能仅用 descriptor 地址判定内容不变。颜色等与深度无关的编辑保持原缓存合同；共享纹理 publication 的保守失效粒度仍可在后续测量后细化。

支持 OPAQUE、MASK、custom Coverage 和 double-sided 的既有合同。位移阶段对未实现的材质/UV/资源组合明确返回 Unsupported。OPAQUE/MASK caster 按不透射几何遮挡，即使其材质有 transmission；这是一种有限近似，不是有色玻璃透射支持。BLEND 明确拒绝，多层透射和体积使用显式 RT/PT 策略。Native strands 属独立 `StrandPass` 外部资产，不在三角形 Scene caster 域内，不向本 VSM 投影，继续使用原 strands 策略。

## 示例、成本与验收边界

独立 [virtual-shadow-map-p4](../Pipelines/Samples/virtual_shadow_map_p4.metallic_graph.json) 使用 [OpenPbrVSMQuality](../Asset/LookDev/OpenPbrDefault/OpenPbrVSMQuality.metallic_scene.json)，包含 Sun/Moon、0.12 m point 和 0.08 m spot。固定 512²、关闭 temporal jitter，SMRT 8 rays × 16 segments、总页预算 2048，receiver 请求 LOD 0。流式初始加载仅保证 terminal cut 驻留，不能把手动 LOD 0 设置等同于首帧已加载完整细节。

P3 每灯屏幕纹理数组架构及内存/复制成本仍在：capacity=8、512² 的公开 R8+RGBA8 数组裸存储为 60 MiB；2048 个 128² D32 页的有效深度容量为 128 MiB，实际方形 atlas 分配还含取整浪费，另有私有输出、work buffers 与在途资源。当前没有直接 Deferred 页表查询，也没有同工作负载 RT/VSM 性能收益结论。Receiver mask、shadow HZB、静动双层未在缺少测量依据时添加。

本阶段不承诺局部 near plane 内 caster、任意透明内容、主视图自适应细分完全同拓扑或长程大型场景质量。P3 的 near、Stream terminal cut、帧重叠和预算重建瞬时内存限制继续适用。

## 验证记录

使用既有 `build-pass-stages-nrd` Release、MSVC 14.51.36231 / Windows SDK 10.0.26100.0，没有改动配置。

| 检查 | 实际结果 |
| --- | --- |
| Metallic / MetallicRHITests / MetallicSceneTests / MetallicShaderRequestsTests 构建 | 通过 |
| Slang module ownership | 386 files / 274 modules / 112 implementation units，通过 |
| Deferred / Trusted / NativeShaderWarmupRequests CPU shader tests | 28/28 通过 |
| Scene light / lighting / photometry / component round trip | 11/11 通过 |
| `virtual_shadow_map_quality` | 83 帧 Cached / Redraw 全 24 层逐像素一致，Vulkan / synchronization validation 0 消息 |
| `virtual_shadow_map_content` | 42 帧 Resident / Stream 缓存对照一致，10 个输入边界负例通过，validation 0 消息 |
| 实际 runtime SPIR-V 与 scalar-layout Vulkan 1.4 校验 | 10 个入口、8 个 CPU/Slang 布局通过；无 RayQuery/RT capability |
| P0–P3、RT+SIGMA、GPUScene lights / publication / recovery、RG array / sample、普通及递归位移回归 | 19/19 通过，无跳过 |
| 四灯 OpenPBR：8×16 与独立 32×32 对照配置 | 两套配置各自 Cached / Force / Flat 输出完全一致且有限，validation / 编译错误 0 |
| 真实 DisplacedPlane：512²、主视图自适应细分、VSM factor 8 + 额外两级 | Cached / Force 输出一致且有限，validation 0；16K 虚拟分辨率质量档已检查实际画面 |

质量矩阵在 193×129、无 RayQuery/RTAS 的独立 preview device 上执行。页大小 128、virtual axis 16、总页预算 2048、32 rays × 32 segments、搜索距离 12 m、bias 0.002 m。CPU 参考以 authored triangles、双精度相机/光线求交和 256 个面积光源样本构造；起点是实际 receiver，只有 `1e-7` 求交 epsilon，没有沿法线移动参考点。

合计 331,006 个独立接收样本：平面 ROI 每像素检查，Room 每 3 像素检查；这些区域内 receiver depth 错误为零。最大平面 MAE 为 0.00297434，最大半影 MAE 为 0.10562824，低于预定 0.04 / 0.20 门槛，亮区/本影中误差超过 0.1 的样本为零。接触处软像素数 Directional / Spot / Point 分别从 477 / 628 / 628 缩小至 6 / 37 / 34。相机、灯及 caster 移动、光源尺寸编辑、零尺寸严格回到 Hard、PCF、极低预算粗页回退和恢复均有断言；关闭阴影/零预算还要求全层 255、统计全零，并确认不进入 caster preparation。

点光单层平面将实际阴影边缘放在两面接缝与三面交点上，seam / corner MAE 为 0.00550643 / 0.02146739，均有独立 CPU 半影和 GPU 软像素。多层 Room 在 Resident / Stream 各执行 20 个视角，覆盖全部 12 条接缝和 8 个 corner，各有 2,416 / 380 个 seam / corner 样本；该多层场景记录近似误差，不套用单层平面的质量门槛。Room 使用 32 个 quad primitives 保留相同的全部 64 个三角形，并解码 terminal cut 断言数量一致。原单 primitive 的 32-triangle 粗级与 GPU 的 262 个深度差异完全吻合；没有以删除几何或放宽误差阈值处理此问题。该 fixture 不构成任意简化 Stream cut 的质量保证。

位移内容矩阵覆盖镜像非均匀变换、单面 winding、关闭位移、混合基础材质、负幅度/center 编辑、因子 4→8、额外层级 0→1→2→0、9×9→5×5 物理纹理重新发布，以及 MASK alpha publication。GPU 解码位置对独立双精度高度采样参考的误差门槛 `2e-5 m` 不变；额外层级还检查 UV 有向面积和一阶矩，防止遗漏或重复叶 patch。Stream Reference codec 的法线量化先作用于输入，随后独立计算变换和高度；没有用 GPU 结果反校参考。纹理测试验证不可变 image + publication revision 生命周期，未模拟完整 KTX 异步 mip 流送。

真实位移场景保留主视图 `tessellationMaxFactor=8 / tessellationMaxSplitDepth=2 / tessellationEdgePixels=1`，阴影质量档为：

```json
{
  "filter": "hard",
  "virtualPagesPerAxis": 128,
  "physicalPageBudget": 2048,
  "maxWorkItems": 4194304,
  "tessellationFactor": 8,
  "tessellationSubdivisionDepth": 2,
  "shadowBias": 0.002
}
```

这是可选质量档，不是默认配置。128 个 128 px 页组成 16K 虚拟轴；该资产的 128 个基础三角形生成 196,608 个阴影微三角，是没有额外层级时的 16 倍。没有以增大 bias 或降低主视图细分解决暗斑。低分辨率诊断、同网格 CPU BVH 参考及额外高密度参考保留在 `vsm-p4-displaced-{matched,mask,fine}/`；同网格 N16→N128 的显著误阴影从 966 降到 76，保留的真实遮挡从 132 增至 179。receiver-plane 公式在掠射角下跨越曲面时存在有限 texel 近似误差，残余误差仍不能当作精确几何光线结果。

额外 768² 直接 R8 mask 验证先审计 producer 实际边细分率，确认两级叶 patch 全部饱和 factor 8，再使用真实 StreamAsset、196,608 个三角形做独立 CPU 相机首交和向太阳求交。GPU / CPU 相机首交距离误差 P99 为 22.16 µm，只有 3 个轮廓像素的覆盖不同；参考起点无 normal offset，忽略自身三角形，自交 epsilon 为 `1e-7 m`。129,522 个直接受光面样本中有 15 个误阴影、9 个漏阴影、8 个正确阴影，其余正确受光。全部表面（含背光面）为 15 个误阴影、68 个漏阴影；这是保留 2 mm bias 的有限精度结果，不宣称零误差或全部微小接触阴影被保留。该检查与原始 512² 自适应主视图的缓存/重画及人工画面检查分别记录。

四灯 OpenPBR 在 512² 下每变体运行 6 帧、读取第 0/1/5 帧 HDR，再额外读取显示输出和 receiver depth。8×16 和 32×32 均保留 Hard 对照并人工检查 PNG；提高样本数后仍有复杂曲面的颗粒状阴影，不能据此宣称达到默认画质。专用 GPU 矩阵和这个启用 RayQuery 的 baseline harness 分开记录，后者不证明全局没有 RTAS 成本，也不构成 RT/VSM 性能 A/B。

主要复现入口：

```powershell
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_quality --rhi-validation --output-dir .cache/vsm-p4
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --filter virtual_shadow_map_content --rhi-validation --output-dir .cache/vsm-p4
.\build-pass-stages-nrd\tests\MetallicShaderRequestsTests.exe '--gtest_filter=*Deferred*:*Trusted*:NativeShaderWarmupRequests.*'
.\build-pass-stages-nrd\tests\MetallicSceneTests.exe '--gtest_filter=*Light*:*Lighting*:Photometry.*:SceneEditing.ComponentPropertyRoundTrip'
```

本机证据位于忽略的 `.cache/`，不作为源码提交：`vsm-p4-quality-run10.log`、`vsm-p4-content-run6.log`、`vsm-p4-regression.log`、`vsm-p4-shader-tests-final.log`、`vsm-p4-scene-tests-final.log`，以及 `vsm-p4/{VirtualShadowMapQuality,VirtualShadowMapContent,spirv,diagnostics}/`。真实场景的 Cases/graphs/脚本分别保留于 `vsm-p4-openpbr-config/`、`vsm-p4-openpbr-high-config/`、`vsm-p4-displaced-final-config/`，输出对应 `vsm-p4-openpbr/`、`vsm-p4-openpbr-high/`、`vsm-p4-displaced-final/`，其中 `Comparison.json` 记录逐输出差异。

未重新运行整个历史测试集；[P2](VirtualShadowMapsP2.md) 已记载的 Deferred 法线对照失败及 CPU shader suite 陈旧计数断言仍未解决。长程大场景、复杂曲面质量、峰值显存、同工作负载性能和默认图切换继续保留验收门槛。

## SMRT 搜索距离漏影回归（2026-10-29）

`virtual_shadow_map_smrt_distance` 使用独立几何参考检查方向光、聚光和点光。遮挡板高度为 1.5 / 12 m，软搜索距离为 1 / 10 m，射线数为 1 / 8，每条射线 16 段；局部光位于 20 m 高处。24 组组合另加三类光源各一组 `1e-6` 非零尺寸控制，共 27 组。亮区和本影内部逐像素断言，并检查主视图深度、Hard 对照、热缓存与强制重绘；不会借精确零尺寸的 Hard 旁路掩盖错误。旧实现的基础 24 组中 18 组失败，修复后全部 27 组通过。

```powershell
cmake --build build-release --target MetallicRHITests MetallicVSMSample --parallel 8
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_smrt_distance:*virtual_shadow_map_quality:*virtual_shadow_map_filter_demand:*virtual_shadow_map_compact' --rhi-validation --output-dir .cache/VSMSMRTFix/AfterRegression
python -B Tools/CheckShaderModules.py
```

上述四项 GPU 测试和 shader module 审计通过，无跳过。另在 Sponza Sample 的普通 GPUDriven 管线上验证 Hard、SMRT 1×16 和 8×16，保留 10 m 搜索距离及原相机、太阳、bias 和页预算；1280×720 静态各 32 帧，8×16 沿长廊移动相机 64 帧，再以 1920×1080 验证 1×16 共 32 帧。检查实际 PNG 后确认建筑投影从近到远连续保留。静态第 31 帧、移动第 31/63 帧的阴影及 HDR 与强制重绘逐字节一致；Hard 修复前后 PNG 也逐像素一致。各实景进程均确认无 RayQuery/RTAS、无光追分配，Vulkan / synchronization validation 消息为零。

本机证据保留于忽略的 `.cache/VSMSMRTFix/`：`BeforeRegression`、`AfterRegression`、`BeforeSMRT1Settled`、`AfterHard`、`AfterSMRT1`、`AfterSMRT8`、`AfterMotionSMRT8`、`AfterNativeSMRT1`。实景的 2048 页预算会触发粗级回退，修复前后均存在；本次无 raster work overflow 或无法解析的粗页查询，不把该检查当作全分辨率驻留或长时间显存压力测试。端点查询仍是单层深度近似，远处半影扩展仍受软搜索距离限制。

## Unreal 源码对照后的 SMRT 修正（2026-10-29）

上一节只验证了远处本影保留，未能识别全屏共用采样和严重粗页回退。当前实现已替换其独立线段检测加末端补查方案。参考的是本机 `F:/UnrealEngine` 的 5.7.4 源码：

- `Engine/Shaders/Private/VirtualShadowMaps/VirtualShadowMapSMRTTemplate.ush`：远到近追踪、首样本深度比较、历史斜率外推。
- 同目录 `VirtualShadowMapProjectionDirectional.ush` / `VirtualShadowMapProjectionSpot.ush`：receiver 选级、纹理抖动、局部光角度截断及投影插值。
- 同目录 `VirtualShadowMapSMRTCommon.ush`：逐像素光源采样。
- `Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/VirtualShadowMapArray.cpp`：斜率与角度默认值。方向光 5 cm 转为本项目的 0.05 m。

这里使用 Metallic 的平面圆盘光源合同、米制搜索上限、确定性像素散列和点光逆径向历史坐标；没有引入 Unreal 的蓝噪声资源、时域累积或自适应射线终止。

新增 `virtual_shadow_map_smrt_penumbra`，以 12 m 遮挡板和独立 256 样本 CPU 几何参考验证真实太阳角 `.00465`、较宽方向光 `.04`、聚光和点光。32 条射线检查每像素误差，1 条射线检查半影三个窄带的空间平均。旧实现的 8 个窄带失败，修正后通过；32 条射线的半影 MAE 为 0.0256 / 0.0263 / 0.0852 / 0.0444。`virtual_shadow_map_filter_demand` 新增 16 段、小光源、10 m 用例，旧实现无法满足页面需求，新实现完整准入且不回退粗页。

质量矩阵改用虚拟轴 128，保留所有原误差阈值、广角接触及近远半影收缩检查。原因是旧轴 16 在 receiver 选定级只有 31.25 mm texel，无法解析约 24 mm 的方向光接触半影；它此前依靠错误的逐段细级提升偶然获得更细 texel。轴 128 使该级 texel 为 3.906 mm。额外接触近景对三类光源各检查 516 个独立半影像素，避免仅依赖原广角的 6 / 34 / 34 个样本；近景半影 MAE 为 0.0797 / 0.0520 / 0.0705，门槛仍为 0.20。旧低分辨率失败证据保留在 `AfterRegression` 和 `DitherRegression`，没有放宽断言掩盖它。

Sponza 使用原场景、相机、太阳、2048 页预算和 10 m 搜索距离，1920×1080 原生分辨率、普通 GPUDriven 管线。1×16 与 8×16 静态各 32 帧，8×16 移动相机 64 帧。静态页需求分别为 692 / 697，全部准入，过滤粗级查询为零；旧 1280×720 的 8×16 则请求 7758 页并有超过 90% 的过滤查询回退全场景 128² 粗图。新截图的长廊投影连续，保留半影与细小遮挡，消除了粗图造成的大块锯齿。静态第 31 帧及移动第 31/63 帧的阴影和 HDR 均与强制重绘逐字节一致，无 RayQuery/RTAS 或光追分配，Vulkan / synchronization validation 消息为零。

复现回归：

```powershell
cmake --build build-release --target MetallicRHITests MetallicVSMSample --parallel 8
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_smrt_penumbra:*virtual_shadow_map_filter_demand:*virtual_shadow_map_smrt_distance:*virtual_shadow_map_quality:*virtual_shadow_map_compact' --rhi-validation --output-dir .cache/SMRTUE/Regression
python -B Tools/CheckShaderModules.py
```

上述 5 项 GPU 回归均通过，无跳过；shader module 审计通过，Sample 已重新构建。本机输出在 `.cache/SMRTUE/` 下的 `BeforeRegression`、`DitherRegression`、`ResolvedQuality`、`FinalCoreRegression`、`SponzaSMRT8`、`SponzaNativeSMRT1`、`SponzaMotionSMRT8`。仅对单层几何质量及所列实景做验收；多层遮挡仍有单层深度近似误差，当次验收未进行独立斜面／掠射接收面质量门槛或长时间显存压力测试。1 条射线仍会有空间噪声，当次验收的 Sample 尚未引入时域降噪。

## 三种过滤共享接收点选级（2026-10-29）

`virtual_shadow_map_filter_demand` 增加 Hard 的有效需求准入、中心 debug 级别、零 PCF 半径和零 SMRT 光源尺寸的逐面需求／回退／输出对照，以及关闭 footprint 后恢复最细级需求的检查。旧实现首先在 `Directional-PCF-Cold` 的 Hard 页预算断言失败。原有 scene/cache/local-light 夹具显式使用 `receiverFootprint=false`：它们的页边界、驱逐压力和固定分辨率几何参考依赖最细级；未放宽其误差或覆盖断言。接收点匹配策略由过滤需求与质量矩阵单独检查。

Sponza 使用同一普通 GPUDriven 图、原始相机／太阳、1920×1080、2048 页、关闭 jitter，三种模式各运行 32 帧。Hard 的稳定需求从 7756 页、准入 2047 页降为 653 页且全部准入，中心粗级查询从 1,195,569 降至零；PCF 和 SMRT 8×16 分别请求 654 / 697 页，均无丢页或中心／过滤粗级回退。检查实际 PNG 后确认 Hard 长廊投影不再出现粗图大块锯齿。Hard 另运行移动相机 64 帧，末帧 595 页全部准入；各静态第 31 帧和移动第 31/63 帧的阴影及 HDR 均与强制重绘逐字节一致，无 RT 能力或分配，Vulkan / synchronization validation 消息为零。这些为短程正确性检查，不代表性能提升计时或长时间显存验收。

选级矩阵共 54 个状态通过，其中 6 个降低虚拟页轴后的 PCF / SMRT 状态覆盖非末级：每状态对照 24,897 个中心查询，均与 Hard 选级一致；方向光／聚光各有 24,897 个非末级像素，点光为 24,385 个。10 项 GPU 回归通过、无跳过：`scene`、`cache`、`local_lights`、`compact`、`filter_demand`、`smrt_distance`、`smrt_penumbra`、`quality`、`remap`、`content`（均带 `virtual_shadow_map_` 前缀）。Shader module 审计通过，`MetallicRHITests`、`MetallicVSMSample` 与 `Metallic` Release 构建通过。

本机证据保留于 `.cache/VSMReceiverLevel/`；`BeforeRegression` 和 `BeforeHard` 保存旧实现失败及画面，`AfterRegression` 保存过滤／SMRT／质量／紧凑输出回归，`FinalRegression` 保存最终 54 状态及固定精度夹具回归；`HardRegression` 中另保留夹具迁移前的失败和已通过的 remap/content。`Afterhard`、`Afterpcf`、`Aftersmrt`、`AfterHardMotion` 保存实景对照。

## 接收面质量门槛（2026-10-29）

`virtual_shadow_map_receiver_quality` 覆盖 30° 斜面、中心 `NdotL=0.1` 的掠射平面，以及半径 3 m、32 段的凸／凹柱面。每种几何测试无遮挡、沿中心光线相距 0.15 m 的近接触遮挡、1.2 m 的分离遮挡，再组合方向光／聚光／点光，得到 36 个状态、108 项 Hard / PCF / SMRT 质量结果。每状态执行 SMRT 冷／热缓存和 PCF，共 108 帧，并与强制重绘逐字节对照。

测试使用 193×129 正交视图，近接触视野高度 0.4 m，其余 1.4 m；相机沿接收面法线相距 2 m，保证最细方向光 clipmap 覆盖接收面。局部光距原点 8 m、半径 0.32 m，方向光角半径 0.04 rad。页尺寸 128、虚拟页轴 128、4 层、第一层半宽 4 m，每面 1024 个物理页；开启 receiver footprint，bias 为 0.002 m。SMRT 使用 32 条射线、每条 32 段、32 m 搜索距离；PCF 半径为 1 texel。

参考由独立的 double 精度 authored-triangle 射线求交产生，BVH 与原始暴力求交在固定子集上交叉验证。先验证接收面的深度和 triangle 身份，再使用 256 个均匀圆盘样本计算面积可见度，不读取阴影 atlas 作为真值。平面遮挡的 50% 边缘使用连续解析投影，避免有限 CPU 样本的 0.5 平台末端造成像素偏移；解析位置在三个截面上分别验证中心光线两侧的 1/0 可见度及圆盘中心的 0.5 可见度。曲面继续使用三角形几何积分参考。

| 门槛 | 要求 |
| --- | --- |
| Hard / SMRT 接收区平均绝对误差 | < 0.04 |
| SMRT 半影平均绝对误差 | < 0.20，至少 20 个半影像素 |
| 50% 阴影边缘位移 | 每行误差的 P95 ≤ 2 px，至少 20 条有效边缘行 |
| 内区漏光与误阴影 | 本影可见度 > 0.1、纯亮可见度 < 0.9 的比例分别 < 1%；有足够样本的 `NdotL` 桶分别执行 |
| 有效覆盖 | 至少 500 个 receiver、100 个纯亮内区；有遮挡的 Hard / SMRT 至少 50 个本影内区 |
| 页面与缓存 | 零丢页、零 raster work overflow、零未解析／全场景粗图查询；冷／热／强制重绘一致 |

PCF 与点光源几何参考比较边缘和稳定内区；其预期的平滑带按核半径与 `NdotL` 排除，不使用面积光源 MAE 门槛。无遮挡状态检查自身误阴影；24 个遮挡状态另要求全亮、全暗和偏移 4 px 的负对照被拒绝。JSON 记录角度桶、中心级别直方图、解析与离散边缘的差值，输出 GPU／CPU／误差 PNG。当前覆盖约 0.1 的掠射及 `(0.35,1]` 内区角度，`(0.15,0.35]` 桶无样本；这不是连续角度扫描。曲面边缘也没有平面解析参考同等的精度保证。

```powershell
.\build-release\tests\MetallicRHITests.exe --gtest_filter=*virtual_shadow_map_receiver_quality* --rhi-bindless --rhi-validation --output-dir .cache/vsm-receiver-quality
```

新门槛暴露了仅按相机世界像素大小选级的问题：在掠射接收面上，光空间 texel 展开后大于屏幕像素。当前使用接收面到光空间 Jacobian 的最小奇异值修正三种过滤共享的 footprint。细化增加页需求后，分配器改为粗层优先，保留覆盖再补细节；18 个配置各重复两次的独立串行 allocator 对照覆盖准入顺序、resident pin 和先空闲后驱逐。

移动实景还暴露了方向光整数页 remap 后的浮点舍入差：先生成随相机移动的 clipmap 坐标、再减页边界，会改变裁剪插值和深度。方向光精细页现在从 snapped page-grid origin 与虚拟页索引构造固定的世界页中心，在顶点点积前合并页面平移，以固定页边界裁剪；现有 projection 字段传递该原点，资源布局不变。局部光和全场景粗图维持原投影。

最终 108 项结果全部通过：Hard / PCF / SMRT 最大接收区 MAE 分别为 0.00909 / 0.01550 / 0.02495；SMRT 最大半影 MAE 为 0.14064，最大边缘 P95 为 1.99370 px，Hard / PCF 最大边缘 P95 均为 1 px。三种模式的内区漏光和误阴影计数、接收面深度错误均为零。使用同一最终测试复测旧选级实现时有 8 项失败，包括 2% 的 Hard 内区漏光和最高 3.99370 px 的 SMRT 边缘位移；没有通过放宽门槛消除这些失败。

`virtual_shadow_map_scroll_precision` 直接回读生产 atlas D32，覆盖半宽 `4/.01 m × 页轴 128/127 × 滚动 1/2/7 页` 共 12 个检查点、28 帧。要求实际 remap/reuse、足够非空深度、准入物理槽不变，缓存／强制重画的深度位模式和阴影字节完全相同。旧 raster 路径失败 8 项、累计 434,203 个深度 texel 位差；固定页坐标后全部为零。该 A/B 保留相同 CPU canonical grid，仅回退 raster 路径。

Sponza 最终使用原普通 GPUDriven 图、1920×1080、2048 页和关闭 jitter，三种过滤各静态 32 帧，Hard / SMRT 8×16 各移动 64 帧。静态 Hard / PCF / SMRT 请求 3016 / 3032 / 3709 页，各准入 2047 个细页及保底粗页；超额细页使用较粗 clipmap，所有静态／移动帧的中心与过滤全场景粗图查询、work overflow 均为零。静态第 31 帧和移动第 31/63 帧的阴影及 HDR 与强制重画逐字节一致。已检查最终截图；无 RT 能力／分配，Vulkan / synchronization validation 零消息。精细页需求增加是质量成本，这些诊断回放不作为性能提升或长期显存验收。

13 项相关 GPU 回归已通过、无跳过：`scene`、`cache`、`local_lights`、`compact`、`filter_demand`、`smrt_distance`、`smrt_penumbra`、`quality`、`receiver_quality`、`allocator`、`remap`、`content`、`scroll_precision`（均加 `virtual_shadow_map_` 前缀）。`MetallicRHITests`、`MetallicVSMSample`、`Metallic` Release 构建及 shader module 审计通过。本机证据在 `.cache/VSMReceiverQuality/`：`FinalBaseline` 保存旧选级失败；`StableRegression` 保存最终质量与场景回归；`BeforeStableScroll` / `StableScroll` 保存页面精度 A/B；`StableSponza*` 保存最终实景回放。早期 `CoreRegression` 的一次 quality 运行被并行源码写入触发的 shader epoch 检查中止，冻结源码后的 `FinalQuality` 和 `StableRegression` 均通过。

## 页请求与过滤统计开销（2026-10-29）

SMRT 在每个 receiver 内为最多 12 个细级保存上次请求的页，跨射线保留，跳过同一级连续重复的全局 OR。选级、射线和采样数、父级请求及最终请求集合不变。Hard 和九点 PCF 保留原短路径，避免引入无收益的缓存初始化和展开成本。

过滤查询数、粗图查询数及未解析查询数先在线程内累计，再由 8×8 工作组汇总，每组每个非零统计槽只做一次全局 Add。点光跨 face 查询仍归属于 receiver 的统计缓冲；槽 30/31/9 的条件及全部 32 槽含义不变。尾组、无效 receiver 和 point face 拒绝从 helper 返回，保证过滤组的所有线程经过两道 barrier；Hard 直接跳过这些同步。不增加 subgroup 能力、资源绑定或 push 布局要求。

RTX 5060、Release、普通 GPUDriven Sponza，原生 1920×1080、固定 LOD 0、2048 页、关闭 jitter，沿用相机、太阳和 10 m SMRT 搜索距离。PCF 半径 1 texel，SMRT 8×16。每种模式优化前后各 3 个独立进程，每次 256 帧，前 32 帧预热；表中为每个进程剩余 224 帧中位数再取三次中位数。使用相同输入哈希，所记录渲染器源码仅 VSM shader 不同。磁盘缓存保留，关闭验证及图像/统计回读；未锁频或独占 GPU。范围是串行预览中的 GPU pass 时间，不包含编辑器呈现，也不代表端到端帧率提升。

| 过滤 | Shadows GPU，之前 → 之后（ms） | 变化 |
| --- | ---: | ---: |
| Hard | 1.070 → 1.075 | +0.5% |
| PCF | 3.842 → 3.840 | −0.1% |
| SMRT | 32.977 → 27.325 | −17.1% |

SMRT 的 `VSM page demand, cache and raster` 从 24.855 降至 19.133 ms（−23.0%）；`Project compact VSM lights` 从 8.186 降至 8.001 ms（−2.3%）。主要收益来自页请求阶段，不能把逻辑原子次数的减少等同于投影时间同比下降。SMRT 三次 Shadows 中位数由 33.338 / 32.977 / 32.389 ms 变为 27.458 / 27.214 / 27.325 ms。

新增 `virtual_shadow_map_demand_equivalence`：三类光源 × 三种过滤 × 冷/热/强制重画，另含点光 12 级、32×64 采样的三个状态，共 30 状态。129×97 尾组、至少三个点光 receiver face、全部 12 级实际请求及提前终止均有覆盖断言。报告保留完整请求页 ID、32 槽统计和阴影字节。`filter_demand` 的 54 状态增加逐 face PCF 精确九次查询、SMRT 查询范围和零尺寸退化断言，以及阴影/debug/depth 哈希。使用同一测试执行旧/新 shader，84 状态的全部证据字段完全一致；现有点光精确面边界的域外计数也保持一致。

上述两项及原 12 项相关回归全部通过，无跳过，Vulkan/synchronization validation 零消息；108 项斜面/掠射/曲面质量门槛未改动且全部通过。最大采样用例额外验证了实际请求覆盖 0–11 级。三种静态 Sponza 各 32 帧，Hard/SMRT 移动各 64 帧；与上一节旧 shader 的全部逐帧统计和截图像素哈希相同，检查点的阴影与深度哈希相同。每个回放的缓存/强制重画阴影和 HDR 逐字节一致，无 RT 能力或分配。静态请求数仍为 3016 / 3032 / 3709，细页超额时仍使用较粗 clipmap，未通过缩小请求集合获得性能收益。

`MetallicRHITests`、`MetallicVSMSample`、`Metallic` 构建及 shader module 审计通过。本机证据位于 `.cache/VSMDemandCost/`：`Before-*` / `Final-*` 为计时与源码/输入/二进制身份记录；`TimingSummary.json` 为进程中位数；`BeforeEquivalence` / `FinalRegression`、`EquivalenceSummary.json` 为精确 A/B；`FinalSponza-*`、`SponzaSummary.json` 为实景回放。中间候选保留在其他目录，不进入最终计时汇总。

```powershell
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_filter_demand:*virtual_shadow_map_demand_equivalence' --rhi-bindless --rhi-validation --output-dir .cache/vsm-demand-check
python -B Tools/Perf/VirtualShadowMapAcceptance.py run --cases .cache/VSMDemandCost/smrtTiming.json --exe build-release/tests/MetallicRHITests.exe --output .cache/vsm-demand-repeat --runs 3
```
## 低射线时域与搜索范围（2026-10-29）

多灯 pass 新增独立可见度历史。每盏稳定 ID 的灯只有一份屏幕空间历史，点光六面完成后只解析一次，支持 compact 和 legacy 输出。使用实际光栅相机和 jitter 重投影，四个历史 tap 分别经过深度重建法线与双向平面距离检查；显露或无效区域使用当前原始结果。历史保存半精度可见度、累积次数和原始 receiver depth，达到帧数上限后继续以 `1/maxFrames` 更新。不会用单条射线的二值邻域范围裁掉有效半影。

普通相机移动、采样相位变化和 Force Redraw 保留历史；camera cut、断帧、分辨率/投影、相关设置、灯光几何和 caster 内容变化使历史失效。灯光排序变化由每灯签名处理，纯灯光失效原因不再推进几何重投影版本，原有辐射度历史失效版本保持原行为。相机属性从光栅设置版本的比较中排除，实际相机仍单独发布和检查。取消提交不提升历史，同帧重新执行使用相同前驱，不重复累积。

历史使用三个受完成信号保护的 RG32Uint 纹理槽，不建立逐帧引用链。关闭时域、切出 SMRT、零光源尺寸、禁用或预算拒绝的灯会释放其历史所有权。`Shadows.memory` 的第 6 槽（从零计）报告历史池实际纹理分配字节；1080p 单灯逻辑容量为 47.46 MiB，本机 Vulkan 实际分配为 50.625 MiB。当前没有 caster motion vector，任意 caster 内容变化会保守清空所有灯的历史，持续动画不能依靠这份历史保持低射线收敛。

`smrtRayLengthScale > 0` 时，方向光和局部光共同采用 `min(smrtMaxRayLength, smrtRayLengthScale * receiverCameraDistance)`；透视相机使用欧氏距离，正交相机使用轴向线性深度，使平移不改变搜索范围。局部光再应用 `smrtMaxRayAngle` 和原有光源距离约束。标页和过滤调用同一个长度函数。缩短搜索限制半影展开，不移除中心深度比较或搜索范围以外的本影；默认比例 0 与角度 0.03 保持旧搜索配置。

新增 `virtual_shadow_map_smrt_temporal` 使用 193×129 透视视图、256 样本独立 CPU 几何参考，方向光/聚光/点光各覆盖 1/2 条射线。每例包含 32 帧静态、16 帧相机移动和 12 帧失效控制。Cached、Force Redraw 和原始相位对照在同帧的独立节点执行，缓存/重画可见度逐字节一致。结果如下（可见度范围 0–1）：

| 光源 / 射线 | 静态半影 MAE | 静态帧差 RMS / 原始 RMS | 移动半影 MAE |
| --- | ---: | ---: | ---: |
| 方向光 / 1 | 0.03813 | 0.01879 / 0.71875 | 0.02545 |
| 方向光 / 2 | 0.02886 | 0.01048 / 0.35614 | 0.02082 |
| 聚光 / 1 | 0.05498 | 0.01749 / 0.66166 | 0.04331 |
| 聚光 / 2 | 0.04835 | 0.01018 / 0.34363 | 0.04117 |
| 点光 / 1 | 0.05593 | 0.01775 / 0.66737 | 0.04570 |
| 点光 / 2 | 0.05041 | 0.01044 / 0.35045 | 0.04340 |

六例各 480 个真实显露像素均在 1/255 内匹配当前原始值；点光双灯场景额外执行 33 帧，145 个 seam 半影样本中有 126 个形成软阴影。两个 source slot/base layer 实际交换后，5922 个已累积软像素无异常跳变，compact/legacy 全图一致。质量门槛为静态 receiver MAE ≤0.04、半影 MAE ≤0.14、半影误差至少降低 35%、帧差 RMS ≤原始 35%；移动半影 MAE ≤0.16、重投影帧差 RMS ≤原始 60%，全部通过。

`virtual_shadow_map_smrt_search` 的 17 个状态通过：三类灯的视距比例/固定米制控制结果一致，世界上限生效；方向光短搜索请求 42 页，对照 106 页，半影没有越过独立几何上界；0.1 m 搜索仍保留 12 m 外遮挡物的本影。局部光角度范围的需求和重画对照也通过。`virtual_shadow_map_temporal_lifecycle` 覆盖同帧重复、未提交录制、取消、断帧、cut、设置/内容失效、尺寸变化、输出层变化、点光 canonical face 和资源池平台期。`history_invalidation_scopes` 覆盖灯光/几何/材质/显式失效及原版本合同。

原有 14 项 VSM 回归和 6 项历史/灯光合同检查全部通过，无跳过；新增时域质量、搜索及生命周期共 3 项通过。原有 108 项斜面、掠射和曲面质量门槛未改动且全部通过。Release 编辑器、Sample、RHI 测试构建成功，Slang 审计通过（387 files / 275 modules / 112 implementation units）。复用 `build-release`，仅将原先关闭的 `METALLIC_BUILD_TESTS` 开启，未更换编译器或 SDK。

Sponza 使用普通 GPUDriven、原生 1920×1080、2048 页、关闭 jitter：1/2 rays ×16 各静态 96 帧，2 rays 移动 96 帧；另有相同 2 rays、历史上限 1 的 96 帧控制。Vulkan/synchronization validation、未解析查询、全场景粗图查询和 raster work overflow 均为零，无 RT 能力或分配。已检查最终静态/移动截图。历史分配从一槽增长到三槽后固定为 50.625 MiB；这些短回放不是长期显存压力验收。

2-ray 静态控制与 32 帧历史的全部 96 帧页需求/过滤统计相同。最终 sRGB 图像第 63/95 帧的 RGB 差 RMS 从 0.00471666 降至 0.00029788（两张相隔 32 帧的图像，不是逐帧运动质量指标）。旧 acceptance 的 `comparisonFrames` 会依次渲染不同帧，不适用于开启相位轮转后要求缓存/重画逐字节相同；该合同由上述同帧三节点测试验证。

RTX 5060 的耗时控制固定同一 Sponza、2×16 采样和逐帧相位，只改变历史上限 1/32。每配置 3 个独立进程 ×256 帧，前 32 帧预热，关闭验证和回读、保留磁盘缓存，未锁频或独占 GPU。每进程后 224 帧中位数再取三次中位数：`Stabilize SMRT shadows` 为 0.132 → 0.848 ms，完整 Shadows pass 为 15.137 → 15.840 ms。前者包含当前帧写入和历史解析，增加约 0.716 ms；不是端到端帧率数据，也未把减少射线数计入时域收益。

证据位于本机 `.cache/VSMTemporal/`：`Quality2`、`Quality1/VirtualShadowMapSMRTSearch`、`Regression`、`Sponza*Run`、`Timing2*Run`、`TimingSummary.json` 和 `SponzaStability.json`。`Quality1` 保留首次发现灯光重排全局失效的失败，修复后重排检查在 `Quality2` 通过；所有最终实景/耗时回放的源码、输入和二进制完整性检查通过。

```powershell
.\build-release\tests\MetallicRHITests.exe '--gtest_filter=*virtual_shadow_map_smrt_temporal:*virtual_shadow_map_smrt_search:*virtual_shadow_map_temporal_lifecycle:*history_invalidation_scopes' --rhi-bindless --rhi-validation --output-dir .cache/vsm-temporal-check
```
