# M7 — Native Strand Rendering

对应外部路线图 **M7 / Phase 16**。提供原生折线、变半径、有限多层 coverage 的小型 groom 后端，独立生成 Strand Visibility，再由共享 Fiber Material Program / Closure / PreparedClosure 着色。无需将曲线转成 DOTS 三角形，也不依赖 RTXCR SDK 的几何导入组件。

## 使用

在编辑器的样本选择器中选择 **Material LookDev → Native Strand Groom**，或运行：

```powershell
.\build-scheduling-release\Source\LookDev.exe --sample native-strands
```

图为 `Pipelines/Samples/native_strands.metallic_graph.json`。示例 `Asset/Strands/NativeGroom.strands.json` 包含 72 根、576 段曲线，显式选择 8 层并引用既有 `ChestnutFiber.material`。共享材质定义的 `RTXCRChiang.DOTS` 名称为兼容旧资产保留；原生 strand 消费相同 Program，不经过 DOTS 几何路径。

`Strands` 的运行时选项控制每像素层数、形变 phase / amplitude 和密度 LOD；`Fiber` 控制灯强、strand 阴影及溢出策略。视口相机由共享 RenderView 驱动，支持透视、正交与 camera cut。未绑定 RenderView 的图预览使用节点 camera。

## 资产与生命周期

JSON version 1 包含 `materialRoot`、`materials` URI 数组和 `strands`。每根 strand 有唯一 `id`、`material` 索引、可选 `opacity` / 稳定 `normal`，以及至少两个 `points`。每个点包含 `position`、正值 `radius`，可选 `previousPosition` 默认等于当前位置。材质根相对于曲线文件解析。

- IDs 保留完整 uint32 精度，`UINT32_MAX` 留作无命中标记；文件顺序不改变 identity。segment ID 为原始点索引，root-to-tip 参数按加载时的弧长归一化。
- 加载拒绝重复或非整数 ID、无效材质索引、退化段、非有限/溢出的长度与法线。失败不发布候选资产。
- `MaterialAssetLibrary` 解析 Fiber Definition / Instance、继承参数，再构建不可变 MaterialGeneration；Surface 资产不可用于本后端。
- 图重新准备时重新加载资产及材质；运行时参数不重新编译 Program。当前没有曲线文件监听或骨骼 groom 模拟。内置 root-pinned 形变使用固定控制点参数，并保留上一帧相机、phase 与 amplitude。
- 上传缓冲由图持有；pass 不启用 frame overlap / pipelined submission。录制取消、几何改变、resize 和 camera cut 会使 motion history 无效。

## 可见性与合成契约

| 项目 | 契约 |
| --- | --- |
| 容量 | 最多 4096 段、64 个材质，每像素 K=1…8，默认 4 |
| 记录 | 每条 16 字节：segment index、局部 u、coverage、正向 view Z |
| 内存 | `width × height × K × 16`；记录超过 512 MiB 时准备失败并给出诊断，不静默降质 |
| 轮廓 | 裁剪后的变半径 projected capsule，加一像素 box coverage 近似；接缝以最大 coverage 合并 |
| 接缝 | 相邻且深度接近的段先归并，再做 K 截断；分离的自重叠仍可成为独立层 |
| 排序 | 正向 view Z 从近到远，同深度按稳定 strand / segment ID 排序 |
| 合成 | `C += T × coverage × radiance; T *= 1 - coverage`，最后叠加 `T × opaqueColor` |
| 溢出 | 默认颜色标为洋红；可选 nearest-K，计数仍保留，不静默丢弃 |
| LOD | 基于稳定 strand ID 的密度排序和连续淡入；不重编号，不使用逐帧随机覆盖 |

本阶段是解析屏幕足迹近似，深度使用中心线，不是精确圆柱求交。像素覆盖内的横截面 h 用四点确定性求积，再直接构建 RTXCR/Fiber interaction；不额外乘 Surface `NdotL`。这不意味着极端低粗糙度已达到离线积分精度。

## 图接口与消费者

- `Strands.counts`：RGBA32Uint，retained / overflow / candidate count / historyValid。candidate count 在 K 截断前统计。
- `Fiber.color`：RGBA32Float，场景线性合成色，alpha 为累计 strand coverage。
- `Fiber.identity`：RGBA32Uint，最前层 strand ID / segment ID / root-to-tip float bits / coverage float bits；空像素前两分量为 `UINT32_MAX`。
- `Fiber.motion`：RGBA32Float，xy 为 **previousPixel − currentPixel**，单位为 render pixel；zw 为 root-to-tip 参数和正向 view Z。无有效历史时 xy 为零。多层的重投影需要同时检查 identity、historyValid 与遮挡；本阶段未直接接入 DLSS / NRD / 时域重建。
- 可选 `Strands.opaqueDepth` 必须是同相机、同尺寸的 **正向线性 view Z**，不可直接连接 Vulkan device depth / reversed-Z；未命中用远平面距离。
- 可选 `Fiber.opaqueColor` 为同尺寸、场景线性 RGB。没有连接时使用 0.025 背景。
- 整数诊断输出支持预览及精确 raw readback；8-bit 可视预览仅做 clamp，不能用来恢复身份数据。

照明为固定方向主光和补光。主光支持按 strand opacity 衰减的解析 strand 自阴影；不是 deep opacity map，未接入网格阴影、HDRI、多次散射或全局 Fiber 输运。扫描复杂度随像素数和段数增长，当前定位为小型 groom 验证后端；大规模 groom 的 tile binning、GPU 曲线导入/动画、LSS / secondary ray 是后续工作，不以本次结果声明其性能或正确性。

## 验收与证据

Windows Release，RTX 5070 Ti，Vulkan validation 开启。复用 `build-scheduling-release`：

```powershell
cmake --build build-scheduling-release --target Metallic LookDev MetallicSceneTests MetallicRHITests
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter=StrandAssets.*:MaterialAssets.*
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.native_strand_visibility:RHIRendering.material_fiber_native_and_stages:RHIRendering.material_fiber_asset_rendering --output-dir build/strand-m7-final
```

CPU 覆盖资产拒绝/事务发布、样本排序及共享材质资产。GPU 使用真实生产 pass，覆盖 129×97 非整组分辨率、多层 alpha、32 位 ID、静态逐位一致、形变/相机解析运动、camera cut、近裁剪、正交投影、亚像素平移覆盖守恒、LOD、接缝归并、overflow、opaque 背景/遮挡、strand 阴影及材质实例重载；并保存 512×512 groom 图。既有 M6 上游数值 oracle 与 Claire 资产场景回归验证 DOTS 消费未被新 h 有效位改变。

2026-10-05 最终结果：四个目标构建通过，CPU **22/22**、GPU **3/3** 通过，日志中无 Vulkan validation error；LookDev 的样本列表已验证包含 `native-strands`。实际查看了 512×512 groom 输出，53,094 个覆盖像素，最多 8 个候选层，溢出像素 **0**。4 层不足以覆盖该样本最密的交叠，故图显式选择 8 层。测试另外验证 K=1/2 的计数、洋红诊断和 nearest-K 退化策略。

`NativeStrandMetrics.json` 保存容量、覆盖数及响应指标：受控阴影用例中心 RGB 总衰减约 0.06569，melanin 重载 RGB 总变化约 0.006389，coverage 保持相同。这些是功能断言，不是性能指标。静态输出逐位重复，细 strand 亚像素平移的总 coverage 相对偏差要求小于 8%；不据此声称所有视角和材质的时域无闪烁。

生成的日志、报告和截图保存在 `build/strand-m7-final/` 及相邻 `strand-m7-*.log`，不加入源码控制。交互式操作和长时间动画运行不由这些有界 GPU 测试代替。
