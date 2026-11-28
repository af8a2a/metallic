# Ancient Rift Valley

1 km × 1 km 的原创干旱峡谷地形。S 形主谷串接两条支谷，谷地两侧有台地、砂岩陡壁、浅色沉积层和碎石坡。高度约为 20.35–413.06 m，高差约 392.71 m。视觉方向参考 Epic 的 Moab 干旱峡谷场景。 [Valley of the Ancient](https://dev.epicgames.com/documentation/en-us/unreal-engine/valley-of-the-ancient-sample-game-for-unreal-engine)

## Houdini 工程

在 Houdini 22.0.368 打开 `AncientRiftValley.hipnc`。`/obj/AncientRiftValley` 的 **Terrain Art Direction** 提供高差、谷宽、侵蚀沟细节和层间对比控件。`OUT_LAYERS` 输出高度与八个命名遮罩。视口可通过 `Material_Weights_Preview` 检查分层，再选择 `Canyon_Entrance` 或 `Aerial_Overview` 观察地形。

`Materials/TerrainMaterialLibrary.hip` 含八个可编辑的 MaterialX 材质。每层有 1024² 的 sRGB 底色、OpenGL +Y 切线法线和线性粗糙度。贴图按 4–12 m 的世界尺度平铺，材质和贴图随工程目录一起移动。

| EXR | R | G | B | A |
|---|---|---|---|---|
| `Maps/Weights_0.exr` | 峡谷沙 | 冲积砾石 | 枯草 | 湿泥沙 |
| `Maps/Weights_1.exr` | 红砂岩 | 浅色盖层 | 崩积碎石 | 荒漠灌木 |

坡度、高程、沟道与局部凹凸决定各层贡献。西侧支谷的冲积扇上留有一块八层同时参与的区域，便于后续检查多层求值。

`Preview/CanyonVista.png` 和 `Preview/WorldOverview.png` 使用 Blender Cycles 渲染实际导出的 2049² 高度、权重及全部八套 PBR 贴图。权重使用全局 XZ，砂岩和浅色盖层细节在预览中使用世界坐标三平面投影，以减轻陡壁的拉伸。鸟瞰中的竖向剖面和底板用于展示覆盖范围。

`Preview/EditablePreview.blend` 可用于检查八层混合节点，使用 257² 代理网格。移动资产时一起保留 `Preview`、`Maps` 和 `Materials` 目录，贴图以相对路径加载。

## 导出与分区

`Maps/Height.exr` 使用 `Y` 通道的 32 位浮点米制高度，缩放为 1、偏移为 0。两张权重图均为线性半精度 RGBA。三张 EXR 都是 2049²，行从世界 Z −500 m 走向 +500 m，列从世界 X −500 m 走向 +500 m。`flipZ` 为 `false`。每格约 0.488 m。

`TerrainAsset.json` 记录材质路径、通道、平铺尺度和 4×4 区域切分窗口。每区 250 m × 250 m，使用含端点的 513² 个样本。可从全局 EXR 按窗口裁切；全部 24 条共享边的高度和八层权重已逐样本比对。`Verification.json` 包含工程重开复算、EXR 和贴图检查结果。

Metallic 的 **Ancient Rift Valley 4x4 + VSM** 样例使用完整 16 区、两张权重图和八层 OpenPBR 材质。红砂岩和浅色盖层的 `triplanar` 设置随材质导入，中央 2×2 样例保留 XZ 基线。
