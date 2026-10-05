# Painter 材质验证资产

通过 Adobe Substance 3D Painter 官方远程 Python API 创建真实 `.spp`，使用公开的 JavaScript shader API 分配独立 OpenPBR Shader Instance。当前参数配置绑定 Painter **12.1.4**，不会把 ASM 或 glTF sheen 当成 OpenPBR。

本机已完成的资产和验收结果见 [验证包报告](../../build/MaterialValidation/PainterValidation/Report.md)。生成的工程、贴图和截图保留在 `build/MaterialValidation/`，不提交到源码库。

## 覆盖范围

| 组 | 用例 |
|---|---|
| M01 | 中性介电体，roughness 0.05 / 0.2 / 0.5 / 0.8 |
| M02 | 有色金属，roughness 0.1 / 0.35 / 0.7，金属度分区贴图 |
| M03 | 红色底层 + coat 关闭 / roughness 0.03 / 0.25 |
| M04 | 各向异性 0.8，切线 0° / 45° / 90°，方向贴图 |
| M05 | Fuzz 0 / 0.5 / 1，fuzz 权重贴图 |
| M06 | 闭合玻璃，网格厚度 10 / 50 mm、roughness 0 / 0.15 |
| M07 | 发光 1 / 10 / 100 nits，彩色发光棋盘 |
| M08 | 双面卡片，MASK 阈值 0.5，含 0.499 / 0.501 分区和 OpenGL 法线 |
| H01 | HDR 补充：1,000 / 10,000 / 100,000 nits |
| S01–S06 | 六组 5×5 扫描：金属度、IOR、coat、各向异性、透射、fuzz |

M/H 组各含 uniform 和 textured 两个项目；S 组各一个，共 **24 个项目**。JSON 保存每个材质的参数真值、通道映射和实际回读，不能只传递 EXR 而丢失 shader uniform。

## 重建

外部 Python 需要 `numpy` 和 `OpenEXR`；只有几何/PNG 准备步骤使用标准库。Painter 自带 Python 运行 `PainterAPI.py`，无需向 Painter 安装这些外部依赖。Painter 必须已激活，使用 `--enable-remote-scripting` 启动，并且没有未保存项目。脚本连接官方 `localhost:60041/run.json`。

从仓库根目录执行；输出目录必须全新：

```powershell
python Tools/MaterialValidation/PreparePainter.py --output build/MaterialValidation/NewPainterReference
python Tools/MaterialValidation/ConfigurePainter.py --root build/MaterialValidation/NewPainterReference --painter-root 'E:/SubstancePainter/Adobe Substance 3D Painter'
python Tools/MaterialValidation/BuildPainterSuite.py --root build/MaterialValidation/NewPainterReference --iray-glass
python Tools/MaterialValidation/ValidatePainterExports.py --root build/MaterialValidation/NewPainterReference
python Tools/MaterialValidation/VerifyPainterReload.py --root build/MaterialValidation/NewPainterReference
python Tools/MaterialValidation/GenerateReport.py --root build/MaterialValidation/NewPainterReference
```

`--only M03_CoatedPaint` 可以限制创建范围。已完成项目通过 BuildReceipt 跳过；失败项目保留现场，不自动叠加重试。`AddHDRSupplement.py` 只用于给较早的 22 项目包追加 H01，新包已经包含它。

`Painter12.1.4Profile.json` 保存本次确认的映射；换 Painter 版本时必须重新检查官方 SDK、shader 参数和导出结果，不能直接改版本号绕过检查。ConfigurePainter 会绑定本机模板/OCIO 路径并生成中性 HDR 环境。

## 检查边界

- `ValidatePainterExports.py` 检查每张 EXR 的有限值、SPP 哈希、常量数值，以及纹理中心 UV 区域与原始 RGB16 PNG 的一致性；常量容差 2e-5，纹理容差 1e-4。边缘 padding、GPU mip/过滤和压缩误差不属于这些检查。
- `VerifyPainterReload.py` 实际重新打开每个 SPP，检查 shader 定义、通道常量、贴图色彩空间，以及内嵌贴图和环境可访问性。验证期间关闭当前项目，避免用户同时编辑。
- 颜色使用 ACEScg，数据通道使用 raw，法线使用 OpenGL +Y；EXR 为 32 位浮点。GeometryOnly.gltf 只供几何复用，其 extras 不会自动成为 Metallic 的材质定义。
- UI PNG 来自真实 Painter 窗口，已经经过显示变换；它不是线性辐射亮度 EXR。实际 Iray 截图的采样数和场景尺度必须人工复核，不能把截图当成收敛的定量真值。
- M06 的输入网格厚度已经检查；Iray 的场景归一化尺度与 transmissionDepth 的绝对长度尚未标定。本次玻璃截图仅可作定性参考，不能用于 Beer–Lambert 数值验收。S05 的实时预览同样不能验证体积传输。
- Painter 的发光 shader 使用 emissionLuminance / 1000；H01 覆盖大于 1 的 shader 输入响应，但屏幕截图不能验证线性 HDR 输出是否夹取。
- Painter 导出检查本身不验证引擎渲染。Metallic 场景执行检查见 [LookDev 验收记录](../../Documentation/PainterLookDevValidation.md)；它不代表阴影 MASK、能量守恒或随机采样收敛已经通过。

## 下一步对照

已有 HDRI + 实拍色卡可使用 `BuildStudioLookDev.py` 构建独立灯光版本，
具体参数、照片测量与反射率的区别见 [White Studio LookDev](../../Documentation/WhiteStudioLookDev.md)。

Metallic 整组场景导入入口为 `BuildLookDevScenes.py --root <Painter验证包>`。
默认输出 `build/MaterialValidation/PainterLookDev`，重启 LookDev 后在
**RenderGraph → Graph Editor Settings → Painter Material Scene** 下拉框选择。详见
[LookDev 场景选项说明](../../Documentation/OpenPbrLookDev.md)。
导入包保留明确色彩标签和独立参数/资源文件，不再直接使用 GeometryOnly.gltf
中的占位材质；预览纹理当前量化为 RGBA8，原始 EXR 保持完整。

1. 从 Reference.json、AuthoringReceipt.json 和 EXR 共同构建 Metallic Material Definition / Instance，明确 ACEScg 到引擎工作空间的转换。不要将 ACEScg 贴图标记为 sRGB。
2. 对齐几何、切线、相机、环境、曝光和显示变换；先检查 M01/M02，再检查 coat、各向异性和 fuzz。
3. 校准玻璃绝对场景尺度后，用 PT 与充分采样的线性参考检查厚度吸收；无需要求不支持体积传输的路径给出相同结果。
4. 单独检查 M08 在 raster / RT / shadow 的覆盖一致性，以及 H01 的线性高亮值。保存逐项结果后再作材质系统正确性结论。

参考：[官方远程控制 API](https://experienceleague.adobe.com/en/docs/substance-3d-dev/painter-python/tutorials/remote-control)、[官方 Iray 设置与场景尺度](https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/iray-renderer/iray-settings)。
