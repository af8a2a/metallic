"""Generate a local index from actual build and validation receipts."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    load=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    manifest=load('Manifest.json')
    exported=load('ExportValidation.json')
    reloaded=load('ReloadValidation.json')
    expected={(c['id'],m) for c in manifest['cases'] for m in c['modes']}
    assert {(p['case'],p['mode']) for p in exported['projects']}==expected
    assert {(p['case'],p['mode']) for p in reloaded['projects']}==expected
    lines=['# Painter 材质验证包', '',
           f'已保存 {len(expected)} 个 Painter 12.1.4 原生 OpenPBR 项目，包含 M01–M08、六组 5×5 参数扫描及 H01 HDR 发光补充。', '',
           '**本报告验收的是 Painter 资产制作、保存和导出链路。Metallic 渲染一致性尚未测试。**', '',
           '## 数值与重载检查', '',
           f'- EXR 检查通过：{exported["passed"]}；文件 {exported["files"]:,} 张，均检查 NaN / Inf。',
           f'- 常量检查 {exported["constantChecks"]:,} 项，最大误差 {exported["maxConstantError"]:.9g}。',
           f'- 纹理中心 UV 区域检查 {exported["textureChecks"]} 项，最大误差 {exported["maxTextureError"]:.9g}。',
           f'- SPP 实际重载通过：{reloaded["passed"] and reloaded["complete"]}；{len(reloaded["projects"])} 个项目，校验 shader、参数、内嵌纹理和色彩空间。',
           '- 色彩通道 ACEScg；数值通道 raw；OpenGL +Y 法线；512×512、32f EXR。', '',
           '[数值报告](ExportValidation.json) · [重载报告](ReloadValidation.json) · [参数总表](Manifest.json)', '',
           '## 工程入口', '',
           '打开 SPP 后，每个 Texture Set 对应一个参数变体。单行用例从左到右排列；扫描网格从上到下为 Row0–Row4，从左到右为 Col0–Col4。具体参数见各组 Reference.json。', '',
           '| 用例 | 模式 / SPP | 参数 | 实际窗口截图 |', '|---|---|---|---|']
    for case in manifest['cases']:
        for mode in case['modes']:
            folder=f'{case["id"]}/{mode}'
            spp=f'{folder}/source/{case["id"]}_{mode}.spp'
            image=f'{folder}/reference/PainterViewportWindow.png'
            assert (root/spp).is_file() and (root/image).is_file()
            references=f'[Viewport]({image})'
            if (root/f'{folder}/reference/PainterIrayWindow.png').is_file():
                references+=f' / [Iray]({folder}/reference/PainterIrayWindow.png)'
            lines.append(f'| {case["id"]} | [{mode}]({spp}) | [Reference]({case["id"]}/Reference.json) / [实际回读]({folder}/AuthoringReceipt.json) | {references} |')
    lines+=['', '## 对照前必须明确的事项', '',
        '- M06 玻璃的 10 / 50 mm 是输入网格厚度。Iray 的独立场景尺度和绝对吸收距离未在本自动化中标定，不能据此验收 Beer–Lambert。',
        '- Iray 截图需人工检查实际采样数；其余主要是实时 viewport 参考。所有 PNG 都经过显示变换，不是线性渲染 EXR。',
        '- M07 保留 1 / 10 / 100 nits；H01 增加 1,000 / 10,000 / 100,000 nits。导出 emission color 与 emission luminance 是分离的，亮度在 shader 参数中。',
        '- 未烘焙 AO，预设请求 Mixed AO 时可能给出警告；本验证关闭 AO，所需材质通道均单独检查。',
        '- GeometryOnly.gltf 只含几何和参数 extras；不能直接加载它就认为 Metallic 已使用完整 OpenPBR 材质。',
        '- 待完成：Metallic 的参数/纹理接入、相机灯光和工作空间对齐、PT/VBuffer 对照、MASK 阴影一致性、玻璃尺度校准，以及线性 HDR/能量测试。', '',
        '## 文件用途', '',
        '- source/*.spp：可编辑工程；textures/*.exr：实际 Painter 导出；reference/*.png：真实窗口截图。',
        '- AuthoringReceipt.json：请求值、实际回读、shader 设置和版本绑定。BuildReceipt.json：SPP 哈希。',
        '- PreparationChecks.json 是最初生成时的历史快照；若追加过用例，另见其 SupplementChecks.json。Manifest.json 是当前完整清单。',
        '- NeutralStudio.exr 为数值合成的中性环境光，不是拍摄的 HDRI；用于排除彩色环境造成的误判。', '',
        '参考：[Adobe Iray 设置](https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/iray-renderer/iray-settings)。场景大小和采样限制的判断来自本包保存的实际界面。', '']
    if (root/'VisualReview.json').exists():
        review=load('VisualReview.json')
        lines+=['## 本次人工检查记录', '',f'记录日期：{review["date"]}。', '']
        lines+=['- '+note for note in review['notes']]+['']
    (root/'Report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(root/'Report.md')


if __name__=='__main__':
    main()
