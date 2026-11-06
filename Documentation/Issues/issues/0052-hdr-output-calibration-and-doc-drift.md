---
id: "0052"
title: "Output: HDR 输出缺显示器峰值检测与独立窗口 HDR，且 HdrOutput.md 与已实现的 HDR10 路径不一致"
area: material
kind: task
priority: P2
state: open
milestone: M3-Hardening
labels: ["area:material", "kind:task", "priority:P2"]
depends_on: []
evidence:
  - Documentation/HdrOutput.md:27
  - Documentation/DisplayOutput.md:18-22
  - Documentation/DisplayOutput.md:31-35
  - Documentation/DisplayOutput.md:45-47
  - Documentation/ACEScgMigrationValidation.md:108-111
---

## 背景 / Context

输出边界是 `ColorGradingLUTPass → FinalBlit → ImGui 合成 → 交换链`。工作空间为
ACEScg 后，显示编码（sRGB / scRGB / HDR10-PQ）必须保持固定含义，且 HDR 的绝对亮度
标定不能依赖猜测。

## 现状 / Current state

- `DisplayOutput.md` 描述了三个 profile：`SDR_sRGB`、`HDR_scRGB`（RGBA16F +
  `EXTENDED_SRGB_LINEAR_EXT`）、`HDR10_PQ`（A2B10G10R10 + `HDR10_ST2084_EXT`），
  并在可用时通过 `VK_EXT_hdr_metadata` 写入 BT.2020/D65 mastering primaries 与所选
  峰值（`DisplayOutput.md:18-22`、`:45-47`）。
- 但 `HdrOutput.md` 的能力声明仍是旧的：「HDR10/PQ、HDR metadata、自动显示器峰值
  检测与独立窗口 HDR 尚未实现」（`HdrOutput.md:27`）。其中 HDR10/PQ 与 metadata
  已在 `DisplayOutput.md` 与代码侧实现，两份文档对同一能力给出相反结论。
- 两个真实缺口在两个文档里是一致的：
  1. **无自动显示器峰值检测/标定。** 峰值是手填的 mastering target（默认 1000 nits），
     「Set peak to the intended mastering target (for example 600 or 1000 nits); this is
     not automatic display calibration」（`DisplayOutput.md:31-35`）。
  2. **独立窗口仍是 SDR。** 「Detached ImGui windows currently use SDR previews」
     （`DisplayOutput.md:47`）；scRGB 视口在其中按纸白转换并裁切高光。
- 硬件验收仍是空白：「Physical HDR monitor calibration, display-profile switching and
  extended interactive viewing were not validated by these checks」
  （`ACEScgMigrationValidation.md:108-111`）。

## 问题或缺口 / Problem

1. **文档冲突会直接导致错误的实机标定。** 按 `HdrOutput.md` 行事的人会认为 HDR10 与
   metadata 不存在，从而只测 scRGB；按 `DisplayOutput.md` 行事的人会以为 HDR10 已经
   完整验收。
2. **没有显示器能力查询。** 峰值亮度、纸白对齐（跟随 Windows SDR 白电平）都依赖用户
   手填；在多显示器/不同 HDR 硬件之间切换时没有任何自动或半自动路径，也没有
   「当前设置超出显示器能力」的告警。
3. **独立窗口不支持 HDR。** 把 LookDev 比较视图拖出主窗口后，scRGB 视口会被按纸白
   裁切，跨窗口比较的亮度语义不一致。

## 期望结果 / Expected outcome

- 一份权威的 HDR 输出能力表（profile × 已实现 × 已验证 × 硬件依赖），
  `HdrOutput.md` 与 `DisplayOutput.md` 不再冲突，二者之一链接到另一份。
- 至少半自动的显示标定：能读取（或引导用户输入）显示器峰值/纸白，并在超出能力时告警。
- 独立窗口 HDR 有明确结论（实现，或写明不支持并给出替代方案）。

## 验收标准 / Acceptance criteria

- [ ] 统一两份文档：删除 `HdrOutput.md` 中已不成立的「HDR10/PQ 与 metadata 尚未实现」，
      改为链接 `DisplayOutput.md`；两份文档的 profile 表逐项一致。
- [ ] 新增显示能力查询/标定入口：至少能把当前协商到的格式/色彩空间、纸白、峰值写入
      日志与 UI；超范围设置给出可见告警。
- [ ] 在真实 HDR 硬件上完成一次端到端记录：`METALLIC_OUTPUT_PROFILE` 三种 profile 依次
      切换，记录协商结果与校准图读数（80/203/400/1000 nit 色块），并与
      `hdr-calibration` 样例的期望值对照。
- [ ] 独立窗口：要么实现 HDR 交换链（含 scRGB 语义），要么在文档与 UI 明确标注 SDR-only
      并说明裁切行为。
- [ ] `ACEScgMigrationValidation.md` 的 HDR 章节更新「已由哪次运行覆盖」。

## 相关文档 / References

- [HdrOutput.md](../../HdrOutput.md)
- [DisplayOutput.md](../../DisplayOutput.md)
- [ACEScgMigrationValidation.md](../../ACEScgMigrationValidation.md)
- [AutoExposure.md](../../AutoExposure.md)
