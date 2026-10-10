# LookDev color grading

**ACES 2.0 is the default display transform.** Select the **ColorGrading** node
(ColorGradingLUTPass) to edit transforms, global grading and custom LUTs.
FinalBlit samples its output; it no longer evaluates ACES per screen pixel.
All shipped scene-linear pipelines connect this node. Diagnostic sRGB and HDR
calibration graphs retain their direct display path.

    Lighting -> AutoExposure -> FinalBlit.source -> editor composition -> swapchain
    ColorGradingLUTPass.lut --> FinalBlit.lut

The renderer stays scene-linear working RGB. Windows HDR still defaults to scRGB;
HDR10 encodes BT.2020/PQ after FP16 UI composition. See [display profiles](DisplayOutput.md).

## RenderGraph contract

- The LUT is a native **Texture3D, 64 x 64 x 64, RGBA16F**, owned by RenderGraph
  (2 MiB texels). It is not an atlas or a private FinalBlit texture.
- The producer declares a storage write; FinalBlit declares a sampled read.
  Graph scheduling and transitions cover graphics/compute queues. LUT dimensions
  are independent of the viewport. A 2D/3D edge mismatch is rejected.
- Coordinates use log2(1 + linear / 0.01) / log2(1 + 65504 / 0.01) per channel.
  Black maps exactly to zero; the finite nonnegative FP16 range is covered.
  Out-of-domain input clamps. Sampling is trilinear with texel-center coordinates.
- Generation derives dispatch coverage and endpoint coordinates from the actual
  volume dimensions, matching the dimension-based lookup. All 64 cubed voxels
  are written. Large near-white regions in SDR volume previews are expected:
  the log-shaped domain extends to 65504, well into the tone mapper's highlights.
- SDR cube values are sRGB display code values; HDR values are absolute scRGB
  (1 = 80 nits). PQ is not baked into the cube. Display EV applies once before lookup.
- The cube regenerates each execution, supporting live edits, reload, resize and
  overlapping frames without a stale private cache. Allocation and custom image /
  ACES table upload occur at graph compilation, never per frame.
- Profile/peak changes recompile the graph and rebuild the matching ACES tables.
  The ordinary 2D viewport cannot directly display a volume; inspect FinalBlit.color.

An unconnected LUT retains the old simple FinalBlit path for standalone custom or
 diagnostic graphs. It does not run ACES implicitly. For old custom graphs, move
 grading and toneCurve properties from FinalBlit to ColorGradingLUTPass, connect
 ColorGrading.lut to FinalBlit.lut and retain the scene-color source connection.

## Independent Slang module

Import ColorGrading and use Metallic.ColorGrading. Public functions are
 evaluateColorGrading, gradingLutEncode, gradingLutDecode and applyGradingLut.
Parameters and texture/sampler resources are explicit; library code has no fixed
binding slots or consumer macros. The module is split into ColorMath, ACES1,
ACES2, Film, Parameters and Transform. CPU table generation lives in ACESTables.cpp.

UE engine snapshots, .ush includes and compatibility macros have been removed.
Unused output encodings, top-level inverse transforms and dead blocks were pruned.
Inverse appearance-model helpers required by forward ACES rendering remain.
Formulas were adapted from UE 5.7.4 and Academy ACES; attribution and applicable
Unreal/Academy/OpenColorIO notices are retained in Shaders/Licenses/ColorGrading.
There is no engine installation requirement. Refactoring does not change licensing.

## Controls

| toneCurve | Result |
| --- | --- |
| aces2 (default) | ACES2 forward transform, AP1 limiting gamut, scene multiplier 1.5; SDR 100 nits, HDR selected peak |
| unreal | UE Film in SDR; ACES 1.3 in HDR with dark surround, min 0.0001 nits, mid 15 nits, multiplier 1.5 and stretch black |
| reinhard / exponential / none | Legacy alternatives composed into the same LUT |

Global RGB/master saturation, contrast, gamma, gain and offset operate in AP1.
Film slope/toe/shoulder/clips, blue correction and tone amount affect only UE SDR.
Defaults remain 0.88 / 0.55 / 0.26 / 0 / 0.04, blue correction 0.6, expansion 1,
tone amount 1. Shadow/midtone/highlight controls and white balance are not exposed.
HDR ACES produces absolute luminance: paper white affects UI/SDR composition,
not another scale on scene ACES output. Invalid numeric inputs are sanitized.

Custom LUT controls retain UE legacy behavior:

- lut1 through lut4 accept absolute or repository-relative **256 x 16** images
  (prefer PNG): 16-cubed data, red inside a tile, green vertically, blue across tiles.
  Upload uses RGBA8 UNORM without sRGB decoding. Two bilinear samples interpolate
  blue slices. Custom LUT evaluation occurs during cube composition.
- lut1Weight through lut4Weight blend four textures plus neutral. First defaults
  to 1, others 0. The remaining weight below one is neutral; contributors below
  1/512 are dropped, duplicate paths keep the strongest slot, and weights normalize.
- Legacy custom LUTs operate in sRGB display space **in SDR only**. HDR bypasses
  them, matching UE. This feature does not import .cube or scene-linear HDR LUTs.
- Enter commits a path and rebuilds resources. Grading and weight changes update
  live. Unreadable or wrong-sized LUTs report an error.

Example ColorGradingLUTPass properties:

    { "toneCurve": "aces2", "colorSaturation": [1, 1, 1, 1],
      "lut1": "Asset/LookDev/MyGrade.png", "lut1Weight": 0.5 }

## Validation

Build Metallic and MetallicRHITests in the existing MSVC build-scheduling-release
configuration, then run:

    .\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=*color_grading_*:*hdr_*:*final_blit* --rhi-validation --output-dir .cache/lut-validation

GPU coverage includes native volume allocation, full 64-cubed finite RGB/alpha
readback with a nonmatching viewport, default ACES2, the Film 18% anchor,
custom LUT axes/interpolation/weights, live updates without recompilation, exposure
placement, black, reload, profile/peak changes and dimension mismatch rejection.
Fixed OpenPBR captures use 384 x 384, 64 frames / 256 spp.

Against the previous direct-evaluation SDR captures, ACES2's mean absolute channel
error is **0.081/255**, 99th percentile **1/255**, maximum **2/255**. UE Film's mean is
**0.145/255**, maximum **2/255**. Both images were inspected. This measures one scene;
it is not a bound for arbitrary grading or custom LUTs.

Logs/images live under .cache/lut-*. No UE editor screenshot parity, calibrated HDR
TV appearance, long-duration VRAM result or GPU speedup is claimed. The unrelated
missing EOS/Validation.json loader manifests remain visible in logs.

2026-10-01: 11 color/display GPU and pipeline tests plus 10 graph contract/lifetime
tests passed. The real Windows editor rendered 16 LookDev frames while switching
scRGB -> HDR10 -> SDR -> scRGB. All profiles negotiated and presented successfully.
The editor log is .cache/lut-editor-smoke.log.
