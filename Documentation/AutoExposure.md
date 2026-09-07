# Physical lighting and auto exposure

Metallic meters physical HDR radiance before tone mapping. The implementation follows the extended EV100 convention and histogram eye adaptation in the local Unreal source:

- `E:/UnrealEngine/Engine/Source/Runtime/Renderer/Private/PostProcess/PostProcessEyeAdaptation.cpp`: lens attenuation, 18% middle gray, EV100 limits, directional adaptation speeds.
- `E:/UnrealEngine/Engine/Shaders/Private/PostProcessHistogramCommon.ush`: percentile-clipped mean log luminance and adaptation in logarithmic space.

No Unreal code is copied. Metallic uses an analytically integrated linear/exponential transition so splitting a frame timestep does not change adaptation, including when crossing the transition boundary.

## Use

The physical realtime, path tracing (including SHaRC/NRC), RTXDI/NRD and DLSS sample graphs include the new stage. In **Physical Lighting**, enable **Auto Exposure (Histogram)** and adjust **Eye Adaptation**. Newly loaded scenes default to automatic exposure. Existing sidecars with a lighting block but no `autoExposure` block retain manual exposure. Settings are saved with the scene.

For a custom graph:

1. Enable **HDR Output (Auto Exposure)** (`outputLinear: true`) on `SceneRealtimeLightingPass`, `ScenePathTracePass` or `SceneRtxdiPass`.
2. If using RTXDI + NRD, enable it on `RtxdiCompositePass` too; all radiance and emissive inputs must use the same exposure convention.
3. Connect the final HDR color, after NRD composition or DLSS reconstruction, to `AutoExposurePass.source`.
4. Connect `AutoExposurePass.color` to `FinalBlitPass.source`.

The source must be a floating-point HDR texture at the pass resolution. LDR/tonemapped color is rejected. Debug material/normal/RTXDI visualizations should be connected directly to `FinalBlitPass`, bypassing exposure.

`AutoExposurePass` keeps the existing Reinhard + gamma display curve by default; the RTXDI sample selects its exponential curve. `artisticExposure` is an optional final multiplier. The older RTXDI/composite `exposure` properties apply only to their inline LDR output. `sourceExposure` removes a *known, fixed* input pre-exposure before metering; leave it at 1 for Metallic's HDR outputs.

## Settings and calibration

For luminance `L` in cd/m² and the current lens convention (`q = 0.78`, saturation luminance `Lmax = 2^EV100`):

```text
metered L = exp2(percentile-clipped mean(log2(pixel luminance)))
target EV100 = clamp(log2(metered L / 0.18), minEV100, maxEV100)
display multiplier = exp2(compensation - adapted EV100)
manual multiplier = exp2(compensation - exposureEV100)
```

| Scene setting | Default | Meaning |
| --- | --- | --- |
| `minEV100`, `maxEV100` | -10, 20 | Exposure limits; equal values fix the exposure immediately |
| `compensation` | 0 | Stops; +1 doubles displayed linear brightness |
| `lowPercent`, `highPercent` | 70, 90 | Percentile interval retained for metering |
| `histogramMinEV100`, `histogramMaxEV100` | -10, 20 | Histogram log₂ luminance range under the current lens convention |
| `speedUp`, `speedDown` | 3, 1 | Stops/second entering brighter/darker environments; zero holds that direction |
| `transitionDistance` | 1.5 | Switch to exponential convergence near the target; zero uses only linear motion |

First use, graph recompilation/resizing, scene identity changes, mode changes and the node's **Reset Adaptation** action discard exposure history. Light/intensity changes preserve it and adapt smoothly. Camera controllers can increment `resetSerial` for a camera cut. Ordinary camera motion does not reset exposure. Timing uses elapsed monotonic time capped at one second; offline rendering/tests can set `adaptationDeltaSeconds` to a fixed positive timestep.

## GPU and output contract

The GPU builds a 64-bin histogram per 16×16 tile, reduces it, then applies exposure and tone mapping. Adjacent bins share fixed-point sample weights to reduce stepping. Pixels at or below the histogram floor are excluded, following Unreal's default zero black-bucket influence, so a black background does not blow out the subject. An entirely empty meter produces finite output and cannot seed the next frame's adaptation. Partial edge tiles, negative values and non-finite input are handled. No luminance data is read back to the CPU. Adaptation history belongs to each pass instance/view and stays on the graphics queue; explicit barriers order its use between overlapping frames. Canceled submissions invalidate history.

`exposure` is a 16-byte buffer containing four floats: display multiplier, adapted EV100, target EV100 and metered luminance. `histogram` exposes the per-tile counts for GPU inspection. `color` is opaque, gamma-encoded RGBA8 for the existing final blit.

For reference LookDev, `toneCurve: "none"` (**None (sRGB)**) applies exposure and
the exact piecewise linear-to-sRGB transfer without highlight compression.
Display values above one clip, while the upstream HDR output remains available.
The existing Reinhard and Exponential options retain their original gamma-2.2
encoding. See [OpenPBR LookDev](OpenPbrLookDev.md) for a calibrated example.

Ordinary HDR lighting and path-trace accumulation use RGBA32F. The existing NRC/DLSS/NRD paths retain their FP16 resource contracts; inputs exceeding their representable range require pre-exposure before entering those integrations. This change does not implement feedback pre-exposure, metering masks, exposure compensation curves or local exposure.

## Validation

- `MetallicSceneTests`: `SceneEditing.PhysicalLightingRoundTrip`, `SceneEditing.AutoExposureValidationAndLegacyLoading` cover persistence, old sidecars and invalid settings.
- `MetallicRhiTests --filter auto_exposure --rhi-validation`: physical gray calibration, percentile rejection, manual/automatic modes, limits, compensation, both speeds, zero speeds, timestep invariance and invalid pixels.
- `MetallicRhiTests --filter photometric --rhi-validation`: real lighting, identical inline/manual-post-exposure pixels and a 10-stop light increase compensated by auto exposure. Produces `auto-exposure-realtime.png` in the selected output directory.
