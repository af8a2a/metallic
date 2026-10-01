# LookDev color and output profiles

Metallic uses **scene-linear Rec.709/sRGB primaries, D65** inside the renderer.
Lighting, accumulation, denoising and upscaling keep floating-point HDR radiance.
AutoExposure outputs exposed linear RGBA16F in every profile; it does not tone map
or encode sRGB. Path tracing and RTXDI always output linear radiance. The old
`outputLinear` switch no longer changes their behavior.

The display boundary is ColorGradingLUTPass -> FinalBlit followed by editor composition
and presentation. Scene pipelines default to ACES 2.0 baked into a native 64-cubed
FP16 RenderGraph texture; FinalBlit samples it. See [color grading](ColorGrading.md).
Changing the output profile does not select a different lighting/exposure format.
Diagnostic textures (normals, IDs, material colors) retain explicit display encodings.
Floating-point storage alone is not an encoding declaration: use `SceneLinear`,
`ExposedLinear`, `scRGB` or `sRGB` on graph resources.

| Profile | Swapchain format / Vulkan color space | Purpose |
| --- | --- | --- |
| `SDR_sRGB` | BGRA8/RGBA8 sRGB preferred; `SRGB_NONLINEAR_KHR` | SDR reference, nominal 100 nits |
| `HDR_scRGB` | RGBA16F; `EXTENDED_SRGB_LINEAR_EXT` | Windows HDR editor and LookDev default |
| `HDR10_PQ` | A2B10G10R10 UNORM; `HDR10_ST2084_EXT` | BT.2020 + ST 2084 game/TV verification |

Use **Display Output** to select a profile. The requested and actual profile are
separate: HDR requires Windows HDR and an advertised matching format/color-space
pair. An unavailable HDR profile falls back to SDR with a visible status and log.
It never silently selects the other HDR encoding. SDR UNORM fallback is supported
with explicit sRGB encoding; sRGB attachments receive linear values, avoiding a
second transfer function.

Default HDR paper white is fixed at **203 nits**, peak **1000 nits**, display
exposure **0 EV**. Set peak to the intended mastering target (for example 600 or
1000 nits); this is not automatic display calibration. Following the Windows SDR
white level is opt-in. SDR's nominal 100-nit reference requires a calibrated
display/OS configuration; an ordinary SDR swapchain does not enforce luminance.

With the legacy display transform, exposed linear 1 maps to HDR paper white;
the highlight shoulder approaches peak. The optional Unreal/ACES transforms use
their own absolute luminance mapping; see [color grading](ColorGrading.md).
scRGB uses **1 = 80 nits**, so 203-nit white is 2.5375 and 1000 nits is 12.5.
Both HDR profiles keep FinalBlit's output in absolute scRGB for composition.
HDR10 composites UI in a per-swapchain-image FP16 target, converts linear Rec.709
to BT.2020, then applies PQ once in a fullscreen pass with blending disabled.
PQ code value 1 means 10000 nits, independently of the selected mastering peak.
When available, `VK_EXT_hdr_metadata` receives BT.2020/D65 mastering primaries and
the selected peak. MaxCLL/MaxFALL remain unknown (zero), rather than fabricated.
Detached ImGui windows currently use SDR previews.

Display transforms (`aces2`, `unreal`, `reinhard`, `exponential`, `none`) belong to
**ColorGradingLUTPass**. Repository scene graphs have been migrated. For custom
graphs, connect its `lut` output to FinalBlit's `lut` input and move grading controls
to the producer. Unconnected diagnostic graphs retain the simple FinalBlit path.
SDR transform evaluation uses the exact
sRGB transfer function. Display exposure is separate from physical scene exposure.
DLSS-NR currently requires display-referred SDR input; its node preserves exposed
linear HDR through a logged bypass in every profile. Disabling its fallback reports
Unsupported. DLSS-SR/RR continue to process linear HDR.
An explicit `displayReferredInput: true` on DLSS-NR supports separate SDR
display-only graphs after FinalBlit; it requires an already sRGB-encoded RGBA8 input.

For reproducible launches, set `METALLIC_OUTPUT_PROFILE` to one of the exact profile
names before starting Metallic. Unknown names fail initialization. With no override,
the editor requests `HDR_scRGB` and negotiates SDR when unavailable.

Validation covers surface-pair negotiation, GPU exposure/display pixels, fixed
linear intermediates across all profiles, sRGB encoding, and the real editor
FP16 composition -> RGB10A2 PQ shader including translucent UI. These checks do
not replace visual inspection on a calibrated HDR monitor or an HDR10 television.

References: [Windows Advanced Color and scRGB](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range),
[Vulkan WSI color spaces and HDR metadata](https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html).

## Validation record — 2026-10-01

- Built `Metallic` and `MetallicRHITests` in the existing MSVC
  `build-scheduling-release` configuration.
- Focused RHI run: 14 passed, 3 skipped. Coverage includes display surface pairs,
  HDR/scRGB/PQ GPU pixels, linear UI alpha composition, exposure adaptation and
  cancellation, exact SDR transfer, physical lighting, graph assets and lifetimes.
  The skipped tests require an explicitly enabled Streamline DLSS-NR SDK session.
- Real Windows editor calibration run: 16 frames, switching scRGB -> HDR10 ->
  SDR -> scRGB. Actual negotiated Vulkan format/color-space pairs were
  `(97, 1000104002)`, `(64, 1000104008)`, `(50, 0)` respectively; all frames presented.
  Set `METALLIC_SMOKE_TEST_SAMPLE=hdr-calibration`, `METALLIC_SMOKE_TEST_FRAMES=16`
  and `METALLIC_SMOKE_TEST_OUTPUT_PROFILES=1`, then run `Metallic.exe --smoke-test`.
- Inspected the physical lighting test's SDR output. HDR television appearance,
  calibrated luminance and long-duration temporal/VRAM behavior remain unverified.
- Local logs: `.cache/display-build.log`, `.cache/display-final-tests.log`,
  `.cache/display-smoke-pq.log`, `.cache/display-smoke-switch.log`. The Vulkan loader
  reports two pre-existing missing layer manifests (EOS overlay and `E:\Validation.json`);
  these GENERAL loader messages are retained separately from workload validation errors.
