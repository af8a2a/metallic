# ACEScg migration validation

The subsequent correctness review and its repairs are recorded in
[ACEScg migration review fixes](ACEScgMigrationFixes.md). The results below
retain the original migration run; use the supplement for the reviewed fixes.

Validated on 2026-10-04, Windows x64 / MSVC Release, NVIDIA GeForce RTX 5070 Ti,
NVIDIA driver 616.92, Vulkan core validation enabled for the GPU regressions.
Metallic now defaults to scene-linear ACEScg/AP1/ACES white. Rec.709 compatibility
is selected before process startup with `METALLIC_WORKING_COLOR_SPACE=rec709`.
See [Color Pipeline](ColorPipeline.md) for the source, working and display contracts.

## Implementation scope

The shared CPU/Slang core owns RGB/XYZ and white adaptation matrices, process
selection, shader defines and shader-cache identity. Material factors, lights and
environment radiance convert at their upload boundaries. Color texture samples
carry separate source primaries/transfer and Color/Data usage; factor modulation
occurs in the declared source basis. Authored glTF/editor/material values remain
linear Rec.709 and survive saving. USD Preview Surface texture tags and material
asset tags are retained. Environment conversion precedes mips, SH, filtering and
importance sampling. Legacy Material Value IR retains its authoring arithmetic.

OpenPBR, PathTrace, VBuffer and RTXDI use working RGB and working Y. NRD has an
explicit Rec.709 adapter; SR/RR retain linear HDR working inputs and NR retains
display sRGB. ACES 2.0 receives AP1 once; direct legacy display paths and color
diagnostics convert to display Rec.709. scRGB, HDR10 and UI encodings stay fixed.

## Results

| Check | Result | Evidence relative to repository root |
| --- | --- | --- |
| Editor/RHI/Scene/ShaderRequest/ShaderCompiler build | Passed | `build-scheduling-release/acescg-complete-build.log`, `acescg-build-verified.log`, `acescg-capture-build-verified.log` |
| Scene CPU tests, including tagged textures and environment save/load | 119 passed, 9 skipped | `build-scheduling-release/acescg-scene-final.log` |
| Shader request/Float16 contracts | 8 passed | `build-scheduling-release/acescg-cache-final.log` |
| Working-color static audit | Passed | `build-scheduling-release/acescg-audit.log` |
| Complete configured ACEScg shader warmup | 170 requests, 0 failures | `build-scheduling-release/acescg-warmup-verified.log` |
| Complete configured Rec.709 shader warmup | 170 requests, 0 failures | `build-scheduling-release/rec709-warmup-verified.log` |
| ACEScg CPU/GPU color and scene regressions | 24 passed | `build-scheduling-release/acescg-validated.log` |
| Rec.709 CPU/GPU compatibility regressions | 12 passed | `build-scheduling-release/rec709-validated.log` |
| NRD plans and all supported native shader permutations | 5 passed | `build-pass-stages-nrd/acescg-contract.log` |
| NRD REFERENCE/REBLUR/RELAX/SIGMA and shadow/history GPU tests | 6 passed | `build-pass-stages-nrd/acescg-gpu.log` |
| RTXDI -> RELAX -> Composite scene chromaticity, 16 frames per on/off path | Passed in both modes, per-channel normalized RGB tolerance 0.015 | `build-pass-stages-nrd/acescg-scene-color-verified.log`, `rec709-scene-color-verified.log` |
| Actual DLSS-SR, DLSS-RR and DLSS-NR paths | 3 passed | `build-scheduling-release/acescg-sdk-verified.log` |
| DLSS-RR camera/history/odd-size resize in compatibility mode | Passed | `build-scheduling-release/rec709-rr-verified.log` |
| OpenUSD-enabled scene library build | Passed | `build-asset-rebuild-usd/acescg-usd-build.log` |
| Editor smoke test | Submitted and presented a Vulkan frame | `build-scheduling-release/acescg-editor-smoke.log` |

The 24-test ACEScg run includes CPU/GPU transforms, known primaries and neutrals,
negative/HDR signals, sRGB and native AP1 inputs, Color/Data bypass, source-basis
modulation, NRD frontend packing, forbidden per-shader basis overrides, exposure,
LookDev PathTrace/VBuffer, custom/material-asset programs, deferred transmission,
HDR output/UI, constant environment filtering, ReGIR lifecycle and KTX2 resources.
Earlier GPU runs also exercised streamed shading/transmission and RTXDI typed
post-processing/history (`build-scheduling-release/acescg-runtime.log`).

DLSS-SR checks three output/settings variants, camera motion and guides, including
optional NR. RR checks 24 frames, camera motion and resize from 384x256 to 321x217.
NR checks the display-domain fixture, tuning and exact zero-intensity bypass.
These tests establish execution and basic history/color correctness, rather than
model accuracy for every wide-gamut input or long-duration temporal stability.

## A/B references

Before source edits, six existing color/display tests passed and their PNGs were
saved under `build-scheduling-release/tests/acescg-baseline`. After migration,
**all eight corresponding image pixel arrays in Rec.709 compatibility are
identical** (MAE=0, maximum difference=0). See
`build-scheduling-release/tests/compatibility-image-comparison.json`.

New raw scene references are in `build-scheduling-release/tests/acescg-validated`
and `tests/rec709-validated`, named `OpenPBRDefault-<space>.rgba32f`, with adjacent
JSON source/working-space manifests. They capture `PathTrace.color`, 384x384,
64 frames at 4 spp/frame, before exposure and display transform. Both are finite
and retain values above 1 (ACEScg max 2.002015; Rec.709 max 2.078454).

After conversion to common D65 XYZ, mean absolute difference is 0.0000617117,
versus mean absolute reference signal 0.268169: **relative MAE 0.0230%**.
Maximum absolute XYZ difference is 0.00905692. This is one fixed 256-spp LookDev
workload, including Monte Carlo and per-channel BSDF basis effects; it does not
assert all scenes or all saturated materials are invariant under a basis change.
See `build-scheduling-release/tests/linear-reference-comparison.json`.

`Asset/LookDev/OpenPbrDefault/ACEScgRuntimeReference.json` records the new runtime
contract. Historical `Reference.json`, MaterialX source and captures retain their
original meaning. Generated images, readbacks and shader caches remain local
build outputs.

## Regression repairs

ACEScg-to-display conversion amplified a deferred FP16 error to 3/255 at one
transmission pixel, exceeding the existing 2/255 limit. ACEScg keeps energy color
vectors and metalness/transmission inputs in FP32; other bounded scalar weights
retain FP16. The original threshold now passes. ReGIR source-color assertions
convert raw working readbacks to their authored Rec.709 basis before checking
RGB transport; PDF/power validation remains in the actual working basis.

Initial RR test failures came from the test's structural guide configuration and
then a null shared-view assumption in a legacy per-pass-camera graph. The final
regression binds one shared view and uses the production RR graph's color path.
First-run logs remain preserved; the corrected runs above supersede them.

## Limits and outstanding external validation

- The 9 CPU skips cover OpenUSD-disabled cases and unavailable full Zorah /
  SuperSponza assets. The OpenUSD importer compiled in a separate enabled tree;
  its full runtime color-management cases were not exercised.
- SDR/scRGB/HDR10 and ImGui composition have GPU pixel checks. The editor smoke
  used the locally persisted material-visualization sample. Physical HDR monitor
  calibration, display-profile switching and extended interactive viewing were
  not validated by these checks.
- Painter/OCIO authoring screenshots and full MaterialX color-management import
  were not performed. EXR ingestion and an OCIO runtime graph remain outside
  this migration; the existing HDR decoder accepts explicitly tagged sources.
- NRD vendor sanitization can clip negative Rec.709 components of native AP1
  signals outside the Rec.709 gamut. The checked in-gamut colored-light scene
  passed; fully native wide-gamut NRD specialization needs separate work.
- Streamline logs an exception/minidump during shutdown and Vulkan reports one
  leaked semaphore after successful feature runs. **The same shutdown behavior
  reproduces in both ACEScg and Rec.709** (`acescg-rr-verified.log` and
  `rec709-rr-verified.log`); successful frame tests do not establish clean SDK
  shutdown. No vendor SDK changes are included.
- NRC/SHARC packing limits and precision were documented but not runtime-tested.
  No full-scene performance speedup, VRAM result or long-term stability claim is
  made. Warmup request counts reflect this configured tree's enabled SDKs.

## Reproduction

Use an x64 MSVC developer shell and the existing configured directories; do not
change SDK/compiler settings. Build selected test executables before running.

```powershell
cmake --build build-scheduling-release --target Metallic MetallicRHITests MetallicSceneTests MetallicShaderRequestsTests MetallicShaderCompiler -j 8
cmake --build build-pass-stages-nrd --target MetallicNRDTests -j 8
$env:METALLIC_WORKING_COLOR_SPACE = 'acescg'
cmake --build build-scheduling-release --target MetallicShaderWarmup
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-bindless '--gtest_filter=ColorSpace.*:RHIRendering.working_color_cpu_gpu_texture_contract:RHIRendering.working_color_lookdev_linear_capture'
.\build-pass-stages-nrd\tests\MetallicNRDTests.exe --gtest_filter=NRDWorkingColor.*
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-realtime '--gtest_filter=RHIRendering.realtime_clustered_dlss_pipeline:RHIRendering.working_color_dlss_rr_history:RHIRendering.dlss_nr_runtime'
ctest --test-dir build-scheduling-release -R '^MetallicWorkingColorAudit$' --output-on-failure
$env:METALLIC_WORKING_COLOR_SPACE = 'rec709'
cmake --build build-scheduling-release --target MetallicShaderWarmup
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-bindless '--gtest_filter=ColorSpace.*:RHIRendering.working_color_cpu_gpu_texture_contract:RHIRendering.working_color_lookdev_linear_capture'
```

Set `--output-dir` to retain distinct capture sets. Full selected filters are
recorded at the top of the corresponding validation logs. Shader warmup remains
an optional manual CMake target outside default editor/sample dependencies.
