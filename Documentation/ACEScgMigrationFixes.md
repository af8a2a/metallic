# ACEScg migration review fixes

Validated on 2026-10-04, Windows x64 / MSVC Release, RTX 5070 Ti, Vulkan.
This supplements [the migration validation](ACEScgMigrationValidation.md) with
repairs from the subsequent correctness review. See [Color Pipeline](ColorPipeline.md)
for the resulting source/working/display contracts.

## Corrected contracts

1. **Tagged LDR environments decode once.** Explicit source tags use normalized
   8/16-bit decoder samples before their declared transfer function. Untagged LDR
   retains the previous gamma 2.2 behavior; explicit linear Rec.709 remains raw.
   Source tag changes, including untagged/explicit-linear changes, reload the same
   path. Serialization retains whether a default source space was explicitly set.
2. **Texture semantics follow the material slot.** Lowering restores Color/Data
   defaults from the destination slot rather than inheriting previous metadata.
   Empty USD color bindings retain Color semantics, so subsequently adding an
   ACEScg color resource succeeds and untagged color resources decode normally.
3. **Debug output carries its display encoding through the graph.** BaseColor and
   shadow transmittance use display-linear Rec.709; normal/ID/FrontFace retain
   their display code values. Diagnostic backgrounds are black. Exposure, grading,
   copies, sliders and SR/RR respect the encoding. Per-frame metadata is prepared
   in graph order before recording-worker snapshots. SR/RR spatially resize debug
   output without SDK evaluation and reset temporal history on returning to beauty.
   The resize kernel is optional on devices without bindless; same-size debug
   copies and the existing RR beauty/RR/SR Off compilation remain available.
4. **Material Value IR bounds apply in working RGB.** Legacy IR arithmetic stays
   in authoring Rec.709, but base/emission bounds apply after conversion back to
   the working basis. Native AP1 primaries and HDR emission therefore survive
   signed/out-of-range intermediate Rec.709 coordinates.
5. **Procedural direct-light radiance converts at its source.** The authored
   Rec.709 key-light constant now enters the active working basis consistently
   with procedural sky radiance.

## Current verification

Evidence paths below are relative to the repository root. SDK tests actually
execute SR/RR; their shutdown problem is called out separately rather than
counted as a clean successful process run.

| Check | Result | Evidence |
| --- | --- | --- |
| Editor, RHI, Scene and ShaderRequest targets | Built with the existing MSVC Release configuration | `build-scheduling-release/acescg-fixes-build.log`, `acescg-fixes-build-complete.log`, `acescg-fixes-compat-build.log` |
| USD-enabled scene target | Built | `build-asset-rebuild-usd/acescg-fixes-usd-build.log` |
| USD untextured-material import and tagged/untagged lowering | Actual enabled importer and lowering succeeded | `build-asset-rebuild-usd/acescg-fixes-usd-probe.log` |
| Scene suite, ACEScg and Rec.709 | 121 passed, 9 skipped in each mode | `build-scheduling-release/acescg-fixes-scene.log`, `rec709-fixes-scene.log` |
| Shader request and native Float16 contracts | 8 passed | `build-scheduling-release/acescg-fixes-requests.log` |
| Core color/material/environment/display/exposure GPU regressions | 12 passed in each mode | `build-scheduling-release/acescg-fixes-core.log`, `rec709-fixes-core.log` |
| OpenPBR debug views and VBuffer transmission/parity | Both passed in each mode | `build-scheduling-release/acescg-fixes-scenes.log`, `rec709-fixes-scenes.log` |
| Multi-queue submit and parallel GPU fanout | Both passed | `build-scheduling-release/acescg-fixes-scenes.log` |
| Exposure stage export/cancel | Passed in both modes | `build-scheduling-release/acescg-fixes-stages.log`, `rec709-fixes-stages.log` |
| RTXDI -> RELAX -> Composite scene chromaticity | Passed in both modes, 16 frames per on/off path, tolerance 0.015 | `build-pass-stages-nrd/acescg-fixes-nrd-scene.log`, `rec709-fixes-nrd-scene.log` |
| Actual SR/RR debug bypass and RR history/resize | All 3 test assertions passed in each mode; process shutdown failed (see below) | `build-scheduling-release/acescg-fixes-dlss.log`, `rec709-fixes-dlss.log` |
| Final SR/RR debug bypass after capability compatibility adjustment | 2 passed in each mode, both processes exit 0 | `build-scheduling-release/acescg-fixes-dlss-final.log`, `rec709-fixes-dlss-final.log` |
| Original non-bindless RR beauty/RR Off/SR Off compile contracts | Passed, separate SDK device with heap disabled, exit 0 | `build-scheduling-release/acescg-fixes-non-bindless.log` |
| Editor smoke | Submitted and presented an actual Vulkan frame, exit 0 | `build-scheduling-release/acescg-fixes-editor-smoke.log` |
| Working-color static audit and whitespace check | Passed | `Tools/CheckWorkingColor.cmake`, `git diff --check` |

The core tests read back a real tagged LDR environment across successive source
changes, run fourteen production IR texture/arithmetic cases (including native
AP1 primaries/HDR), and check the production procedural light function. The
runtime display test switches encodings without graph recompilation through
CopyColor/AutoExposure/FinalBlit with grading and exposure enabled, using both
Joined and Pipelined recording and 1/4 workers.

The production OpenPBR/VBuffer tests retain their normal exposure/LUT chain.
BaseColor, geometry normal, FrontFace, triangle ID, tangent and bitangent image
arrays are **identical between ACEScg and Rec.709** (maximum difference and MAE 0).
See `build-scheduling-release/tests/fixes-debug-image-comparison.json`; PNGs remain
under `tests/acescg-fixes-scenes` and `tests/rec709-fixes-scenes` within that tree.
This comparison concerns display/data diagnostics, not stochastic beauty output.

## Limits

- The 9 Scene skips cover disabled USD tests and unavailable full scene assets.
  The separate enabled USD probe covers the reviewed empty-binding/lowering
  regression; it does not establish complete USD/MaterialX color-management support.
- Streamline still records a shutdown exception/minidump and one leaked Vulkan
  semaphore in both modes. Both combined three-test history runs exit with access-violation
  status `0xC0000005` after all selected frame assertions pass. The final separate
  two-test debug-bypass runs exit 0 and produce no SDK shutdown warning. The same problem
  predates these fixes and is documented in the original validation. SDK shutdown
  is still unresolved; vendor code was not modified.
- Without bindless, same-size diagnostic copies are supported; diagnostic
  resizing returns `Unsupported`. RR beauty and RR/SR Off retain their previous
  requirements.
- NRD's Rec.709 adapter retains its documented native wide-gamut clipping limit.
  The tested colored-light scene is within that adapter's valid gamut.
- The editor smoke used the persisted material-visualization scene and
  `--skip-shader-warmup`; it does not establish extended interactive or temporal
  correctness. The new resize shader is compiled/executed by the real SDK bypass
  regressions and remains in the optional warmup catalog.
- Physical HDR calibration/profile switching, Painter/OCIO authoring and full
  MaterialX color management remain unvalidated, as in the original migration.

## Reproduction

Build selected targets first in an x64 MSVC developer shell, retaining existing
compiler, generator and SDK settings. Run GPU processes serially. Set
`METALLIC_WORKING_COLOR_SPACE` before process startup to `acescg` or `rec709`.
Full selected filters and capture paths are recorded in the corresponding logs.

```powershell
cmake --build build-scheduling-release --target Metallic MetallicRHITests MetallicSceneTests MetallicShaderRequestsTests -j 8
cmake --build build-pass-stages-nrd --target MetallicNRDTests -j 6
.\build-scheduling-release\tests\MetallicSceneTests.exe
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='RHIRendering.environment_ldr_source_color_space:RHIRendering.material_value_ir_texture_footprints:RHIRendering.display_encoding_runtime_parallel_propagation'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='RHIResource.working_color_dlss_non_bindless_compile' # own process, omit --rhi-streamline
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-streamline --gtest_filter='RHIRendering.working_color_dlss_sr_debug_bypass:RHIRendering.working_color_dlss_rr_debug_bypass:RHIRendering.working_color_dlss_rr_history'
.\build-pass-stages-nrd\tests\MetallicNRDTests.exe --gtest_filter='NRDWorkingColor.*'
cmake -DSOURCE_DIRECTORY=E:/metallic -P Tools/CheckWorkingColor.cmake
```
