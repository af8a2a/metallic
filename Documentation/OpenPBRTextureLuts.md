# OpenPBR texture LUTs

The production OpenPBR implementation supports **texture mode only**. The
native Slang module contains no LUT arrays or integration loops; requesting
`OPENPBR_USE_TEXTURE_LUTS=0` at the renderer adapter is a compile error. The
unchanged Adobe snapshot remains an oracle and the source of precomputed data.

## Precomputed texture payloads

[`OpenPBRLutData.h`](../Source/Runtime/Render/Material/OpenPBRLutData.h) embeds
Adobe's offline-integrated energy and sheen tables, in their original axis
order. No new numerical integration, runtime array-to-RGBA expansion, Python
dependency, or external asset-path lookup is needed to start the renderer.
Updating the vendored data updates these payloads at C++ build time.

| Adobe IDs | Contents | Dimensions | Texture format |
|---|---|---|---|
| 0, 3 | Ideal/opaque dielectric energy complement | 32 x 32 x 32 | R16_UNORM |
| 1, 2, 4, 5 | Average energy/reflection ratio/metal energy | 32 x 32 | R16_UNORM |
| 6 | Average metal energy complement | 32 x 1 | R16_UNORM |
| 7 | Sheen LTC coefficients | 32 x 32 | RGBA32_FLOAT |

Energy values use the original `uint16 / 65535` encoding without requantization.
LTC coefficients keep their sign and full float precision; alpha is baked as 1.
Payload/upload bytes fall from **1,131,008 to 155,712** (86.23% less). These are
texel payload sizes, not Vulkan allocation sizes; device alignment can differ.
Textures have one mip and no sRGB conversion or compression. Upload occurs once
per resource lifetime, with the existing submission rollback and lifetime rules.

## Sampling and integration

[`OpenPBRTextureLUT.slang`](../Shaders/Modules/OpenPBRTextureLUT.slang) is shared
by PathTrace, production Deferred, legacy visibility preview and the GPU probe.
2D lookup uses one hardware bilinear sample instead of four loads. 3D lookup
uses two hardware-filtered XY slices and float Z interpolation instead of eight
loads. Both 3D energy textures have 32 slices. Adobe's half-texel remapping and
IOR extrapolation remain in the BSDF module.

A dedicated linear, clamp-to-edge sampler is bound through
`SceneResourceParameters.openPBRLutSampler` (CPU input ID 98). Material Value IR
keeps input ID 97. CPU/Slang resource records and every affected named layout
were updated together. Legacy preview clients must supply the sampler handle
through `GPU_DRIVEN_OPENPBR_LUT_SAMPLER` alongside their existing LUT bases.

Hardware filtering is not bitwise equivalent to float interpolation. Vulkan
specifies finite sub-texel precision in its
[sampling contract](https://docs.vulkan.org/spec/latest/chapters/textures.html).
On the tested RTX 5070 Ti, direct 3D hardware filtering exceeded the probe's
local interpolation error bound. The two-slice implementation passes that
same bound; the test tolerance was not enlarged to accept the first candidate.
This observation is device-specific, not a claim about all GPU interpolators.

## Validation and timing

Build `MetallicRHITests`, then run:

```powershell
build-scheduling-release/tests/MetallicRHITests.exe --gtest_filter=RHIRendering.material_openpbr_texture_luts --output-dir build/openpbr-textures-check
python -B Tools/Perf/OpenPBRBenchmark.py run --texture-luts --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/openpbr-textures-timing --runs 5
python -B Tools/Perf/OpenPBRBenchmark.py verify build/openpbr-textures-timing
```

The LUT test compares old RGBA32F/manual interpolation and the production
compact texture provider independently with a double-precision CPU oracle.
It checks all texel centers, random coordinates, exact 0/1 and outside-domain
clamp on every axis: 262,144 cases per coordinate mode, eight tables. The bound
is local maximum edge gradients summed over axes / 256, plus `2e-5 *
max(1,abs(value))` for conversion/rounding; centers omit the gradient term.
This is an eight-bit sub-texel quality target, not a universal Vulkan minimum.
Actual SPIR-V, outputs, oracle values and bounds are saved for verification.

Timing uses GPU-local textures/output, separate validation-disabled processes,
two warmup and eight alternating AB/BA measured rounds. Each timestamp covers
eight dispatches with WAW barriers, divided by eight; uploads, CPU oracle,
readback, shader compilation and PSO creation are excluded. This measures LUT
lookup throughput, **not complete BSDF evaluation, a production pass, or frame
time**. The original native/vendor BSDF test remains an independent exact-LUT
oracle; its tight tolerance is unchanged.

Production acceptance additionally renders the 1024 spp LookDev and tests
MaterialGraph texture/normal execution through PT and Deferred, SDK equivalence,
program binning and instance updates. Generated timing/visual evidence is kept
under `build/`, outside source control.

## Acceptance — 2026-10-05

Windows x64 / Release MSVC; RTX 5070 Ti, driver 616.92. Six focused checks passed:
texture LUT oracle, native/vendor BSDF equivalence, material shader contract,
guide compilation, LookDev and MaterialGraph scene execution. Vulkan validation
was enabled for correctness checks and disabled for ordinary timing.

- Texture centers: maximum scaled error `5.96e-8`. Random/clamp coordinates:
  `0.001758`; all per-cell gradient bounds pass. Filtering is not bitwise exact.
- Graph/SDK and binned/unbinned relative errors: zero. Texture/normal graph
  execution: `9.43e-8` (PT), `9.40e-8` (Deferred). Instance changes reach lighting.
- Same 768 x 768 / 1024 spp LookDev before/after: display-space mean absolute
  difference `0.00001582`, RMSE `0.00028265`; output image inspected. Individual
  path samples can change, so this does not imply bitwise image equivalence.

| LUT query pattern | Discovery, 5 processes | Confirmation, 3 processes |
|---|---|---|
| Texel centers | 55.72% reduction [55.09, 56.36]% | 48.11% [14.08, 82.13]% |
| Random/clamp | 58.87% reduction [56.25, 61.49]% | 56.59% [54.96, 58.22]% |

Brackets are two-sided 95% Student t intervals over process-level relative
reductions. All four lower bounds exceed the declared 3% threshold. The wider
center-query confirmation interval is retained; no outlier was removed and no
extra significance-seeking runs were added. Acceptance is scoped to this LUT
kernel on this GPU. Full-pass/frame performance and other GPUs are unmeasured.

Local evidence:

- `build/openpbr-texture-validation-slices/`: validation, bound oracle/readbacks.
- `build/openpbr-texture-production-fixed/`: LookDev and PT/Deferred graph checks.
- `build/openpbr-texture-production/`: native equivalence and shader contracts;
  also retains the initial scene binding conflict, fixed before final acceptance.
- `build/openpbr-texture-discovery/`, `build/openpbr-texture-confirmation/`:
  hashed sources, executable, actual bound SPIR-V, raw readbacks and timings.
- `build/openpbr-texture-report/`: verified chart, report, numerical/visual
  summaries and reproducible `plot_report.py` (run with `python -B -X utf8`).
