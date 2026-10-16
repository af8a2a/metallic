# OpenPBR primary-guides split experiment

`ScenePathTracePass` accepts an experimental `splitPrimaryGuides` property. It is
**off by default** and is supported only with this combination:

```json
{
  "bsdf": "openpbr",
  "exportDenoiserGuides": true,
  "samples": 1,
  "cacheMode": "off",
  "splitPrimaryGuides": true
}
```

Other combinations fail explicitly. The experiment does not add kernels to
normal shader warmup. `MetallicShaderPipelineBench` appends the two requests to
its manual catalog; use `--list` to discover their indices.

## Boundary and costs

`OpenPBRPrimaryGuides.slang` performs the existing `queryOpenPbrPrimaryStage`,
including independent guide material evaluation and guide RNG. It writes the
six primary guide outputs and the **authored** `HitInfo`. It does not export a
closure or a filtered/evaluated shading normal for reuse by the path.

`OpenPBRPathContinuation.slang` runs the existing path loop, replacing only its
first primary query with a record load. The original `primaryQueryConsumed`
guard survives BLEND depth retries. Path RNG, material evaluation, ray cone
propagation, shadow queries, radiance/history and specular-hit-distance output
remain in the continuation. Misses carry an explicit found flag. Camera-ray
arithmetic retains the fused expression grouping.

The record is 100 bytes per pixel: 24 authored scalar words and a word containing
found/front-face flags. Position and bitangent are preserved rather than
reconstructed, to retain the original floating-point values. This first version
does not quantize or compact active pixels. A new recording owns its allocation;
the command buffer retains both allocation and descriptor leases. The render
graph orders the storage write before the continuation read.

Scratch capacity is 25 MiB at 512 x 512, about 197.8 MiB at 1920 x 1080, and
791.0 MiB at 3840 x 2160 **per outstanding recording**. There is an additional
dispatch, storage traffic and a fresh device-buffer allocation per recording.
This can cost GPU time and peak memory even when native creation improves.
The shared scene parameter ABI remains 384 bytes; only the split uses the
392-byte extension packet.

## Native creation measurement, 2026-10-16

Base HEAD: `03c4d423333eb97f77d5f31353bad734a72de41f`, with this experiment applied.
Release/MSVC, RTX 5060, NVIDIA 616.92, Slang 2026.18.2, O1, mapped descriptors,
no SPIR-V postprocess. The fused input SPIR-V is byte-identical to the captured
HEAD request (`9a5e2e8d5d1331c9`, 3,222,616 bytes). Primary and continuation are
380,808 and 2,874,180 bytes respectively; their sum is slightly larger.

These measurements replay exact PSO recipes exported by the rendering test,
including its explicit cache-off specialization. They are not interchangeable
with the earlier catalog-default Native PSO measurement. Each sample uses an
independent process, one creation, an empty application pipeline cache,
`METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`, validation off and no executable
statistics/ISA capture. Slang compilation is bypassed by recipe replay.
Hidden compiler/OS cache state is not guaranteed cold. No build or other GPU
test ran concurrently.

| Repeat | Fused create (s) | Primary create (s) | Continuation create (s) | Split max (s) | Split sum (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.566 | 1.897 | 41.903 | 41.903 | 43.801 |
| 2 | 51.127 | 2.145 | 45.167 | 45.167 | 47.311 |
| 3 | 47.542 | 2.086 | 43.989 | 43.989 | 46.075 |
| Median | 51.127 | 2.086 | 43.989 | 43.989 | 46.075 |

The median longest creation decreases **14.0%**; the median sum decreases
**9.9%**. Both improve in all three paired repeats, but fused process timings
span 47.5–59.6 seconds. These are observations on one GPU/driver, not a stable
cross-device speedup guarantee. The continuation still dominates creation.
For the sum, compute primary + continuation within each repeat before taking
the median; do not add unrelated percentile statistics. These metrics do not
include frontend compilation, resource setup or end-to-end application startup.

## Correctness evidence

Both arms of `RHIRendering.material_phase0_baseline` passed with Vulkan
validation enabled. Opaque1, Mixed1 (MASK/BLEND, normal maps and nested
dielectrics), Sky, LegacyShadow and ABeautifulGame have byte-identical final HDR
and captured outputs, including seven guide channels where enabled. Temporal
captures include camera motion and ABeautifulGame frame 96. LegacyShadow is an
unchanged negative control. Correctness resolutions are 128 x 128 except
ABeautifulGame at 512 x 512. The evidence comparator also checks matching
settings and presence of both split timestamp scopes.

The shader module audit passes (337 sources), all 48 `ShaderRequests.*` tests
pass, and the Python evidence tests cover rejected workload changes, missing
split scopes, wrong Native kernel roles and cache contamination. This does not
establish correctness for unsupported multisample, SHaRC or realtime/deferred
paths, nor full-scene peak memory at high resolution.

Mixed1 and ABeautifulGame readbacks were also inspected as matched-exposure
previews. The one-sample images are noisy in both arms; no visual difference
was observed. The raw byte comparison, rather than preview tone mapping,
provides the differential correctness check.

## GPU cost

Three independent processes per arm, alternating fused/split order, ran all
five cases at 512 x 512 with fixed cameras, 32 warmup and 64 measured frames.
Validation was off, driver caching enabled, disk caches retained and no
measured frames filtered. Each value below is the median of process medians.
Compilation, scene loading and readback are excluded from these GPU timestamps.

| Case | Fused pass (ms) | Split pass (ms) | Change | Fused graph (ms) | Split graph (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opaque1 | 12.507 | 12.899 | +3.1% | 12.563 | 12.916 |
| Mixed1 | 32.515 | 33.104 | +1.8% | 32.543 | 33.126 |
| Sky | 0.798 | 1.084 | +35.9% | 0.846 | 1.107 |
| LegacyShadow (unchanged) | 11.578 | 11.238 | -2.9% | 11.611 | 11.271 |
| ABeautifulGame | 2.681 | 2.910 | +8.5% | 2.713 | 2.932 |

All three performance pairs also pass the HDR/guide comparison exactly. Small
Opaque1/Mixed1 differences are comparable to process drift, including the
unchanged control. Sky and ABeautifulGame show a clearer additional cost.
These pass/graph observations are not end-to-end frame-rate measurements.
The default remains fused: the observed native creation reduction does not
justify enabling this first version broadly with its storage and runtime cost.
The short validation-enabled correctness runs are not used for this table.

## Reproduce

Build `MetallicShaderPipelineBench`, `MetallicRHITests` and
`MetallicShaderRequestsTests` in a compatible configured tree. The commands below
assume `build-release`; replace it consistently for another tree.

```powershell
python -B Tools/CheckShaderModules.py
.\build-release\tests\MetallicShaderRequestsTests.exe --gtest_filter=ShaderRequests.*
ctest --test-dir build-release -R 'Metallic(ShaderModuleOwnershipAudit|PathTraceQueryEvidenceTests|PrimaryGuidesSplitTests)$' --output-on-failure

python -B Tools/Perf/PrimaryGuidesSplit.py prepare build-release/split-correctness
$env:METALLIC_MATERIAL_BASELINE_CASES = 'build-release/split-correctness/fused/Cases.json'
$env:METALLIC_PIPELINE_RECIPE_DIRECTORY = 'build-release/split-recipes/fused'
.\build-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.material_phase0_baseline --rhi-validation --output-dir build-release/split-render-fused
# Repeat with split/Cases.json, a separate recipe directory and output directory.
python -B Tools/Perf/PrimaryGuidesSplit.py compare build-release/split-render-fused/MaterialBaseline.json build-release/split-render-split/MaterialBaseline.json

# Fixed cameras, 512 x 512, 32 warmup + 64 measured frames for GPU timing.
python -B Tools/Perf/PrimaryGuidesSplit.py prepare build-release/split-performance --phase performance
# Run the generated arms in independent processes with --rhi-no-validation.
# Alternate their order and compare each pair; retain all measured frames.

$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
# Select the exported recipe by shaders.compute.module_name, not file order.
.\build-release\Source\MetallicShaderPipelineBench.exe --recipe-file <recipe.json> --iterations 1 --pipeline-cache empty --output-dir <fresh-directory>
python -B Tools/Perf/PrimaryGuidesSplit.py native --group <fused-report.json> <primary-report.json> <continuation-report.json>
# Repeat --group for each independent round. Restore environment overrides after use.
```

See [ShaderPipelineBench.md](ShaderPipelineBench.md) for recipe capture/replay and
cache controls. The Native aggregation validates three distinct, stable roles,
device identity and measurement policy. Catalog reports must have Single=1;
recipe reports do not carry preprocessor defines, so their role check verifies
module/source-entry identity, not the absent macro metadata. Preserve the
render fixture, source hashes and recipe capture provenance as well.

Local evidence is under
`build-release/primary-guides-split-03c4d423/`: `rendering-provenance.json`,
`rendering-comparison.json`, `native-comparison.json`, `gpu-comparison.json`,
`CorrectnessPreview.png`, recipes and individual process reports/logs.
Generated artifacts remain outside source control.
