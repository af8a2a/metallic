# Opt-in SPIR-V dead-branch cleanup

The 2026-10-13 implementation offers two optional SPIR-V cleanup paths: a
separate in-process optimizer after Slang, and two additive passes inside
Slang's existing downstream optimizer. The production default remains **none**.
The rendering and native/GPU measurements below qualify the first path,
`dead-branches`, for three fixed mapped-descriptor workloads. The newer
`slang-dead-branches` path has runtime API, cache/recipe and frontend evidence;
its current Guides output matches the first path byte for byte. A three-pair
fresh-source/native-creation comparison records a 7.72% paired-median reduction
for this single Guides request, including its extra compiler cost. This does
not establish complete renderer startup savings. Both cleanup paths pass the
subsequent fixed MixedSharc HDR/guide comparison. These results use the current
owned Slang modules, not the earlier 2026-10-12 source snapshot.

## Compiler path and controls

[`SlangCompiler.cpp`](../Source/Runtime/Render/Core/SlangCompiler.cpp) runs the
following sequence for `dead-branches` on a cache miss:

1. Generate SPIR-V through Slang's existing component/link/code-generation path.
2. Apply the existing Vulkan target finalization.
3. For `dead-branches`, validate the input for Vulkan 1.4 with scalar block layout,
   then run exactly `CreateDeadBranchElimPass()` followed by
   `CreateSimplificationPass()` through the SPIRV-Tools C++ API.
4. Validate the resulting SPIR-V with the same environment/layout and persist
   only successful code whose source dependencies still match.

The two transforms correspond to `spirv-opt --eliminate-dead-branches
--simplify-instructions`, in that order. No external optimizer process is
launched. Bindings and specialization constants are preserved by optimizer
options; this does not add Vulkan specialization constants or a new shader ABI.

For `slang-dead-branches`, one Slang `CompilerOptionName::DownstreamArgs` entry
uses `stringValue0="spirv-opt"` and a newline-separated `stringValue1` containing
`--eliminate-dead-branches` then `--simplify-instructions`. The passes append to
the selected Slang optimization preset; they do not replace O1 or promote O0 to
O1/O2. The returned Slang blob already includes these passes. Metallic then
applies its existing target finalization and one final Vulkan 1.4 scalar-layout
validation. It does not run the separate optimizer or its input validation a
second time. The qualified SDK is the pinned **Slang 2026.18.2**; alternate SDK
and missing-downstream-library fault cases have not been exercised.

`SlangShaderDesc::postprocessMode` exposes `Default`, `None`, `DeadBranches` and
`SlangDeadBranches`. An explicit mode overrides the process environment.
`Default` resolves `METALLIC_SPIRV_POSTPROCESS`: unset, empty, `default` or `none`
selects `None`; the other accepted values are `dead-branches` and
`slang-dead-branches`. Invalid values fail explicitly.

The Benchmark exposes
`--spirv-postprocess default|none|dead-branches|slang-dead-branches`. The native/GPU
experiment runner accepts the three explicit modes, defaults to `none`, and
sets both the Benchmark argument and process environment. The two-arm cleanup
comparison alternates AB, BA, AB across independent-process blocks. Existing
unchanged-code arm names and historical evidence schedules retain their
compatibility. The separate frontend runner below rotates three arms so each
occupies every position over three blocks.

Cleanup currently supports **mapped descriptors only**. A native descriptor-heap
request fails before compilation/cache lookup; validation is not skipped to
accommodate an unqualified layout. The optional CMake setting
`METALLIC_ENABLE_SPIRV_TOOLS` defaults to `ON` and discovers installed optimizer
libraries for `MetallicShaderWarmupCore`. Discovery does not enable a transform.
A build without the optional SPIRV-Tools dependency supports `none` and
explicitly rejects either requested cleanup before cache lookup. Local
optimizer/final-validation failure rejects the result and does not publish or
cache substituted original code. The pinned SDK's actual API forwarding is
covered by a resolved-branch O1 probe and the current Guides byte identity;
these checks are not a general guarantee for arbitrary `SLANG_ROOT` installs.

Captured PSO recipes already contain finalized bytecode. Replay their recorded
code and policy without supplying a new `--spirv-postprocess` override.

## Metrics and cache identity

Per-call `compilation_phases` now includes:

| Field | Measured meaning |
| --- | --- |
| `postprocess_ms` | `dead-branches`: separate optimizer setup, input validation, both transforms and final validation. `slang-dead-branches`: final validation only; its transforms remain in `code_generation_ms`. `none`: zero |
| `raw_spirv_bytes` | Slang's returned blob before Metallic target finalization. For `slang-dead-branches`, this already includes the appended transforms. `0` on a cache hit means unavailable, not an empty shader |
| `output_spirv_bytes` | Size of the returned finalized code, including cleanup when selected; also populated on cache hits |

`code_generation_ms` remains Slang's API interval, including its own IR and
downstream work. `target_finalize_ms` measures the existing preceding target
finalization. `total_ms` and `compile_wall_ms` include cleanup on a miss; SDK
compiler elapsed counters overlap existing phases and must not be added again.
The returned bytecode is still distinct from any later device-specific RHI
rewrites, which the Benchmark records separately. None of these byte counts is
Driver ISA size.

The `none` mode performs precisely the original request-hash operations and
preserves existing shader-cache identities. The existing `dead-branches` key
also remains unchanged: its suffix identifies the pass order and Vulkan 1.4
scalar-layout validation, revision `1`, and the linked
`spvSoftwareVersionDetailsString()`. `slang-dead-branches` has a distinct suffix
and revision `1`, plus the runtime `spGetBuildTagString()` and its mode-specific
tool version. Each extends the existing compiler/options identity and retains
source-dependency validation. Original, separate-cleanup and Slang-cleanup
requests therefore have separate cache entries even when emitted bytes agree.
Hits return stored transformed code without rerunning compilation/cleanup;
compiler phase counters, `postprocess_ms` and raw byte count remain zero.

`slangShaderPostprocessToolVersion(mode)` preserves the old no-argument result
through its default `DeadBranches` parameter. The new mode reports
`Slang 2026.18.2 downstream spirv-opt + <linked SPIRV-Tools version> validator`.
The linked version identifies Metallic's final validator, not Slang's bundled
optimizer. Experiment inputs also retain the actual executable/DLL hashes.

Benchmark reports record requested/effective cleanup policy and the corresponding
tool version. Runtime logs and recipe shader metadata also record the effective
policy; both cleanup modes require a nonempty, available tool version on recipe
export/import. The runner rejects missing or conflicting cleanup confirmation,
tool-version drift, and cache-hit metrics
that claim new cleanup work. Source/device SPIR-V identity maps are separate
for every Slang level and postprocess mode. Native object reuse and driver
binary/application caches continue to use their existing distinct layers.
When identical bytecode has several source producers, recipes record
`source_identity_ambiguous=true`; concrete source fields then identify one
candidate rather than the current request uniquely. Exact-bytecode replay
remains possible, but the experiment runner rejects ambiguous provenance.

## Current rendering qualification

The sealed [summary](../build-release/shader-cleanup-correctness-20261013/summary.json),
[workload](../build-release/shader-cleanup-correctness-20261013/Cases.json) and
[manifest](../build-release/shader-cleanup-correctness-20261013/Manifest.json)
record one process per arm. Both processes used the Release RHI harness,
NVIDIA GeForce RTX 5060 / driver 616.92, O1, mapped descriptors, shader debug
disabled, normal Pipeline creation and Vulkan validation enabled. The
controllable internal Driver cache was enabled, and runtime application caches
were private to this experiment. This is a correctness batch, not a cold Driver
or ordinary GPU timing comparison.

All cases used 128 x 128, depth 12, SHARC off, accumulation off and the same
studio HDR environment at intensity 1 and rotation 25 degrees. `Opaque1` and
`Mixed1` used one sample; `Sky` used two samples and a camera facing away from
the mixed geometry. The opaque fixture has a nonmetallic untextured material.
Mixed includes a normal map, MASK/BLEND coverage, textured transmission and
two different IOR/absorbing-volume materials.

The fixed workload warms frames 0–3 and records timing frames 4–7. HDR captures
cover frames 8–15, including a camera change at frame 11. The seven guide outputs
are captured individually at frames 16–22, with a 0.02-unit camera step on X
between guide frames; they are not seven simultaneous same-frame attachments.
The final HDR artifact equals its frame-15 capture in each arm and is therefore
a repeated terminal-frame check.

The original absolute/relative tolerance remained `2e-5` for both. Independent
readback recomputation matched the summary exactly:

| Case | Maximum HDR error over frames 8–15 | Final HDR maximum error | Seven guides |
| --- | ---: | ---: | --- |
| Opaque1 | `4.76837158203125e-7` | `4.76837158203125e-7` | Byte-identical |
| Mixed1 | `9.179115295410156e-6` at frame 14 | `1.6689300537109375e-6` | Byte-identical |
| Sky | `0` | `0` | Byte-identical |

All 48 candidate-versus-baseline comparisons pass their predeclared tolerance: 24 temporal HDR
captures, three repeated final-HDR checks and 21 guide readbacks. All 21 guide
files and all Sky HDR files are byte-identical. Opaque/Mixed HDR has small
floating-point differences; it must not be reported as zero difference.
The guides cover albedo, specular albedo, normal/roughness, motion vectors,
linear depth, specular hit distance and depth. Raw artifacts remain authoritative:
[original report](../build-release/shader-cleanup-correctness-20261013/block-00-O1-pipeline-normal/MaterialBaseline.json)
and [cleanup report](../build-release/shader-cleanup-correctness-20261013/block-00-O1-pipeline-normal-dead-branches/MaterialBaseline.json).

This batch does not qualify Fiber, custom Slab programs, SHARC Update/Query,
native descriptor heaps, full-scene memory behavior or long temporal sequences.
Its short timing window with validation enabled cannot establish a GPU speedup.

## Actual current Guides bytecode

The raw recipe files were read and their byte counts/SHA-256 values recomputed.
Both variants use `OpenPBRRayQueryPathTraceGuides.openPbrRayQueryPathTraceGuidesMain`.
The single-sample variant is used by Opaque1/Mixed1; Sky selects the general
multisample variant. Graph properties, the production
`METALLIC_PT_SINGLE_SAMPLE` selection and per-graph PSO logs establish that mapping.
The optimizer was `SPIRV-Tools v2026.4 v2026.4.rc2-0-gef96ed76`.

| Variant | Original bytes | Cleanup bytes | Byte reduction | Original FNV → cleanup FNV |
| --- | ---: | ---: | ---: | --- |
| Single sample | 7,027,216 | 3,395,020 | 51.69% | `21e59a3ab818ac4f` → `f610841e8cd9a663` |
| General multisample | 6,947,064 | 3,314,908 | 52.28% | `2d09e5ccab53bd72` → `16a583307413a7e4` |

| Variant / mode | SHA-256 |
| --- | --- |
| Single / none | `25664d7e654d91dad70d9de87bcf288f498f5e742e8eb6657771af3610c3638b` |
| Single / cleanup | `c025d68c6fad0c84fac5d8ec5648b2158af5b12aae87d960d05bff06fb0bf197` |
| General / none | `645602e7448196a055ccc8a5e7f33a42d22a8d1031a303d9e54e856cb299d270` |
| General / cleanup | `ca442308a2b31c7b8d06534ad530c8fab95955ecd37de211d7d64ce382f6c4ba` |

Byte reduction alone does not establish fewer executed instructions or improved
GPU dispatch. The current measurements below retain prefilled-SPIR-V and
uncached-source scenarios as distinct scopes.

## Current compilation and GPU measurements

The sealed [qualification](../build-release/shader-cleanup-qualification-20261013/Qualification.json)
and its [recomputation script](../build-release/shader-cleanup-qualification-20261013/Qualify.py)
verify the raw reports, bytecode/settings and artifact inventories. They use the
same RTX 5060 / 616.92 device, mapped O1 Guides request 146, normal Pipeline
creation, disabled debug and disabled validation for timing. The Driver cache
UUID is `e686062a10e5dfb6adcf90d3a502d334`.

### Native creation with prefilled SPIR-V

Each measured process loaded the fixed original or cleaned SPIR-V, bypassed
native-object reuse, used an empty application pipeline cache and requested
`METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`. All samples reported valid
feedback, a native cache miss and no metadata hit. This does not prove every
Driver compiler or OS state was cold.

| Scope, 3 processes per arm | O1 none median [min, max] | O1 cleanup median [min, max] |
| --- | ---: | ---: |
| `native_create_ms` | 74.253 s [61.024, 74.426] | 60.842 s [55.876, 65.330] |
| Feedback `driver_ms` | 74.253 s [61.024, 74.426] | 60.842 s [55.876, 65.330] |
| `cache_lock_wait_ms` | 0.0009 ms | 0.0008 ms |

The three **paired** cleanup/original ratios are 0.879836, 0.817484 and
0.915640: median 0.879836, or **12.02% less native creation time**. Do not
substitute the ratio of independent arm medians, which is a different
estimator. The range shows substantial process variability; these are small
samples for this workload, not a universal speedup.

The original [three-pair run](../build-release/shader-cleanup-native-20261013/summary.json)
retains all samples. Its block 1 overlapped a Python structural regression test
and is excluded from the headline calculation. A separate
[BA replacement pair](../build-release/shader-cleanup-native-recheck-20261013/summary.json)
retained identical device, source and final-device bytecode identities. The
selected chronological order is original block 0 AB, original block 2 AB,
then replacement BA; it is not presented as the original planned AB/BA/AB
schedule. Lock waiting in these serial requests says nothing about contention
in shared-cache concurrent creation.

### Uncached source compilation

Six separate compile-only processes used disabled shader caching and alternating
AB/BA/AB order, producing exactly the bytecode used by the native comparison.
No device or Pipeline creation was included.

| Scope, 3 processes per arm | O1 none median [min, max] | O1 cleanup median [min, max] |
| --- | ---: | ---: |
| `compile_wall_ms` | 9.856 s [9.271, 10.131] | 17.953 s [17.406, 18.653] |
| `postprocess_ms` | 0 | 8.171 s [8.000, 8.315] |

Evidence: [frontend reports and metrics](../build-release/shader-cleanup-frontend-20261013/summary.json).
The additional compilation cost consumes much of the native benefit when both
source and Driver work are uncached. These separately measured scopes are not
an end-to-end startup measurement; do not claim an overall cold-start speedup
by adding their independent medians. Prefilled SPIR-V avoids repeating cleanup.

### Production GPU work

The [GPU batch](../build-release/shader-cleanup-gpu-20261013/summary.json) ran
OpaquePerf and MixedPerf at 512 x 512, two samples/frame, depth 12 and SHARC off.
Each of six independent processes warmed 32 frames and measured 64, in
AB/BA/AB order. Compilation, upload, readback, denoising and presentation are
excluded from the reported GPU scopes. Ordinary Driver caching was enabled;
this batch does not measure cold creation. Clocks were unaltered, and exclusive
ownership of the desktop GPU is not certified.

| Workload | PathTrace pass paired-median change | Shading dispatch paired-median change |
| --- | ---: | ---: |
| OpaquePerf | -0.48% | -0.51% |
| MixedPerf | -0.33% | -0.26% |

These small changes, including individual positive and negative pairs, do not
establish a GPU execution improvement or a systematic regression. All requested
HDR/guide comparisons passed the unchanged `2e-5` absolute/relative tolerance;
the largest absolute error in this batch was `1.722574234008789e-5`. The separate
moving-camera validation batch above remains the temporal correctness evidence.

### Slang downstream API measurements

The pinned compiler accepted an O1 command with two additive arguments:

```text
-Xspirv-opt --eliminate-dead-branches
-Xspirv-opt --simplify-instructions
```

One frozen-source CLI diagnostic took **10.742 s**, followed by **2.090 s**
of separate Vulkan 1.4 scalar-layout validation. Its output was byte-identical
to the qualified current multisample cleanup result: 3,314,908 bytes, SHA-256
`ca442308a2b31c7b8d06534ad530c8fab95955ecd37de211d7d64ce382f6c4ba`.
Evidence: [command, logs, identities and comparison](../build-release/link-time-current-20261013/downstream-cleanup-probe/summary.json).
This single CLI diagnostic led to the opt-in `SlangDeadBranches` runtime API
path described above. Its timing remains separate from the API experiment.

The sealed [runtime frontend batch](../build-release/slang-cleanup-api-frontend-20261013/summary.json)
contains nine independent compile-only processes for current request 146:
three per mode, mapped O1, debug/validation disabled. Every measured process
uses `--shader-cache reuse` with a newly created private directory and reports
a cache miss with `compiler_executed=true`. This includes cache writing; it is
a **fresh shader-cache miss**, not the earlier disabled-cache experiment.
No device, Driver creation or GPU execution occurs in this scope. The recorded
block orders are none/external/Slang, external/Slang/none, Slang/none/external.

| Mode, 3 processes | `compile_wall_ms` median [min, max], s | `code_generation_ms` median, s | `postprocess_ms` median, s |
| --- | ---: | ---: | ---: |
| `none` | 10.214 [10.155, 11.160] | 9.223 | 0 |
| `dead-branches` | 20.609 [19.442, 20.857] | 10.220 | 9.182 |
| `slang-dead-branches` | 14.002 [12.103, 14.319] | 10.560 | 2.391 |

All six cleanup samples emit the same 3,314,908 bytes and SHA-256
`ca442308a2b31c7b8d06534ad530c8fab95955ecd37de211d7d64ce382f6c4ba`.
The integrated path has lower measured compiler wall time than the separate
path, but cleanup still adds source-compilation time relative to `none`.
Compare the complete compiler wall interval: subtracting the new `postprocess_ms` alone
would omit its appended passes inside code generation. These independently
measured frontend results do not prove net cold startup or Driver savings.

A [schema-v2 corpus probe](../build-release/slang-cleanup-api-corpus-20261013/summary.json)
uses the same fresh-cache scope with one process per mode for each request.
Its timings are exploratory single samples, with no repeated estimate:

| Current request | `none` / external / Slang compiler wall, s | Original / cleanup bytes | External versus Slang output |
| --- | ---: | ---: | --- |
| 137: `SceneRTXDI.sceneRtxdiMain` | 3.637 / 4.727 / 3.824 | 1,358,548 / 1,307,132 | Same SHA-256 |
| 149: Guides, SHARC Update | 19.130 / 26.928 / 20.893 | 6,433,468 / 2,811,392 | Same size, different SHA-256 |
| 152: Guides, SHARC Query | 10.106 / 18.126 / 13.080 | 6,974,240 / 3,342,116 | Same SHA-256 |

For request 149 the external SHA-256 is
`32033799d7908be010ebd0caaf5f8e42085bab11747b0046e1e058246e95259a`, while the
Slang path emits `55f6479149c4e5c8520af5c27761561a9ea7387f706ed60b86b112e3a57af20b`.
Equal sizes do not establish equivalent code. The affected SHARC rendering
qualification below provides a workload-specific correctness check.
The [follow-up static comparison](../build-release/slang-cleanup-update-id-analysis-20261013/summary.json)
validated both outputs and still found different bytes after `--compact-ids`.
Both contain 181,870 instructions and 22,351 basic blocks. Their opcode-count
difference is one additional `OpConstantComposite` in the external path versus
one additional `OpCompositeConstruct` in the Slang path; the equal aggregate
counts do not prove semantic equivalence. The corpus probe alone qualifies
neither a Driver improvement nor rendering.

### Fresh source plus native creation in one process

The [six-process cold batch](../build-release/slang-cleanup-api-cold-total-20261013/summary.json)
compares `none` with `slang-dead-branches` on current Guides request 146 in
AB/BA/AB order. Every process uses a new private shader cache and reports a
source miss, creates an empty application Pipeline cache, disables the
controllable Driver cache and reports valid native-miss feedback. Mapped O1,
normal Pipeline optimization, disabled debug/validation and the RTX 5060 /
616.92 device remain fixed. Final device SPIR-V equals the qualified frontend
bytes for each mode; neither native objects nor native binary imports are reused.

| Scope, 3 processes per mode | `none` median [min, max], s | `slang-dead-branches` median [min, max], s |
| --- | ---: | ---: |
| Source `compile_wall_ms` | 10.904 [10.632, 11.754] | 15.656 [13.618, 16.758] |
| `native_create_ms` | 67.128 [66.751, 69.136] | 57.706 [56.313, 57.895] |
| Compile + module + Pipeline load/create | 77.787 [77.681, 80.915] | 71.983 [71.338, 74.667] |
| Whole Bench process wall | 78.292 [78.243, 81.535] | 72.549 [71.827, 75.113] |

The compile/module/load/create subtotal uses nonoverlapping wall intervals
returned by the **same process**, and excludes device initialization, cache
saving and teardown. Its three paired ratios are 0.922782, 0.918340 and
0.925382: median **0.922782**, or **7.72% less time**. Whole-process paired
ratios have median 0.921231 (**7.88% less time**), including device initialization
and teardown. These scopes measure one Bench request, not complete renderer
startup, and do not prove fully cold OS or Driver compiler state.

The [independent qualification](../build-release/slang-cleanup-api-cold-qualification-20261013/Qualification.json)
and [verifier](../build-release/slang-cleanup-api-cold-qualification-20261013/VerifySlangCleanupCold.py)
recompute the artifact inventories, six distinct processes, exact frontend
identities, source-dependency hashes, per-mode device bytecode/PSO identity,
actual empty-cache preparation, native misses and paired aggregates. The
Slang-integrated cleanup's extra compilation cost is included in the observed
net reduction. Default compilation remains `none`; qualification is specific
to this workload and pinned toolchain.

### SHARC rendering qualification for both cleanup paths

The [validation-on batch](../build-release/slang-cleanup-api-sharc-correctness-20261013/summary.json)
ran the existing MixedSharc fixture once per mode (`none`, external cleanup,
Slang cleanup), with mapped O1, shader debug disabled and normal Pipeline
optimization. The 128 x 128 workload uses two samples/frame, depth 12,
SHARC enabled, 16 main frames and a camera change at frame 11. Each mode
compares final HDR, eight temporal HDR captures at frames 8–15 and seven
subsequent guide captures against the original. All processes passed Vulkan
validation and the unchanged `2e-5` absolute/relative readback threshold.

Both cleanup paths have maximum absolute HDR error
`5.46872615814209e-6` versus the original, and all seven guide comparisons are
numerically exact. Direct external-versus-Slang comparison is numerically
exact for all 16 readbacks, including the moving-camera captures. The
[independent qualification](../build-release/slang-cleanup-api-sharc-qualification-20261013/Qualification.json)
binds all nine relevant runtime recipes (requests 146/149/152 in three modes)
to the actual byte hashes from the frontend/corpus analysis. Thus the Update
shader with the documented composite-constant representation difference was
exercised in this scene. This is a fixed-scene correctness check, not a proof
of semantic equivalence for every input.

The [fixed-exposure HDR preview](../build-release/slang-cleanup-api-sharc-preview-20261013/HDRPreview.png)
was visually inspected at frames 8 and 15; geometry, shading and camera motion
agree across the three modes. Ordinary Driver caching was enabled, and the
Slang arm reused the external arm's identical base/query Pipeline cache data.
Do not interpret these creation logs or validation-on GPU timestamps as cold
Driver or dispatch performance comparisons. Fiber/custom-Slab and native-heap
coverage remain outside this batch; the default stays `none`.

## Reproduce

See [Compute pipeline experiments](ComputePipelineExperiments.md) for separate
frontend, native and dispatch commands. Reproduce fresh-cache frontend costs
without including Pipeline creation:

```powershell
python -B Tools/Perf/MeasureShaderPostprocess.py run `
  --executable .\build-release\Source\MetallicShaderPipelineBench.exe `
  --request-indices 146 --postprocesses none dead-branches slang-dead-branches `
  --pairs 3 --output-dir build-release/cleanup-frontend-<fresh>
python -B Tools/Perf/MeasureShaderPostprocess.py verify `
  build-release/cleanup-frontend-<fresh>
```

Current schema v2 first performs one excluded dependency-discovery compilation
per request. It freezes all compiler-reported dependency paths/content hashes,
including imported vendor headers and C++/Slang interop, before measuring that
request. Sources, executable/DLLs and imported profiling helpers are hashed;
checks before/after each measured process reject changes or new dependencies.
Verification requires one successful discovery per request, frozen dependency
coverage, the recorded schedule, fresh misses, effective O1/mapped settings,
disabled debug/validation, successful processes and exact cache/output bytes.
It also binds compiler identity within each mode and request/SDK/target/ABI
identity across modes/discovery, rejecting nonfinite or invalid timing values.
The completed nine-sample Guides batch retains its original schema-v1 runner
and excluded discovery evidence; v2 strengthens helper snapshots and
independent dependency verification without rewriting that historical batch.

Verify the completed external-cleanup correctness batch without rerunning its
GPU work:

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py verify `
  build-release/shader-cleanup-correctness-20261013
```

The implementation and evidence checks are covered by compiler/cache tests,
`tests/perf/TestComputePipelineExperiments.py` and
`tests/perf/TestMeasureShaderPostprocess.py`. Rendering and performance
qualification remain workload-specific; keep generated outputs outside source
control and use a fresh directory for every experiment.
