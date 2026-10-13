# Opt-in SPIR-V dead-branch cleanup

The 2026-10-13 implementation adds an optional in-process SPIR-V cleanup after
Slang compilation. The production default remains **none**. The current
correctness experiment qualifies this transformation for its three fixed
mapped-descriptor workloads. Current independent-process measurements show a
Driver creation benefit, substantial extra compiler cost, and approximately
unchanged GPU execution. These results use the current owned Slang modules,
not the earlier 2026-10-12 source snapshot.

## Compiler path and controls

[`SlangCompiler.cpp`](../Source/Runtime/Render/Core/SlangCompiler.cpp) runs the
following sequence on a cache miss:

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

`SlangShaderDesc::postprocessMode` exposes `Default`, `None` and `DeadBranches`.
An explicit mode overrides the process environment. `Default` resolves
`METALLIC_SPIRV_POSTPROCESS`: unset, empty, `default` or `none` selects `None`;
`dead-branches` selects the cleanup. Invalid values fail explicitly.

The Benchmark exposes `--spirv-postprocess default|none|dead-branches`. The
experiment runner exposes `--postprocesses none dead-branches`, defaults to
`none`, and sets both the explicit Benchmark argument and process environment.
The two-arm cleanup comparison alternates AB, BA, AB across independent-process
blocks. Existing unchanged-code arm names and historical evidence schedules
retain their compatibility.

Cleanup currently supports **mapped descriptors only**. A native descriptor-heap
request fails before compilation/cache lookup; validation is not skipped to
accommodate an unqualified layout. The optional CMake setting
`METALLIC_ENABLE_SPIRV_TOOLS` defaults to `ON` and discovers installed optimizer
libraries for `MetallicShaderWarmupCore`. Discovery does not enable a transform.
A build without SPIRV-Tools-opt supports `none` and explicitly rejects requested
cleanup. Compilation, optimizer or validation failure never falls back to
uncleaned code while reporting cleanup.

Captured PSO recipes already contain finalized bytecode. Replay their recorded
code and policy without supplying a new `--spirv-postprocess` override.

## Metrics and cache identity

Per-call `compilation_phases` now includes:

| Field | Measured meaning |
| --- | --- |
| `postprocess_ms` | Optimizer setup, input validation, the two transforms and final output validation; excludes Slang generation and target finalization |
| `raw_spirv_bytes` | Slang's emitted blob before target finalization/cleanup; `0` on a cache hit means unavailable, not an empty shader |
| `output_spirv_bytes` | Size of the returned finalized code, including cleanup when selected; also populated on cache hits |

`code_generation_ms` remains Slang's API interval, including its own IR and
downstream work. `target_finalize_ms` measures the existing preceding target
finalization. `total_ms` and `compile_wall_ms` include cleanup on a miss; SDK
compiler elapsed counters overlap existing phases and must not be added again.
The returned bytecode is still distinct from any later device-specific RHI
rewrites, which the Benchmark records separately. None of these byte counts is
Driver ISA size.

The `none` mode performs precisely the original request-hash operations and
preserves existing shader-cache identities. Enabled cleanup appends its policy
name, revision `1`, and the linked `spvSoftwareVersionDetailsString()` to the
existing compiler/options/dependency identity. The policy identifies the pass
order and Vulkan 1.4 scalar-layout validation. Thus cleaned and original code
cannot occupy the same request cache entry. The cache stores the transformed
output; hits return it without running the compiler or optimizer again.
Cache-hit compiler/cleanup counters and raw byte count remain zero.

Benchmark reports record requested/effective cleanup policy and optimizer
version. Runtime logs and recipe shader metadata also record the effective
policy; cleanup recipes retain the tool version. The runner rejects missing or
conflicting cleanup confirmation, tool-version drift, and cache-hit metrics
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

### Cheaper compiler-downstream candidate

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
This is one CLI diagnostic rather than paired runtime API timing. It supports
testing the same late passes within Slang's existing optimizer invocation,
potentially avoiding a second optimization context and repeated validation.
It is not wired into the production compiler policy yet; keep the default
`none` until API, cache identity and broader workload qualification are complete.

## Reproduce

See [Compute pipeline experiments](ComputePipelineExperiments.md) for fresh
compile/dispatch commands using `--postprocesses none dead-branches`. Verify the
completed correctness batch without rerunning its GPU work:

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py verify `
  build-release/shader-cleanup-correctness-20261013
```

The implementation and evidence checks are covered by compiler/cache tests and
`tests/perf/TestComputePipelineExperiments.py`. Rendering and performance
qualification remain workload-specific; keep generated outputs outside source
control and use a fresh directory for every experiment.
