# Path-tracing material Backend Type experiment (2026-10-13)

Selecting a complete material implementation removes the two selected Fiber
paths from the current O1 Guides shader, while linking that implementation
produces almost the same code size as selecting it concretely. The matched
concrete/link outputs still differ after identical dead-branch cleanup and
compact-ID normalization. Three paired native measurements against the
uniformly cleaned **original source** observe a **16.31% paired-median
reduction** for No-Fiber. The exact five cleaned artifacts pass validation-on
GPU comparison in three fixed scenes. A direct original-source GPU batch
retains byte-identical HDR, temporal and guide outputs, but dispatch changes
are mixed: Opaque -2.60%, Mixed +13.70% and Sky -20.17%, with substantial Mixed
variation. This candidate is not qualified as a general runtime improvement.
Earlier linked-General controls are retained separately below; their timings
must not be combined with the original-source batches. Qualification remains
restricted to the stated workloads.
Production shader sources and defaults remain unchanged.

This follows the [boolean capability experiment](PathTracingLinkTimeSpecialization.md),
where resolved O1 feature gates retained their bodies. Here the experiment
selects implementations of larger functions through a Slang Type instead of
replacing small boolean-returning methods. The experiment uses the pinned
Slang 2026.18.2 SDK, mapped descriptors, O1, `CUSTOM_MATERIALS=0`, SHARC off,
disabled shader debug and unchanged runtime SPIR-V postprocessing (`none`).
Link-time Types are resolved before SPIR-V generation; this introduces no
Vulkan specialization constants or Driver-time Type selection.

## Five controls and the selection boundary

[`PathTraceBackendOverlay.py`](../Tools/Perf/PathTraceBackendOverlay.py) reads
frozen shader sources and returns owned source modifications without editing
production files. [`ValidatePathTraceBackend.py`](../Tools/Perf/ValidatePathTraceBackend.py)
copies the complete shader/include graph, applies those modifications and
audits each overlay before invoking the compiler.

| Arm | Type selection | Found-material dispatch / guide implementation |
| --- | --- | --- |
| `original-general` | Current production source | Unchanged original |
| `backend-concrete-general` | Concrete `typealias` | General |
| `backend-link-general` | Consumer `extern struct` plus one exported provider | General |
| `backend-concrete-no-fiber` | Concrete `typealias` | No-Fiber |
| `backend-link-no-fiber` | Consumer `extern struct` plus one exported provider | No-Fiber |

The new `IScenePathTraceMaterialBackend` boundary selects both
`shadeFoundMaterial` and `evaluateGuides`. The former owns the complete
found-material tail: full material loading, Fiber dispatch, coverage/alpha
handling, conditional Slab handling and the existing OpenPBR fallback. The
common vertex prefix stays outside selection, preserving miss handling, ray
cone propagation, volume attenuation and throughput termination. The second
method selects the independent surface-guide evaluation used by the Guides
pass. The General implementations copy the original bodies; No-Fiber removes
exactly the found-vertex Fiber branch and the independent Fiber-guide branch.

The overlay retains function signatures and `inout` state, including RNG,
interior stack, depth, throughput and guide state. It retains the existing TBN,
material resource access, report macros and conditional Slab/SHARC declarations.
The registered compute module, entrypoint and `[numthreads(8, 8, 1)]` are
unchanged. These are source-preservation checks, not a claim of identical GPU
results after code generation.

Generated `PathTraceMaterialBackendExperiment.slang` declares
`implementing OpenPBRPathTracing;` and is included only by its owning module.
Each linked provider declares its own module, imports `OpenPBRPathTracing` and
exports one qualified `ScenePathTraceMaterialBackend` alias. The two linked
arms have identical consumer sources and differ only in their selected
provider. Providers are compiled and supplied explicitly; an extra search
path alone does not choose an exported implementation.

Compare concrete and linked arms with the **same** feature set to assess
link-time selection. Compare General and No-Fiber to assess the explicit scene
restriction. The latter cannot establish a benefit from the linker itself.

## Frozen baseline and compiler stages

The [preparation](../build-release/path-backend-type-20261013/preparation/Preparation.json)
and [reference report](../build-release/path-backend-type-20261013/preparation/ReferenceReport.json)
retain the selected request, source/tool/SDK hashes, generated-source hashes,
provider bindings, exact commands and five successful ownership audits.
Current catalog request 146 is
`Features/PathTracing/OpenPBRRayQueryPathTraceGuides.openPbrRayQueryPathTraceGuidesMain`,
profile `spirv_1_6`. Its fresh-cache Release API compilation took 11.090 s and
returned 6,947,064 bytes, FNV `2d09e5ccab53bd72`, SHA-256
`645602e7448196a055ccc8a5e7f33a42d22a8d1031a303d9e54e856cb299d270`.
This is a compile-only shader-cache miss, with no Device or Pipeline creation.

The [sealed compiler run](../build-release/path-backend-type-20261013/compile/summary.json)
contains one exploratory sample per arm, in the table's order. Each stage is
a separate CLI process. Original direct-source generation and original
serialized-module linking both match the API reference byte for byte.

| Arm | Consumer source-to-IR, s | Provider source-to-IR, s | IR-to-SPIR-V link/generation, s |
| --- | ---: | ---: | ---: |
| Original General | 0.919 | — | 9.034 |
| Concrete General | 0.906 | — | 9.172 |
| Linked General | 0.921 | 0.915 | 9.671 |
| Concrete No-Fiber | 0.920 | — | 9.745 |
| Linked No-Fiber | 1.026 | 0.988 | 9.297 |

Original direct-source CLI generation took a separate 9.026 s. It is a baseline
identity check, not an additional cost required by the serialized workflow.
The commands retain CLI detailed-performance reporting flags. These are single
CLI wall samples, not production API `link_ms`, repeated timing estimates,
GPU timings or startup measurements. Imported modules remain frozen source
and may be parsed again during linking. Serializing Consumer/Provider IR is
not a complete dependency-IR cache; the Provider compilation also imports its
owner graph.

The measured rows do not demonstrate faster frontend compilation from either
link-time selection or the No-Fiber restriction. Any later reuse experiment
must record exactly which IR was reused and keep initial precompile costs
separate from regeneration costs.

## Raw and uniformly cleaned SPIR-V

The [validated raw analysis](../build-release/path-backend-type-20261013/analysis/Analysis.json)
and [uniform cleanup analysis](../build-release/path-backend-cleanup-analysis-20261013/summary.json)
retain exact hashes, structural counts and comparisons. Cleanup applies only
`--eliminate-dead-branches` then `--simplify-instructions`, under Vulkan 1.4 with
scalar block layout, to **all five** original inputs. The SDK optimizer was
`SPIRV-Tools v2026.4 v2026.4.rc2-0-gef96ed76`. Raw/cleaned results and their
compact-ID counterparts were validated. Compact IDs only renumber identifiers.

| Arm | Raw bytes | Raw instructions | Raw basic blocks |
| --- | ---: | ---: | ---: |
| Original General | 6,947,064 | 449,813 | 60,501 |
| Concrete General | 6,947,524 | 449,854 | 60,510 |
| Linked General | 6,947,364 | 449,835 | 60,501 |
| Concrete No-Fiber | 6,442,996 | 416,201 | 55,232 |
| Linked No-Fiber | 6,442,836 | 416,182 | 55,223 |

| Arm | Cleaned bytes | Cleaned instructions | Cleaned basic blocks |
| --- | ---: | ---: | ---: |
| Original General | 3,314,908 | 215,311 | 28,217 |
| Concrete General | 3,315,368 | 215,352 | 28,226 |
| Linked General | 3,315,208 | 215,333 | 28,217 |
| Concrete No-Fiber | 2,844,440 | 184,173 | 23,498 |
| Linked No-Fiber | 2,844,280 | 184,154 | 23,489 |

Every row has one function and zero function calls after O1 generation. All
five raw outputs contain zero Vulkan specialization-constant instructions.
Compared with its matching General arm, No-Fiber removes 504,528 raw bytes
and 33,653 instructions for both selection mechanisms. The same scene
restriction therefore supplies the large reduction; linking supplies a
160-byte difference between matched implementations. Byte/instruction counts
are SPIR-V structure, not Driver ISA, executed instruction counts or GPU speed.

For both feature sets, concrete minus linked code contains one extra
`OpConstant`, nine extra `OpLabel` and nine extra `OpBranch` instructions:
19 instructions and 160 bytes. This difference survives the identical
two-pass cleanup and compact-ID normalization. The transformed General arms
also differ from production General; wrapping the dispatch adds control/load/
store scaffolding. The experiment does not establish stronger general-purpose
dead-code elimination from link-time Types.

## Separate merge-block diagnostic

The [merge-block diagnostic](../build-release/path-backend-cleanup-analysis-20261013/diagnostic-merge-blocks/summary.json)
adds `--merge-blocks` and then `--compact-ids` to the already cleaned outputs.
It is explicitly excluded from cleanup-policy and native-measurement inputs.
Its extra transform must not be attributed to the runtime two-pass policy.

Merging removes the matched concrete/link nine-label/nine-branch count
difference. General then has 24,513 blocks in both arms; No-Fiber has 20,389
blocks in both. Concrete still has one additional constant: merged General
is 3,255,960 bytes / 207,926 instructions versus linked 3,255,944 / 207,925;
merged No-Fiber is 2,794,696 / 177,955 versus linked 2,794,680 / 177,954.
The pairs remain byte-different after compact IDs.

The [operand inspection](../build-release/path-backend-cleanup-analysis-20261013/diagnostic-merge-blocks/OperandEvidence.json)
identifies that remaining constant as signed int32 literal `7`. In the concrete
General output it has 11 uses excluding its definition, including 10 stores;
concrete No-Fiber has two uses, including one store. The linked counterparts
do not contain that literal. These are actual operand uses, including stores,
not an established unused declaration or pure ID-renumbering difference.
Equal block counts and nearly equal instruction counts do not prove semantic
or numerical equivalence.

## Scene eligibility and GPU qualification

No-Fiber requires scene-wide eligibility across every reachable material and
streamed asset, including indirect hits. Restricting only the currently visible
objects is insufficient. Existing shadow traversal can still reference Fiber
code, so this experiment does not remove Fiber globally. Translucency and
metallic paths remain enabled; sheen/coat and dispersion retain their original
baseline values. Conditional Slab code remains in source, although the initial
reference deliberately selects `CUSTOM_MATERIALS=0`; SHARC variants are outside
this qualification.

The [five-process artifact rendering batch](../build-release/path-backend-rendering2-20261013/summary.json)
passed on RTX 5060 / NVIDIA 616.92 with validation enabled and shader debug
disabled. Each arm rendered Opaque1, Mixed1 and Sky at 128 x 128, two samples
per frame and depth 12, with SHARC off, `CUSTOM_MATERIALS=0`, mapped descriptors
and the pass camera (`GLOBAL_VIEW=0`). The frame-11 camera change and subsequent
guide camera steps are identical across arms. The loaded material sets are
inspected before device initialization, including unreferenced materials.

Each scene contributes final HDR, eight temporal HDR frames and seven guides:
48 captured arrays per arm, 240 across all five arms. All four transformed arms
match original General with **zero maximum absolute error**, and both direct
static/link comparisons (General and No-Fiber) are also numerically exact.
Fifteen independently captured target PSO recipes, one per arm/scene, confirm
the exact artifact SHA256/FNV, source tag, native entry, descriptor ABI, 12-byte
push data and pipeline-state hash. The reported source identity is unambiguous;
every case consumes the artifact and reports zero rejected target requests.
Logs contain no validation error or error-material fallback.

The predeclared tolerance remains absolute `2e-5` plus relative `2e-5`; exactness
is an observed result, not a tightened post-hoc threshold. Numerical identity
does not imply identical SPIR-V or Driver ISA. Preview inspection confirms
rendered geometry, mixed materials and the environment, without an error
material; the two-sample geometry images retain expected Monte Carlo noise.
Implicit scene sidecars are disallowed and their absence is checked before and
after every process. glTF, buffers, textures, geometry caches and environment
inputs are frozen, as are the source/tool/executable and compiler/cleanup
provenance chains. The output manifest seals reports, recipes and readbacks.

Long temporal, dynamic material changes, larger scenes and full-scene memory
remain outside this fixed fixture. The validation-on process times are not
used as compilation or Dispatch speedups; separate GPU measurements follow
below. No-Fiber cannot serve
a scene that contains Fiber; custom programs and SHARC variants remain outside
qualification. Production sources and defaults remain unchanged.

### Larger validation-off GPU comparison

The [512 x 512 dispatch batch](../build-release/path-backend-dispatch-512-20261013/summary.json)
adds one validation-on qualification pair followed by three independent
validation-off General/No-Fiber pairs in AB/BA/AB order. These are the exact
linked two-pass artifacts above. Each process renders the same Opaque1,
Mixed1 and Sky scenes, with samples 2, depth 12, SHARC off, `CUSTOM=0` and the
pass camera. Frames 0-15 warm up, 16-47 supply GPU timestamps, and 48-55 supply
eight HDR captures with the camera change at frame 51. Seven moving-camera
guides follow at frames 56-62. Readback and the camera change are outside the
timing window. Clocks are unaltered; the controllable driver cache is enabled
and every process uses its own application PSO-cache directory.

All eight processes pass: **384 readback arrays are numerically identical**,
including the validation-off outputs against validation-on General. Twenty-four
per-case target recipes confirm the exact bytecode and ABI. Each case's eleven
ordinary stage identities also match across all eight processes, so differences
are confined to the target artifact. Each measured process supplies 32 valid
timestamp rows per scene, with unique execution IDs and the same queue topology.
Compilation and native creation are outside these GPU intervals.

| Path trace shading interval | General median (ms) | No-Fiber median (ms) | Paired-median change |
| --- | ---: | ---: | ---: |
| Opaque1 | 23.255568 | 22.678592 | -1.96% |
| Mixed1 | 61.684864 | 60.673200 | -1.64% |
| Sky | 0.493904 | 0.394416 | -20.20% |

The two median columns summarize three independent-process medians. The last
column is the median of the three within-block No-Fiber/General ratios, a
different estimator. Opaque ratios are 0.948546, 0.980412 and 0.983148; Mixed
ratios are 0.983599, 0.970364 and 0.987524; Sky ratios are 0.798023, 0.810522
and 0.795912. Frames are correlated samples, not 96 independent experiments.
General's Opaque process medians decline from 23.908800 through 23.255568 to
22.526880 ms (about 5.8% across the batch), exceeding the roughly 2% paired
effect. Several Mixed processes also drift by roughly 2-3% between their
first and last sixteen timed frames. All original frames are retained. The
geometry results are a small observed improvement in three pairs, not a
statistically established general gain. The Sky difference is about 0.1 ms
in absolute dispatch time.

The corresponding paired-median Graph changes are -2.03%, -1.64% and -10.93%;
Pass changes are -2.04%, -1.64% and -11.01%. Graph and Pass are broader GPU
intervals and can include GPU upload commands. The archived tool's original
`scope` text claimed uploads were excluded from all intervals; that wording
was too broad and is corrected in the current tool. Raw data and the sealed
archived verifier remain untouched. `Path trace shading` is the marked dispatch
stage, not standalone ISA execution or an end-to-end frame-rate measurement.

Per-process frame P95s remain diagnostic. Their three-process median is
24.765504 -> 24.036960 ms for Opaque, 65.056832 -> 65.416448 ms for Mixed and
0.503040 -> 0.398592 ms for Sky. The Mixed tail does not improve even though its
median does. These three pairs show a small geometry-stage improvement and a
larger sky-stage improvement for this workload, without proving a general GPU
benefit or removing the earlier scene restrictions. A performance comparison
against original-cleaned General, larger real scenes and other material/
SHARC permutations remains required before production promotion.

[`MeasurePathTraceBackendDispatch.py`](../Tools/Perf/MeasurePathTraceBackendDispatch.py)
freezes the actual verifier and import helpers inside the output. Historical
`verify` replays that archived source; `check` additionally requires current
inputs and GPU inventory to match. Sixteen before/after snapshots record source,
artifact, executable, fixture, sidecar-absence and GPU-inventory invariants.
Failed process reports and partially completed measurements are sealed as
failures. The 23 CPU evidence tests pass through the registered
`MetallicPathTraceBackendDispatchEvidenceTests` suite; the 17 existing rendering
evidence tests pass after parameterizing the common validation qualifier.
An independent audit verifies 1,369 sealed files, all sixteen snapshots,
eight distinct non-overlapping successful processes, 24 target recipes and
264 ordinary shader-stage records. It independently reconstructs PSO hashes,
checks all 384 readbacks are finite and byte-identical, and recomputes the
paired statistics and nearest-rank P95 from 576 raw validation-off timing
frames. The 384 warmup rows across all eight processes remain excluded.

```powershell
python -B Tools/Perf/MeasurePathTraceBackendDispatch.py run `
  --preparation build-release/path-backend-type-20261013/preparation `
  --compile build-release/path-backend-type-20261013/compile `
  --analysis build-release/path-backend-cleanup-analysis-20261013 `
  --reference build-release/path-backend-type-20261013/preparation/ReferenceReport.json `
  --cases build/compute-create-fixtures/Correctness.json `
  --executable build-release/tests/MetallicRHITests.exe `
  --output-dir build-release/path-backend-dispatch-512-<fresh>
python -B Tools/Perf/MeasurePathTraceBackendDispatch.py verify `
  build-release/path-backend-dispatch-512-20261013
```

### Exact artifact rendering test

[`ValidatePathTraceBackendRendering.py`](../Tools/Perf/ValidatePathTraceBackendRendering.py)
consumes the exact `Cleaned.spv` from all five arms, produced by the same two
passes above. It excludes compact-ID and merge-block diagnostic outputs. A
test-only scoped resolver is installed before preview initialization through
`METALLIC_SHADER_ARTIFACT_MANIFEST`. The loader verifies immutable bytes,
SHA-256/FNV, compute `main`, structural counts, capabilities and provenance
manifest hashes. It matches the original request's module, authored entry,
profile, search paths, macros and effective O1/mapped/debug-disabled/ACEScg
settings. A different permutation of that target fails; other shaders use
ordinary compilation with postprocessing `none`. Artifact tags identify the
actual bytecode producer rather than claiming production-source provenance.

Each independent arm runs the frozen `Opaque1`, `Mixed1` and `Sky` fixtures at
128 × 128, two samples per frame and maximum depth 12, with SHARC off and
`METALLIC_CUSTOM_MATERIALS=0`. The pass owns its camera
(`METALLIC_GLOBAL_VIEW=0`). All arms use
the same assets, lighting, RNG/frame sequence and camera changes. No-Fiber
eligibility is checked before initialization against the entire loaded material
set, including materials outside primary visibility, and the fixed graph's
scene references. Fiber or custom material programs reject qualification.
This is a static check of the frozen fixture, not a guarantee for assets or
materials introduced later at runtime.

Each case writes **16 readback files**: eight frame-indexed HDR captures
(frames 8–15), seven guide outputs and a separately stored final HDR image.
The guides are read after the timing window. Each case captures its successful
PSO recipes in a fresh directory; the verifier checks actual shader bytes,
source tags, device/descriptor ABI and PSO hashes. Per-case consumption counts,
target rejection/error records and early failures persist in
`MaterialBaseline.json`. An unused artifact or ErrorMaterial fallback fails
qualification. Vulkan validation is enabled throughout; the passed results
provide correctness evidence, **not a GPU performance conclusion**. The
numerical results and fixed qualification scope are recorded above.

## Native creation cost

The [six-process native batch](../build-release/path-backend-native-20261013/summary.json)
replays the exact two-pass-cleaned linked General/No-Fiber outputs, with three
AB/BA/AB pairs. All runs use RTX 5060 / NVIDIA 616.92, cache UUID
`e686062a10e5dfb6adcf90d3a502d334`, mapped descriptors, normal Driver optimization,
12-byte user push data, disabled shader debug/validation, an empty application
Pipeline cache and `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`. Every call
reports valid native-miss feedback, no metadata hit, no native-object reuse and
no native-binary import. Slang compilation, optimization and Dispatch are absent.

| Scope, 3 processes per arm | Linked General median [min, max] | Linked No-Fiber median [min, max] |
| --- | ---: | ---: |
| `native_create_ms` | 65.188 s [64.396, 65.978] | 49.794 s [47.609, 50.121] |
| Feedback `driver_ms` | 65.188 s [64.396, 65.978] | 49.794 s [47.609, 50.121] |
| `cache_lock_wait_ms` | 0.0006 ms | 0.0008 ms |

The three native No-Fiber/General paired ratios are 0.759659, 0.763851 and
0.739318: median **0.759659**, or **24.03% less native creation time**. This
estimator differs from the ratio of independent arm medians. Serial near-zero
lock waiting does not establish anything about concurrent cache contention.
These small samples show the observed cost for this fixed program and Driver;
the controllable cache override does not prove fully cold OS/compiler state.

The [measurement and independent verifier](../build-release/MeasurePathBackendNative.py)
reproduce the original captured PSO hash before generating recipes, preserve
the complete state/device recipe, archive provider/analysis/tool provenance and
check frozen input hashes before/after every process. Verification re-reads all
six raw reports, exact commands, per-call metrics, source/final-device SPIR-V
identities, native misses and paired aggregates. The recipe module tags identify
experimental artifacts rather than presenting them as production source.
Both clean inputs remain distinct from the extra merge-block diagnostics.

Only the native creation window is measured. Device initialization, recipe
loading, shader-module preparation, cache saving and whole-process wall time
are excluded. There is no GPU execution or complete renderer startup claim.
The smaller bytecode helps Driver creation in this experiment, while the
single compiler samples above do not establish a frontend benefit. Rendering
qualification and the scene restriction remain required before adoption.

## Original-source native control

The [original-control batch](../build-release/path-backend-native-original-20261013/summary.json)
compares uniformly cleaned production General directly with linked No-Fiber,
instead of using the wrapped linked General as the denominator. It uses six
fresh processes in AB/BA/AB order on the same RTX 5060 / NVIDIA 616.92, with
normal Compute Pipeline creation, empty application caches, the controllable
Driver cache disabled, and debug/validation disabled. Both inputs are the
same previously qualified two-pass artifacts. Target frontend compilation,
object reuse and native binary import/export are absent; every creation has
valid feedback and reports a native miss and metadata miss.

| Pair and order | Original General native, s | Linked No-Fiber native, s | No-Fiber reduction |
| --- | ---: | ---: | ---: |
| 0: General / No-Fiber | 62.365 | 39.658 | 36.41% |
| 1: No-Fiber / General | 48.986 | 40.998 | 16.31% |
| 2: General / No-Fiber | 49.627 | 43.157 | 13.04% |

The median paired No-Fiber/General ratio is **0.836942**, or **16.31% less
native creation time**. Arm medians are 49.627 s and 40.998 s; their ratio is
a different statistic and is not the paired result. Driver-duration ratios
agree to six decimal places, and native minus Driver duration is only
0.027--0.039 ms per sample. Cache-lock waits are 0.0007--0.0011 ms.

The original arm varies by 13.379 s across processes. These three pairs are
exploratory observations, not a statistical confidence estimate. Disabling
the controllable cache does not prove all Driver/compiler state is cold.
Do not combine this batch with the earlier linked-General batch to infer
wrapping overhead: their process state and execution times differ.

Relative to original General, the selected No-Fiber artifact removes
14.20% of bytes, 14.47% of instructions and 16.76% of basic blocks after the
same cleanup. This reduction includes both implementation selection and the
wrapper change; matched static/link controls above still isolate the much
smaller effect of the linking mechanism itself. This batch measures only
`vkCreateComputePipelines`, excluding frontend cost, initialization and GPU
execution.

[`MeasurePathTraceBackendNative.py`](../Tools/Perf/MeasurePathTraceBackendNative.py)
reconstructs the captured v6 Compute PSO ABI and checks final device bytecode,
recipe state, returned per-call metrics, batch counts and sums, cache controls,
process identities, nonoverlapping intervals and paired aggregates. It seals
failed runs without admitting them as samples and archives the actual helper
sources and source/compile/cleanup provenance for historical verification.
Its 17 mocked CPU tests passed; they do not substitute for native execution.

## Original-source GPU control

The [direct original-source batch](../build-release/path-backend-dispatch-original-512-20261013/summary.json)
selects `original-general` as the General arm. Both arms use identical two-pass
cleanup; it launches two validation-on qualification processes followed by
six validation-off timing processes in AB/BA/AB order. Settings match the
earlier 512x512 batch: two samples, depth 12, SHARC off, `CUSTOM_MATERIALS=0`,
`METALLIC_GLOBAL_VIEW=0`, 16 warmup frames, 32 timing frames and eight later HDR
captures per scene, with seven guide captures after the camera change. Driver
internal caching is enabled and each process has a private application cache.
Neither frontend nor native creation is part of the GPU timestamp intervals.

All eight processes and all 24 scene cases pass. The 384 readback arrays are
finite and byte-identical to their original-General qualification reference.
The target recipes prove the exact input artifacts were consumed, while all
264 ordinary shader-stage identities match the reference; no fallback or
validation error is admitted. The statistical unit remains one process median
per arm/pair, with 576 timed frames across the six performance processes.

| Scene | Original dispatch median, ms | No-Fiber dispatch median, ms | Paired dispatch change | Paired Pass change | Paired Graph change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opaque | 29.412992 | 28.929600 | -2.60% | -2.64% | -2.66% |
| Mixed | 68.644080 | 76.880288 | +13.70% | +13.70% | +13.66% |
| Sky | 0.499376 | 0.400096 | -20.17% | -7.23% | -6.69% |

The arm columns are medians of three process medians. Percentage changes use
the median of three within-pair candidate/control ratios, not the ratio of
those arm columns. Dispatch ratios are Opaque `0.969976 / 0.973997 / 0.988613`,
Mixed `1.174111 / 1.136992 / 0.943438`, and Sky
`0.798308 / 0.839074 / 0.795761`. Positive change means slower.

The Mixed original process medians span 62.612--81.490 ms; No-Fiber spans
73.514--78.048 ms. Comparing the first 16 with the last 16 timed dispatch
frames, original block 1 changes by +20.61% for Opaque and -18.92% for Mixed;
No-Fiber block 0 Mixed changes by +7.99%. These are chronological diagnostics,
not extra independent repetitions. The pair direction changes, and the
observed variation is much larger than the Opaque benefit. This is evidence of an
unresolved runtime cost or measurement-state effect, not proof of its cause.
Do not adopt the smaller artifact based only on native creation or SPIR-V size.
Compare resources and executable statistics on these exact artifacts, then
repeat a controlled Mixed measurement before deciding on a runtime policy.

As a separate tail diagnostic, medians of the three per-process frame P95s
are Opaque 30.683296 / 30.036544 ms, Mixed 83.371744 / 79.354272 ms, and Sky
0.503360 / 0.523104 ms (original / No-Fiber). Sky's median improves while this
tail statistic worsens; Mixed's tail statistic improves while its paired
median worsens. Neither diagnostic erases the primary paired result or
provides a confidence estimate from three processes.

Graph/Pass intervals may contain GPU upload commands. The narrower `Path trace
shading` interval includes its existing resource-stage work; it is not an
ISA-only duration. Readbacks, presentation and shader/native compilation are
outside these intervals. Sky's dispatch improvement is about 0.1 ms and must
not be presented as a 20% whole-frame improvement. The fixed scenes do not
exercise Fiber, custom programs, metal/coat/sheen/dispersion combinations,
SHARC, long temporal runs or full-scene memory behavior.

The [original-control preview](../build-release/path-backend-dispatch-original-preview-20261013/OriginalGeneral512.png)
was inspected. It shows the expected noisy two-sample geometry outputs and
sky view, with no error material. It converts ACEScg to Rec.709, applies
Reinhard and sRGB for display; the exact raw readbacks remain authoritative.
The preview and its input hashes are outside the sealed measurement directory.

The runner's 31 CPU evidence tests passed, including eight tests for explicit
General-arm selection, original-source binding, role-aware aggregation and
historical replay. Tests preserve the original default linked-control behavior.

## Exact-artifact Driver resource diagnostic

The [separate executable capture](../build-release/path-backend-executable-original-20261013/summary.json)
creates each exact original-control artifact once, with the same sealed recipe
and copied SPIR-V, without frontend compilation or Dispatch. It requests
`METALLIC_VK_PIPELINE_STATISTICS=1` and sets
`METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY` to a fresh per-arm directory,
enabling `CAPTURE_STATISTICS` and `CAPTURE_INTERNAL_REPRESENTATIONS`. Those flags
can change Driver compilation even though the ordinary v6 PSO hash is the same.
All capture creation times have `timing_eligible=false` and are excluded from
the ordinary native and GPU comparisons.

Both manifests report successful properties, statistics and representation
queries, with one CS executable and subgroup size 32. The input and final-device
SPIR-V, full recipe ABI, Driver/cache identity and native-miss payload checks
pass. Five before/after input-hash snapshots match and the result is sealed.
The original capture environment and reports remain intact; only an explicitly
documented report copy removes the two diagnostic environment fields when
reusing the ordinary native payload checker. This validates artifact/state/
cache contracts and does not qualify the captured time as an ordinary sample.

| Driver statistic | Original General | Linked No-Fiber |
| --- | ---: | ---: |
| Register Count | 255 | 255 |
| Binary Size, bytes | 1,757,056 | 1,437,312 |
| Stack Size | 0 | 0 |
| Shared Memory Size, bytes | 12,288 | 12,288 |
| Local Memory Size, raw UINT64 | 68,719,489,584 | 68,719,485,840 |

Driver-reported Binary Size decreases by 319,744 bytes (**18.20%**). Registers,
stack and shared memory do not decrease. The large Local Memory value is
retained with its original format/name/description and is not truncated,
decoded or interpreted as spill. These are static executable statistics;
they do not establish occupancy, executed instruction counts or the cause
of Mixed's result.

Both representation queries succeed with **zero representations**, so this
capture supplies no Driver ISA or executable representation bytes. The ignored
`Capture.py`, exact commands, explicit capture environment, original logs,
reports, manifests and input hashes are retained in the evidence directory.
Use runtime hardware counters and repeated measurements with better state
control to investigate Mixed before promoting a candidate; do not infer an
execution speedup from the smaller Binary Size.

## Reproduce and verify

List the current catalog first; indices can change. Create one fresh mapped O1
reference without postprocessing, then prepare five overlays. Every output
directory below must be fresh; replace `N` with the selected SHARC-off Guides request.

```powershell
$bench = '.\build-release\Source\MetallicShaderPipelineBench.exe'
& $bench --list --filter OpenPBRRayQueryPathTraceGuides
python -B Tools/Perf/MeasureShaderPostprocess.py run `
  --executable $bench --request-indices N --postprocesses none --pairs 1 `
  --output-dir build-release/backend-reference-<fresh>
python -B Tools/Perf/ValidatePathTraceBackend.py prepare `
  --report build-release/backend-reference-<fresh>/request-N-block-00-O1-pipeline-normal/report.json `
  --request-index N --slang-root External/slang `
  --output-dir build-release/backend-type-<fresh>/preparation
python -B Tools/Perf/ValidatePathTraceBackend.py run `
  --preparation build-release/backend-type-<fresh>/preparation `
  --repetitions 1 --output-dir build-release/backend-type-<fresh>/compile
python -B Tools/Perf/ValidatePathTraceBackend.py verify `
  build-release/backend-type-<fresh>/compile
python -B Tools/Perf/ValidatePathTraceBackend.py analyze `
  --input-dir build-release/backend-type-<fresh>/compile `
  --output-dir build-release/backend-type-<fresh>/analysis
```

The uniform cleanup and merge diagnostics retain their own scripts, commands,
tool hashes and manifests under the linked analysis directories. They remain
separate from the compiler harness and runtime cleanup policy. Keep generated
shader overlays, modules and analysis artifacts outside source control.

Reproduce the validation-on artifact comparison from the sealed current
preparation/compiler/cleanup inputs. Geometry caches for the fixed scenes must
already be populated, and the output directory must be fresh (choose a new
suffix when repeating the command). This command
retains the reference request and applies the same two-pass artifacts; it does
not recompile the target shader or measure native creation performance.

```powershell
python -B Tools/Perf/ValidatePathTraceBackendRendering.py `
  --preparation build-release/path-backend-type-20261013/preparation `
  --compile build-release/path-backend-type-20261013/compile `
  --analysis build-release/path-backend-cleanup-analysis-20261013 `
  --reference build-release/path-backend-type-20261013/preparation/ReferenceReport.json `
  --cases build/compute-create-fixtures/Correctness.json `
  --executable build-release/tests/MetallicRHITests.exe `
  --output-dir build-release/path-backend-rendering-recheck-20261013 `
  --absolute-tolerance 2e-5 --relative-tolerance 2e-5
```

Reproduce the direct original-source controls using the same sealed artifacts.
The native runner disables the controllable Driver cache and performs one
creation per independent process; the dispatch runner enables that cache and
uses separate correctness and timing processes. Run the batches sequentially,
keep all `Tools/Perf` helpers frozen during each run and choose fresh output
directories. Pass `--general-arm original-general` explicitly to the GPU
runner, whose default still selects linked General.

```powershell
python -B Tools/Perf/MeasurePathTraceBackendNative.py run `
  --preparation build-release/path-backend-type-20261013/preparation `
  --compile build-release/path-backend-type-20261013/compile `
  --analysis build-release/path-backend-cleanup-analysis-20261013 `
  --reference build-release/path-backend-type-20261013/preparation/ReferenceReport.json `
  --template-recipe build-release/slang-cleanup-api-sharc-correctness-20261013/block-00-O1-pipeline-normal/recipes/compute-9ddcf6da408be75c.recipe.json `
  --executable build-release/Source/MetallicShaderPipelineBench.exe `
  --general-arm original-general `
  --output-dir build-release/path-backend-native-original-recheck-20261013
python -B Tools/Perf/MeasurePathTraceBackendDispatch.py run `
  --preparation build-release/path-backend-type-20261013/preparation `
  --compile build-release/path-backend-type-20261013/compile `
  --analysis build-release/path-backend-cleanup-analysis-20261013 `
  --reference build-release/path-backend-type-20261013/preparation/ReferenceReport.json `
  --cases build/compute-create-fixtures/Correctness.json `
  --executable build-release/tests/MetallicRHITests.exe `
  --general-arm original-general `
  --output-dir build-release/path-backend-dispatch-original-recheck-20261013
python -B Tools/Perf/MeasurePathTraceBackendNative.py verify `
  build-release/path-backend-native-original-20261013
python -B Tools/Perf/MeasurePathTraceBackendDispatch.py verify `
  build-release/path-backend-dispatch-original-512-20261013
```

Historical verification executes the actual archived helpers and checks the
sealed inputs and raw outputs. Adding `--current-inputs` also compares the
current source, binaries and helper inventory, and can appropriately reject
an older batch after repository changes.

The preparation's five generated-overlay ownership audits passed. The
27 CPU cases in [`TestPathTraceBackend.py`](../tests/perf/TestPathTraceBackend.py)
also passed, including the registered `MetallicPathTraceBackendEvidenceTests`
CTest suite. Tests cover source preservation, module ownership, strict
reference settings, provider/stage binding, artifact/plan identity, complete
schedules, early baseline failure, timeout and nonfinite metrics. They mock
compiler processes and provide no GPU qualification.

The rendering extension additionally passes 17 mocked evidence-gate tests in
`MetallicPathTraceBackendRenderingEvidenceTests`, 10 CPU scoped-override tests
in `MetallicShaderRequestsTests` and two complete-material eligibility tests
in `MetallicRHITests`. The runtime/test/Bench targets build in Release; the
repository module audit passes for 332 authored Slang files. Existing
postprocess tests pass (five passed, one optimizer-unavailable configuration
test skipped because this build has SPIRV-Tools enabled).

The first rendering attempt rejected the artifact before executing it because
the test loader incorrectly required an unused Ballot capability declaration.
Slang capability requests permit operations but do not require unused
`OpCapability` declarations. The corrected loader retains exact request
capability matching and sealed Vulkan 1.4 validation; the successful batch uses
a new output directory. The failed report is retained separately.

## Optimization implication

The coarse implementation boundary and declared material restriction provide
the measured code reduction and original-control native benefit. The Mixed
runtime result prevents selecting the current No-Fiber implementation as a
general default; correctness qualification alone is insufficient. Link-time
Type selection itself improves the
matched static output by only 19 instructions / 160 bytes after two-pass
cleanup. It has no established frontend advantage from the single compiler
samples. Prefer explicit scene capabilities and coarse, independently selected
material implementations for the next experiment; use the static arm as the
control for every link-time change. Reusing serialized consumer IR can be
investigated separately, with its preparation and relink costs measured rather
than assuming that linking supplies stronger dead-code elimination.

```powershell
python -B tests/perf/TestPathTraceBackend.py
ctest --test-dir build-release -R '^MetallicPathTraceBackendEvidenceTests$' --output-on-failure
```
