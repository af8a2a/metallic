# Path-tracing resource narrow reads (2026-10-12)

Path-tracing resource access now reads the required fields instead of loading
the complete 384-byte `SceneResourceParameters` packet. The original wire
packet, 24-byte compute root, descriptor resolution, resource leases, RNG,
normal/TBN handling and feedback/write order remain unchanged. Sparse
`SHARC_UPDATE` retains its whole-packet reads.

## Change and bounds contract

`getResourceParameterField<Packet,Field>` remains the sole slicing mechanism.
It requires a complete compute root and complete Scene packet, verifies field
size/offset/alignment, and returns zero on an invalid/truncated input. It keeps
valid optional-resource sentinel values unchanged. Resource/span resolution
and `resolveNonUniform` calls remain at their original sites.

The shared CPU/Slang header adds byte offsets checked by `offsetof` and small
read-only views with size/member-offset assertions. It does not rearrange or
extend the actual Scene packet. The hot consumers now read:

| Resource | Read size | Packet byte offset |
| --- | ---: | ---: |
| Scene acceleration-structure address | 8 | 0 |
| Environment map/PDF | 4 each | 32 / 36 |
| Lights/PDF/ReGIR handle | 4 each | 88 / 92 / 260 |
| Material sampler | 4 | 108 |
| Material textures span | 12 | 112 |
| Material values/materials handle | 4 each | 136 / 140 |
| OpenPBR 2D/3D LUT spans and sampler | 28 | 172 |
| Ray-stream changes | 4 | 220 |
| Ray-stream header/instances/page table/pages | 4 each | 244 / 248 / 252 / 256 |
| Stream-page count | 4 | 316 |
| Texture feedback | 4 | 332 |
| Physical environment resources | 28 | 356 |

Existing guide/output/history/cache/sharc slices remain in use. The new reads
cover Surface/OpenPBR material adapters, direct/fiber lighting, traversal and
texture-feedback helpers. Other consumers of the shared helpers include
SceneRTXDI, SceneMaterialVisualize and VisibilityBuffer lighting. Whole-packet
getters still visible in unused legacy integrator or SHARC update source are
not evidence that the measured normal Guides entry calls them.

## Frozen-input compiler comparison

The baseline was rebuilt from a HEAD snapshot into an ignored overlay and
verified byte-for-byte against the current original Benchmark cache. The
candidate was separately frozen after the change. Each arm used Slang
2026.18.2, the same Guides request/module/entry/macros/capabilities, O1, mapped
descriptors and disabled debug information. Six independent CLI processes ran
sequentially in paired order baseline/candidate, candidate/baseline,
baseline/candidate. Each reproduced its expected SPIR-V hash. Shader/IR disk
cache reuse was absent; OS filesystem caches were allowed to remain warm.

| CLI phase, process median | Baseline | Narrow reads |
| --- | ---: | ---: |
| Wall time | 21.262 s [20.524, 21.354] | 12.426 s [11.834, 12.432] |
| SDK downstream | 19.250 s [18.920, 19.320] | 10.210 s [9.690, 10.240] |
| SDK `compileInner` | 20.995 s | 12.178 s |
| SDK `generateOutput` | 20.124 s | 11.276 s |
| SDK `linkAndOptimizeIR` | 0.686 s | 0.900 s |
| SDK `frontEndExecute` | 0.871 s | 0.904 s |

The paired median wall-time ratio is **0.584447 (-41.56%)**; downstream ratio
is **0.528468 (-47.15%)**. The observed benefit is in downstream SPIR-V processing,
not source parsing or the `linkAndOptimizeIR` counter. These SDK phases overlap
and must not be added. This measures compiler work, not renderer launch time
or Driver/GPU speedup.

Evidence: [six-process compiler summary](../build-release/narrow-read-frontend-paired/summary.json),
its `Manifest.json`, raw commands/logs/SPIR-V and frozen input identities.

## Final SPIR-V structure

| Metric | Baseline | Narrow reads |
| --- | ---: | ---: |
| Bytes | 6,940,612 | 6,945,092 |
| FNV-1a64 | `ef751a428c7a9c48` | `f0282250bfa994e8` |
| Instructions | 447,484 | 449,692 |
| Basic blocks | 58,220 | 60,520 |
| CFG edges | 74,427 | 77,877 |
| Function-storage variables | 26,347 | 24,745 |
| Function-storage loads | 40,669 | 39,642 |
| Function-storage stores | 49,761 | 49,423 |
| StorageBuffer loads | 4,509 | 4,509 |

Both have one function, zero calls/Phi nodes and 495 loop merges. The final
module did not become smaller: temporary Function storage decreased, while
control flow increased. Function storage is not a measurement of GPU spills,
and identical StorageBuffer load counts do not establish lower memory traffic.
The smaller ID bound is not an instruction-count or runtime-cost metric.
Reduced whole-packet intermediate construction can lower compiler processing
cost even when its final code has similar size; the measured phase timings
support the cost reduction without assigning it to an individual optimizer pass.
See [exact structural delta](../build-release/narrow-read-complexity/Delta.json).

## Native creation comparison

Three qualified pairs of independent recipe-replay processes compared ordinary
Compute Pipeline creation. Both arms use their verified precompiled SPIR-V,
identical full PSO state/mapped ABI/12-byte user push, debug/validation/statistics
disabled, a new application cache and disabled controllable Driver internal
cache. There is no native-object reuse and every pipeline reports valid native
cache-miss feedback. Source/module preparation and cache export are outside
`native_create_ms`; this is not a total application-startup timer.

| Native phase, process median [min, max] | Baseline | Narrow reads |
| --- | ---: | ---: |
| `native_create_ms` | 61,527.948 [61,221.048, 67,412.761] | 62,181.839 [60,032.194, 63,408.556] |
| `driver_ms` | 61,527.902 [61,220.994, 67,412.722] | 62,181.804 [60,032.166, 63,408.518] |

Paired native ratios are 0.890517, 1.035731 and 1.010628. The median ratio is
1.010628; signs vary and three pairs do not establish a stable gain or
regression. The original first pair overlapped a CPU ID-compaction diagnostic
at 17:14:47–17:15:01 Asia/Shanghai. Both members were excluded and the full pair
was resampled; all original logs remain intact. This exclusion is due to known
interference rather than a measured duration. Eight processes were executed,
six are used in the qualified comparison.

The Driver's ~60-second cold creation remains the largest measured cost.
Narrow reads improve the Slang compilation path, not this native bottleneck.
Separate compiler/native medians must not be added and reported as a measured
end-to-end speedup. See [qualified Native evidence](../build-release/narrow-read-native-qualified.json),
which verifies both original/replacement manifests and lists selected/excluded
samples, exact recipes, bytecode identities, feedback and commands.

## Runtime qualification

Release Bench and RHI builds passed. The extended
`named_resource_field_views_and_bounds` test passed with Vulkan validation.
Mapped/native modes each exercised full packet, short packet, all-ones sentinel
and a 20-byte short root. Sixty-two output words cover new individual fields,
views, both halves of the scene address and invalid/unaligned offsets. The
short-root output lookup deliberately uses only its valid bytes 12–19; field
reads still return zero because the complete root is absent.

Final rebuilt tests for field bounds, optional-resource sentinels and named
environment ABI passed with Vulkan validation. Final LUT/ReGIR identifier cleanup
was recompiled and verified to produce the exact same measured candidate SPIR-V;
see [final source identity proof](../build-release/narrow-read-final-source-proof.json).

O1 mapped/debug-disabled compile smoke and Vulkan 1.4 scalar-layout validation
passed for SceneRTXDI, SceneMaterialVisualize, legacy Guides and OpenPBR Guides
SHARC update/query (catalog indices 137/138/144/149/152 at measurement time).
The SHARC update raw IDs differ, but its opcode sequence/count is identical,
and a separate ID-only compaction produces identical validated bytecode.
This is structural evidence; SHARC training GPU/temporal cache behavior was not
directly tested. All original raw modules were retained.

The actual OpaquePerf renderer ran at 512×512, two samples, depth 12, SHARC off,
32 warmup and 64 timing frames in three independent processes. With fixed
`atol=rtol=2e-5`, final HDR and seven guides were compared against the sealed
original O1 Pipeline baseline: **24/24 outputs are byte-for-byte equal, maximum
error zero**, with no nonfinite values. Cases/settings, resolved graph, 55 asset
hashes, device/driver/features and full PSO state were independently checked.
The recipe changes only its bytecode-dependent PSO identity
`5aa0734d26f3cc25` → `2dea6aa029a50f50`, with the same mapped ABI and 12-byte
user-push layout.

Candidate dispatch-scope process median is 24.099 ms [23.154, 24.481]. That
reference is historical rather than a same-session paired GPU experiment;
no GPU speedup or regression is claimed. Mixed MASK/BLEND/transmission,
fiber-specific, custom-material and temporal SHARC workloads remain broader
qualification work rather than being covered by this opaque fixture.

The [actual HDR comparison preview](../build-release/narrow-read-preview.png)
was inspected. Both panels use the same ACEScg-to-Rec.709 conversion, exposure
1 and sRGB transfer without tone mapping. The low two-sample fixture retains
its original noise; byte-equal readbacks establish unchanged output here.

Evidence:

- [Runtime workload/process evidence](../build-release/narrow-read-gpu-512/summary.json).
- [Independent original-to-candidate output proof](../build-release/narrow-read-old-baseline-proof.json)
  and its sibling reproducible qualification script.
- [Affected-request compile smoke](../build-release/narrow-read-smoke/SmokeEvidence.json)
  and `SharcUpdateStaticDifference.json`.

Generated images, captures and experiment data stay outside source control.

## Material-record follow-up (2026-10-13)

The next experiment narrows the initial 720-byte `SceneMaterialData` transport
in found-vertex shading and guide classification. It preserves the full
General implementation, including Fiber and conditional Slab paths. It does
not use the material-backend Type experiment's No-Fiber eligibility restriction.
Production shader sources remain unchanged; the experiment generates owned
frozen overlays with [`PathTraceMaterialReadOverlay.py`](../Tools/Perf/PathTraceMaterialReadOverlay.py).

The raw-buffer candidate reads the program/alpha header at byte 56 (8 bytes
for the vertex, 4 for guides), Fiber parameters at byte 112 (64 bytes) inside
the existing Fiber branches, and coverage only inside the existing BLEND
condition. Coverage reads alpha at 12, cutoff at 40, program presence at 92
and the 48-byte texture record at 176, reusing the header's alpha mode.
It preserves float-to-uint program classification, normal handling, RNG,
alpha-layer updates and the existing texture evaluation order. Surface model
parameter loading, shadow traversal and Slab loading are unchanged.

The original material load has no local bounds/default behavior. The candidate
retains its valid resolved buffer, complete 720-byte record, valid index and
non-overflowing byte-index preconditions. It adds no `GetDimensions` gate or
missing-record zero/default rule. The existing Scene packet/root resolver and
uniform handle resolution remain in use, including the SHARC-update getter.
The 720-byte record and 48-byte texture ABI remain unchanged.

The [raw-read six-process compiler batch](../build-release/path-material-reads-20261013/compile/summary.json)
uses the current Guides API reference (O1, mapped, debug disabled, SHARC off,
CUSTOM_MATERIALS=0, runtime postprocess none), sequential AB/BA/AB, no module/IR
cache and final Vulkan 1.4 scalar-layout validation. All three original outputs
match the API reference byte for byte; each candidate repetition has the same
hash. Current source differs from the preceding 2026-10-12 packet experiment,
so the older timing medians must not be combined with these measurements.

| Raw-read measurement | Original General | Raw field reads |
| --- | ---: | ---: |
| Compiler process wall median, s | 10.093093 | 10.271174 |
| Raw SPIR-V bytes | 6,947,064 | 6,952,452 |
| Raw instructions / blocks | 449,813 / 60,501 | 450,141 / 60,543 |
| Uniformly cleaned bytes | 3,314,908 | 3,320,116 |
| Uniformly cleaned instructions / blocks | 215,311 / 28,217 | 215,630 / 28,259 |

Paired wall ratios are `0.988148 / 1.001384 / 1.017644`; their median change
is **+0.14%**, with mixed directions and no established compiler benefit.
The [cleanup analysis](../build-release/path-material-reads-20261013/analysis/summary.json)
applies only `--eliminate-dead-branches` then `--simplify-instructions` equally
to both inputs. Function variables increase by six and total `OpLoad` count
by 48 in both raw and cleaned modules. These are static declaration/opcode
counts, not material bytes, dynamic traffic, registers or spill. No native
creation or GPU dispatch performance gain is claimed for this candidate.

[`SceneMaterialReadProbe.slang`](../tests/rhi/shaders/SceneMaterialReadProbe.slang)
and the `scene_material_read_overlay_layout` RHI test exercise the actual
overlay helpers against full-record reads and CPU `SceneMaterialData`. Two
complete records cover index/stride, Surface/Fiber IDs, alpha/coverage values,
all four Fiber vectors, all texture fields/padding/transforms and the all-ones
texture sentinel. Mapped and native modes pass with Vulkan validation for the
raw candidate. These field readbacks do not establish HDR, temporal/guide,
SHARC or full-scene output equivalence, and do not read undefined invalid
records. The test skips when no explicit frozen overlay is supplied.

The harness freezes the full source/include graph, CPU ABI header, SDK and
actual helpers, checks them around each process, and binds each declared arm
to its generated sources. Failed runs are sealed without sample admission.
Historical verification uses the archived helpers; current-input checks also
reject later source/tool changes. The new compiler evidence tests passed,
including candidate selection and historical verification of the raw batch.

### Typed-member control

The [separate typed-member compiler batch](../build-release/path-material-member-reads-20261013/compile/summary.json)
keeps the same classification/Fiber/coverage boundary and public probe helper
interfaces, while projecting fields through the existing typed material buffer.
Its active readers use no explicit byte address arithmetic. It retains General
model branches and the original valid-record preconditions. The raw and original
overlay generation outputs remain byte-identical to their first versions.
Mapped/native layout probes also pass with Vulkan validation for this arm.

| Typed-member measurement | Same-batch Original General | Typed member reads |
| --- | ---: | ---: |
| Compiler process wall median, s | 10.172642 | 9.943602 |
| Raw SPIR-V bytes | 6,947,064 | 6,955,860 |
| Raw instructions / blocks | 449,813 / 60,501 | 450,435 / 60,617 |
| Uniformly cleaned bytes | 3,314,908 | 3,323,704 |
| Uniformly cleaned instructions / blocks | 215,311 / 28,217 | 215,933 / 28,333 |
| Cleaned Function variable declarations | 24,775 | 24,799 |
| Cleaned total `OpLoad` | 21,545 | 21,635 |

The paired wall ratios are `0.995486 / 0.942821 / 1.000704`, with median
**-0.45%**. One pair dominates the range and the signs vary. This exploratory
three-pair sample does not establish a stable frontend improvement. Its arm
median ratio is a different statistic and is not the paired result. Both
resource-read variants increase final structure after the same two-pass cleanup.
The [typed analysis](../build-release/path-material-member-reads-20261013/analysis/summary.json)
retains exact hashes and validation. Do not compare raw/member wall medians
across their separate batches as if those were a matched experiment.

The current harness passes 15 mocked CPU evidence tests. The Release RHI target
builds and the repository ownership audit passes for 333 authored Slang files;
each frozen generated overlay has its own successful ownership audit. These
results support continued experimentation, not deployment or a Driver/GPU gain.

```powershell
python -B Tools/Perf/ValidatePathTraceMaterialReads.py prepare `
  --report build-release/path-backend-type-20261013/preparation/ReferenceReport.json `
  --request-index 146 --candidate-arm material-member-general `
  --slang-root External/slang --output-dir build-release/material-member-repeat/preparation
$env:METALLIC_MATERIAL_READ_OVERLAY = 'E:/metallic/build-release/material-member-repeat/preparation/overlays/material-member-general/Shaders'
.\build-release\tests\MetallicRHITests.exe --filter scene_material_read_overlay_layout
Remove-Item Env:METALLIC_MATERIAL_READ_OVERLAY
python -B Tools/Perf/ValidatePathTraceMaterialReads.py run `
  --preparation build-release/material-member-repeat/preparation `
  --spirv-val C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe `
  --output-dir build-release/material-member-repeat/compile
python -B Tools/Perf/ValidatePathTraceMaterialReads.py analyze `
  --input-dir build-release/material-member-repeat/compile `
  --spirv-tools C:/VulkanSDK/1.4.363.0/Bin `
  --output-dir build-release/material-member-repeat/analysis
```

Use fresh output directories and keep the shader/include graph and all
`Tools/Perf` helpers frozen during each batch. Choose `material-narrow-general`
as the candidate for the raw-read control. Run the layout test before timing;
the test's field readbacks are correctness evidence and not a performance sample.

### Separate material-read native comparison

The [nine-process native batch](../build-release/path-material-reads-native-20261013/summary.json)
replays each exact two-pass-cleaned artifact in a fresh process. Three blocks
rotate Original/raw/member, raw/member/Original, member/Original/raw. Each uses
normal Compute Pipeline creation, mapped ABI, the same captured full state and
12-byte user push, empty application cache, controllable Driver cache disabled,
debug/validation disabled and no executable-capture flags. Frontend work,
initialization, cache export and GPU execution are outside the native window.

| Block order | Original native, s | Raw native, s | Member native, s |
| --- | ---: | ---: | ---: |
| Original / raw / member | 72.155366 | 62.629773 | 63.067065 |
| raw / member / Original | 72.740915 | 65.890563 | 68.528735 |
| member / Original / raw | 63.589289 | 66.140541 | 71.333216 |

Raw/Original ratios are `0.867985 / 0.905825 / 1.040121`, with paired median
**-9.42%**. Member/Original ratios are `0.874045 / 0.942093 / 1.121780`, with
paired median **-5.79%**. Driver-duration ratios agree to six decimal places;
cache-lock waits are 0.0007--0.0011 ms. All nine calls have valid native-miss
feedback, no metadata hits, object reuse, binary import/export or frontend
compilation. Actual input/final-device bytecode, recipe v6 ABI, PSO hashes,
device/Driver/cache identities and returned per-call/batch metrics pass.

Both candidates reverse direction in the last block (+4.01% raw, +12.18%
member), while Original varies by 9.152 s across processes. These observations
do not establish stable Driver gains. Disabling the controllable cache does
not prove all compiler state is cold. Keep all blocks; do not discard the
reversal or add separately measured compiler/native medians into a startup
speedup. The larger SPIR-V and ambiguous native result also do not prove that
the Source read boundary is a Driver bottleneck.

The ignored `Measure.py`, qualification records, source/raw/cleaned/recipe arm
bindings, actual commands, process IDs/intervals, original reports and archived
helpers are retained. Nineteen before/after/final input snapshots match and
the completed result is sealed. Production shader defaults remain unchanged.

### Fixed-scene material-read qualification

The [three-arm rendering qualification](../build-release/path-material-reads-rendering3-20261013/summary.json)
directly injects the exact cleaned artifacts for Original, raw and typed reads.
Each Validation-on process renders Opaque, Mixed MASK/BLEND/transmission and
Sky at 128x128, two samples and depth 12, SHARC off, CUSTOM_MATERIALS=0 and
GLOBAL_VIEW=0. It retains 16 main frames, warmup 4, captures 8--15, camera
change at 11 and seven later guide readbacks. All recorded timings are
ineligible for performance comparison.

All **144 readback arrays** (3 arms x 3 scenes x 16 outputs) are finite and
byte-identical to the corresponding Original reference. Nine target PSO
recipes prove exact artifact consumption, one per case, with the same ordinary
shader/device identities and no fallback or Validation error. Source/helper/
executable/fixture/dependency input snapshots match before and after execution,
and the completed evidence is sealed. Whole-scene eligibility for the actual
CUSTOM_MATERIALS=0 fixture is retained; this is not qualification of custom
programs, Fiber shading, SHARC, other permutations or long temporal/memory
behavior. Fiber field decoding is separately covered by the layout probe.

Three earlier preparations were rejected before rendering: an incorrect
frame-count assumption, mismatched producer/consumer nested-manifest inventory
rules, and a nonexistent `.cpp` path for the header-only artifact fixture.
Those records remain intact. The successful script uses the material producer's
own seal validation and freezes all actual dependencies. No target Slang
compilation is admitted; the ignored `Validate.py`, commands, manifests,
runtime reports and raw readbacks are retained in the successful directory.

An independent audit verified the evidence inventory, exact recipe/device
artifact identities and all 144 decoded readbacks. The
[qualification preview](../build-release/path-material-reads-preview-20261013/OriginalGeneral128.png)
was inspected for the three Original scenes; both candidates have identical
raw outputs. The preview uses ACEScg-to-Rec.709 conversion, Reinhard mapping
and sRGB encoding for display. Raw floating-point readbacks remain the
comparison evidence; the expected two-sample noise is visible in the preview.

The material read alternatives pass fixed-workload functional checks, but have
no established stable compilation/native gain. Keep them in the harness and
leave production unchanged. Compare actual function boundaries/maximum CFG
size next, using these same output and Driver-cache controls; preserve the
separation between compiler, native and GPU measurement scopes.
