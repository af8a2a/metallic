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
