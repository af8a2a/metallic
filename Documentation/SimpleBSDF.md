# SimpleBSDF material generic experiment

`SimpleBSDF` wraps the existing full `vk_gltf_renderer` BSDF port in Metallic's
three-stage material structure. The name identifies the wrapper, not nvpro's
separate simplified BSDF. The imported lobe implementation is unchanged.

```text
Material<SimpleBSDFMaterialProgram<SceneSimpleBSDFSource>>
    -> evaluate(context, instance)
    -> SimpleBSDFClosure
    -> prepare(wo, Radiance)
    -> SimpleBSDFPreparedClosure
    -> generic NEE evaluation / continuation sampling
```

The scene source owns parameter/texture access and mutable texture RNG. The
resolved closure owns values only; the prepared closure adds the outgoing
direction and transport validity. Prepared evaluation does not reload textures
or retain the source. Concrete Slang generic types select the implementation at
compile time; material instance references select data, not shader types.

The source BSDF evaluates a randomly selected lobe and returns its PDF. Its
projected contributions already include cosine, its continuation weight already
includes division by PDF, and delta samples retain the source's `-1` sentinel.
These differ from the deterministic full-mixture contract of
`IPreparedSurfaceClosure`. As with the existing separate Fiber contract,
`SimpleBSDF` extends the shared `Material<TProgram>` wrapper with its own explicit
random-evaluation interfaces. It does not reinterpret the selected-lobe PDF as a
mixture PDF or invent a deterministic evaluation. Importance transport fails
closed; the experiment uses the same Radiance convention as the direct path.

## Selection and comparison

Both arms use the same experimental RT integrator, raygen entry point, scene
mapping, RNG order, guides, ordered transparency and three hit stages. Change
only `bsdf` from `vkGltf` (direct calls) to `simple` (material generics):

```json
{
  "traversalBackend": "rayPipeline",
  "bsdf": "simple",
  "shaderExecutionReorder": false,
  "cacheMode": "off",
  "splitPrimaryGuides": false,
  "debugView": "final"
}
```

The generic raygen request adds `METALLIC_SIMPLE_BSDF=1`. Direct requests and hit
requests retain their existing definitions. The pass shader key and accumulation
state distinguish the two models. OpenPBR remains the existing independent path;
this experiment does not change the default or extend scene material support.
See [the traversal experiment](PathTraceTraversal.md) for the supported material
fields and limitations inherited by both arms.

The A/B fixes SER off. Correctness requires the existing HDR absolute/relative
bound and byte-identical guides, including same-arm repetitions. Native PSO
measurements replay actual captured recipes and time only
`vkCreateRayTracingPipelinesKHR`, with empty application caches and controllable
driver caching disabled. Compiler measurements use four separate fresh `slangc`
processes and require each output stage to match its actual runtime SPIR-V.
Compiler/OS/hidden driver state is not guaranteed cold. Per-frame GPU speed is
outside this experiment's requested measurement scope.

Results and plots are generated in new local build output directories; historical
vkGltf/SER evidence remains unchanged.

## Measurement on 2026-10-17

RTX 5070 Ti, driver 617.42, Slang 2026.18.2, existing Release build. Each role
uses four independent processes per arm in AB/BA/AB/BA order. Every sample is
retained. Times below are medians in seconds; percentage changes are computed
per paired block, so their median is not the ratio of the two time medians.

| Scope | Direct A (s) | Generic B (s) | Paired median B/A | Four-pair range |
| --- | ---: | ---: | ---: | --- |
| Native PSO, single | 9.303 | 9.397 | +2.15% | -7.71% to +7.08% |
| Native PSO, general | 9.927 | 9.698 | +0.67% | -3.58% to +3.26% |
| Four-stage compile, single | 4.680 | 4.780 | +2.44% | -1.50% to +7.00% |
| Four-stage compile, general | 4.567 | 4.786 | +4.44% | -0.11% to +13.96% |

No consistent native PSO regression was observed across the paired samples.
The spread and n=4 cannot exclude a small regression. Compiler times were
somewhat higher for B in this batch, with mixed-sign paired results. This does
not establish that generics are free or that another workload will behave alike.

The 12 validation-enabled correctness processes covered Opaque1, Mixed4 and
ABeautifulGame. All six A/B pairs (88 readbacks) and six same-arm pairs (88
readbacks) were byte-identical, including temporal HDR and guides. All 64 timed
compiler outputs matched their corresponding runtime stage SPIR-V byte for byte.

Actual runtime raygen SPIR-V grew by 6,996 bytes / 502 instructions (about 0.32%)
in both roles. This was local-storage/copy/control-flow growth, with equal
floating-point opcode counts, no surviving function calls, and no replication
of the shared NEE evaluation by source type. Driver machine-code/register costs
were not measured. The CPU wrapper comparison had 81,920 byte-identical samples;
it did not encounter an actual Dirac sample. The final four evidence/module CTest
suites passed. An existing full-request warmup fixture still expects three
requests where the pre-existing Deferred canonicalization produces two; this
unrelated assertion remains documented rather than hidden.

Local evidence: `build-release/simple-bsdf-generic-20261017/report-final/report.md`,
`ImplementationValidation.md`, `ir-review/Review.md`, and
`ir-runtime-match/RuntimeIdentity.md`. Reproduce using
[the measurement tools](../Tools/Perf/SimpleBSDFGeneric.md). No realtime GPU
performance claim is made by this PSO experiment.

## GPU pass follow-up on 2026-10-17

The separate [GPU runner](../Tools/Perf/SimpleBSDFGPU.md) compares the wrapper at
both EXT SER settings. The renderer/shader implementation and Release executable
are unchanged from the PSO experiment. RTX 5070 Ti / driver 617.42; all three cases
render at 512x512 with depth 12, cache off and guides enabled. Opaque1 uses 1 spp,
Mixed4 uses 4 spp, and ABeautifulGame uses 2 spp. Each arm/case has four independent
processes in Williams order, each with 32 warmup frames and 64 timed frames.
Validation is disabled for timing and enabled in the separate correctness stage.

The primary scope is the PathTrace pass GPU timestamp range. Times below are
medians of process medians in milliseconds. Percentage changes are the median
and range of four **paired** block percentages, not the ratio of arm medians.

| Case | SER | Direct (ms) | SimpleBSDF (ms) | Paired median | Four-pair range |
| --- | --- | ---: | ---: | ---: | --- |
| Opaque1 | off | 11.473 | 10.792 | -0.24% | -15.37% to +4.15% |
| Opaque1 | on | 33.454 | 33.140 | -1.10% | -5.15% to -0.02% |
| Mixed4 | off | 156.226 | 159.118 | +0.13% | -7.18% to +2.23% |
| Mixed4 | on | 241.690 | 249.147 | +2.97% | +1.70% to +4.32% |
| ABeautifulGame | off | 2.295 | 2.303 | +0.46% | -0.19% to +4.08% |
| ABeautifulGame | on | 6.268 | 6.272 | -0.05% | -1.50% to +0.26% |

The SER-off wrapper comparisons have mixed-sign paired changes and substantial
dispersion in Opaque1/Mixed4; they do not establish a stable speedup or zero cost.
With SER on, Mixed4 is slower in all four wrapper pairs, an observed regression
of approximately 3% in this run. Opaque1 is slightly faster in all four SER-on
wrapper pairs. These observations have n=4 and unmodified clocks; no confidence
interval, significance claim, or machine-code explanation is inferred.

Both same-SER wrapper contrasts pass the combined numerical gate in all three
cases: 176 correctness-stage plus 192 timing-stage readback comparisons pass the
unchanged HDR tolerance and byte-exact guide requirements. All 192 timing-stage
wrapper readbacks are byte-identical. The SER-on temporal correctness capture has
19 HDR readbacks that differ within the existing tolerance, so the complete
correctness set is not described as byte-identical. Same-arm repeats also pass.

Switching SER itself fails the combined equivalence gate for both implementations
in all three cases. The observed direct-path SER pass-time increases are +191.46%,
+57.06%, and +172.69% (paired medians for Opaque1, Mixed4, ABeautifulGame); these
remain **diagnostic measurements, not qualified SER optimizations**. Failed
cross-SER contrasts do not erase the separately qualified same-SER wrapper
comparisons. Ordinary timing also reads final HDR at frame 96 and seven guides
on additional frames 97..103, outside the timed window.

All 72 GPU processes completed without runtime/validation errors or shader
identity drift. Every captured stage matched its corresponding fresh factory
reference SPIR-V. The current verifier retains global `numeric_failure` because
of the cross-SER failures, while reporting each contrast's qualification.
All five selected evidence/module CTest suites passed. GPU clocks and background
activity were not controlled; whole-device telemetry is retained without using
it to remove samples. These pass/dispatch/graph ranges are not editor FPS.

Local evidence and Python charts:
`build-release/simple-bsdf-gpu-20261017/report-final/report.md`.

## SER correctness repair and qualified GPU rerun on 2026-10-17

This later experiment fixes the cross-SER failures above. The earlier packages
remain historical diagnostic evidence; they are not overwritten or retroactively
qualified. The HDR bound remains `2e-5 + 2e-5*abs(reference)` and all seven guides
still require raw byte equality. No output quantization or tolerance change is used.

Two local arithmetic boundaries needed explicit SPIR-V `NoContraction` in the
experimental integrator. Hardware-depth projection differed by one ULP despite
identical same-frame linear depth; controlling the intermediates and final Fma
removed that discrepancy. A bounded HDR diagnostic then exposed a one-ULP primary
camera-direction difference before shadow traversal. The final uninstrumented
candidate fixes the guides-enabled perspective camera's affine combination and
normalization order, retaining the shared camera basis, projection inputs, fallback
and threshold. Instrumentation perturbed some results, so the final strict GPU
matrix, rather than debug logging alone, establishes acceptance.

These camera changes affect the guides-enabled primary rays, not just guide
writeout. They can change low HDR bits relative to the old shader. RNG, BSDF, NEE,
ray origins/cones, ordered transparent shadows and the non-guides camera branch
remain unchanged; shared OpenPBR shaders are untouched. SER uses an explicit fused
`OpHitObjectTraceReorderExecuteEXT` wrapper because Slang 2026.18.2 lacks that free
API. Final IR contains two real EXT fused calls when enabled and two `TraceRayKHR`
calls when disabled. The fused wrapper alone did not fix the numerical failures.

All 32 production-factory SPIR-V stages compile and pass Vulkan 1.4 validation.
Relative to the depth-only candidate, the camera repair adds 24 NoContraction
decorations and 804 bytes / 51 instructions per raygen; 24 hit stages are unchanged.
These IR counts say nothing about native register allocation or spills.

On RTX 5070 Ti / driver 617.42 / Slang 2026.18.2, the repaired candidate passes all
24 correctness processes and all 48 ordinary timing processes. Every contrast and
same-arm repeat passes, including temporal/camera-change HDR and byte-exact guides.
Actual runtime stages match the fresh reference SPIR-V. This is the same three-case
512x512, depth-12, guides-enabled workload and four-arm Williams timing order
described above, with 32 warmup and 64 timed frames per independent process.

PathTrace pass medians below are in milliseconds, with four process medians per
cell. SER changes are the median of four block-paired percentages, not ratios of
the table's medians. No sample was removed.

| Case | Direct off | Simple off | Direct on | Simple on | Direct SER paired change |
| --- | ---: | ---: | ---: | ---: | --- |
| Opaque1 | 10.132 | 10.065 | 32.740 | 32.715 | +222.91% [199.49%, 225.49%] |
| Mixed4 | 158.853 | 157.112 | 244.134 | 248.117 | +60.76% [53.66%, 96.19%] |
| ABeautifulGame | 2.600 | 2.573 | 6.957 | 6.999 | +169.93% [165.73%, 176.39%] |

Wrapper paired medians (off/on) are -0.51%/-0.03%, +0.52%/-0.48% and
-1.14%/+0.43%, respectively. Mixed4 wrapper ranges are -19.69% to +29.11% with
SER off and -6.00% to +11.27% with SER on; a stable wrapper gain/regression is not
established. Frame inspection shows sustained fast/slow periods, primarily in the
shading interval. Whole-device telemetry has no exact frame-time join and cannot
identify a clock, thermal, cache or background-load cause. The SER regression is
present in every paired block and remains much larger than the wrapper medians.

### Shadow-SER policy control

A separate temporary candidate keeps the same numerical repair, direct BSDF and
main-ray fused EXT call, but uses ordinary TraceRay for ordered transparent shadow
queries. All four SER-off raygens and 24 shared hit stages are byte-identical to the
repaired full policy. SER-on has one fused call and one TraceRay; its outgoing
payload variables now have separate locations, so this changes native codegen as
well as the shadow reordering policy. The payload value structure is unchanged.

The control passes 12 correctness processes across all three scenes, followed by
eight ordinary Mixed4 timing processes in AB/BA/AB/BA order, with the original
readback and same-arm gates. Direct SER-off pass medians are 117.374608, 117.765552,
123.399840 and 116.601792 ms; main-only SER medians are 90.246512, 80.928576,
81.397328 and 87.435728 ms. Its four within-period changes are -23.11%, -31.28%,
-34.04% and -25.01% (median -28.15%). This reverses the sign of the earlier
Mixed4 full-policy SER comparison, strongly implicating shadow reordering and its
associated codegen as an important contributor to that workload's regression.

The full policy and control were measured in separate periods and used different
four-arm/two-arm schedules. Their identical SER-off SPIR-V nevertheless produced
different timing distributions. Do not pair blocks across periods or attribute
the complete absolute-time difference to shadow SER. The control covers direct
BSDF timing in one scene and does not qualify a four-arm production replacement;
the working-tree shader is restored to the full policy that passed 24+48 processes.

The integrator invokes shadow queries inside a loop of up to 16 transparent layers.
All materials share one hit SBT record, closest-hit writes a small payload, and
expensive material work runs later in raygen; no explicit material coherence hint
is supplied. Repeated state save/restore with insufficient coherence benefit is a
plausible explanation, consistent with the
[Khronos SER guidance](https://www.khronos.org/blog/boosting-ray-tracing-performance-with-shader-execution-reordering-introducing-vk-ext-ray-tracing-invocation-reorder).
It is not a measured spill or occupancy diagnosis. A single hit shader/no hint
does not preclude SER gains, as the same article's vk_gltf_renderer example shows.
Native RT live-state/coherence counters would be needed to separate those costs.

The result establishes cross-variant equivalence on the declared perspective,
guides-enabled workloads, not reference-renderer truth or cross-driver bitwise
identity. Orthographic, no-guides GPU rendering, accumulated output and extreme
FOV behavior were not separately exercised. GPU pass/shading/graph timestamps
exclude shader compilation and PSO creation and are not editor FPS measurements.

Local evidence under `build-release/ser-correctness-20261017/`:

- `report-final-v2/report.md`, `PassProcesses.png`, `PassPairedChanges.png`, and
  `report-tools/PlotReport.py`: verified full matrix and visually reviewed charts.
- `policy-control-report/report.md`, `PolicyControl.png`, and
  `control-report-tools/PlotPolicyControl.py`: restricted control and its charts.
- `candidate-09b-camera-affine/{correctness,performance}` and
  `candidate-10-main-only-ser/{control-correctness,control-performance}`: sealed raw
  processes, readbacks, references, identities and original thresholds.
- `variation-review/VariationReview.md`: all Mixed4 timed frames and telemetry
  limits; `ProjectionModel/README.md` and `Candidate09bEvidenceNote.md`: numerical
  investigation, including rejected candidates and instrumentation limitations.
- `candidate09b-identity-review.json`, `candidate10-identity-review.md`,
  `PolicySource-restore.json`: IR evidence and exact source restoration.

Use the current maintained `Tools/Perf/SimpleBSDFGPU.py verify <capture>` with the
restored source to recheck the full matrix. The temporary policy's current-source
verifier requires its separately preserved overlay; its report records verification
while that source was active. Do not execute archived evidence scripts blindly or
weaken the source-identity checks to verify a different installed candidate.
