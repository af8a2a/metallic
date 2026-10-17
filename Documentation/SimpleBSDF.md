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
