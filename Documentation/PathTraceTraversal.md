# Experimental path tracing backends

The same vkGltf integrator also has a `bsdf: "simple"` material-generic variant.
See [SimpleBSDF](SimpleBSDF.md) for the wrapper contract and direct/generic PSO
comparison; this is separate from the OpenPBR/vkGltf/SER comparison below.

## Isolated vk_gltf_renderer BSDF and EXT SER experiment

An additional opt-in material/integrator path is available:

```json
{
  "traversalBackend": "rayPipeline",
  "bsdf": "vkGltf",
  "shaderExecutionReorder": true,
  "cacheMode": "off",
  "splitPrimaryGuides": false,
  "debugView": "final"
}
```

Set `shaderExecutionReorder` to `false` for its matching ordinary RT pipeline.
The default remains unchanged. SER requires the enabled
`VK_EXT_ray_tracing_invocation_reorder` device feature; unsupported requests fail
explicitly. The device's reordering hint is recorded separately and does not
prove a particular ray launch was reordered. Only ray generation uses
`HitObject::TraceRay`, `ReorderThread`, and `HitObject::Invoke`; closest-hit
continues to return the original raw CLAS hit. No NV invocation-reorder fallback
is used. SER PSO recipes use schema 4; existing non-SER RT recipes remain schema 3.

The new `VkGltfPathTracing` module imports the isolated
[full nvpro BSDF port](../Shaders/Modules/VkGltfBSDF/README.md), not an OpenPBR
closure, LUT, sampler, or integrator. It shares Metallic's scene upload ABI,
texture sampling, geometry reconstruction, camera, lights and environment.
The common `SceneSurface` resource adapter still knows OpenPBR *parameter data*
types; those are not used to evaluate the experimental BSDF.

The current scene adapter maps builtin metallic/roughness, base color, normal,
emission/unlit, opacity, occlusion, specular factor/color, transmission,
diffuse transmission, thickness, attenuation and IOR, including their existing
textures. Custom Value programs, Fiber and custom closures are rejected.
The full BSDF library also implements clearcoat, sheen, anisotropy, iridescence,
dispersion and retroreflection, but the current scene ABI has no corresponding
fields: these remain at their disabled defaults. CPU library probes exercise
those lobes; scene performance measurements do not measure them. The upstream
spectral-to-RGB helpers assume Rec.709; additional scene mapping of dispersion
or iridescence must first resolve that working-color-space boundary.

This is a distinct renderer experiment, not a claim of OpenPBR image equivalence.
Normal mapping retains Metallic's reconstruction policy. The upstream BSDF uses
stochastic lobe evaluation, and the new integrator shares one evaluation site
for local/celestial/environment NEE. The interior stack tracks absorption;
surface IOR follows the source air/material convention, not a complete nested
dielectric IOR solver. Ordered straight-through transmission shadows are an
approximation. Guides are derived from the new material model and are therefore
not required to match OpenPBR guides. SER on/off uses the same guide contract.

[Measurement tools](../Tools/Perf/VkGltfSER.md) separate three arms: OpenPBR RT,
vkGltf RT, and vkGltf RT with EXT SER. OpenPBR/vkGltf differences are recorded
descriptively; SER on/off and repeated same-arm outputs retain strict numeric
checks. Shader compile time, native PSO creation and ordinary GPU shading time
are measured separately. Generated results stay under ignored build output.

### vkGltf / EXT SER measurements (2026-10-17)

The frozen three-arm experiment used an RTX 5070 Ti, driver 617.42, Slang
2026.18.2, Vulkan SDK 1.4.350.0 and the existing Release build. Each cell below
is the median of three independent processes, with balanced OGS/GSO/SOG order.
GPU shading scopes used 512x512, 32 warmup frames and 64 timed frames, with
validation disabled. Compilation sums four sequential `slangc` process wall
times; all 72 emitted stages matched the actual runtime SPIR-V byte for byte.
Native PSO timing covers only `vkCreateRayTracingPipelinesKHR`, with empty
application caches and controllable driver caching disabled. Hidden compiler,
OS and driver state is not guaranteed cold.

| Measured scope | OpenPBR RT | vkGltf RT | vkGltf EXT SER |
| --- | ---: | ---: | ---: |
| Four-stage compile, single (s) | 6.060 | 4.728 | 4.612 |
| Four-stage compile, general (s) | 6.011 | 4.646 | 4.871 |
| Native PSO, single (s) | 20.417 | 8.992 | 10.244 |
| Native PSO, general (s) | 56.644 | 9.440 | 10.579 |
| Opaque1 shading (ms) | 19.060 | 12.269 | 34.621 |
| Mixed1 shading (ms) | 69.954 | 42.636 | 59.436 |
| ABeautifulGame shading (ms) | 4.493 | 2.045 | 6.446 |

These are different material/integrator models and supported feature scopes;
O/G timing differences do not establish equal-quality speedups. All three
paired G/O shading comparisons improved in each measured workload, while
this SER implementation regressed against G in every paired shading sample.
ABeautifulGame's O process medians ranged from 4.461 to 5.330 ms; retain this
dispersion rather than treating the displayed median as an exact prediction.
SER currently reorders both primary/path and ordered shadow traversal sites;
these results do not isolate their individual costs or establish the cause of
the regression. No production default was changed.

Strict SER correctness remains **unqualified**. All 78 validation-enabled
correctness processes completed, 226/226 SER HDR readbacks passed, all 39
same-arm pairs (591 readbacks) were byte-identical, and all six lighting
contribution controls passed. However, 22 hardware-depth guide comparisons
failed byte equality (maximum absolute error 1.1920928955078125e-7). The 512x512
performance readbacks additionally had six failing HDR components in
ABeautifulGame in each of its three SER comparisons (maximum absolute error
0.0008562114089727402); their same-arm repetitions were identical. Thresholds
were not relaxed and these performance observations are not an accepted SER
optimization.

Actual captured single-guide raygen SPIR-V shrank from 2,851,044 bytes for O to
2,189,144 for G; S is 2,189,792 bytes. The glTF variants retain one rolled NEE
source loop and one common inlined BSDF evaluation region. S contains two each
of EXT Trace/Reorder/Execute and no NV invocation-reorder operations. This is
SPIR-V evidence, not driver machine-code or register-pressure evidence.

Local raw packages, per-process values, plots, numerical failure details,
manifests and reproduction commands are under
`build-release/vkgltf-ser-20261017/`; start at `report-final/report.md`,
`ImplementationValidation.md` and `ir-review/Review.md`. These generated
artifacts remain outside source control.

## Original OpenPBR traversal experiment

`ScenePathTracePass` can run the existing OpenPBR integrator through either a
compute shader with inline RayQuery (the default) or a Vulkan ray-tracing PSO.
Select the experimental backend in the pass properties:

```json
{
  "traversalBackend": "rayPipeline",
  "bsdf": "openpbr",
  "cacheMode": "off",
  "splitPrimaryGuides": false
}
```

Use `traversalBackend: "rayQuery"` to return to the original backend. Changing
this setting rebuilds the graph and invalidates accumulation. The experimental
path requires an enabled `DeviceCapabilities::rayTracingPipeline`, CLAS, and the
same bindless resource ABI as the existing path. Editor and preview devices request
RT pipeline support as an optional feature. Unsupported devices retain the default
backend; selecting the experiment reports an error instead of silently falling back.

This first experiment supports OpenPBR path tracing with the radiance cache off.
It supports the existing single-sample guides and general/multi-sample integrator.
SHARC, split-primary guides, legacy glTF BSDFs, realtime lighting and Deferred are
outside this backend's current scope and are explicitly rejected.

## Shared shading and ordered transparency

The split follows the raw-hit design in `vk_gltf_renderer`: ray generation owns
the integrator, while closest-hit returns distance, instance/cluster/primitive IDs
and barycentrics. `traceStreamClosest` reconstructs the surface with the same
function as the inline-query implementation. Any-hit uses the existing deterministic
coverage test. It consumes no path RNG. Layered BLEND and nested-medium shadow
transmittance continue through the existing ordered closest-hit loop.

The experiment does not enable SER or replace the OpenPBR evaluation, sampling,
light selection, guides, accumulation or material-value programs. The request helper
preserves the canonical shader defines and native specialization values; a separate
global `METALLIC_PT_RAY_PIPELINE` policy selects the traversal implementation.

The native pipeline has raygen, miss and one triangle hit group containing both
closest-hit and any-hit. Its recursion depth is one because path recursion remains
an iterative raygen loop. `RayTracingKernel` retains the four shader modules,
pipeline, SBT and encoded parameters through command completion. Render graph
accesses use `RayTracingShader` stages, and command diagnostics count `traceRaysCalls`
separately from compute dispatches.

CLAS pipelines set
[`VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV`](https://docs.vulkan.org/refpages/latest/refpages/source/VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV.html)
and the hit stages read `ClusterIDNV` using Slang's `GetClusterID()`.

## Measurement contract

`Tools/Perf/PathTraceTraversal.py` prepares matched graphs and runs the production
MaterialBaseline harness. The only graph difference between arms is
`traversalBackend`. Correctness captures use Vulkan validation and retain HDR,
guides and temporal readbacks. Ordinary GPU timings disable validation and use
the same dimensions, cameras, quality, cache policy and warmup. Independent
processes run in AB, BA, AB order; frame samples are not independent experiments.

`Tools/Perf/PathTraceTraversalPSO.py` replays actual captured recipes through
`MetallicShaderPipelineBench`. For example:

```powershell
python -B Tools/Perf/PathTraceTraversalPSO.py run --exe build-pass-stages-nrd/Source/MetallicShaderPipelineBench.exe --pair single-guides <compute.recipe.json> <ray-tracing.recipe.json> --output <new-output-directory>
python -B Tools/Perf/PathTraceTraversalPSO.py verify <output-directory>
```

RT recipes use schema version 3 and carry all four stage inputs, specialization
values, recursion depth, CLAS policy and bindless ABI. Compute v1/v2 and graphics
v1 recipes remain readable. Native timing covers only `vkCreateComputePipelines`
or `vkCreateRayTracingPipelinesKHR`; it excludes Slang compilation, shader-module
creation and SBT allocation/upload. Application pipeline caches are empty and
controllable driver internal caching is disabled for this measurement. This does
not guarantee that hidden compiler or operating-system state is cold.

Keep captures, plots, raw logs and generated fixtures in local build output.
Results apply to the measured pass and workload, not editor FPS or other scenes.
Compilation and SPIR-V validation alone do not qualify a new traversal backend.

`PathTraceTraversal.py run` keeps its original strict gates: HDR component error
must be at most `2e-5 + 2e-5 * abs(A)` and guide readbacks must be byte-identical.
Numeric failures retain their raw evidence and return a nonzero exit code.
`verify <directory> --allow-numeric-failure` checks a complete sealed capture for
diagnosis but still reports `correctness_qualified: false`; it does not relax the
thresholds or qualify performance acceptance. The native PSO verifier canonicalizes
the recipe's hexadecimal hash and the benchmark's numeric hash before comparison.

## Initial experiment (2026-10-17)

On RTX 5070 Ti / driver 617.42, the additional backend remains experimental.
All 28 correctness processes completed with core and synchronization validation,
and both direct-lighting controls responded, but all 14 configurations failed the
strict HDR/guide equivalence gate. An isolation probe using the original RayQuery
inside raygen still differed, so TraceRay/payload is not necessary for the
differences. Their precise driver-codegen source has not been established.

With three independent process pairs, single-sample guides native PSO creation
medians were 41.65 s (RayQuery) and 20.99 s (RT); general guides were 46.37 s and
56.37 s. These exclude frontend compilation and SBT setup. At 512x512, shading
scope medians were 7.48/14.96 ms (Opaque1), 33.45/68.78 ms (Mixed1), and
1.99/4.58 ms (ABeautifulGame). Mixed1 had substantial run-to-run timing variation.
The RT path was slower in all measured GPU pairs, and its numerical differences
mean these measurements do not qualify an equivalent replacement.

Local evidence lives under `build-release/openpbr-ray-pipeline-20261017`:
`correctness-full`, `performance-recovered`, `native-pso`, and `report-final`.
An execution-session interruption during an uncompleted native creation required
resuming four GPU processes; fourteen complete processes were preserved verbatim,
and no complete sample was excluded. The recovery record documents cache and
timing-continuity limitations. The default backend remains `rayQuery`.
