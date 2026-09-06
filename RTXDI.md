# RTXDI / ReSTIR DI Sample

Metallic includes a native many-light direct-illumination pass inspired by the
fused spatiotemporal ReSTIR DI flow in `E:\RTXDI\Samples\Minimal Sample`, with
the signal decomposition and NRD RELAX integration patterned after
`E:\RTXDI\Samples\FullSample`. The implementation uses Metallic's render graph,
scene buffers, ray-query path, history-resource manager, and vendored NRD
integration. It does not copy or link the NVIDIA RTXDI SDK source, so the
repository does not acquire a machine-local RTXDI SDK dependency or redistribute
SDK-licensed code.

## What is implemented

`SceneRtxdiPass` defaults to `lightSource = "scene"` and evaluates the same
directional, point and spot virtual lights used by the real-time renderer and
path tracers, including imported glTF lights. Source-node visibility and native
light enablement are resolved during collection; converted import metadata
does not emit a duplicate light. Each pixel performs:

1. ReGIR grid selection for local lights, hierarchical PDF-mipmap importance
   sampling for fallback and the HDR environment, optional initial visibility,
   and weighted reservoir updates.
2. Previous-frame reprojection with position and normal rejection.
3. Temporal reservoir combination with a bounded history length.
4. Spatial combination from reprojected neighboring reservoirs.
5. One RayQuery visibility test for the selected light.
6. FullSample-style diffuse/specular demodulation and RELAX signal packing.

The pass stores double-buffered reservoir, world-position, and shading-normal
textures through `HistoryResourceManager`. Authored world-space geometric and
shading normals remain stable while constructing TBN; face-forwarding is applied
only after normal-map evaluation.

Following the RTXDI FullSample preprocessing convention, finite local-light
base-level weights are proportional to emitted power: point intensity is
integrated over the sphere and spot intensity over its actual squared cosine
falloff. Directional lights use illuminance as a positive importance proxy,
since an infinite directional source has no finite total power. These proposal
weights never replace physical lux/candela evaluation. Environment texel weights are
`luminance * sin(theta)`, which accounts for lat-long texel solid angle. Both
distributions are padded to power-of-two dimensions and reduced into complete
R32_FLOAT mip chains with 2x2 averaging. Mirroring FullSample's
`PrepareLights.hlsl` flow, `PrepareLightsPdf.slang` performs the preprocessing
entirely on the GPU: it writes local-light power into a Z-curve base level,
derives the environment base level directly from the HDR texture, and dispatches
each mip reduction with an explicit UAV dependency barrier. The local-light PDF
is refreshed every frame, while the environment PDF is rebuilt when its source
resource changes; neither path needs a PDF readback or CPU mip upload.

The RTXDI shader descends those mip chains from the root, selects among four
children by relative weight, and accumulates the discrete selection probability.
The environment probability is divided by the sampled texel's solid angle before
reservoir weighting. Zero-energy inputs fall back to a uniform distribution over
the valid lights or environment texels.

Local-light selection uses a Metallic-native Grid ReGIR modeled after
FullSample's `PresampleReGIR.hlsl` flow. `BuildReGIR.slang` covers the scene
bounds with a configurable 3D grid and constructs a fixed number of RIS light
slots per cell on the GPU. Each slot draws multiple candidates from the global
power PDF, evaluates a light target using the fitted average distance to the
cell volume, and stores the selected light with its RIS inverse source weight.
Directional targets are distance-independent. Point and spot targets retain
support throughout the volume: they do not reject a light just because the cell
center lies outside its cone or finite range. Exact range and cone attenuation
are evaluated at the receiving surface. During initial ReSTIR sampling, the
surface position (with configurable cell jitter) selects a cell and one of its
slots. Surfaces outside the grid, or an unavailable grid, use the global
power-PDF selector. An empty slot in a valid cell is a **zero-weight sample**;
it does not trigger another global sample, which would change the estimator.
The structure is rebuilt every frame, including zero-light scenes, with explicit
storage write-to-read barriers before transport consumes it.

`SceneLightResources` owns the compact physical records and shared GPU sampling
resources. Collection is camera-independent and covers the complete resolved
scene plus its eligible virtual lights. DrawSet and clustered LightGrid lists
are view-filtered raster candidates and must not be reused for secondary path
vertices, reflections, or transport outside the current camera view.

The common `PunctualLightSampling.slang` contract uses binding 50 for the
header-prefixed physical light buffer, binding 52 for the three-record ReGIR
header followed by RIS slots, and binding 53 for the global power-PDF mip chain.
These compact indices are not GPUScene's stable source-slot indices. Buffer
replacement retires old resources for in-flight frames. Resolved light changes
invalidate RTXDI reservoir history and the path tracer's accumulation/radiance
caches, preventing deleted or reordered lights from reusing stale indices.
Cancelled GPU recordings invalidate the sampling allocation; the next build
recreates the PDF/grid resources rather than assuming cancelled texture-layout
transitions executed. This keeps first-build and resize retries valid.

Standard and OpenPBR path tracing, NRC/SHARC cache query/update paths, and RTXCR
hair direct lighting use the same selector with one virtual-light sample per
eligible path vertex. Its contribution is multiplied by the RIS inverse source
weight. Directional, point and spot lights are delta distributions: BSDF or
environment direction sampling cannot hit them, so their MIS weight is one.
Environment next-event sampling retains its separate solid-angle PDF and MIS;
there is no second all-lights sum or SH-radiance contribution. Real-time physical
lighting retains its deterministic full-light evaluation.

The sample graph then runs four passes:

1. `SceneRtxdiPass` writes raw preview color plus demodulated diffuse and
   specular radiance/hit-distance signals, packed normal/roughness, screen-space
   motion vectors, linear view depth, base-color/metalness, and emissive data.
2. `RtxdiConfidencePass` follows FullSample's confidence preprocessing flow.
   It selects the brightest direct-light signal in each 3x3 gradient stratum,
   compares current and motion-reprojected previous diffuse/specular luminance,
   applies four A-trous filter iterations, converts the relative gradients to
   confidence, and applies the same `power = 0.25` non-linear short-history
   filter. The resulting diffuse and specular R8_UNORM textures are available
   as graph outputs.
3. `NrdDenoisePass` runs `RELAX_DIFFUSE_SPECULAR`. It receives current and
   previous camera matrices, advances RELAX history across frames, and resets
   history when the camera or graph is reset. The confidence textures are bound
   as `IN_DIFF_CONFIDENCE` and `IN_SPEC_CONFIDENCE`, with
   `isHistoryConfidenceAvailable` enabled for RELAX.
4. `RtxdiCompositePass` remodulates the denoised diffuse signal by diffuse
   albedo, remodulates the denoised specular signal by dielectric/metallic F0,
   adds emissive/background radiance, and performs exposure and tone mapping.

All linear RTXDI outputs, including NRD signals and emissive/background, receive
the same physical EV100 exposure after transport. The composite therefore only
applies its existing artistic exposure and tone map, not EV100 a second time.
There is no constant ambient-light term. Directional shadow rays remain
unbounded, while the exported NRD hit distance is finite for its half-float
signal format. Environment visibility gates both HDR backgrounds and the
procedural preview fallback; hiding the environment never leaves a synthetic
sky in the color or emissive output.

The normal/roughness and radiance/hit-distance encodings use NRD's own front-end
helpers, so they track the configured NRD build. Motion vectors use the NRD
convention `previousUv = currentUv + motionVector`, with a scale of `(1, 1)`.

This is the ReSTIR DI portion of RTXDI with Grid ReGIR. Onion ReGIR, ReSTIR GI,
SDK light structures, and the RTXDI SDK's optional bias-correction modes are
outside the current scope.

## Build and run

For a multi-configuration generator such as Visual Studio:

```powershell
cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON
cmake --build build --target MetallicRtxdiSample --config Debug
build\Source\Debug\MetallicRtxdiSample.exe
```

For a single-configuration Ninja build, the executable is instead
`build\Source\MetallicRtxdiSample.exe` (without a `Debug` subdirectory).

The standalone executable opens the editor with the `RTXDI / ReSTIR DI` sample
selected. A non-interactive eight-frame path is also available; multiple frames
are rendered so RELAX history is exercised:

```powershell
build\Source\Debug\MetallicRtxdiSample.exe --smoke-test
```

The sample graph is
`Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json`. It explicitly selects
`lightSource = "bench"` to preserve the animated many-light test bench. This
mode creates ordinary native point-light records on the host and sends them
through the same GPU power, ReGIR and transport path; shaders no longer generate
a separate synthetic light type. Bench lights replace scene lights for that
pass only and are not inserted into the authored document. Select
**Scene / Virtual Lights** to render imported lights or lights edited in the
Physical Lighting panel. Light count, animation and benchmark intensity affect
only explicit bench mode.

The inspector exposes light source and benchmark light count, local-light importance sampling, Grid ReGIR enablement/resolution,
lights per cell, build samples, sampling jitter, initial local candidates,
environment candidates, HDR environment intensity/rotation/visibility,
environment importance sampling, initial visibility, spatial neighbors, history
length, temporal and spatial reuse, light animation, intensity, exposure, and
debug views for the selected light, ReGIR cells, and reservoir history. RELAX
settings include history lengths, A-trous iteration count, diffuse/specular prepass blur radii,
minimum hit-distance weight, anti-firefly, confidence inputs, disocclusion
threshold, denoising range, and validation mode. Confidence preprocessing
exposes the FullSample defaults of four gradient A-trous passes, sensitivity 8,
darkness bias -12 EV, and a 0.75-frame confidence history. The default sampling budget uses eight initial
local-light candidates, four environment candidates, and one spatial neighbor.
The graph presents `Composite.color` through `FinalBlit.color`. `Rtxdi.color`,
`Confidence.diffuseConfidence`, and `Confidence.specularConfidence` can also be
selected for inspecting the pre-denoise and confidence results.

Build `MetallicRhiTests`, then select the executable path for the configured
generator. These Google Test wildcard filters include all six
`regir_virtual_lights_*` tests: GPU power/edit/delete, empty-reservoir probability
mass, cancelled-recording retry, standard PT, OpenPBR PT and RTXDI temporal
rendering.

```powershell
cmake --build build --target MetallicRhiTests --config Debug
# Ninja / single-configuration:
$rtxdiTestExe = '.\build\tests\MetallicRhiTests.exe'
# Visual Studio / multi-configuration: use this path instead.
# $rtxdiTestExe = '.\build\tests\Debug\MetallicRhiTests.exe'

& $rtxdiTestExe --rhi-validation '--gtest_filter=*render_graph_rtxdi_shader_compile:*importance_pdf_size:*regir_grid_layout:*regir_virtual_lights_*'
& $rtxdiTestExe --rhi-validation '--gtest_filter=*render_graph_rtxdi_preview'
```

A Vulkan device with acceleration-structure and ray-query support is required.
