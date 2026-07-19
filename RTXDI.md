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

`SceneRtxdiPass` generates hundreds of animated analytic lights and evaluates
their direct contribution against glTF materials. Each pixel performs:

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

Following the RTXDI FullSample preprocessing convention, local-light base-level
weights are proportional to emitted power. Environment texel weights are
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
cell volume, and stores the selected light with its estimated inverse source
PDF. During initial ReSTIR sampling, the surface position (with configurable
cell jitter) selects a cell and one of its slots. Surfaces outside the grid and
invalid slots fall back to
the global power-PDF or uniform selector. The structure is rebuilt every frame
so animated procedural lights remain synchronized, with an explicit storage
buffer write-to-read barrier before the screen-space RTXDI dispatch.

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
   adds emissive/ambient radiance, and performs exposure and tone mapping.

The normal/roughness and radiance/hit-distance encodings use NRD's own front-end
helpers, so they track the configured NRD build. Motion vectors use the NRD
convention `previousUv = currentUv + motionVector`, with a scale of `(1, 1)`.

This is the ReSTIR DI portion of RTXDI with Grid ReGIR. Onion ReGIR, ReSTIR GI,
SDK light structures, and the RTXDI SDK's optional bias-correction modes are
outside the current scope.

## Build and run

```powershell
cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON
cmake --build build --target MetallicRtxdiSample --config Debug
build\Source\Debug\MetallicRtxdiSample.exe
```

The standalone executable opens the editor with the `RTXDI / ReSTIR DI` sample
selected. A non-interactive eight-frame path is also available; multiple frames
are rendered so RELAX history is exercised:

```powershell
build\Source\Debug\MetallicRtxdiSample.exe --smoke-test
```

The sample graph is
`Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json`. Its inspector exposes
light count, local-light importance sampling, Grid ReGIR enablement/resolution,
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
The graph's default presentation output is `Composite.color`; `Rtxdi.color`,
`Confidence.diffuseConfidence`, and `Confidence.specularConfidence` remain
marked as debug outputs for inspecting the pre-denoise and confidence results.

The shader and eight-frame GPU preview tests can be run with:

```powershell
build\tests\Debug\MetallicRhiTests.exe --filter render_graph_rtxdi_shader_compile
build\tests\Debug\MetallicRhiTests.exe --filter importance_pdf_mip_chain
build\tests\Debug\MetallicRhiTests.exe --filter regir_grid_layout
build\tests\Debug\MetallicRhiTests.exe --rhi-validation --filter render_graph_rtxdi_preview
```

A Vulkan device with acceleration-structure and ray-query support is required.
