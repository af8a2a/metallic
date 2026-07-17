# RTXDI / ReSTIR DI Sample

Metallic includes a native many-light direct-illumination pass inspired by the
fused spatiotemporal ReSTIR DI flow in `E:\RTXDI\Samples\Minimal Sample`. The
implementation uses Metallic's render graph, scene buffers, ray-query path, and
history-resource manager. It does not copy or link the NVIDIA RTXDI SDK source,
so the repository does not acquire a machine-local SDK dependency or redistribute
SDK-licensed code.

## What is implemented

`SceneRtxdiPass` generates hundreds of animated analytic lights and evaluates
their direct contribution against glTF materials. Each pixel performs:

1. Uniform initial light sampling, optional initial visibility, and weighted
   reservoir updates.
2. Previous-frame reprojection with position and normal rejection.
3. Temporal reservoir combination with a bounded history length.
4. Spatial combination from reprojected neighboring reservoirs.
5. One RayQuery visibility test for the selected light and final tone mapping.

The pass stores double-buffered reservoir, world-position, and shading-normal
textures through `HistoryResourceManager`. Authored world-space geometric and
shading normals remain stable while constructing TBN; face-forwarding is applied
only after normal-map evaluation.

This is the ReSTIR DI portion of RTXDI. ReGIR, ReSTIR GI, SDK light structures,
and the RTXDI SDK's optional bias-correction modes are outside the current scope.

## Build and run

```powershell
cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON
cmake --build build --target MetallicRtxdiSample --config Debug
build\Source\Debug\MetallicRtxdiSample.exe
```

The standalone executable opens the editor with the `RTXDI / ReSTIR DI` sample
selected. A non-interactive two-frame validation path is also available:

```powershell
build\Source\Debug\MetallicRtxdiSample.exe --smoke-test
```

The sample graph is
`Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json`. Its inspector exposes
light count, initial candidates, initial visibility, spatial neighbors, history
length, temporal and spatial reuse, light animation, intensity, exposure, and
debug views for the selected light and reservoir history. The default sampling
budget follows the RTXDI Minimal Sample: eight initial local-light candidates and
one spatial neighbor.

The shader and two-frame GPU preview tests can be run with:

```powershell
build\tests\Debug\MetallicRhiTests.exe --filter render_graph_rtxdi_shader_compile
build\tests\Debug\MetallicRhiTests.exe --filter render_graph_rtxdi_preview
```

A Vulkan device with acceleration-structure and ray-query support is required.
