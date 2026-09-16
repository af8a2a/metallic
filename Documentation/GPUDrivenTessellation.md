# GPUDriven material displacement tessellation

This implements bounded GPU tessellation in the existing visibility pipeline. It does not add a path-tracing pass or change GPUDrivenSample's streamed MiniZorah default. The feature is opt-in (`VBuffer.tessellation`, default false).

## Try it

Select **GPU Driven / Material Displacement** in the editor's sample menu. The repository contains an original procedural plane and height image in `Asset/Tessellation/DisplacedPlane.gltf`. The example uses the real-time deferred graph, DLSS-SR, automatic exposure and optional DLSS-NR. Its initial resident mode allows it to open without a generated stream cache.

To use exactly the same asset through Streamer, cook once from the repository root:

```powershell
build-release/Source/Metallic.exe --build-meshstream Asset/Tessellation/DisplacedPlane.gltf --output Asset/MeshletCache/Tessellation/DisplacedPlane.meshstream.bin
```

Then set `VBuffer.enableMeshletStreaming=true` in the example graph. The stream path is already configured. Cooked pages must carry position, normal and UV0 streams; position-only pages cannot evaluate authored height displacement.

For an existing graph, connect `VBuffer.domain` to `Deferred.domain` in addition to visibility, depth and rasterInfo. The shipped real-time and VBuffer comparison graphs already have that connection. A displaced VBuffer without its matching domain input fails explicitly instead of shading the base triangle by accident.

## Authoring

Displacement is project-specific glTF material **extras**, not a standard glTF extension:

```json
"extras": {
  "METALLIC_displacement": {
    "texture": { "index": 0, "texCoord": 0 },
    "magnitude": 0.45,
    "center": 0.5
  }
}
```

Height is the linear red channel, sampled with repeat addressing and bilinear filtering at mip 0. The texture must be decodable as RGBA8. The texture-info UV transform is supported; only UV0 is currently supported. Height displacement is `(height - center) * magnitude` along the interpolated world-space authored normal. Magnitude is in world units, independent of instance scale; negative values invert displacement. Center must be in [0,1]. Zero magnitude disables patch expansion for that material.

The material inspector exposes magnitude and center under Surface details. They participate in undo/redo, scene-document save/reload and material revisions. An edit republishes the immutable GPU material table and invalidates the HZB history. No graph reload is required.

| VBuffer property | Default | Meaning |
|---|---:|---|
| `tessellation` | false | Enable the task/mesh displacement path |
| `tessellationEdgePixels` | 8 | Target edge length in render pixels, clamped to [1,256] |
| `tessellationMaxFactor` | 4 | Maximum per-source-edge factor, clamped to [1,8] |

Rate uses the unjittered render view, endpoint depth and world edge length. Foreshortening does not suppress detail. Both sides of a shared edge compute the same rate, and boundary positions use a canonical endpoint order. Different normals, UVs, materials, transforms or independently simplified LOD boundaries can still create displacement seams; this is not a general seam-repair system.

## Pipeline

1. Existing Streamer frontier and residency determine the safe geometry cut. No additional detail requests or changes to CLAS publication are introduced.
2. Instance and meshlet frustum/HZB tests inflate world-space spheres by the maximum authored displacement. Normal-cone rejection is disabled when that bound is nonzero. The existing conservative two-pass HZB path remains active.
3. Task shaders expand displaced clusters to source-triangle patches. Resident tasks use a wave prefix sum, with a small shared-memory fallback where task wave operations are unavailable. Stream tasks operate on published active-cut clusters. Unmodified clusters retain a single mesh workgroup.
4. Mesh shaders evaluate an original 512-entry topology table, covering all independent edge rates 1..8. Each source patch emits at most **61 vertices / 96 triangles**; output storage is statically bounded. Clamping reduces quality rather than dropping patches on overflow.
5. Rasterization keeps the original visibility ID. A second RGBA32F attachment stores original-triangle barycentrics, UV/world-area LOD scale, and a 24-bit octahedral microtriangle normal. The depth winner also writes the corresponding domain. Deferred material binning continues to use the original instance/material.
6. Deferred shading reconstructs the displaced point from depth and interpolates original material attributes using domain barycentrics. Its motion/depth guides therefore describe the displaced visible surface. Static displacement works with camera motion and DLSS-SR. The normal is geometric; normal mapping is applied afterward using a stable tangent frame.

The extra attachment costs 16 bytes per render pixel while active (about 31.6 MiB at 1920x1080). When disabled, the graph port uses an R8 placeholder, has no raster writes or shader reads, and keeps the ordinary shader entries and hybrid rasterization. Active tessellation uses hardware rasterization for every cluster in that VBuffer; software microtriangle rasterization is not implemented here.

## Relationship to Nanite and remaining scope

The design takes the concepts of symmetric edge factors, bounded dicing patterns and source-domain reconstruction from the local Unreal 5.7.4 `NaniteTessellation.ush` / `NaniteDice.ush` reference. The topology generator and implementation are original; Unreal source was not copied into this repository. See Epic's [Nanite documentation](https://dev.epicgames.com/documentation/unreal-engine/working-with-naniteenabled-content).

This is not the complete Nanite tessellator: no recursive large-patch splitting, global microtriangle work queue/budget, software dicing raster path, displacement-aware LOD error baking, displacement mips or animated displacement history. Large source triangles can hit the factor cap before reaching the requested pixel size. Supply sufficient base geometry and raise the cap deliberately.

BLAS/CLAS/TLAS still represent the **base mesh**. Ray-traced shadows, reflections and reference path tracing do not intersect the displaced surface. The displacement demo omits the ray-shadow pass and disables Deferred's fallback ray-shadow queries (`debugDisableShadows=true`) for this reason. Resident alpha-masked rasterization retains its coverage test; streamed masked displacement is rejected explicitly at binding creation. Stream pages without the required attributes retain ordinary geometry in the low-level shader fallback. Height decoding/UV0/texture-budget failures are reported at binding creation.

## Verification

`RhiValidation.tessellation_pattern_coverage` checks all 512 patterns: positive winding, exact domain area, manifold interior edges, independent boundary sample counts, and mesh-output bounds.

`RhiRendering.tessellation_displacement_render` compares GPU displacement against explicitly baked geometry, base color and flat microtriangle normals. It covers resident and streamed pages, perspective/orthographic views, both depth conventions, mirrored instances, unbinned drawing, and capped/adaptive independent edge factors (64 image comparisons); it also changes and restores material magnitude without rebuilding the graph. This oracle caught a repeat-seam sampling defect: using nonnegative integer texel coordinates avoids negative-remainder behavior at UV=0/1.

Material scene tests cover scalar validation and document round trips. Final validation on RTX 5060 / 610.47:

- Release builds: Metallic, MetallicGPUDrivenSample, MetallicRhiTests and MetallicSceneTests.
- 13 focused RHI tests passed with Vulkan validation: topology, displaced render oracle, wave distribution, stream cut/budget, hybrid raster equivalence, mixed producer, graph sample loading and PSO cache invalidation. After extending the oracle to adaptive factors, both tessellation tests passed again (64 comparisons plus live edit/restore).
- Both material editing/document round-trip scene tests passed.
- The editor sample ran the complete deferred / DLSS-SR / exposure pipeline; FinalBlit was captured via metallicctl at 1198x438 (799x292 internal render). The original procedural height image is embedded as a glTF bufferView. Raw capture and PNG are in `.cache/tessellation/demo-verified/`.
- The live-edit oracle caught a residency bug: VBuffer's compile reuse key included material revision. It now leaves material refresh to GPUScene/binding updates and retains the stream cut; restoring magnitude reproduces the original image in the first frame.

The original 3000-frame MiniZorah replay completed with tessellation disabled. The final cut matches the prior wave-distribution run: 19,394 active groups, 162,989 selected clusters, early HW/SW 26,806/19,172 and late HW/SW 200/361; zero fallback instances, invalid requests or load failures, zero visible over-target refinements, maximum visible error 1.499928 px.

**Timing cannot establish performance neutrality.** Concurrent Unity 3D utilization had median 57.9%, P95 98.5% (versus 23.8% / 68.7% in the earlier run). Return-hold VBuffer GPU P50 was 8.476 ms versus 4.125 ms, while detail demand was 0.2374 ms versus 0.2380 ms. Keep the slowdown in the record rather than attributing it entirely to either this patch or interference. An isolated A/B run is needed for that conclusion. No other application was stopped. After the completed report and terminal pass result, the existing replay script reclaimed its own process after the known teardown timeout (`forcedCleanupAfterCapture=true`).

[GPUDrivenTessellationResults.json](GPUDrivenTessellationResults.json) retains all phase distributions, final cut data, manifest hashes and the process outcome. Raw artifacts are under `.cache/tessellation/`; no timing frames were removed.

```powershell
./build-release/tests/MetallicRhiTests.exe '--gtest_filter=*tessellation*:*stream_wave_work_distribution:*meshlet_lod_stream*:*hybrid_raster_scene_equivalence:*render_graph_gpu_driven_mixed_producer_render:*render_graph_sample_load:*pipeline_cache_persistence_and_shader_invalidation' --rhi-validation --output-dir .cache/tessellation/validation
./build-release/tests/MetallicSceneTests.exe '--gtest_filter=SceneEditing.Material*'
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/tessellation/roam -Cases m1 -Realtime -QualityWithoutValidation
./build-release/Source/MetallicGPUDrivenSample.exe --sample gpu-driven-tessellation --debug-control
```
