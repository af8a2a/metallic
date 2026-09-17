# GPUDriven material displacement tessellation

This implements recursive GPU tessellation in the existing visibility pipeline. It does not add a path-tracing pass or change GPUDrivenSample's streamed MiniZorah default. The feature is opt-in (`VBuffer.tessellation`, default false).

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
| `tessellationMaxFactor` | 4 | Maximum per-leaf-edge dicing factor, clamped to [1,8] |
| `tessellationMaxSplitDepth` | 2 | Recursive split depth, clamped to [0,3]; 0 retains single-patch dicing |

The three quality controls are live settings: resident and streamed draws send
edge pixels, leaf factor and split depth through push constants every frame.
Editing them does not rebuild the graph, compile shaders or republish the
material/pattern table. Temporal history is invalidated when edited. Only the
`tessellation` feature toggle still requires a graph rebuild to change shader
entries and the domain attachment format.

### Visualize generated triangles

Choose **VBuffer > Visualization > Tessellated Triangle ID**
(`visualization="tessellatedTriangle"`). Each actual diced triangle has an unlit
flat color, exposing both recursive patch boundaries and leaf dicing. The
existing **Triangle ID** continues to show the original source triangles.
Changing the visualization or the three tessellation quality controls is live.
Without tessellation, the new mode shows the ordinary source triangles.

Resident and streamed geometry both support this mode. Colors use geometry
identity, original triangle, integer patch corners and the diced triangle index,
not temporary draw slots or physical stream allocation addresses. A topology
change can change colors. Tessellation raster writes colors to the existing
`VBuffer.color` output as a third MRT (4 additional bytes written per covered
pixel, no extra texture allocation); its depth winner matches visibility/domain.
The generated primitive index is separate from the original visibility ID, so
deferred material lookup, domain normals and motion reconstruction stay intact.
Frozen-camera raster discards viewport domain/color writes. Other diagnostic
modes still use the existing composite and its **Shaded ID Colors** option.

The generated-triangle regression covers resident/streamed and binned/unbinned
draws: source-vs-generated palette size, stationary colors, live quality changes,
unchanged deferred shading, and viewport motion with a frozen culling camera.
Coarse and dense captures are saved as `CoarseTessellatedTriangles*.png` and
`TessellatedTriangles*.png` in the test output directory.

Rate uses the unjittered render view, endpoint depth and world edge length. Foreshortening does not suppress detail. Both sides of a shared edge compute the same rate, and boundary positions use a canonical endpoint order. Different normals, UVs, materials, transforms or independently simplified LOD boundaries can still create displacement seams; this is not a general seam-repair system.

## Pipeline

1. Existing Streamer frontier and residency determine the safe geometry cut. No additional detail requests or changes to CLAS publication are introduced.
2. Instance and meshlet frustum/HZB tests inflate world-space spheres by the maximum authored displacement. Normal-cone rejection is disabled when that bound is nonzero. The existing conservative two-pass HZB path remains active.
3. Resident and stream task shaders process batches of 16 source triangles. Each lane recursively splits its root in source-triangle domain until all edges fit the leaf dicing limit or the depth budget is reached. A wave prefix sum compacts leaf dispatch counts; a 16-element shared-memory scan supports devices without task wave arithmetic. Stream tasks use the published active cut. Unmodified clusters retain a single mesh workgroup.
4. Each leaf gets a mesh workgroup evaluating the original 512-entry topology table, covering independent edge rates 1..8. A leaf emits at most **61 vertices / 96 triangles**. Depth 3 permits up to **64 leaves / 6,144 microtriangles** per source triangle, while individual mesh outputs remain bounded. Leaves at the depth limit are diced with capped rates; none are dropped.
5. Rasterization keeps the original visibility ID. A second RGBA32F attachment stores original-triangle barycentrics, UV/world-area LOD scale, and a 24-bit octahedral microtriangle normal. The depth winner also writes the corresponding domain. Deferred material binning continues to use the original instance/material.
6. Deferred shading reconstructs the displaced point from depth and interpolates original material attributes using domain barycentrics. Its motion/depth guides therefore describe the displaced visible surface. Static displacement works with camera motion and DLSS-SR. The normal is geometric; normal mapping is applied afterward using a stable tangent frame.

The extra attachment costs 16 bytes per render pixel while active (about 31.6 MiB at 1920x1080). When disabled, the graph port uses an R8 placeholder, has no raster writes or shader reads, and keeps the ordinary shader entries and hybrid rasterization. Active tessellation uses hardware rasterization for every cluster in that VBuffer; software microtriangle rasterization is not implemented here.

## Recursive subdivision

`VisibilityTessellationSplit.slang` implements a task-local depth-first work stack. Every child re-evaluates its unjittered edge metric. An edge is bisected only when its rate exceeds `tessellationMaxFactor`: one marked edge yields two children, two yield three, and three yield four. Unmarked outer edges remain intact. This red/green subdivision permits neighboring patches to finish at different depths without inserting an unmatched boundary vertex. It is an original implementation of the split-then-dice design, not a copy of Unreal's tessellation table or queue code.

Patch vertices are packed integer barycentric coordinates on a 1/4096 grid. Midpoints remain exact at all supported recursion depths. Leaf vertices map directly back to the original source domain; positions, UVs and authored normals are evaluated there, then displacement is applied once. In particular, normals are not repeatedly normalized at intermediate split vertices. Canonical edge evaluation and rational `step/rate` dicing preserve shared-edge samples across opposite winding. Discontinuous source normals/UVs or independently simplified meshlet boundaries still have the authoring limitations described above.

The DFS stack reserves 10 pending entries per root and the task payload reserves 64 leaves per root, covering the maximum depth of 3. There is no append race, queue-overflow discard or partial parent replacement. The task/mesh payload ABI is fixed at **12,372 bytes** at every runtime depth; push constants limit actual splitting and emitted leaves. This trades the former smaller depth-specialized payload for immediate depth changes without shader permutations. Lower depths still reduce executed work, but no longer reduce reserved payload storage.

Both prebinned and unbinned draws have GPU-generated tessellation dispatch arguments. Existing ordinary dispatch records retain their layout and behavior; extra records describe eight 16-triangle tasks per cluster and flatten large dispatches into two dimensions. Recursion requires no new frame barrier, CPU readback or global scratch allocation. The per-source depth budget bounds expansion, **not total scene GPU time**; a scene-wide priority/work budget is still separate work.

## Relationship to Nanite and remaining scope

The design takes the concepts of symmetric edge factors, recursive large-patch splitting, bounded leaf dicing and source-domain reconstruction from the local Unreal 5.7.4 `NaniteTessellation.ush`, `NaniteSplit.usf`, `NaniteDice.ush` and `NaniteRasterizer.usf` reference. The topology generator and implementation are original; Unreal source was not copied into this repository. See Epic's [Nanite documentation](https://dev.epicgames.com/documentation/unreal-engine/working-with-naniteenabled-content).

This is not the complete Nanite tessellator: there is no persistent global patch queue, patch-level HZB culling, scene-wide microtriangle budget, software dicing raster path, displacement-aware LOD error baking, displacement mips or animated displacement history. Unlike Nanite's multi-dispatch global split queue, this implementation keeps bounded recursive work inside each task. Very large source triangles can still reach the depth/leaf-rate budget before the requested pixel size (up to 64 segments per source edge at depth 3 and factor 8).

BLAS/CLAS/TLAS still represent the **base mesh**. Ray-traced shadows, reflections and reference path tracing do not intersect the displaced surface. The displacement demo omits the ray-shadow pass and disables Deferred's fallback ray-shadow queries (`debugDisableShadows=true`) for this reason. Resident alpha-masked rasterization retains its coverage test; streamed masked displacement is rejected explicitly at binding creation. Stream pages without the required attributes retain ordinary geometry in the low-level shader fallback. Height decoding/UV0/texture-budget failures are reported at binding creation.

## Verification

`RhiValidation.tessellation_pattern_coverage` checks all 512 patterns: positive winding, exact domain area, manifold interior edges, independent boundary sample counts, and mesh-output bounds.

`RhiRendering.tessellation_displacement_render` compares GPU displacement against explicitly baked geometry, base color and flat microtriangle normals. It covers resident and streamed pages, perspective/orthographic views, both depth conventions, mirrored instances, unbinned drawing, and capped/adaptive independent edge factors (64 image comparisons); it also changes and restores material magnitude without rebuilding the graph. This oracle caught a repeat-seam sampling defect: using nonnegative integer texel coordinates avoids negative-remainder behavior at UV=0/1.

`RhiRendering.tessellation_recursive_render` adds **192 comparisons** using an independent CPU recursive baker, at depths 1, 2 and 3, with both saturated and adaptive rates. It also exercises live material disable/restore on the streamed recursive path. Its live-quality regression performs **40 edits** across resident/streamed and binned/unbinned rendering: depths 0/1/2/3, leaf factors 1/4/8 and edge targets 256/16/1. Every edit must affect the next frame without making the graph dirty or changing its compiled generation; restoring settings must reproduce the baseline image and surface coverage must remain intact. `RhiRendering.tessellation_recursive_topology` reads actual GPU leaves from **512 roots**: all eight split masks, depth budgets 0..3, near-plane crossing, exact positive domain coverage, paired interior edges/rates, identical rational samples on shared source edges, and poison-guarded unused output slots. It reaches the full 64-leaf budget. Runtime render tests exercise the fixed-capacity task payload and mesh interfaces with Vulkan validation.

Runtime push-constant update (2026-09-17): Release builds of Metallic,
MetallicGPUDrivenSample and MetallicRhiTests passed. Seven focused tests passed
with Vulkan validation and no validation errors: both render oracles (256 image
comparisons plus 40 live quality edits), recursive topology, pattern coverage,
stream CPU/GPU LOD equivalence, hybrid raster equivalence and standalone stream
pass smoke. Build and test logs are `.cache/tessellation/runtime-settings-build.log`
and `.cache/tessellation/runtime-settings-tests.log`.

Generated-triangle visualization (2026-09-17): Release builds of Metallic,
MetallicGPUDrivenSample and MetallicRhiTests passed. Eight focused RHI tests
passed across the validation runs: the two displacement render oracles (256
comparisons, 40 live quality edits and the visualization checks above), recursive
topology, pattern coverage, stream LOD equivalence, hybrid raster equivalence,
standalone stream smoke and pipeline cache invalidation. Final render-oracle
results/captures are in `.cache/tessellation/triangle-debug-verified.log` and
`.cache/tessellation/triangle-debug-verified/`; the other six tests are recorded
in `.cache/tessellation/triangle-debug-clean.log`. That earlier run also records
a compiler failure in an intermediate culling fragment; the final implementation
shares a plain helper between entries instead of calling one shader entry from
another, and both render tests pass on re-run. No Vulkan validity errors remain.
Validation reports a non-invalid unused mesh-output warning for location 6
(`debugPatchSeed`) on the frozen-only fragment interface. The frozen path does
not need that color seed and cannot write the viewport MRTs.

Recursive extension results on RTX 5060 / 610.47 (2026-09-16):

- Release builds completed for Metallic, MetallicGPUDrivenSample and MetallicRhiTests. The 256 image comparisons, GPU recursive topology probe and original LUT coverage test passed with Vulkan validation.
- Updating indirect arguments exposed two old standalone test allocations that still held a single command. Both GPU and readback allocations now hold both records; the CPU/GPU frontier oracle additionally verifies the tessellation dispatch. Frontier, stream traversal-demand and standalone stream-pass smoke tests passed with validation after this correction.
- **Validation limitation:** `meshlet_lod_stream_scene_runtime_cut` and `render_graph_gpu_driven_mixed_producer_render` intermittently fault inside the local Vulkan 1.4.341 validation layer during `vkCmdBindResourceHeapEXT`, with tessellation disabled. An exception trace and link map identify `DescriptorHeap::bind` as the caller; the temporary tracing code is not retained. These two cases and three other stream regressions pass without the layer. The full validation-enabled suite is **not** reported as passing. Traces and failed runs remain under `.cache/tessellation/recursive-map.log`, `recursive-repeat.log` and `recursive-final.log`.
- The example ran recursive displacement through deferred lighting, DLSS-SR and exposure. Its 1198x438 FinalBlit capture is `.cache/tessellation/recursive-demo-capture/RecursiveTessellation.png`.
- The unchanged 3000-frame MiniZorah route passed with tessellation disabled. Its final cut remains 19,394 active groups / 162,989 selected clusters; early HW/SW 26,806/19,172 and late 200/361. No fallback instances, invalid requests, load failures or visible over-target refinements; maximum visible error remains 1.499928 px.
- Return-hold VBuffer GPU P50/P95 are 4.137/4.687 ms, versus 8.476/11.495 ms in the earlier run. Background GPU work differed substantially; this is **not a measured speedup from recursive tessellation**. All phases and process utilization samples are summarized in [GPUDrivenRecursiveTessellationResults.json](GPUDrivenRecursiveTessellationResults.json). The replay script again reclaimed only its own process after the completed report/terminal pass result stalled at teardown (`forcedCleanupAfterCapture=true`, process exit -1).

Material scene tests cover scalar validation and document round trips. The **initial bounded implementation** was validated as follows:

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
