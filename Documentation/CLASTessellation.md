# Streamed CLAS displacement tessellation

Metallic can build ray-traceable displaced microgeometry from the published
streaming RT cut. Enable `rtTessellation` on the geometry producer:
`SceneAccelerationStructurePass`, a path-tracing pass, or a
`VisibilityBufferPass` with `enableClusterRtx=true`. The default is disabled.
The **GPU Driven / Material Displacement** sample enables both raster and CLAS
tessellation and connects the resulting TLAS to the ray-traced shadow pass.
The device must support `VK_NV_cluster_acceleration_structure`.

| Property | Default | Meaning |
| --- | --- | --- |
| `rtTessellation` | false | Enable generated CLAS geometry; changing it rebuilds the graph |
| `rtTessellationEdgePixels` | 8 | Unjittered render-pixel edge target, 1–256 |
| `rtTessellationMaxFactor` | 4 | Leaf edge factor, 1–8 |
| `rtTessellationMaxSplitDepth` | 2 | Recursive split depth, 0–3 |
| `rtTessellationMaxPatches` | 16384 | Global generated leaf capacity; changing it reallocates the bounded arenas |

The three quality controls update the next traversal without shader variants.
Raster and ray quality settings are independent; set matching values when
comparing their surfaces. Material authoring, world-unit magnitude, height
sampling, UV0 transforms and seam limitations are described in
[GPUDrivenTessellation](GPUDrivenTessellation.md).

## GPU workflow

1. Existing streaming chooses and publishes complete RT representations and
   resolves failed dynamic admissions to its terminal fallback cut.
2. One workgroup per displaced instance counts recursive leaves over that
   published cut. This includes visible instances outside the primary frustum;
   raster HZB visibility does not remove shadow/reflection contributors.
3. A bounded prefix pass admits complete instances. A second walk emits source
   page/cluster/triangle, integer patch corners and leaf edge rates.
4. One 64-thread group per leaf evaluates the shared raster split/dice pattern.
   It writes at most 64 vertex slots and 96 triangles, then produces the native
   indirect triangle CLAS build record. Actual patterns use at most 61 vertices.
5. GPU indirect CLAS builds write addresses into a contiguous arena. GPU-generated
   BLAS records group those CLAS addresses by instance. Complete BLAS results
   override their base TLAS instance addresses before the TLAS build.
6. Ray queries and ray-pipeline hits recover source barycentrics/UV/material
   identity from retained microtriangle records. Geometric normal, shading normal,
   tangent projection and ray-cone area describe the displaced microtriangle.

The implementation reuses `VisibilityTessellation` rather than duplicating its
topology generator or height sampler. Vertices are displaced in world space and
transformed back into each instance's object space, preserving world-unit
displacement under nonuniform and mirrored transforms. Each instance gets its
own generated geometry.

Generated cluster IDs start after the asset's entire fixed page/cluster ID range;
initialization checks that the generated namespace cannot wrap. Both candidate
alpha/transparent transport and committed surface decoding use the same source
domain mapping. Generated positions and triangle indices are retained because
Metallic does not use ray-tracing position fetch.

## Capacity, history and lifetime

An instance exceeding the remaining patch capacity keeps its complete base BLAS.
Missing required page attributes or a singular transform also reject the whole
instance. No partially tessellated instance is published. This fallback is
geometrically different from the displaced raster surface; increase capacity or
reduce quality when the diagnostics report rejected instances.

The resource inspector exposes `tessellationCounts` (patches, built instances,
capacity rejections, invalid-input rejections), `tessellationLeaves`, and
`tessellationVertices`. The BLAS header exposes generated patch count and rejected
instance count. Allocation or unsupported native build failures propagate through
the existing renderer error path.

Generated storage reuses the runtime's dependency on all previous ray readers.
Its parameter packet is uploaded through a GPU copy into persistent device memory.
Frame retention covers the component across asynchronous consumers and capacity
replacement. GPU semantic history compares the actual published source cut,
patches, edge rates, transforms and material revision; rebuilding identical
geometry does not reset PT/SHARC. Enable/disable and success/fallback transitions
invalidate history. Cancelled command recordings do not publish geometry or
advance history initialization.
Capacity replacements share the prior GPU history and material generation:
success/fallback transitions invalidate once, while moving an unchanged surface
to a differently sized arena does not invalidate it.

## References and scope

The split/dice concepts follow Brian Karis's
[How to tessellate](https://graphicrants.blogspot.com/2026/02/how-to-tessellate.html)
and [Nanite programmable displacement](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine).
The CLAS workflow is informed by NVIDIA's
[vk_tessellated_clusters](https://github.com/nvpro-samples/vk_tessellated_clusters),
also available locally at `E:\vk_tessellated_clusters`. No Unreal source or
tessellation table is copied.

This version uses transient indirect triangle builds. It does not implement
pattern CLAS templates/instantiation, mixed mini-cluster packing, persistent
patch queues, or a temporal geometry cache. The count/emit walk is bounded but
serial within each instance workgroup. Large displaced instances can therefore
be expensive even when total storage is bounded. No performance improvement is
claimed; template instantiation and more parallel source-triangle scheduling
require separate same-scene measurements.

## Validation entry points

- `clas_indirect_triangle_build`: native indirect count/stride, canaries, range
  rejection and cancelled-command resource lifetime.
- `scene_clas_tessellation`: production RenderGraph CLAS build and ray decoding,
  compared with independently evaluated microtriangles; source UV, geometry
  normal, material changes, mirrored/nonuniform transforms, offscreen queries,
  enable/disable, cancellation and capacity fallback.
- Existing `tessellation_*`, `scene_clas_attributes*`, `stream_blas_cut_cache`,
  `stream_rt_representation_lifecycle` and `stream_clas_cut_readiness` protect
  shared geometry, decoding and publication contracts.

## Executed validation (2026-10-25)

Windows x64 Release, RTX 5070 Ti, NVIDIA 621.01, existing
`build-pass-stages-nrd` configuration (NRD enabled, Streamline disabled):

- `MetallicRHITests`, `LookDev` and `Metallic` builds passed. Seven focused
  shader-request tests passed. Shader module ownership passed for 377 files;
  the new tessellation compute shader also passed SPIR-V validation.
- Both new cases passed the strict testbench with synchronization validation,
  `--tb-require-all`, no skips, and zero captured validation messages.
- Ten existing tessellation, CLAS attribute, cut-cache and RT publication cases
  passed. Two raster cases were first interrupted by a concurrent source-epoch
  change; both then passed completely with the source tree held still.
- A native-resolution copy of the sample executed VBuffer, ray-traced shadows
  (SIGMA enabled), deferred shading and FinalBlit at 1444x694. The readback reported
  **2048 patches, 1 generated BLAS, 0 capacity rejections, 0 invalid inputs**.
  Captured leaf records used 61 vertices / 96 triangles; vertex readback confirmed
  displaced positions. The final image was inspected and no Vulkan warnings or
  errors were logged. The original graph was restored byte-for-byte afterward.
- The current capture excludes DLSS because this configured build has no
  Streamline SDK. It does not establish DLSS integration, performance gains,
  large-scene memory behavior, or long camera-motion stability.

Representative commands (use a fresh output directory for each invocation):

```powershell
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --tb-run --tb-suite core --tb-filter '*scene_clas_tessellation:*clas_indirect_triangle_build' --tb-validation sync --tb-require-all --output-dir .tmp/clas-tessellation/strict
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-validation --gtest_filter='*tessellation*:*scene_clas_attributes*:*stream_blas_cut_cache:*stream_rt_representation_lifecycle:*stream_clas_cut_readiness' --output-dir .tmp/clas-tessellation/regressions
python -B Tools/CheckShaderModules.py
```

Local evidence is under `.tmp/clas-tessellation/`: the strict report is
`tb-sync-final-20261025-1538/report.html`, the sample image is
`CLASTessellationSample.png`, and `sample-geometry/manifest.json` records the
three buffer captures. These generated artifacts are not source-controlled.
