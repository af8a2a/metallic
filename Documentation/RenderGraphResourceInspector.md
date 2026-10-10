# Render Graph resource inspection

Open **Render Graph Editor > Resources**, or right-click an output pin and choose
**Inspect Resource**. The list includes Texture and Buffer bindings from executed
passes, plus resources published by their registered debug checkpoints. Filter by
resource or pass name. Input bindings are identified by the consuming pass name.

The source selector offers **All resources**, **Render graph**, and **Subsystems**.
Resources are grouped by owner, then Texture/Buffer; filtering also matches the
subsystem name. Subsystem resources use the same texture (including Texture3D
cube/slice) and typed-buffer viewers, Refresh/Live controls, and capture budget.
The detail panel shows their owner, capture checkpoint and validity notes.

Built-in subsystem publications include:

- `render.gpu-scene`: global instance, geometry, material, vertex/index, meshlet,
  draw and LOD tables. Known layouts are typed; other tables support manual raw
  interpretation. Per-view culling resources retain their producer checkpoints.
- `render.environment`: HDRI radiance/PDF, SH coefficients, prefiltered specular
  and celestial records; physical-atmosphere publications used in this execution additionally
  expose transmittance, multi-scattering, sky view, cloud shadow and aerial data.
  Atmosphere publication revisions distinguish observer/parameter variants.
- `render.streamer`: debug-enabled meshlet sessions, with request headers/lists,
  page tables, active groups and LOD/demand state. Sparse visible-cluster records
  require the original producer checkpoint and its visibility evidence.
  Only sessions recorded in this execution are exposed, not sessions retained
  by another view or executor.

Only resources published by subsystems required by the current execution (and
their dependencies) are listed. Uninitialized or non-readable resources explain
why capture is unavailable. Disappearing resources clear old evidence; changing
an allocation automatically replaces even a frozen capture. Freezing otherwise
keeps the completed snapshot while the resource contents continue changing.

## Publishing subsystem resources

Override `IRenderSubsystem::appendDebugBindings(context, bindings)`. Supply local
resource names, Buffer/Texture pointers, exact current states, buffer ranges,
registered layouts (or `raw`) and allocation versions. Buffers/textures must have
transfer-source usage to be readable. The host prefixes names with
`subsystem.<subsystem-id>.` and supplies owner metadata. Borrowed GPU pointers stay
within the callback; the inspector receives immutable capture metadata/bytes.

The callback runs only with a debug observer, after successful graph execution
and all post-graph hooks, before `endFrame`. Both externally recorded and
self-submitted execution use this boundary; the latter joins every graph queue
first. Existing state-restoring readback and submission tracking apply.
Normal execution does not enumerate or copy debug resources.

`rg.describe` lists these owners separately under `subsystems`, without adding
synthetic graph passes. `capture.batch` routes via
`pass: "subsystem.<subsystem-id>"`, `checkpoint: "AfterGraph"`; each resource uses
its fully qualified ID. Unknown/inactive subsystem checkpoints are rejected.

Select a resource to capture it on the next execution. **Refresh** captures once;
**Live** refreshes at most twice per second, with one pending request. Disabling
Live freezes the last completed snapshot. The displayed execution and generation
identify the data being shown; graph recompilation discards stale snapshots.

- Textures: RGB or individual channels, exposure, min/max display range, fit/zoom,
  scroll, ROI, and exact raw pixel fields/bytes on hover. Display values are
  mapped as `(value * exp2(exposure) - min) / (max - min)` and clamped; non-finite
  results are magenta. This is a diagnostic display, without scene tonemapping.
  Texture3D resources default to **Cube**: drag to rotate a cube whose visible
  faces show the corresponding X/Y/Z boundary slices of the captured ROI.
  Face labels identify the axes; hover reports original voxel coordinates and
  raw values. Rotation and zoom reuse the frozen capture. **Reset view** restores
  the initial orientation. **Z slice** switches to a single XY plane with a slice
  selector. Both modes support channel/exposure/range controls.
- Buffers: first element/count, virtualized rows, hexadecimal values, and
  uint32/int32/float32 interpretation of raw 32-bit words. Producer-declared
  layouts open as named, typed columns. Arrays/vectors expand into component
  columns; hover a header for byte offset/size and bitfield information. More
  than 32 component columns use field pages. **Raw 32-bit words** shows the same
  captured bytes without another GPU copy.

## Typed buffers and StructuredBuffer research

`AutoExposure.exposure` now declares `AutoExposureState`: four `f32` fields named
`multiplier`, `adaptedEV100`, `targetEV100`, and `luminance` at offsets 0/4/8/12,
with a 16-byte element stride. `AutoExposure.histogram` declares `u32`. This
describes the actual `RWBufferSpan<float4>` / `RWBufferSpan<uint>` storage in
`PostProcessParameters.h` and `AutoExposure.slang`; no shader or GPU ABI changed.

An untyped buffer offers **Capture type**, including float/int/uint vectors of
2–4 components and registered schemas compatible with its declared stride.
The scalar `u32`/`i32`/`f32` choices always remain available. This is an explicit
manual interpretation, not automatic type detection. Changing it resets the
range and captures again. First element/count use the selected schema's stride;
the raw-word view labels word indices and retains the original byte offsets.

Pass authors can publish existing schemas on output fields:

```cpp
reflection.addBufferOutput("records")
    .buffer(recordCount * sizeof(VisibleClusterRecord), sizeof(VisibleClusterRecord))
    .bufferLayout("VisibleClusterRecord")
    .storageWrite();
```

Register additional `DebugTypeDesc` schemas in `RenderDebugProviders.cpp`, with
explicit scalar types, byte offsets, array counts and stride. Prefer `offsetof`
and `sizeof` on CPU/shader shared types, plus a GPU layout/readback test. Nested
members can be flattened into named fields with absolute offsets. Padding is
represented by offsets/stride, not displayed as a fictitious member. Input aliases
inherit the producer's schema; only output declarations establish the resource
type. Graph compilation rejects unknown schemas, differing declared strides and
sizes that are not whole elements. Captures carry the schema and its layout hash,
so the native UI and offline `metallicctl` decode the same typed evidence.

The reference `E:/vk_mini_samples/samples/realtime_analysis/realtime_analysis.cpp`
also explicitly supplies `BufferInspectionInfo.format`: e.g. float2 `position`,
`predictedPosition`, `velocity`, `density`. It does not discover a C++/shader struct
from the Vulkan buffer.

Automatic reflection is feasible for genuine `StructuredBuffer<T>`: Slang exposes
the resource element type layout, struct fields and field offsets, array strides,
scalar/vector types and matrix layout. The actual target layout must be used;
type size and array stride are not always equal. See the official
[reflection API guide](https://shader-slang.org/slang/user-guide/reflection.html)
and [StructuredBuffer definition](https://shader-slang.org/stdlib-reference/types/structuredbuffer-0a/).

It is **not implemented automatically in this change**. Metallic currently saves
SPIR-V and dependencies in `ShaderCompileResult`/the disk cache, without serialized
type reflection. Its `BufferSpan<T>` resources resolve `ByteAddressBuffer` and use
`Load<T>`/`Store<T>`; the bound raw descriptor alone does not retain the element
schema or identify a RenderGraph field. A future automatic path needs:

1. Extract target-specific layouts while the Slang linked program is alive and
   serialize them in a versioned shader-cache record, including cache-hit paths.
2. Explicitly map pass fields/parameter paths to element types, covering typed
   spans and multiple views of one allocation as well as StructuredBuffer.
3. Validate scalar widths, nested offsets, array/matrix strides and layout mode;
   preserve generation/hash invalidation across shader reloads. Unsupported or
   ambiguous layouts must remain raw instead of guessing from stride.

The explicit schema path supplies useful type viewing now and can also consume
validated reflection-generated schemas later.

The texture path captures mip 0 and layer 0 of uncompressed 2D images, or
contiguous Z slices of mip 0 of uncompressed 3D images:
RGBA/BGRA8 UNORM/sRGB, R/RG/RGBA16F, R/RG/RGBA32F, R/RG/RGBA32 uint/sint, and D32F.
sRGB images expose their stored UNORM values. Compressed/packed formats
show an explicit unsupported reason. Cube mode reads the volume within the XY
ROI; single-slice mode reads only one plane. Oversized captures require a smaller
ROI or switching to Z slice; the default runtime budget is 16 MiB including evidence metadata. Raw buffer
ranges are measured in 32-bit words, with at most 65,536 requested elements.
For `capture.batch`, each texture resource accepts an optional zero-based `slice`
(default 0) and `sliceCount` (default 1), alongside the existing XY `roi`. Captures
record both in metadata; partial-volume captures have `completeCoverage: false`.
The native bytes are tightly packed in X, then Y, then Z order. Out-of-range
slices are rejected before recording a GPU copy.

Inspection uses the existing RenderDebugRuntime pass-boundary copies and restores
source resource synchronization state. CPU access waits for GPU completion through
polling; opening or closing the local inspector changes graph allocation policy
and causes recompilation. While attached, debug resources are pinned, transfer
source usages are enabled, and debug execution restricts parallel recording.
Therefore timings taken in this mode are not normal renderer performance timings.
Closing Resources detaches the local observer; an explicit `--debug-control`
session keeps its observer active. Preview uploads create separate retained images
so in-flight frames never sample a replaced graph resource or a rewritten preview.

## Native diagnostics without desktop automation

Build `Metallic`, then run from the repository root:

```powershell
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
.\build-release\Source\Metallic.exe --skip-shader-warmup `
    --resource-inspector-smoke .cache\InspectorEvidence
Remove-Item Env:METALLIC_SMOKE_TEST_HIDDEN
```

This uses the real editor and GPU. It verifies known texture/buffer values, preview
GPU upload and display conversion, channel/exposure controls, live/frozen snapshots,
ROI/ranges, invalid ranges, recompile, and close/reopen. Exit code 0 indicates success.
It also checks the first, middle and last slices of a known 3D texture, volume ROI,
invalid slice indices, GPU preview pixels, and exports `volume/` and `volume-ui.ppm`.
Cube checks compare all six uploaded face images with known XYZ values and export
`cube/`, `cube-ui.ppm` and `cube-rotated-ui.ppm`, including opposite-face rotation
without recapturing the volume.
Subsystem checks exercise an independently owned typed buffer, replacement and
disappearance/reappearance, real environment radiance/PDF/SH and GPUScene sentinel
readback, a real physical-atmosphere transmittance LUT, external command recording,
and invalid checkpoint rejection.
Evidence includes `subsystem-buffer/`, `subsystem-environment/` and their `-ui.ppm`
images showing the native grouped Resources panel. `subsystem-atmosphere/` and
`subsystem-atmosphere-ui.ppm` contain the physical-atmosphere LUT evidence.
It also runs the real AutoExposure histogram/reduce/apply shaders on a known HDR
input, verifies typed float fields and histogram weights, tests schema inheritance
through a graph input, checks manual uint4 element offsets, and rejects invalid
producer schemas/strides. Histogram percentiles are fixed to 0–100% in this fixture.
Evidence includes:

- `report.json`: check results, artifact identities and capture provenance.
- `texture/` and `buffer/`: standard `manifest.json` and `0.bin` captures, consumable
  by `metallicctl --capture <directory>` offline.
- `texture-ui.ppm` and `buffer-ui.ppm`: native ImGui rendering of the editor window,
  cropped and read back from an offscreen target. No screenshots, clicks, focus
  changes, or desktop automation are used. HDR UI output is converted to SDR using
  the configured paper white for these images.
- `typed-buffer/` and `exposure/`, plus `typed-buffer-ui.ppm`,
  `typed-buffer-raw-ui.ppm`, and `exposure-ui.ppm`: typed captures and native UI
  evidence, including the actual AutoExposure output.

For example, verify the known fixture's 18th word offline (expected value 292):

```powershell
.\build-release\Source\metallicctl.exe --capture .cache\InspectorEvidence\buffer `
    eval 'buffers["Data.values"][17]' --json
```

In a tests-enabled build, build `Metallic` first and run
`ctest --test-dir <build-dir> -R '^MetallicResourceInspectorSmoke$' --output-on-failure`.
The smoke uses an isolated generated graph, does not save scene/graph changes, and
does not overwrite the user's ImGui layout. Artifacts belong in local output folders.

For a running editor started with `--debug-control`, `metallicctl --pid <pid> hello
--json` exposes `result.engine.resourceInspector`, including the selected resource,
pending job, status, live flag, preview readiness, and captured execution/generation.
Existing `capture.batch`, `jobs.get`, and `artifact.read` commands remain available
for direct capture automation without opening the Resources page.
