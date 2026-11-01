# VSM GPU marker attribution

In the Shadows runtime settings, enable **Detailed GPU Profile**
(`benchmarkGPUProfile`) and, for build details, **Profile Build Substages**
(`benchmarkBuildSubstages`). The Sponza VSM sample enables both. Other graphs keep
the existing opt-in behavior. Disable detailed profiling before collecting a
normal, minimally instrumented performance baseline. CPU attribution
(`benchmarkCPUProfile`) and detailed GPU profiling remain mutually exclusive.

The existing `VSM reset`, `VSM mark`, `VSM allocator`, `VSM build`, `VSM raster`
and `VSM projection` / `VSM one-pass projection` parent names are unchanged.
The Metallic profiler and GPU capture debug-label hierarchy expose these children:

| Parent | Children | Scope |
|---|---|---|
| Reset | prepare, dispatch, publish | Recycled/cache input dependencies, reset kernel, output dependency |
| Mark | dispatch, publish | Receiver marking kernel, page/statistics dependency |
| Allocator | count, allocate, admit, physical, assign | Each kernel **includes its following publication dependency** |
| Build | existing direct/compact, compact sync, indirect, finalize sync, finalize | Optional existing build attribution; finalize includes trailing read/draw dependencies |
| Raster (or static/dynamic layer) | prepare, clear and restore, coarse, fine, publish | Atlas transition/rendering setup, dirty-page clear/static restore, indirect coarse/fine draws, rendering end/atlas publication |
| Raster | diagnostics publish | Cache diagnostic/workload buffer publication |
| Projection, including one-pass | prepare, dispatch, publish | Output dependencies, projection kernel, output/statistics dependency |
| Stabilize SMRT shadows | temporal prepare, temporal dispatch, temporal publish | History dependencies, temporal kernel, shared output dependency |

Cold attachment clearing belongs to raster **prepare**. The **clear and restore**
range may be empty on a cold frame; it measures the explicit dirty-page clear or
static restoration draw when present. No per-batch scopes are added. Conditional
paths such as layering or one-pass projection only appear when actually executed.

Parent durations already contain their children: never add both to a total.
These are command-range measurements using the existing BOTTOM_OF_PIPE timestamp
implementation. Additional queries can alter scheduling; a `publish` interval is
not an isolated barrier-cost estimate, and a `dispatch` interval is not shader
instruction attribution. RenderGraph's automatic input transitions are recorded
outside its stage scope. CPU section time measures command recording, not GPU work.

A single Projection dispatch cannot expose independent receiver/ray/sample timings
through host-side markers. Use shader profiling or separately controlled workload
diagnostics after establishing that the dispatch itself is the hotspot. Splitting
the shader into dispatches only to time it would change the workload being studied.

The profiler has a 256-section per-node budget shared by CPU/GPU sections. Detailed
markers repeat across lights/faces; inspect `profilingOverflow` when scaling beyond
the single-directional-light Sponza case. Overflow or missing timestamps are not
zero-cost work. The acceptance runner rejects incomplete profiling frames.

For a bounded diagnostic run, use `Tools/Perf/VirtualShadowMapAcceptance.py` with
`mode=diagnostic`, an explicit `--runs 1`, fixed camera/settings and post-warmup
frames. Its verified JSON already contains section names, parent indices and GPU
milliseconds. Keep workload atomics disabled when diagnosing marker ranges unless
workload counts are specifically needed; such counts add their own instrumentation.

Generate the nested report without launching another GPU process:

```powershell
python -B Tools/Perf/VirtualShadowMapMarkerProfile.py <acceptance-package> <new-report-directory>
```

The reporter re-verifies the package, checks parent ordering and direct-child
duration sums, then aggregates repeated faces by their complete parent path.
It exports per-process post-warmup medians, frame data and separate parent/child
charts. Conditional missing markers remain missing instead of becoming zeros.
