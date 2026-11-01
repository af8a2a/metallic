# Grid-driven local VSM projection candidate

`VirtualShadowMapLightsPass` accepts `benchmarkLightGridProjection: true` to
test grid-driven one-pass projection. It defaults to false. The candidate keeps
the existing R8 visibility layers, temporal history, Mark, page allocation and
filtering settings. Four-bit shadow packing is a separate future change.

The pass consumes the matching raster producer's `ClusterLightGridSnapshot` and
declares reads of its parameters, cells, indices and candidates. A producer with
no matching grid retains the original projection. The existing one-pass device
and filter eligibility checks still apply.

## Index and execution contracts

- Grid indices are GPUScene source slots. Publication slots 0/1 are Sun/Moon;
  slot `source + 2` names a local light. Projection jobs have a different order,
  so the CPU publishes a separate slot-to-job-range table.
- LightsPass already rejects a source domain exceeding `lightCapacity`, whose
  maximum is 64 including Sun/Moon. A `uint2` mask therefore covers the complete
  publication domain, including holes and budget-rejected entries.
- Each valid depth receiver queries the grid once. Existing overflow and
  out-of-grid fallback retain all bounded candidates; unbounded local lights
  are added separately. The result intersects the admitted-light mask.
- A subgroup OR creates a common job list. All active lanes traverse identical
  jobs; membership gates per-pixel work, while the existing per-job subgroup
  reductions remain outside that branch. Descriptor indices remain uniform
  within each subgroup.
- Celestial and local jobs share one filtering call site. Actual projection
  retains its original per-job receiver reconstruction. An earlier candidate
  shared that reconstruction and used two call sites; it failed exact image
  comparison and is not retained.
- LightsPass clears the external visibility array to white before projection,
  so omitted local receivers need no store. Debug pixels and receiver/background
  counters may differ where an omitted job previously wrote or counted them.
  The candidate does not promise identical debug visualization.

The grid context is a 544-byte CPU/Slang structure. It uses the projection-only
`ShadowPush.arguments` field; existing non-grid direct callers use the invalid
handle and do not read this context. Resource leases and the frame-owned grid
producer preserve lifetime through command completion.

## Diagnostics

`benchmarkGPUWork: true` exposes `Shadows.projectionGridDiagnostics`, eight
unsigned 64-bit counters stored as low/high uint pairs. Ordinary timing disables
these atomics.

| Counter | Meaning |
|---|---|
| receiverLookups | Valid-depth receivers that query the grid |
| gridFallback | Those queries using the complete bounded-candidate fallback |
| localCandidates | Bounded-list plus unbounded-list entries examined |
| localMemberships | Admitted local memberships after mask intersection |
| waveLocalJobs | Local face jobs traversed by subgroups |
| skippedWaveLocalJobs | Local face jobs absent from subgroup unions |
| projectedLaneJobs | Per-pixel local face-helper calls, before canonical-face/domain rejection |
| possibleLaneJobs | Full-screen local face-helper calls in the original path |

Use `benchmarkProjectionProfile: true` with `benchmarkGPUProfile: false` to
collect Projection scopes without exhausting the render graph's 256-section
per-pass budget on a scene containing many point-light faces. Top-level graph
and pass-stage timestamps remain available. This avoids changing the renderer's
global timestamp allocation.

The bounded experiment, raw captures, compilation audit and review report are
local outputs under `build/vsm-grid-projection-20261101-bounded3/`. Earlier
timestamp-capacity and correctness failures remain in `bounded/` and `bounded2/`
respectively. Consult the report for measured results and remaining coverage;
the implementation alone is not a performance acceptance result.
