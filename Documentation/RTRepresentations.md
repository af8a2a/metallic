# Streamed RT representations

The streamed ray-tracing path separates a BLAS owner from a rendered instance.
Raster LOD traversal still produces per-instance cuts. Each primitive has three
RT representation slots, and each instance selects a slot or its terminal BLAS.
`METALLIC_BLAS_REPRESENTATIONS=0` selects the existing per-instance path for A/B
testing; `1` enables the representation path.

## Ownership and admission

- **Merged:** the highest drawable cut of the primitive's currently resident
  hierarchy. Refinement substitutes complete child groups for their parents;
  it does not concatenate overlapping instance cuts.
- **Shared:** a representative instance's complete raster cut. Conservative
  per-LOD enclosing bounds classify other instances; an instance may share only
  when its minimum permissible LOD is no finer than the representative cut's
  actual maximum LOD. Bounds include rotation and nonuniform scale.
- **Discrete cache:** a complete resident discrete LOD cut, retained across
  frames and temporary lack of consumers. This is an opportunistic cache, not
  a mechanism for pinning geometry pages indefinitely.

BLAS storage is charged once per representation. With owner-priority admission,
only dirty/retried representations consume per-frame references and build slots;
clean cached BLAS do not consume that work budget.
The three slots impose a maximum of three dynamic BLAS owners per primitive,
including a discrete-cache refresh. This does not make instance classification,
TLAS creation, raster traversal or all RT preparation independent of instance
count. The selected representation retains the authored TLAS instance identity
and transform.

The compact RT group buffer has its own capacity. Admission publishes complete
primitive ranges, never a truncated group list. Under pressure, idle cache
storage gives way to active demand. A rejected shared/cache selection may use
an already-ready merged representation of the same primitive; otherwise it
uses the existing terminal fallback.

## Owner demand importance

Each owner also carries demand metadata aggregated from **all** instances that
request it, before budget rejection. A shared cut's representative is not its
importance proxy. The primary metric is the maximum estimated terminal-fallback
error in render pixels; minimum surface distance and maximum projected viewport
coverage are independent auxiliary metrics and may come from different instances.
These estimates describe the cost of the terminal fallback, not a measured
image difference or an exact error delta against the current representation.

The immutable metadata buffer appends one fallback sphere/error record per
primitive after its LOD records. It covers the complete terminal cut, including
branches that terminate early. Terminal groups' own errors are stop sentinels;
the approximation error comes from each terminal cluster's generating group,
with zero error for original clusters. The envelope includes both terminal and
generating groups and is rounded outward.

Projection uses a conservative affine scale bound, including shear. Every
consumer contributes a direction-independent ray error; primary-frustum
consumers additionally use the raster projection bound. Orthographic error is
distance-independent. Offscreen and behind-camera consumers remain eligible for
shadow/reflection use, with zero primary-view coverage rather than zero demand.
This is a camera-based proxy, not an estimate of an individual shadow or reflected
ray's footprint.

`rtRepresentationState` appends six words per owner after the instance records:
`fallbackErrorKey`, `nearestSurfaceDistanceBits`, `maxCoverageBits`,
`consumerCount`, `frustumVisibleConsumerCount`, `offscreenConsumerCount`.
Nonnegative finite float bits support atomic max/min. Error key zero means no
consumers; otherwise `asfloat(key - 1)` is the error, so even exact-zero-error
consumers have a nonzero key. Empty owners have maximum finite distance and zero
coverage/counts. The fields reset and aggregate every executed frame; they do
not invalidate cached cuts or change owner identities.

Owner-priority admission is enabled by default with representations and the GPU
BLAS allocator. `METALLIC_BLAS_PRIORITY=0` selects the previous admission policy
for a same-allocator A/B; `1` enables priority. The legacy storage allocator and
per-instance representation mode keep their existing policies.

The GPU sorts stable owner IDs by error descending, distance ascending, coverage
descending, then ID ascending. A 64-element tile sort followed by parallel merge
passes writes two order arrays and an inverse rank after the importance records.
Group admission uses each primitive's highest-priority owner and reserves the
complete primitive range; output groups retain their original owner order so a
change of priority does not invalidate cut comparisons. Reference and build
admission then traverse that order, skipping impossible requests and charging
only dirty/retried builds. Offscreen consumers participate in the same ordering;
this is eligibility, not a guarantee that every low-priority request can fit.

Known actual sizes first select a ranked working set. Newly built actual sizes
finalize storage selection before Move/Publish. Lower-priority clean losers may
be evicted, at most 256 per frame in the runtime configuration, retaining their
measured size and negative cut cache. A priority-only change can therefore admit
a formerly rejected foreground owner without repeatedly building the unchanged
losers. Fragmented selected requests retain their claim while the existing
bounded relocation pass progresses. Waiting requests consume no build/reference
quota, allowing lower-priority work that fits an existing gap to proceed without
consuming the waiting owner's storage claim. Requests denied by this frame's work
budget neither reserve storage nor evict a clean cache. No full repack is introduced.

Dirty cuts retain their previous allocation size only as a scheduling estimate.
A separate projected balance delays known negative-cache retries that are unlikely
to survive the higher-ranked dirty builds. The estimate is charged only after a
dirty request receives build/reference quota; it never establishes actual storage
ownership or a valid negative cache. Deferred retries preserve their measured cut,
consume no work or real storage claim, and do not evict clean owners. An overestimate
can delay a retry until the next stationary frame; an underestimate still undergoes
exact native-size admission. Homogeneous inactive, fitting clean, and non-fitting
negative tiles use parallel fast paths; mixed tiles retain greedy backfill.

Projection-only deferrals are excluded from `storageRejected`, because the next
frame uses this field to decide whether idle discrete caches face real capacity
pressure. They remain in suppressed/overflow and rejected-consumer diagnostics,
with a separate projected-retry count. Benchmark captures label this convention
with `blasStorageRejectedSemantics = physical-and-known-storage-only-v2`; older
captures without that field include projected deferrals in storage rejection.

Feedback reports requested/rejected visible and offscreen consumers, maximum
rejected error and eviction counts. These are **desired-owner selection** metrics:
a rejected shared/cache owner may still use an available merged BLAS. They are
not actual TLAS fallback counts or measured image error. Feedback uses the existing
producer-frame identity and asynchronous readback, with no admission readback.
Priority-path diagnostics separately count unknown-size builds, post-build storage
budget rejection and physical gap allocation failure; the priority-off baseline
retains its aggregate allocation-failure counter without this breakdown. Waiting count/bytes, the
largest waiting request and the compaction target expose reserved capacity that
has not yet become a live allocation. Saturated near-plane error bounds are
reported separately from finite pixel estimates in the comparison tool.
Projected retry deferrals and submitted dirty-size estimates are reported separately
from actual storage claims and are instrumented only on the priority path.

## Residency and submitted work

Every selected cluster must have a drawable page and a published CLAS. Cache
keys include CLAS publication generations, so unchanged topology does not imply
that physical addresses are still valid. Missing dependencies invalidate the
representation before publication. Raster demand continues independently.

Merged and discrete cut masks also have a GPU-resident cache keyed by page-table
mutation serial and CLAS publication revision. Shared masks are compared with
the actual current raster frontier, including representative identity and group
count. This avoids using camera stillness as a proxy for unchanged geometry.
Recorded page-table mutations advance their serial even if the recording is
cancelled; only executed GPU commands publish new cache metadata.

Representation records and buffers are retained by command submissions. The
existing render-graph completion dependencies order writes to the shared BLAS
arena after prior consumers. CLAS chunk retirement also waits for recorded GPU
uses. New initialization/publication state is committed through submission
transactions; cancelling unsubmitted work preserves the last submitted cut.

## Validation

`stream_rt_representation_importance` dispatches production reset and aggregation
over 137 instances across three workgroups, with classified LOD inputs supplied
by the fixture. It checks shared-consumer maxima, permutation invariance,
offscreen/behind-camera demand, orthographic/affine/near-plane cases and 32 frames
of removal and owner migration. It also compares GPU owner sorting and inverse
ranks with a CPU total-order oracle, including partial tiles and merge parity.
`stream_rt_representation_parallel_admission` covers priority group capacity;
`stream_blas_gpu_allocator` covers dirty reference/build quotas, priority-only
storage replacement, cancelled submissions and 32 stationary pressure frames.
It also leaves a high-priority request waiting on fragmentation with relocation
disabled: smaller builds must still use available work quota without consuming
the waiting owner's storage claim, then settle to zero builds for 32 frames.
Prediction regressions cover shrinking/growing dirty cuts, build/reference denial,
cancelled submissions, and stationary recovery. Tile cases cover complete rejected
blocks, a fitting last-ranked request, exact clean-block capacity and overflow,
and inactive lanes.
The actual allocator header is also passed to next-frame representation planning:
projection-only delay must preserve a newly idle cache, while real or mixed
storage rejection must still allow eviction in both admission modes.

`stream_rt_representation_lifecycle` exercises stable frames, sharing, merging,
cache return after temporary disuse, tight budgets, cancellation and retirement
with two frames in flight. It also reads actual fallback/importance metadata and
checks stable values and cancelled-frame recovery. Direct and reflected secondary-ray probes compare
coverage, instance identity and distance with the per-instance oracle. These
probes establish geometric correctness for their fixture, not full-scene
reflection-image quality.

For owner-priority comparisons use `Tools/AnalyzeBLASPriority.py`, which requires
explicit off/on/on/off captures with the same GPU allocator and budgets. Its full
RT preparation timing includes representation classification/cut and BLAS sibling
stages, so sorting cost is included. Dedicated quality runs may select deterministic
`qualitySamples` as well as the final pose; their readback drains disqualify their
timings as performance evidence.

Use `Tools/RunZorahFullRoam.ps1` with identical executable, shader, scene, camera,
resolution and warmup settings for each side. The runner records the mode in
its manifest. `Tools/AnalyzeRTRepresentation.py` consumes A1/B1/B2/A2 captures
and reports both construction savings and added preparation costs. Separate
quality captures export shadow, Deferred and final images; attaching their
debug observer makes them diagnostic runs, unsuitable as performance samples.

The default Zorah sample may have no enabled shadow-casting light. The benchmark
accepts an explicit `shadowPointLight` object with `position` and `intensity`;
use the same light on both sides and verify the captured shadow mask is not a
constant image. This sample does not contain a secondary-reflection pass.
`shadowOverrides` can select deterministic hard shadows for the separate image
comparison. Keep those settings identical on both sides and record them alongside
the image evidence. Alpha shadow sampling binds the same prepared frame's
texture metadata and image generation, including across texture migration.

Quality-only `captureReflections` adds a fixed-sample OpenPBR path-tracing diagnostic
using the VBuffer's published TLAS and streaming session. It includes material
reflections but is not isolated specular radiance or a production temporal
reflection pass. Compare identical sample indices against a larger-capacity
reference, check its rejection counters and residency, and retain texture/CLAS
streaming differences as confounders. This diagnostic is excluded from performance
comparisons.
