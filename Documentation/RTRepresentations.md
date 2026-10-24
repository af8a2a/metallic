# Streamed RT representations

The streamed ray-tracing path separates a BLAS owner from a rendered instance.
Raster LOD traversal still produces per-instance cuts. Each primitive has three
RT representation slots, and each instance selects a slot or its terminal BLAS.
Representation ownership is the sole streamed RT path; per-instance A/B
selection has been removed.

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

The production merged-sharing policy makes a
primitive's demanded merged cut the single owner of all its nonterminal
consumers. The highest drawable cut can satisfy those consumers without also
charging shared/discrete BLAS storage, references and builds. Its importance
aggregates all those consumers, including offscreen ones. A representative's
own broad LOD interval alone does not create merged demand. Primitives with no
real merged consumer retain their shared/discrete choices; terminal-only users
retain the terminal BLAS.

This choice is computed from the completed minimum-LOD histogram after shared
and discrete selection, before per-instance demand aggregation. The first LOD's
fourth histogram word temporarily holds the choice; phase 25 subsequently reuses
it for cached-storage accounting. No extra sort, scan pass, CPU readback or
persistent publication state is added. Existing idle discrete-cache retention and
pressure eviction still apply. A merged cut can cost more traversal work than a
coarser shared cut, so image quality and ray timings remain scene-level checks.
If that merged cut cannot fit, this policy also forgoes smaller independently
requested shared/discrete alternatives; it is not a universal quality guarantee
under every capacity limit. Terminal fallback remains possible for rejected owners.

The compact RT group buffer has its own capacity. Admission publishes complete
primitive ranges, never a truncated group list. Under pressure, idle cache
storage gives way to active demand. A rejected shared/cache selection may use
an already-ready merged representation of the same primitive; otherwise it
uses the existing terminal fallback.

The experimental budgeted intermediate fallback was not accepted: it improved
some hotspots while increasing overall HDR error. Its production opt-in and
diagnostic policy switches have been removed; terminal fallback remains the
fallback policy for rejected owners.

The allocator always uses the accepted mixed resident/rejected tile
preparation shortcut when its exact fit proof holds. It only applies
when every resident claim fits and even the smallest rejected claim cannot fit
before any resident charge; equality retains the serial path. The allocator
regression checks the production result against a CPU reference, including dirty,
unknown-size, eviction-limit and cached stationary frames.

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

Owner-priority admission is the production policy for the GPU BLAS allocator.
The old owner-ID admission and per-instance allocator A/B paths are removed.

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

Exact GPU sort/admission caching is always active. After recomputing owner importance,
the GPU compares all three ordering keys with the last executed sort's snapshot.
Unchanged keys reuse the sorted IDs and inverse ranks; consumer counts still
update each frame. Recorded but cancelled commands do not publish the snapshot.

The allocator caches a completed admission decision only when it submitted no
builds and evicted no owners. Reuse requires unchanged owner demand, no dirty or
unknown-size cut, unchanged ordering, budgets, total free bytes and largest gap.
The existing parallel owner pass restores measured rejections and waiting claims
without repeating the serial ranked scan. Clean relocation remains bounded and
can continue; changed free-space conditions invalidate admission reuse. With no
new native build sizes, exact storage selection is skipped while the common
compaction planner still runs. This preserves waiting-owner recovery instead of
treating camera stillness as permission to stop allocation work.

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
budget rejection and physical gap allocation failure. Waiting count/bytes, the
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

The RHI owner-importance and admission fixtures compare production owner keys,
complete admitted ranges and cache invalidation against CPU reference decisions.
Allocator tests cover exact-fit and fragmented capacity, rejected-owner retry,
clean relocation, idle reuse, cancellation and mixed resident/rejected tiles.
The retired per-instance size-class/Repack fixture is removed with its backend.

`stream_rt_representation_lifecycle` exercises sharing, merging, discrete-cache
return, budgets, cancellation and retirement with work in flight. A cold restart
of the production path provides the independent lifecycle reference; it does
not select the removed per-instance implementation. Direct and reflected probe
checks establish fixture geometry behavior, not full-scene image correctness.

Use the roam `streamingRegression` configuration for bounded production-path
validation with converged frozen geometry/CLAS/texture residency and one
canonical owner/history reset. It exports owner selection, actual TLAS pointers,
cache reuse, builds and images. It does not switch admission policies.
`detailDemandComparison` and `blasQualityComparison` are rejected; offline
analysis tools remain able to inspect historical captures.

Quality capture requires DLSS Off, no temporal jitter, deterministic textures,
`shadowOverrides.sigmaDenoise=false` and fixed sample seeds. Readback/drain
captures are separate from ordinary timing. The reflection diagnostic includes
OpenPBR material transport; it is not isolated specular radiance or a temporal
production reflection pass. See [production LOD regression](StreamLODFrontier.md).

An optional `diagnosticPixels` array records actual OpenPBR paths at up to eight
pixels through a diagnostic shader permutation. `Diagnostics.json` contains the
raw path texture and representation/instance-selection buffers. Integer fields
are bit representations, not radiance. A shared initial seed does not guarantee
identical later rays after geometry or material branches diverge. Use
`Tools/AnalyzeBLASPathDiagnostics.py` to inspect the first differing ray, hit or
lighting contribution. A changed pixel alone does not establish a material,
normal or PDF defect.
