# VSM Page HMip / Receiver Mask candidate

Persistent receiver masks default on for directional SMRT with layered caching.
The Sponza VSM sample enables layered caching. Set `benchmarkReceiverMask` to
`false` on the lights pass for a full-page baseline. The dirty-page hierarchy
remains opt-in through `benchmarkDirtyPageHierarchy`.

## Scope and contracts

- After Assign, `vsmBuildPageHierarchyMain` constructs tight dirty-page XY bounds
  and an OR hierarchy per projection level. Mip zero remains the existing exact
  dirty-page bitset; upper levels use a padded power-of-two axis. Cluster queries
  clip against the tight bounds, choose a mip from rectangle extent, and read at
  most four nodes. Conservative false positives are allowed. Triangle page
  enumeration remains exact and retains the existing bitset.
- Receiver masks store 8×8 subtiles in two uints per virtual page. The existing
  directional SMRT footprint marks the mask before page-request deduplication.
  Mark publishes mask writes before Build reads them. Build intersects triangle
  XY bounds with each page mask before appending work.
- Receiver masks currently require directional SMRT, caching and cache layering.
  Other modes use complete pages. Coarse pages and static-dirty fine pages keep
  complete geometry. Static and dynamic raster passes share the Build list, so
  static-dirty work cannot be filtered by current receivers.
- Partial dynamic depth has a persistent 64-bit valid-coverage mask. Reset
  carries it through the same `oldIndex` remap as the resident page, only when
  the old depth remains valid and its casters have not invalidated it. After
  Mark, Admit keeps clean pages when `requested & ~valid` is zero; new coverage
  makes a resident admitted page dirty through the existing allocation path.
- Successful Finalize replaces valid coverage with the current mask. It records
  full coverage for static-dirty pages because their shared work list draws the
  complete page. It never unions stale coverage after a dynamic redraw. Overflow
  cannot publish valid coverage, and eviction clears it. The page table and mask
  share the existing submission/completion and cancelled-recording lifecycle.
- Valid masks occupy an 8-byte-per-page tail after the residency table and
  optional propagated mappings, with total allocation rounded to 16 bytes.
  Byte-addressed two-word stores avoid writes overlapping adjacent pages. No
  new dispatch or push-constant field is required. Changing the effective mask
  mode, or the propagated-table layout while masks are active, invalidates the
  cache before the old tail could be interpreted using a different offset.
- Build mask bounds include one texel plus a coordinate-magnitude allowance for
  the differing clipmap and canonical page translations. It follows Mark's
  conservative numerical policy; focused rendering equivalence is not a proof
  for every floating-point world-coordinate configuration.

Scratch layout is allocation metadata, dirty bitset, optional receiver masks,
optional page hierarchy, then optional compact-build candidates. CPU sizes,
Reset coverage and shader offsets use the same effective permutations.

This first candidate implements a **PageFlags hierarchy and exact per-page
receiver-mask tests**. It does not yet implement a separate ReceiverMask HMip
pyramid, tile-reduced mask writes, or separate static/dynamic Build lists. Unreal
uses distinct page-flag and optional receiver-mask hierarchy queries; these are
not one combined gather operation. Further hierarchy work should be driven by
the measured mask test/reject counts and total Mark/Build/Raster cost.

## Diagnostics and validation

The optional build diagnostics ABI is version 4, 50 uint64 counters. Previous
indices remain unchanged. New counters are `clusterHierarchyLevelTests`,
`clusterHierarchyNodeTests`, `clusterHierarchyBoundsRejects`,
`receiverMaskPageTests` and `receiverMaskPageRejects`. Hierarchy queries do not
contribute to the old linear-scan histograms. `triangleBitsetPageHits` counts
enumerated bits before mask rejection and work-capacity overflow.

`VSM build page hierarchy` is an independent GPU child scope included in the
inclusive `VSM build` time. Do not add it to Build again. The profile parser
retains versions 1–3 without inventing missing counters.

Focused RHI tests:

- `virtual_shadow_map_page_hierarchy`: production reset/build/query versus an
  independent CPU oracle across 72 axis/layout/pattern combinations, bounds and
  OR-tree readback, at most four node reads, zero false negatives, allocation
  guards, and receiver-mask word/texel boundaries.
- `virtual_shadow_map_receiver_mask_cache`: complete-redraw versus candidate,
  mixed static/dynamic casters, receiver expansion, moving casters, coarse budget
  fallback, disabling/re-enabling the mask and switching to hard shadows.

Bounded scene experiment and raw evidence: `build/vsm-page-hierarchy-20261102/`.
Its A/H/M arms all explicitly enable cache layering; A is a layered baseline,
not the unchanged shipped sample configuration. Ordinary timing is separate
from diagnostics. One symmetric timing block does not establish a confidence
interval or independent confirmation.

## Initial result before persistent valid coverage

The Release build, module audit and 102 Python tests passed. The GPU hierarchy
probe passed 67,212 queries; the cache fixture passed 16 transitions. Sponza's
96 raw image comparisons were byte-exact, with zero validation messages.
Twelve actual bound SPIR-V binaries passed `spirv-val`.

On RTX 5070 Ti (621.01), the stationary warm-cache baseline had 2,047 clean fine
pages. The combined candidate made all 2,047 dirty every frame. Its ordinary
Shadows time rose from approximately 1.78 to 6.12 ms. Mark rose from 0.223 to
approximately 0.514 ms; Build/Raster also increased substantially. The combination
is therefore retained disabled, not accepted as a performance improvement.

HMip alone added approximately 0.9 microseconds to warm Build. During diagnostic
motion frames 40–47 it reduced cluster word/node reads from 1,336,932 to 117,806,
but conservative hits increased decoded triangles from 507,096 to 1,025,432.
Final raster work remained unchanged. This diagnostic work reduction is not a
dirty-scene ordinary-timing result. The receiver mask rejected only 420,826 of
8,053,106 tested triangle/page pairs (5.23%) over the same motion interval.

Before expanding the hierarchy, prioritize persistent valid-coverage masks to
reuse complete-enough dynamic depth, tile/wave aggregation of mask writes, and
separating cached static work from dynamic Build work. HMip needs either exact
confirmation of positive results or bounded descent to control false positives.
See the full raw evidence, timing chart and reproduction script in
`build/vsm-page-hierarchy-20261102/Report.md`.

## Persistent-coverage validation

The subsequent persistent-coverage implementation passed a 30-state GPU cache
test, including actual subpage rejection, coverage growth/shrink/revisit,
replacement after dynamic redraw, virtual-page remapping, physical-slot reuse,
and mapping-tail layout changes. All shadow/depth comparisons matched complete
redraw, with zero validation messages. Sponza supplied another 48 byte-exact raw
image comparisons and unchanged canonical inputs/page demand. Nine actual bound
lifecycle SPIR-V binaries passed `spirv-val`.

Sponza's warm persistent-mask path now retains 2,047 clean fine pages, with zero
dirty fine pages, zero Build triangle decodes and zero fine raster groups.
In one same-runtime ABBA block on RTX 5070 Ti (621.01), Build was approximately
0.017 ms. Shadows was approximately 2.09 ms with persistent masks versus 1.82 ms
with masks disabled (both arms use HMip and layered caching). Mark remains the
main extra cost: approximately 0.482 versus 0.223 ms before tile aggregation.
The earlier every-frame-redraw
results above are historical context, not this ABBA's baseline.

Evidence and reproducible charts: `build/vsm-receiver-validmask-20261102/Report.md`.

## Directional Mark tile aggregation

Directional Mark uses tile aggregation by default when persistent receiver masks
are active. Set `benchmarkAggregateReceiverMask` to `false` to benchmark direct
writes. Each 8×8 group accumulates the two mask words per page in
a 64-slot shared table. A claimed key stays fixed until all lanes finish.
Colliding pages write their original words directly. After a group barrier,
each occupied slot publishes its nonzero words once. Both paths take the OR
of the original footprint contributions. Page requests use their existing
deduplication and cache admission consumes the same mask layout.

On VSMSample-Sponza at 1080p, the median static-frame global receiver-mask OR
count fell from 30.96 million to 3.98 million. The counter measures issued
32-bit mask updates and excludes group-shared operations and page requests.
Six checkpoints spanning camera motion and a 127×73 resize matched the direct
path byte for byte across all 98,305 requested-mask entries and four raw images
per checkpoint. Vulkan validation reported zero messages in both arms.

In one ordinary ABBA block, Mark process medians were 0.494 / 0.476 ms for
direct writes and 0.462 / 0.463 ms for aggregation (4.60% lower). Shadows
changed by only 0.16% and graph time increased by 0.78% in this block.
Measurements, actual Mark SPIR-V and charts are in
`build/vsm-mask-wave-20261103/Report.md`.

## Static depth HZB per physical page

Layered directional shadows use physical-page HZB occlusion queries by default.
Set `benchmarkStaticPageHZB` to false to disable them. Page HMip tracks virtual-page dirtiness. The
shadow HZB holds static occluder depths within each resident physical page.

- Finalize appends each admitted, valid static-dirty page to an indirect job
  list. After static raster, one workgroup reduces that page of the static atlas to
  its root. With reversed Z each node stores the minimum of its four children.
  An uncovered texel contributes clear depth. A clean frame has zero jobs.
- The HZB follows the static atlas through cache publication and reuse. Enabling
  it on an existing cache without a pyramid requires rebuilding static depth.
  A 128×128 page stores 5,461 floats (64×64 through 1×1). A 2,048-slot face
  uses 42.66 MiB for the pyramid plus 8,204 bytes of indirect arguments and jobs.
- Reset can reuse a composite page when every overlapping old and new caster
  bound is strictly behind the retained static depth. The test pads the texel
  rectangle, expands the nearest depth for rounding and reads up to four cells
  of a single mip. Fine pages use the rasterizer's canonical page transform.
  The coarse page checks up to 32 changed bounds. A static-layer change or
  forced redraw bypasses reuse.
- Build and Compact apply the same query after a dirty-page hit, before
  triangle decode, for projected rectangles spanning up to 4×4 pages including
  the existing guard band. A cluster level can be skipped only when every dirty
  admitted page it could touch has valid, unchanged static depth and occludes
  the bounds. Larger rectangles and newly assigned pages follow normal Build.

The initial retained-depth results are in `build/vsm-static-hzb-20261103/Report.md`.

### Dynamic cluster culling after static raster

With `benchmarkStaticPageHZB` enabled, `benchmarkClusterHZBCulling` defaults
to true. The `VSM dynamic cluster HZB cull` dispatch runs between the static
HZB update and dynamic raster. Each group checks a 64-triangle-slot cluster
against the current static depth and writes one 32-bit visibility mask. Bits
cover the fine clipmaps and coarse page. If every relevant dirty page occludes
the bounds, dynamic raster can skip the mesh decode for that level, including
when the static depth was rebuilt in the same frame. Missing bounds, incomplete
pages and large rectangles retain their original work. Dynamic depth does not
contribute occluders.

The pass requires both static and dynamic casters. Finalize dispatches zero
groups if there is no fine or coarse draw work. Mesh group submission and the
shared Build work list are unchanged. Workload schema v5 records cluster levels
tested, levels rejected and dynamic triangle raster work items skipped before
decode. The last count includes repeated submissions across pages and levels.

The `virtual_shadow_map_physical_hzb` Vulkan fixture forces a complete static
redraw on promotion, moves the roof away and then moves the previously hidden
cluster. It compares the receiver depth and shadow outputs to full redraw on
every frame, including after physical slot reuse. See
`build/vsm-current-cluster-hzb-20261103/Report.md` for the diagnostic results.

## Physical page recency and static caster transitions

Each cached light face appends a GPU recency list to its published page table.
Each entry stores a physical slot and its last requested cache frame. Slot zero
remains reserved for coarse coverage. The existing Physical pass stably
partitions the prior order into free, unrequested and pinned slots. Assign
uses free slots first and then evicts the oldest unrequested owners. Slots
used by this frame move to the newest end. A clipmap scroll preserves recency
for retained physical slots. Eviction order does not depend on comparing
32-bit timestamps, so wrapping the frame counter cannot reverse it.
The list costs 16 KiB per 2,048-slot snapshot and adds no dispatch.

For layered caches, `framesStaticThreshold` defaults to 100 submitted,
unchanged cache frames. Set it to zero to keep promoted casters dynamic.
Content changes and dynamic LOD changes restart the interval. Cancelled
recordings do not advance it. On demotion, a valid static page that only
needs the new caster loads its existing depth and draws the transitioned
casters into it and refreshes the static HZB. The composite can keep its
previous depth because it already includes those casters at the same pose.
This requires a valid composite with complete receiver coverage and no
independent composite invalidation. A page with partial receiver coverage
still needs composition before it can publish full coverage. Independently
invalidated static pages get a full static redraw.
The transition list uses the current shadow LOD's original 64-triangle
clusters and a mask of the slots that changed layer. With only fine-page
merges pending, Build decodes those slots. If another fine page needs
composition, Build traverses the usual geometry and filters its output per
page. Coarse static raster independently selects the transition clusters for
a merge or the full static geometry for a redraw.
An unadmitted or incomplete fine page must rebuild before reuse. A subsequent
move promotes the caster again and invalidates its old static footprint.

Workload schema v6 adds counts of incremental static pages and triangle/page
decodes, physical evictions and the sum of evicted page ages. With a nonzero
eviction count, dividing the age sum by that count gives the mean victim age
in submitted cache frames. The allocator GPU fixture checks timestamps and
ownership over eight frames. The physical HZB fixture checks pure demotion,
concurrent motion, subsequent reveals and re-promotion against full redraw.
The receiver mask fixture also checks demotion after a narrow dynamic redraw
and the next receiver coverage expansion.
