# VSM Page HMip / Receiver Mask candidate

2026-11-02: opt-in candidate. Both `VirtualShadowMapSettings::dirtyPageHierarchy`
and `receiverMask` default to false. `VirtualShadowMapLightsPass` exposes the
experimental properties `benchmarkDirtyPageHierarchy` and `benchmarkReceiverMask`.
This is not a production-default change.

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
- Partial dynamic depth is valid for the current receiver footprint only. Reset
  invalidates resident fine dynamic composition each frame while preserving the
  static depth layer. Changing the effective receiver-mask mode invalidates the
  cache, including transitions to hard shadows. This can cost more than it saves
  in a stationary warm-cache scene.
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

## Initial result

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
