# Retired VSM experiments

The established Mark deduplication, local-light Mark sample optimization,
directional Mark, clipmap bounds and page-size specialization, cluster culling,
cluster/triangle dirty-page bitsets and directional Build paths are unconditional.
Their runtime settings and graph switches no longer select an alternative path.
As of 2026-11-04, directional uniform Projection, multi-face one-pass Projection,
directional Mark ray-pages, propagated mappings, persistent/aggregated receiver
masks, static-page HZB and dynamic cluster HZB culling are also standard.
Optional shadow LOD uses hierarchical traversal and its depth-error guard;
optional smart bias uses resource-qualified refined receivers.

`VirtualShadowMapOptimization.py` retains historical four-arm evidence verification
and reporting. Its live `prepare_plan` and `run` entry points now reject all new
four-arm experiments before reading cases, creating output or launching a process.

`VirtualShadowMapBitset.py` rejects these live experiment names at the same boundary:

- `cluster-bitset`
- `directional-build`
- `directional-mark`
- `mark-clipmaps`
- `mark-page-size`
- `mark-ray-pages`
- `directional-projection`
- `one-pass-projection`

All existing protocol identifiers, sealed-package verification, original acceptance
policies and report generation remain available. Use the current tool's `report`
command on old packages; do not execute archived runner copies. Historical
comparison results do not imply that old switches still affect the current renderer.

Live `compact-build` experiments remain available. To prevent ignored settings
from producing a false A/B, their new capture graphs must omit these removed
properties, including properties previously set to the optimized/default value:

| Removed feature | Removed graph properties |
| --- | --- |
| Mark deduplication | `benchmarkLegacyMark` |
| Local Mark sample optimization | `benchmarkMarkSamples` |
| Directional Mark | `directionalMark`, `benchmarkDirectionalMark` |
| Mark clipmap specialization | `markClipmapSpecialization`, `benchmarkMarkClipmapSpecialization` |
| Mark page-size specialization | `markPageSizeSpecialization`, `benchmarkMarkPageSizeSpecialization` |
| Cluster culling | `benchmarkDisableClusterCull` |
| Cluster page bitset | `clusterPageBitset`, `benchmarkClusterPageBitset` |
| Triangle page bitset | `trianglePageBitset`, `benchmarkTrianglePageBitset` |
| Directional Build | `directionalBuild`, `benchmarkDirectionalBuild` |
| Multi-face Projection | `onePassProjection`, `benchmarkOnePassProjection`, `benchmarkLegacyProjection` |
| Directional Projection | `directionalProjection`, `benchmarkDirectionalProjection` |
| Propagated mappings | `propagatedMapping`, `benchmarkPropagatedMapping` |
| Mark ray-pages | `markRayPages`, `benchmarkMarkRayPages` |
| Persistent receiver masks | `receiverMask`, `benchmarkReceiverMask` |
| Coalesced mask writes | `aggregateReceiverMask`, `benchmarkAggregateReceiverMask` |
| Static-page HZB | `staticPageHZB`, `benchmarkStaticPageHZB` |
| Cluster HZB culling | `clusterHZBCulling`, `benchmarkClusterHZBCulling` |
| Shadow LOD hierarchy | `shadowLODHierarchy` |
| Shadow LOD depth agreement | `shadowLODDepthGuard` |
| Refined receiver bias | `shadowLODSmartBiasRefine` |

This rejection applies only to live planning; historical graph normalization is
unchanged so sealed evidence remains reproducible. The remaining candidate
controls, including compact Build and Projection statistics, are not retired by
this change. Quality, resource budgets, cache-layering and top-level optional
LOD/smart-bias functionality remain configurable.

`PrepareVirtualShadowMapLODImpact.py` and the `VirtualShadowMapLODImpact.py`,
`VirtualShadowMapLODSelfShadow.py`, `VirtualShadowMapLODSmartBias.py` and
`VirtualShadowMapLODSmartBiasRefine.py` runners reject live preparation/execution:
their comparisons require flat traversal or bypassing the standard depth guard
or refined receiver implementation. Historical normalize/verify/report and
offline quality/oracle helpers remain available. The demand-budget experiment
remains live, emits no retired switches and rejects them before process launch.

CPU regression tests exercise live rejection before launch, all remaining live
plans, stale graph properties in both boolean states, immutable historical
reports and shader/diagnostic identity verification:

```powershell
python -X utf8 -B tests/perf/TestVirtualShadowMapOptimization.py
python -X utf8 -B tests/perf/TestVirtualShadowMapBitset.py
```
