# MiniZorah spatial demand A/B

**Retired: verification-only.** Spatial candidates were withdrawn after the bounded experiments. `run` now fails immediately, before inspecting inputs or creating files, locks or processes. Historical `verify`, decoding and statistical policies remain unchanged. Collection commands below document the historical protocol and cannot launch a new experiment.


`MeasureSpatialDemand.py` historically ran the editor `streamingRegression` harness with
`METALLIC_SPATIAL_DEMAND=0/1` in separate processes. It does not compile, modify source,
change the production default, or clear persistent shader/driver caches.

The final historical collections used `metallic-spatial-demand-ab-v4`: A must report `ordered-tiles`,
64 groups per leaf, 8 nodes per task and zero spatial metadata; B must report
`ordered-task-spatial-v4`, 8 groups per leaf, spatial task-node budget 128 and positive spatial
metadata no larger than **64 MiB**. The candidate preserves original ordered root tasks (budget 8), with its
spatial node/group-reference data in a separate buffer. This cap applies to
that buffer's actual `spatialDemandBytes` and is an acceptance gate only; it does
not change runtime allocation or residency budgets.

The original topology size (`state.immutableMetadata.topologyBytes`) must be equal
across both arms and all captures. Top-level `originalTopologyHash` and
`primitiveMetadataHash` must also match: both are canonical decimal uint64 strings,
preserving every bit across JSON consumers. The independent spatial allocation may
exceed the original topology allocation; only the declared 64 MiB cap limits it.
The plan records both topology contracts and the cap before any process starts.
Existing v1 and v2 evidence remains verifiable and plottable under its original
four-plus-four timing budget and demand/graph acceptance gates, without requiring
v3 fields or rewriting any evidence. In v2, candidate `group-refs-v2` uses leaf8/task32
and its spatial bytes remain part of the total topology. Never combine protocol
versions or reuse another candidate's performance data in a new experiment.

The predeclared schedule is quality **ABBA**, discovery **four ABBA blocks**, and
independent confirmation **six ABBA blocks** (44 bounded processes total). Quality
failure or a completed stage whose CI demonstrates regression stops the schedule;
otherwise confirmation runs even when discovery is inconclusive. No sample-size
extension or outlier rejection occurs. Each process uses 16 settle and 64 timing
frames after the harness converges and freezes residency and resets owner history.
`benchmarkDemandTraversal` forces real demand evaluation instead of cached history.

```powershell
python -B Tools/Perf/MeasureSpatialDemand.py run `
  --exe build-release/Source/MetallicGPUDrivenSample.exe `
  --config <deterministic-MiniZorah-config.json> --assets <verified-mini-assets.json> `
  --output build/spatial-demand-new
python -B Tools/Perf/MeasureSpatialDemand.py verify build/spatial-demand-new
python -B Tools/Perf/PlotSpatialDemand.py build/spatial-demand-new --output build/spatial-demand-report-new
```

Requires Python/numpy (plotting also matplotlib), PowerShell 7 (`pwsh`), and the
existing renderer dependencies. The config must select `gpu-driven-sample`, disable
DLSS, temporal jitter, stochastic texture filtering and shadow sigma denoising.
Use a build configured with `METALLIC_ENABLE_STREAMLINE=TRUE`: this graph retains
`StreamlineDLSSSRPass` and requires its SDK even when `dlssMode` is `Off`. An RHI
test build with Streamline disabled does not qualify this editor graph executable.
Supply explicit resident/CLAS budgets, resolution and scene lighting as appropriate.
The asset input is the existing `metallic-declared-assets-v1` MiniZorah manifest;
size/mtime is rechecked around every process, not a new hash of the large asset closure.
The selected config, asset manifest, source/shader/executable/DLL hashes, commands,
process intervals, GPU/process telemetry, raw snapshots and logs are retained.

Primary metric: VBuffer `Stream traversal` GPU scope. Acceptance requires its
confirmation reduction CI lower bound to be greater than zero. The lower bounds
for whole `RenderGraph GPU envelope`, `LOD clear / demand seed`, and `LOD frontier`
reductions must each be greater than -2%. `Detail demand` remains a reported
secondary metric; speeding it up alone cannot accept the candidate. At a completed
stage, a primary CI upper bound below zero or any guard upper bound below -2%
declares regression and stops further stages. Otherwise the fixed confirmation
budget runs, even if discovery is inconclusive. Discovery cannot accept the
candidate, and the runner never applies a candidate automatically.

The CI uses independent ABBA block log time ratios: four blocks in discovery and
six in v3 confirmation, not 64 frames as independent samples. Plot labels report
the actual block and process counts. No adaptive extension, pooling of discovery
and confirmation, or selection of favorable frames/blocks is allowed.

Quality is collected in separate processes with seeds 0 and 1, reflections enabled,
and RGBA32F Deferred output required. All six captured image resources must be finite
and bitwise equal; max absolute error and RMSE are still reported for failures. The
14-buffer decoder checks actual TLAS addresses against captured owner allocations
or fallback addresses, then compares semantic page/CLAS states, instance data, cut
records, effective TLAS owners, requests and texture publication. Physical addresses
are normalized; both ordered and order-independent cut hashes are compared, and
authored cut fields are preserved.
Traversal counters may differ and are reported separately. Diagnostic source strings
are not assumed to be portable physical-allocation identities.

Ordinary timing captures diagnostic snapshots only after measured frames. Raw GPU
competition files are retained and their presence checked; this is not a proof of
exclusive GPU use. Review them before accepting a performance claim. PDH retry
messages are retained as diagnostics; failed samples are never replaced with zero.
This tool adds no background telemetry mechanism. Shader identity
checks cover reported frontier/mask identities and the bound demand shader identity
in demandStats metadata, cross-checked against the actual scheduler word.
The fixed view is a controlled comparison, not a roaming-performance claim. Use
`Tools/RunMetallicCfgReplay.ps1` / `RHIRendering.minizorah_fixed_baseline` separately
for the same camera replay under live residency, where asynchronous arrival may
produce different intermediate cuts and cannot satisfy the frozen-source oracle.


## v4 ordered-task spatial protocol

New runs use `metallic-spatial-demand-ab-v4`, layout `ordered-task-spatial-v4`, leaf groups 8, spatial task nodes 128. Seed and dispatch tasks retain original ordered roots (budget 8). Candidate and baseline must have identical `demandTaskCount`, actual `seededTasks` and `processedTasks` across snapshots. Original topology bytes and both decimal-u64 identity hashes remain equal; separate spatial storage is capped at 64 MiB.

V3 and v4 both retain fixed quality ABBA, discovery 4 ABBA blocks and confirmation 6 ABBA blocks. Primary is total stream traversal; graph, Seed and Frontier retain -2% guards and completed-stage regression stopping. V1/v2 retain their original 4+4 policy and Demand primary. Historical v3 layout remains `separate-group-refs-v3`.
