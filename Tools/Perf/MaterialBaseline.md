# Material GPU baselines

`MaterialBaseline.py` drives the existing production render-graph RHI baseline test.
The default remains the three Phase 0 fixtures. An explicit `--cases Cases.json`
selects a generated catalog without editing scene assets or production shaders.

```powershell
python -B Tools/Perf/PrepareMaterialBaseline.py build/material-baseline-config-new
cmake --build build-scheduling-release --target MetallicRHITests
python -B Tools/Perf/MaterialBaseline.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --cases build/material-baseline-config-new/Cases.json --output build/material-baseline-new --timeout 3600
python -B Tools/Perf/MaterialBaseline.py verify build/material-baseline-new
python -B Tools/Perf/PlotMaterialCatalog.py build/material-baseline-new build/material-baseline-report-new
```

Use new directories. The local White Studio/Painter catalog and RTXCR assets must
already exist; the generator fails when a required file is missing. It generates
56 cases with the current catalog: 24 surface scenes in Deferred/PT, three Slab
variants in Deferred/PT, one Claire Fiber PT scene, and one native strand scene.
Camera and geometry come from the authored lookdev graph. All internal outputs
are 512 x 512, linear HDR. Graphs contain only the measured rendering branch;
display transforms, comparison sliders and auto exposure are excluded.

- Deferred: sparse program binning, FP16 weights, advanced IBL budget 64,
  deterministic texture filtering. The simple material path may use prefiltered
  IBL instead; budget 64 does not imply identical per-pixel work for every BSDF.
- Surface PT: 4 spp/frame, depth 12, accumulation; Fiber PT: 4 spp/depth 4.
- Native Strands: authored small groom, 576 segments, eight layers, independent
  geometry/lighting. Its graph cost includes visibility plus Fiber lighting.
- Each case starts a fresh renderer and history. Three independent processes,
  each with 32 warmup frames and 64 timed frames. Final HDR readback is frame 96,
  outside the timing window. Warmup timestamps are retained separately.
- Required HDR environment must be ready throughout. Explicit graph cameras or
  authored document cameras and manual exposure are required for scene cases.

`run --runs N` selects 1 through 10 independent processes per package; the default
remains 3. `Process.json.runCount` records the requested count, and verification
requires exactly that many reports and successful exit codes. Legacy packages
without `runCount` retain the three-process requirement. Use `--runs 1` when an
external experiment driver controls interleaved A/B ordering; the package alone
does not establish A/A stability. Its `imageAA` result is explicitly unqualified
with null difference/exact values. The external driver must retain the run order
and compare repeat images and process timings across packages. `compare` does
not infer interleaving or automatically accept a speedup.
`PlotMaterialCatalog.py` remains a three-process catalog report; use a separate
experiment report for packages captured with another run count.

The runner disables validation and sanitizes inherited `METALLIC_*` flags. Run
correctness/validation separately, e.g. set `METALLIC_MATERIAL_BASELINE_CASES` and
invoke `RHIRendering.material_phase0_baseline` without `--rhi-no-validation`.
Do not mix those timings into the baseline. Disk caches are retained, GPU clocks
are unchanged, all workloads run serially with a process timeout. Cooperative
locks do not establish GPU exclusivity. The runner records whole-device telemetry
and background process inventory; it does not kill other applications or attribute
their GPU utilization. Normal timing is not compatible with injected diagnostics.

Evidence contains source/binary/input hashes, actual graph JSON, raw HDR images,
per-frame graph/node/nested-section GPU milliseconds, process logs and telemetry.
Git status and the archived diff cover the superproject and do not inspect optional
submodule worktrees; `Identity.json.statusScope` records this boundary. Staged
external entries and the explicit source/binary/workload hashes remain recorded.
Compilation/loading/CPU waits/readback and editor presentation are outside the
reported GPU scopes. Assets are identified, not copied into a portable snapshot.
Keep the generated configuration directory because material URIs reference it.
`verify` rechecks exact artifact hashes, case identity, complete ordered frame
windows, GPU times and finite HDR values. Image A/A differences are reported, not
silently treated as an accepted quality tolerance. Version 2 catalog captures
retain nonfinite HDR outputs with explicit `validHDR=false` and component counts,
then continue other cases. Capture completion is not a quality pass. The verifier
recounts nonfinite values from raw files; a case invalid in any process is excluded
from all performance tables/charts. Version 1 still fails immediately, and
optimization comparison excludes the union of invalid cases from both packages.
Deferred's VisibilityBuffer uses two forked compute/graphics raster branches,
joined before shading. Graph timing is the enclosing graphics-queue timestamp
span including joins, not a sum of concurrent GPU intervals. The verifier requires
this backend-specific queue contract (two branches for Deferred, zero for PT and
native strands); an explicit `expectedAsyncComputeBranches` can register another
contract. All enclosing node timestamps in these fixtures are graphics intervals.
A version 2 case can declare `expectedNodeQueues`, an exact mapping from node name
to RHI queue ID (0 graphics, 1 compute, 2 copy), for production graphs containing
nodes on other queues. Verification requires that exact node set and queue values;
extra, missing or duplicated nodes are rejected. Without this explicit mapping,
the existing all-graphics-node requirement remains. These node intervals are not
summed across queues, and `expectedAsyncComputeBranches` remains independently
checked. `Process.json.cache` records the actual configured warmup frame window.

The plotter exports raw plotting data, per-process medians, charts, previews and
a standalone report outside the sealed evidence. A/A uses the relative range of
three process medians, with a 10% threshold applied to both graph and material-pass
scopes. An unstable result
is `inconclusive`, never an accepted optimization. No measured frames are removed.
Graph/pass timings are scene baselines, not isolated BSDF costs or editor FPS;
geometry, coverage, environment and quality must match in future comparisons.
Do not sum nested GPU sections or compare Fiber/native and surface scenes as if
they used the same geometry or estimator.
