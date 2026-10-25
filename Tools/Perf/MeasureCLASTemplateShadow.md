# Three-arm CLAS template Shadow qualification

This runner uses the independent `metallic-clas-tessellation-ab-v2` protocol.
It reuses the production MaterialBaseline collector and v2 log-ratio statistics;
it does not use WorkRaster source swapping. `MeasureCLASTessellation.py` retains
the original two-arm v1 behavior. The new `verify` also delegates v1 bundles to
that existing verifier without changing their decisions or evidence.

The scene, camera, native resolution, light, material, displacement quality,
residency freeze, workload size and exact image checks remain fixed. The arms are:

| Arm | Backend | Template shape | Instantiation flags |
|---|---|---|---|
| A | triangles | right | fastTrace |
| B | original templates | right | fastTrace |
| C1 | candidate templates | right | none |
| C2 | candidate templates | equilateral | fastTrace |

Each experiment selects exactly one of C1 or C2 as arm C. The combined
`equilateral/none` strategy is not registered, and arbitrary graph/property sweeps
are rejected. Predeclare whether to run C1, C2 or both; do not add candidates or
increase budgets after looking at timings. Each selected candidate gets its own
fresh fixture and fixed discovery/confirmation budget. Discovery observations
are never pooled into confirmation or between candidates.

```powershell
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py prepare --output build/clas-shadow-c1-fixture-new --template-shape right --instantiation-flags none --native-sizes
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-shadow-c1-fixture-new --output build/clas-shadow-c1-diagnostic-new --diagnostics-only
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-shadow-c1-fixture-new --output build/clas-shadow-c1-new --blocks 3 --confirmation-blocks 5
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py verify --input build/clas-shadow-c1-new
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py report --input build/clas-shadow-c1-new --output build/clas-shadow-c1-report-new
```

C2 preparation substitutes `--template-shape equilateral --instantiation-flags
fastTrace`. Defaults remain 1444 x 694, 32 warmup and 64 timed frames. Build the
same Release executable separately; the runner never changes SDK options, source
files, clocks or user processes. Cooperative GPU locks and whole-device telemetry
are retained, but do not prove exclusive use of the GPU. Ordinary processes have
validation and pipeline recipe export disabled.

Each stage uses symmetric **ABCCBA** six-process blocks. The A/B, A/C and B/C
contrasts each extract their chronological ABBA subsequence from the same block.
The statistical unit is the complete six-process block; its three contrasts
share samples and are not independent replications. The helper computes one mean
log-ratio per contrast per block and its Student-t 95% interval. Fixed defaults are
three discovery blocks and, if all point estimates qualify, five new confirmation
blocks: 48 ordinary processes plus three diagnostic processes when confirmation
runs. No outlier removal, adaptive stopping, extra sampling or pooled discovery
is permitted. Per-arm range above 10% remains a warning rather than a reason to
discard a paired result.

All of these confirmation bounds are mandatory. Gain is `1 - candidate/reference`:

| Contrast and GPU scope | Required lower 95% gain bound |
|---|---:|
| C/A complete Shadows pass | at least -2% |
| C/A whole measured graph | at least -2% |
| C/A CLAS build/instantiate | at least +50% |
| C/B complete Shadows pass | strictly above 0% |
| C/B CLAS build/instantiate | at least -5% |

Discovery never accepts. It advances only if every point estimate meets its
registered gate; a confidence interval wholly failing any gate can reject. At
confirmation, a crossing interval is inconclusive, not success or proof of no
effect. The report shows all six scopes and all three contrasts, including any
regression in the trace, copy, complete shadow, tessellation or graph intervals.

`Shadow trace` and existing `Publish shadow` GPU sections separate trace recording
from the graph copy. They are supporting measurements: complete Shadows pass
time owns the consumer gate and includes its other work and synchronization.
Tessellation is record plus publication; its nested CLAS section is not added a
second time, and final TLAS construction remains in whole-graph time.

Each arm first exports actual bound tessellation and shadow pipeline recipes and
SPIR-V in a separate diagnostic process. Tessellation's hash must match the
production runtime receipt. All arms must bind the same shadow kernel bytes.
Geometry/material/count identities and exact HDR/depth/shadow bytes must agree
before ordinary timing starts and throughout both stages. Actual generated
stream files are archived and normalized only under the existing narrowly
validated v9 runtime marker schema. Source, assets, fixtures, executable, DLLs,
CMake configuration and scripts are hashed before collection and checked again
between processes.

With `--native-sizes`, only DiagnosticGraph.json sets `benchmarkCLASBuildSizes`
true and only DiagnosticCases.json sets `captureTessellationNativeSizes` true.
The actual GPU u32 size array must cover exactly the live patch count; the
verifier recomputes its byte count, positive sizes, total, minimum and maximum.
Ordinary graphs explicitly disable the native-size operation, must not contain
its receipt, and must not export pipeline recipes. Between diagnostic and ordinary
runtime identities, only the `diagnosticBuildSizes` flag is exempted; all other
strategy and shader identity fields must remain equal. Native size diagnostics
are not timing samples. Persistent template, auxiliary, CLAS and scratch
allocations remain reported even though initialization time is excluded.

Use new report directories outside the sealed evidence bundle. The Python charts
record their source data and dependency versions. Inspect their axes, stage
boundaries and legends before making claims; this result describes the fixed
native production branch, not editor FPS or every scene.
