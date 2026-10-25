# Three-arm CLAS template Shadow qualification

New collections use the independent `metallic-clas-tessellation-ab-v3` protocol.
It reuses the production MaterialBaseline collector and v2 log-ratio statistics;
it does not use WorkRaster source swapping. `MeasureCLASTessellation.py` retains
the original two-arm v1 behavior. `verify` delegates v1 bundles to that verifier
and retains the original v2 fixture/analysis policy for sealed C1/C2 experiments.
Neither old protocol requires the new bounded-path receipts, and their decisions
and evidence remain unchanged. New v2 or source-swap experiments cannot be started
through this runner.

The scene, camera, native resolution, light, material, displacement quality,
residency freeze, workload size and exact image checks remain fixed. The arms are:

| Arm | Backend | Template shape | Instantiation flags | Template bounds |
|---|---|---|---|---|
| A | triangles | right | fastTrace | none |
| B | original templates | right | fastTrace | none |
| C3 | candidate templates | right | fastTrace | geometry |

Only C3 is registered for new collections. Historical v2 C1 (`right/none`) and C2
(`equilateral/fastTrace`) remain verifiable but are not rerun. Arbitrary strategy
combinations and graph/property sweeps are rejected. Do not add candidates or
increase budgets after looking at timings. C3 gets a fresh fixture and fixed
discovery/confirmation budget; observations from C1/C2 are never pooled into it.

```powershell
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py prepare --output build/clas-shadow-c3-fixture-new --template-bounds geometry
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-shadow-c3-fixture-new --output build/clas-shadow-c3-new --blocks 3 --confirmation-blocks 5
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py verify --input build/clas-shadow-c3-new
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py report --input build/clas-shadow-c3-new --output build/clas-shadow-c3-report-new
```

The run includes its three diagnostic processes before discovery; a separate
qualification run is unnecessary. Defaults remain 1444 x 694, 32 warmup and 64 timed frames. Build the
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

In v3, native size and actual-bound diagnostics are mandatory. Only DiagnosticGraph.json sets `benchmarkCLASBuildSizes`
true and only DiagnosticCases.json sets `captureTessellationNativeSizes` true.
The actual GPU u32 size array must cover exactly the live patch count; the
verifier recomputes its byte count, positive sizes, total, minimum and maximum.
Ordinary graphs explicitly disable the native-size operation, must not contain
its receipt, and must not export pipeline recipes. C3's diagnostic also copies
the actual 24-byte template bounding-limit buffer. Its six finite f32 values must
match the production identity byte-for-byte and have max >= min on every axis.
A/B must have no bounded-template receipt. C3 must report 1024 native templates
and ready bounded templates; B has 512 and A has none. The two template sets share
one pool: the first 512 entries are unbounded fallback templates, the second 512
are bounded templates. Total retained storage and auxiliary bytes include both
sets; the report does not hide them as excluded initialization cost.

No per-frame GPU readback is added to ordinary timing. The final untimed receipt
reads the last-frame fallback count and a GPU sticky bit which records any fallback
since allocation. C3 requires zero and false respectively. Pool generation must
remain unchanged throughout every warmup and measured frame and match the final
receipt, preventing a replacement allocation from silently clearing the proof.
Generation numbers are process-local provenance, not cross-process identities.
Between diagnostic and ordinary backend identities, only `diagnosticBuildSizes`
and that separately verified generation are exempted; all other strategy, bound,
allocation and shader identity fields must remain equal. Native size and bound
diagnostics are not timing samples. Template, auxiliary, CLAS and scratch
allocations remain reported even though initialization time is excluded.

The diagnostic explicitly classifies C3's actual live CLAS bytes versus B as
reduced, unchanged or larger, and reports the exact total, range and reduction
fraction. This size observation does not add a sixth performance gate, imply a
causal traversal explanation, or impose a bounds-volume threshold. Reports also
show retained pool capacity separately from those live native output bytes.

Use new report directories outside the sealed evidence bundle. The Python charts
record their source data and dependency versions. Inspect their axes, stage
boundaries and legends before making claims; this result describes the fixed
native production branch, not editor FPS or every scene.
