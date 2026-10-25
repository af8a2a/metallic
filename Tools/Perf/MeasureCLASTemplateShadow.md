# Three-arm CLAS template Shadow qualification

New collections use the independent `metallic-clas-tessellation-ab-v5` protocol.
It reuses the production MaterialBaseline collector and v2 log-ratio statistics;
it does not use WorkRaster source swapping. `MeasureCLASTessellation.py` retains
the original two-arm v1 behavior. `verify` delegates v1 bundles to that verifier
and retains the original v2 fixture/analysis policy for sealed C1/C2 experiments
and v3 bounded-path proof for C3. V4 remains `DYADIC_PROTOCOL`, with its original
C4 guide, budgets and metadata checks. Their decisions and evidence remain unchanged;
older protocols do not acquire v5 variant-count requirements. Only v5 can start new collections.
The runner never swaps source files.

The scene, camera, native resolution, light, material, displacement quality,
residency freeze, workload size and exact image checks remain fixed. The arms are:

| Arm | Backend | Template shape | Instantiation flags | Template bounds |
|---|---|---|---|---|
| A | triangles | right | fastTrace | none |
| B | original templates | right | fastTrace | none |
| C5 | candidate templates | rightCorner | fastTrace | none |

Only C5 is registered for new collections. It keeps three cyclic right-triangle
guide banks, using the existing quantized barycentric coordinates `q / 65535`.
For selected corner `c`, the guide XY is `(q[(c+1)%3], q[(c+2)%3]) / 65535`.
The actual generated corner vertices are at indices `0`, `rx`, `rx + ry` for the
pattern's edge rates. The GPU selects the corner opposite the longest actual XYZ
edge with the production FP32 comparison and first-corner tie rule, then selects
that bank's template address. Actual instantiated vertex positions, indices,
displacement, sampler settings, truncation bits and material behavior remain
unchanged. GPU selection and synchronization remain inside production timing.
Native replay and production controls must use this same guide definition and
selection rule; their byte agreement must be established by focused GPU tests, not
inferred from native size or the runner's metadata checks.
Historical v2 C1 (`right/none`), C2 (`equilateral/fastTrace`) and v3 C3
(`right/fastTrace/geometry`) and v4 C4 (`rightDyadic/fastTrace/none`) remain verifiable
but are not rerun. C4's guide remains `round(q * 32768 / 65535) / 32768`; C5 does
not inherit that coordinate change. Arbitrary strategy
combinations and graph/property sweeps are rejected. Do not add candidates or
increase budgets after looking at timings. C5 gets a fresh fixture and a declared
budget of three discovery and five confirmation blocks, enforced by both fixture
and verifier. Observations from C1/C2/C3/C4 or native replay are never pooled into it.

```powershell
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py prepare --output build/clas-shadow-c5-fixture-new --template-shape rightCorner --instantiation-flags fastTrace --template-bounds none
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-shadow-c5-fixture-new --output build/clas-shadow-c5-new --blocks 3 --confirmation-blocks 5
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py verify --input build/clas-shadow-c5-new
python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py report --input build/clas-shadow-c5-new --output build/clas-shadow-c5-report-new
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
log-ratio per contrast per block and its Student-t 95% interval. The v5 fixed budget is
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

In v5, native size diagnostics are mandatory for all three arms on every experiment.
Only DiagnosticGraph.json sets `benchmarkCLASBuildSizes`
true and only DiagnosticCases.json sets `captureTessellationNativeSizes` true.
The actual GPU u32 size array must cover exactly the live patch count; the
verifier recomputes its byte count, positive sizes, total, minimum and maximum.
Ordinary graphs explicitly disable the native-size operation, must not contain
its receipt, and must not export pipeline recipes. A/B/C must report pattern counts
`0/512/512`, native template counts `0/512/1536` and `templateVariantCount` values
`0/1/3`. Each field must be a JSON integer, excluding booleans and integral floats.
All arms use bounds mode `none`, null bounds and no native bounds receipt; A has
no template pool. B's one bank and C5's three banks remain resident. Their full
storage and auxiliary capacities are reported alongside actual native live CLAS
bytes; the extra persistent bank memory is not hidden as excluded initialization.
Reports use `candidateTemplateBytes` and `NativeSizeDiagnostic.json` without claiming
a bounded-template effect. V4 keeps its original 512-template B/C requirement and
does not require a variant-count field.

Historical v3 retains its original stricter bounded-path contract. C3's diagnostic also copies
the actual 24-byte template bounding-limit buffer. Its six finite f32 values must
match the production identity byte-for-byte and have max >= min on every axis.
A/B must have no bounded-template receipt. C3 must report 1024 native templates
and ready bounded templates; B has 512 and A has none. The two template sets share
one pool: the first 512 entries are unbounded fallback templates, the second 512
are bounded templates. Total retained storage and auxiliary bytes include both
sets; the report does not hide them as excluded initialization cost.

No per-frame GPU readback is added to ordinary timing. The final untimed receipt
reads the last-frame fallback count and a GPU sticky bit which records any fallback
since allocation. C3, C4 and C5 require a zero count and a false sticky bit;
v4/v5 also check the unbounded A/B receipts are zero and false. Pool generation must
remain unchanged throughout every warmup and measured frame and match the final
receipt, preventing a replacement allocation from silently clearing the proof.
Generation numbers are process-local provenance, not cross-process identities.
Between diagnostic and ordinary backend identities, only `diagnosticBuildSizes`
and that separately verified generation are exempted; all other strategy, bound,
allocation and shader identity fields must remain equal. Native size and bound
diagnostics are not timing samples. Template, auxiliary, CLAS and scratch
allocations remain reported even though initialization time is excluded.

The diagnostic explicitly classifies the candidate's actual live CLAS bytes versus B as
reduced, unchanged or larger, and reports the exact total, range and reduction
fraction. This size observation does not add a sixth performance gate, imply a
causal traversal explanation, or impose a bounds-volume threshold. Reports also
show retained pool capacity separately from those live native output bytes.
C5's three-bank persistent memory is reported, without adding an acceptance gate.

Use new report directories outside the sealed evidence bundle. The Python charts
record their source data and dependency versions. Inspect their axes, stage
boundaries and legends before making claims; this result describes the fixed
native production branch, not editor FPS or every scene.
