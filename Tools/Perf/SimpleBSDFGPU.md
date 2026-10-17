# SimpleBSDF GPU comparison

`SimpleBSDFGPU.py` compares direct vkGltf calls and the SimpleBSDF generic wrapper
at both SER settings. It is independent of the earlier PSO/compiler and three-arm
OpenPBR tools. It never changes their evidence or thresholds.

| Arm | `bsdf` | `shaderExecutionReorder` |
| --- | --- | --- |
| D0 | `vkGltf` | false |
| S0 | `simple` | false |
| D1 | `vkGltf` | true |
| S1 | `simple` | true |

The four mandatory strict contrasts are `wrapper_ser_off` (D0/S0),
`wrapper_ser_on` (D1/S1), `ser_direct` (D0/D1), and `ser_simple` (S0/S1).
Every arm also repeats against its first process. Different `bsdf` strings do not
exempt the wrapper from numerical equivalence. HDR uses the existing componentwise
`abs(B-A) <= 2e-5 + 2e-5 * abs(A)` bound; all seven guides require raw byte equality.
Nonfinite readbacks, runtime/validation errors, missing samples, and identity drift
are invalid evidence, not numeric failures.

## Workloads and process order

Preparation reads the sealed three-arm fixtures. It retains the original graphs,
scene/asset list, material, environment, camera, and sample counts: Opaque1 is 1 spp,
Mixed4 is 4 spp, and ABeautifulGame is 2 spp. All arms use ray pipelines, cache mode
off, final output, guides enabled, and the same integrator. Preparation checks the
exported production factory requests and their exporter/source hashes. Only the
simple raygen adds `METALLIC_SIMPLE_BSDF=1`; SER changes its declared macro and EXT
capability. All hit-stage requests remain identical.

Both stages use 512 × 512 pixels. Correctness keeps the temporal captures, camera
change examples, and seven guide captures with core/synchronization validation.
Two reverse orders, D0/S0/D1/S1 then S1/D1/S0/D0, produce 24 independent processes
for three cases. These two blocks balance pair direction; they do not give full
four-position balance. `--case Opaque1` gives a bounded 8-process smoke.

Performance uses no core or synchronization validation. It removes temporal HDR
captures, camera changes, guide-camera steps, and RenderView overrides. It retains
the fixed graph camera and seven guide outputs. Frames 0–31 warm up; GPU timestamps
use frames 32–95; frame 96 supplies final HDR. Guides are separately rerendered and
read back at frames 97–103, after the timing window, rather than copied from frame
96. Both windows' execution IDs and warmup flags are checked. Final HDR and all
guide readbacks are compared across arms and repeated processes.

Four Williams orders produce 48 independent processes:

1. D0 / S0 / S1 / D1
2. S0 / D1 / D0 / S1
3. D1 / S1 / S0 / D0
4. S1 / D0 / D1 / S0

Every arm occupies every position once; all twelve directed adjacent arm pairs
occur once. Each independent process supplies one median of 64 GPU timestamps.
Report pass time (`pass_ms`) as the primary measurement, with shading scope
(`dispatch_scope_ms`) and graph (`graph_ms`) separately. Statistics retain all
four process values and four block-paired percentage changes, median and range.
Positive percentages mean slower. Frames are not 64 independent experiments;
these timings are neither end-to-end latency nor PSO/compiler cost.

## Running and verification

Run from the repository root after all runtime sources, binaries, and tools are
frozen. Generate `FactoryRequests.json` with the production C++ factories and
place the path-to-SHA256 exporter/source/executable map in `ExporterIdentity.json`.
The request structure is `roles.single|general.D0|S0|D1|S1.stages.<stage>`; each
stage includes module, entry, profile, capabilities, defines, and search_paths.
Generate all 32 reference SPIR-V stages from those exact requests, recording their
absolute paths and SHA256 in `reference-spv/Reference.json`. Its `roles` layout
matches the requests, each stage containing `path` and `sha256`; it also records
`schema_version: 1`, `factory_requests_sha256`, `requests_path`, and a `tools`
path-to-SHA256 map. Every actual runtime stage must match its reference bytecode,
so identical module/entry names cannot conceal a missing SimpleBSDF macro.
Preparation does not execute the exporter or compiler.

```powershell
python -B Tools/Perf/SimpleBSDFGPU.py prepare --requests build-release/simple-bsdf-gpu-20261017/FactoryRequests.json --reference-spirv build-release/simple-bsdf-gpu-20261017/reference-spv/Reference.json
python -B Tools/Perf/SimpleBSDFGPU.py run --stage correctness --output-dir build-release/simple-bsdf-gpu-20261017/correctness
python -B Tools/Perf/SimpleBSDFGPU.py run --stage performance --correctness build-release/simple-bsdf-gpu-20261017/correctness --output-dir build-release/simple-bsdf-gpu-20261017/performance
python -B Tools/Perf/SimpleBSDFGPU.py verify build-release/simple-bsdf-gpu-20261017/performance
python -B tests/perf/TestSimpleBSDFGPU.py
```

`--prepared`, `--executable`, `--timeout` and repeatable `--case` override the
declared locations or select a bounded subset. Output directories must be new.
Performance requires the complete three-case correctness capture, identical
frozen source/binary/tool/fixture hashes, and the same actual shader/native PSO
identities. Validation flags alone may differ between the stages. Each stage
holds the shared cooperative lock and launches bounded children serially. Freeze
checks run before and after each child. Fresh application PSO directories are
separate by arm and reused by that arm across blocks; OS/compiler/driver caches
are not claimed cold. External GPU observation may be recorded separately; the
lock does not establish system-wide exclusivity.

The default performance gate requires all correctness contrasts and repeats to
pass. An explicitly authorized diagnostic run can add
`--allow-unqualified-correctness`, only for a complete, sealed `numeric_failure`
capture with successful runtime and unchanged inputs. The override preserves
global failure and the failed contrast's qualification. Each contrast is recorded
independently, so a cross-SER failure cannot hide or disqualify a passing same-SER
wrapper comparison. Local performance readback failures further disqualify the
affected contrast. Never describe diagnostic timing as globally qualified.

The verifier rehashes raw evidence, current frozen inputs, actual SPIR-V and
recipes; checks the archived runner, process commands/environment/order, source
preparation, and linked correctness; and recomputes every numeric comparison and
timing statistic. Actual SER requires EXT SPIR-V identity and enabled native/device
state; NV and shared-hit SER are forbidden. All three hit stages and unrelated
PSOs must match across arms. `verify --allow-numeric-failure` verifies intact
diagnostic evidence without passing its numeric gate. Numeric failure exits 2;
an incomplete or invalid capture cannot qualify even with the override.

The returned `verified` field means evidence integrity. `qualification` in
`Summary.json` holds final case/contrast and global eligibility, including the
linked correctness stage. `analysis` retains local raw comparisons, repeated-arm
results, process medians, and statistics. Report both scope and eligibility.
