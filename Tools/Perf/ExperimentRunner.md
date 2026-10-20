# Shader experiment runner

`ExperimentRunner.py` implements the first M3 closed loop for the qualified
MiniZorah WorkControl history case. It compiles the host target, applies one
declared shader candidate, runs independent processes in ABBA order, checks
outputs and workload identity, makes a decision, and restores the baseline.
No candidate is installed permanently, including an accepted candidate.

## Run

Use an x64 Visual Studio developer shell with CMake, Python and `nvidia-smi` on
PATH. GPU execution needs the same host permissions as the M2 runs.

```powershell
python -B Tools/Perf/ExperimentRunner.py run `
  --case Tools/Perf/WorkloadCase.MiniZorahHistory.json `
  --candidate Tools/Perf/Candidate.CompactWorkVertices.json `
  --assets build/m2-assets-20260925/mini.json `
  --exe build-release/Source/MetallicGPUDrivenSample.exe `
  --build-dir build-release `
  --output build/my-new-experiment

python -B Tools/Perf/ExperimentRunner.py verify build/my-new-experiment
```

The output directory must be new. Fix both stage budgets before starting:
defaults are three discovery ABBA blocks and, for a promising discovery result,
five independent confirmation ABBA blocks. `--blocks` and
`--confirmation-blocks` each accept 2–10 (at most 80 processes across both
stages). Each process has a default 300-second timeout, configurable with
`--timeout`; the case retains its per-process rounds and frame counts. Processes
are serial. There is no adaptive sample extension or significance-based early
stopping. Evidence or correctness failures can stop collection.

Build failures, process failures, missing evidence and restoration conflicts
produce `inconclusive`, never a speedup. Exit code 0 means a completed
accept/reject decision, 2 means inconclusive, and 1 means invalid invocation or
failed verification. Inspect the classifications and `nextAction` in
`Decision.json`, not just the exit code. A discovery result may be inconclusive
and still schedule confirmation.

## Candidate and build contract

The first version permits only
`Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang`. The candidate JSON
records its hypothesis, baseline byte SHA-256, and exact text replacements with
expected counts. A mismatched hash/context, unknown path or textual no-op fails.
The example changes shared screen vertices from four words to three words.
This is a hypothesis about shared storage/bank layout; actual allocated storage,
registers and occupancy are not inferred from the source declaration.

The runner invokes `cmake --build ... --target MetallicGPUDrivenSample --config
Release`. It requires a single-config Release cache for this source tree and the
matching target executable; the cache and its hash are archived. Candidate shaders are compiled by the production Slang path when each
process loads the graph, before warmup and measurement. The bound SPIR-V must be
stable within each arm and different between arms. Identical compiled shaders
produce `inconclusive`, even if noisy timings appear faster. Arbitrary C++,
pipeline, quality-setting and multi-file candidates are outside this version.

The complete source inventory and runtime binary hashes are checked around each
run. Source copies, candidate bytes/patch, tool copies/hashes, build logs, case,
declared asset hashes, GPU identity, process records, PDH telemetry, raw readbacks,
frame timings and decisions are retained. Asset originals stay at their declared
paths; size/mtime are rechecked against the M2 content-hash manifest. This is not
a portable copy of the 71 GB MiniZorah asset closure or a fresh content rehash on
every run. Do not modify source/assets or run another GPU experiment concurrently.

## Gates

1. Every run must satisfy `WorkloadCase.analyze_run`, including nonzero late,
   overflow/binding checks, stable per-frame residency, valid readback bytes,
   complete timings and exclusion of known capture instrumentation. The process
   must exit normally and PDH must cover every round's measurement window.
   The sampler waits for the target process's GPU instance before opening its
   continuous query, and retains target zero values. Background instances are
   those visible when this query is opened; later-created competitors may be
   absent, so it does not certify global GPU exclusivity.
2. All diagnostic depth and visibility readbacks must match exactly across A/B.
   An A/B mismatch after input and reference-baseline identity checks pass rejects the
   candidate. This establishes equivalence to the baseline for these
   outputs/camera histories, not correctness against a separate renderer or a
   final HDR image metric.
3. Available workload invariants must match across arms: case, camera, render
   extent, graph, history policy, cut/page hashes, software lists, indirect groups,
   residency and production binding fields. Only the compiled SPIR-V fingerprint
   may differ. Confirmation must also match discovery's inputs, outputs and each
   corresponding arm's complete identity, including its SPIR-V fingerprint.
   Version 2 checks input drift before attributing differing outputs to candidate
   correctness. An unplanned identity change is inconclusive. This does not claim
   a complete byte snapshot of every HZB/history input.
4. The case's existing per-arm spread limit for software total and graph time
   remains a diagnostic threshold (10% by default). Exceeding it adds a warning;
   it does not discard otherwise valid A/B evidence or bypass gain calculation.
   The standalone `WorkloadCase` A/A qualification rule is unchanged. Background
   GPU activity remains archived; it is not forcibly excluded or independently
   proven harmless.
5. One complete ABBA block is the statistical unit. For each metric, take each
   process's median, then compute the block's mean log ratio:
   `(log(B1/A1) + log(B2/A2)) / 2`. The two adjacent pairs within a block are not
   independent samples. Compute the two-sided 95% Student-t interval across
   block log ratios, then transform the estimate and reversed interval bounds
   to gains with `1 - exp(logRatio)`. Each stage has 2–10 blocks built from new
   processes; frame samples are not independent experiments. This accounts for
   pairing and within-block correlation, but does not establish independence
   across blocks or rule out environmental drift.
6. Discovery screens candidates using its full fixed budget. An upper gain bound
   below zero establishes a software regression; a graph upper gain bound below
   -2% exceeds the graph regression budget. A software upper gain bound below
   3% is below the target. These results reject. Otherwise, a software point
   estimate of at least 3% is promising and schedules independent confirmation,
   even when discovery cannot establish the final acceptance bounds. Other
   uncertain results retain the candidate and evidence.
7. Only confirmation can accept a new candidate. Its own software gain interval
   must have a lower bound of at least 3%, and its own graph gain lower bound
   must be at least -2%. Discovery data are not pooled into this decision.
   Confirmation rejects established regression or a software upper gain bound
   below 3%; other results remain inconclusive when the fixed budget is spent.
   No threshold tuning, automatic extension or repeated retries are performed.

## Result classification

The top-level `accept / reject / inconclusive` status is retained. Version 2 also
records `stage`, `evidenceValidity`, `effectAssessment`, and `nextAction`, so a
missing measurement and a promising but imprecise measurement do not imply the
same follow-up.

| `effectAssessment` | Meaning and next step |
| --- | --- |
| `not_assessed` | `evidenceValidity` is `invalid` or `incomplete`: a collection, identity, integrity or restoration problem prevents a performance conclusion; follow the recorded repair/collection action. |
| `correctness_failure` | Recorded outputs differ from the baseline; `reject_candidate` regardless of timing. Here `evidenceValidity=valid` describes the evidence, not the candidate's correctness. |
| `no_shader_change` | Both arms bind the same compiled shader; `revise_candidate`, without a speedup conclusion. |
| `confirmed_regression` | The interval gives evidence of a software slowdown or a graph regression beyond the 2% budget; `reject_candidate`. `stage` identifies where this was observed; rejection does not require independent confirmation. |
| `below_target` | The software interval's upper bound is below the 3% target; `reject_candidate` under this policy. A positive gain below 3% is not “no benefit.” |
| `promising` | Discovery's software estimate is at least 3%, without established regression or an upper bound below target; reason `promising_gain`, status `inconclusive`, `nextAction=run_confirmation`. |
| `uncertain` | The fixed budget does not resolve the decision; `inconclusive` with `nextAction=retain_candidate`. |
| `confirmed_gain` | Independent confirmation satisfies both lower bounds; `accept_candidate`, subject to successful restoration. |

Every result retains the candidate patch and collected evidence. Spending the
budget without confirmation does not disprove the hypothesis. A later independent
confirmation plan must declare its budget before observing its new data and
retain all earlier rounds; do not repeatedly rerun until one interval passes.
The current runner starts a new experiment rather than resuming or expanding an
archived stage.

## Restoration and recovery

The repository shader is temporarily changed while the runner owns
`build/shader-experiment.lock`. The journal and baseline bytes are written before
mutation. `finally` restores the baseline on normal decisions, exceptions and
timeouts. A concurrent user edit is preserved rather than overwritten; the
decision becomes inconclusive and the lock/evidence remain for inspection.

After a hard interruption, recover only after the owning process has stopped:

```powershell
python -B Tools/Perf/ExperimentRunner.py recover build/my-new-experiment
```

Recovery checks the lock owner and baseline/candidate hashes. Unknown current
bytes require manual reconciliation from `Baseline.slang`; they are never
overwritten. Recovery does not turn an interrupted experiment into an accepted
one. The lock coordinates these runners only; it cannot prevent editors or
unrelated applications from accessing the repository or GPU.

## Offline evidence

`verify` checks all archived file hashes, the parser/tool revision, replacement
result, readback bytes and raw frame data, recomputes each stage decision, and
requires a qualifying discovery result, independent confirmation and successful
restoration for acceptance.
It checks process lifetimes, nonoverlap and containment of measurement windows;
copying a run into another directory does not create an independent sample.
For version 2, it also checks the declared stage budgets, complete ABBA ordering
and the decision that authorized confirmation. Live collection and verification
both require confirmation to preserve discovery's inputs, outputs and the
complete identity of each corresponding arm.
Original producer tool bytes/hashes are preserved in the bundle. The current
verifier checks those archived hashes without executing archived code and reports
its own hash separately.

Candidate JSON remains `metallic-shader-experiment-v1`; new experiment manifests
use `metallic-shader-experiment-v2`. The verifier selects the exact supported
policy from the manifest protocol and requires agreement with saved decisions:

| Manifest | Recomputed policy |
| --- | --- |
| `metallic-shader-experiment-v1` | Original per-arm spread hard gate, adjacent-pair gain Student-t interval, and discovery acceptance before confirmation. |
| `metallic-shader-experiment-v2` | Spread warnings, ABBA-block log-ratio interval, discovery screening, independent confirmation and explicit result classifications. |

Unknown protocols or altered/unknown policies fail verification. Existing v1
archives retain their original decisions and are never rewritten or reclassified
to fit v2. Any new analysis belongs outside the original evidence bundle and must
identify its separate policy; it does not replace the historical verdict.
The manifest is an integrity index, not a cryptographically signed attestation
against intentional rewriting of all evidence and hashes.

Full Zorah is excluded from this first loop because M2 found cross-process texture
residency differences. Nsight source exports may support a hypothesis but are
diagnostic evidence; this runner does not automate the remaining M1 UI adapter
or require a UI session to decide native timing results.
