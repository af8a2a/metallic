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

The output directory must be new. Default budgets are three discovery ABBA
blocks and, only after discovery acceptance, two confirmation ABBA blocks.
Each run has a 300-second timeout. `--blocks` and `--confirmation-blocks`
accept 2–5; the case retains its per-process rounds and frame counts. Processes
are serial. Build failures, process failures, missing evidence and restoration
conflicts produce `inconclusive`, never a speedup. Exit code 0 means a completed
accept/reject decision, 2 means inconclusive, and 1 means invalid invocation or
failed verification. Inspect `Decision.json`, not just the exit code.

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
   A mismatch rejects the candidate. This establishes equivalence to the baseline
   for these outputs/camera histories, not correctness against a separate renderer
   or a final HDR image metric.
3. Available workload invariants must match across arms: case, camera, render
   extent, graph, history policy, cut/page hashes, software lists, indirect groups,
   residency and production binding fields. Only the compiled SPIR-V fingerprint
   may differ. An unplanned identity change is inconclusive. This does not claim
   a complete byte snapshot of every HZB/history input.
4. Each arm must pass the case's existing A/A spread limit for software total and
   graph time. The default 10% limit is unchanged. Background GPU activity remains
   archived; it is not forcibly excluded or independently proven harmless.
5. The statistical unit is an adjacent pair of independent-process medians. ABBA
   alternates AB and BA pair order. For each pair, gain is `1 - B/A`. The runner
   uses the two-sided 95% Student-t interval on pair gains, with 4–10 pairs per
   stage. Frame samples are not counted as independent experiments. Small-sample
   intervals and order balancing reduce uncertainty but do not prove absence of
   correlated environmental drift.
6. Acceptance requires the software-total gain interval's lower bound to be at
   least 3%, and the graph gain lower bound to be at least -2%. An upper bound
   below either gate rejects; an interval crossing a gate is inconclusive.
   Discovery acceptance requires a separate confirmation stage with the same
   gates. No threshold tuning or repeated retries are performed automatically.

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
requires discovery plus confirmation and successful restoration for acceptance.
It checks process lifetimes, nonoverlap and containment of measurement windows;
copying a run into another directory does not create an independent sample.
Original producer tool bytes/hashes are preserved in the bundle. The current
verifier checks those archived hashes without executing archived code, recomputes
the result under its current policy, requires agreement with saved decisions and
reports its own hash separately. A changed policy or incompatible recomputation
fails; original evidence is never rewritten to fit a newer verifier.
The manifest is an integrity index, not a cryptographically signed attestation
against intentional rewriting of all evidence and hashes.

Full Zorah is excluded from this first loop because M2 found cross-process texture
residency differences. Nsight source exports may support a hypothesis but are
diagnostic evidence; this runner does not automate the remaining M1 UI adapter
or require a UI session to decide native timing results.
