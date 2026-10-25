# Frozen MiniZorah streaming NvPerf diagnostics

**Retired: verification-only.** Spatial candidates were withdrawn after the bounded experiments. `run` now fails immediately, before inspecting inputs or creating files, locks or processes. Historical `verify`, decoding and statistical policies remain unchanged. Collection commands below document the historical protocol and cannot launch a new experiment.


`MeasureSpatialDemandCounters.py` uses the existing frozen `streamingRegression`
scene and preserves `captureQuality=false` / `captureReflections=false`. It is a
separate diagnostic protocol (`metallic-spatial-demand-counters-v2`), never an
ordinary timing experiment or candidate acceptance mechanism. Historical v1 counter evidence remains bound to spatial v3; new v2 manifests
explicitly declare spatial v4. Historical evidence is unchanged.

Each requested metric group has exactly six serial processes in **ABBAAB** order:
three independent captures per arm. `--runs 1` instead uses **AB**, explicitly a
pilot with n=1 and no repeatability claim. The schedule is fixed; failures stop
without automatic retries or metric substitution. Each process profiles one frame,
with precisely `Streaming/Seed`, `Streaming/Demand`, `Streaming/Frontier` around the
actual graphics-queue dispatches, excluding their surrounding barriers. A complete
single SDK pass and zero dropped ranges/trace bytes are mandatory.

```powershell
python -B Tools/Perf/MeasureSpatialDemandCounters.py run `
  --exe <existing-NvPerf-Release>/Source/MetallicGPUDrivenSample.exe `
  --build-dir <existing-NvPerf-Release> --config <frozen-MiniZorah-config.json> `
  --assets <verified-mini-assets.json> --metrics <metric-groups.json> `
  --output <fresh-output-directory> --runs 3 --timeout 240
python -B Tools/Perf/MeasureSpatialDemandCounters.py verify <output-directory>
```

The tool does not build, configure, install an SDK, change clocks, or start an
additional background collection service. It uses `RunZorahFullRoam.ps1` for the
existing bounded renderer lifecycle and GPU/process telemetry. The cooperative lock
and those files do not establish system-wide GPU exclusivity. An outer watchdog
only reclaims its own launcher process tree. PDH failures remain diagnostic logs;
no missing counter or telemetry sample is replaced by zero.

## Exact metric candidates

`--metrics` is a JSON object mapping a simple group name to 1-16 unique exact metric
names. Omitting it requests these three declared candidates. Their metric bases and
definitions were checked in SDK 2026.3's `NvPerf/doc/metrics/MetricsEnumerator-GB20X.html`.
Suffix evaluation, device counter availability and whether a group fits a single
pass still require the SDK at runtime. A directory listing is not hardware support
validation. Do not silently switch names or split a rejected multipass group.

```json
{
  "packing": [
    "gpu__time_duration.sum",
    "sm__ctas_launched.sum",
    "sm__warps_launched.sum",
    "smsp__inst_executed.sum",
    "smsp__thread_inst_executed.sum",
    "smsp__thread_inst_executed_pred_on.sum"
  ],
  "memory": [
    "gpu__time_duration.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_realtime.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_realtime.sum",
    "lts__t_sectors.sum",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum"
  ],
  "issue": [
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct"
  ]
}
```

- `ctas_launched` and `warps_launched` are launch counts, not the application's task
  count. A persistent worker may process many tasks.
- `thread_inst_executed / inst_executed` estimates average active lanes per warp
  instruction; the pred-on numerator additionally excludes instruction-predicated
  lanes. Neither ratio is occupancy.
- Global-load L1 requested/hit sector counts yield a matching hit fraction. The L2
  hit percentage is its broader SDK-defined domain; it is not specifically the
  spatial metadata buffer's hit rate. Retain the absolute sector counts as well.
- DRAM read/write counters are absolute byte counts. Preserve SDK dimensional units
  for every metric; do not silently convert sectors to bytes using an assumed size.
- Active warps are in-flight warps; eligible warps can issue. Long-scoreboard reports
  waiting for L1TEX-operation dependencies, not necessarily DRAM misses. Barrier and
  membar waits are distinct. Stall percentages are not recoverable execution time.

## Evidence and limits

The collector archives the selected config, verified asset manifest, source/tool,
Release cache, executable/DLL and matched SDK header/runtime hashes. It checks this
identity before and after each process and reuses the existing asset metadata
checks. The complete output inventory is hashed, including raw availability,
configuration, prefix and counter image files. Fresh output and lock ownership are
required. SDK/Vulkan official support status is retained, never inferred from a
successful collection.

The existing 14-buffer decoder validates before/after snapshots: actual TLAS
addresses, cut ordering, page/CLAS state, instances, requests, texture publication,
original topology/primitive hashes and A/B metadata layout. These snapshots explicitly
are **not the measured target frame**. Existing-graph depth and visibility images
must be finite and byte-identical before/after and across every process; no
ReflectionQuality resource is required. The producer warms 16 ordinary frames
after the before readback and before starting counters. Actual target range metadata independently
records and verifies phase, shader module/entry/SPIR-V, workgroup/subgroup, thread
count and dispatch XYZ against the snapshot's bound shader identity. No readbacks
are inserted inside counter ranges. The two stability snapshots must agree in work
counts before those counts can serve as denominators.

Every arm reports all raw values, median and relative spread. Demand `.sum` metrics
also report ratios per snapshot processed task, visited node and tested group;
these are deferred-stability-workload normalizations, not direct target-frame
per-task attribution. A zero denominator remains null. Ratios are derived only when
both counters were collected together in the same command range. Missing metrics
remain missing. n=1 pilots do not establish repeatability, confidence intervals,
normal-timing speedups or optimization acceptance.

The verifier rechecks file inventory, raw buffer semantics, exact range/dispatch
identity and recomputes summaries. It does not reevaluate the raw counter image with
the SDK (`counterImageReevaluated=false`). Command ranges do not establish isolated
replay, source-line/PC/SASS attribution, or behavior under roaming/live residency.


## Counter protocol v2

New counter manifests use `metallic-spatial-demand-counters-v2` and explicit `spatialProtocol: metallic-spatial-demand-ab-v4`. New layout and task-count equality gates are inherited from the v4 ordinary evidence validator. Counter protocol v1 is permanently decoded as spatial v3, regardless of the current default protocol. Existing single-pass three-range, shader/dispatch identity, 14-buffer semantic and exact before/after image checks remain unchanged. No ordinary timing acceptance is inferred.

## Measured GB203 qualification (2026-10-25)

On the local RTX 5070 Ti with SDK 2026.3, the three default **candidate** groups
above required 6, 2 and 9 passes respectively and were rejected. They remain
explicit capability probes, not qualified single-pass defaults. For a repeat of
the measured workload, pass an explicit JSON metric plan with `--metrics`; the
following group was collected successfully in three independent processes per
arm for both spatial v3 and v4:

```json
{
  "participation_load": [
    "gpu__time_duration.sum",
    "sm__inst_executed_realtime.sum",
    "sm__thread_inst_executed_pred_on_realtime.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_realtime.sum",
    "lts__t_sectors_op_read_realtime.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum"
  ]
}
```

This is machine/SDK-specific evidence; the backend still checks required passes
at capture time. The realtime pred-on thread/warp-instruction ratio is a
participation proxy, not the unavailable original SMSP ratio or occupancy.
Requested L1 sectors and hit sectors did not fit together in one pass; do not
divide values from separate groups to invent a same-run hit fraction.

`Streaming/Seed` includes the entire Clear dispatch (sparse state/history clearing,
root filtering and task compaction). Demand excludes DemandReset; Frontier
excludes the separate Mask/Prefix/Emit dispatches. These counter scopes differ
from the wider ordinary timestamp scopes, including their barriers.

See [the MiniZorah counter and A/B report](../../Documentation/MiniZorahSpatialDemandCountersAB.md)
for qualification failures, raw-image offline positive controls, repeatability,
the rejected v4 candidate and the preserved Vulkan 1.4 support warning.
