# CPU job scheduling measurements — 2026-11-06

The measurements support a shared range API and workload-specific grain tuning.
They do **not** justify replacing stdexec's worker deque or changing the default
TaskSystem scheduling policy. The unified range bridge is not uniformly faster
than the old Meshlet worker pool. No result here establishes renderer FPS or
actual geometry-cooking speedup.

## Reproduction and scope

- Machine: AMD Ryzen 7 7800X3D, 8 physical cores / 16 logical processors,
  Windows 11 build 26300; MSVC 19.44.35229, Release `/O2`, x64. No process/thread
  affinity or ISA flag was imposed. Other agent builds/downloads were paused
  during the complete measurement window; normal desktop/OS activity remained.
- Locked stdexec `6d7ad689f4d4831c5136e4abe1c601f9a3b64e43`; oneTBB
  `a00cc3b8b5fb4d8115e9de56bf713157073ed68c` (2021.9.0). oneTBB's optional
  allocator, IPO and automatic HWLOC lookup were disabled for this CPU-only
  comparison. No vendor sources changed.
- The legacy pool is a benchmark-only copy from repository `8e0d94587`.
  **TaskGraph is measured from this branch's local implementation**, including
  the admission/shutdown lock and parent-stop-token TLS changes; it is not an
  exact issue-start binary. It still uses the existing directed stdexec
  scheduling policy. The raw stdexec comparison bypasses TaskGraph entirely.
  Each `Metadata.json` includes the binary hash and explicit source hashes,
  including new/untracked sources, rather than relying on the base commit alone.
- Uniform/random/long-tail/locality use 65,536 elements; bandwidth reads and
  writes 16,777,216 `uint32_t` values (128 MiB input plus output total). Fixed
  seeds, identical kernels and checksum validation for every sample. Long-tail
  puts 8,192 mix rounds in the first eight elements of each 1,024-element block,
  versus eight rounds elsewhere; random uses 1–128 rounds.
- 1/2/4/8 workers; 20 warm repeats plus one first invocation for each case.
  Candidate order rotates inside every round. Buffer setup, clearing and
  checksum verification are outside the timing window. Timings cover per-call
  state allocation, scheduling, useful work and join. TaskGraph construction is
  outside; `submit` validation and completion snapshot copying are inside.
- The old pool includes the calling thread among its worker count. The unified
  runtime uses that many background workers and an external joining caller.
  Comparison pools coexist but remain idle while another backend is timed.
  This is an explicit implementation comparison, not identical thread topology.
- `first_call` is a first backend invocation on initialized/serial-validated
  buffers, **not flushed-cache execution**. Pool-construction times are retained
  separately in raw rows; construction order is fixed, so those single samples
  are not a comparative startup study. Warm repeats reuse buffers; the large
  bandwidth working set still exceeds the CPU's cache capacity.

The exact build/run commands are in [the benchmark instructions](../../../Tools/CPUJobs/README.md)
and each metadata file. The retained sets are:

| Set | Purpose | Raw samples |
| --- | --- | ---: |
| [Baseline](Baseline/Metadata.json) | 256-leaf flat/fan-out DAGs, raw stdexec ordinary/directed, old pool and oneTBB; grain 64; diagnostics off/on | 6,720 |
| [Range](Range/Metadata.json) | Static/Dynamic/AdaptiveHalfSteal versus old pool and oneTBB; grain 64; diagnostics off/on | 4,200 |
| [Grain1](Grain1/Metadata.json) | Original single-item claim versus candidate strategies, uniform/long-tail; diagnostics off | 1,848 |
| [Grain256](Grain256/Metadata.json) | Uniform/long-tail range-grain calibration; diagnostics off | 840 |
| [SmallTasks](SmallTasks/Metadata.json) | 1,024 elements split into 1,024 DAG leaves; diagnostics off/on | 1,344 |

All **14,952** sample checksums matched the serial reference. `Raw.jsonl` retains
every invocation; `Summary.json` retains P50/P95/P99, throughput and available
diagnostics. Quantiles use `floor(q*(n-1))`; twenty repeats do not establish
production tail probabilities. P95 and P99 often select the same sample.

## Scheduler decision

Warm P50 milliseconds, optional diagnostics off:

| 256-leaf fan-out/join | Workers | Raw ordinary | Raw directed | TaskGraph directed |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4 | 3.118 | 3.048 | 3.348 |
| Long-tail | 4 | 3.462 | 3.643 | 4.331 |
| Uniform | 8 | 1.762 | 1.637 | 1.759 |
| Long-tail | 8 | 1.852 | 1.855 | 2.207 |
| SmallTasks uniform, 1,024 leaves | 4 | 0.198 | 0.175 | 1.190 |
| SmallTasks uniform, 1,024 leaves | 8 | 0.223 | 0.206 | 1.685 |

Raw `fanout_join` schedules leaves from a pool callback and joins with
`async_scope::on_empty`; TaskGraph also has explicit root/join graph nodes.
The raw comparison isolates upstream ordinary versus directed scheduling, not
all of TaskGraph's dependency/lifecycle/observability costs.

Ordinary scheduling is not consistently better than the current directed
policy. The tiny-node TaskGraph gap suggests graph-level per-node bookkeeping,
synchronization and snapshots deserve focused profiling; these samples do not
isolate which of those costs dominates. It would be incorrect to blame the
stdexec deque or claim a custom deque would eliminate that gap. The range
bridge amortizes graph work across a bounded number of lane tasks. M4 remains
untriggered: retain the mature backend and its existing policy.

## Range results and grain selection

Warm wall P50/P95 milliseconds, grain 64, optional diagnostics off:

| Workload | Workers | Legacy atomic | Static | Dynamic | Half-steal | oneTBB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4 | 2.954 / 3.103 | 3.087 / 3.410 | 3.018 / 3.122 | 2.988 / 3.144 | 2.967 / 3.147 |
| Long-tail | 4 | 3.400 / 3.553 | 3.522 / 4.045 | 3.416 / 3.651 | 3.374 / 3.513 | 3.416 / 3.612 |
| Bandwidth | 4 | 12.858 / 14.687 | 3.615 / 4.395 | 12.245 / 14.348 | 3.524 / 4.091 | 3.569 / 4.075 |
| Uniform | 8 | 1.569 / 1.729 | 1.606 / 2.511 | 1.556 / 1.798 | 1.569 / 1.891 | 1.532 / 1.693 |
| Long-tail | 8 | 1.726 / 1.832 | 1.737 / 2.136 | 1.787 / 1.875 | 1.700 / 1.876 | 1.744 / 1.883 |
| Bandwidth | 8 | 10.250 / 10.619 | 3.515 / 3.967 | 13.202 / 13.513 | 3.381 / 3.790 | 3.399 / 3.901 |

Compute-heavy cases are similar; there is no general win over the legacy pool
or oneTBB. On this contiguous bandwidth kernel, small globally claimed batches
are costly, while static/local interval claims avoid that shared bottleneck.
This explains a candidate tuning direction, not a universal half-steal gain.
These long-tail workloads vary callback cost across 65,536 elements, which
divides evenly into each measured grain (1, 64, 256). They do not exercise a
short final batch. When a victim holds one full grain plus a short tail,
AdaptiveHalfSteal leaves the tail with the owner to avoid a sub-grain steal.

Four-worker warm P50 milliseconds for the uniform and long-tail calibration:

| Grain | Uniform legacy / Dynamic / Half-steal | Long-tail legacy / Dynamic / Half-steal |
| ---: | ---: | ---: |
| 1 | 3.496 / 3.471 / 3.123 | 4.295 / 4.538 / 3.506 |
| 64 | 2.954 / 3.018 / 2.988 | 3.400 / 3.416 / 3.374 |
| 256 | 3.031 / 3.096 / 3.079 | 3.379 / 3.391 / 3.487 |

Grain groups were collected in separate processes/windows, so cross-grain
changes include that variance; candidates within a group are interleaved.
Grain 1 is a compatibility default, not an optimum. Real meshlet task costs
differ from these synthetic kernels. Keep the production meshlet policy at
Dynamic/grain 1 until geometry-cooker measurements justify a different grain;
do not transfer the synthetic grain 64 or half-steal policy automatically.

## Observability and limits

TaskGraph snapshots and instrumented raw stdexec tasks expose queue delay and
per-physical-thread active intervals. Instrumented range/legacy callbacks expose
per-lane active time; range diagnostics also expose claim attempts and
half-steal attempts/successes. Four-worker bandwidth half-steal had median 49
steal attempts / 33 successes and 262,183 claim attempts; Dynamic had median
346,985 claim attempts. These are range CAS counters, not upstream deque steals.

Instrumentation is material for tiny batches. In the Range bandwidth group,
four-worker Dynamic P50 changed from 12.245 ms to 16.498 ms with diagnostics;
half-steal changed from 3.524 ms to 4.023 ms. Off/on groups are separate runs,
so their difference is not a precise instrumentation-cost estimate. Existing
TaskGraph snapshot instrumentation remains enabled in both groups.

Actual worker sleep/idle transitions, wakeup delay distinct from queue delay,
upstream stdexec/oneTBB steal counters, atomic time isolated from useful work,
system-wide CPU utilization and hardware SIMD utilization are unavailable.
`processCpuNs / wallNs` reports process CPU cores with coarse Windows accounting;
it is noisy for short tasks. Wall minus callback active time includes scheduling
and bookkeeping and must not be reported as measured worker sleep time.

The archived standalone Release regression binary passed **30/30 Task and
ParallelFor tests**, including 1/2/4/8 workers, exact coverage, empty/tail/large
ranges, exceptions, cancellation, nested inline execution, concurrent submit
and shutdown. These are correctness checks, not speed assertions. Windows
ThreadSanitizer was unavailable. Actual meshlet/scene and Vulkan regressions,
scene critical path, frame statistics and resource peaks require separate
runtime evidence and are not established by this report.
