# CPU job baseline

`MetallicCPUJobBench` builds independently of the renderer. It uses the locked
stdexec and optional oneTBB checkouts; it neither initializes Vulkan nor creates
rendering worker pools. Run from a matching x64 Visual Studio developer shell:

```powershell
cmake -S Tools/CPUJobs -B build-cpu-jobs -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DMETALLIC_CPU_JOBS_RANGE=ON `
  -DMETALLIC_CPU_JOBS_TBB=ON
cmake --build build-cpu-jobs --target MetallicCPUJobBench MetallicTaskTests
ctest --test-dir build-cpu-jobs --output-on-failure
python -B Tools/MeasureCPUJobs.py `
  --executable build-cpu-jobs/MetallicCPUJobBench.exe `
  --output build-cpu-jobs/evidence --mode all --repeats 20
```

With CMake 4, the locked oneTBB version needs
`'-DCMAKE_POLICY_VERSION_MINIMUM=3.5'`. The benchmark disables oneTBB's optional
allocator, IPO and automatic HWLOC search; its `parallel_for` still uses the
normal oneTBB runtime. `METALLIC_STDEXEC_ROOT`, `METALLIC_GOOGLETEST_ROOT` and
`METALLIC_CPU_JOBS_TBB_ROOT` can point to existing checkouts at the locked
revisions. No dependency is downloaded by this standalone project.

`METALLIC_CPU_JOBS_RANGE=OFF` excludes the candidate implementation; `--mode
baseline` measures the old paths even in a candidate-enabled build. oneTBB is
opt-in, and the fallback configuration requires no oneTBB. To build just the
benchmark without GoogleTest, set `METALLIC_CPU_JOBS_TESTS=OFF`.

The benchmark-only `LegacyMeshletBuildParallel.*` preserves the old independent
pool from `8e0d94587`, with only its namespace/include changed. It never links
into scene/runtime targets. At the same requested worker count, it uses the
calling thread plus `workers-1` background threads; the shared TaskSystem uses
`workers` background threads while the caller joins. Other comparison pools
exist but remain idle during a sample.

Raw records, configuration/host metadata and summaries are separate files.
Each measurement round rotates candidate order. The fixed-seed serial checksum
is checked after every sample, and output is cleared before each invocation.
`first_call` means first backend invocation on already initialized buffers;
it does not mean cold CPU caches. Optional diagnostics are measured separately
with `--diagnostics both`. TaskGraph's existing snapshot instrumentation is
always present and cannot be disabled by the benchmark flag.

The timing window includes scheduling, kernel and join. Graph construction and
buffer initialization are outside it. TaskGraph validation and completion
snapshot copying remain inside it. Process CPU accounting is coarse on short
Windows runs. Per-worker active time has different scopes: graph callback
dispatch/finish versus range/legacy kernel callbacks. Actual sleep/wakeup
transitions and upstream stdexec/oneTBB steal counters are unavailable; queue
delay must not be described as wakeup delay. Use enough repeats for tails, and
do not convert these measurements into renderer FPS claims.

## Range identity example

`MetallicRangeDiagnosticsExample` is a manual `EXCLUDE_FROM_ALL` target available
when `METALLIC_CPU_JOBS_RANGE=ON`. It performs a deterministic producer/range/
nested-consumer correctness example and writes one JSON file. It checks each
lane's graph, execution, node, thread and time envelope against the completed
TaskGraph snapshot before accepting the output. It is not a benchmark.

From a root build, explicitly build the target with
`cmake --build build-issue73 --target MetallicRangeDiagnosticsExample`, then run
`build-issue73/cpu-jobs/MetallicRangeDiagnosticsExample.exe <fresh-output.json>`.
For the standalone CPUJobs project, the executable is directly in its build
directory. Compiler, stdexec and build options are the same as the Task library.
No renderer, Vulkan, Slang, ISPC or JSON library is added to the task layer.

An [archived example](../../Documentation/Measurements/CPUJobsProduction20261107/RangeDiagnostics/Example.json)
and [provenance](../../Documentation/Measurements/CPUJobsProduction20261107/RangeDiagnostics/Metadata.json)
show four asynchronous lanes and one nested inline lane. The inline lane
borrows its parent's task identity and has null queue delay. Async queue delay
is derived from the TaskGraph's scheduled/start times; it is not wakeup delay.
Operation creation time may precede submission by an arbitrary interval.

These diagnostics are opt-in through `ParallelForOptions::diagnostics` and an
optional generic `backendLabel`. The production meshlet adapter still uses
null diagnostics. A Task event observer can capture its actual range-task
graph/node/thread intervals, but no range steal count exists unless that
owning caller explicitly opts into a diagnostics object. Missing counters
must not be fabricated from worker or task counts.
