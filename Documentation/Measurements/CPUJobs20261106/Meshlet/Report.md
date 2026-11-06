# Shared-runtime meshlet migration

The shared TaskSystem implementation produces identical logical geometry to
the original independent meshlet pool, including every meshlet/LOD field.
It removes the extra production pool, but does not establish a cooker speedup:
several measured multi-worker cases are modestly slower than the legacy pool.
The production policy remains Dynamic with one geometry group per claim;
ordered publication inside the existing builder requires monotonic claims.

The real meshlet builder processed generated 64/128-cell grids, with plain,
attributed and protected-seam geometry. Six alternating AB/BA process rounds,
1/2/4/8 workers, serial references, warmup and fresh output per invocation
produced 288 steady samples. All serial/parallel/repeated/cross-executable
geometry hashes agreed. These are generated geometry workloads, not production
scene loading, renderer CPU critical paths or FPS.

Warm P50 milliseconds (six samples per cell):

| Input | Cells | Legacy 1 | Shared 1 | Legacy 4 | Shared 4 | Legacy 8 | Shared 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain | 64 | 7.431 | 7.065 | 5.341 | 5.635 | 5.299 | 5.805 |
| Plain | 128 | 31.120 | 31.253 | 19.461 | 20.515 | 18.166 | 18.403 |
| Attributes | 64 | 9.594 | 9.536 | 6.755 | 6.938 | 6.457 | 6.805 |
| Attributes | 128 | 39.294 | 40.981 | 23.156 | 24.994 | 20.579 | 22.367 |
| Seams | 64 | 7.469 | 7.476 | 5.512 | 5.472 | 5.205 | 5.069 |
| Seams | 128 | 31.963 | 30.162 | 22.923 | 24.039 | 22.667 | 23.322 |

P95/P99 and paired ratios are retained in [Summary.json](Summary.json).
With six observations, nearest-rank P95/P99 are maxima, not rare-tail estimates.
The measurement window had no concurrent agent compilation or benchmarks;
normal desktop activity remained. Input generation/copy, hashing, runtime
construction and destruction are outside the timed builder call. The first
serial builds and warmup records are retained separately in [Raw.jsonl](Raw.jsonl).
The machine, execution order, source/binary hashes and exact conditions are in
[Metadata.json](Metadata.json). Original per-process records are under `Processes/`.

```powershell
cmake --build build-issue73 --target MetallicMeshletSceneBench MetallicMeshletSceneBenchLegacy
python -B Tools/CPUJobs/MeasureMeshletScene.py `
  --new build-issue73/cpu-jobs/MetallicMeshletSceneBench.exe `
  --legacy build-issue73/cpu-jobs/MetallicMeshletSceneBenchLegacy.exe `
  --output <fresh-directory> --rounds 6
```

The normal Release build also built Task/Scene tests and both offline cookers.
Task tests passed 30/30. The sixteen range tests passed another 100 consecutive
iterations, including shutdown/admission races, nested cancellation and exact
coverage. Scene tests passed 160/169 with nine explicit skips for unavailable
USD/local assets. Windows sandbox temporary-directory atomic replacement
returned error 5; rerunning Scene tests with normal file permissions and a
worktree-local `.cache` temporary directory passed in 7.6 seconds. The meshlet
fixture separately verifies deep LOD equality for one/four workers and scratch
slot exclusivity. ThreadSanitizer was unavailable on this Windows toolchain.
