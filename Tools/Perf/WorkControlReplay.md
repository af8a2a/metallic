# WorkControl isolated replay

Contract: [resource closure and gates](../../Documentation/WorkControlReplayContract.md).
This is an opt-in diagnostic path. Ordinary frames, shader code and optional
shader warmup targets are unchanged. No shader candidate is installed.

```powershell
python -B -X utf8 Tools/Perf/WorkControlReplay.py run --case Tools/Perf/WorkloadCase.MiniZorahHistory.json --assets build-release/shader-trace-p2-verified-20260927/Assets.json --exe build-release/Source/MetallicGPUDrivenSample.exe --phase early --output build-release/replay-new
python -B -X utf8 Tools/Perf/WorkControlReplay.py verify build-release/replay-new
```

Use a fresh directory. Default: three serial independent processes, 240 seconds
per process. `--runs 1` is a pilot, not a three-process acceptance. `--phase late`
selects the late dispatch from the primed target frame. Both phases reuse the
existing WorkloadCase camera/history/asset configuration and P3-style before /
control / after checkpoints. The selected capture is armed AFTER the four prime
frames, immediately before the target frame.

The production pipeline's retained `PreparedExecution` executes against private
buffer allocations at the same typed descriptor indices. Full push bytes,
compiler/device SPIR-V, original/scratch addresses, descriptor stride and size,
indirect bytes and nested bins/raster-binding indices are archived. Physical
addresses differ by design. This is executable and logical binding equivalence,
not physical descriptor equality. The first slice requires subgroup minimum 32;
the low-wave fallback is rejected. Snapshots have a 1 GiB aggregate source-byte
budget. Copies and archives can consume several GiB; they are outside ranges.

Only the actual target dispatch is replayed. No cull, resolve, frame advance,
residency/page request consumer or history publisher runs in the isolated
submission. All RHI queue submissions are serialized by the replay owner after
the control frame drains. Input restoration, copy/readback and comparison each
use separate submissions. All scratch inputs are reset and byte-checked before
EVERY dispatch, including nominally read-only buffers; the full pixel output is
compared to the immediately captured same-frame control. Read-only scratch bytes
and protected production page/request/LOD/HZB/visibility buffers are checked too.

## Counters and multi-pass

Build the optional backend with a matching SDK as in [NvPerf.md](NvPerf.md), then
add `--counters`. Defaults remain the existing two metrics, GPU duration and SM
active cycles. Two unprofiled restore/output checks and a production-state check
must pass before the NvPerf session is created. Every subsequent SDK pass gets
another independently restored dispatch and output check. SDK completion drives
the loop, with a hard limit of 16 passes and one range per pass.

`--split-metric-passes` deliberately schedules the same requested metrics into
separate SDK pass groups. This exercises multi-pass restoration without adding
counter names. The actual required pass count comes from the SDK config image;
the backend rejects this option if the SDK still schedules fewer than two. The
archive records `one-metric-per-pass-group`; this is not optimal counter
scheduling and must not be mixed with ordinary timing. `--metrics <json>` uses
the existing 1–16 exact metric-name contract.

| Evidence | Protocol / scope |
| --- | --- |
| Existing in-frame ranges | `metallic-nvperf-v1` / `in-frame-command-ranges-on-graphics-queue` |
| Isolated correctness only | `metallic-work-control-replay-v1` / `isolated-correctness-only`, counterEligible=false |
| Isolated counter image | `metallic-nvperf-isolated-v1` / `isolated-dispatch-RHI-exclusive` |

The last scope means a single dispatch per target submission, with other RHI
submissions excluded within this process. It does not prove whole-system GPU
exclusivity, prevent native submissions by other software, or provide source-line
or PC attribution. SDK metric domain semantics and external activity still
apply. Physical bindings, cache state after restoration and diagnostic times
must not be presented as normal production timing or an optimization win.

## Failure and offline review

Creating `app/replay/Cancel.request` requests cancellation before the next
submission or after an accepted submission has retired. Failed submissions do
not advance the ledger. A real unknown completion, timeout or device loss marks
the process poisoned and exits instead of releasing in-flight allocations into
normal rendering. An outer process timeout handles driver cleanup that hangs.
Restore/byte mismatches fail the run; scratch data is never published.

`--fault <name> --runs 1` exercises the failure gates in a standalone GPU process
without counters. Names: `cancel-before-submit`, `cancel-after-submit`,
`submit-failure`, `device-error`, `timeout`, `restore-failure`, `state-leak`.
These are explicitly injected failures, NOT induced driver/device faults.
Timeout/cancellation-after-submit injection happens after safe GPU retirement.
Version 2 submit/device/wait injections enter the real typed-error handling and
poisoned-process paths; restore failure skips the second pixel-buffer reset so
the GPU readback detects carried-over output. State-leak injection changes the
comparison observation, not production GPU state. Version 1 failure archives
remain separately identifiable and verifiable. A passing negative test is `expected-failure-verified`, never capture
success. Real driver loss and genuinely hung GPU retirement remain separate
environmental failure cases.

`verify` uses current code, verifies the exact artifact inventory/hashes,
recomputes binary equality, rechecks the ordered submit/restore ledger, nested
binding indices, actual SPIR-V fingerprint, same-frame and across-process
WorkloadCase identity, and distinct profiler scope. It never executes archived
scripts. It does not re-decode CounterDataImage; `counterImageReevaluated=false`
remains explicit. Failed or truncated archives cannot be promoted to successful
evidence. Hashes are integrity checks, not provenance signatures.
