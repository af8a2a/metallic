# ZorahFull Nsight performance root cause - 2026-09-30

The main remaining Deferred slowdown is triggered by GPU texture-demand feedback written directly into a `HostReadback` storage buffer. With one frozen working set in one injected process, switching this buffer path off / host / device / host / off changes the inclusive Deferred shading GPU scope from 3.64 / 82.77 / 3.32 / 83.60 / 3.71 ms. The Device diagnostic retains the same sampling metadata and demand atomics. This identifies the expensive resource-access path; it does not establish Nsight's private driver/interceptor mechanism.

There are two distinct contributors to the original live capture-ready slowdown:

- The earlier same-build HVVM experiment identified a large streaming-traversal cost associated with Nsight's default host-visible VRAM demotion policy: 34.250 ms with demotion versus 4.416 ms with CPU hashing, after geometry settled. CPU hashing failed actual capture replay, so it remains rejected. See [the first investigation](ZorahFullNsightPerformance20260930.md).
- This investigation identifies texture feedback as the dominant remaining Deferred cost. These atomics are expensive in the normal process too, and the injected process magnifies their measured cost. Disabling supplementary path tracing does not remove texture feedback.

## Controlled workload

Evidence: `build/nsight-root-cause-20260930-02/`. Source HEAD: `d21e01b9729dd561a60e6a9c56b393d7d0d58124`, with the archived diagnostic patches. Windows / RTX 5070 Ti 16 GB / driver 616.92 / Nsight Graphics 2026.3.1 build 38722833, SDK 0.9.2, MSVC Release.

Output is fixed at **2560 x 1440**, DLSS Quality internal extent **1707 x 960**, fixed sample camera, temporal jitter, VSync, two frame slots, hidden window, validation disabled. Existing asset/shader caches are reused. GPU experiments run serially in fresh processes. No F11 capture or replay occurs during these measurements.

The scene warms for 60 seconds after readiness. GPU submissions/queries are drained, then geometry cut/CLAS/TLAS and texture publication/mip reveal are frozen. Sixteen unmeasured settle frames precede another drain and the measurement. Freezing intentionally removes streaming traversal/build work and normally suppresses GPU texture-demand writes; this is a diagnostic workload, not a new live performance baseline.

The strongest experiment uses five 20-second stages in the same process. Each transition drains prior GPU work. The first two seconds of each stage and both transition-adjacent frames are excluded from stable statistics. Camera, scene/stream generation, raster identity, texture residency, sampling-floor transitions and cumulative publication counters remain fixed across the five stages.

Frame ms is the start-to-start editor loop, including CPU wait/pacing, recording and present. GPU envelope/pass timings are inclusive RenderGraph timestamp intervals, excluding editor composite/present and independent texture submissions. Tool-inserted work within an interval can contribute to it. These are resource-path ablations, not pure shader-instruction measurements or end-to-end renderer speedups. Do not add nested scopes.

## Same-process feedback stages

`Host` means the original `HostReadback` binding 95, with nonzero source dimensions and `InterlockedMin` / `InterlockedAdd`. `Device` keeps those dimensions and atomics, but initializes a cached Device storage buffer from the host seed before shading. Frozen CPU demand is discarded in all stages, so this diagnostic deliberately performs no Device-to-host result copy. It cannot implement live texture streaming.

| Process | Stage | Stable frames | Deferred shading GPU mean / P95 ms | Frame mean ms |
|---|---|---:|---:|---:|
| Normal | Feedback off | 1301 | 0.913 / 1.238 | 13.824 |
| Normal | Host feedback | 585 | 12.021 / 12.900 | 30.679 |
| Normal | Device feedback | 1343 | 0.921 / 1.224 | 13.378 |
| Normal | Host repeat | 585 | 12.213 / 13.163 | 30.650 |
| Normal | Off repeat | 1353 | 0.907 / 1.213 | 13.277 |
| Nsight injected | Feedback off | 1178 | 3.637 / 4.186 | 15.271 |
| Nsight injected | Host feedback | 187 | 82.769 / 86.108 | 95.475 |
| Nsight injected | Device feedback | 1168 | 3.318 / 3.878 | 15.291 |
| Nsight injected | Host repeat | 182 | 83.602 / 85.934 | 98.223 |
| Nsight injected | Off repeat | 1165 | 3.712 / 4.251 | 15.323 |

In both processes, returning to Host restores the high cost and returning to off restores the low cost. Relative to off, Host adds about 11 ms in the normal process and 79-80 ms in the injected process. Their separately warmed texture working sets differ, so these differences do not establish an exact normal-to-injected amplification ratio.

Both processes retain 61,661 geometry pages, 3,758,094,848 geometry bytes and 1,515,217,664 CLAS bytes across all stages. Normal texture residency remains 316,633,600 bytes / 848 refined images; injected residency remains 218,690,048 bytes / 558 refined images. Injected pending texture work remains 8 images / 1,503,232 bytes: GPU drain is not CPU decode/publication drain, and pending work is frozen rather than declared complete. No per-image mip hash is available across processes; the same-process repeat experiment avoids that limitation.

GPU clocks in the slow injected Host stages are approximately 2914 MHz, versus approximately 2895 MHz in fast stages. Normal stages differ by roughly 1% in clock. DWM and Edge have background GPU activity of similar scale across stages. Windows process counters report errors in some runs and an invalid 2119% copy sample; raw telemetry is preserved, invalid samples are excluded from interpretation. Clock throttling and a between-process residency difference do not explain the repeated within-process Host/off response.

## Independent runs and SDK check

The first four 20-second frozen runs share one executable. Their separate-process aggregate texture residencies differ, so use them as supporting evidence rather than a replacement for the five-stage control.

| Mode | Frame mean ms | Deferred shading GPU mean ms |
|---|---:|---:|
| Normal, frozen Host feedback | 21.901 | 13.092 |
| Injected, frozen feedback off | 11.801 | 3.072 |
| Injected, frozen Host feedback | 83.210 | 74.509 |
| Injected, frozen Device feedback | 13.481 | 3.068 |
| Injected Host, `noVulkanCaptureReplayMemory=true` | 92.771 | 79.223 |

The last SDK setting is recorded both in the manifest policy and the actual settings log (`hvvm=0`, `noCaptureReplayMemory=1`). It stops Nsight's forced capture/replay memory allocation flags, but does not restore the fast regime. It is not a strict identical-mip cross-process A/B, so this does not prove zero effect from that setting. The SDK's definition does not promise a performance fix. [Official capture CLI](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)

All seven diagnostic runs export complete frame/GPU timing coverage. Frozen counters and raw first/last/min/max agree. Load failures, request overflows and BLAS error counters are zero, with no device-loss/validation error in workload logs. `allocationFailures=1` is retained from pre-freeze maintenance; it is not summed as a new failure every frame.

## Source-level cause and exclusions

The feedback buffer is created in `Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp` as `MemoryLocation::HostReadback`. Its 32-byte-per-texture entries contain source dimensions, mip state, minimum demanded mip and hit count. `Shaders/Features/PathTracing/ScenePathTrace.slang::loadMaterialTextureForMaterial()` reads this metadata, then performs two atomics for enabled texture demand. Many shaded pixels contend on the entries for the same material textures. The frozen Device test preserves those shader operations while changing the buffer access path.

Default ZorahFull Deferred has supplementary path tracing disabled. Its stream shader removes the corresponding RTAS bindings and has no reachable own ray queries. Its large geometry/material buffers are Device resources, rather than the large static HostUpload group/node/LOD tables used by streaming traversal. This excludes Deferred's supplementary ray traversal as the observed feedback-stage cause. Final SPIR-V capability/SASS equivalence was not independently proven.

Application-side CaptureSymbols retains Slang optimization; only ShaderDebug sets optimization NONE. The earlier symbols-only control stayed close to normal. Descriptor heaps report memory type 3 / flags 14 / system heap 1 in both normal and injected logs; this is application-visible accounting, not proof of physical residency after interception. NVIDIA supports descriptor-heap capture/replay from 2026.2; its documented GPU-written-heap limitation does not directly apply to Metallic's CPU descriptor writes. No public evidence establishes heap emulation or a specific shader-instrumentation bug as this slowdown's root cause. [Official support statement](https://developer.nvidia.com/nsight-graphics/getting-started/release-note-v2026.2)

The measured trigger is precise: GPU reads/atomics on the original HostReadback texture-feedback path become very expensive under capture injection. Which private Nsight/driver mechanism amplifies that path remains unresolved. HVVM policy, capture/replay allocation flags, host-memory tracking and invisible tool work must not be collapsed into an unmeasured single mechanism.

## Retained changes and validation

Temporary SDK policies, forced frozen demand, Device-feedback substitution and timed environment stages are removed from production source. Their exact patches/configurations/logs are retained under the evidence directory. Both final executables are rebuilt from restored runtime source and checked to contain no `METALLIC_NSIGHT_DIAG` strings. SDK-default capture behavior remains unchanged; the failed CPU-hash workaround remains absent.

The retained benchmark adds default-off `benchmarkFreezeStreaming`, validated `deferredOverrides`, freeze snapshots and counter ranges. These runs are marked diagnostic. The offline analyzer accepts intentionally absent frozen RTAS build scopes while requiring complete Deferred shading GPU coverage. The actual failed frozen report now passes; a raw-copy test with one Deferred GPU query removed is rejected. Live-run RTAS coverage requirements remain in place.

Build: `cmake --build build-release --target Metallic MetallicGPUDrivenSample -j 8` in the existing x64 MSVC developer environment. Logs: `BuildDiagnostic.log`, `BuildPhases.log`, `BuildRestored.log`. Final ordinary-path verification in `final-normal-validation/` passes: 10-second warmup, 10-second measurement, 326 frames, complete live RTAS/shading timing coverage, actual injection false, shader mode disabled, freeze false and diagnostic false. Frame mean / P95 is 30.731 / 34.265 ms under the current background workload. This short validation is not substituted for the archived baseline and does not assert a 30-FPS acceptance result.

The production repair direction is Device feedback plus asynchronous readback into HostReadback, retaining initialization barriers, GPU-to-copy/host visibility, per-frame completion and generation lifetime, cancellation and demand-consumption semantics. The frozen no-copy diagnostic establishes the performance direction but does not validate live texture refinement or capture replay correctness for such a repair. No production feedback migration is introduced by this investigation.

Raw evidence includes each run's `Manifest.json`, `Config.json`, `Policy.json`, `Capture.json`, `Frames.jsonl`, `Uploads.jsonl`, stdout/stderr, GPU sensors and process counters. `PhaseEvidence.md` is the concise phase report; `CompactEvidence.json` retains the detailed checks. Recreating the temporary probe requires the archived `TemporaryRuntime.patch` and `FeedbackPhases.patch` against the retained frozen benchmark, followed by a rebuild; `RunCase.ps1 -Phases` alone is not a production feature.
