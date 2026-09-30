# ZorahFull Nsight capture-ready performance - 2026-09-30

`--nsight-capture` reproduces a substantial live-performance drop before F11 is pressed. The optimized shader-symbol control remains close to normal rendering; Nsight injection and capture tracking are the distinguishing factor in these measurements. The CPU-hash experiment partially reduces the drop, but actual replay of its first capture fails with GPU device loss. The original SDK demotion control exports three captures and its first capture passes three-loop replay. Metallic therefore retains the original SDK capture path and removes the experimental HVVM options. The live injection overhead remains; this investigation does not establish a renderer performance regression or a complete performance fix.

## Measurement conditions

One serial fresh-process run per mode on 2026-09-30: Windows, RTX 5070 Ti (16 GB), driver 616.92, Nsight Graphics 2026.3.1 / build 38722833 / SDK 0.9.2, MSVC Release. The scene is `gpu-driven-zorah-full`, using the existing cooked stream asset and normal runtime budgets. Output is fixed at 2560 x 1440; DLSS Quality renders at 1707 x 960. Camera and route are fixed, temporal jitter is enabled, VSync is enabled, there are two frame slots, and the hidden editor runs without validation or detailed per-work-item telemetry. This camera is the controlled sample view, not a reconstruction of the supplied screenshot's camera.

Each process completes scene initialization, then the stated 10- or 40-second warmup, then a 20-second measurement. Existing shader/asset caches are reused rather than cleared; this is not a cold-cache test. No F11 frame capture is requested during timing. Injected rows measure the live capture-ready application before a capture, not capture-file generation or replay.

Frame ms is the start-to-start editor loop interval. GPU envelope and pass times are inclusive RenderGraph timestamp scopes, excluding editor composite/present and independent texture submissions. CPU scopes measure host work recording those scopes and are not the elapsed GPU work. Do not sum nested/concurrent scopes or call pass changes end-to-end GPU speedups.

## Results

The `injected` row records SDK-default HVVM demotion. The `demote-*` rows also use demotion; the `cpu-hash*` and `eager-settled` rows use CPU hashing. The editor retains demotion as its default; these mode labels and timings describe the recorded settings.

| Mode | Warmup s | Frames | Frame mean ms | Frame P95 ms | GPU envelope mean ms | Deferred shading mean ms | CLAS build mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 10 | 895 | 22.367 | 29.335 | 21.514 | 11.373 | 0.000113 |
| symbols | 10 | 880 | 22.737 | 30.294 | 21.919 | 11.839 | 0.000119 |
| injected | 10 | 135 | 148.225 | 199.356 | 148.357 | 78.471 | 26.476517 |
| demote-control | 10 | 147 | 136.274 | 181.015 | 136.231 | 70.570 | 25.600235 |
| cpu-hash | 10 | 200 | 100.030 | 130.029 | 93.230 | 67.112 | 14.487876 |
| cpu-hash-settled | 40 | 230 | 87.162 | 131.752 | 85.700 | 73.135 | 0.000124 |
| eager-settled | 40 | 227 | 88.181 | 104.098 | 86.629 | 73.948 | 0.000126 |
| cpu-hash-uncached-control | 40 | 207 | 97.096 | 119.486 | 91.488 | 77.963 | 0.000125 |
| cpu-hash-uncached-preserved | 40 | 201 | 99.791 | 135.861 | 92.022 | 78.468 | 0.000125 |
| demote-settled | 40 | 156 | 128.454 | 144.311 | 128.241 | 84.516 | 0.000124 |

Executable SHA-256 groups:

- `4540ADBC6AD7665B5F3DAA275042A1EEBD817DA04110F80ED0621BC6AD7B2C57`: normal, symbols, injected.
- `E2F8D26F0B82F58D6603A0083F5316668EB638E1AB958DAE93506DFD4681ADDB`: demote-control, cpu-hash.
- `1F3CF51183F2365F614F3F0C6E68FAED9AC7EB1DF026EF8A2415794BD5BA015E`: cpu-hash-settled, eager-settled.
- `4D307FF5F547A492DFE39F005F8F67C35F8A93F043F14009BEC434A0B87405F7`: cpu-hash-uncached-control, cpu-hash-uncached-preserved, demote-settled.

The symbols-only control stays near ordinary rendering: frame means 22.367 vs 22.737 ms, GPU envelopes 21.514 vs 21.919 ms. This rules out an unoptimized Slang shader mode as the explanation for the measured capture slowdown. `CaptureSymbols` adds standard debug information and direct SPIR-V emission; only the separate `ShaderDebug` branch requests `SLANG_OPTIMIZATION_LEVEL_NONE`.

Within the same 10-second-warmup executable, changing SDK HVVM handling from default demotion to CPU hashing reduces frame mean 136.274 to 100.030 ms and GPU envelope 136.231 to 93.230 ms. This is partial mitigation, with different live residency progress. It does not restore normal rendering performance. After 40 seconds of warmup with CPU hashing, stable geometry/CLAS still leaves an 87.162 ms frame mean and 73.135 ms Deferred shading GPU mean.

The same-build eager-data-collection pair shows no mean improvement: frame 87.162 to 88.181 ms and envelope 85.700 to 86.629 ms. The same-build uncached-memory-preservation pair also shows no improvement: frame 97.096 to 99.791 ms, P95 119.486 to 135.861 ms, envelope 91.488 to 92.022 ms, shading 77.963 to 78.468 ms. These single runs do not quantify a repeatable regression, but they provide no evidence to retain either experiment as a remedy. All experimental HVVM, eager-collection and uncached-memory options were removed; the table preserves their measured results as investigation evidence.

## Residency and comparison limits

`injected`, `demote-control`, and `cpu-hash` still change geometry residency during measurement. Their respective resident-page ranges are 42511-59337, 43365-61286, and 46375-61660; missing-CLAS maxima are 1364, 1176, and 1117. Those live timings include a changed streaming/build workload and are not a frozen-state shader-cost comparison.

All five 40-second-warmup runs keep geometry residency constant. For every measured frame, `clasBuiltClusters`, `clasMovedClusters`, `blasBuildCount`, `blasMissingClasInstances`, geometry `ioActive`, `ioQueued`, and `uploads` are zero. CLAS build GPU means are about 0.000125 ms. The remaining Deferred GPU cost therefore occurs without measured CLAS rebuilds.

The strongest persistent HVVM comparison uses `demote-settled` and `cpu-hash-uncached-control`: the same executable SHA-256, 40-second warmup, 20-second measurement, and default uncached-memory policy. Both runs keep 61661 resident pages and have zero measured missing-CLAS/build/move/BLAS-build counts. CPU hashing reduces frame mean 128.454 to 97.096 ms and frame P95 144.311 to 119.486 ms. GPU envelope decreases 128.241 to 91.488 ms; VBuffer 41.214 to 10.972 ms; Stream traversal 34.250 to 4.416 ms; Deferred 84.641 to 78.087 ms; Deferred shading 84.516 to 77.963 ms. CLAS GPU time stays near 0.000125 ms in both. This shows mitigation after geometry settles, rather than only faster loading. Texture refinement still differs: demote has 342-537 refined images and 168882688-212136448 resident texture bytes, versus CPU-hash 506-720 and 201454080-266301952 bytes. It is not a fully frozen texture-state comparison.

Texture refinement continues in every 40-second run, as the refined-image and resident-byte ranges show below. Geometry/CLAS stability does not establish fully frozen resources or identical texture residency. There is one run per mode and no repeatability interval. Configuration/camera/graph/shader-source identity matches within each executable hash group; cross-group timings are descriptive only. The source hash does not assert byte-identical compiled shaders across symbol modes.

`loadFailures`, `requestOverflows`, and `blasOverflowCount` are zero in all ten measured runs. The optional-allocation backpressure counter `allocationFailures` reaches 1 in several runs; the report does not describe every diagnostic counter as zero.

| Mode | Resident pages min-max | Missing-CLAS instances max | Texture refined images min-max | Texture resident bytes min-max |
|---|---:|---:|---:|---:|
| normal | 61661-61661 | 0 | 505-831 | 206041600-312111616 |
| symbols | 61661-61661 | 0 | 520-838 | 209187328-313094656 |
| injected | 42511-59337 | 1364 | 73-279 | 128365056-156103168 |
| demote-control | 43365-61286 | 1176 | 78-300 | 129479168-160363008 |
| cpu-hash | 46375-61660 | 1117 | 112-397 | 135213568-181137920 |
| cpu-hash-settled | 61662-61662 | 0 | 543-758 | 212660736-279933440 |
| eager-settled | 61661-61661 | 0 | 545-756 | 214954496-279245312 |
| cpu-hash-uncached-control | 61661-61661 | 0 | 506-720 | 201454080-266301952 |
| cpu-hash-uncached-preserved | 61660-61660 | 0 | 492-704 | 200208896-262861312 |
| demote-settled | 61661-61661 | 0 | 342-537 | 168882688-212136448 |

## SDK behavior and retained configuration

Nsight injection is active from graphics-context startup. NVIDIA documents nonzero overhead even before a capture is triggered; the normal-case description of small overhead does not bound this particular ZorahFull workload. [Nsight Graphics SDK guide](https://docs.nvidia.com/nsight-graphics/UserGuide/sdk.html)

The capture CLI documents HVVM demotion to system memory as the default, and CPU hashing as a way to preserve HVVM while tracking CPU updates. It separately documents converting uncached write-combined memory to cached memory and the potential application GPU cost. These are capture-tool memory policies; the observations here do not prove that every affected resource was physically demoted. [Graphics Capture CLI documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)

The editor and sample executables retain SDK-default system-memory demotion when they inject Graphics Capture. Experimental HVVM CPU hashing is removed from the runtime and benchmark launcher because the tested capture failed actual replay, despite successful export. Eager resource collection and uncached-memory preservation are also removed. External `ngfx` launchers use their own capture settings.

Launch the main editor's supported capture path, then load ZorahFull from the sample UI:

```powershell
& ./build-release/Source/Metallic.exe --nsight-capture
```

The main editor does not accept `--sample`. For direct ZorahFull startup, use the GPU sample wrapper's `--zorah-full` argument and request SDK injection through the environment; this wrapper does not accept `--nsight-capture`:

```powershell
$env:METALLIC_NSIGHT_GRAPHICS_CAPTURE = '1'
& ./build-release/Source/MetallicGPUDrivenSample.exe --zorah-full
Remove-Item Env:METALLIC_NSIGHT_GRAPHICS_CAPTURE
```

The benchmark launcher retains `-NsightCapture` and `-ShaderCaptureSymbols`. The runtime report records actual `graphicsCaptureInjected` and `shaderDebugMode`; analysis labels capture/debug instrumentation as diagnostic runs so they cannot silently replace ordinary performance baselines. Example measuring the original capture-ready path using a fixed-camera configuration:

```powershell
./Tools/RunZorahFullRoam.ps1 -Executable ./build-release/Source/MetallicGPUDrivenSample.exe -OutputRoot ./build/nsight-check -RouteConfig ./build/nsight-performance-20260930-01/Fixed1440.json -Runs 1 -WarmupSeconds 40 -DurationSeconds 20 -NsightCapture
```

An external driver-instrumentation comparison did not complete: the driver-on launcher failed its connection timeout and produced no benchmark `Capture.json`. The driver setting has therefore not been runtime-verified, and no result is inferred for driver-off.

## Capture and replay validation

The CPU-hash smoke run completed three SDK captures and exited with code 0 after 227.724 seconds. `capture-smoke-cpu-hash-02/Result.json` identifies the same executable hash `4D307FF5F547A492DFE39F005F8F67C35F8A93F043F14009BEC434A0B87405F7` as the final timing group. Its log confirms `HVVM mode=cpu-hash`, uncached-memory preservation disabled, and scene readiness at frame 90 with 65398/65398 readiness entries.

| Capture filename | Smoke viewport | Bytes |
|---|---|---:|
| `MetallicGPUDrivenSample_2026_09_30_16_39_25.ngfx-capture` | 1797 x 660 | 11287552968 |
| `MetallicGPUDrivenSample_2026_09_30_16_40_34.ngfx-capture` | 1957 x 750 | 11922556920 |
| `MetallicGPUDrivenSample_2026_09_30_16_41_30.ngfx-capture` | 1797 x 660 | 12470852376 |

Fresh files are under `Captures/NsightGraphics/`; raw logs/results are under `build/nsight-performance-20260930-01/capture-smoke-cpu-hash-02/`. This capture-export smoke run resizes the editor viewport and is separate from the fixed 2560 x 1440 performance runs.

The first capture's embedded PNG was inspected and contains the ZorahFull image. `MetadataErrors.log` reports no embedded messages with severity >= 2; neither observation establishes replay correctness. Actual three-loop replay exits with code 1 after 76.040 seconds, with `FATAL EXECUTION ERROR: GPU device lost` in `capture-replay-cpu-hash-02/stdout.log`. `Result.json` records the failed exit and the requested loop count, not three completed loops.

The SDK-default demotion smoke control also exports three captures, exits with code 0, and takes 128.883 seconds. Its `Result.json` records executable SHA-256 `3F812E7BAC06344F0E97973ACBDE6B27C5476A2E1EED784AA5ACD63A509C10AC`; its stdout confirms `HVVM mode=default-demote` and the same 65398/65398 readiness entries at frame 90.

| Capture filename | Smoke viewport | Bytes |
|---|---|---:|
| `MetallicGPUDrivenSample_2026_09_30_16_56_31.ngfx-capture` | 1797 x 660 | 11284812120 |
| `MetallicGPUDrivenSample_2026_09_30_16_57_17.ngfx-capture` | 1957 x 750 | 11948945208 |
| `MetallicGPUDrivenSample_2026_09_30_16_57_44.ngfx-capture` | 1797 x 660 | 12495955320 |

The demotion control's first capture passes actual three-loop replay: exit code 0 after 74.629 seconds and `Replay completed successfully` in `capture-replay-demote/stdout.log`. Raw smoke evidence is under `capture-smoke-demote/`, and replay evidence under `capture-replay-demote/`, within the same local investigation directory. The other two demotion exports were not replayed in this check. These capture runs use different executable hashes; they do not isolate the internal cause of CPU-hash device loss.

Both replay logs warn that capture version 2026.3.1 is newer than replayer version 2026.3, although both report build 38722833. Both report the NGX SuperSampling artifact warning. The successful demotion replay has these warnings too, so they are recorded without attributing device loss to them. The CPU-hash embedded PNG is capture-side evidence, not newly rendered replay output; no replay-output pixel comparison was performed.

The original SDK path has successful export and replay evidence; CPU-hash replay has failed and is not retained as a fix. Live timing JSON files named `Capture.json` are benchmark reports, not `.ngfx-capture` files. The ten timing runs did not generate frame captures or validate replay.

## Final ordinary GPU sample validation

Two ordinary final-build `MetallicGPUDrivenSample.exe` runs use the same executable SHA-256 `DBD0BB6758CB1DD854AD51AE2142C31476FB0FA2256A7FB5E11150AA4D105E84`, 10-second warmup and 20-second fixed-camera measurement. They are independent validation records, not Main editor measurements or a newly published baseline; the archived baseline and the initial same-build normal-versus-injected comparison remain unchanged. Both have actual injection false, shader mode disabled, diagnostic false and validation false.

| Measurement | Initial normal | normal-sample-final | normal-sample-final-02 |
|---|---:|---:|---:|
| Frames | 895 | 648 | 740 |
| Frame mean ms | 22.367 | 30.890 | 27.044 |
| Frame P95 ms | 29.335 | 35.781 | 33.904 |
| CPU frame mean ms | 21.824 | 30.251 | 26.367 |
| CPU Reflex pacing mean ms | 9.068 | 17.805 | 10.375 |
| CPU slot-wait-before-input mean ms | 2.440 | 0.254 | 2.439 |
| GPU envelope mean ms | 21.514 | 25.669 | 26.368 |
| VBuffer GPU mean ms | 8.175 | 9.186 | 9.317 |
| Deferred shading GPU mean ms | 11.373 | 13.898 | 14.470 |

Camera, graph, runtime/requested configuration, 2560 x 1440 output / 1707 x 960 render, VSync, frame-slot count, asset path/size/mtime, shader source hash and flags match initial `normal`. All 79 loaded shader-cache keys match across the three runs, and the recorded raster SPIR-V FNV-1a-64 is the same `12715239286116297052`; cache keys are not an independent byte hash of every compiled shader. Geometry residency is fixed at 61661, 61661 and 61662, with zero measured CLAS builds/moves/missing instances. Texture refinement differs: 505-831, 395-816 and 446-822 refined images.

Between the two final runs, frame mean falls 3.846 ms while GPU envelope rises 0.699 ms. The first final run's extra wall-clock time is therefore chiefly visible in CPU/Reflex waiting; this is not a measured GPU speedup. Both final GPU envelopes remain above the earlier normal value, and their different executable hash and background GPU activity prevent attributing that cross-build difference to a source optimization regression. Nested scopes cannot be summed to allocate the frame increase.

Associated whole-GPU sensor clocks are 2900.3, 2902.9 and 2899.1 MHz; temperatures are 61.75, 61.56 and 61.95 C, without observed clock loss. GPU utilization is 94.2, 84.2 and 94.8 percent. Process counters in the approximate Reflex-associated windows show msedge 3D means 0.968, 7.652 and 7.066 percent (14, 17 and 18 recorded samples); the final runs also have video-decode means 6.346 and 5.832 percent. DWM 3D means are 0.671, 3.338 and 3.880 percent. Each run records one Metallic workload process. This visible Edge/video/compositor activity is a comparison confound, not a quantified causal explanation of every millisecond.

`FinalNormalComparison.json` preserves identity checks, separate CPU/GPU distributions, shader-cache key sets, process/sensor records, streaming ranges and evidence hashes. Process percentages are means of recorded counters, not sums or zero-filled missing values; timestamp windows are approximate because driver reports are delayed. The failed `normal-main-final` attempt generated no benchmark report and is excluded.

## Evidence and source points

Raw local evidence is preserved under `build/nsight-performance-20260930-01/`, including each mode's manifest, frame samples, GPU/process sensor samples, and stdout/stderr. `Summary.json/md` record configuration comparisons, executable groups, raw evidence hashes, inclusive scope distributions and driver heap/domain snapshots. `TimingEvidence.md` also lists CPU recording scopes separately from GPU scopes. Generated evidence and captures remain outside source control.

For ordinary performance measurements, set `METALLIC_NSIGHT_GRAPHICS_CAPTURE=0` and `METALLIC_SHADER_CAPTURE_SYMBOLS=0`, and omit `--nsight-capture` and `--nsight-shader-debug`. The ordinary controls and final validation above are GPU sample-executable measurements, not a main-editor baseline. The `normal-main-final/` attempt started the main editor's default A Beautiful Game path-tracing sample and produced no benchmark `Capture.json`; its partial logs are preserved, and it is excluded from the timing data.

Relevant source: `Source/Editor/EditorApplication.cpp` selects shader mode and initializes capture; `Source/Runtime/Render/Core/SlangCompiler.cpp` distinguishes optimized capture symbols from unoptimized shader debugging; `Source/Runtime/Render/Profiling/NsightGraphicsCapture.{h,cpp}` retains the original SDK configuration; `Source/Editor/EditorFullRoamBenchmark.cpp` records actual injection/debug state. `Tools/RunZorahFullRoam.ps1` controls bounded measurement runs, and `Tools/AnalyzeZorahFullRoam.py` retains the diagnostic conditions.
