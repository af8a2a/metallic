# CPU-job evidence plots

`Tools/CPUJobs/PlotProductionCPUJobs.py` reads evidence and creates PNG figures,
an HTML report, a Markdown report and `PlotManifest.json`. It launches no
renderer, cooker, compiler or benchmark. Use Python with matplotlib; image
inspection also uses numpy, already required by matplotlib. The output must be
a new empty directory. Archived evidence and its manifests are never rewritten.
Each new report saves the exact as-run `PlotProductionCPUJobs.py` source beside
its manifest; `plotterSHA256` hashes that snapshot, not a later workspace edit.

```powershell
python -B Tools/CPUJobs/PlotProductionCPUJobs.py `
  --historical Documentation/Measurements/CPUJobs20261106 `
  --production build-issue73-production/performance `
  --production-cook build-issue73-production/cold-cook `
  --output build-issue73-production/plots-final `
  --title 'SuperSponza CPU jobs: matched production and archived evidence'
```

Replace the input directories with the actual evidence roots. Each production
input contains `Metadata.json`, `Raw.jsonl`, and its relative run-artifact paths.
`--historical`, `--production` and `--production-cook` are independently optional.
The historical overview uses the archived raw data, rather than inventing
process-level samples from percentile summaries.

No frame/RAM/VRAM budget is assumed. Optional `--frame-budget-ms`,
`--cpu-memory-budget-mib` and `--vram-budget-mib` draw explicitly supplied
reference lines. A texture-subsystem budget is not a total application VRAM
budget; whole-device capacity is also not an application budget.

## Definitions retained by the plots

- Production frame quantiles belong to individual processes and their measured
  frame windows. Correlated frames are not pooled as independent A/B trials.
  Block, worker limit, process identity, sample counts and executable hashes are
  shown. Validation, performance-off and instrumented diagnostic data have
  separate contexts; validation timings are not performance results.
- Time-series pages show every recorded frame without smoothing. Explicit
  `measuredFrameRange` supplies window shading; absent windows are not guessed.
  All pages use common y limits, including recorded outliers.
- Loading scopes show Editor initialization (when explicitly logged), capture
  loading after initialization through scene ready, and the whole OS process
  lifetime as independent markers in seconds. They are not stacked or subtracted
  to invent warmup duration. SceneLoad owner steps retain their logged durations;
  separate complete PNG decode/mip/upload intervals remain NA when unavailable.
- Inclusive CPU scopes are separate occurrence distributions. Nested/parallel
  scopes are not added. Scope occurrence counts can differ from frame counts.
- Coordinator stacks require the explicit non-overlapping interval contract and
  are checked against each measured frame envelope. Means use the same complete
  frames; sums of P50s are never used as a breakdown.
- Diagnostic TaskGraphs use actual nodes, edges and steady-clock intervals.
  The latest-predecessor completion chain separates node wall, observed
  ready-to-start gaps and completion epilogue. A longest dependency-node-wall
  path is computed separately. The largest eligible graph in each diagnostic
  process is selected explicitly, not presented as a typical/performance-off
  graph. Node wall includes preemption; gaps are not measured worker sleep.
  TaskSystem's recorded thread identity is a `std::thread::id` hash, not an OS
  affinity or Windows TID contract.
- OS commit/working-set high-waters are observed through the last successful
  query. Current-value sampled peaks, sampled thread peaks and average process
  CPU cores have separate definitions. Average CPU cores are not peak CPU use.
- RHI event high-waters cover tracked allocations and exclude untracked
  SDK/driver allocations. Heap allocation/block/usage samples are distinct
  accounting views. Host+device totals and per-domain peaks are not a VRAM-only
  partition; independently timed domain peaks are not summed.
- Cold-cook process wall and cooker elapsed have different boundaries. The
  separate validation subprocess is excluded. New output paths do not imply a
  flushed source/OS cache. Aligned device payload sizes are not measured VRAM.
- Missing metrics remain absent/unavailable. No artificial zeros, interpolation
  or synthetic frame/resource samples are created.

## Render output inspection

The image helper accepts the app's `Capture.json` plus its `image.file` raw
output. It checks the extent, format, byte count and all RGBA values for
finiteness. It emits native-resolution RGB previews and raw numeric statistics.
Supported formats are tightly packed `RGBA8Unorm` and little-endian
`RGBA16Sfloat`. Alpha is inspected but ignored in the visual RGB preview.

```powershell
python -B Tools/CPUJobs/PlotProductionCPUJobs.py `
  --image-capture build-issue73-production/validation-shared/Capture.json `
  --output build-issue73-production/image-preview
```

Auto preview follows the current `DisplayColorEncoding` contract: sRGB values
are clipped/displayed directly; scRGB or DisplayLinearRec709 receive a fixed
linear-to-sRGB transfer after clipping to raw `[0,1]`. SceneLinear/ExposedLinear
primaries are not guessed. There is no adaptive exposure, tone mapping or
replacement of NaN/Inf with zero. Original HDR data remains in the raw binary.

Comparison requires an explicit request and a caller-supplied description of
camera, temporal history, frame-index/seed, exposure and priming controls:

```powershell
python -B Tools/CPUJobs/PlotProductionCPUJobs.py `
  --image-capture build-issue73-production/validation-legacy/Capture.json `
  --image-capture build-issue73-production/validation-shared/Capture.json `
  --compare-images `
  --comparison-context 'Replace with the actual fixed-camera and temporal/priming protocol' `
  --output build-issue73-production/image-comparison
```

Scene, camera, graph, extents, format and encoding must match. The helper reports
raw-coordinate MAE/RMSE/max difference and a visual heatmap; it applies no
automatic bitwise/quality threshold and cannot establish temporal equivalence
on its own. `RenderImageComparison.json` retains the exact caller context and
source hashes. Nonfinite output makes image inspection fail after preserving
the diagnostic statistics/report.

## Restoring archived run artifacts

The repository keeps small metadata, compact per-run summaries, hashes and
plots. Full original Raw/Summary JSON and per-run archives are in the versioned
[release evidence](Storage.md); `CompactSummary.json` is not a plotter input.
Restore the original relative `run-*` layout before plotting:

```powershell
gh release download pr-74-cpu-jobs-evidence-20261107 --repo af8a2a/metallic --pattern metallic-cpu-jobs-production-20261107.zip --dir build-issue73-evidence-download
python -B Documentation/Measurements/CPUJobsProduction20261107/RestoreEvidence.py --archive build-issue73-evidence-download/metallic-cpu-jobs-production-20261107.zip --output build-issue73-evidence-restored --bundle FrameReduced
python -B Documentation/Measurements/CPUJobsProduction20261107/FinalPlots/PlotProductionCPUJobs.py --production build-issue73-evidence-restored/Data/FrameReduced `
  --output build-issue73-production/restored-plots
```

The original inner archives use ZIP LZMA, supported by Python's standard library;
PowerShell `Expand-Archive` does not handle this compression. The restore helper
checks the release and original file hashes. The plotter reads ordinary restored files; it has no special archive/gzip
execution path. The runner's artifact hashes and archive manifest provide
integrity checks. `PlotManifest.json` records the actual input-file SHA256s used
and the plotting script SHA256; inputs are rechecked after generation.
