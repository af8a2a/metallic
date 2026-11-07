# Final plot populations and visual inspection

No renderer, cooker, compiler or test was launched to generate these reports. The final plot run reads only completed evidence. Every listed report retains its exact as-run plotting source, source SHA256, input SHA256s and figure SHA256s.

Full original records now live in the [versioned evidence release](Storage.md),
with byte-preserving restore instructions and per-file hashes. Source control
keeps compact per-run summaries and provenance. This storage migration does
not change the as-run reports, their snapshots or any numerical conclusion.

| Report | Population | Figures |
| --- | --- | ---: |
| [FinalPlots](FinalPlots/Report.html) | Ordinary diagnostics-off frames: one complete W1 ABBA block, two processes per backend, 600 frames per process; eight prewarm processes excluded. Separately, all 48 cold-cook processes, three blocks at 1/2/4/8 workers. | 11 |
| [SoakPlots](SoakPlots/Report.html) | Instrumented diagnostic residency soak: four W8 processes × 6000 frames. No ordinary FPS conclusion. | 11 |
| [HistoricalPlotsFinal](HistoricalPlotsFinal/Report.html) | Archived scheduler, range, SIMD and generated meshlet raw measurements. Scope labels and original sample/quantile definitions retained. | 5 |
| [ValidationPlots](ValidationPlots/Report.html) | Existing core-validation four-process ABBA, 120 captured frames each. Existing plots were preserved, not regenerated. | 12 |

The original ordinary frame plan was stopped at the user's request to reduce task duration. The reduced root preserves the original record bytes and contains the first completed W1 ABBA block; it does not claim completion of the original 56-process, three-block plan. The eight excluded prewarm processes cover both candidates at 1/2/4/8 workers. A later one-sided W2 record and an interrupted counterpart remain in the original evidence and are not paired or included in these plots. This was an external budget stop, not a GPU failure.

## Ordinary frame observations

| Process | Candidate | CPU frame P50 / P95 / P99 (ms) | GPU envelope P50 / P95 / P99 (ms) |
| --- | --- | --- | --- |
| 8 | Legacy | 5.5279 / 6.1239 / 6.4502 | 3.854784 / 4.188256 / 4.485984 |
| 9 | Shared | 5.7130 / 6.6274 / 7.0597 | 3.900128 / 4.649248 / 4.849472 |
| 10 | Shared | 5.8290 / 7.1159 / 8.0692 | 4.108256 / 4.496320 / 4.861184 |
| 11 | Legacy | 6.5399 / 59.1116 / 71.1660 | 4.360640 / 45.051968 / 61.944448 |

The two-process means of CPU P50 are 6.0339 ms legacy and 5.7710 ms shared (legacy/shared = 1.04556). The final legacy process has pronounced CPU and GPU tails, visible without trimming in both the percentile and complete time-series plots. One block and two processes per side do not establish a stable causal scheduler gain or a general FPS improvement. The recorded tail remains evidence; no outcome-dependent retry or exclusion was added.

All four ordinary processes have sampled peak thread count 48 and RHI tracked device-local event high-water 7278.872 MiB. Observed OS commit high-waters span 8385.793–8399.367 MiB. These measurements cover the process lifecycle and have different ownership from VRAM or worker counts.

Initialization, post-initialization scene-ready loading and whole-process wall are independent markers in [ProductionLoadingScopes.png](FinalPlots/ProductionLoadingScopes.png). Their boundaries differ; no subtraction is used to infer warmup. The post-initialization loading envelopes are 32.9522, 31.9976, 31.6912 and 30.7445 seconds in process order. These include compilation and uploads and retain shared caches; they are not cold-cache loading results. Separate complete PNG decode/mipmap/upload boundaries are unavailable.

## Cold-cook observations

The complete cold-cook population is 48 fresh-output processes, six per candidate and worker count, with separate all-page and source-attribute validation subprocesses excluded from timing. The three block ratios of legacy/shared whole-process wall are:

| Workers | Block 0 | Block 1 | Block 2 |
| --- | ---: | ---: | ---: |
| 1 | 1.00269 | 0.87264 | 0.99470 |
| 2 | 1.01930 | 1.01140 | 1.05738 |
| 4 | 0.97857 | 0.98252 | 0.97303 |
| 8 | 0.93647 | 1.00412 | 0.99653 |

Ratios above one favor shared. Results vary with worker count and block; this is not a uniform production speedup. Cooker-reported elapsed and full process lifetime remain separate series. Output payload byte counts are not reported as VRAM.

## Visual QA

Opened the final ordinary frame quantile/time-series, cook process, loading scope, resource peak, coordinator path, historical SIMD/packing, and diagnostic DAG chain/timeline PNGs. Axis units, identities, process/sample counts, captions and outliers are visible without clipping. The diagnostic DAG timeline zooms to recorded node intervals; the separate chain plot retains the submission-to-completion wall and observed readiness gaps. The selected graph is explicitly the largest observed graph, not a typical frame or a performance-off sample.

Opened all four validation render previews and representative cross-backend/same-backend difference maps. Geometry and textured architectural details align visually, without obvious missing blocks or error colors. Difference maps predominantly show sparse small changes; their individual observed maxima set the color scales and are not acceptance thresholds. The fixed-view, finite-output and numerical checks are diagnostic and do not establish every view or statistical temporal equivalence. The separate [SoakAudit](SoakAudit/Report.md) retains the longer-run finite/residency checks and same-backend comparison reference.

The earlier `HistoricalPlots` and `PilotShared2` directories are draft/pilot artifacts and are excluded from the final report population. Their old manifests were not rewritten.
