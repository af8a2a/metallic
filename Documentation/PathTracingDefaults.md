# Path tracing defaults

The interactive PathTracingSample graphs use one camera ray per pixel per frame
and query SHaRC at the first indirect surface. The cache estimates the remaining
multi-bounce radiance using independently traced training paths.

| Property | Default | Meaning |
|---|---:|---|
| `samples` | 1 | Camera paths per pixel; shadow and training rays are additional work |
| `cacheMode` | `sharc` | Sparse update, resolve, then full-resolution query |
| `sharc.queryMinDepth` | 1 | First eligible query after one indirect ray; camera surface is depth 0 |
| `maxDepth` | 12 | Hard surface-iteration limit, including cache-miss and specular fallback |
| `sharc.updateMaxDepth` | 12 | Independent multi-bounce training limit |

This is a target of one exact indirect bounce followed by cache reuse, not an
unconditional one-bounce cutoff. A complete eligible cache hit terminates the
path. A partial hit replaces only the cached BSDF components. Missing components,
immature cache entries, small footprints, delta paths, transparent materials and
unsupported layered closures retain the existing transport, up to `maxDepth`.
No new cache eligibility or confidence thresholds are relaxed by this preset.

Do not set `maxDepth=1` to request one indirect bounce: that evaluates only the
camera surface. `maxDepth=2` evaluates the first indirect surface but truncates
uncached tails. The default keeps those tails for cold-cache and glass quality.
Russian roulette can still end deep paths before the hard limit. SHaRC training
uses its own depth limit for both the standard and OpenPBR integrators.

The UI labels `maxDepth` as **Path Depth Limit (Including Fallback)** and `samples`
as **Camera Rays Per Pixel**. Changing the first query depth invalidates image
accumulation. Existing explicit graph settings remain authoritative.

`Pipelines/default.metallic_graph.json`, the four ABeautifulGame path-tracing
samples (plain, SR, RR, NR), and the dedicated meet_mat SHaRC sample explicitly
store the interactive preset. LookDev/material reference graphs, the legacy
meet_mat validation sample and RTXCR showcase explicitly disable caching and keep
their prior sample/depth settings. The query-implementation benchmark likewise
pins its original 2-sample, depth-12, cache-off workload.

Realtime deferred lighting keeps its own settings. Standard-material denoiser
guides retain their existing cache-disabled fallback; cached guides require
OpenPBR. The preset does not alter denoiser, normal/TBN, alpha coverage, nested
IOR/absorption or guide encoding contracts.

## Validation

`RHIRendering.openpbr_sharc_default_budget` compares implicit defaults against an
explicit 1-rpp/depth-12/cache-off oracle. It checks exact cache-miss equivalence,
actual cache use, radiance error, seven exact guides, moving-camera history, and
cache invalidation after lighting/grid changes. The runtime-settings test checks
the UI defaults. Two additional controls cap rendering at one indirect surface,
with caching off/on and independent deep training, to demonstrate that the cache
itself supplies the multi-bounce tail without relying on deep render fallback.
GPU correctness timings include validation/readback and must not
be treated as production performance measurements.

## User-view performance baseline

`Tools/Perf/PathTraceBaseline.ABeautifulGame.json` registers the user-supplied
2026-10-11 view at **1920 x 1080** internal rendering resolution, perspective
45 degrees, eye `(-0.447789, 0.206681, 0.134049)`, center
`(1.574037, -0.888370, -0.601834)`, up `(0, 1, 0)`, and clip range `0.001..100`.
The camera uses the six decimals displayed by the editor. The rendered output
has a 16:9 aspect ratio; the screenshot's docked viewport rectangle is not the
render target size. This replaces the earlier 512 x 512/default-camera workload
for subsequent budget comparisons; do not mix timings from the two workloads.

Generate fresh fixtures and run the existing measurement runner:

```powershell
python -B Tools/Perf/PreparePathTraceDefaults.py build/pt-user-view/fixtures
python -B Tools/Perf/MaterialBaseline.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --cases build/pt-user-view/fixtures/Cases.json --output build/pt-user-view/performance --timeout 600
python -B Tools/Perf/MaterialBaseline.py verify build/pt-user-view/performance
```

Three arms retain the previous 2-rpp/cache-off reference, a 1-rpp/cache-off
reference, and implicit SHaRC defaults. Each of three independent processes
uses 128 warmup frames, 128 timed frames and a final untimed HDR readback.
All arms use the same fixed studio environment, OpenPBR, guides, and disabled
image accumulation. The screenshot does not specify its complete lighting or
graph settings; this baseline adopts its camera/resolution, not its appearance
or full RenderGraph timing. Measurements cover the PathTrace pass including
SHaRC maintenance where enabled, excluding denoising, upscaling and presentation.
