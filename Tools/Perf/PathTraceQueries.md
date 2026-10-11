# Path tracing query experiments

`PreparePathTraceQueries.py` creates small deterministic glTF fixtures (opaque,
MASK, BLEND, nested glass, normal and transmission textures), a nine-case GPU
correctness suite, and three production PathTrace timing workloads. It preserves
source assets. Python dependencies: numpy, scipy, psutil, Pillow and matplotlib.

The production changes share **only authored primary HitInfo**, including a known
miss, between RR samples and guides. BSDF/texture/coverage RNG streams and guide
material evaluation remain independent. Reused camera intersections are consumed
once so alpha continuation at depth zero traces again. Non-guide multi-sample
paths have different camera jitter and retain their existing behavior.
The first sample caches its raw hit inside the original path loop, before material
evaluation. Hoisting reconstruction outside that loop caused sparse floating-point
drift in ABeautifulGame; that production scene and frame 96 are regression cases.

The one-rpp guide variant writes fixed guides in its primary stage. Multi-sample
guide variants do the same immediately after caching the first authored hit in
the original loop, before evaluating the transport material. Their independent
guide RNG, texture footprint and post-normal-map face-forward policy must remain
unchanged; do not reuse the transport closure. Only the first sample's specular
hit distance is written after continuation. SHARC Update never writes guides.
Changes to this scheduling should cover multi-sample MASK/BLEND, glass, sky,
history, and normal/texture debug overrides as well as the fixed production view.
Source-level lifetime reduction is not proof of lower physical register usage.

Boolean resident-scene shadows accept the first coverage-approved blocker. The
OpenPBR transmittance query still finds the nearest surface: a non-BLEND material
with an exactly zero transmission factor returns opaque before full attribute
decode. Other surfaces keep their original texture evaluation, nested medium
stack, attenuation, offsets and traversal order. Streamed queries conservatively
retain full decoding. No BSDF, path-depth, sample-budget or denoiser setting changes.

## Capture

Save the four shader files listed in `PathTraceQueries.FILES` before editing into
`<baseline>`, and after primary-hit-only changes into `<primary>`. The current
workspace must contain the final candidate. The runner archives all three and
checks source, executable/DLL and declared asset hashes. Use a current Release
`MetallicRHITests` build. It uses `material_phase0_baseline`, not a substitute
microbenchmark. The general `ExperimentRunner` remains limited to WorkControl.

```powershell
python -B Tools/Perf/PreparePathTraceQueries.py build/pt-fixtures-new
python -B Tools/Perf/PathTraceQueries.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --cases build/pt-fixtures-new/Correctness.json --baseline <baseline> --primary <primary> --output build/pt-correctness-new --phase correctness
python -B Tools/Perf/PathTraceQueries.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --cases build/pt-fixtures-new/Performance.json --baseline <baseline> --primary <primary> --output build/pt-performance-new --phase performance
python -B Tools/Perf/PathTraceQueries.py verify build/pt-performance-new
python -B Tools/Perf/PlotPathTraceQueries.py --repo . --correctness build/pt-correctness-new --performance build/pt-performance-new --output build/pt-report-new
```

Correctness runs A/B/C with Vulkan validation, exact guide bytes and declared
HDR tolerances (absolute and relative 2e-5). All temporal captures occur outside
the timing interval. Explicit RenderView camera updates exercise actual motion
history; changing graph properties alone is not sufficient for that contract.

Performance runs six balanced triplets: ABC, CBA, BAC, CAB, ACB, BCA. Each of the
18 independent processes measures 64 frames after 32 warmup frames. It records
PathTrace and graph GPU timestamps at 512 x 512, 2 samples/frame, maxDepth 12 and
SHARC off, with fixed studio HDR lighting. ABeautifulGame uses the sample camera.
It does **not** measure DLSS/NRD, presentation or end-to-end frame rate.

Compile, loading, warmup and readback are outside the measurement window. Clocks
are unaltered. Other applications are not closed; the cooperative experiment
lock does not establish global GPU exclusivity. The unit of statistics is the
process median; the six paired gains use a 95% Student-t interval. Improvement
requires a PT gain lower bound of 3%, a graph lower bound of -2%, and <=10%
process-median spread in each arm. Results crossing a gate remain inconclusive;
do not repeat unchanged experiments until they pass.

The original installed candidate is restored in `finally`, including on capture
failure. Unknown concurrent shader bytes are never overwritten. After a hard
interruption, inspect the lock owner and use:

```powershell
python -B Tools/Perf/PathTraceQueries.py recover <experiment-directory>
```

Reports are immutable after sealing. Verification uses current trusted source,
checks manifests, capture sequences and shader payload hashes, and recomputes
differences/statistics. Cached production SPIR-V is archived by content hash,
because the compiler can reuse a cache filename after shader dependencies change.
It is compiler/PSO input evidence, not a capture of driver-rewritten machine code.
Generate plots outside the sealed evidence directories. Preserve failed attempts.
