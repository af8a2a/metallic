# Default LookDev performance baseline: 1 spp per frame

Use this profile for default LookDev shader performance experiments. The production
`openpbr_lookdev` graph uses one path-tracing sample per pixel per frame with
accumulation enabled. The case fixes the default scene/camera at 1920×1080 and
depth 12, renders 768 frames (768 accumulated spp), excludes 256 warmup frames,
and measures frames 256–511. Remaining frames complete the image readback.
768 is the nominal accumulated sample budget; the harness does not read back
the internal accumulation counter. Environment changes are recorded separately.

This replaces the 4-spp workload used for the October 19 stream-decoding
experiments. Historical 4-spp evidence and the separate material-system Phase 0
fixtures remain unchanged. Do not pool those timings with this baseline, divide
them by four, or transfer an optimization decision between sampling workloads.

From the repository root, use a compatible Release test executable:

```powershell
python -B Tools/Perf/MaterialBaseline.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --cases LookDev/Performance/PathTrace1Spp/Cases.json --runs 3 --timeout 600 --output .cache/lookdev-1spp-new/Capture
python -B Tools/Perf/ReportLookDevBaseline.py .cache/lookdev-1spp-new/Capture --output .cache/lookdev-1spp-new/Report
```

Use fresh output directories. The three serial A/A processes keep source,
binary, assets, camera and graph fixed. Qualification requires finite bitwise
identical raw images and at most 10% range/median across independent process
medians for both PathTrace and the graph. Record all samples; an unstable result
is inconclusive, not grounds to retry until it passes. This qualifies a baseline,
not an optimization. Retain existing disk caches; exclude diagnostic instrumentation
from normal timing. No GPU-exclusivity claim follows from the cooperative lock.
