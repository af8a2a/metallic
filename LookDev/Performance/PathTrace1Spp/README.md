# Default LookDev performance baseline: 1 spp per frame

## Default Nsight diagnostic reference

As selected by the user on 2026-10-20, use
`Captures/NsightGraphics/LookDev_2026_10_20_20_40_25.ngfx-gputrace`
for subsequent LookDev Nsight analysis. [NsightBaseline.json](NsightBaseline.json)
records its SHA-256, preserved screenshot, rounded screenshot metrics and
source hashes observed at registration. This replaces earlier Nsight references.

The screenshot shows Path trace shading at **4.96 ms**, the enclosing PathTrace
pass at **5.44 ms**, and frame 0 at **6.68 ms** (active **6.38 ms**). Keep these
scopes distinct. The selected event reports **9.3% compute warp occupancy**;
`openPbrRayQueryPathTraceMain` reports **54.39% Long Scoreboard** within its
sampled stalls. These are one capture's UI observations, not repeated-run means.

Registration confirmed that current shader files match archived combined B
(committed-hit specialization plus batched attributes and branchless decoding).
This is source provenance at registration, not a verified capture-time pipeline
binding. Scene/camera/resolution/1-spp setup follows the established conversation;
raw trace metadata, tool versions and the screenshot's Duration warning text
have not been extracted. Compare future captures against the same scope and
settings. Keep ordinary A/A timings below separate from diagnostic measurements.

## Ordinary timing calibration

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
