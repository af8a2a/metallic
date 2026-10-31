# Mark page-size specialization experiment

**Historical protocol only.** Page-size specialization is now unconditional for
directional Mark. Live `run --experiment mark-page-size` and `prepare_plan` reject
new experiments; the production and benchmark switches below have been removed.
Existing sealed packages can still be verified and reported without modification.
See [retired VSM experiments](VirtualShadowMapRetiredExperiments.md). The remaining
description and capture commands document the historical protocol, not a supported
current capture workflow.

`VirtualShadowMapBitset.py --experiment mark-page-size` uses independent protocol
`virtual-shadow-map-mark-page-size-v1`. It compares dynamic integer page-size
division against a constant shift in directional Mark. The runtime setting is
`specializeMarkPageSize`, the production property is `markPageSizeSpecialization`,
and the explicit boolean `benchmarkMarkPageSizeSpecialization` overrides it.
The benchmark switch must be `false` in A and `true` in B; all other settings match.

This experiment requires one directional face with per-face Projection. Declare
the same power-of-two `pageSize` in [1,128] and `clipmapCount` in [1,12] in both
graphs. Existing directional Mark, clipmap specialization and ray-page fast paths
remain enabled, while directional Projection specialization remains disabled.
Sampling, page requests and logical work are unchanged.

The shader identity gate checks five entries: Mark, Build, Projection, raster mesh
and depth fragment. Only Mark SPIR-V may change between arms. Actual
`[VSMShader] markPageShift` is -1 in A; B's Mark uses `log2(pageSize)` in [0,7],
and all other entries retain -1. The gate also checks the existing Mark cache
bounds, ray-page fast path, production flags and directional light type.

Diagnostic verification requires byte-identical PNG, linear HDR, depth and shadow
captures at frames 16/32/63, all 32 legacy statistics, all 45 build diagnostics and
all 15 stable workload counters from frame 16 onward. Only the established
scheduling-dependent `markPageRequests` exception applies. Mark sample positions
and level tests must remain identical, unlike the earlier ray-page experiment.

## Fixed budget and commands

Prepare `Diagnostic-ACases.json`, `Diagnostic-BCases.json`, `Timing-ACases.json`, and
`Timing-BCases.json`, each with one case referring to its corresponding graph.
Use 64 diagnostic frames and 64 or 96 timing frames; warmup is 32 frames, streaming
freeze and camera cut occur at frame 16, and `comparisonFrames` stays empty.
Timing disables validation, captures and diagnostic readbacks. For the established
Sponza workload, retain native 1920x1080, six clipmaps, 128-texel pages, 2048
physical pages, SMRT 2x16 and the fixed camera. Copy case JSON data into a new
directory; do not execute archived helper scripts.

The fixed budget is two diagnostic processes, two discovery ABBA blocks (eight
fresh processes), then three independent confirmation blocks (12 processes) only
if discovery requests confirmation. Each process has a 600-second timeout. The
primary metric is `markMs`: confirmation requires its gain lower 95% bound to
reach 3%, with RenderGraph gain lower bound at least -2%. Frames are not independent
statistical samples; the complete ABBA block is the statistical unit.

```powershell
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment mark-page-size --phase diagnostic --cases-dir build/vsm-mark-page-size/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-mark-page-size/Diagnostic
python -B Tools/Perf/VirtualShadowMapBitset.py report build/vsm-mark-page-size/Diagnostic build/vsm-mark-page-size/DiagnosticReport
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment mark-page-size --phase discovery --cases-dir build/vsm-mark-page-size/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-mark-page-size/Discovery --diagnostic build/vsm-mark-page-size/Diagnostic
python -B Tools/Perf/VirtualShadowMapBitset.py report build/vsm-mark-page-size/Discovery build/vsm-mark-page-size/DiscoveryReport --diagnostic build/vsm-mark-page-size/Diagnostic
```

For conditional confirmation, use `--phase confirmation`, a new output directory,
the same diagnostic link and `--reference` pointing to Discovery. Pass both links
to `report`. Inspect `Decision.json`, retain all outcomes, and keep discovery and
confirmation separate. Reports include Python-generated process medians, block
gain intervals and diagnostic workload plots. Do not extend the budget after
inspecting confidence intervals.

CPU regression command: `python -X utf8 -B tests/perf/TestVirtualShadowMapBitset.py`.
These tests validate the evidence protocol; they do not establish GPU correctness
or performance.
