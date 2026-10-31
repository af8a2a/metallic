# Directional Projection experiment

`VirtualShadowMapBitset.py --experiment directional-projection` uses protocol
`virtual-shadow-map-directional-projection-v1`. It compares the generic single-face
Projection program against its orthographic-only compiled variant. The explicit
boolean `benchmarkDirectionalProjection` must be `false` in A and `true` in B;
all other case and graph settings must match. Production `directionalProjection`
is overridden by this benchmark switch.

The established Mark/Build/bitset optimizations are now unconditional. Remove
their retired production and benchmark properties from new capture graphs; the
live runner rejects them rather than treating ignored controls as valid A/B
settings. Historical evidence still verifies under its recorded protocol. See
[retired VSM experiments](VirtualShadowMapRetiredExperiments.md).

This experiment requires one directional face and per-face projection scopes.
Mixed-light one-pass projection is outside this candidate. The shader receipts
require `directionalProjection=true` only for B's `vsmProjectMain`, and require
different Projection SPIR-V but identical Mark, Build, raster mesh and depth
fragment SPIR-V. The workload retains production Mark and direct directional
Build, cluster/page bitsets, and disables projection-statistics optimization.

## Inputs and fixed budget

Supply `Diagnostic-ACases.json`, `Diagnostic-BCases.json`, `Timing-ACases.json`, and
`Timing-BCases.json` in a cases directory. Each contains one case and refers to its
corresponding graph. Use 64 diagnostic frames or 64/96 timing frames, with 32 warmup
frames, streaming freeze and camera cut at frame 16. Diagnostic capture frames are
exactly 16, 32, 63; `comparisonFrames` is empty. Timing disables validation and all
diagnostic readbacks. The current local Sponza reference cases are under
`build/vsm-mark-ray-pages/ABCases`; copy their JSON data into a new directory, retain
ray-page Mark in both arms, and change only the new Projection switch between arms.
Do not execute archived helper scripts.

The runner fixes the budget at two diagnostic processes, two discovery ABBA blocks
(eight fresh processes), and, only after promising discovery, three independent
confirmation blocks (12 processes). Each process has a 600-second timeout.
The primary metric is `projectionMs`: confirmation requires its gain lower 95%
bound to reach 3%, and the RenderGraph gain lower bound to remain at least -2%.
Statistics use independent complete ABBA blocks, not individual frames.

Diagnostic verification requires exact PNG, linear HDR, depth and shadow outputs,
all 32 legacy statistics, all 45 build diagnostics and all stable work counters
from frame 16 onward. Only the existing scheduling-dependent `markPageRequests`
counter is excluded. In particular, sample positions and level tests must match;
the relaxed work-counter rules of the Mark ray-page experiment do not apply.

```powershell
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment directional-projection --phase diagnostic --cases-dir build/vsm-directional-projection/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-directional-projection/Diagnostic
python -B Tools/Perf/VirtualShadowMapBitset.py report build/vsm-directional-projection/Diagnostic build/vsm-directional-projection/DiagnosticReport
python -B Tools/Perf/VirtualShadowMapBitset.py run --experiment directional-projection --phase discovery --cases-dir build/vsm-directional-projection/Cases --exe build-release/tests/MetallicRHITests.exe --output build/vsm-directional-projection/Discovery --diagnostic build/vsm-directional-projection/Diagnostic
python -B Tools/Perf/VirtualShadowMapBitset.py report build/vsm-directional-projection/Discovery build/vsm-directional-projection/DiscoveryReport --diagnostic build/vsm-directional-projection/Diagnostic
```

If discovery requests confirmation, use `--phase confirmation`, a new output
directory, the same `--diagnostic`, and `--reference` pointing to Discovery. Pass
the same links to `report`. Reports contain process medians, ABBA gain confidence
intervals, diagnostic counts, original evidence links and reproducible plotting.
Always inspect `Decision.json`; finishing a run is not acceptance. Keep discovery
and confirmation results separate, preserve inconclusive or rejected evidence,
and do not extend the fixed budget after inspecting a confidence interval.

CPU-only runner regression:

```powershell
python -X utf8 -B tests/perf/TestVirtualShadowMapBitset.py
```

The CPU fixture test is not GPU correctness or performance evidence.
