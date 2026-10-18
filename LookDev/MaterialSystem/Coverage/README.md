# Coverage scene A/B fixture

`Cases.json` runs the existing `material_phase0_baseline` RHI test against the
authored OpenPBR shaderball and calibration mesh, camera, HDR environment and
manual exposure. Set `METALLIC_MATERIAL_BASELINE_CASES` to its absolute path.

All six cases retain the original OpenPBR surface parameters. Material 0 uses
MASK with cutoff 0.5; the fixture changes only its Coverage expression:

- `control`: default texture/base alpha, no custom Coverage program.
- `simple`: `alpha * fract(uv * 8)`, producing visible repeated cutouts.
- `complex`: five nested `fract(sin(value * scale) + bias)` layers multiplied by
  alpha, exercising a larger expression while retaining visible cutouts.

Each workload runs independently through the existing Deferred-only and
Pathtrace-only graphs at 768 x 768. A fresh preview starts its frame sequence at
zero. Frames 0–31 warm up, frames 32–95 are timed, and frame 96 performs the raw
linear HDR readback. Path tracing uses four samples per frame, depth 12,
accumulation enabled and cache mode off; the final image contains 388 samples
per pixel. No readback enters the timed interval. Both arms must run this exact
sequence with the same graph and material inputs.

`MaterialBaseline.json` records graph GPU duration and per-node timings, resolved
graph/material inputs, environment readiness, and raw HDR filenames/formats.
Graph GPU duration covers the offscreen scene render; CPU, presentation, shader
compilation and final readback are outside that scope. These are deliberately
Coverage-active LookDev fixtures, not a representative scene distribution.
