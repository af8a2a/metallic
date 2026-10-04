# Material system Phase 0 fixtures

`Cases.json` and the three graph files freeze the current OpenPBR PT, OpenPBR
VBuffer Deferred and RTXCR Chiang workloads. Run from the repository root.
The runner hashes referenced assets and these fixtures before and after capture.
Do not silently regenerate or edit the reference when comparing a candidate.

Images, timestamps, diagnostic captures and reports belong in a new ignored
`build/material-phase0-*` directory, not in this source directory.

See [the Phase 0 contract, ABI and validation](../../Documentation/MaterialSystemPhase0.md).
