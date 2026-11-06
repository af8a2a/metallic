# Directional Mark receiver coverage diagnostics

`VirtualShadowMapReceiverCoverage.py` is an offline analyzer. It never launches the renderer, captures Nsight, or measures/predicts GPU time. Input is the acceptance receipt containing `cases[].receiverCoverageCaptures[]`. It uses one recorded frame and the same data for 1/2/4/8-pixel-side groups and 1/2 texel numerical padding; side 1 is the receiver baseline.

```powershell
python -B Tools/Perf/VirtualShadowMapReceiverCoverage.py path/to/VirtualShadowMapAcceptance.json --output build/vsm-receiver-coverage-20261101/Analysis
python -B tests/perf/TestVirtualShadowMapReceiverCoverage.py
```

`--frame` defaults to 32. Relative binary paths resolve against the input JSON directory (`--root` overrides this). A direct capture descriptor is also accepted; pass `--gpu-mark-level-tests` if it lacks the same-frame work counter. Full acceptance receipts automatically supply matching directional face-0 counters. The report records null verification status if a direct descriptor provides no GPU counter; that is not validated workload accounting.

## Input protocol

Capture version 1 declares width/height, pageSize, virtualPagesPerAxis, clipmapCount, physicalPageCount, rayCount, samplesPerRay, angularRadius, lightIndex, frame, recordWords=22 and recordsEnabled=true. The experiment is restricted to one directional face with two rays. Each binary descriptor has `file`, `byteLength`, `rawSha256`; the analyzer verifies all three.

Records are viewport-order little-endian uint32 words:

| Word | Meaning |
|---|---|
| 0 | Receiver valid |
| 1–3 | Original world position, float bits |
| 4 | Original effective ray length, float bits |
| 5 | Original receiver level |
| 6–7 | Original ray 0/1 residual sweep masks |
| 8 | Undithered primary center page, or UINT_MAX; hypothetical policy input, not necessarily requested |
| 9–20 | GPU ray-0 t=0 witness per level, UINT_MAX if absent |
| 21 | Original linearDepth, float bits |

A successful original single-page ray certificate supplies its already requested page as witness. A residual level supplies the original ray-0 t=0 sample page. Consequently W contains real requests from both certificate successes and failures, not just the successful certificate set. Every W page must occur in the real baseline requested flags, or analysis fails.

Projections are 112-byte `ShadowProjection` records: float4 rowX/Y/Z/W/metric, uint4 page, float4 originNear. Includes the final coarse record. Pages are uint4 entries including coarse; requested is `.y & 1`, admitted is `.y & 2`.

## Coverage model

For each receiver with residual work at a level, Q bounds every point in the directional light cone up to the original effective ray length plus independent half-texel dither. Per projection row, the support uses the normalized light axis, perpendicular row component, cone angle and receiver-selected world texel size. It does not regenerate random rays or replay every sample. Z rejection is omitted, making Q looser. Q is clipped to the virtual XY domain and converted to a page rectangle.

For a tile/level, Q is the bounding rectangle of residual receivers' Q; W is the union of all active receivers' actual witnesses at that level. Q is considered potentially covered only when every page in its rectangle occurs in W. Integer containment is exact and rejects invalid/cross-level/duplicate witnesses correctly. **The CPU double support plus empirical 1/2 texel padding is not a strict GPU floating-point/SFU error proof. All savings are potential; proven savings remain zero.** Empty intersections can remove residual work in the mathematical model, but remain subject to the same numerical caveat.

Original residual tests are `sum(popcount(raySweepMasks)) * (samplesPerRay + 1)`. Original producer tests are `rayCount * sum(clipmapCount - receiverLevel)`. Their sum must equal the same-frame GPU `markLevelTests` when supplied. Receiver count and reconstructed sample-position count are also checked when present. Original successful ray certificates are excluded from savings. No claim is made that fewer logical level tests implies a faster GPU pass.

Extra model counts are separate: two support row dots per residual receiver/level, explicit ray-0 witness preparation where its certificate failed, scalar min/max reduction combines, witness slots visited, and membership queries after an ideal unique-count rejection gate. They are not interchangeable operations and are never subtracted into a synthetic GPU time. The model does not choose/measure the required shared-memory hash, sorting, barriers or register implementation.

**Data availability is a separate cost.** Q is computed only for receivers/levels already known to have residual masks; successful witness pages reuse the original ray certificate producer. An online implementation must first obtain those original ray certificate results. It may need retained ray state, a separate producer/barrier phase, or ray regeneration. None of these state/storage/scheduling/recomputation costs is measured here. This offline model must not be described as an online algorithm with no preliminary ray generation or prescan.

Savings are split into `potentialSavedEmptyDomainSampleLevelTests` (Q has no pages after XY-domain intersection) and `potentialSavedWitnessCoveredSampleLevelTests` (nonempty Q contained in W), including per-level counts. Empty-domain removal requires no witness lookup; it still lacks a strict floating-point bound. Full residual ray-sweep removal is split into witness-only, empty-domain-only and mixed reasons, so a large total cannot be mistaken for effective tile witness sharing.

Tile diagnostics include valid-receiver counts, distinct receiver levels, mixed-level fraction, absolute depth span in world units and relative span `(maxDepth-minDepth)/minDepth`. These help identify discontinuity-driven rectangle inflation.

## Hypothetical page policies and budget

Report exactly six bounded examples: primary-only; primary plus actual ray-0 parent witnesses; primary plus one-page 8-neighbor halo plus parents; and tile 2/4/8 same-level primary bounding rectangle plus one-page halo plus parents. Parent witnesses are GPU-recorded pages on levels strictly above each receiver's selected level, not child-coordinate shifts or CPU-reprojected approximations. Fine-grid halo is clamped independently per level. These policies deliberately change demand and are not exact rendering candidates.

The real allocator admits requested fine pages by descending virtual-page index with one reserved coarse slot. Analyzer replays that rank and fails if requested/admitted flags disagree. Each policy reports extra/missing requests and baseline-admitted pages displaced. It does not simulate physical mapping reuse, cache dirtiness, multi-light feedback, geometry work, image quality or temporal history.

Outputs: `Coverage.json` (versioned schema, hashes, validation, baseline accounting, eight coverage rows with per-level detail, tile diagnostics, six policies) and `Coverage.png` (potential coverage, unweighted work counts, hypothetical budget effects). Diagnostic GPU timings are not consumed.
