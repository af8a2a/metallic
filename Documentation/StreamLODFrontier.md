# Production streaming LOD

RT demand uses conservative aggregated refinement bounds (C1) and the offscreen
hysteresis threshold lower bound (C2). Only completely offscreen subtrees omit
raster error. Leaf history is read only inside the hysteresis band. History
uses two words per group and publishes both the token and wanted bit; ray-first,
packed history and sparse-history-write experiments have been removed.

## Single-wave frontier

Distributed split Frontier uses `streamCooperativeLod32Main` on devices with a
fixed subgroup size of 32 and compute ballot/arithmetic support. Unsupported
configurations retain the 64-thread `streamCooperativeLodMain`. This is a hardware
compatibility choice, with no runtime or environment A/B override. Clear, Mask,
Emit and Prefetch still use their 64-thread entry.

The 32-thread frontier processes wider leaves in ordered batches. Wave prefix
and sum replace its cross-wave LDS scan; descending sparse order, cross-LOD
barriers, final publication, history and residency dependencies are retained.

## Frontier active mirror

Cooperative Frontier mirrors active flags in a 4096-bit workgroup bitmap for
primitives with at most 4096 groups. It clears only the used words and reuses
the existing initialization barrier. Both 32- and 64-thread frontiers retain
the same global active writes, sparse order, masks, Clear and final publication;
larger primitives and the scalar fallback keep global parent queries.

Parent queries and bit insertion use LDS atomics: a parent and a current-level
child can share a bitmap word even though their bits differ. Between LOD levels,
the bitmap path uses a group-memory barrier only when transition telemetry is
disabled. Telemetry still reads parent blocked stamps from global memory and
keeps the all-memory barrier. The final all-memory barrier is unchanged because
combined Mask consumes global active flags and sparse IDs. Split Mask is also
unchanged; the bitmap does not persist between dispatches or frames.

## Terminal-only split Mask

After split Frontier publishes readiness and the complete terminal count,
`ready && activeCount == terminalCount` proves that no nonterminal group is
active. Mask directly writes full masks (including 32-cluster groups) and the
selected count, without refined-target queries or a workgroup reduction.
Combined Frontier/Mask retains its original mask generation because its header
is published afterward. This does not implement full Frontier direct output.

The condition can hold while fine demand is blocked by residency. Demand,
history, page requests, active words and sparse ordering continue through the
same production path. No ownership, cache-key or submission-lifetime change is
introduced. `benchmarkTerminalMaskStats=true` only adds untimed counters in
demand words 29–31: ready instances, terminal-only instances and their total
cluster count. It does not select an implementation.

## Regression and diagnostics

`meshlet_lod_stream_rt_demand` validates production demand against independent
per-leaf references. `meshlet_lod_stream_frontier32` directly compares the
hardware-compatible 64-thread entry with the 32-thread entry.
`meshlet_lod_stream_frontier_lds` checks 288 consecutive WG64/WG32 frames at
31/32/33/4095/4096/4097 groups against independent leaf demand, active and mask
references. It covers shared-word parent/child atomics, multiple parents,
complete sparse ordering, residency and visibility changes, telemetry blocked
ancestry, and recorded-but-unsubmitted work.
`meshlet_lod_stream_terminal_mask` checks 48 consecutive production frames,
including mixed-depth 1/31/32-cluster terminals, unavailable detail and blocked
parents, missing terminals, visibility/manual transitions, history boundaries,
unsubmitted commands and diagnostic counters on/off.

The roam config `streamingRegression` runs one production implementation after
source convergence and a canonical owner/history reset. It accepts bounded
`sampleFrames` (16–64), `settleFrames` (4–16), and one or two fixed `seeds`.
Timing and readback snapshots are separate. Optional `prepareProfile` and
`cpuThreadDiagnostics` are diagnostics, not ordinary timing. Effective shader
module, entry, workgroup size and RHI SPIR-V content hash are exported.

The old `detailDemandComparison` and `blasQualityComparison` configuration
sections are rejected. Their archived captures remain historical evidence;
they are not executable A/B policies in the current renderer. The old C1/C2,
history mode, Frontier32 and terminal Mask properties/environment switches no
longer select alternate implementations.
