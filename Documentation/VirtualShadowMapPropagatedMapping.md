# Propagated VSM projection mappings

As of 2026-11-04, compact propagated mappings are the standard Projection
implementation. The `propagatedMapping` setting and
`benchmarkPropagatedMapping` opt-out have been removed. Old baseline measurements
below describe historical evidence, not a selectable runtime path.
The implementation prepares sampling mappings once
per virtual page so a resident page or an integer-mapped ancestor can be resolved with
one derived-table read. It does not change Mark, physical-page admission, raster
ownership, filtering, the configured ray sequence, or adaptive ray count.

## Storage and publication

The original `uint4` page-table prefix retains its existing ownership contract:
physical page in `x`, requested/admitted flags in `y`, the page's own projection
level in `z`, and composite/static valid/dirty flags in `w`. These entries are
consumed by allocator ownership reconstruction, eviction, cache remapping, static
depth reuse, and rasterization. An ancestor physical address must not replace
`pages[id].x`, since multiple virtual pages would then claim the same physical
owner on the next frame.

With the candidate enabled, the page buffer has a second `uint4` per virtual
page. `vsmPropagateMappingMain` rebuilds this tail after Finalize completes:

1. Finalize updates validity, including fine-work overflow handling.
2. A compute write/read dependency publishes the ownership prefix.
3. Propagation reads only the stable prefix and writes only its own tail record.
4. The following whole-buffer dependency publishes the result to compute and
   pre-raster consumers. Raster ownership continues to use the prefix.

There is no cross-invocation read/write overlap within the propagation pass.
It cannot be merged into the current Finalize dispatch by reading neighboring
validity flags: those flags may still be changing in other workgroups.

Cache reset reads only the previous prefix, so switching between one- and
two-table allocations does not copy stale derived mappings. Current tail records
are rebuilt for every page, including unsupported or unused records. One-pass
lookup derives the tail offset from each job's `clipmapCount * VPA * VPA + 1`;
it does not rely on the first job's `totalPages`. The common VPA is enforced by
the existing shared-job settings contract and an assertion.

The extra logical storage is `totalPages * 16` bytes per face. Sponza's six
128-by-128 clipmaps plus coarse page add 1,572,880 bytes, approximately 1.5 MiB,
for its admitted directional light. This describes the additional logical table,
not a measurement of allocator or driver VRAM residency.

## Record format

The tail stores `uint4(physicalPage, targetLevel, packedParentPage, mode)`:

| Mode | Meaning |
| --- | --- |
| 0 | Unsupported or uncertain mapping; preserve the original search. |
| 1 | This page is admitted, resident, and valid; return its own address. |
| 2 | Directional integer ancestor or coarse fallback, with relative projected coordinates. |
| 3 | Local-light dyadic mip ancestor or coarse fallback; no sample boundary certificate is needed. |

`targetLevel == clipmapCount` means use the unchanged coarse fallback path;
its stored physical field is `UINT32_MAX`. The filtered lookup still reads the
coarse entry and validates it. Thus a coarse result can require a derived-table
read plus a coarse-table read. Hard projection retains its existing independent
coarse handling. Global work overflow still bypasses all fine-level lookup.

## Directional mapping and numerical contract

The current fast subset requires a power-of-two VPA of at least two. It uses
the canonical snapped origins and power-of-two extent sequence produced by
`makeProjections`. Every visited ancestor, including the mapped target, must
have exactly scaled XY coefficients, the same Z row, and a supported signed
integer origin range. VPA is bounded by the CPU settings contract to 128, so
packing the parent's X and Y page coordinates into two 16-bit fields is sufficient.
The integer parent page is:

```
((sourcePage + sourceOrigin - axis/2) >> levelDelta)
    - parentOrigin + axis/2
```

The shift is signed arithmetic shift, so negative coordinates use floor
division. Parent pages on the outer row or column disable the mapping:
the original half-texel containment inset can cut through those page interiors.

The mapping stores the integer parent's physical address, level, and packed page
coordinate. Lookup transforms the source's already-computed continuous NDC XY:

```
scale = 2^-(targetLevel - sourceLevel)
translation = targetTranslation - sourceTranslation * scale
targetProjected.xy = sourceProjected.xy * scale + translation
targetProjected.z = sourceProjected.z
```

All fine directional levels produced by `makeProjections`, including reused
cache projections, share the same depth row. Metallic therefore preserves source
Z without the level-dependent depth scaling used by Unreal's representation.
The complete coarse projection remains independent and uses the old fallback.

Continuous target XY stays unclamped for receiver-plane depth correction. Only
the floating-point texel address is clamped to the integer parent's texel range,
before conversion to unsigned integers. The nonnegative conversion implements
floor and cannot address a neighboring physical page due to rounding. Integer
origins are shifted in page units; they are not multiplied by page size, avoiding
overflow for large supported world origins.

This is a changed numerical contract, not a bit-exact rewrite of the previous
world-dot projection. Independent world projection can round across a page
boundary where the integer hierarchy chooses the other parent; relative XY can
also change the final texel or receiver-plane correction. Lookup now trusts the
integer parent's ownership rather than checking a conservative per-sample error
certificate. The outer-page slow path retains the existing containment inset;
unsupported mappings retain the original search. SMRT's separate projection of
the undithered ray point for depth comparison remains unchanged.

## Local mip mapping

Local mip misses use mode 3 when the source axis is a power of two, each visited
parent axis matches the halved/clamped sequence, and all four projection rows
equal the source rows. Local levels consequently share the same projected
coordinates and inclusive containment domain. For the resulting nonnegative
normalized address, multiplication by a power-of-two resolution and flooring
commute with a right shift. The existing upper-edge clamp also commutes: the
largest source texel shifts to the largest target texel. The same argument holds
once the mip axis reaches one and subsequent levels repeat that resolution.

Unused fixed-stride slots remain unsupported. Non-dyadic source mips use the
original search. Point-light samples select their face before resolving the
mapping, as they did before this candidate.

## Scope, counters, and evidence

This is a constant-table-read fast path, not a worst-case O(1) replacement for
every query. A sample outside its initial clipmap still searches for a containing
level. Directional outer-page cases and unsupported mappings still execute
the original level loop. A coarse result retains its own validity lookup.

Earlier jump-only metadata supplied a level to resume the search, so it still
needed a residency-table read at the destination after paying metadata and guard
costs. That approach did not establish a worthwhile net benefit. This candidate
instead supplies the final physical address and level, including the own-page
case, to remove that destination probe. Its preparation pass, extra storage,
relative-coordinate cost, and slow paths still require measurement; fewer probes alone
do not establish a GPU-time improvement.

Workload diagnostics append raw page-table reads, derived mapping reads,
successful mapped fine ancestors, and mapping slow paths. Baseline and candidate
both count original lookup reads; diagnostic atomics are excluded from ordinary
timing. The `VSM propagate mappings` GPU marker is nested under build/finalize;
do not add it again to their inclusive times.

The bounded experiment report is a local output at
`build/vsm-propagated-mapping-20261101/Report.md`. For the previous exact-certificate
candidate, the 17-case focused GPU test
and five paired Sponza captures passed exact comparisons. The diagnostic
snapshot showed 31.55% fewer combined original/derived table reads. A single
ordinary ABBA block nevertheless increased Projection time by 18.50% and
Shadows time by 11.52%. The implementation remains disabled by default; no
performance adoption or independent statistical confirmation is claimed. These
results do not validate the revised integer-parent numerical contract above.

The revised candidate has separate evidence at
`build/vsm-relative-mapping-20261101/Report.md`: 19 focused GPU cases (114 lookup
pairs) passed the integer/relative-coordinate oracle, including documented
differences from the old boundary contract. Twelve paired Sponza captures,
including consecutive camera-motion frames and a resize/cut, had byte-identical
depth, visibility, HDR and shadow outputs. Combined lookup reads fell 32.81% in
the fixed diagnostic frame, but one ordinary ABBA block still increased
Projection time by 13.36% (within-arm spread below 0.5%). Broader Shadows/graph
timings drifted and do not establish an end-to-end improvement.

Actual ordinary SPIR-V confirms the directional mapping branches contain no
target world-dot or error-certificate calculation. The generic fallback and
local-light branches remain compiled, and target projection metadata is still
loaded. Native register/spill attribution is unavailable: both bounded Nsight
attempts failed to render the scene because the injected target could not write
its scene-streaming source identity. The Present attempt exported only editor
work, which was rejected as VSM evidence. No additional timing runs were added,
and the revised candidate remains disabled by default.

A subsequent capture-only repair is documented in
[Nsight capture runtime integrity](NsightCaptureRuntimeIntegrity.md), with the
new report at `build/vsm-nsight-repair-20261101/Report.md`. The original EXE's Low
integrity label caused the write failures; fresh, byte-identical Medium runtime
files enabled valid SDK frame-48 A/B traces without rerunning correctness or
ordinary timing. The actual bound Projection shaders use 127/128 registers per
thread, 109056/130816 native code bytes, and 12/5904 static shared bytes per block.
Both capture SPIR-V programs explicitly declare only 12 shared bytes, so the
additional allocation is compiler-generated. B also introduces local-memory
loads/stores and much more shared traffic, while occupancy remains approximately
30.4/30.3 percent. This strongly supports temporary-state/register-pressure
costs as the leading regression mechanism. Exact spill variables/instructions
remain unproved, and capture-symbol native code is not assumed identical to
ordinary-mode native code. The candidate remains disabled.

## Compact lookup state

The propagated lookup now returns the selected projection-table handle and level
instead of a complete 112-byte `ShadowProjection`. The table handle preserves
point-light face selection. PCF and SMRT load the selected projection only after
a valid lookup, while the projection buffers remain immutable for this pass.
SMRT still projects the undithered comparison point independently.

The mapping helper accepts only source XY translation and virtual-page index;
it does not mutate or return a projection aggregate. Directional mode 2 reads
only target XY translation and page-axis length, and returns before the generic
local-light target projection. Mode 0 and rejected mode 3 preserve source
coordinates for the existing fallback. Coarse containment and validity handling
are unchanged. This does not enable the separate directional-projection macro
or change integer-parent ownership and relative-coordinate arithmetic.

Bounded evidence is in `build/vsm-compact-state-20261101/Report.md`. The focused
19-case GPU test passed, and twelve Sponza frames were byte-identical for depth,
visibility, HDR and shadows both against the baseline and the previous candidate.
Actual ordinary SPIR-V confirms late selected-projection loads, no complete
projection-state copies in the six directional mapping branches, and fewer
vec4 Function loads/stores (608/526 to 364/238 versus the previous candidate).
These IR operations are not physical local-memory transactions.

One ordinary ABBA block measured Projection at 0.999568/1.000336 ms for A and
0.861632/0.861936 ms for B: 13.82 percent lower, with within-arm spreads below
0.1 percent. Build increased by approximately 0.0041 ms including propagation.
Broader Shadows/graph ranges drifted; no end-to-end gain or independent
statistical confirmation is claimed.

Two valid SDK captures show current A/B at 127/96 registers per thread and
12/784 native shared bytes per block, with only 12 explicit SPIR-V shared bytes
in each. Candidate local-memory throughput is zero, and shared load/store rates
fell substantially from the previous candidate's capture. Occupancy is
30.33/37.54 percent. The additional 772 shared bytes and elevated shared traffic
remain; exact spilled variables/instructions are not established. The ordinary
and capture-symbol binaries are separately identified. Both A and B SPIR-V
identities changed with the refactor, so performance conclusions use the current
controlled A/B rather than subtracting timings across versions.

The experiment completed its fixed nine-process budget (one focused, two
diagnostic, four ordinary, two Nsight). No full-scene local-light/point-face or
mixed-light one-pass acceptance was added. It remained disabled until the
default-path adoption below.

## Default-path adoption

The compact-state mapping is now enabled by default. The pass inherits the
settings default when `benchmarkPropagatedMapping` is absent. `Programs`' initial
false value is cache initialization, overwritten from settings before shader
compilation; it is not an independent feature default.

`build/vsm-compact-default-20261101/Report.md` records one additional 64-frame
Sponza diagnostic run with the override property removed. Both actual shader
entries reported `propagatedMapping=true` and exactly matched the verified
candidate's SPIR-V hashes. All 64 frames contained the propagation marker and
identical shadow statistics; the twelve captured frames' depth, visibility,
HDR and shadow outputs matched the candidate byte-for-byte. Validation and
synchronization checks were enabled. Release builds of MetallicRHITests and
MetallicVSMSample passed.

Only the settings default changed; shader code was unchanged. Existing
correctness, ABBA and Nsight evidence was reverified offline and reused, with no
new ordinary timing or capture runs. The previous single-block performance and
Sponza-only runtime coverage limits still apply. Adoption changes the default,
not the strength or scope of that evidence.
