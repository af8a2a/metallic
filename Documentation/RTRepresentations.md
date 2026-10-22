# Streamed RT representations

The streamed ray-tracing path separates a BLAS owner from a rendered instance.
Raster LOD traversal still produces per-instance cuts. Each primitive has three
RT representation slots, and each instance selects a slot or its terminal BLAS.
`METALLIC_BLAS_REPRESENTATIONS=0` selects the existing per-instance path for A/B
testing; `1` enables the representation path.

## Ownership and admission

- **Merged:** the highest drawable cut of the primitive's currently resident
  hierarchy. Refinement substitutes complete child groups for their parents;
  it does not concatenate overlapping instance cuts.
- **Shared:** a representative instance's complete raster cut. Conservative
  per-LOD enclosing bounds classify other instances; an instance may share only
  when its minimum permissible LOD is no finer than the representative cut's
  actual maximum LOD. Bounds include rotation and nonuniform scale.
- **Discrete cache:** a complete resident discrete LOD cut, retained across
  frames and temporary lack of consumers. This is an opportunistic cache, not
  a mechanism for pinning geometry pages indefinitely.

BLAS references, build slots and storage are charged once per representation.
The three slots impose a maximum of three dynamic BLAS owners per primitive,
including a discrete-cache refresh. This does not make instance classification,
TLAS creation, raster traversal or all RT preparation independent of instance
count. The selected representation retains the authored TLAS instance identity
and transform.

The compact RT group buffer has its own capacity. Admission publishes complete
primitive ranges, never a truncated group list. Under pressure, idle cache
storage gives way to active demand. A rejected shared/cache selection may use
an already-ready merged representation of the same primitive; otherwise it
uses the existing terminal fallback.

## Owner demand importance

Each owner also carries demand metadata aggregated from **all** instances that
request it, before budget rejection. A shared cut's representative is not its
importance proxy. The primary metric is the maximum estimated terminal-fallback
error in render pixels; minimum surface distance and maximum projected viewport
coverage are independent auxiliary metrics and may come from different instances.
These estimates describe the cost of the terminal fallback, not a measured
image difference or an exact error delta against the current representation.

The immutable metadata buffer appends one fallback sphere/error record per
primitive after its LOD records. It covers the complete terminal cut, including
branches that terminate early. Terminal groups' own errors are stop sentinels;
the approximation error comes from each terminal cluster's generating group,
with zero error for original clusters. The envelope includes both terminal and
generating groups and is rounded outward.

Projection uses a conservative affine scale bound, including shear. Every
consumer contributes a direction-independent ray error; primary-frustum
consumers additionally use the raster projection bound. Orthographic error is
distance-independent. Offscreen and behind-camera consumers remain eligible for
shadow/reflection use, with zero primary-view coverage rather than zero demand.
This is a camera-based proxy, not an estimate of an individual shadow or reflected
ray's footprint.

`rtRepresentationState` appends six words per owner after the instance records:
`fallbackErrorKey`, `nearestSurfaceDistanceBits`, `maxCoverageBits`,
`consumerCount`, `frustumVisibleConsumerCount`, `offscreenConsumerCount`.
Nonnegative finite float bits support atomic max/min. Error key zero means no
consumers; otherwise `asfloat(key - 1)` is the error, so even exact-zero-error
consumers have a nonzero key. Empty owners have maximum finite distance and zero
coverage/counts. The fields reset and aggregate every executed frame; they do
not invalidate cached cuts or change owner identities.

This stage exposes importance only. Group, reference, build-slot and storage
admission still use their existing order; nonzero offscreen demand does not yet
guarantee admission or prevent starvation under capacity pressure.

## Residency and submitted work

Every selected cluster must have a drawable page and a published CLAS. Cache
keys include CLAS publication generations, so unchanged topology does not imply
that physical addresses are still valid. Missing dependencies invalidate the
representation before publication. Raster demand continues independently.

Merged and discrete cut masks also have a GPU-resident cache keyed by page-table
mutation serial and CLAS publication revision. Shared masks are compared with
the actual current raster frontier, including representative identity and group
count. This avoids using camera stillness as a proxy for unchanged geometry.
Recorded page-table mutations advance their serial even if the recording is
cancelled; only executed GPU commands publish new cache metadata.

Representation records and buffers are retained by command submissions. The
existing render-graph completion dependencies order writes to the shared BLAS
arena after prior consumers. CLAS chunk retirement also waits for recorded GPU
uses. New initialization/publication state is committed through submission
transactions; cancelling unsubmitted work preserves the last submitted cut.

## Validation

`stream_rt_representation_importance` dispatches production reset and aggregation
over 137 instances across three workgroups, with classified LOD inputs supplied
by the fixture. It checks shared-consumer maxima, permutation invariance,
offscreen/behind-camera demand, orthographic/affine/near-plane cases and 32 frames
of removal and owner migration. It does not test priority admission.

`stream_rt_representation_lifecycle` exercises stable frames, sharing, merging,
cache return after temporary disuse, tight budgets, cancellation and retirement
with two frames in flight. It also reads actual fallback/importance metadata and
checks stable values and cancelled-frame recovery. Direct and reflected secondary-ray probes compare
coverage, instance identity and distance with the per-instance oracle. These
probes establish geometric correctness for their fixture, not full-scene
reflection-image quality.

Use `Tools/RunZorahFullRoam.ps1` with identical executable, shader, scene, camera,
resolution and warmup settings for each side. The runner records the mode in
its manifest. `Tools/AnalyzeRTRepresentation.py` consumes A1/B1/B2/A2 captures
and reports both construction savings and added preparation costs. Separate
quality captures export shadow, Deferred and final images; attaching their
debug observer makes them diagnostic runs, unsuitable as performance samples.

The default Zorah sample may have no enabled shadow-casting light. The benchmark
accepts an explicit `shadowPointLight` object with `position` and `intensity`;
use the same light on both sides and verify the captured shadow mask is not a
constant image. This sample does not contain a secondary-reflection pass.
`shadowOverrides` can select deterministic hard shadows for the separate image
comparison. Keep those settings identical on both sides and record them alongside
the image evidence. Alpha shadow sampling binds the same prepared frame's
texture metadata and image generation, including across texture migration.
