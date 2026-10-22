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

`stream_rt_representation_lifecycle` exercises stable frames, sharing, merging,
cache return after temporary disuse, tight budgets, cancellation and retirement
with two frames in flight. Direct and reflected secondary-ray probes compare
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
