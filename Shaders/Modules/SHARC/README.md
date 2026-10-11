# SHARC Slang module

`import SHARC; using Metallic.SHARC;` exposes the renderer radiance-cache API.
The module has no resources or parameter root of its own: callers supply typed
`RWStructuredBuffer` fields in `SHARCParameters`. `PathTraceStageParameters.h`
imports the same owner for the maintenance handles.

- `Config.slang`: typed constants and hash key/index aliases.
- `Types.slang`: hash parameters and the two 16-byte cache records.
- `HashGrid.slang`: spatial keys, logarithmic levels, probing and atomic insertion.
- `Cache.slang`: update propagation, packed radiance, query, temporal resolve and eviction.

This is a native Slang port of Metallic's RTXGI v2 SHaRC 1.8.3 adaptation.
The pre-migration files in `../../ThirdParty/RadianceCache/Sharc/` remain unchanged
as a reference snapshot, with their NVIDIA copyright/license notices. Each ported
source retains its NVIDIA notice. Production shaders do not include that snapshot.

The supported configuration matches the existing renderer: 64-bit keys and atomic
compare-exchange, normal bits, 16-slot buckets, the existing empty-slot termination
rule, depth-two update propagation with cache resampling, and adjacent-level resolve
blending. Radiance/sample count remain fp16 halves packed in `uint2`; both cache
records stay 16 bytes. Responsive lighting and SH encoding remain unsupported.
Unused compact-key and lock-buffer branches are not part of this module.

Library configuration uses constants, aliases and typed resource parameters rather
than textual resource/prefix macros. `SHARC_UPDATE` and `SHARC_QUERY` select pass
code only. Update functions always perform updates; query passes must not call them.
`SHARCState` is the update state independently of importer defines. Query-only
use is eliminated by the compiler. The OpenPBR component cache uses the same
hash-grid and radiance helpers without depending on pass-local macro propagation.

`ScenePathTrace.slang` imports the module for update/query, and
`SceneSharcMaintenance.slang` imports it for clear/resolve. Public entry points use
lower-camel function names, types preserve the SHARC acronym, and implementation
helpers are internal. The compiler tracks the module and its `__include` files as
transitive cache/hot-reload dependencies.

Validation: compile clear/resolve plus plain/update/query paths, including guides
and custom materials; run `MetallicRHITests --filter sharc` with mapped and
native (`--rhi-bindless`) resources. These GPU tests cover queried cache transport,
cache misses, guide equivalence, camera movement, history cancellation/retry,
maintenance bounds/eviction and module ownership/mode isolation.
