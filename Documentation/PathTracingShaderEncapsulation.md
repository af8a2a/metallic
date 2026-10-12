# Path-tracing shader encapsulation experiment (2026-10-12)

The proposed optimization has two separate objectives: remove dependencies
which an entry does not use, and reuse the compiler's work on stable modules.
Neither an `import` nor a wrapper function removes code which the entry calls
from its final SPIR-V. This document records an isolated prototype; production
shader files and compilation/session policy remain unchanged.

## Concrete dependency split

The production entry is `OpenPBRRayQueryPathTraceGuides`; the legacy
`ScenePathTraceGuides` is a separate catalog program. Both produce HDR path
tracing as well as seven guides. They are not independent guide-only passes.

The production dependency currently reaches the legacy implementation:

```text
OpenPBRRayQueryPathTraceGuides
  -> OpenPBRRayQueryPathTrace
     -> OpenPBRSurface -> SceneSurface
     -> ScenePathTrace
        -> traversal, cache support
        -> legacy pathTrace and scenePathTraceMain (not called by OpenPBR)
     -> OpenPBRPathVertex / OpenPBRRadianceCache
```

The first change should extract `ScenePathTraceCommon.slang` from
`ScenePathTrace.slang`: the prefix containing traversal/cache declarations and
`makeCacheContext`, while leaving `pathTrace` and `scenePathTraceMain` in the
legacy implementation. Both integrators explicitly include `SceneFiberPath`
because both use its fiber shading. OpenPBR then includes only Common.
Screen-space shadow and ray-query test consumers should likewise depend on
their actual traversal/camera API rather than the legacy integrator.

This eliminates unused legacy source from OpenPBR parsing/semantic checking.
It should be reported as a front-end dependency change unless final reachable
SPIR-V also changes. The legacy Guides entry actually calls legacy `pathTrace`;
this extraction alone cannot remove that integral computation from it.

## Slang module facade

A low-risk module prototype keeps all request-specific implementation in one
module and exports only a dispatch function with a built-in argument:

```slang
// Library: private types, resources, macros and early-guide callback stay here.
module PathTraceFacadePrototype;
__include "../Features/PathTracing/PathTraceFacadeImplementation.slang";
// Implementation exports:
// public void dispatchPathTraceGuides(uint3 dispatchThreadId) { ... }
```

```slang
// Program: same entry name, thread-group size and parameter ABI.
import PathTraceFacadePrototype;

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceGuidesMain(uint3 id : SV_DispatchThreadID)
{
    dispatchPathTraceGuides(id);
}
```

The isolated prototype moves the original entry body into that public function
without reordering its operations. It does not require making every field of
`ScenePathTracePush`, `HitInfo`, prepared closures or RNG state public. SHARC and
early-guide preprocessor definitions remain beside their implementation; macros
do not leak through module imports. This is a practical packaging boundary,
not a promise of reduced target optimization or Driver work.

After dependency pruning, a more reusable module API can separate:

- `SceneTraceTypes`: the shared wire/geometry contracts with explicit public
  fields and unchanged scalar layout.
- `SceneTrace`: traversal and visibility; page decoding and alpha candidate
  handling stay private.
- A separate radiance-cache adapter: SHARC modes must not force all traversal
  consumers to carry SHARC state/types.
- `OpenPBRPathIntegrator`: the actual reachable path algorithm and material
  adapter. Keep generated custom-material/STF adapters at the request boundary.
- A Guides policy invoked at the existing first authored-hit point. Avoid a
  circular import from Integrator back to Guides; use an explicit statically
  specialized provider or keep both in the request module.

Moving guide evaluation to the end of tracing is not equivalent to this
packaging experiment. Preserve independent guide RNG, authored hit data,
coverage/BLEND behavior, guide writes and closure live ranges at the existing
call site. Preserve the normal/TBN contract through all boundaries.

## What module reuse requires

`SlangCompiler.cpp` deliberately creates a fresh global/session for every
uncached request, because a session caches imported modules. Therefore simply
changing text includes to imports does not enable cross-request IR reuse in
the current RHI. The full SPIR-V disk cache already bypasses compilation on a
hit; module IR reuse would mainly help source misses and related permutations.

A serialized stable-module cache is a separate change. Its key must include
the pinned SDK, target/profile/capabilities, compiler and floating-point/debug
options, descriptor ABI, effective macros, all transitive sources and generated
material adapter identities. Load validated IR into a fresh request session;
do not retain a shared mutable session and silently reuse stale hot-reload or
SHARC/custom-material permutations. Precompile/export cost and final
link/code-generation cost must be recorded separately.

Pinned Slang 2026.18.2 tiny probes already demonstrate that source `import`,
text include and binary-only ordinary module import produce identical final
O1 bytecode, with one function and zero calls. Even the experimental embedded
target-IR route still produces one function after final linking/optimization.
Neither route supplies an opaque native Compute Pipeline library.

[Slang's compilation model](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/08-compiling.html)
describes linking and target code generation;
[module documentation](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/04-modules-and-access-control.html)
describes source and precompiled modules. These support separate front-end
reuse, not an assumed Driver cold-compilation benefit.

## Qualification and decision

Two real Guides prototypes completed with pinned Slang 2026.18.2, O1, mapped
descriptors and disabled debug information. Both passed
`spirv-val --target-env vulkan1.4`; actual depfiles identify the intended overlay
sources, and all original source/dependency hashes stayed unchanged.

| Final SPIR-V | Original | Common dependency pruning | Imported module facade |
| --- | ---: | ---: | ---: |
| Bytes | 6,940,612 | 6,940,612 | 6,940,172 |
| Module instructions | 447,484 | 447,484 | 447,466 |
| Function count / calls | 1 / 0 | 1 / 0 | 1 / 0 |
| Largest function instructions | 446,028 | 446,028 | 446,028 |
| Basic blocks / CFG edges | 58,220 / 74,427 | 58,220 / 74,427 | 58,220 / 74,427 |

Common pruning is **byte-for-byte identical** to the original SPIR-V.
The module facade changes only the opcode counts for `OpName` (60 → 42);
every non-`OpName` opcode count is unchanged. The public dispatch facade is
fully inlined at O1. These static counts do not prove numerical equivalence of
the changed module bytecode, but they establish that wrapping the large
implementation did not shrink its executable function body. The module
facade's SHA-256 is
`d0858c055793d934e3a84e55af4234832aaaf6d8f84c294e7e0c0e49e444cc8f`.

Next, nine independent CLI processes ran sequentially, three per arm, with
balanced ordering `[original, common, facade]`, `[common, facade, original]`,
`[facade, original, common]`. Each run reproduced its arm's exact validated
bytecode hash. No Shader/IR disk cache was used; OS filesystem caches were
allowed to remain warm. No Driver Pipeline creation or GPU dispatch occurred.
Median SDK-reported phases, in seconds, were:

| Compiler phase | Original | Common | Module facade |
| --- | ---: | ---: | ---: |
| `compileInner` | 19.633 | 19.840 | 19.332 |
| `SemanticChecking` | 0.770 | 0.791 | 0.791 |
| `generateOutput` | 18.767 | 19.072 | 18.540 |
| Downstream | 18.040 | 18.400 | 17.900 |

The facade's paired `compileInner` ratios were 0.943, 0.990 and 0.995;
the paired median is approximately **1% lower**, with a much larger first-block
effect. Original durations ranged 19.425–20.299 s, facade 19.138–19.438 s.
Three pairs establish a small observed difference, not a stable improvement
or a Native/GPU gain. Common pruning showed no measured compile reduction in
this sample. Neither arm reduces the ~18 s downstream optimization burden
substantially. These nested durations must not be added together.

Evidence and complete commands:

- [Current RHI baseline](../build-release/pathtrace-encapsulation-baseline/bench/report.json).
- [Common prototype and source identity](../build-release/pathtrace-common-prototype/summary.json),
  with its sibling `Manifest.json` preserving 359 artifacts.
- [Module facade prototype](../build-release/pathtrace-module-facade-prototype/Summary.json)
  and sibling `StaticDiff.json`; the actual thin entry/library/implementation
  are under that directory's `overlay/Shaders`.
- [Nine-process timing summary](../build-release/pathtrace-encapsulation-timing/summary.json)
  and sibling `Manifest.json`; each output has its command, logs and depfile.

Compare the same Guides request, macro set, O1, mapped descriptors, debug
disabled and exact source state. Record dependency/module-load, code-generation,
SDK downstream, total compiler wall, finalized SPIR-V identity and structure.
Do not sum nested compiler timers. The fresh current RHI baseline is 20.656 s
compiler wall, 716.876 ms module load, 19.752 s code generation and 18.842 s
SDK downstream; it matches the prior 6,940,612-byte Guides reference exactly.
This single baseline sample is diagnostic, not a paired speedup result.

Continue with independent sequential processes and interleaved variants before
claiming a stable front-end gain. Include-only pruning and import packaging
remain separate arms.
If final SPIR-V is byte-for-byte equal, do not claim Driver compile reductions
from either arm. If bytes change, qualify actual HDR plus seven guides and
ordinary GPU timings before adoption. Compile SHARC off/update/query, single
and multi-sample, custom-material and relevant descriptor/debug variants; test
hot reload and source invalidation as part of a module-cache change.

For a large reduction in cold native creation, the actual reachable material,
visibility and integrator IR must become easier to compile, or execution must
be split into independently compiled kernels with measured queue/bandwidth
costs. Encapsulation provides a useful boundary for those experiments; it does
not replace them. See [complexity evidence](ShaderCompilationComplexity.md).

The concrete decision is to treat Common extraction and a private library with
a narrow dispatch facade as dependency work, then test a validated stable-module
IR cache separately before adopting a performance claim. Do not merge the
facade merely to hide a large include and report that as a cold-compilation fix.
Fresh-session binary-IR loading is preferable to casually sharing one mutable
session across permutations. The pinned SDK's global session and its derived
objects are not generally thread-safe; use worker-local ownership or explicit
synchronization for any future reuse. Full-IR and target-IR precompile routes
need separate cost accounting, because final linking/optimization and native
Driver compilation still remain.
