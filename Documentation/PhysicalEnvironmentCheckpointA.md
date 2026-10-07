# Checkpoint A — Celestial Lighting Architecture

Implementation and validation recorded on 2026-10-06 against the user-supplied
`D:/Metallic_Physical_Environment_Roadmap.md` (M0–M2).

The architecture checkpoint is implemented: production lights are local point/spot
lights, Sun/Moon have fixed environment slots, and raster, path tracing, ReSTIR DI
and ray-traced shadows consume the celestial abstraction. Physical atmosphere,
astronomy, day/night, weather and finite-disk direct-light sampling remain later
milestones. Existing procedural sky and HDRI precomputation remain in use.

## Ownership and resource contract

- [WorldEnvironment](../Source/Runtime/Environment/WorldEnvironment.h) owns Sun
  (slot 0) and Moon (slot 1). Detached snapshots carry celestial and lighting
  revisions; atmosphere/weather revisions are reserved at zero.
- Scenes persist `world.environment.sun` and `.moon`. RenderWorld adopts the
  bound scene's state unless explicitly overridden. Independently resolved scenes
  retain their own environment. Editor controls expose enabled state, direction,
  linear Rec.709 color, illuminance in lux and angular radius.
- [GPUCelestialLight](../Source/Runtime/Render/Environment/CelestialLighting.h)
  is a 48-byte C++/Slang record. The environment buffer always contains two records,
  including disabled slots, and is exposed at binding 55 through
  `SceneResourceParameters::environment.celestialLights`. The scene parameter block
  remains 448 bytes; the nested environment handle uses its former tail padding.
- Celestial irradiance preserves the existing lux transport scale and converts
  linear Rec.709 color to ACEScg once. Disk radiance is prepared from irradiance
  and angular radius; current direct estimators remain delta-light estimators.
  Ray-traced celestial shadows use the authored angular radius.
- EnvironmentLightingSubsystem publishes immutable, frame-retained buffers and
  reuses identical contents through a bounded cache. Passes receive their own
  publication. PT and ReSTIR history compare actual celestial records, avoiding
  resets when different scenes alternate or a cached allocation is evicted.
- GPUScene and ClusterLightGrid contain only bounded/unbounded local lights.
  Compact punctual buffers, the local power PDF and ReGIR contain no Sun/Moon.
  Raster evaluates both celestial slots separately; PT and ReSTIR DI evaluate
  them outside the local-light proposal/reservoir domain.
- Shadow selection has fixed Sun/Moon entries followed by local source slots.
  Tagged identities distinguish celestial from local sources. Auto selects Sun,
  then Moon, then a local light. SIGMA history responds to source/radius changes.

See [Physical lighting](PhysicalLighting.md) and
[Ray-traced shadows](ScreenSpaceShadows.md) for authoring, units and shader details.

## Migration

| Input | Result |
| --- | --- |
| No legacy directional light | Disabled default Sun/Moon; existing no-distant-light scenes remain unlit by celestial sources |
| One legacy sidecar directional light | Explicit migration to Sun with a save warning; preserves world direction, color, enabled state and physical intensity |
| Multiple legacy directional lights | Transactional load failure; requires explicit migration |
| Legacy directional plus explicit Sun | Transactional load failure; prevents ambiguous ownership |
| glTF directional light | Ignored with a warning; never silently becomes Sun |

Legacy incident EV100 is converted using `lux = 2.5 * 2^EV100`; SI/Lux values are
unchanged. Source transforms are resolved before freezing the migrated direction.
Missing imported bindings leave the migrated Sun disabled. Saved local-light
arrays contain only point/spot records. The OpenPBR LookDev asset and its generator
now author Sun directly.

The old graph `shadowAngularRadius` property is ignored; its authored replacement
is the Sun/Moon angular radius in radians. Explicit local shadow selection indices
need the fixed two-slot prefix. Low-level directional parsing/unit conversion is
retained only for compatibility migration, not as a production emitter.

## Validation

Existing Ninja Release trees were reused without changing their compiler or SDK
options. `build-scheduling-release` supplies Scene/RHI/LookDev; the existing
NRD-enabled `build-pass-stages-nrd` supplies NRD tests. Commands below run from the
repository root in an x64 Visual Studio developer shell.

```powershell
cmake --build build-scheduling-release --target MetallicSceneTests MetallicRHITests LookDev
cmake --build build-pass-stages-nrd --target MetallicNRDTests
ctest --test-dir build-scheduling-release -R '^MetallicSceneTests$' --output-on-failure

.\build-scheduling-release\tests\MetallicRhiTests.exe --gtest_filter='*celestial*:*gpu_scene_light_collection:*gpu_scene_light_frustum:*cluster_light_grid*:*regir_virtual*:*render_graph_light_grid_debug*:*photometric*:*lookdev_render_paths' --output-dir build/celestial-checkpoint-a/after
.\build-scheduling-release\tests\MetallicRhiTests.exe --gtest_filter='*celestial_fixed_slots_named_environment_abi:*celestial_publication_override_reuse:*photometric_gpu_units_falloff_sh' --output-dir build/celestial-checkpoint-a/verified

.\build-pass-stages-nrd\tests\MetallicNRDTests.exe --gtest_filter='NRDPlan.*:NRDRayTracingGPU.*:NRDGPU.SigmaLitOccludedResetResizeAndDiscard'
.\build-pass-stages-nrd\tests\MetallicNRDTests.exe --gtest_filter='NRDRayTracingGPU.RayTracedShadowOcclusionAndHistory'
.\build-pass-stages-nrd\tests\MetallicNRDTests.exe --gtest_filter='NRDWorkingColor.RTXDIRelaxSceneChromaticity'
.\build-scheduling-release\Source\LookDev.exe --smoke-test --skip-shader-warmup --sample lookdev-vbuffer

ctest --test-dir build-scheduling-release -R '^(MetallicWorkingColorAudit|MetallicShaderRegistryUsageAudit)$' --output-on-failure
```

| Check | Result and scope |
| --- | --- |
| C++ targets above | Built successfully |
| Full Scene suite | Passed: 139 tests, 9 skips for disabled USD or unavailable optional scene fixtures |
| Targeted RHI suite | 22 unique tests passed across the broad run and corrected focused rerun; includes local grids/ReGIR, photometry, named-resource ABI, publication lifetime and rendering |
| Celestial raster/PT | 32 frames for each of disabled/Sun/Moon/both at 128²; finite HDR, independent slots and additive contribution within test tolerance |
| NRD | 10 unique tests passed, including fixed-slot selection, lux preservation, local-only resources, ownership overrides, SIGMA and RTXDI/RELAX chromaticity |
| Moon shadow follow-up | Passed with Sun disabled: slot 1 selection, flat receiver visibility and shadows from offscreen geometry; existing Sun radius/history cases also passed |
| LookDev editor smoke | Recorded, submitted and presented a comparison frame successfully |
| Direct Slang compilation | Eight affected shader/probe targets compiled successfully |
| ShaderRegistry usage audit | Passed |
| WorkingColor audit | Existing failure: `Shaders/Modules/RTXCRHair/BSDFUtils.slang` uses Rec.709 luminance coefficients; the offending line is unchanged from HEAD |

The initial broad RHI run passed 20 tests and exposed two test-fixture issues:
an SH expectation used Rec.709 rather than the renderer's AP1 working color, and
the ABI probe used a device without the required bindless heap. Both were corrected;
the focused rerun passed all three selected tests. Windows sandbox path
canonicalization also rejected some Scene/report operations; rerunning the same
tests outside that sandbox succeeded.

Direct Slang checks used `External/slang/bin/slangc.exe`, include paths
`Shaders/Modules` and `Shaders/Interop`, and
`-target spirv -profile glsl_460 -stage compute`. Outputs are under
`build/celestial-checkpoint-a/`:

| Source | Entry point | Additional options / output |
| --- | --- | --- |
| `tests/rhi/shaders/PhotometricProbe.slang` | `photometricProbeMain` | `PhotometricProbe.spv` |
| `tests/rhi/shaders/CelestialLightingProbe.slang` | `main` | `CelestialLightingProbe.spv` |
| `Shaders/Features/Lighting/SceneRealtimeLighting.slang` | `sceneRealtimeLightingMain` | `-I External/RTXCR/Include`; `SceneRealtimeLighting.spv` |
| `Shaders/Features/ReSTIR/SceneRTXDI.slang` | `sceneRtxdiMain` | `SceneRTXDI.spv` |
| `Shaders/Features/PathTracing/OpenPBRRayQueryPathTrace.slang` | `openPbrRayQueryPathTraceMain` | `OpenPBRRayQueryPathTrace.spv` |
| `Shaders/Features/Lighting/ScreenSpaceShadows.slang` | `rayTracedShadowsMain` | `ScreenSpaceShadows.spv` |
| `Shaders/Features/Lighting/SceneRealtimeLighting.slang` | `sceneRealtimeLightingMain` | `-D METALLIC_REALTIME_DEFERRED=1 -D METALLIC_DEFERRED_LIGHT_GRID=1`; `SceneDeferredLighting.spv` |
| `Shaders/Features/VisibilityBuffer/VisibilityBufferDeferred.slang` | `visibilityBufferDeferredMain` | Same two deferred defines; `VisibilityDeferred.spv` |

These eight checks generated SPIR-V successfully; they do not constitute a full
production permutation warmup.

## Legacy Sun HDR comparison

Before changing the renderer, `lookdev_render_paths` captured the legacy
directional-light scene. The migrated scene was captured with the same camera,
settings, 256² resolution and 32 accumulation frames. Application shader/PSO caches
were reused; changed shaders compiled on demand. These are correctness captures,
not performance measurements or a complete shader-warmup run.

| Path | Before vs migrated Sun |
| --- | --- |
| DeferredOnly | Linear HDR byte-for-byte identical; relative RMSE and maximum absolute RGB error are zero |
| PathTraceOnly | Mean RGB changes from 0.2727779171 to 0.2729156331 (about +0.0505%); relative pixel RMSE 4.8971%, mean absolute RGB error 0.00548645 |

The PT pixels differ after moving celestial evaluation out of local ReGIR sampling;
the random sequence and local proposal domain changed. The small mean difference
and independent/additive tests support preserved lighting semantics, but a 32-frame
stochastic comparison does not establish pixel-exact equivalence or a converged
visual error bound. Deferred before/after and PT output images were inspected.
No separate legacy SIGMA shadow A/B capture was taken; new GPU shadow behavior
is covered by the Sun/Moon geometry, radius and history tests.

Raw evidence remains in ignored local output, outside source control:

- [HDR comparison metrics](../build/celestial-checkpoint-a/LegacySunComparison.json)
- [Legacy deferred image](../build/celestial-checkpoint-a/before/DeferredOnly.png)
- [Migrated deferred image](../build/celestial-checkpoint-a/after/DeferredOnly.png)
- [RHI run](../build/celestial-rhi.log) and [focused verification](../build/celestial-rhi-verified.log)
- [Scene CTest](../build/celestial-scene-unsandboxed.log)
- [NRD tests](../build/celestial-nrd-tests.log), [Moon shadow](../build/celestial-nrd-moon.log) and [RTXDI/RELAX](../build/celestial-rtxdi-color.log)
- [Editor smoke](../build/celestial-editor-smoke.log) and [audits](../build/celestial-audits.log)

Editor control persistence is covered by SceneDocument round trips and compiled
editor integration; interactive editing/save/discard was not separately exercised.
No full Zorah scene residency, long-duration temporal run or atmosphere validation
is claimed by this checkpoint.
