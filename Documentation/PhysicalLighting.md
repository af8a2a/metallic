# Physical lighting

See [Color Pipeline](ColorPipeline.md) for ACEScg working-space and source/display contracts.

See [Checkpoint A implementation and evidence](PhysicalEnvironmentCheckpointA.md)
and [Checkpoint B physical environment](PhysicalEnvironmentCheckpointB.md)
for the celestial architecture migration and current validation results.

Select **Real-time / Physical Lighting** in Samples, then use the **Physical Lighting** panel to add, disable, edit or remove point and spot lights. Edit Sun and Moon in **Environment**. Local lights and exposure are saved in `world.lighting`; celestial state is saved in `world.environment.sun` and `.moon`. These settings survive reload, and Discard restores the saved state. Changing local-light units preserves the physical intensity (zero cannot be converted to a finite EV).

## Celestial ownership

[WorldEnvironment](../Source/Runtime/Environment/WorldEnvironment.h) is the authority for the only two infinite sources: Sun at slot 0 and Moon at slot 1. Production `LightingSettings` accepts only point/spot lights. Scene nodes cannot author generic directional emission.

Each celestial light stores an emitted world-space `direction`, linear Rec.709 `color`, perpendicular `illuminance` in lux, `angularRadius` in radians, and `enabled`. Shading points toward the source using `-normalize(direction)`. The renderer converts color into its ACEScg working space and packs two 48-byte `GPUCelestialLight` records. Disabled slots stay present with zero contribution. Sun and Moon default to disabled, preserving scenes that previously had no distant lighting.

`Scene::environmentSnapshot()` and `RenderWorld::environmentSnapshot()` return detached values with fixed `celestial[2]` slots. Celestial edits advance the celestial and lighting revisions; atmosphere/provider edits advance atmosphere and lighting revisions; HDRI edits advance lighting without changing the celestial or atmosphere revision. Weather remains reserved. A newly bound RenderWorld adopts the scene's environment. An explicit RenderWorld override applies to that scene or an unbound preview; independently resolved or `sourceOverride` scenes retain their own environment.

The environment **Source** selects HDRI or Physical atmosphere, saved as `world.environment.source`. HDRI retains lux and linear Rec.709 celestial color. Physical atmosphere uses Sun/Moon `topOfAtmosphereIrradiance` samples at 680/550/440 nm in W/m²/nm, attenuated at each shading position before conversion through XYZ to ACEScg. The atmosphere controls and double-precision planet center are saved in `world.environment.atmosphere`. Both providers feed the same radiance capture, SH, specular prefilter and environment PDF. Physical primary backgrounds add the explicitly attenuated celestial disks; lighting captures exclude them. Surface direct-light estimators retain the delta-light approximation, and the angular radius controls ray-traced shadow penumbrae. Camera aerial perspective composites `T * surface + in-scattering` before exposure. Astronomy, weather and clouds remain later milestones.

Legacy sidecars with one directional light explicitly migrate it to Sun, preserving color, direction, enabled state and physical intensity, and report a warning requesting a save. Incident-meter EV100 becomes `lux = 2.5 × 2^EV100`; SI/Lux values are unchanged. Bound source transforms are resolved before freezing the migrated world direction. Missing imported source nodes keep the migrated Sun disabled. Zero legacy directional lights retain the disabled default Sun. Multiple legacy directional lights, or a legacy directional light alongside an explicit `world.environment.sun`, fail loading and require explicit migration. Saved documents contain only local lights in `world.lighting.lights`.

## Imported glTF lights

Each active glTF node that references a point or spot light creates its own native `PunctualLight`, including separate instances of the same glTF light definition. Directional imports are ignored with a warning; the importer never silently adopts them as Sun or Moon. The document identifies local sources with `(sourceId, sourceNodeIndex)`, so repeated assets in a composed scene remain distinct. Names, linear colors, intensity in candela, range and cone angles are preserved. An omitted or zero range remains unbounded.

Imported virtual lights retain a source-node binding. Their position and emission direction follow the complete parent hierarchy and source mount, with glTF's local -Z emission axis. The Physical Lighting panel displays the current world-space pose and the import source; editing that pose stores a source-local offset rather than detaching the light. Parent/source visibility and the light's own Enabled switch both control its contribution. Manual virtual lights without an import binding continue to use independent world-space poses.

The source `LightComponent` remains hierarchy and import metadata, not a second emitter. Its Inspector edits the same native light properties, including intensity units and undo/redo; removing the native light makes the old Inspector read-only. Saved edits and removals survive reload, and removed imported lights are not recreated from their original defaults. Scene replacement uses the new document's lights instead of appending another copy. The common runtime light packing resolves the live source transform and emits a converted light only once across real-time shading, path tracing and clustered LightGrid consumers.

A scene resolved independently by a render-graph pass or selected through `sourceOverride` uses its own authored lights, adding only manually created, unbound world lights; it never borrows imported-light bindings from another scene.

## Units and calibration

Scene distances are **metres**. CPU light colors are linear Rec.709 multipliers. Imported point/spot intensities follow [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md) in candela. The source `si` convention and native Candela authoring unit represent the same local-light value; celestial authoring uses lux directly.

| Light | Authoring units | Conversion to transport units |
| --- | --- | --- |
| Sun / Moon | Lux | Perpendicular incident illuminance; no distance attenuation |
| Point | Candela, Lumens or local-light EV | `cd = lm / (4π)`; `cd = 2^EV` |
| Spot | Candela, Lumens or local-light EV | `cd = lm / Ωeffective`; `cd = 2^EV` |

Local-light EV follows Unreal's fixed reference-area convention, not the incident-light meter convention. See [Unreal physical lighting units](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-physical-lighting-units-in-unreal-engine) and local sources `Engine/Source/Runtime/Engine/Private/Components/LocalLightComponent.cpp`, `PointLightComponent.cpp`, and `Engine/Source/Runtime/RenderCore/Public/RenderUtils.h`.

Point and spot illumination is `I / distance²` before the material's cosine term. Range 0 is unbounded; an optional finite range multiplies illumination by `max(1 - (distance/range)^4, 0)` to smoothly cut off at the authored range. The singularity alone is capped at a 1 mm distance. Spot attenuation is a squared linear interpolation in cosine space between outer and inner half-angles. Lumen normalization integrates this actual profile: `Ωeffective = 2π [(1 - cos(inner)) + (cos(inner) - cos(outer))/3]`. It is not a uniform-cone approximation.

Manual exposure follows Unreal's `EV100ToLuminance(1, EV)` saturation-luminance convention: `Lmax = 2^EV cd/m²`, display scale `1/Lmax`. It is applied after light transport, including SHaRC and denoiser outputs; it does not alter light powers or the environment PDF. This is a manual exposure setting, not an automatic exposure algorithm.

An HDR file is not inherently calibrated. To compare it against physical lamps, its linear pixels must represent cd/m² after multiplying by Environment Intensity. This multiplier is therefore also the HDR calibration scale. Arbitrary HDRIs and the procedural preview sky are not guaranteed to have an absolute physical calibration.

## Rendering paths

- `SceneRealtimeLightingPass` evaluates OpenPBR direct lighting with one primary ray and shadow rays for fixed Sun/Moon plus point/spot lights. Diffuse environment GI is a deterministic nine-coefficient SH lookup. It is independent of the diagnostic `VisibilityBufferPass` and requires Vulkan ray-query support. It has no stochastic accumulation, multi-bounce transport, environment specular prefilter or indirect visibility solver; SH GI is an unoccluded low-frequency diffuse approximation modulated by material occlusion.
- `ScenePathTracePass` evaluates both celestial slots separately and selects one local-light sample per eligible path vertex through the shared GPU ReGIR selector. The standard/OpenPBR paths, SHARC cache query/update paths, and RTXCR hair direct lighting apply the RIS inverse source weight only to the selected local contribution. These are delta-light next-event estimates (MIS weight 1), separate from the existing environment solid-angle PDF/MIS sampler. They do not sample irradiance SH as radiance or add a second full-light contribution. OpenPBR preserves transmission-aware shadow transport; the standard material path uses its direct-light BRDF and opaque shadow queries.
- `SceneRTXDIPass` defaults to `lightSource = "scene"`, collecting imported and manually authored point/spot lights into its ReSTIR DI reservoirs. Sun/Moon direct lighting is evaluated separately and never enters local ReGIR or local reservoirs. `lightSource = "bench"` substitutes native benchmark point-light records for that pass; only this mode uses the benchmark count, animation and intensity controls. The bundled RTXDI sample selects bench mode explicitly. This remains a Metallic-native implementation modeled after RTXDI Grid ReGIR/ReSTIR DI, not a copied or linked RTXDI SDK integration.
- Environment PDF construction, SH projection and cosine convolution run entirely on the GPU. SH covers bands l=0,1,2 (nine RGB coefficients), with convolution factors π, 2π/3, π/4. Consumers evaluate irradiance in the rotated environment frame and apply diffuse albedo/π exactly once. The procedural sky is also projected on the GPU.

Shader authors can use the typed [spherical harmonics GI API](SphericalHarmonics.md) for L1/L2 projection, arithmetic, interpolation, rotation and irradiance evaluation. The environment float4 buffer layout and physical normalization are unchanged.

- Light buffers are immutable per update and retained for in-flight frames. Resolved light revisions invalidate path-tracing accumulation/radiance caches and RTXDI reservoir history, including light removal and compact-index changes. Imported native lights and manually added world lights are additive; converted source metadata does not emit a duplicate contribution. Source visibility and virtual-light enabled state exclude inactive lights from the buffer. RTXDI applies physical exposure consistently to direct color, linear NRD signals and emissive/background before the composite's artistic exposure.

## Shared full-scene ReGIR selection

`SceneLightResources` owns compact point/spot records and shared GPU importance resources. Binding 50 is the 64-byte-per-record `gPunctualLights` buffer, whose first record contains count and exposure. Binding 52 is the ReGIR header and RIS slots; binding 53 is the global local-light power PDF mip chain. `PunctualLightSampling.slang` is consumed by both path tracing and RTXDI. GPUScene/LightGrid source-slot IDs must not be used as indices into these compact buffers. `EnvironmentLightingSubsystem` owns the separate two-record celestial buffer at binding 55, exposed through `EnvironmentResourceParameters::celestialLights`; it has no count/exposure header.

The global local-light proposal is built on the GPU from actual point/spot parameters. Weights integrate candela over the corresponding angular emission profile. Sun/Moon never enter this PDF, the ReGIR grid or compact punctual buffers. Grid ReGIR draws power-PDF candidates and resamples them using conservative cell-volume targets. These targets preserve support near spot and finite-range boundaries rather than rejecting lights from a cell-center test. The selected light is evaluated with its exact physical attenuation at the receiving surface.

Selection outside the grid falls back to the global power PDF, and a zero-power global distribution uses uniform selection over available records. An empty RIS slot in a valid cell contributes zero, without drawing a second fallback sample. This preserves the estimator's probability mass. A zero-light scene returns no local contribution. Neither proposal construction nor use requires GPU readback, CPU preintegration, or camera-visible light counts.

The source collection is **camera-independent**: full-scene lights can illuminate secondary vertices and reflections outside the primary view. The camera-filtered DrawSet/LightGrid lists described below serve raster work and are not the transport proposal. The real-time physical pass continues evaluating its light set deterministically. See [RTXDI / ReSTIR DI](RTXDI.md) for reservoir flow, benchmark controls, limits and validation commands.

Cancelled GPU recordings mark the sampling resources for reconstruction. A retry recreates the PDF/grid instead of trusting texture-layout transitions that were only recorded, including cancellation during the first build or after resizing.

## DrawSet light candidates

`GPUSceneSubsystem::beginFrame` synchronizes scene and virtual world lights, including worlds without a Scene. Shared `SceneLightRecord` packing resolves imported native-light bindings against the current source pose and suppresses duplicate imported emission. It preserves source slots and the physical `GPUPunctualLight` data; disabled, invalid, black and zero-intensity sources retain their slots but do not enter `GPUSceneDrawSet::lights`.

`GPUScene::prepareView` collects candidates independently for each View/frame slot, alongside coarse mesh collection. It follows Unreal's `ComputeLightVisibility` (`Renderer/Private/SceneVisibility.cpp`) and `FSphere::FromCone` (`Core/Public/Math/Sphere.h`): point lights use their attenuation sphere; spot lights use a conservative sphere enclosing their radial range and outer cone. Non-normalized inward frustum planes are supported, tangency is retained, and invalid/zero planes are skipped. Neither mesh visibility predicates nor HZB occlusion remove lights. There is no screen-size or brightness threshold.

`GPUSceneSubsystem::visibleLights(view, frameSlot)` returns two deterministic local source-ID lists:

- `localLights`: finite-range point/spot candidates for the clustered LightGrid.
- `unboundedLocalLights`: range-zero point/spot lights, kept separately because a finite grid bound cannot represent their influence.

Resolve each ID through `light(id)` to obtain its source indices/object, unmodified GPU payload and coarse `boundingSphere`. The sphere is not a replacement for the original position/range/cone used in future grid intersection. The collection has its own `lightGeneration`/`lightRevision`; parameter edits preserve IDs, while source-slot topology changes invalidate them. Check `visibleLights` or `lights.validFor(...)` before using a snapshot: light-only changes expire light lists without changing mesh DrawSet revisions or GPU allocations. GPUScene's geometric HZB state is independent; the renderer's existing global radiance-history invalidation/camera-cut policy remains unchanged.

VisibilityBuffer and GPUDrivenStreamAsset provide the same camera used for mesh culling. VisibilityBuffer supports frozen and orthographic cameras; disabling `instanceFrustumCull` disables only the coarse light frustum rejection, not clustered intersection. Environment PDF and SH are not local-light candidates.

## GPU clustered LightGrid

After coarse collection, both raster paths call `GPUSceneSubsystem::recordLightGrid` before traversal/culling/raster work. `lightGrid(view, frameSlot)` exposes the current GPU buffers and parameters; it returns null after the view is re-prepared, its light revision changes, its recording is cancelled, or its shader program is replaced. The RTAS diagnostic path does not construct a raster grid. Raster shading evaluates Sun/Moon independently, then the clustered bounded locals and unbounded locals. Path tracing continues using the camera-independent local proposal.

The default is **64×64 screen pixels, 32 depth slices, 64 stored lights per cell**. `ClusterLightGridDesc` exposes these limits and the camera to callers. Perspective slices use the Unreal-style mapping `z = log2(depth * B + O) * S`, with `B=1/near`, `O=0`, and `S=sliceCount/log2(far/near)`; this chooses ordinary logarithmic spacing without Unreal's centimetre-scale near bias. Orthographic views use linear spacing. Depth is positive view-space distance, independent of reversed-Z. Partial edge tiles are clamped to the actual viewport; the frozen culling camera keeps its original aspect ratio even when the viewport resizes.

`ClusterLightGrid.slang` dispatches one 64-thread group per cell. Eight view-space tile/depth corners define a conservative AABB. Threads test bounded local candidates against their original range spheres; spot lights additionally use Unreal's conservative cone/AABB separating-plane test. The emission point, radial range, outer cone and physical intensity are preserved. No CPU cell-light calculation, GPU readback, receiver-depth/HZB rejection or screen-size/brightness heuristic is used by the runtime.

GPU data contract:

| Buffer | Layout |
| --- | --- |
| `parameters` | One 128-byte `ClusterLightGridParams` |
| `lights` | 64-byte physical records at stable GPUScene source slots, **no metadata header** |
| `candidates` | Source indices: bounded locals, then unbounded locals; no celestial records |
| `cells` | 16-byte `{offset, count, overflow, totalCount}`, indexed `(z*gridY+y)*gridX+x` |
| `lightIndices` | Cell source-index lists; fixed segment `cellIndex*maxLightsPerCell` |

`ClusterLightGridCommon.slang` provides `clusterLightGridLookup`, `clusterLightGridLocalLightIndex`, and global-list helpers for unbounded point/spot lights only. The former directional-count word in the parameter ABI is reserved zero. Overflow never silently truncates lighting: a cell with more matches than capacity returns the **complete bounded-local candidate list** through the lookup helper. Outside the grid's XY/depth domain, the helper also falls back to that candidate list instead of incorrectly clamping into the last cell. This is still a view-filtered list, not a substitute for a full-scene query outside that view or on secondary rays. Unbounded locals must be evaluated once in addition to local lookup. Stored cell order is GPU-atomic order and is not stable; source IDs and candidate ordering are stable. Do not pass these source indices to the compact/header-prefixed `gPunctualLights` buffer.

Each View/frame slot owns independently versioned resources, reused only after its tracked `RenderFrameContext` submission completes; resizing grows allocations and retains old buffers for in-flight commands. Legacy commands without a frame receive independent buffers and a separate program/descriptor table per recording; their submission state retains these until command reset (the RHI caller must finish GPU work before resetting). Output buffers end in `ShaderRead`. Every cell header is rewritten, including empty scenes, so unused index storage need not be cleared. Invalid/degenerate cameras and oversized allocations fail before dispatch (256 MiB cell/header budget per slot, depth slices 1–256, capacity 1–1024). Shader reload stages replacement programs for every active grid and commits only after all subsystem preparations succeed; old programs remain retained by their frames or command submissions.

The design references local Unreal `Renderer/Private/LightGridInjection.cpp`, `Shaders/Private/LightGridInjection.usf`, and `RenderCore/Public/RenderUtils.h`. This implementation deliberately uses fixed cell segments and explicit fallback instead of UE's linked-list pool, compaction and overflow feedback.

## Validation

`WorldEnvironment.*` covers fixed slots, detached snapshots, domain revisions, invalid values, glTF directional-ignore policy, legacy migration, Sun/Moon persistence and transactional migration rejection. `Photometry.UnitsAndValidation` and `SceneEditing.PhysicalLightingRoundTrip` cover units, negative EV, validation, local imports and persistence. RHI tests `photometric_gpu_units_falloff_sh` and `photometric_realtime_render` cover GPU inverse-square attenuation, celestial invariance, spot cutoff, unit equivalence, exposure, constant-environment irradiance and live light edits.

`SceneEditing.ImportedglTFLights*` covers per-node native-light creation, physical parameters, saved edits/deletions, legacy document loading, composed-source identities, transactional loads/edits, stale-setting rejection and orphaned bindings. RHI tests `imported_virtual_light_*` cover single-emitter ownership, live hierarchy/visibility binding, GPUScene candidate collection and independently resolved scene/source-override isolation.

The `gpu_scene_light_*` tests cover source collection/lifetime, independent light revisions, per-view/frame-slot snapshots, conservative point/spot culling, perspective/orthographic camera planes and light-only world synchronization.

The `cluster_light_grid_*` tests verify CPU camera/depth contracts and GPU-produced point/spot cell lists, actual shader lookup and fallback, source-slot holes, global-light separation, growth/shrink reuse, cancellation, View/frame-slot isolation and staged shader reload.

The `regir_virtual_lights_*` GPU tests cover physical local-light power-PDF construction, point/spot estimates, global fallback, edits/deletion, zero-weight RIS slot probability mass, cancelled-recording retry, and standard PT/OpenPBR PT/RTXDI temporal rendering. Run `MetallicRHITests.exe --rhi-validation --gtest_filter="*regir_virtual_lights_*"` in a configured test build. See [RTXDI validation commands](RTXDI.md#build-and-run) for the shader, PDF-size, grid-layout and sample-preview filters. Test descriptions identify coverage; current execution results belong in the checkpoint evidence report.
