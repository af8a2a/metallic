# Physical lighting

See [Color Pipeline](ColorPipeline.md) for ACEScg working-space and source/display contracts.

See [Checkpoint A implementation and evidence](PhysicalEnvironmentCheckpointA.md)
and [Checkpoint B physical environment](PhysicalEnvironmentCheckpointB.md)
for the celestial architecture migration and current validation results.

Select **Real-time / Physical Lighting** in Samples, then use the **Physical Lighting** panel to add, disable, edit or remove point, spot and rect lights. Edit Sun and Moon in **Environment**. Local lights and exposure are saved in `world.lighting`; celestial state is saved in `world.environment.sun` and `.moon`. These settings survive reload, and Discard restores the saved state. Changing point/spot units preserves the physical intensity (zero cannot be converted to a finite EV).

## Celestial ownership

[WorldEnvironment](../Source/Runtime/Environment/WorldEnvironment.h) owns the two infinite sources, Sun at slot 0 and Moon at slot 1. `LightingSettings` holds the local point, spot and rect lights.

Each celestial light stores an emitted world-space `direction`, linear Rec.709 `color`, perpendicular `illuminance` in lux, `angularRadius` in radians, and `enabled`. Shading points toward the source using `-normalize(direction)`. The renderer converts color into its ACEScg working space and packs two 48-byte `GPUCelestialLight` records. Disabled slots stay present with zero contribution. Sun and Moon default to disabled, preserving scenes that previously had no distant lighting.

`Scene::environmentSnapshot()` and `RenderWorld::environmentSnapshot()` return detached values with fixed `celestial[2]` slots. Celestial edits advance the celestial and lighting revisions; atmosphere/provider edits advance atmosphere and lighting revisions; HDRI edits advance lighting without changing the celestial or atmosphere revision. Weather remains reserved. A newly bound RenderWorld adopts the scene's environment. An explicit RenderWorld override applies to that scene or an unbound preview; independently resolved or `sourceOverride` scenes retain their own environment.

The environment **Source** selects HDRI or Physical atmosphere, saved as `world.environment.source`. HDRI retains lux and linear Rec.709 celestial color. Physical atmosphere uses Sun/Moon `topOfAtmosphereIrradiance` samples at 680/550/440 nm in W/m²/nm, attenuated at each shading position before conversion through XYZ to ACEScg. The atmosphere controls and double-precision planet center are saved in `world.environment.atmosphere`. Both providers feed the same radiance capture, SH, specular prefilter and environment PDF. Physical primary backgrounds add the explicitly attenuated celestial disks; lighting captures exclude them. Surface direct-light estimators retain the delta-light approximation, and the angular radius controls ray-traced shadow penumbrae. Camera aerial perspective composites `T * surface + in-scattering` before exposure. Astronomy, weather and clouds remain later milestones.

In **Environment → Atmosphere**, **Aerial perspective start depth (km)** sets how far each camera ray travels before air contributes to the visible surface. The scene field `world.environment.atmosphere.aerialPerspectiveStartDepthKm` defaults to 0 km. The Sponza GI / Physical Atmosphere sample sets it to 0.1 km (100 m) so its covered arcade retains the surface lighting from the courtyard. **Aerial LUT range (km)** controls the distance covered by precomputed samples.

Under **Environment → Sun** or **Moon**, enable **Cast shadows on atmosphere** to let opaque geometry shadow that light's single scattering in the camera view. The scene field `castShadowsOnAtmosphere` defaults to false for each light. Sponza GI / Physical Atmosphere enables it for the Sun. To inspect shadows inside its short arcade, set the aerial perspective start depth to 0 km.

In a realtime graph, enable `publishWorldShadows` on `VirtualShadowMapLightsPass` and `directionalAtmosphereShadows` on `VisibilityBufferDeferredPass`, and connect their `worldShadowFaces` pins. The current view samples the Sun and Moon VSMs along its aerial and visible-sky rays. The atmosphere's extinction and multiple scattering still come from the medium. Shared environment lighting and reflection captures use the global sky. A light without an admitted shadow map retains its unoccluded contribution.

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
| Rect | Nits (cd/m²) | Uniform emitted luminance across the rectangle |

Local-light EV follows Unreal's fixed reference-area convention, not the incident-light meter convention. See [Unreal physical lighting units](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-physical-lighting-units-in-unreal-engine) and local sources `Engine/Source/Runtime/Engine/Private/Components/LocalLightComponent.cpp`, `PointLightComponent.cpp`, and `Engine/Source/Runtime/RenderCore/Public/RenderUtils.h`.

Point and spot illumination is `I / distance²` before the material's cosine term. Range 0 is unbounded; an optional finite range multiplies illumination by `max(1 - (distance/range)^4, 0)` to smoothly cut off at the authored range. The singularity alone is capped at a 1 mm distance. Spot attenuation is a squared linear interpolation in cosine space between outer and inner half-angles. Lumen normalization integrates this actual profile: `Ωeffective = 2π [(1 - cos(inner)) + (cos(inner) - cos(outer))/3]`. It is not a uniform-cone approximation.

Manual exposure follows Unreal's `EV100ToLuminance(1, EV)` saturation-luminance convention: `Lmax = 2^EV cd/m²`, display scale `1/Lmax`. It is applied after light transport, including SHaRC and denoiser outputs; it does not alter light powers or the environment PDF. This is a manual exposure setting, not an automatic exposure algorithm.

An HDR file is not inherently calibrated. To compare it against physical lamps, its linear pixels must represent cd/m² after multiplying by Environment Intensity. This multiplier is therefore also the HDR calibration scale. Arbitrary HDRIs and the procedural preview sky are not guaranteed to have an absolute physical calibration.

## Spotlights and IES profiles

In **Scene List → Physical Lighting**, use **Add spot** to create a spotlight.
Set its emission direction, range and inner and outer half-angles. In the
Scene Graph, authored spotlights appear under **Fog & Lights** unless the scene
document assigns them to a source node. **Bounds** shows their outer cone and
finite range.

Use **Load .ies** on a point or spot light in Physical Lighting or Inspector.
Metallic reads Type C LM-63 files with `TILT=NONE` and interpolates the measured
angles, including the file's horizontal symmetry. Profile roll turns the
distribution about the emission axis. The scene stores the `iesProfile` path
relative to the scene document when possible, together with `iesRollDegrees`
and `useIESIntensity`. Keep the `.ies` file with the scene when moving assets.

By default, the profile scales the authored light intensity by a normalized
angular distribution. **Use measured peak** takes the peak candela from the
file, including its photometric multipliers. A spotlight's cone and range
continue to limit the result in either mode. Set intensity in candela or use
the measured peak when calibrating a light to an IES file. The Lumens control
uses the underlying point or spot conversion before profile modulation.

The profile modulates direct punctual illumination in realtime shading, path
tracing, RTXDI, Lumen Surface Cache and volumetric fog. Spatial culling and
shadow coverage continue to use the light's geometric cone and range.
[ANSI LM-63 overview](https://blog.ansi.org/ansi/standard-file-photometric-data-ies-lm-63-19/)
· [Autodesk Type C file guide](https://help.autodesk.com/cloudhelp/2024/ENU/AutoCAD-Core/files/GUID-45CAAF1C-7C9D-49A7-B18D-00CA5E2ED380.htm)

## Rectangular area lights

In **Physical Lighting**, select **Add rect light**. Position and emission direction are in world space. Width, height and optional attenuation range are in metres. Roll rotates the rectangle about its emission direction. Zero range is unbounded. The scene saves the settings in `world.lighting.rectLights`. At constant luminance, increasing the emitting surface also increases its total light output.

The ray-query realtime pass and Visibility Buffer Deferred integrate unoccluded Lambert diffuse and isotropic GGX base reflection for OpenPBR surfaces. The rectangle emits from its front face. The optional center-based range uses Metallic's quartic cutoff. The solid-angle integral accounts for distance falloff. Each pass evaluates rectangles from its LightGrid cell, and a surface that receives a rectangular contribution prepares its LTC basis and coefficients once. Rect shadows, source textures, barn doors, layer and transmission lobes, RTXDI and path-traced area sampling are outside this implementation.

Surface Cache evaluates unoccluded Lambert irradiance with the identity LTC transform and the card's world-space position and normal. `SurfaceCache.directIrradiance` feeds Radiosity when connected. Both card resolve paths evaluate up to `rectLightSamples` rectangles per covered texel, with a default of 8. At or below that light count they sum the exact integrals. Set `rectLightSamples` to 0 for the exhaustive reference.

For larger sets, the existing Grid ReGIR builder resamples rectangle power proposals into 64 RIS slots per cell on an 8×8×8 grid around the card bounds. It draws 8 construction candidates per slot. The cell target uses the rectangle area, fitted distance to the cell volume, and emitting plane. A finite-range rectangle is excluded only when the volume lies wholly outside its influence sphere. Receivers outside the grid use the global power proposal with the same draw budget. Each draw evaluates the existing irradiance integral with its inverse source weight, and an empty slot contributes zero.

The GPU light buffer, PDF and grid are reused across partial relights while the packed rectangles and grid bounds match. Changes to those inputs trigger relighting and invalidate affected indirect history through the existing card update budget. The implementation extends Metallic's native RIS builder and normalization. Rect inputs use full-float dimensions and ACEScg luminance, with the center-based range test adapting the volume target to Metallic's authored range. [NVIDIA RTXDI ReGIR integration](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)

Persistent Surface Cache graphs can set `maxStochasticDirectLightingCardsPerFrame` to add a round-robin Rect refresh budget alongside the deterministic relight budget. The Realtime Lighting and Realtime Lighting VSM samples each set it to 4. A value of 0 retains the fixed sample sequence and updates it with ordinary relights.

The stochastic refresh uses the captured card data and cached deterministic irradiance. It accumulates the Rect mean and batch count in `stochasticDirectHistory`, and the final direct irradiance includes both terms. After 32 batches, each new batch contributes 1/32 of the updated mean. The per-texel sequence advances on refresh, and the grid reservoirs rotate on a nominal card sweep with a minimum interval of four frames. The power PDF is reused on reservoir rotation. A captured or invalidated card starts a new mean when its mandatory relight is processed. Ordinary stochastic refresh updates Radiosity's outgoing lighting while preserving its indirect history. Sun and punctual relights preserve the Rect mean, and a transition to exact Rect evaluation clears it.

The Realtime Lighting graph enables `SurfaceCache.hardwarePunctualShadows` and connects `VBuffer.accelerationStructure` to `SurfaceCache.accelerationStructure`. This mode requires `shadowedLighting` and a scene TLAS with ray query support. It evaluates every eligible point and spot light on the current card update list. Its opaque shadow rays use the UE 5.7.4 card slope bias with a 1 cm base offset, reject front faces, and stop at the first hit up to the light distance or 200 m. The card atlas stores irradiance before the receiver material response. Sun shadows use the GDF and Rect lighting uses the existing LTC path.

The reference uses inline Vulkan RayQuery with the stream TLAS instance opacity flags. It covers opaque geometry represented by that TLAS, including resident or fallback geometry outside the camera frustum. Card relighting follows light, scene transform, visibility and material revisions. A stream-only ray LOD change can affect visibility and is not a separate card relight trigger. The Realtime Lighting VSM graph continues to use software card shadows.

Set `SurfaceCache.hardwareRectShadows` with `hardwarePunctualShadows` to sample an untextured Rect's emitting surface and trace to the sampled point. The shape sampler uses the UE 5.7.4 spherical rectangle fit and small-angle branch. Its irradiance weight is the solid angle times the sampled receiver cosine and source radiance. Finite range in this mode uses NvRTX's squared quartic fade. Metallic supplies the authored axes, scene-linear radiance and per-texel frame seed. The adapter covers one-sided Rects without barn doors or source textures.

Set `rectLightSamples` to 0 for the all-Rect hardware reference, which takes one shape sample per eligible Rect on each refresh. Above the configured sample budget, Metallic's existing Rect ReGIR supplies light indices and inverse selection weights to the same shape and visibility calculation. With `maxStochasticDirectLightingCardsPerFrame` enabled, both cases accumulate the sampled irradiance and reset that history when the Rect inputs or opaque shadow geometry change. NvRTX's shared point, spot and Rect grid, cell normalization and per-texel MIS reservoir are separate work.

Unreal Rect Light informs the authoring model and the separate matrix and amplitude lookups. The shader ports the LTC authors' horizon-clipped quad integration, and embeds the original 64×64 RGBA16F DDS texels from `selfshadow/ltc_code` (commit `31e5e96b54f98f33098f8503003119ba2231a1c6`). The integration uses their Lambert normalization and GGX Fresnel fit. [Unreal Rect Light](https://dev.epicgames.com/documentation/en-us/unreal-engine/rectangular-area-lights-in-unreal-engine) · [LTC reference](https://github.com/selfshadow/ltc_code/tree/31e5e96b54f98f33098f8503003119ba2231a1c6) · [License](../Shaders/Licenses/PolygonalLightLTC.txt)

Port deviations from the LTC demo are limited to the one-sided coplanar rejection, a tangent basis for the degenerate normal-incidence projection, and avoiding the unused fifth vertex read when clipping yields a triangle. Metallic supplies the rectangle pose, range, ACEScg light color and OpenPBR base-lobe inputs. The demo's horizon-clipped mode is used throughout.

## Rendering paths

- `SceneRealtimeLightingPass` evaluates OpenPBR direct lighting with one primary ray, shadow rays for Sun/Moon and point/spot lights, and LTC for rect lights. Diffuse environment GI is a deterministic nine-coefficient SH lookup. It is independent of the diagnostic `VisibilityBufferPass` and requires Vulkan ray-query support. It has no stochastic accumulation, multi-bounce transport, environment specular prefilter or indirect visibility solver; SH GI is an unoccluded low-frequency diffuse approximation modulated by material occlusion.
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

After coarse collection, both raster producers call `GPUSceneSubsystem::recordLightGrid` before traversal and rasterization. `lightGrid(view, frameSlot)` exposes the current GPU buffers and parameters. Re-preparing the view, changing its punctual light revision, cancelling recording or replacing the grid shader invalidates the snapshot. The realtime ray-query pass builds a grid for its own camera and scene. Both realtime shading paths use `ClusteredSurfaceLights` to sample Sun/Moon and the grid's local light candidates through the same direct-light integrator. Raster and ray-query shadow visibility remain specific to their respective paths. Path tracing uses the camera-independent punctual proposal.

The default is **64×64 screen pixels, 32 depth slices, 64 stored lights per cell**. `ClusterLightGridDesc` exposes these limits and the camera to callers. Perspective slices use the Unreal-style mapping `z = log2(depth * B + O) * S`, with `B=1/near`, `O=0`, and `S=sliceCount/log2(far/near)`; this chooses ordinary logarithmic spacing without Unreal's centimetre-scale near bias. Orthographic views use linear spacing. Depth is positive view-space distance, independent of reversed-Z. Partial edge tiles are clamped to the actual viewport; the frozen culling camera keeps its original aspect ratio even when the viewport resizes.

`ClusterLightGrid.slang` dispatches one 64-thread group per cell. Eight view-space tile/depth corners define a conservative AABB. Threads test finite-range lights against their center-based range spheres. Spot lights additionally use Unreal's conservative cone/AABB separating-plane test. Rect lights reject cells wholly behind the emitting plane, including rects with unlimited range. Cells intersecting the plane retain the candidate for the per-receiver evaluation. Grid construction runs entirely on the GPU using the camera grid and authored light bounds.

GPU data contract:

| Buffer | Layout |
| --- | --- |
| `parameters` | One 128-byte `ClusterLightGridParams` |
| `lights` | 64-byte records. Stable punctual slots are followed by compact rectangle culling records |
| `rectLights` | 80-byte rectangle shading records in the same publication. Rectangle ID minus punctual slot count gives the index |
| `candidates` | Source indices for finite-range punctual lights and all rects, followed by unbounded punctual lights |
| `cells` | 16-byte `{offset, count, overflow, totalCount}`, indexed `(z*gridY+y)*gridX+x` |
| `lightIndices` | Cell source-index lists; fixed segment `cellIndex*maxLightsPerCell` |

`ClusterLightGridCommon.slang` provides `clusterLightGridLookup`, `clusterLightGridLocalLightIndex`, and global-list helpers for unbounded point and spot lights. The parameter counts describe clustered candidates, punctual slots, unbounded punctual candidates and total source slots, in that order. Overflow and receivers outside the grid's XY/depth domain use the complete clustered candidate list, then evaluate the unbounded punctual suffix once. Stored cell order follows GPU atomics. Punctual source IDs retain their GPUScene and shadow-table mapping, and rect IDs are scoped to the current grid publication.

Each View/frame slot owns independently versioned resources, reused only after its tracked `RenderFrameContext` submission completes; resizing grows allocations and retains old buffers for in-flight commands. Legacy commands without a frame receive independent buffers and a separate program/descriptor table per recording; their submission state retains these until command reset (the RHI caller must finish GPU work before resetting). Output buffers end in `ShaderRead`. Every cell header is rewritten, including empty scenes, so unused index storage need not be cleared. Invalid/degenerate cameras and oversized allocations fail before dispatch (256 MiB cell/header budget per slot, depth slices 1–256, capacity 1–1024). Shader reload stages replacement programs for every active grid and commits only after all subsystem preparations succeed; old programs remain retained by their frames or command submissions.

The design references local Unreal `Renderer/Private/LightGridInjection.cpp`, `Shaders/Private/LightGridInjection.usf`, and `RenderCore/Public/RenderUtils.h`. This implementation deliberately uses fixed cell segments and explicit fallback instead of UE's linked-list pool, compaction and overflow feedback.

## Validation

`WorldEnvironment.*` covers fixed slots, detached snapshots, domain revisions, invalid values, glTF directional-ignore policy, legacy migration, Sun/Moon persistence and transactional migration rejection. `Photometry.UnitsAndValidation` and `SceneEditing.PhysicalLightingRoundTrip` cover units, negative EV, validation, local imports and persistence. RHI tests `photometric_gpu_units_falloff_sh` and `photometric_realtime_render` cover GPU inverse-square attenuation, celestial invariance, spot cutoff, unit equivalence, exposure, constant-environment irradiance and live light edits.

`SceneEditing.ImportedglTFLights*` covers per-node native-light creation, physical parameters, saved edits/deletions, legacy document loading, composed-source identities, transactional loads/edits, stale-setting rejection and orphaned bindings. RHI tests `imported_virtual_light_*` cover single-emitter ownership, live hierarchy/visibility binding, GPUScene candidate collection and independently resolved scene/source-override isolation.

The `gpu_scene_light_*` tests cover source collection/lifetime, independent light revisions, per-view/frame-slot snapshots, conservative point/spot culling, perspective/orthographic camera planes and light-only world synchronization.

The `cluster_light_grid_*` tests verify CPU camera/depth contracts and GPU-produced point/spot cell lists, actual shader lookup and fallback, source-slot holes, global-light separation, growth/shrink reuse, cancellation, View/frame-slot isolation and staged shader reload.

The `regir_virtual_lights_*` GPU tests cover physical local-light power-PDF construction, point/spot estimates, global fallback, edits/deletion, zero-weight RIS slot probability mass, cancelled-recording retry, and standard PT/OpenPBR PT/RTXDI temporal rendering. Run `MetallicRHITests.exe --rhi-validation --gtest_filter="*regir_virtual_lights_*"` in a configured test build. See [RTXDI validation commands](RTXDI.md#build-and-run) for the shader, PDF-size, grid-layout and sample-preview filters. Test descriptions identify coverage; current execution results belong in the checkpoint evidence report.
