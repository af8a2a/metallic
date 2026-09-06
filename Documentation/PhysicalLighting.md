# Physical lighting

Select **Real-time / Physical Lighting** in Samples, then use the **Physical Lighting** panel to add, disable, edit or remove directional, point and spot lights. World lights and manual exposure are saved in `world.lighting` in the scene document; they survive reload and Discard restores the saved state. Imported glTF lights remain editable through their LightComponent Inspector, including intensity units and undo/redo. Changing units preserves the physical intensity (zero cannot be converted to a finite EV).

## Units and calibration

Scene distances are **metres**. Colors are linear RGB multipliers. The default imported `si` unit follows [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md): directional intensity is lux; point/spot intensity is candela.

| Light | Authoring units | Conversion to transport units |
| --- | --- | --- |
| Directional | Lux or incident-meter EV100 | `lux = 2.5 × 2^EV100` (ISO 100, incident calibration constant C=250) |
| Point | Candela, Lumens or local-light EV | `cd = lm / (4π)`; `cd = 2^EV` |
| Spot | Candela, Lumens or local-light EV | `cd = lm / Ωeffective`; `cd = 2^EV` |

Local-light EV follows Unreal's fixed reference-area convention, not the incident-light meter convention. See [Unreal physical lighting units](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-physical-lighting-units-in-unreal-engine) and local sources `Engine/Source/Runtime/Engine/Private/Components/LocalLightComponent.cpp`, `PointLightComponent.cpp`, and `Engine/Source/Runtime/RenderCore/Public/RenderUtils.h`.

Point and spot illumination is `I / distance²` before the material's cosine term. Range 0 is unbounded; an optional finite range multiplies illumination by `max(1 - (distance/range)^4, 0)` to smoothly cut off at the authored range. The singularity alone is capped at a 1 mm distance. Spot attenuation is a squared linear interpolation in cosine space between outer and inner half-angles. Lumen normalization integrates this actual profile: `Ωeffective = 2π [(1 - cos(inner)) + (cos(inner) - cos(outer))/3]`. It is not a uniform-cone approximation.

Manual exposure follows Unreal's `EV100ToLuminance(1, EV)` saturation-luminance convention: `Lmax = 2^EV cd/m²`, display scale `1/Lmax`. It is applied after light transport, including NRC and denoiser outputs; it does not alter light powers or the environment PDF. This is a manual exposure setting, not an automatic exposure algorithm.

An HDR file is not inherently calibrated. To compare it against physical lamps, its linear pixels must represent cd/m² after multiplying by Environment Intensity. This multiplier is therefore also the HDR calibration scale. Arbitrary HDRIs and the procedural preview sky are not guaranteed to have an absolute physical calibration.

## Rendering paths

- `SceneRealtimeLightingPass` evaluates OpenPBR direct lighting with one primary ray and shadow rays for directional/point/spot lights. Diffuse environment GI is a deterministic nine-coefficient SH lookup. It is independent of the diagnostic `VisibilityBufferPass` and requires Vulkan ray-query support. It has no stochastic accumulation, multi-bounce transport, environment specular prefilter or indirect visibility solver; SH GI is an unoccluded low-frequency diffuse approximation modulated by material occlusion.
- `ScenePathTracePass` consumes the same physical lights as delta-light next-event estimates (MIS weight 1), alongside the existing environment PDF/MIS sampler. It does not sample the irradiance SH as radiance. OpenPBR supports transmission-aware shadow transport; the standard material path uses its direct-light BRDF and opaque shadow queries.
- Environment PDF construction, SH projection and cosine convolution run entirely on the GPU. SH covers bands l=0,1,2 (nine RGB coefficients), with convolution factors π, 2π/3, π/4. Consumers evaluate irradiance in the rotated environment frame and apply diffuse albedo/π exactly once. The procedural sky is also projected on the GPU.
- Light buffers are immutable per update and retained for in-flight frames. Light changes invalidate path-tracing history and radiance caches. Imported scene lights and world lights are additive; light visibility/enabled state excludes them from the buffer.

The existing RTXDI many-light benchmark retains its own synthetic benchmark lights; the physical scene-light path described here is the new real-time pass and the standard/OpenPBR path tracers.

## DrawSet light candidates

`GPUSceneSubsystem::beginFrame` synchronizes imported scene lights and virtual world lights, including worlds without a Scene. Shared `SceneLightRecord` packing preserves source slots and the physical `GpuPunctualLight` data; disabled, invalid, black and zero-intensity sources retain their slots but do not enter `GPUSceneDrawSet::lights`.

`GPUScene::prepareView` collects candidates independently for each View/frame slot, alongside coarse mesh collection. It follows Unreal's `ComputeLightVisibility` (`Renderer/Private/SceneVisibility.cpp`) and `FSphere::FromCone` (`Core/Public/Math/Sphere.h`): point lights use their attenuation sphere; spot lights use a conservative sphere enclosing their radial range and outer cone. Non-normalized inward frustum planes are supported, tangency is retained, and invalid/zero planes are skipped. Neither mesh visibility predicates nor HZB occlusion remove lights. There is no screen-size or brightness threshold.

`GPUSceneSubsystem::visibleLights(view, frameSlot)` returns three deterministic source-ID lists:

- `localLights`: finite-range point/spot candidates for a future LightGrid.
- `directionalLights`: global directional lights, unaffected by camera frustum.
- `unboundedLocalLights`: range-zero point/spot lights, kept separately because a finite grid bound cannot represent their influence.

Resolve each ID through `light(id)` to obtain its source indices/object, unmodified GPU payload and coarse `boundingSphere`. The sphere is not a replacement for the original position/range/cone used in future grid intersection. The collection has its own `lightGeneration`/`lightRevision`; parameter edits preserve IDs, while source-slot topology changes invalidate them. Check `visibleLights` or `lights.validFor(...)` before using a snapshot: light-only changes expire light lists without changing mesh DrawSet revisions or GPU allocations. GPUScene's geometric HZB state is independent; the renderer's existing global radiance-history invalidation/camera-cut policy remains unchanged.

VisibilityBuffer and GPUDrivenStreamAsset provide the same camera used for mesh culling. VisibilityBuffer supports frozen and orthographic cameras; disabling `instanceFrustumCull` also disables light frustum rejection. This is the CPU coarse collection stage only, not LightGrid construction or GPU grid upload. Existing physical lighting and path tracing continue consuming the complete light buffer; camera-culled lists must not replace secondary-path lighting. Environment PDF and SH are not local-light candidates.

## Validation

`Photometry.UnitsAndValidation` and `SceneEditing.PhysicalLightingRoundTrip` verify units, negative EV, validation, imported overrides, virtual light persistence and discard. RHI tests `photometric_gpu_units_falloff_sh` and `photometric_realtime_render` verify GPU inverse-square attenuation, directional invariance, spot cutoff, unit equivalence, exposure, constant-environment irradiance and live light add/edit/remove.

The `gpu_scene_light_*` tests cover source collection/lifetime, independent light revisions, per-view/frame-slot snapshots, conservative point/spot culling, perspective/orthographic camera planes and light-only world synchronization.
