# Physical lighting

Select **Real-time / Physical Lighting** in Samples, then use the **Physical Lighting** panel to add, disable, edit or remove directional, point and spot lights. Imported glTF lights appear in the same panel as native virtual lights. World lights and manual exposure are saved in `world.lighting` in the scene document; they survive reload and Discard restores the saved state. Changing units preserves the physical intensity (zero cannot be converted to a finite EV).

## Imported glTF lights

Each active glTF node that references a punctual light creates its own native `PunctualLight`, including separate instances of the same glTF light definition. The document identifies the source with `(sourceId, sourceNodeIndex)`, so repeated assets in a composed scene remain distinct. Names, linear colors, intensity, range and cone angles are preserved; directional intensity is lux and point/spot intensity is candela. An omitted or zero range remains unbounded.

Imported virtual lights retain a source-node binding. Their position and emission direction follow the complete parent hierarchy and source mount, with glTF's local -Z emission axis. The Physical Lighting panel displays the current world-space pose and the import source; editing that pose stores a source-local offset rather than detaching the light. Parent/source visibility and the light's own Enabled switch both control its contribution. Manual virtual lights without an import binding continue to use independent world-space poses.

The source `LightComponent` remains hierarchy and import metadata, not a second emitter. Its Inspector edits the same native light properties, including intensity units and undo/redo; removing the native light makes the old Inspector read-only. Saved edits and removals survive reload, and removed imported lights are not recreated from their original defaults. Scene replacement uses the new document's lights instead of appending another copy. The common runtime light packing resolves the live source transform and emits a converted light only once across real-time shading, path tracing and clustered LightGrid consumers.

A scene resolved independently by a render-graph pass or selected through `sourceOverride` uses its own authored lights, adding only manually created, unbound world lights; it never borrows imported-light bindings from another scene.

## Units and calibration

Scene distances are **metres**. Colors are linear RGB multipliers. glTF imports follow [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md): directional intensity is lux; point/spot intensity is candela. The source `si` convention and native Lux/Candela authoring units represent the same physical values.

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
- Light buffers are immutable per update and retained for in-flight frames. Light changes invalidate path-tracing history and radiance caches. Imported native lights and manually added world lights are additive; converted source metadata does not emit a duplicate contribution. Source visibility and virtual-light enabled state exclude inactive lights from the buffer.

The existing RTXDI many-light benchmark retains its own synthetic benchmark lights; the physical scene-light path described here is the new real-time pass and the standard/OpenPBR path tracers.

## DrawSet light candidates

`GPUSceneSubsystem::beginFrame` synchronizes scene and virtual world lights, including worlds without a Scene. Shared `SceneLightRecord` packing resolves imported native-light bindings against the current source pose and suppresses duplicate imported emission. It preserves source slots and the physical `GpuPunctualLight` data; disabled, invalid, black and zero-intensity sources retain their slots but do not enter `GPUSceneDrawSet::lights`.

`GPUScene::prepareView` collects candidates independently for each View/frame slot, alongside coarse mesh collection. It follows Unreal's `ComputeLightVisibility` (`Renderer/Private/SceneVisibility.cpp`) and `FSphere::FromCone` (`Core/Public/Math/Sphere.h`): point lights use their attenuation sphere; spot lights use a conservative sphere enclosing their radial range and outer cone. Non-normalized inward frustum planes are supported, tangency is retained, and invalid/zero planes are skipped. Neither mesh visibility predicates nor HZB occlusion remove lights. There is no screen-size or brightness threshold.

`GPUSceneSubsystem::visibleLights(view, frameSlot)` returns three deterministic source-ID lists:

- `localLights`: finite-range point/spot candidates for the clustered LightGrid.
- `directionalLights`: global directional lights, unaffected by camera frustum.
- `unboundedLocalLights`: range-zero point/spot lights, kept separately because a finite grid bound cannot represent their influence.

Resolve each ID through `light(id)` to obtain its source indices/object, unmodified GPU payload and coarse `boundingSphere`. The sphere is not a replacement for the original position/range/cone used in future grid intersection. The collection has its own `lightGeneration`/`lightRevision`; parameter edits preserve IDs, while source-slot topology changes invalidate them. Check `visibleLights` or `lights.validFor(...)` before using a snapshot: light-only changes expire light lists without changing mesh DrawSet revisions or GPU allocations. GPUScene's geometric HZB state is independent; the renderer's existing global radiance-history invalidation/camera-cut policy remains unchanged.

VisibilityBuffer and GPUDrivenStreamAsset provide the same camera used for mesh culling. VisibilityBuffer supports frozen and orthographic cameras; disabling `instanceFrustumCull` disables only the coarse light frustum rejection, not clustered intersection. Environment PDF and SH are not local-light candidates.

## GPU clustered LightGrid

After coarse collection, both raster paths call `GPUSceneSubsystem::recordLightGrid` before traversal/culling/raster work. `lightGrid(view, frameSlot)` exposes the current GPU buffers and parameters; it returns null after the view is re-prepared, its light revision changes, its recording is cancelled, or its shader program is replaced. The RTAS diagnostic path does not construct a raster grid. The grid is currently a producer and shared shader query interface; existing physical-lighting/path-tracing shaders have not been changed to use camera-filtered lists for transport.

The default is **64×64 screen pixels, 32 depth slices, 64 stored lights per cell**. `ClusterLightGridDesc` exposes these limits and the camera to callers. Perspective slices use the Unreal-style mapping `z = log2(depth * B + O) * S`, with `B=1/near`, `O=0`, and `S=sliceCount/log2(far/near)`; this chooses ordinary logarithmic spacing without Unreal's centimetre-scale near bias. Orthographic views use linear spacing. Depth is positive view-space distance, independent of reversed-Z. Partial edge tiles are clamped to the actual viewport; the frozen culling camera keeps its original aspect ratio even when the viewport resizes.

`ClusterLightGrid.slang` dispatches one 64-thread group per cell. Eight view-space tile/depth corners define a conservative AABB. Threads test bounded local candidates against their original range spheres; spot lights additionally use Unreal's conservative cone/AABB separating-plane test. The emission point, radial range, outer cone and physical intensity are preserved. No CPU cell-light calculation, GPU readback, receiver-depth/HZB rejection or screen-size/brightness heuristic is used by the runtime.

GPU data contract:

| Buffer | Layout |
| --- | --- |
| `parameters` | One 128-byte `ClusterLightGridParams` |
| `lights` | 64-byte physical records at stable GPUScene source slots, **no metadata header** |
| `candidates` | Source indices: bounded locals, then directionals, then unbounded locals |
| `cells` | 16-byte `{offset, count, overflow, totalCount}`, indexed `(z*gridY+y)*gridX+x` |
| `lightIndices` | Cell source-index lists; fixed segment `cellIndex*maxLightsPerCell` |

`ClusterLightGridCommon.slang` provides `clusterLightGridLookup`, `clusterLightGridLocalLightIndex`, and independent global-light helpers. Overflow never silently truncates lighting: a cell with more matches than capacity returns the **complete bounded-local candidate list** through the lookup helper. Outside the grid's XY/depth domain, the helper also falls back to that candidate list instead of incorrectly clamping into the last cell. This is still a view-filtered list, not a substitute for a full-scene query outside that view or on secondary rays. Global lights must be evaluated once in addition to local lookup. Stored cell order is GPU-atomic order and is not stable; source IDs and candidate ordering are stable. Do not pass these source indices to the old compact/header-prefixed `gPunctualLights` buffer.

Each View/frame slot owns independently versioned resources, reused only after its tracked `RenderFrameContext` submission completes; resizing grows allocations and retains old buffers for in-flight commands. Legacy commands without a frame receive independent buffers and a separate program/descriptor table per recording; their submission state retains these until command reset (the RHI caller must finish GPU work before resetting). Output buffers end in `ShaderRead`. Every cell header is rewritten, including empty scenes, so unused index storage need not be cleared. Invalid/degenerate cameras and oversized allocations fail before dispatch (256 MiB cell/header budget per slot, depth slices 1–256, capacity 1–1024). Shader reload stages replacement programs for every active grid and commits only after all subsystem preparations succeed; old programs remain retained by their frames or command submissions.

The design references local Unreal `Renderer/Private/LightGridInjection.cpp`, `Shaders/Private/LightGridInjection.usf`, and `RenderCore/Public/RenderUtils.h`. This implementation deliberately uses fixed cell segments and explicit fallback instead of UE's linked-list pool, compaction and overflow feedback.

## Validation

`Photometry.UnitsAndValidation` and `SceneEditing.PhysicalLightingRoundTrip` verify units, negative EV, validation, imported overrides, virtual light persistence and discard. RHI tests `photometric_gpu_units_falloff_sh` and `photometric_realtime_render` verify GPU inverse-square attenuation, directional invariance, spot cutoff, unit equivalence, exposure, constant-environment irradiance and live light add/edit/remove.

`SceneEditing.ImportedGltfLights*` covers per-node native-light creation, physical parameters, saved edits/deletions, legacy document loading, composed-source identities, transactional loads/edits, stale-setting rejection and orphaned bindings. RHI tests `imported_virtual_light_*` cover single-emitter ownership, live hierarchy/visibility binding, GPUScene candidate collection and independently resolved scene/source-override isolation.

The `gpu_scene_light_*` tests cover source collection/lifetime, independent light revisions, per-view/frame-slot snapshots, conservative point/spot culling, perspective/orthographic camera planes and light-only world synchronization.

The `cluster_light_grid_*` tests verify CPU camera/depth contracts and GPU-produced point/spot cell lists, actual shader lookup and fallback, source-slot holes, global-light separation, growth/shrink reuse, cancellation, View/frame-slot isolation and staged shader reload.
