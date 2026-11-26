# Volumetric fog

Open **GI Playground → Sponza / Volumetric Fog** to see sunlight through the arcade, a shadowed point light in a low bank of fog, and indirect light scattered by the air. The sample uses fixed exposure, the streamed scene TLAS for opaque directional fog shadows, and VSM for surfaces and the arcade lamp's fog shadow. In the Environment panel, expand **Volumetric fog** to edit the two exponential height layers, scattering, emission and view distance. Expand **Local fog volumes** to add, move or rotate a sphere.

## Render graph

Connect the same visibility pass's `depth` and `rasterInfo` to `VolumetricFogPass`. When `rayTracedShadows` is enabled, also connect its `accelerationStructure` and enable cluster ray tracing on the producer. An unshadowed configuration needs only the first two inputs. The fog pass publishes the medium coefficients, lit scattering and integrated radiance/transmittance as RGBA16F 3D textures.

The Sponza sample uses the renderer's default budget of 1,048,576 VSM work items per shadow face. Its sun needed 712,042 items on the first frame of the 641 × 361 acceptance view. If the work queue overflows, surface shadows fall back to a single coarse page and the sunlit edge on the courtyard floor develops visible steps.

Connect `Fog.integrated`, `Fog.view`, the original reversed-Z depth and the fully lit RGBA32F scene color to `VolumetricFogCompositePass`. Pass its scene-linear `color` to exposure or upscaling. To enable fog in the separate Sponza GI atmosphere graph, insert the composite between `Reflections.color` and `DLSSSR.inputColor` and preserve the DLSS depth and motion guides. The dimensions of the froxel grid follow the negotiated input depth resolution.

The scene stores fog under `world.environment.volumetricFog`. Density and height falloff use Unreal's authoring scale, distances use metres and colors are linear Rec.709. Missing settings give disabled fog. The sample's `gridPixelSize=16` and `gridSizeZ=64` reproduce the reference High shadow-quality grid. For deterministic single-frame comparisons, switch off `temporalReprojection` and `conservativeDepth` on both engines.

Each point or spot light can set `volumetricScatteringIntensity` independently of its surface intensity. The default is 1. Set it to 0 to retain the surface light and remove its contribution to fog.

To shadow the fog from a point or spot light, enable **Cast volumetric shadow** on that light (`castVolumetricShadow` in the scene document, default false). On `VirtualShadowMapLightsPass`, enable `publishWorldShadows` and connect `worldShadowFaces` to the fog pass's input of the same name. Both passes must consume the same `rasterInfo` output. The fog uses the selected light's VSM at each froxel position, including geometry outside the camera view. Each admitted shadow face reserves a coarse page for positions without finer resident pages. A light dropped by the VSM page budget still illuminates fog through the unshadowed path.

Author local spheres in the scene's `volumetricFog.localVolumes` array. Position and radius use metres. Rotation uses local-to-world Euler angles in degrees, applied about X, Y and Z in order. Height offset is measured in unit-sphere coordinates. The radial and height extinction controls combine according to Unreal's local-volume coverage formula. For a near-camera effect, set the graph's `localFogStartDistanceMeters` to suit the scene. The sample uses 1 m and the reference default is 20 m.

### Indirect lighting

Connect the same visibility pass's `depth` and `rasterInfo` to `LumenTranslucencyVolumePass`. Supply the quantized Global Distance Field and its object grid, the Surface Cache cards and depths, and outgoing card radiance. Sponza uses `Radiosity.lighting` for this radiance. Connect all three outputs (`giAmbient`, `giDirectional`, `giGrid`) to the matching fog inputs. Disconnect all three to compare direct-only fog at the same exposure.

The volume traces from positions in the air. Geometry hits pick up outgoing surface radiance and the remaining visibility admits the scene skylight. This gives lit walls a color bounce and lets a roof block light from above. The fog evaluates the two-band spherical harmonics at each scattering sample using its phase parameter. The result contributes to in-scattering; extinction still comes from the medium. The Sponza volume uses 32-pixel cells and extends to 80 m. Its GI geometry comes from the opaque Sponza distance-field and card bakes. Masked foliage and cloth retain their raster and direct-shadow roles.

`maxRayIntensity` limits the brightest channel of each ray before filtering and scales RGB together. Its default is 20. Metallic applies the limit in scene-linear units with pre-exposure equal to one. When comparing against a pre-exposed UE view, use the UE limit divided by that view's pre-exposure.

## Reference and deviations

The numerical reference is the local Unreal Engine 5.7.4 NvRTX checkout at `8284a0e654e8`. The relevant sources are `SceneCore.cpp`, `RenderUtils.h::CalculateGridZParams`, `VolumetricFog.cpp`, `VolumetricFog.usf`, `HeightFogCommon.ush`, `HeightFogPixelShader.usf`, `GenerateConservativeDepth.usf` and `LocalFogVolumeRendering.cpp`.

- Metallic supplies a Y-up camera and TLAS in metres. Froxel depths, ray step lengths and medium coefficients use the centimetre convention of the reference. The projection adapter handles Metallic's finite reversed-Z depth.
- Scene colors use Metallic's working color space. The HDR pipeline supplies unexposed radiance, corresponding to a pre-exposure factor of one for both the current frame and history. The physical-atmosphere sun uses Metallic's atmosphere and cloud transport.
- The scene supplies one directional fog light. Point and spot lights use UE's inverse-square attenuation with zero source length, the default distance-bias scale of one and spot soft fading off. Metallic supplies its view light grid and converts candela to the reference centimetre scale. A glTF range of zero gives unlimited range. The renderer derives the UE spot-cone constants from its stored cone cosines. Rect lights, light functions and MegaLights have separate lighting paths.
- Shadowed point and spot lights follow the VSM branch of UE's `InjectShadowedLocalLight` and `SampleVirtualShadowMapLocal`: a single reversed-Z comparison at the sample position with zero ray-start and surface-normal bias. Metallic uses its point-face convention, resident page table and atlas. For an unlimited light, Metallic bounds the shadow projection by its casters; beyond that far plane a clear texel remains lit and a stored blocker still shadows. The fog shader evaluates each light once in its existing clustered loop. Shadowed lights use full-precision light inputs and share the fog pass's jitter/history. This scheduling omits UE's separate pre-exposed FP16 local-light accumulation target and its intermediate rounding.
- Local spheres contribute extinction, scattering and density-weighted emission to the material grid. The pass sorts and packs up to 32 instances per view, culls them against 128-pixel tiles, and stores per-voxel emission in RGBA16F. Tile culling uses the UE permutation without a furthest-depth texture. The global phase parameter applies to all participating media in this froxel path. With local volumes disabled, constant global emission is rounded to the same half precision in the scattering pass.
- The directional RT pass uses inline Vulkan ray queries against opaque TLAS geometry. Alpha-tested any-hit and thin-shadow material evaluation are outside this path. The Sponza sample requires the streamer's cluster acceleration structure support.
- Conservative depth reduces the corresponding texels of the main raster depth directly over the UE HZB footprint. The history manager discards history when the view extent or medium changes and retains it during ordinary camera motion.
- The indirect lighting reference is the software `r.Lumen.TranslucencyVolume.TraceFromVolume=1`, `r.Lumen.TranslucencyVolume.RadianceCache=0` path in `LumenTranslucencyVolumeLighting.usf`. Trace radiance, spatial filter results and ambient SH use RGBA16F, matching UE's `r.Lumen.LightingDataFormat=1`. Directional SH also uses RGBA16F to preserve signed coefficients. The shaders use Metallic's metre-space GDF and card radiance, and sample its lat-long environment in working color space in place of UE's skylight cubemap. The physical atmosphere provides its diskless sky capture.
- The nine octahedral directions occupy adjacent Z slices in the trace atlas. The visibility reduction takes the furthest reversed-Z depth over the reference HZB footprint. The fog reads the current-frame SH before UE's separate transparency intensity scale and history filter. Metallic's fog history accumulates the resulting scattering. The SH volume sampler preserves the reference clip-W depth mapping, including constant clip W in an orthographic view.
- This pass integrates to the configured view distance. Distant analytic height fog, transparent-surface sampling, standalone local-volume splats, volume-domain materials and the translucency-volume Radiance Cache permutation remain outside its coverage.

## Cross-engine reference capture

A 128 × 96 unlit white plane in homogeneous fog was rendered on the same GPU in Unreal D3D12 and Metallic Vulkan with temporal reprojection and conservative depth disabled. The following linear transmittances agreed exactly at the half-float precision of the captured UE scene color.

| Plane distance | Centre (64, 48) | Off-axis (30, 24) | Near corner (6, 8) |
| --- | ---: | ---: | ---: |
| 10 m | 0.79785156 | 0.76757813 | 0.72216797 |
| 20 m | 0.63525391 | 0.58691406 | 0.51953125 |
| 40 m | 0.40209961 | 0.34350586 | 0.26855469 |

The local-light and local-volume regression uses a 40 m wall, an 8 × 6 × 64 grid and five image positions. It compares eight configurations against the UE D3D12 capture. The maximum absolute error across the 40 linear HDR readings was 0.000244140625. Lights with zero scattering intensity, lights outside their range and pixels outside the reference spot or sphere matched the global-only baseline. The test also checks the raster depth, grid dimensions, unbounded lights and overflow of the clustered light list.

The punctual VSM regression places a wall with a doorway between each lamp and the fog, entirely outside the camera view. With seven physical pages (six point faces and one spot face), the point light illuminates the central camera ray at 14.86 m and the spot at 25.15 m. The test checks zero in-scattering behind the solid wall and lintel and verifies that toggling each lamp's shadow off restores its original scattering. A matching UE D3D12 SM5 capture reproduced the occlusion pattern. Across six image positions the maximum absolute linear HDR differences with shadows enabled were 0.000610 for the point light and 0.017944 for the spot. The UE capture's local shadow backend was not independently identified.

The Cornell fog regression gives a spotlight zero direct volumetric intensity and points it at the wall. Changing the wall from gray to red changed the sampled mean linear fog RGB from (0.3647, 0.3647, 0.3647) to (0.2332, 0.0471, 0.0252). With only a white HDRI, removing the ceiling increased the mean from 0.0737 to 0.1170. The mean extinction stayed at 0.002499 in all configurations. A separate shader probe checks signed directional SH against the reference phase expression at 48 view directions for each of `g=-0.7` and `g=0.7`.

A UE D3D12 SM6 capture used the same perspective view, 128 × 96 render target, 32 × 32 measurement region, fixed pre-exposure of one and direct volume tracing. With card radiosity disabled, the maximum absolute difference in the red wall test was 0.0108 after conversion to linear ACEScg. To isolate skylight visibility, the UE control used absorbing surfaces with distance fields baked at the walls' final dimensions, waited for the skylight capture and destroyed the roof actor for the open measurement.

| Mean linear fog light | Metallic Vulkan | Unreal D3D12 |
| --- | ---: | ---: |
| Gray wall, spotlight illuminating only surfaces | 0.36466 | 0.36838 |
| White sky, roof present | 0.07369 | 0.07413 |
| White sky, roof removed | 0.11696 | 0.11758 |

White diffuse walls with full card radiosity still show a cross-engine difference. A second UE material exposed the white wall to Lumen card capture and rendered it black to the camera, allowing a direct HDR reading of the fog. With the roof present, Metallic measured 0.14796 and UE measured 0.15416. After removing it, Metallic rose to 0.20531 and UE fell to 0.12475. The difference persists when the underlying surface color is excluded. The isolated wall and skylight checks above cover the direct volume trace. The white-room result also depends on the upstream card lighting.

Run the focused rendering probes from the directory containing the Release test executable.

```powershell
.\MetallicRHITests.exe --rhi-realtime --rhi-validation --gtest_filter="*volumetric_fog*:*local_fog_volume_view_packing:*history_volume_lifecycle"
```

The [sample graph](../Pipelines/Samples/sponza_volumetric_fog.metallic_graph.json) and [scene](../Asset/Sponza/SponzaVolumetricFog.metallic_scene.json) provide a complete setup.
