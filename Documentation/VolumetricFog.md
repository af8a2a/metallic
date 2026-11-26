# Volumetric fog

Open **GI Playground → Sponza / Volumetric Fog** to see sunlight through the arcade and a warm point light in a low bank of fog. The sample uses fixed exposure and the streamed scene TLAS for opaque directional fog shadows. Surface lighting uses VSM. In the Environment panel, expand **Volumetric fog** to edit the two exponential height layers, scattering, emission and view distance. Expand **Local fog volumes** to add, move or rotate a sphere.

## Render graph

Connect the same visibility pass's `depth` and `rasterInfo` to `VolumetricFogPass`. When `rayTracedShadows` is enabled, also connect its `accelerationStructure` and enable cluster ray tracing on the producer. An unshadowed configuration needs only the first two inputs. The fog pass publishes the medium coefficients, lit scattering and integrated radiance/transmittance as RGBA16F 3D textures.

The Sponza sample uses the renderer's default budget of 1,048,576 VSM work items per shadow face. Its sun needed 712,042 items on the first frame of the 641 × 361 acceptance view. If the work queue overflows, surface shadows fall back to a single coarse page and the sunlit edge on the courtyard floor develops visible steps.

Connect `Fog.integrated`, `Fog.view`, the original reversed-Z depth and the fully lit RGBA32F scene color to `VolumetricFogCompositePass`. Pass its scene-linear `color` to exposure or upscaling. In the Sponza GI atmosphere graph this goes between `Reflections.color` and `DLSSSR.inputColor`. Preserve the existing DLSS depth and motion guides. The dimensions of the froxel grid follow the negotiated input depth resolution.

The scene stores fog under `world.environment.volumetricFog`. Density and height falloff use Unreal's authoring scale, distances use metres and colors are linear Rec.709. Missing settings give disabled fog. The sample's `gridPixelSize=16` and `gridSizeZ=64` reproduce the reference High shadow-quality grid. For deterministic single-frame comparisons, switch off `temporalReprojection` and `conservativeDepth` on both engines.

Each point or spot light can set `volumetricScatteringIntensity` independently of its surface intensity. The default is 1. Set it to 0 to retain the surface light and remove its contribution to fog.

Author local spheres in the scene's `volumetricFog.localVolumes` array. Position and radius use metres. Rotation uses local-to-world Euler angles in degrees, applied about X, Y and Z in order. Height offset is measured in unit-sphere coordinates. The radial and height extinction controls combine according to Unreal's local-volume coverage formula. For a near-camera effect, set the graph's `localFogStartDistanceMeters` to suit the scene. The sample uses 1 m and the reference default is 20 m.

## Reference and deviations

The numerical reference is the local Unreal Engine 5.7.4 NvRTX checkout at `8284a0e654e8`. The relevant sources are `SceneCore.cpp`, `RenderUtils.h::CalculateGridZParams`, `VolumetricFog.cpp`, `VolumetricFog.usf`, `HeightFogCommon.ush`, `HeightFogPixelShader.usf`, `GenerateConservativeDepth.usf` and `LocalFogVolumeRendering.cpp`.

- Metallic supplies a Y-up camera and TLAS in metres. Froxel depths, ray step lengths and medium coefficients use the centimetre convention of the reference. The projection adapter handles Metallic's finite reversed-Z depth.
- Scene colors use Metallic's working color space. The HDR pipeline supplies unexposed radiance, corresponding to a pre-exposure factor of one for both the current frame and history. The physical-atmosphere sun uses Metallic's atmosphere and cloud transport.
- The scene supplies one directional fog light. Point and spot lights use the UE unshadowed inverse-square branch with zero source length, the default distance-bias scale of one and spot soft fading off. Metallic supplies its view light grid and converts candela to the reference centimetre scale. A glTF range of zero gives unlimited range. The renderer derives the UE spot-cone constants from its stored cone cosines. Rect lights, light functions and MegaLights have separate lighting paths.
- Local spheres contribute extinction, scattering and density-weighted emission to the material grid. The pass sorts and packs up to 32 instances per view, culls them against 128-pixel tiles, and stores per-voxel emission in RGBA16F. Tile culling uses the UE permutation without a furthest-depth texture. The global phase parameter applies to all participating media in this froxel path. With local volumes disabled, constant global emission is rounded to the same half precision in the scattering pass.
- The directional RT pass uses inline Vulkan ray queries against opaque TLAS geometry. Alpha-tested any-hit and thin-shadow material evaluation are outside this path. The Sponza sample requires the streamer's cluster acceleration structure support.
- Conservative depth reduces the corresponding texels of the main raster depth directly over the UE HZB footprint. The history manager discards history when the view extent or medium changes and retains it during ordinary camera motion.
- This pass integrates to the configured view distance. Distant analytic height fog, transparent-surface sampling, standalone local-volume splats, volume-domain materials, local-light shadow injection and indirect volume lighting remain outside its coverage.

## Cross-engine reference capture

A 128 × 96 unlit white plane in homogeneous fog was rendered on the same GPU in Unreal D3D12 and Metallic Vulkan with temporal reprojection and conservative depth disabled. The following linear transmittances agreed exactly at the half-float precision of the captured UE scene color.

| Plane distance | Centre (64, 48) | Off-axis (30, 24) | Near corner (6, 8) |
| --- | ---: | ---: | ---: |
| 10 m | 0.79785156 | 0.76757813 | 0.72216797 |
| 20 m | 0.63525391 | 0.58691406 | 0.51953125 |
| 40 m | 0.40209961 | 0.34350586 | 0.26855469 |

The local-light and local-volume regression uses a 40 m wall, an 8 × 6 × 64 grid and five image positions. It compares eight configurations against the UE D3D12 capture. The maximum absolute error across the 40 linear HDR readings was 0.000244140625. Lights with zero scattering intensity, lights outside their range and pixels outside the reference spot or sphere matched the global-only baseline. The test also checks the raster depth, grid dimensions, unbounded lights and overflow of the clustered light list.

Run the focused rendering probes from the directory containing the Release test executable.

```powershell
.\MetallicRHITests.exe --rhi-realtime --rhi-validation --gtest_filter="*volumetric_fog*:*local_fog_volume_view_packing:*history_volume_lifecycle"
```

The [sample graph](../Pipelines/Samples/sponza_volumetric_fog.metallic_graph.json) and [scene](../Asset/Sponza/SponzaVolumetricFog.metallic_scene.json) provide a complete setup.
