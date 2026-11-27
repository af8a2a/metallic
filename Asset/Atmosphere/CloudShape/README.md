# Default volumetric cloud shape textures

These texels come from the Windows platform data of Unreal Engine 5.7.4
NvRTX (revision `8284a0e654e8`). The instance
`/Engine/EngineSky/VolumetricClouds/m_SimpleVolumetricCloud_Inst` references
the following assets in the same directory.

| Metallic asset | UE asset | Platform mip 0 | Stored format |
| --- | --- | --- | --- |
| Pattern.rgba16f | T_CloudPattern | 256 × 256 | RGBA16F, linear |
| Profile.rgba16f | T_Profile_08 | 256 × 256 | RGBA16F, linear |
| Noise.rgba8srgb | VT_PerlinWorley_Balanced | 128 × 128 × 128 | RGBA8, sRGB view |

Each file contains tightly packed mip-zero texels in X, Y, Z order. The
2D inputs retain all four half channels. The volume import swaps the
platform's BGRA bytes to RGBA. The sampler repeats the texture, including
along the volume's Z axis. The default material's weather mask is eliminated
by its instance's zero RGB mask dot product.

## Reproduction

Use the Windows cook of the referenced packages or export the bound GPU
textures from a frame capture as DX10 DDS. An editor source DDS contains
the original import data, including the 1024 × 1024 pattern before UE
builds its 256 × 256 platform mip.

For a source-engine extraction, request the Windows target with
`ITargetPlatformManagerModule::FindTargetPlatform`, call
`UTexture::BeginCacheForCookedPlatformData`, and wait for
`IsCachedCookedPlatformDataLoaded`. The resulting `FTexturePlatformData`
contains `PF_FloatRGBA` for the two 2D textures and `PF_B8G8R8A8` for
the volume. Export the unmodified top mip as `<UE asset>.mip0` or use a
GPU DDS export named `<UE asset>.dds`. Keep the input in platform byte
order. Run the importer from the repository root.

```powershell
python -B Tools/ImportCloudShapeTextures.py <export-directory>
```

The importer verifies the SHA-256 of every decoded texture before writing
the three runtime files. The reference package SHA-256 values are recorded
below.

| UE asset | Source package SHA-256 |
| --- | --- |
| T_CloudPattern | `0a8225097ad44c42ea9f5d5d7cba61ff0576ec43fbb328458dd357c1b3d70225` |
| T_Profile_08 | `25334827f0073977519a189bbb3dc784baf1aaa99f97b4cafc20f321b837ef15` |
| VT_PerlinWorley_Balanced | `89542fbe266d8f0d6d01ed3861cf72311c756d3a30d19c7c555041d44893c007` |

See the repository's [Unreal license notice](../../../Shaders/Licenses/ColorGrading/UNREAL_LICENSE.md).
