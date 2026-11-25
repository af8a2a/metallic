#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Material/SceneMaterialData.h"
namespace metallic::render {
inline constexpr uint32_t kDeferredDecalTileSize = 16;
#else
public static const uint kDeferredDecalTileSize = 16u;
#endif

// The three affine rows map world coordinates to a unit box centered at the origin.
#ifdef __cplusplus
struct DeferredDecalData
{
    float worldToDecal[3][4];
    uint32_t materialIndex;
    uint32_t receiverMask;
    uint32_t reserved[2];
};
struct DeferredDecalMaterial
{
    float baseColor[4];
    float roughness;
    float roughnessOpacity;
    uint32_t reserved[2];
    SceneMaterialData::TextureInfo baseColorTexture;
    SceneMaterialData::TextureInfo roughnessTexture;
    uint32_t baseColorSourceExtent[2];
    uint32_t roughnessSourceExtent[2];
};
static_assert(sizeof(DeferredDecalData) == 64);
static_assert(offsetof(DeferredDecalData, materialIndex) == 48);
static_assert(sizeof(DeferredDecalMaterial) == 144);
static_assert(offsetof(DeferredDecalMaterial, baseColorTexture) == 32);
static_assert(offsetof(DeferredDecalMaterial, roughnessTexture) == 80);
static_assert(offsetof(DeferredDecalMaterial, baseColorSourceExtent) == 128);
} // namespace metallic::render
#else
struct DeferredDecalData
{
    float4 worldToDecal[3];
    uint materialIndex;
    uint receiverMask;
    uint2 reserved;
};
struct DeferredDecalMaterial
{
    float4 baseColor;
    float roughness;
    float roughnessOpacity;
    uint2 reserved;
    MaterialTextureInfo baseColorTexture;
    MaterialTextureInfo roughnessTexture;
    uint2 baseColorSourceExtent;
    uint2 roughnessSourceExtent;
};
#endif
