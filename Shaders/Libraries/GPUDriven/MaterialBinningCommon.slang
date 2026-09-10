#ifndef METALLIC_MATERIAL_BINNING_COMMON
#define METALLIC_MATERIAL_BINNING_COMMON

#include "../Scene/SceneMaterial.slang"

// One spatial tile and one 32-bit coverage mask per NVIDIA warp. Classification
// is by executable BSDF features, not source material identity.
static const uint kMaterialTileWidth = 8u;
static const uint kMaterialTileHeight = 4u;
static const uint kMaterialBinGroupSize = 32u;
static const uint kMaterialBinMaxGroupsX = 65535u;
static const uint kMaterialClassBackground = 0u;
static const uint kMaterialClassDielectric = 1u;
static const uint kMaterialClassConductor = 2u;
static const uint kMaterialClassOpaque = 3u;
static const uint kMaterialClassTransmission = 4u;
static const uint kMaterialClassCount = 5u;

struct MaterialBin
{
    uint offset; // offset into the tile task list
    uint count; // number of tile tasks, not pixels
};

struct MaterialTile
{
    uint tileIndex;
    uint laneMask;
};

uint2 materialTilePixel(MaterialTile tile, uint lane, uint width)
{
    uint columns = (width + kMaterialTileWidth - 1u) / kMaterialTileWidth;
    return uint2(tile.tileIndex % columns, tile.tileIndex / columns) *
        uint2(kMaterialTileWidth, kMaterialTileHeight) + uint2(lane & 7u, lane >> 3u);
}

uint classifyDeferredMaterial(PathTraceMaterial material)
{
    // Conservative classification needs no texture sampling. Even a currently
    // black transmission texel stays in the ray-continuation permutation.
    if (material.glassParams.x > 0.0) { return kMaterialClassTransmission; }
    if (material.params.x <= 0.0) { return kMaterialClassDielectric; }
    bool hasMetalnessTexture = material.metallicRoughnessTexture.textureIndex != 0xffffffffu ||
        material.metallicRoughnessTexture.ntcTextureSetIndex != 0xffffffffu;
    if (material.params.x >= 1.0 && !hasMetalnessTexture) { return kMaterialClassConductor; }
    return kMaterialClassOpaque;
}

#endif // METALLIC_MATERIAL_BINNING_COMMON
