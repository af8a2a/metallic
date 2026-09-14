#language slang 2026
implementing GPUDriven;

import Material;
using Metallic.Material;

namespace Metallic.GPUDriven
{
// One spatial tile and one 32-bit coverage mask per NVIDIA warp. Classification
// is by executable BSDF features, not source material identity.
public static const uint kMaterialTileWidth = 8u;
public static const uint kMaterialTileHeight = 4u;
public static const uint kMaterialBinGroupSize = 32u;
public static const uint kMaterialBinMaxGroupsX = 65535u;
public static const uint kMaterialClassBackground = 0u;
public static const uint kMaterialClassDielectric = 1u;
public static const uint kMaterialClassConductor = 2u;
public static const uint kMaterialClassOpaque = 3u;
public static const uint kMaterialClassTransmission = 4u;
public static const uint kMaterialClassCount = 5u;

public struct MaterialBin
{
    public uint offset; // offset into the tile task list
    public uint count; // number of tile tasks, not pixels
};

public struct MaterialTile
{
    public uint tileIndex;
    public uint laneMask;
};

public uint2 materialTilePixel(MaterialTile tile, uint lane, uint width)
{
    uint columns = (width + kMaterialTileWidth - 1u) / kMaterialTileWidth;
    return uint2(tile.tileIndex % columns, tile.tileIndex / columns) *
        uint2(kMaterialTileWidth, kMaterialTileHeight) + uint2(lane & 7u, lane >> 3u);
}

public uint classifyDeferredMaterial(PathTraceMaterial material)
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

} // namespace Metallic.GPUDriven
