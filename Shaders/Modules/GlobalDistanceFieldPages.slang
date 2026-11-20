module GlobalDistanceFieldPages;

import GlobalDistanceField;

namespace Metallic.DistanceField
{
// UE 5.7.4 GlobalDistanceFieldShared.ush:89. The last sample in a page
// is shared with the first sample of its neighbor.
public static const uint kGlobalDistanceFieldPageCells = 7u;
public static const uint kGlobalDistanceFieldPageSamples = 8u;
public static const uint kGlobalDistanceFieldPositivePage = 0xffffffffu;
public static const uint kGlobalDistanceFieldNegativePage = 0xfffffffeu;
public static const uint kGlobalDistanceFieldOverflowPage = 0xfffffffdu;

public uint3 globalDistanceFieldPageAtlasOffset(uint pageIndex, uint3 atlasPages)
{
    return uint3(pageIndex % atlasPages.x, (pageIndex / atlasPages.x) % atlasPages.y,
        pageIndex / (atlasPages.x * atlasPages.y));
}

public struct SparseGlobalDistanceFieldSampler : IGlobalDistanceFieldSampler
{
    public Texture3D<uint> pageTable;
    public Texture3D<float> pageAtlas;
    public SamplerState sampler;
    public GlobalDistanceFieldDomain domain;

    public bool sample(float3 worldPosition, out float distance)
    {
        uint3 tableDimensions;
        pageTable.GetDimensions(tableDimensions.x, tableDimensions.y, tableDimensions.z);
        float3 gridPosition = max((worldPosition - domain.minAndVoxelSize.xyz) / domain.minAndVoxelSize.w, 0.0f);
        // Keep an exact positive endpoint in the last page when the resolution
        // is a multiple of seven. It samples the shared eighth texel.
        uint3 pageCoordinate = uint3(min(gridPosition / float(kGlobalDistanceFieldPageCells), float3(tableDimensions - 1u)));
        uint pageIndex = pageTable.Load(int4(pageCoordinate, 0));

        float influenceRadius = domain.maxAndInfluenceRadius.w;
        if (pageIndex >= kGlobalDistanceFieldOverflowPage) {
            distance = pageIndex == kGlobalDistanceFieldNegativePage ? -influenceRadius : influenceRadius;
            return pageIndex != kGlobalDistanceFieldOverflowPage;
        }

        uint3 atlasDimensions;
        pageAtlas.GetDimensions(atlasDimensions.x, atlasDimensions.y, atlasDimensions.z);
        uint3 pageAtlasOffset = globalDistanceFieldPageAtlasOffset(pageIndex,
            atlasDimensions / kGlobalDistanceFieldPageSamples);
        // UE 5.7.4 GlobalDistanceFieldShared.ush:158. The +0.5 addresses texel
        // centers and keeps linear filtering within this page's eight samples.
        float3 pagePosition = clamp(gridPosition - float3(pageCoordinate * kGlobalDistanceFieldPageCells),
            0.0f, float(kGlobalDistanceFieldPageCells));
        float3 atlasPosition = float3(pageAtlasOffset * kGlobalDistanceFieldPageSamples) + pagePosition + 0.5f;
        distance = pageAtlas.SampleLevel(sampler, atlasPosition / float3(atlasDimensions), 0.0f);
        return true;
    }
};
} // namespace Metallic.DistanceField
