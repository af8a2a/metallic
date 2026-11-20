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
public static const uint kGlobalDistanceFieldMipFactor = 4u;
public static const float kGlobalDistanceFieldMipTransition = 0.625f;

// UE 5.7.4 GlobalDistanceFieldShared.ush:183.
public float encodeGlobalDistanceFieldPageDistance(float worldDistance, float influenceRadius)
{
    return saturate(worldDistance / (2.0f * influenceRadius) + 0.5f);
}

public float decodeGlobalDistanceFieldPageDistance(float encodedDistance, float influenceRadius)
{
    return (encodedDistance * 2.0f - 1.0f) * influenceRadius;
}

public uint3 globalDistanceFieldPageAtlasOffset(uint pageIndex, uint3 atlasPages)
{
    return uint3(pageIndex % atlasPages.x, (pageIndex / atlasPages.x) % atlasPages.y,
        pageIndex / (atlasPages.x * atlasPages.y));
}

public uint loadGlobalDistanceFieldPage(Texture3D<uint> pageTable, float3 gridPosition, out float3 pagePosition)
{
    uint3 tableDimensions;
    pageTable.GetDimensions(tableDimensions.x, tableDimensions.y, tableDimensions.z);
    // An exact positive endpoint uses the last page's shared eighth texel.
    uint3 pageCoordinate = uint3(min(max(gridPosition, 0.0f) / float(kGlobalDistanceFieldPageCells),
        float3(tableDimensions - 1u)));
    pagePosition = clamp(gridPosition - float3(pageCoordinate * kGlobalDistanceFieldPageCells),
        0.0f, float(kGlobalDistanceFieldPageCells));
    return pageTable.Load(int4(pageCoordinate, 0));
}

float sampleGlobalDistanceFieldResidentPage(Texture3D<float> pageAtlas, SamplerState sampler,
    uint pageIndex, float3 pagePosition)
{
    uint3 atlasDimensions;
    pageAtlas.GetDimensions(atlasDimensions.x, atlasDimensions.y, atlasDimensions.z);
    uint3 pageAtlasOffset = globalDistanceFieldPageAtlasOffset(pageIndex,
        atlasDimensions / kGlobalDistanceFieldPageSamples);
    // UE 5.7.4 GlobalDistanceFieldShared.ush:158.
    float3 atlasPosition = float3(pageAtlasOffset * kGlobalDistanceFieldPageSamples) + pagePosition + 0.5f;
    return pageAtlas.SampleLevel(sampler, atlasPosition / float3(atlasDimensions), 0.0f);
}

public float sampleGlobalDistanceFieldEncodedPage(Texture3D<float> pageAtlas, SamplerState sampler,
    uint pageIndex, float3 pagePosition)
{
    if (pageIndex >= kGlobalDistanceFieldOverflowPage) {
        return pageIndex == kGlobalDistanceFieldNegativePage ? 0.0f : 1.0f;
    }
    return sampleGlobalDistanceFieldResidentPage(pageAtlas, sampler, pageIndex, pagePosition);
}

public struct SparseGlobalDistanceFieldSampler : IGlobalDistanceFieldSampler
{
    public Texture3D<uint> pageTable;
    public Texture3D<float> pageAtlas;
    public SamplerState sampler;
    public GlobalDistanceFieldDomain domain;

    public bool sample(float3 worldPosition, out float distance)
    {
        float3 gridPosition = (worldPosition - domain.minAndVoxelSize.xyz) / domain.minAndVoxelSize.w;
        float3 pagePosition;
        uint pageIndex = loadGlobalDistanceFieldPage(pageTable, gridPosition, pagePosition);

        float influenceRadius = domain.maxAndInfluenceRadius.w;
        if (pageIndex >= kGlobalDistanceFieldOverflowPage) {
            distance = pageIndex == kGlobalDistanceFieldNegativePage ? -influenceRadius : influenceRadius;
            return pageIndex != kGlobalDistanceFieldOverflowPage;
        }

        distance = sampleGlobalDistanceFieldResidentPage(pageAtlas, sampler, pageIndex, pagePosition);
        return true;
    }
};

public struct QuantizedGlobalDistanceFieldSampler : IGlobalDistanceFieldSampler
{
    public Texture3D<uint> pageTable;
    public Texture3D<float> pageAtlas;
    public Texture3D<float> coarseMip;
    public SamplerState sampler;
    public GlobalDistanceFieldDomain domain;

    public bool sample(float3 worldPosition, out float distance)
    {
        const float3 volumePosition = worldPosition - domain.minAndVoxelSize.xyz;
        const float3 uv = volumePosition / (domain.maxAndInfluenceRadius.xyz - domain.minAndVoxelSize.xyz);
        const float coarse = coarseMip.SampleLevel(sampler, uv, 0.0f);
        const float radius = domain.maxAndInfluenceRadius.w;
        distance = decodeGlobalDistanceFieldPageDistance(coarse, float(kGlobalDistanceFieldMipFactor) * radius);

        float3 pagePosition;
        uint pageIndex = loadGlobalDistanceFieldPage(pageTable, volumePosition / domain.minAndVoxelSize.w, pagePosition);
        // UE 5.7.4 GlobalDistanceFieldUtils.ush:139. A negative constant page
        // supplies the same fine value as a resident page filled with zero.
        if ((pageIndex < kGlobalDistanceFieldOverflowPage || pageIndex == kGlobalDistanceFieldNegativePage) &&
            coarse < kGlobalDistanceFieldMipTransition) {
            float fine = sampleGlobalDistanceFieldEncodedPage(pageAtlas, sampler, pageIndex, pagePosition);
            distance = decodeGlobalDistanceFieldPageDistance(fine, radius);
        }
        // A saturated positive mip carries no propagated geometry into an
        // overflow page. Preserve the unavailable state for this case.
        return pageIndex != kGlobalDistanceFieldOverflowPage || coarse < 1.0f;
    }
};
} // namespace Metallic.DistanceField
