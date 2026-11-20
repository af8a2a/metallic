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

public uint loadGlobalDistanceFieldPage(Texture3D<uint> pageTable, float3 gridPosition, out float3 pagePosition,
    uint clipmapIndex = 0u)
{
    uint3 tableDimensions;
    pageTable.GetDimensions(tableDimensions.x, tableDimensions.y, tableDimensions.z);
    // An exact positive endpoint uses the last page's shared eighth texel.
    uint3 pageCoordinate = uint3(min(max(gridPosition, 0.0f) / float(kGlobalDistanceFieldPageCells),
        float(tableDimensions.x - 1u)));
    pagePosition = clamp(gridPosition - float3(pageCoordinate * kGlobalDistanceFieldPageCells),
        0.0f, float(kGlobalDistanceFieldPageCells));
    pageCoordinate.z += clipmapIndex * tableDimensions.x;
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
    public uint clipmapIndex;

    public bool sample(float3 worldPosition, out float distance)
    {
        const float3 volumePosition = worldPosition - domain.minAndVoxelSize.xyz;
        float3 uv = volumePosition / (domain.maxAndInfluenceRadius.xyz - domain.minAndVoxelSize.xyz);
        uint3 mipDimensions;
        coarseMip.GetDimensions(mipDimensions.x, mipDimensions.y, mipDimensions.z);
        if (mipDimensions.z != mipDimensions.x) {
            float mipSize = float(mipDimensions.x);
            // A packed Z slice has its own clamp-to-edge boundary.
            uv.z = (clamp(uv.z * mipSize, 0.5f, mipSize - 0.5f) + float(clipmapIndex) * mipSize) /
                float(mipDimensions.z);
        }
        const float coarse = coarseMip.SampleLevel(sampler, uv, 0.0f);
        const float radius = domain.maxAndInfluenceRadius.w;
        distance = decodeGlobalDistanceFieldPageDistance(coarse, float(kGlobalDistanceFieldMipFactor) * radius);

        float3 pagePosition;
        uint pageIndex = loadGlobalDistanceFieldPage(pageTable, volumePosition / domain.minAndVoxelSize.w, pagePosition,
            clipmapIndex);
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

// UE 5.7.4 GlobalDistanceFieldUtils.ush:15 and Common.ush:1931.
uint computeGlobalDistanceFieldClipmapIndex(StructuredBuffer<GlobalDistanceFieldDomain> domains, uint clipmapCount,
    float3 worldPosition)
{
    for (uint clipmapIndex = 0u; clipmapIndex < clipmapCount; ++clipmapIndex) {
        GlobalDistanceFieldDomain domain = domains[clipmapIndex];
        float3 center = (domain.minAndVoxelSize.xyz + domain.maxAndInfluenceRadius.xyz) * 0.5f;
        float3 extent = (domain.maxAndInfluenceRadius.xyz - domain.minAndVoxelSize.xyz) * 0.5f;
        float3 distancesToMin = max(worldPosition - center + extent, 0.0f);
        float3 distancesToMax = max(center + extent - worldPosition, 0.0f);
        float3 closestDistances = min(distancesToMin, distancesToMax);
        float distanceFromClipmap = min(closestDistances.x, min(closestDistances.y, closestDistances.z));
        if (distanceFromClipmap > domain.minAndVoxelSize.w * 0.5f) {
            return clipmapIndex;
        }
    }
    return 0u;
}

// UE 5.7.4 GlobalDistanceFieldUtils.ush:90. Search from fine to coarse and resume at the previous exit.
public GlobalDistanceFieldTrace traceGlobalDistanceFieldClipmaps(QuantizedGlobalDistanceFieldSampler distanceField,
    StructuredBuffer<GlobalDistanceFieldDomain> domains, uint clipmapCount, float3 rayStart, float3 rayDirection,
    float minTraceDistance, float maxTraceDistance, out uint hitClipmapIndex,
    float voxelSizeRelativeBias = 0.0f, float voxelSizeRelativeRayEndBias = 0.0f)
{
    GlobalDistanceFieldTrace result = {-1.0f, 0u, kGlobalDistanceFieldOutside, 0.0f};
    hitClipmapIndex = ~0u;
    uint minClipmapIndex = computeGlobalDistanceFieldClipmapIndex(domains, clipmapCount,
        rayStart + minTraceDistance * rayDirection);
    float minRayTime = minTraceDistance;
    bool fullyCovered = true;

    for (uint clipmapIndex = minClipmapIndex; clipmapIndex < clipmapCount; ++clipmapIndex) {
        QuantizedGlobalDistanceFieldSampler clipmap = distanceField;
        clipmap.domain = domains[clipmapIndex];
        clipmap.clipmapIndex = clipmapIndex;
        float voxelExtent = clipmap.domain.minAndVoxelSize.w * 0.5f;
        float clipmapRayBias = voxelExtent * voxelSizeRelativeBias;
        float clipmapRayLength = maxTraceDistance - voxelExtent * voxelSizeRelativeRayEndBias;
        if (clipmapRayLength <= minRayTime) {
            break;
        }
        float2 intersectionTimes = clipGlobalDistanceFieldRay(clipmap.domain, rayStart, rayDirection, clipmapRayLength);
        intersectionTimes.x = max(intersectionTimes.x, minRayTime);
        intersectionTimes.x = max(intersectionTimes.x, clipmapRayBias);
        if (!(intersectionTimes.x < intersectionTimes.y)) {
            continue;
        }

        fullyCovered = fullyCovered && intersectionTimes.x <= max(minRayTime, clipmapRayBias);
        minRayTime = intersectionTimes.y;
        bool reachedSegmentEnd;
        GlobalDistanceFieldTrace segment = traceGlobalDistanceFieldSegment(clipmap, voxelExtent,
            rayStart, rayDirection, intersectionTimes, reachedSegmentEnd, clipmapRayBias);
        result.stepCount += segment.stepCount;
        if (segment.classification == kGlobalDistanceFieldHit) {
            hitClipmapIndex = clipmapIndex;
            result.hitDistance = segment.hitDistance;
            result.surfaceExpansion = segment.surfaceExpansion;
            result.classification = kGlobalDistanceFieldHit;
            return result;
        }
        if (!reachedSegmentEnd) {
            result.classification = kGlobalDistanceFieldUnavailable;
            return result;
        }
        result.classification = kGlobalDistanceFieldPartial;
    }

    if (result.classification == kGlobalDistanceFieldPartial && fullyCovered && minRayTime >= maxTraceDistance) {
        result.classification = kGlobalDistanceFieldMiss;
    }
    return result;
}
} // namespace Metallic.DistanceField
