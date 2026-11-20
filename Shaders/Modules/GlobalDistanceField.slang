module GlobalDistanceField;

import MeshDistanceField;

namespace Metallic.DistanceField
{
public static const uint kGlobalDistanceFieldMaxTraceSteps = 256u;
public static const uint kGlobalDistanceFieldOutside = 0u;
public static const uint kGlobalDistanceFieldMiss = 1u;
public static const uint kGlobalDistanceFieldHit = 2u;
public static const uint kGlobalDistanceFieldUnavailable = 3u;

public struct GlobalDistanceFieldDomain
{
    public float4 minAndVoxelSize;
    public float4 maxAndInfluenceRadius;
};

public struct GlobalDistanceFieldTrace
{
    public float hitDistance;
    public uint stepCount;
    public uint classification;
    public float surfaceExpansion;
};

public interface IGlobalDistanceFieldSampler
{
    bool sample(float3 worldPosition, out float distance);
};

public float3 globalDistanceFieldVoxelCenter(GlobalDistanceFieldDomain domain, uint3 texel)
{
    // UE's half-voxel composition bias cancels the half-voxel physical page border.
    return float3(texel) * domain.minAndVoxelSize.w + domain.minAndVoxelSize.xyz;
}

// UE 5.7.4 GlobalDistanceFieldCompositeObjects.usf:46, opaque objects without a coverage atlas.
public float compositeGlobalDistanceField(MeshDistanceFieldResources meshes,
    StructuredBuffer<MeshDistanceFieldInstance> instances, uint instanceCount,
    float3 worldPosition, float influenceRadius)
{
    float minDistance = influenceRadius;
    for (uint index = 0u; index < instanceCount; ++index) {
        float maxQueryDistance = min(influenceRadius, max(minDistance, 0.0f));
        float distanceToOccluder = distanceToNearestMeshSurface(meshes, instances[index], worldPosition, maxQueryDistance);
        minDistance = min(minDistance, distanceToOccluder);
    }
    // UE's page encode/decode saturates to this range. Keep the distances in meters without UNORM8 rounding.
    return clamp(minDistance, -influenceRadius, influenceRadius);
}

// The ray clip and callers' coverage tests own the fixed volume bounds.
public float sampleGlobalDistanceField(Texture3D<float> distanceField, SamplerState sampler,
    GlobalDistanceFieldDomain domain, float3 worldPosition)
{
    float voxelSize = domain.minAndVoxelSize.w;
    uint3 physicalDimensions;
    distanceField.GetDimensions(physicalDimensions.x, physicalDimensions.y, physicalDimensions.z);
    float3 uv = ((worldPosition - domain.minAndVoxelSize.xyz) / voxelSize + 0.5f) / float3(physicalDimensions);
    return distanceField.SampleLevel(sampler, uv, 0.0f);
}

public struct DenseGlobalDistanceFieldSampler : IGlobalDistanceFieldSampler
{
    public Texture3D<float> distanceField;
    public SamplerState sampler;
    public GlobalDistanceFieldDomain domain;

    public bool sample(float3 worldPosition, out float distance)
    {
        distance = sampleGlobalDistanceField(distanceField, sampler, domain, worldPosition);
        return true;
    }
};

// UE 5.7.4 GlobalDistanceFieldUtils.ush:90, one level and the opaque visualization settings.
public GlobalDistanceFieldTrace traceGlobalDistanceField<T : IGlobalDistanceFieldSampler>(T distanceField,
    GlobalDistanceFieldDomain domain, float3 rayStart, float3 rayDirection,
    float minTraceDistance, float maxTraceDistance)
{
    GlobalDistanceFieldTrace result = {-1.0f, 0u, kGlobalDistanceFieldOutside, 0.0f};
    float voxelExtent = domain.minAndVoxelSize.w * 0.5f;
    float3 rayEnd = rayStart + rayDirection * maxTraceDistance;
    float2 intersectionTimes = lineBoxIntersect(rayStart, rayEnd,
        domain.minAndVoxelSize.xyz + voxelExtent, domain.maxAndInfluenceRadius.xyz - voxelExtent);
    intersectionTimes *= maxTraceDistance;
    intersectionTimes.x = max(intersectionTimes.x, minTraceDistance);
    if (!(intersectionTimes.x < intersectionTimes.y)) {
        return result;
    }

    result.classification = kGlobalDistanceFieldMiss;
    float sampleRayTime = intersectionTimes.x;
    const float expandSurfaceFalloff = 2.0f * voxelExtent;
    for (uint step = 0u; step < kGlobalDistanceFieldMaxTraceSteps; ++step) {
        float3 worldPosition = rayStart + rayDirection * sampleRayTime;
        float distance;
        if (!distanceField.sample(worldPosition, distance)) {
            result.classification = kGlobalDistanceFieldUnavailable;
            break;
        }
        ++result.stepCount;
        float surfaceExpansion = voxelExtent * saturate(sampleRayTime / expandSurfaceFalloff);
        if (distance < surfaceExpansion) {
            result.hitDistance = max(sampleRayTime + distance - surfaceExpansion, 0.0f);
            result.classification = kGlobalDistanceFieldHit;
            result.surfaceExpansion = surfaceExpansion;
            break;
        }

        float stepDistance = max(distance, voxelExtent);
        sampleRayTime += stepDistance;
        if (sampleRayTime > intersectionTimes.y) {
            break;
        }
    }
    return result;
}

// UE 5.7.4 GlobalDistanceFieldShared.ush:291. Rescale the encoded gradient threshold for meters.
public float3 calculateGlobalDistanceFieldNormal<T : IGlobalDistanceFieldSampler>(T distanceField,
    GlobalDistanceFieldDomain domain, float3 worldPosition, float3 fallbackNormal)
{
    float extent = domain.minAndVoxelSize.w * 0.5f;
    float3 x = float3(extent, 0.0f, 0.0f);
    float3 y = float3(0.0f, extent, 0.0f);
    float3 z = float3(0.0f, 0.0f, extent);
    float right, left, front, back, up, down;
    if (!distanceField.sample(worldPosition + x, right) || !distanceField.sample(worldPosition - x, left) ||
        !distanceField.sample(worldPosition + y, front) || !distanceField.sample(worldPosition - y, back) ||
        !distanceField.sample(worldPosition + z, up) || !distanceField.sample(worldPosition - z, down)) {
        return fallbackNormal;
    }
    float3 gradient = float3(right - left, front - back, up - down);
    float gradientLength = length(gradient);
    float threshold = 0.001f * (2.0f * domain.maxAndInfluenceRadius.w);
    return gradientLength > threshold ? gradient / gradientLength : fallbackNormal;
}
} // namespace Metallic.DistanceField
