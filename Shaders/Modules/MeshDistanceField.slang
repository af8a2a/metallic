module MeshDistanceField;

namespace Metallic.DistanceField
{
public static const uint kMeshDistanceFieldMipCount = 3u;
public static const uint kMeshDistanceFieldMaxTraceSteps = 64u;
static const uint kUniqueBrickSize = 7u;
static const uint kBrickSize = 8u;
static const uint kInvalidBrick = 0xffffffffu;

public struct MeshDistanceFieldMip
{
    public uint4 dimensionsAndIndirectionOffset;
    public float4 volumeToVirtualUVScaleAndDistanceScale;
    public float4 volumeToVirtualUVAddAndDistanceBias;
};

public struct MeshDistanceFieldInstance
{
    public float4 worldToVolume0;
    public float4 worldToVolume1;
    public float4 worldToVolume2;
    public float4 volumeExtentAndWorldScale;
    public uint4 indices; // Finest mip record index, debug instance ID, scene object ID, reserved.
};

public struct MeshDistanceFieldResources
{
    public StructuredBuffer<MeshDistanceFieldMip> mips;
    public StructuredBuffer<uint> indirection;
    public Texture3D<float> atlas;
    public SamplerState atlasSampler;
    public uint bricksPerAxis;
};

public float3 meshDistanceFieldWorldToVolume(MeshDistanceFieldInstance instance, float3 worldPosition)
{
    float4 position = float4(worldPosition, 1.0f);
    return float3(dot(instance.worldToVolume0, position), dot(instance.worldToVolume1, position),
        dot(instance.worldToVolume2, position));
}

float3 volumeToIndirectionScale(MeshDistanceFieldMip mip)
{
    return mip.volumeToVirtualUVScaleAndDistanceScale.xyz * float3(mip.dimensionsAndIndirectionOffset.xyz);
}

float maxEncodedDistance(MeshDistanceFieldMip mip)
{
    return mip.volumeToVirtualUVScaleAndDistanceScale.w + mip.volumeToVirtualUVAddAndDistanceBias.w;
}

// UE 5.7.4 DistanceFieldLightingShared.ush:392, linear 32-bit indirection table.
public float sampleSparseMeshSignedDistanceField(MeshDistanceFieldResources resources,
    float3 sampleVolumePosition, MeshDistanceFieldMip mip)
{
    uint3 dimensions = mip.dimensionsAndIndirectionOffset.xyz;
    float3 indirectionPos = sampleVolumePosition * volumeToIndirectionScale(mip) +
        mip.volumeToVirtualUVAddAndDistanceBias.xyz * float3(dimensions);
    int3 indirectionCoord = int3(indirectionPos);
    uint indirectionIndex = (uint(indirectionCoord.z) * dimensions.y + uint(indirectionCoord.y)) *
        dimensions.x + uint(indirectionCoord.x);
    uint brickIndex = resources.indirection[mip.dimensionsAndIndirectionOffset.w + indirectionIndex];
    float distanceField = maxEncodedDistance(mip);
    if (brickIndex != kInvalidBrick) {
        float3 brickLocalUV = indirectionPos - float3(indirectionCoord);
        uint3 brickOffset = uint3(brickIndex % resources.bricksPerAxis,
            (brickIndex / resources.bricksPerAxis) % resources.bricksPerAxis,
            brickIndex / (resources.bricksPerAxis * resources.bricksPerAxis));
        float atlasTexelSize = 1.0f / float(resources.bricksPerAxis * kBrickSize);
        float3 atlasUV = float3(brickOffset) * (float(kBrickSize) * atlasTexelSize) +
            brickLocalUV * (float(kUniqueBrickSize) * atlasTexelSize) + 0.5f * atlasTexelSize;
        float encodedDistanceField = resources.atlas.SampleLevel(resources.atlasSampler, atlasUV, 0.0f);
        distanceField = encodedDistanceField * mip.volumeToVirtualUVScaleAndDistanceScale.w +
            mip.volumeToVirtualUVAddAndDistanceBias.w;
    }
    return distanceField;
}

// UE 5.7.4 MeshDistanceFieldCommon.ush:9 and DistanceFieldLightingShared.ush:452.
public float distanceToNearestMeshSurface(MeshDistanceFieldResources resources, MeshDistanceFieldInstance instance,
    float3 worldPosition, float maxDistance)
{
    float3 volumePosition = meshDistanceFieldWorldToVolume(instance, worldPosition);
    float3 extent = instance.volumeExtentAndWorldScale.xyz;
    float worldScale = instance.volumeExtentAndWorldScale.w;
    float3 clampedVolumePosition = clamp(volumePosition, -extent, extent);
    float3 toBox = (abs(volumePosition) - extent) * worldScale;
    float distanceToBoxOutside = length(max(0.0f, toBox));
    float distanceToBoxInside = min(0.0f, max(toBox.x, max(toBox.y, toBox.z)));
    float distanceToBox = distanceToBoxOutside + distanceToBoxInside;

    float distanceToOccluder = maxDistance;
    if (distanceToBox < maxDistance) {
        float distanceField = 0.0f;
        for (int mipIndex = int(kMeshDistanceFieldMipCount) - 1; mipIndex >= 0; --mipIndex) {
            MeshDistanceFieldMip mip = resources.mips[instance.indices.x + uint(mipIndex)];
            distanceField = sampleSparseMeshSignedDistanceField(resources, clampedVolumePosition, mip);
            if (abs(distanceField) > 0.25f * maxEncodedDistance(mip)) {
                break;
            }
        }
        distanceToOccluder = distanceField * worldScale + max(0.0f, distanceToBox);
        distanceToOccluder = max(distanceToOccluder, distanceToBox);
    }
    return distanceToOccluder;
}

// UE 5.7.4 Common.ush:1889. Returned times parameterize the finite segment.
public float2 lineBoxIntersect(float3 rayOrigin, float3 rayEnd, float3 boxMin, float3 boxMax)
{
    float3 invRayDir = 1.0f / (rayEnd - rayOrigin);
    float3 firstPlaneIntersections = (boxMin - rayOrigin) * invRayDir;
    float3 secondPlaneIntersections = (boxMax - rayOrigin) * invRayDir;
    float3 closestPlaneIntersections = min(firstPlaneIntersections, secondPlaneIntersections);
    float3 furthestPlaneIntersections = max(firstPlaneIntersections, secondPlaneIntersections);
    return saturate(float2(
        max(closestPlaneIntersections.x, max(closestPlaneIntersections.y, closestPlaneIntersections.z)),
        min(furthestPlaneIntersections.x, min(furthestPlaneIntersections.y, furthestPlaneIntersections.z))));
}

// UE 5.7.4 LumenSoftwareRayTracing.ush:124, full coverage, finest mip, uniform instance scale.
// hitDistance is initialized to the maximum ray distance and shortened only by accepted hits.
public bool rayTraceSingleMeshSDF(MeshDistanceFieldResources resources, MeshDistanceFieldInstance instance,
    float3 worldRayStart, float3 worldRayDirection, float minTraceDistance, float maxTraceDistance,
    bool expandUsingRayTime, float initialMaxDistance, inout float hitDistance, out uint stepCount)
{
    stepCount = 0u;
    MeshDistanceFieldMip finestMip = resources.mips[instance.indices.x];
    float3 extent = instance.volumeExtentAndWorldScale.xyz;
    // Default r.DistanceFields.SurfaceBiasExpand and the renderer's finest-mip bias calculation.
    float volumeSurfaceBias = length(0.25f * extent /
        float3(finestMip.dimensionsAndIndirectionOffset.xyz * kUniqueBrickSize));
    maxTraceDistance = min(maxTraceDistance, hitDistance + volumeSurfaceBias);

    float3 worldRayEnd = worldRayStart + worldRayDirection * maxTraceDistance;
    float3 volumeRayStart = meshDistanceFieldWorldToVolume(instance, worldRayStart);
    float3 volumeRayEnd = meshDistanceFieldWorldToVolume(instance, worldRayEnd);
    float3 volumeRayDirection = volumeRayEnd - volumeRayStart;

    float volumeMaxTraceDistance = length(volumeRayDirection);
    float volumeMinTraceDistance = volumeMaxTraceDistance * (minTraceDistance / maxTraceDistance);
    volumeRayDirection /= volumeMaxTraceDistance;

    float2 intersectionTimes = lineBoxIntersect(volumeRayStart, volumeRayEnd, -extent, extent);
    intersectionTimes *= volumeMaxTraceDistance;
    intersectionTimes.x = max(intersectionTimes.x, volumeMinTraceDistance);
    if (!(intersectionTimes.x < intersectionTimes.y)) {
        return false;
    }

    float sampleRayTime = intersectionTimes.x;
    float minStepSize = 1.0f / (16.0f * float(kMeshDistanceFieldMaxTraceSteps));
    uint stepIndex = 0u;
    bool hit = false;
    float maxDistance = initialMaxDistance;
    for (; stepIndex < kMeshDistanceFieldMaxTraceSteps; ++stepIndex) {
        float3 sampleVolumePosition = volumeRayStart + volumeRayDirection * sampleRayTime;
        float distanceField = sampleSparseMeshSignedDistanceField(resources, sampleVolumePosition, finestMip);
        ++stepCount;

        maxDistance = max(distanceField, maxDistance);
        float expandSurfaceTime = expandUsingRayTime ? sampleRayTime : maxDistance;
        float expandSurfaceFalloff = 2.0f * volumeSurfaceBias;
        float expandSurfaceAmount = volumeSurfaceBias * saturate(expandSurfaceTime / expandSurfaceFalloff);
        if (distanceField < expandSurfaceAmount) {
            sampleRayTime = clamp(sampleRayTime + distanceField - expandSurfaceAmount,
                intersectionTimes.x, intersectionTimes.y);
            hit = true;
            break;
        }

        float stepDistance = max(distanceField, minStepSize);
        sampleRayTime += stepDistance;
        if (sampleRayTime > intersectionTimes.y + expandSurfaceAmount) {
            break;
        }
    }

    if (stepIndex == kMeshDistanceFieldMaxTraceSteps) {
        hit = true;
    }
    if (hit) {
        float newHitDistance = length(volumeRayDirection * sampleRayTime * instance.volumeExtentAndWorldScale.w);
        if (newHitDistance < hitDistance) {
            hitDistance = newHitDistance;
            return true;
        }
    }
    return false;
}

// UE 5.7.4 DistanceFieldLightingShared.ush:476 and LumenSoftwareRayTracing.ush:278.
public float3 calculateMeshSDFWorldNormal(MeshDistanceFieldResources resources, MeshDistanceFieldInstance instance,
    float3 hitWorldPosition)
{
    float3 extent = instance.volumeExtentAndWorldScale.xyz;
    float3 p = clamp(meshDistanceFieldWorldToVolume(instance, hitWorldPosition), -extent, extent);
    MeshDistanceFieldMip mip = resources.mips[instance.indices.x];
    float3 volumeSpaceVoxelSize = (1.0f / float(kUniqueBrickSize)) / volumeToIndirectionScale(mip);
    float3 offset = 0.5f * volumeSpaceVoxelSize;
    float right = sampleSparseMeshSignedDistanceField(resources, float3(p.x + offset.x, p.y, p.z), mip);
    float left = sampleSparseMeshSignedDistanceField(resources, float3(p.x - offset.x, p.y, p.z), mip);
    float front = sampleSparseMeshSignedDistanceField(resources, float3(p.x, p.y + offset.y, p.z), mip);
    float back = sampleSparseMeshSignedDistanceField(resources, float3(p.x, p.y - offset.y, p.z), mip);
    float up = sampleSparseMeshSignedDistanceField(resources, float3(p.x, p.y, p.z + offset.z), mip);
    float down = sampleSparseMeshSignedDistanceField(resources, float3(p.x, p.y, p.z - offset.z), mip);
    float3 volumeGradient = float3(right - left, front - back, up - down);
    float volumeGradientLength = length(volumeGradient);
    float3 volumeNormal = volumeGradientLength > 0.0f ? volumeGradient / volumeGradientLength : 0.0f;
    float3 worldGradient = mul(volumeNormal, float3x3(instance.worldToVolume0.xyz,
        instance.worldToVolume1.xyz, instance.worldToVolume2.xyz));
    float worldGradientLength = length(worldGradient);
    return worldGradientLength > 0.0f ? worldGradient / worldGradientLength : 0.0f;
}
} // namespace Metallic.DistanceField
