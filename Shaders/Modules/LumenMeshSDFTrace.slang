module LumenMeshSDFTrace;

import LumenSurfaceCache;
import MeshDistanceField;

using Metallic.DistanceField;

namespace Metallic.Lumen
{
public struct LumenMeshSDFTraceScene
{
    public MeshDistanceFieldResources meshes;
    public StructuredBuffer<MeshDistanceFieldInstance> instances;
    public StructuredBuffer<uint> candidateIndices;
    public LumenSurfaceCache cache;
};

public struct LumenMeshSDFTraceInput
{
    public float3 origin;
    public float3 direction;
    public float minTraceDistance;
    public float maxTraceDistance;
    public float voxelTraceStartDistance;
    public uint meshSDFStartOffset;
    public uint numMeshSDFs;
    public bool expandUsingRayTime;
};

public struct LumenMeshSDFTraceResult
{
    public float3 radiance;
    public float transparency;
    public float hitDistance;
    public float3 normal;
    public bool cardCovered;
    public uint instanceIndex;
};

public struct LumenMeshSDFInstanceMotion
{
    public float4 previousVolumeToWorld0;
    public float4 previousVolumeToWorld1;
    public float4 previousVolumeToWorld2;
};

public float3 lumenMeshSDFPreviousPosition(LumenMeshSDFInstanceMotion motion, float3 volumePosition)
{
    float4 position = float4(volumePosition, 1.0f);
    return float3(dot(motion.previousVolumeToWorld0, position), dot(motion.previousVolumeToWorld1, position),
        dot(motion.previousVolumeToWorld2, position));
}

float3 sampleLumenMeshSDFRadiance(LumenSurfaceCache cache, MeshDistanceFieldInstance instance,
    float3 hitPosition, float3 hitNormal, out bool cardCovered)
{
    // The fixed-resolution atlas uses a single resident tile per card.
    const float surfaceCacheBias = 0.20f; // 20 cm in UE.
    LumenCardSampleAccumulator accumulator = initLumenCardSampleAccumulator();
    sampleLumenMeshCards(cache, instance.indices.z, hitPosition, hitNormal, surfaceCacheBias, accumulator);
    cardCovered = accumulator.cardIndex != ~0u;
    return evaluateLumenCardRadiance(accumulator);
}

public struct LumenMeshSDFValidationBounds
{
    public float4 minimum;
    public float4 maximum;
};

// The prefix verifier uses actual local mesh bounds and every resident instance.
public bool validateLumenMeshSDFPrefix(MeshDistanceFieldResources meshes,
    StructuredBuffer<MeshDistanceFieldInstance> instances, StructuredBuffer<LumenMeshSDFValidationBounds> bounds,
    uint instanceCount, LumenSurfaceCache cache,
    float3 origin, float3 direction, float minDistance, float maxDistance,
    out float hitDistance, out float3 radiance, out bool inconclusive)
{
    hitDistance = maxDistance;
    radiance = 0.0f;
    float earliestUnconfirmed = maxDistance;
    uint hitObject = ~0u;
    const float3 end = origin + direction * maxDistance;
    for (uint objectIndex = 0u; objectIndex < instanceCount; ++objectIndex) {
        const MeshDistanceFieldInstance instance = instances[objectIndex];
        const LumenMeshSDFValidationBounds objectBounds = bounds[objectIndex];
        const float2 interval = lineBoxIntersect(meshDistanceFieldWorldToVolume(instance, origin),
            meshDistanceFieldWorldToVolume(instance, end), objectBounds.minimum.xyz, objectBounds.maximum.xyz) * maxDistance;
        const float entry = max(minDistance, interval.x);
        if (entry > interval.y || entry >= hitDistance) {
            continue;
        }
        uint stepCount;
        if (rayTraceSingleMeshSDF(meshes, instance, origin, direction, entry, interval.y,
                false, 0.0f, hitDistance, stepCount, true)) {
            hitObject = objectIndex;
        } else {
            // The baked field can miss an edge even when the original mesh bounds intersect.
            earliestUnconfirmed = min(earliestUnconfirmed, entry);
        }
    }
    inconclusive = earliestUnconfirmed < maxDistance && earliestUnconfirmed <= hitDistance;
    if (inconclusive || hitObject == ~0u) {
        return false;
    }
    const MeshDistanceFieldInstance instance = instances[hitObject];
    const float3 hitPosition = origin + direction * hitDistance;
    const float3 hitNormal = calculateMeshSDFWorldNormal(meshes, instance, hitPosition);
    bool cardCovered;
    radiance = sampleLumenMeshSDFRadiance(cache, instance, hitPosition, hitNormal, cardCovered);
    return true;
}

// UE 5.7.4 LumenSoftwareRayTracing.ush:589,828. The receiver supplies the
// candidates and the opaque, uniform-scale Mesh SDF path uses its finest mip.
public LumenMeshSDFTraceResult traceLumenMeshSDFs(LumenMeshSDFTraceScene scene, LumenMeshSDFTraceInput input)
{
    LumenMeshSDFTraceResult result = (LumenMeshSDFTraceResult)0;
    result.transparency = 1.0f;
    result.hitDistance = input.maxTraceDistance;

    if (input.voxelTraceStartDistance > input.minTraceDistance) {
        const float maxTraceDistance = input.voxelTraceStartDistance;
        float hitDistance = maxTraceDistance;
        uint hitObject = 0u;
        for (uint gridCulledMeshSDFIndex = 0u; gridCulledMeshSDFIndex < input.numMeshSDFs; ++gridCulledMeshSDFIndex) {
            const uint objectIndex = scene.candidateIndices[input.meshSDFStartOffset + gridCulledMeshSDFIndex];
            uint stepCount;
            if (rayTraceSingleMeshSDF(scene.meshes, scene.instances[objectIndex], input.origin, input.direction,
                    input.minTraceDistance, maxTraceDistance, input.expandUsingRayTime, 0.0f, hitDistance, stepCount)) {
                hitObject = objectIndex;
            }
        }

        if (hitDistance < maxTraceDistance) {
            const MeshDistanceFieldInstance instance = scene.instances[hitObject];
            const float3 interpolatePosition = input.origin + input.direction * hitDistance;
            result.normal = calculateMeshSDFWorldNormal(scene.meshes, instance, interpolatePosition);

            result.radiance = sampleLumenMeshSDFRadiance(scene.cache, instance, interpolatePosition, result.normal,
                result.cardCovered);
            result.transparency = 0.0f;
            result.instanceIndex = hitObject;
        }

        result.hitDistance = hitDistance;
    }

    return result;
}
} // namespace Metallic.Lumen
