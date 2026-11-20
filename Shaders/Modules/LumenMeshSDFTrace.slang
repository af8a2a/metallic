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
};

public struct LumenMeshSDFTraceResult
{
    public float3 radiance;
    public float transparency;
    public float hitDistance;
    public float3 normal;
};

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
                    input.minTraceDistance, maxTraceDistance, true, 0.0f, hitDistance, stepCount)) {
                hitObject = objectIndex;
            }
        }

        if (hitDistance < maxTraceDistance) {
            const MeshDistanceFieldInstance instance = scene.instances[hitObject];
            const float3 interpolatePosition = input.origin + input.direction * hitDistance;
            result.normal = calculateMeshSDFWorldNormal(scene.meshes, instance, interpolatePosition);

            // The fixed-resolution atlas uses a single resident tile per card.
            const float surfaceCacheBias = 0.20f; // 20 cm in UE.
            LumenCardSampleAccumulator accumulator = initLumenCardSampleAccumulator();
            sampleLumenMeshCards(scene.cache, instance.indices.z, interpolatePosition, result.normal,
                surfaceCacheBias, accumulator);
            result.radiance = evaluateLumenCardRadiance(accumulator);
            result.transparency = 0.0f;
        }

        result.hitDistance = hitDistance;
    }

    return result;
}
} // namespace Metallic.Lumen
