#include "StreamRayGeometry.slang"
#include "../../Interop/MeshletStreamBLAS.h"

bool streamRayGeometryChanged()
{
#if SHARC_UPDATE
    StructuredBuffer<uint> changes = resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().rayStreamChanges);
#else
    StructuredBuffer<uint> changes = resolveBuffer<StructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamChangesOffset));
#endif
    return changes[METALLIC_STREAM_BLAS_SEMANTIC_CHANGE_WORD] != 0u;
}

float3 accumulateStreamPathTrace(uint2 pixel, float3 color, ScenePathTracePush push)
{
    // Actual cut changes reset globally, including offscreen geometry that can
    // affect reflections and shadows. AS address changes do not change radiance.
    // Store the effective frame count in history so a GPU reset remains valid
    // on the following frame, independently of the CPU accumulation counter.
    float count = 1.0;
    if (push.hasHistory != 0u && !streamRayGeometryChanged()) {
        float4 previous = gHistoryPrevious[pixel];
        count = min(previous.a + 1.0, 16777216.0);
        color = lerp(previous.rgb, color, rcp(count));
    }
    gHistoryCurrent[pixel] = float4(color, count);
    return color;
}

bool streamRayAlphaAccepts(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, ScenePathTracePush push)
{
    uint count, stride;
    gRayStreamInstances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    uint materialIndex = gRayStreamInstances[instanceIndex].materialIndex;
    gMaterials.GetDimensions(count, stride);
    if (materialIndex >= count) { return false; }
    PathTraceMaterial material = loadSceneMaterial(materialIndex);
    if (material.textureParams.w < 0.5) { return true; }
    HitInfo hit;
    if (!streamRaySurface(instanceIndex, clusterId, triangle, bary, ray, t, push.bitangentFlip, hit)) { return false; }
    return evaluateSceneCoverage(materialIndex, material, hit.texcoord, push).accepted;
}

bool traceStreamClosest(Ray ray, float nearDistance, float farDistance, ScenePathTracePush push,
    out HitInfo hit, bool occlusionOnly = false)
{
    hit = (HitInfo)0;
    RayDesc desc;
    desc.Origin = ray.origin; desc.Direction = ray.direction;
    desc.TMin = nearDistance; desc.TMax = farDistance;
    // Force candidate callbacks even for CLAS built with the opaque default.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(gScene, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, desc);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            streamRayAlphaAccepts(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), ray, query.CandidateTriangleRayT(), push)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
    // Coverage is already accepted; shadow callers need no committed surface.
    if (occlusionOnly) { return true; }
    return streamRaySurface(query.CommittedInstanceID(), uint(query.CommittedClusterID()),
        query.CommittedPrimitiveIndex(), query.CommittedTriangleBarycentrics(), ray, query.CommittedRayT(), push.bitangentFlip, hit);
}
