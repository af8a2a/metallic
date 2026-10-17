implementing ScenePathTracing;

import Core;
import MaterialTextureDecode;
import RTXCRMaterialAdapter;
import RayMaterialExecution;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import StreamRayGeometry;
#if METALLIC_PT_RAY_PIPELINE
import StreamRayHit;
#endif
using Metallic.StreamGeometry;

namespace Metallic.ScenePathTracing
{
#include "../../Interop/MeshletStreamBLAS.h"

public bool streamRayGeometryChanged()
{
#if SHARC_UPDATE
    StructuredBuffer<uint> changes = resolveBuffer<StructuredBuffer<uint>>(getSceneResourceParameters().rayStreamChanges);
#else
    StructuredBuffer<uint> changes = resolveBuffer<StructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamChangesOffset));
#endif
    return changes[METALLIC_STREAM_BLAS_SEMANTIC_CHANGE_WORD] != 0u;
}

public float3 accumulateStreamPathTrace(uint2 pixel, float3 color, ScenePathTracePush push)
{
    // Actual cut changes reset globally, including offscreen geometry that can
    // affect reflections and shadows. AS address changes do not change radiance.
    // Store the effective frame count in history so a GPU reset remains valid
    // on the following frame, independently of the CPU accumulation counter.
    float count = 1.0;
    if (push.hasHistory != 0u && !streamRayGeometryChanged()) {
        float4 previous = sceneHistoryPrevious()[pixel];
        count = min(previous.a + 1.0, 16777216.0);
        color = lerp(previous.rgb, color, rcp(count));
    }
    sceneHistoryCurrent()[pixel] = float4(color, count);
    return color;
}

public bool streamRayAlphaAccepts(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, ScenePathTracePush push)
{
    uint materialIndex;
    if (!streamRayMaterialIndex(instanceIndex, materialIndex)) { return false; }
    uint count, stride;
    sceneMaterials().GetDimensions(count, stride);
    if (materialIndex >= count) { return false; }
    MaterialCoverageParameters material = sceneMaterialCoverageParameters(loadSceneMaterial(materialIndex));
    if (material.alphaMode < 0.5) { return true; }
    HitInfo hit;
    if (!streamRaySurface(instanceIndex, clusterId, triangle, bary, ray, t, push.bitangentFlip, hit)) { return false; }
    return evaluateSceneCoverage(materialIndex, material, hit.texcoord, push).accepted;
}

public bool traceStreamClosest(Ray ray, float nearDistance, float farDistance, ScenePathTracePush push,
    out HitInfo hit, bool occlusionOnly = false)
{
    hit = (HitInfo)0;
    RayDesc desc;
    desc.Origin = ray.origin; desc.Direction = ray.direction;
    desc.TMin = nearDistance; desc.TMax = farDistance;
    // Force candidate callbacks even for CLAS built with the opaque default.
#if METALLIC_PT_RAY_PIPELINE
    StreamRayHit payload = (StreamRayHit)0;
    payload.t = -1.0;
    // One shared triangle hit group and miss record. Keep full closest-hit
    // traversal even for shadows: transparent media are integrated in order
    // by the existing caller, never accumulated in unordered any-hit callbacks.
    TraceRay(sceneAccelerationStructure(), RAY_FLAG_FORCE_NON_OPAQUE, 0xff, 0, 0, 0, desc, payload);
    if (payload.t < 0.0) { return false; }
    if (occlusionOnly) { return true; }
    return streamRaySurface(payload.instanceId, payload.clusterId, payload.primitiveId,
        payload.barycentrics, ray, payload.t, push.bitangentFlip, hit);
#else
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(sceneAccelerationStructure(), RAY_FLAG_FORCE_NON_OPAQUE, 0xff, desc);
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
#endif
}
} // namespace Metallic.ScenePathTracing
