// CLAS IDs are pageIndex * maxPageClusters + localCluster. IDs survive arena moves.
import GPUDriven;
using Metallic.GPUDriven;
#include "../VisibilityBuffer/VisibilityStreamDecode.slang"
struct StreamRayInstance
{
    uint primitiveIndex, materialIndex, visible, gpuSceneInstanceIndex;
    float4 world0, world1, world2, world3;
    float4 bounds;
};
#define gRayStreamPages getResource<StructuredBuffer<uint>>(90)
#define gRayStreamPageTable getResource<StructuredBuffer<uint2>>(91)
#define gRayStreamInstances getResource<StructuredBuffer<StreamRayInstance>>(92)
#define gRayStreamHeader getResource<StructuredBuffer<uint>>(93)

float3 streamRayVector(StreamRayInstance instance, float3 v)
{
    return instance.world0.xyz * v.x + instance.world1.xyz * v.y + instance.world2.xyz * v.z;
}
float3 streamRayNormal(StreamRayInstance instance, float3 n)
{
    float3 c0 = cross(instance.world1.xyz, instance.world2.xyz);
    float3 c1 = cross(instance.world2.xyz, instance.world0.xyz);
    float3 c2 = cross(instance.world0.xyz, instance.world1.xyz);
    return safeNormalize((c0 * n.x + c1 * n.y + c2 * n.z) *
        (dot(instance.world0.xyz, c0) < 0.0 ? -1.0 : 1.0), n);
}

bool streamRaySurface(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, ScenePathTracePush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    uint count, stride;
    gRayStreamInstances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    StreamRayInstance instance = gRayStreamInstances[instanceIndex];
    gMaterials.GetDimensions(count, stride);
    if (instance.materialIndex >= count) { return false; }
    uint clusterStride = gRayStreamHeader[2];
    if (clusterStride == 0u) { return false; }
    uint pageIndex = clusterId / clusterStride, cluster = clusterId % clusterStride;
    float3 a, b, c;
    if (!decodeStreamTriangleAtPage(pageIndex, cluster, triangle, gRayStreamPages, gRayStreamPageTable, streamPageWordCount(getResource<ByteAddressBuffer>(94)), a, b, c)) { return false; }
    float3 n[3]; float2 uv[3]; float4 tangents[3];
    uint attributes = decodeStreamAttributesAtPage(pageIndex, cluster, triangle, gRayStreamPages, gRayStreamPageTable, n, uv, tangents);
    float3 weights = float3(1.0 - bary.x - bary.y, bary);
    float3 localGeometry = safeTriangleNormal(b - a, c - a, float3(0, 1, 0));
    float3 localNormal = safeNormalize(n[0] * weights.x + n[1] * weights.y + n[2] * weights.z, localGeometry);
    hit.normal = streamRayNormal(instance, localNormal);
    hit.geometryNormal = streamRayNormal(instance, localGeometry);
    hit.texcoord = uv[0] * weights.x + uv[1] * weights.y + uv[2] * weights.z;
    float3 wt = streamRayVector(instance, b - a), wb = streamRayVector(instance, c - a);
    float uvArea = abs((uv[1].x - uv[0].x) * (uv[2].y - uv[0].y) - (uv[1].y - uv[0].y) * (uv[2].x - uv[0].x));
    hit.rayConeLodConstant = uvArea > 1e-7 ? 0.5 * log2(uvArea / max(length(cross(wt, wb)), 1e-20)) : 0.0;
    float3 triangleTangent, triangleBitangent;
    computeTriangleTangentSpace(float3(0), wt, wb, uv[0], uv[1], uv[2], hit.normal, triangleTangent, triangleBitangent);
    float3 axis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 fallback = safeNormalize(cross(axis, hit.normal), float3(1, 0, 0));
    float3 tangent = triangleTangent;
    float sign;
    if ((attributes & 16u) != 0u) {
        float4 authored = tangents[0] * weights.x + tangents[1] * weights.y + tangents[2] * weights.z;
        tangent = streamRayVector(instance, authored.xyz);
        sign = (authored.w < 0.0 ? -1.0 : 1.0) *
            (dot(instance.world0.xyz, cross(instance.world1.xyz, instance.world2.xyz)) < 0.0 ? -1.0 : 1.0);
    } else { sign = dot(cross(hit.normal, tangent), triangleBitangent) < 0.0 ? -1.0 : 1.0; }
    tangent = safeNormalize(tangent - hit.normal * dot(hit.normal, tangent), fallback);
    hit.tangent = float4(tangent, sign * push.bitangentFlip);
    hit.bitangent = cross(hit.normal, tangent) * hit.tangent.w;
    hit.position = ray.origin + ray.direction * t;
    hit.t = t;
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;
    hit.materialIndex = instance.materialIndex;
    hit.instanceIndex = instance.gpuSceneInstanceIndex;
    hit.primitiveIndex = instance.primitiveIndex;
    hit.triangleIndex = triangle;
    return true;
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
    PathTraceMaterial material = gMaterials[materialIndex];
    if (material.textureParams.w < 0.5) { return true; }
    HitInfo hit;
    if (!streamRaySurface(instanceIndex, clusterId, triangle, bary, ray, t, push, hit)) { return false; }
    float alpha = material.baseColor.a * loadMaterialAlphaCoverage(material.baseColorTexture, hit.texcoord, push);
    return material.textureParams.w < 1.5 ? alpha >= material.params.z : alpha > 0.0;
}

bool traceStreamClosest(Ray ray, float nearDistance, float farDistance, ScenePathTracePush push, out HitInfo hit)
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
    return streamRaySurface(query.CommittedInstanceID(), uint(query.CommittedClusterID()),
        query.CommittedPrimitiveIndex(), query.CommittedTriangleBarycentrics(), ray, query.CommittedRayT(), push, hit);
}
