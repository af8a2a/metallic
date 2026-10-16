#language slang 2026
module StreamRayGeometry;

import Core;
import GPUDriven;
import ScenePathTypes;
import ScenePathMath;

namespace Metallic.StreamGeometry
{
using Metallic.GPUDriven;

// CLAS IDs are pageIndex * maxPageClusters + localCluster. IDs survive arena moves.
struct StreamRayInstance
{
    uint primitiveIndex, materialIndex, visible, gpuSceneInstanceIndex;
    float4 world0, world1, world2, world3;
    float4 bounds;
};
#define gRayStreamPages resolveBuffer<StructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPagesOffset))
#define gRayStreamPageTable resolveBuffer<StructuredBuffer<uint2>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPageTableOffset))
#define gRayStreamInstances resolveBuffer<StructuredBuffer<StreamRayInstance>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamInstancesOffset))
#define gRayStreamHeader resolveBuffer<StructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamHeaderOffset))
#define gRayStreamPageCount resolveBuffer<ByteAddressBuffer>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneStreamPageCountOffset))

// Keep resource resolution lazy and preserve the original bounds-check/load order.
public bool streamRayMaterialIndex(uint instanceIndex, out uint materialIndex)
{
    materialIndex = 0u;
    uint count, stride;
    gRayStreamInstances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    materialIndex = gRayStreamInstances[instanceIndex].materialIndex;
    return true;
}

float3 streamRayTriangleNormal(float3 a, float3 b, float3 fallback)
{
    float3 n = cross(a, b);
    float n2 = dot(n, n);
    return n2 > dot(a, a) * dot(b, b) * 1e-12 && n2 > 0.0 ? n * rsqrt(n2) : fallback;
}

float3 streamRayVector(StreamRayInstance instance, float3 v)
{
    return instance.world0.xyz * v.x + instance.world1.xyz * v.y + instance.world2.xyz * v.z;
}
float3 streamRayNormal(StreamRayInstance instance, float3 n)
{
    float3 c0 = cross(instance.world1.xyz, instance.world2.xyz);
    float3 c1 = cross(instance.world2.xyz, instance.world0.xyz);
    float3 c2 = cross(instance.world0.xyz, instance.world1.xyz);
    return Metallic.ScenePath.safeNormalize((c0 * n.x + c1 * n.y + c2 * n.z) *
        (dot(instance.world0.xyz, c0) < 0.0 ? -1.0 : 1.0), n);
}

public bool streamRaySurface(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, float bitangentFlip, out HitInfo hit)
{
    hit = (HitInfo)0;
    uint count, stride;
    gRayStreamInstances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    StreamRayInstance instance = gRayStreamInstances[instanceIndex];
    uint clusterStride = gRayStreamHeader[2];
    if (clusterStride == 0u) { return false; }
    uint pageIndex = clusterId / clusterStride, cluster = clusterId % clusterStride;
    float3 a, b, c;
    if (!decodeStreamTriangleAtPage(pageIndex, cluster, triangle, gRayStreamPages, gRayStreamPageTable, streamPageWordCount(gRayStreamPageCount), a, b, c)) { return false; }
    float3 n[3]; float2 uv[3]; float4 tangents[3];
    uint attributes = decodeStreamAttributesAtPage(pageIndex, cluster, triangle, gRayStreamPages, gRayStreamPageTable, n, uv, tangents);
    float3 weights = float3(1.0 - bary.x - bary.y, bary);
    float3 localGeometry = streamRayTriangleNormal(b - a, c - a, float3(0, 1, 0));
    float3 localNormal = Metallic.ScenePath.safeNormalize(n[0] * weights.x + n[1] * weights.y + n[2] * weights.z, localGeometry);
    hit.normal = streamRayNormal(instance, localNormal);
    hit.geometryNormal = streamRayNormal(instance, localGeometry);
    hit.texcoord = uv[0] * weights.x + uv[1] * weights.y + uv[2] * weights.z;
    float3 wt = streamRayVector(instance, b - a), wb = streamRayVector(instance, c - a);
    float uvArea = abs((uv[1].x - uv[0].x) * (uv[2].y - uv[0].y) - (uv[1].y - uv[0].y) * (uv[2].x - uv[0].x));
    hit.rayConeLodConstant = uvArea > 1e-7 ? 0.5 * log2(uvArea / max(length(cross(wt, wb)), 1e-20)) : 0.0;
    float3 triangleTangent, triangleBitangent;
    Metallic.ScenePath.computeTriangleTangentSpace(float3(0), wt, wb, uv[0], uv[1], uv[2], hit.normal, triangleTangent, triangleBitangent);
    float3 axis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 fallback = Metallic.ScenePath.safeNormalize(cross(axis, hit.normal), float3(1, 0, 0));
    float3 tangent = triangleTangent;
    float sign;
    if ((attributes & 16u) != 0u) {
        float4 authored = tangents[0] * weights.x + tangents[1] * weights.y + tangents[2] * weights.z;
        tangent = streamRayVector(instance, authored.xyz);
        sign = (authored.w < 0.0 ? -1.0 : 1.0) *
            (dot(instance.world0.xyz, cross(instance.world1.xyz, instance.world2.xyz)) < 0.0 ? -1.0 : 1.0);
    } else { sign = dot(cross(hit.normal, tangent), triangleBitangent) < 0.0 ? -1.0 : 1.0; }
    tangent = Metallic.ScenePath.safeNormalize(tangent - hit.normal * dot(hit.normal, tangent), fallback);
    hit.tangent = float4(tangent, sign * bitangentFlip);
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


} // namespace Metallic.StreamGeometry
