#language slang 2026
module StreamRayGeometry;

import Core;
import GPUDriven;
import ScenePathTypes;
import ScenePathMath;
#include "../Interop/StreamCLASTessellation.h"

namespace Metallic.StreamGeometry
{
using Metallic.GPUDriven;

#include "../Interop/MeshletStreamBLAS.h"

// Production CLAS IDs use the asset-validated fixed stride. Explicit strides
// retain the generic decoder for direct-header fixtures.
public uint2 streamRayClusterAddress(uint clusterId, uint clusterStride = METALLIC_STREAM_CLAS_CLUSTER_STRIDE)
{
    if (clusterStride == METALLIC_STREAM_CLAS_CLUSTER_STRIDE) {
        return uint2(clusterId >> METALLIC_STREAM_CLAS_CLUSTER_SHIFT,
            clusterId & (METALLIC_STREAM_CLAS_CLUSTER_STRIDE - 1u));
    }
    uint pageIndex = clusterId / clusterStride;
    return uint2(pageIndex, clusterId - pageIndex * clusterStride);
}

// CLAS IDs are pageIndex * maxPageClusters + localCluster. IDs survive arena moves.
struct StreamRayInstance
{
    uint primitiveIndex, materialIndex, visible, gpuSceneInstanceIndex;
    float4 world0, world1, world2, world3;
    float4 bounds;
};
#define gRayStreamPages resolveBuffer<StructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPagesOffset))
#define gRayStreamPageTable resolveBuffer<StructuredBuffer<uint2>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPageTableOffset))
#define gRayStreamInstances resolveBuffer<StructuredBuffer<StreamRayInstance>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamInstancesOffset))
#define gRayStreamHeader resolveBuffer<StructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamHeaderOffset))
#define gRayStreamPageCount resolveBuffer<ByteAddressBuffer>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneStreamPageCountOffset))

// The optional sidecar is published in the same command sequence as the TLAS.
// Short legacy fixture headers have no sidecar and keep the base decoder.
bool streamRayTessellationTriangle(uint clusterId, uint triangle,
    out StreamCLASTessellationLeaf leaf, out float3 positions[3], out float3 domains[3])
{
    leaf = (StreamCLASTessellationLeaf)0;
    positions[0] = positions[1] = positions[2] = float3(0);
    domains[0] = domains[1] = domains[2] = float3(0);
    let changes = resolveBuffer<StructuredBuffer<uint>>(
        getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamChangesOffset));
    uint count, stride;
    changes.GetDimensions(count, stride);
    if (count < METALLIC_STREAM_BLAS_HEADER_WORDS) { return false; }
    uint64_t address = uint64_t(changes[METALLIC_STREAM_BLAS_TESS_ADDRESS_WORD]) |
        (uint64_t(changes[METALLIC_STREAM_BLAS_TESS_ADDRESS_WORD + 1u]) << 32u);
    if (address == 0u) { return false; }
    let geometry = *((StreamCLASTessellationGeometry*)address);
    if (clusterId < geometry.baseClusterId || clusterId - geometry.baseClusterId >= geometry.maxPatches) { return false; }
    uint index = clusterId - geometry.baseClusterId;
    leaf = ((StreamCLASTessellationLeaf*)geometry.leaves)[index];
    uint3 indices = ((uint3*)geometry.triangles)[index * METALLIC_TESS_TRIANGLES_PER_PATCH + triangle];
    for (uint i = 0u; i < 3u; ++i) {
        let vertex = ((StreamCLASTessellationVertex*)geometry.vertices)[index * METALLIC_TESS_VERTICES_PER_PATCH + indices[i]];
        positions[i] = float3(vertex.x, vertex.y, vertex.z);
        domains[i] = float3(1.0 - vertex.sourceU - vertex.sourceV, vertex.sourceU, vertex.sourceV);
    }
    return true;
}

// Keep resource resolution lazy and preserve the original bounds-check/load order.
public bool streamRayMaterialIndex(uint instanceIndex, out uint materialIndex)
{
    materialIndex = 0u;
    let instances = gRayStreamInstances;
    uint count, stride;
    instances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    materialIndex = instances[instanceIndex].materialIndex;
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

// Ray-query candidates reference the published CLAS cut, whose cluster stride
// and resident geometry remain valid until the frame's GPU completion.
public float2 streamRayTexcoord(uint clusterId, uint triangle, float2 bary,
    uint clusterStride = METALLIC_STREAM_CLAS_CLUSTER_STRIDE)
{
    StreamCLASTessellationLeaf leaf;
    float3 positions[3], domains[3];
    if (streamRayTessellationTriangle(clusterId, triangle, leaf, positions, domains)) {
        float3 weights = float3(1.0 - bary.x - bary.y, bary);
        bary = (domains[0] * weights.x + domains[1] * weights.y + domains[2] * weights.z).yz;
        clusterId = leaf.pageIndex * clusterStride + leaf.clusterIndex;
        triangle = leaf.sourceTriangle;
    }
    uint2 address = streamRayClusterAddress(clusterId, clusterStride);
    float2 uv[3];
    decodeStreamTexcoordsAtPage(address.x, address.y, triangle, gRayStreamPages, gRayStreamPageTable, uv);
    float3 weights = float3(1.0 - bary.x - bary.y, bary);
    return uv[0] * weights.x + uv[1] * weights.y + uv[2] * weights.z;
}

public bool streamRaySurface(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, float bitangentFlip, out HitInfo hit, bool shadowOnly = false,
    uint clusterStride = METALLIC_STREAM_CLAS_CLUSTER_STRIDE, bool committedHit = false)
{
    hit = (HitInfo)0;
    let instances = gRayStreamInstances;
    uint count, stride;
    instances.GetDimensions(count, stride);
    if (instanceIndex >= count) { return false; }
    StreamRayInstance instance = instances[instanceIndex];
    if (clusterStride == 0u) { return false; }
    StreamCLASTessellationLeaf leaf;
    float3 microPositions[3], microDomains[3];
    bool tessellated = streamRayTessellationTriangle(clusterId, triangle, leaf, microPositions, microDomains);
    if (tessellated) {
        float3 weights = float3(1.0 - bary.x - bary.y, bary);
        bary = (microDomains[0] * weights.x + microDomains[1] * weights.y + microDomains[2] * weights.z).yz;
        clusterId = leaf.pageIndex * clusterStride + leaf.clusterIndex;
        triangle = leaf.sourceTriangle;
    }
    uint2 address = streamRayClusterAddress(clusterId, clusterStride);
    float3 a, b, c;
    StreamTriangleDecodeContext triangleContext;
    let pages = gRayStreamPages;
    if (committedHit) {
        if (!decodeStreamCommittedTriangleAtPage(address.x, address.y, triangle, pages, gRayStreamPageTable,
            a, b, c, triangleContext)) { return false; }
    } else {
        if (!decodeStreamTriangleAtPage(address.x, address.y, triangle, pages, gRayStreamPageTable,
            streamPageWordCount(gRayStreamPageCount), a, b, c, triangleContext)) { return false; }
    }
    float3 n[3]; float2 uv[3]; float4 tangents[3];
    uint attributes = decodeStreamAttributes(pages, triangleContext, n, uv, tangents, !shadowOnly);
    float3 weights = float3(1.0 - bary.x - bary.y, bary);
    hit.texcoord = uv[0] * weights.x + uv[1] * weights.y + uv[2] * weights.z;
    if (tessellated) {
        a = microPositions[0]; b = microPositions[1]; c = microPositions[2];
        float2 sourceUv[3] = {uv[0], uv[1], uv[2]};
        for (uint i = 0u; i < 3u; ++i) {
            uv[i] = sourceUv[0] * microDomains[i].x + sourceUv[1] * microDomains[i].y + sourceUv[2] * microDomains[i].z;
        }
    }
    float3 localGeometry = streamRayTriangleNormal(b - a, c - a, float3(0, 1, 0));
    hit.geometryNormal = streamRayNormal(instance, localGeometry);
    hit.position = ray.origin + ray.direction * t;
    hit.t = t;
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;
    hit.materialIndex = instance.materialIndex;
#if METALLIC_PT_PATH_DIAGNOSTICS
    hit.diagnosticStreamInstance = instanceIndex;
    hit.diagnosticCluster = clusterId;
    hit.diagnosticBarycentrics = bary;
#endif
    hit.instanceIndex = instance.gpuSceneInstanceIndex;
    hit.primitiveIndex = instance.primitiveIndex;
    hit.triangleIndex = triangle;
    // Shadow transport consumes geometric orientation and UVs, but no shading frame or ray cone.
    if (shadowOnly) { return true; }

    float3 localNormal = Metallic.ScenePath.safeNormalize(n[0] * weights.x + n[1] * weights.y + n[2] * weights.z, localGeometry);
    hit.normal = tessellated ? hit.geometryNormal : streamRayNormal(instance, localNormal);
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
    return true;
}


} // namespace Metallic.StreamGeometry
