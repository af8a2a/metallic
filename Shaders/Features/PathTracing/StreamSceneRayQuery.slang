#ifndef METALLIC_STREAM_SCENE_RAY_QUERY
#define METALLIC_STREAM_SCENE_RAY_QUERY
#include "StreamSceneDecode.slang"

// Statically specialized at the caller. The query layer owns no descriptor
// slots, parameter roots, or material/texture globals.
interface IStreamRayMaterials
{
    bool load(uint index, out PathTraceMaterial material);
    float alpha(PathTraceTextureInfo texture, float2 uv);
};

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

bool streamRaySurface<Materials : IStreamRayMaterials>(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    Ray ray, float t, StreamSceneParameters scene, Materials materials, float bitangentFlip, out HitInfo hit)
{
    hit = (HitInfo)0;
    StreamRayInstance instance;
    if (!loadStreamRayInstance(scene, instanceIndex, instance)) { return false; }
    PathTraceMaterial material;
    if (!materials.load(instance.materialIndex, material)) { return false; }
    float3 a, b, c;
    float3 n[3]; float2 uv[3]; float4 tangents[3];
    uint attributes;
    if (!decodeStreamRaySurface(scene, clusterId, triangle, a, b, c, n, uv, tangents, attributes)) { return false; }
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

bool streamRayAlphaAccepts<Materials : IStreamRayMaterials>(uint instanceIndex, uint clusterId, uint triangle, float2 bary,
    StreamSceneParameters scene, Materials materials)
{
    StreamRayInstance instance;
    if (!loadStreamRayInstance(scene, instanceIndex, instance)) { return false; }
    PathTraceMaterial material;
    if (!materials.load(instance.materialIndex, material)) { return false; }
    // Validate geometry even for opaque candidates before committing the hit.
    uint pageIndex, cluster;
    float3 a, b, c;
    if (!decodeStreamRayTriangle(scene, clusterId, triangle, pageIndex, cluster, a, b, c)) { return false; }
    if (material.textureParams.w < 0.5) { return true; }
    // Alpha coverage needs UV only, not world transforms, normals or a TBN.
    float3 normals[3]; float2 uvs[3]; float4 tangents[3];
    decodeStreamAttributesAtPage(pageIndex, cluster, triangle,
        StreamDataBuffer<uint>(scene.pages), StreamDataBuffer<uint2>(scene.pageTable), normals, uvs, tangents);
    float2 uv = uvs[0] * (1.0 - bary.x - bary.y) + uvs[1] * bary.x + uvs[2] * bary.y;
    float alpha = material.baseColor.a * materials.alpha(material.baseColorTexture, uv);
    return material.textureParams.w < 1.5 ? alpha >= material.params.z : alpha > 0.0;
}

bool traceStreamClosest<Materials : IStreamRayMaterials>(Ray ray, float nearDistance, float farDistance,
    RaytracingAccelerationStructure accelerationStructure, StreamSceneParameters scene,
    Materials materials, float bitangentFlip, out HitInfo hit)
{
    hit = (HitInfo)0;
    RayDesc desc;
    desc.Origin = ray.origin; desc.Direction = ray.direction;
    desc.TMin = nearDistance; desc.TMax = farDistance;
    // Force candidate callbacks even for CLAS built with the opaque default.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(accelerationStructure, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, desc);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            streamRayAlphaAccepts(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), scene, materials)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
    return streamRaySurface(query.CommittedInstanceID(), uint(query.CommittedClusterID()),
        query.CommittedPrimitiveIndex(), query.CommittedTriangleBarycentrics(), ray, query.CommittedRayT(), scene, materials, bitangentFlip, hit);
}

#endif
