import Core;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#ifndef MATERIAL_CLASS
#define MATERIAL_CLASS -1
#endif

#include "../Lighting/OpenPBRDirectLighting.slang"

ScenePathTracePush deferredViewPush()
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
#if METALLIC_GLOBAL_VIEW
    ViewConstants view = getResource<StructuredBuffer<ViewConstants>>(80)[0];
    push.eye = view.current.eye;
    push.center = view.current.center;
    push.upProjection = view.current.upProjection;
    push.viewport = view.current.viewport;
    push.clipOrtho = view.current.clipOrtho;
    push.previousEye = view.previous.eye;
    push.previousCenter = view.previous.center;
    push.previousUpProjection = view.previous.upProjection;
    push.previousViewport = view.previous.viewport;
    push.previousClipOrtho = view.previous.clipOrtho;
    push.jitterOffsetX = view.jitter.x;
    push.jitterOffsetY = view.jitter.y;
    push.previousCameraValid = view.frame.y;
    push.temporalJitter = view.frame.z;
#endif
    return push;
}

#define gVisibility getResource<Texture2D<uint>>(60)
#define gVisibilityDepth getResource<Texture2D<float>>(61)
#define gRasterVertices getResource<StructuredBuffer<GPUDrivenPreviewVertex>>(62)
#define gRasterMeshlets getResource<StructuredBuffer<GPUDrivenPreviewMeshlet>>(63)
#define gRasterRecords getResource<StructuredBuffer<VisibleClusterRecord>>(64)
#define gRasterMeshletVertices getResource<StructuredBuffer<uint>>(65)
#define gRasterTriangles getResource<StructuredBuffer<uint>>(66)
#define gRasterGeometries getResource<StructuredBuffer<GPUDrivenPreviewGeometry>>(67)
#define gRasterInstances getResource<StructuredBuffer<GPUDrivenPreviewInstance>>(68)
#define gRasterMaterials getResource<StructuredBuffer<GPUDrivenPreviewMaterial>>(69)
#include "VisibilityStreamDecode.slang"
#define gStreamRecords getResource<StructuredBuffer<VisibleClusterRecord>>(83)
#define gStreamGroups getResource<StructuredBuffer<StreamActiveGroup>>(84)
#define gStreamPages getResource<StructuredBuffer<uint>>(85)
#define gStreamPageTable getResource<StructuredBuffer<uint2>>(86)
#define gRasterFrameInfo getResource<ByteAddressBuffer>(87)
#define gRasterDomain getResource<Texture2D<float4>>(88)

float4 rasterDomain(uint2 pixel)
{
    return (gRasterFrameInfo.Load(104) & 1u) != 0u ? gRasterDomain.Load(int3(pixel, 0)) : float4(-1, -1, 0, 0);
}
float3 domainNormal(float packed)
{
    uint bits = uint(packed);
    float2 xy = float2(bits & 4095u, (bits >> 12u) & 4095u) / 4095.0 * 2.0 - 1.0;
    float3 n = float3(xy, 1.0 - abs(xy.x) - abs(xy.y));
    if (n.z < 0.0) { n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); }
    return normalize(n);
}
float rasterRayDistance(uint2 pixel, Ray ray, ScenePathTracePush push)
{
    float depth = gVisibilityDepth.Load(int3(pixel, 0));
    float standard = push.clipOrtho.w > 0.5 ? 1.0 - depth : depth;
    float viewDepth = push.upProjection.w > 0.5 ? lerp(push.clipOrtho.x, push.clipOrtho.y, standard)
        : push.clipOrtho.x * push.clipOrtho.y / max(push.clipOrtho.y - standard * (push.clipOrtho.y - push.clipOrtho.x), 1e-6);
    float3 forward = cameraForwardFrom(push.eye, push.center);
    return (viewDepth - dot(ray.origin - push.eye.xyz, forward)) / max(dot(ray.direction, forward), 1e-6);
}
#define gMaterialBins getResource<StructuredBuffer<MaterialBin>>(70)
#define gMaterialTiles getResource<StructuredBuffer<MaterialTile>>(71)
#if METALLIC_DEFERRED_UPSCALER_GUIDES
#define gUpscalerMotion getResource<RWTexture2D<float2>>(72)
#define gUpscalerDepth getResource<RWTexture2D<float>>(73)
#endif

float3 rasterWorldPosition(GPUDrivenPreviewInstance instance, float3 position)
{
    return instance.world0.xyz * position.x + instance.world1.xyz * position.y +
        instance.world2.xyz * position.z + instance.world3.xyz;
}

float3 rasterWorldNormal(GPUDrivenPreviewInstance instance, float3 normal)
{
    float3 c0 = cross(instance.world1.xyz, instance.world2.xyz);
    float3 c1 = cross(instance.world2.xyz, instance.world0.xyz);
    float3 c2 = cross(instance.world0.xyz, instance.world1.xyz);
    float determinant = dot(instance.world0.xyz, c0);
    // Preserve authored orientation under mirrored transforms, as the ray path does.
    return safeNormalize((c0 * normal.x + c1 * normal.y + c2 * normal.z) *
        (determinant < 0.0 ? -1.0 : 1.0), normal);
}

float3 rasterWorldVector(GPUDrivenPreviewInstance instance, float3 v)
{
    return instance.world0.xyz * v.x + instance.world1.xyz * v.y + instance.world2.xyz * v.z;
}

float rasterHandedness(GPUDrivenPreviewInstance instance)
{
    return dot(instance.world0.xyz, cross(instance.world1.xyz, instance.world2.xyz)) < 0.0 ? -1.0 : 1.0;
}

bool reconstructStreamSurface(uint2 pixel, uint recordIndex, uint triangleIndex, Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    if (gRasterFrameInfo.Load(92) == 0u) { return false; }
    VisibleClusterRecord record;
    float3 a, b, c;
    if (!decodeStreamTriangle(recordIndex, triangleIndex, gStreamRecords, gStreamGroups,
        gStreamPages, gStreamPageTable, record, a, b, c)) { return false; }
    uint count, stride;
    gRasterInstances.GetDimensions(count, stride);
    if (record.instanceIndex >= count) { return false; }
    GPUDrivenPreviewInstance instance = gRasterInstances[record.instanceIndex];
    float3 localNormal = safeTriangleNormal(b - a, c - a, float3(0, 1, 0));
    float3 p = rasterWorldPosition(instance, a);
    hit.geometryNormal = rasterWorldNormal(instance, localNormal);
    hit.normal = hit.geometryNormal;
    float4 domain = rasterDomain(pixel);
    float denominator = dot(ray.direction, hit.geometryNormal);
    if (domain.x < 0.0 && abs(denominator) < 1e-8) { return false; }
    hit.t = domain.x >= 0.0 ? rasterRayDistance(pixel, ray, push) : dot(p - ray.origin, hit.geometryNormal) / denominator;
    hit.position = ray.origin + ray.direction * hit.t;
    float3 world[3] = {p, rasterWorldPosition(instance, b), rasterWorldPosition(instance, c)};
    float3 e0 = world[1] - p, e1 = world[2] - p, crossEdges = cross(e0, e1);
    float area = dot(crossEdges, crossEdges);
    if (area < 1e-20) { return false; }
    float2 yz = float2(dot(cross(hit.position - p, e1), crossEdges), dot(cross(e0, hit.position - p), crossEdges)) / area;
    if (domain.x >= 0.0) {
        yz = domain.xy;
        hit.geometryNormal = domainNormal(domain.w);
    }
    float3 bary = float3(1.0 - yz.x - yz.y, yz);
    float3 normals[3]; float2 uv[3]; float4 tangents[3];
    uint attributes = decodeStreamAttributes(record, triangleIndex, gStreamGroups, gStreamPages, gStreamPageTable, normals, uv, tangents);
    hit.normal = domain.x >= 0.0 ? hit.geometryNormal : rasterWorldNormal(instance,
        safeNormalize(normals[0] * bary.x + normals[1] * bary.y + normals[2] * bary.z, localNormal));
    hit.texcoord = uv[0] * bary.x + uv[1] * bary.y + uv[2] * bary.z;
    float2 du = uv[1] - uv[0], dv = uv[2] - uv[0];
    float uvArea = abs(du.x * dv.y - du.y * dv.x);
    hit.rayConeLodConstant = domain.x >= 0.0 ? domain.z : (uvArea > 1e-7 ? 0.5 * log2(uvArea / sqrt(area)) : 0.0);
    float3 uvTangent, uvBitangent;
    computeTriangleTangentSpace(world[0], world[1], world[2], uv[0], uv[1], uv[2], hit.normal, uvTangent, uvBitangent);
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;
    // Keep the authored geometric normal stable until normal map evaluation.
    float3 axis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = safeNormalize(uvTangent - hit.normal * dot(hit.normal, uvTangent), safeNormalize(cross(axis, hit.normal), float3(1, 0, 0)));
    float tangentSign = dot(cross(hit.normal, tangent), uvBitangent) < 0.0 ? -1.0 : 1.0;
    if ((attributes & 16u) != 0u && domain.x < 0.0) {
        float4 authored = tangents[0] * bary.x + tangents[1] * bary.y + tangents[2] * bary.z;
        float3 worldTangent = rasterWorldVector(instance, authored.xyz);
        tangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), tangent);
        tangentSign = (authored.w < 0.0 ? -1.0 : 1.0) * rasterHandedness(instance);
    }
    hit.tangent = float4(tangent, tangentSign * push.bitangentFlip);
    hit.bitangent = cross(hit.normal, tangent) * tangentSign * push.bitangentFlip;
    gRasterMaterials.GetDimensions(count, stride);
    if (instance.identity.y >= count) { return false; }
    hit.materialIndex = gRasterMaterials[instance.identity.y].identity.z;
    gMaterials.GetDimensions(count, stride);
    if (hit.materialIndex >= count) { return false; }
    hit.instanceIndex = instance.identity.z;
    hit.primitiveIndex = gStreamGroups[record.dataIndex].primitiveIndex;
    hit.triangleIndex = triangleIndex;
    return true;
}

bool reconstructVisibilitySurface(uint2 pixel, Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return false; }
    uint residentCount = gRasterFrameInfo.Load(88);
    if (recordIndex >= residentCount) {
        return reconstructStreamSurface(pixel, recordIndex - residentCount, triangleIndex, ray, push, hit);
    }
    uint count, stride;
    gRasterRecords.GetDimensions(count, stride);
    if (recordIndex >= count) { return false; }
    VisibleClusterRecord record = gRasterRecords[recordIndex];
    if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return false; }
    gRasterMeshlets.GetDimensions(count, stride);
    if (record.clusterIndex >= count) { return false; }
    GPUDrivenPreviewMeshlet meshlet = gRasterMeshlets[record.clusterIndex];
    if (triangleIndex >= meshlet.ranges.w) { return false; }
    gRasterGeometries.GetDimensions(count, stride);
    if (record.dataIndex >= count) { return false; }
    GPUDrivenPreviewGeometry geometry = gRasterGeometries[record.dataIndex];
    gRasterInstances.GetDimensions(count, stride);
    if (record.instanceIndex >= count) { return false; }
    GPUDrivenPreviewInstance instance = gRasterInstances[record.instanceIndex];
    uint3 localIndices = uint3(
        loadPackedMeshletTriangleIndex(gRasterTriangles, meshlet.ranges.z, triangleIndex, 0u),
        loadPackedMeshletTriangleIndex(gRasterTriangles, meshlet.ranges.z, triangleIndex, 1u),
        loadPackedMeshletTriangleIndex(gRasterTriangles, meshlet.ranges.z, triangleIndex, 2u));
    uint3 indices = geometry.payload.x + uint3(
        gRasterMeshletVertices[meshlet.ranges.x + localIndices.x],
        gRasterMeshletVertices[meshlet.ranges.x + localIndices.y],
        gRasterMeshletVertices[meshlet.ranges.x + localIndices.z]);
    GPUDrivenPreviewVertex v0 = gRasterVertices[indices.x];
    GPUDrivenPreviewVertex v1 = gRasterVertices[indices.y];
    GPUDrivenPreviewVertex v2 = gRasterVertices[indices.z];
    float3 p0 = rasterWorldPosition(instance, v0.position.xyz);
    float3 p1 = rasterWorldPosition(instance, v1.position.xyz);
    float3 p2 = rasterWorldPosition(instance, v2.position.xyz);
    hit.t = rasterRayDistance(pixel, ray, push);
    float3 position = ray.origin + ray.direction * hit.t;
    float3 e0 = p1 - p0, e1 = p2 - p0, offset = position - p0;
    float3 crossEdges = cross(e0, e1);
    float determinant = dot(crossEdges, crossEdges);
    if (determinant < 1e-20) { return false; }
    float b1 = dot(cross(offset, e1), crossEdges) / determinant;
    float b2 = dot(cross(e0, offset), crossEdges) / determinant;
    float4 domain = rasterDomain(pixel);
    if (domain.x >= 0.0) { b1 = domain.x; b2 = domain.y; }
    float3 bary = float3(1.0 - b1 - b2, b1, b2);
    hit.position = domain.x >= 0.0 ? position : p0 * bary.x + p1 * bary.y + p2 * bary.z;
    float3 localNormal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
    float3 localGeometryNormal = safeTriangleNormal(v1.position.xyz - v0.position.xyz,
        v2.position.xyz - v0.position.xyz, safeNormalize(localNormal, float3(0, 1, 0)));
    localNormal = safeNormalize(localNormal, localGeometryNormal);
    hit.geometryNormal = rasterWorldNormal(instance, localGeometryNormal);
    hit.normal = rasterWorldNormal(instance, localNormal);
    if (domain.x >= 0.0) { hit.geometryNormal = domainNormal(domain.w); hit.normal = hit.geometryNormal; }
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;
    hit.texcoord = v0.texcoord * bary.x + v1.texcoord * bary.y + v2.texcoord * bary.z;
    float4 localTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;
    float3 triangleTangent, triangleBitangent;
    computeTriangleTangentSpace(v0.position.xyz, v1.position.xyz, v2.position.xyz,
        v0.texcoord, v1.texcoord, v2.texcoord, localNormal, triangleTangent, triangleBitangent);
    float3 axis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 fallback = safeNormalize(cross(axis, hit.normal), float3(1, 0, 0));
    bool authoredTangent = (v0.flags & 2u) != 0u;
    float3 tangent = rasterWorldVector(instance, authoredTangent ? localTangent.xyz : triangleTangent);
    tangent = safeNormalize(tangent - hit.normal * dot(hit.normal, tangent), fallback);
    float sign = (localTangent.w < 0.0 ? -1.0 : 1.0) * rasterHandedness(instance);
    float3 bitangent;
    if (authoredTangent) {
        bitangent = safeNormalize(cross(hit.normal, tangent) * sign, cross(hit.normal, fallback));
    } else {
        bitangent = rasterWorldVector(instance, triangleBitangent);
        bitangent = safeNormalize(bitangent - hit.normal * dot(hit.normal, bitangent), cross(hit.normal, tangent));
        sign = dot(cross(hit.normal, tangent), bitangent) < 0.0 ? -1.0 : 1.0;
    }
    hit.tangent = float4(tangent, sign * push.bitangentFlip);
    hit.bitangent = bitangent * push.bitangentFlip;
    gRasterMaterials.GetDimensions(count, stride);
    if (instance.identity.y >= count) { return false; }
    uint sourceMaterial = gRasterMaterials[instance.identity.y].identity.z;
    gMaterials.GetDimensions(count, stride);
    hit.materialIndex = sourceMaterial < count ? sourceMaterial : 0u;
    hit.instanceIndex = instance.identity.z;
    hit.primitiveIndex = geometry.source.x;
    hit.triangleIndex = triangleIndex;
    float2 uv0 = v1.texcoord - v0.texcoord, uv1 = v2.texcoord - v0.texcoord;
    float uvArea = abs(uv0.x * uv1.y - uv0.y * uv1.x);
    hit.rayConeLodConstant = domain.x >= 0.0 ? domain.z : (uvArea > 1e-7 ? 0.5 * log2(uvArea / sqrt(determinant)) : 0.0);
    return true;
}

// Both scheduling paths share the exact per-pixel estimator and RNG seed.
// Primary visibility stays rasterized, including for volume transmission.
void shadeVisibilityPixel(uint2 pixel, ScenePathTracePush push)
{
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 samplePixel = float2(pixel) + 0.5;
    float2 rayPixel = samplePixel + float2(push.jitterOffsetX, push.jitterOffsetY);
    Ray ray;
    ray.origin = rayOriginFromPixel(rayPixel, push);
    ray.direction = rayDirectionFromPixel(rayPixel, push);
    uint seed = hash(pixel.x + pixel.y * push.width +
        (push.enableAccumulation != 0u ? push.accumulationFrame * 0x9e3779b9u : 0u));
    HitInfo hit;
    float3 radiance;
    bool hasSurface = reconstructVisibilitySurface(pixel, ray, push, hit);
#if METALLIC_DEFERRED_UPSCALER_GUIDES
    float deviceDepth = gVisibilityDepth.Load(int3(pixel, 0));
    gUpscalerDepth[pixel] = push.clipOrtho.w > 0.5 ? 1.0 - deviceDepth : deviceDepth;
    gUpscalerMotion[pixel] = hasSurface ? motionVectorToPrevious(hit.position, samplePixel, push)
        : skyMotionVectorToPrevious(ray.direction, samplePixel, push);
#endif
#if MATERIAL_CLASS == 0
    radiance = environmentRadiance(ray.direction, push, true);
#else
    if (!hasSurface) {
        radiance = environmentRadiance(ray.direction, push, true);
    }
#if (!METALLIC_REALTIME_DEFERRED || METALLIC_STREAM_RAY_QUERIES) && (MATERIAL_CLASS < 0 || MATERIAL_CLASS == 4)
    else if ((gMaterials[hit.materialIndex].glassParams.x > 0.0 || gMaterials[hit.materialIndex].textureParams.w > 1.5) &&
        !openPbrDebugFlag(push, kOpenPBRDebugDisableTransmission) &&
        push.debugView == kOpenPBRDebugViewFinal) {
        // Ray continuation is isolated to transmissive materials. Reusing the
        // OpenPBR reference integrator preserves refraction, exit interfaces,
        // total internal reflection, colored shadows and Beer-Lambert absorption.
        ScenePathTracePush transmissionPush = push;
        transmissionPush.maxDepth = clamp((push.deferredSettings >> 21u) & 31u, 2u, 16u);
        uint samples = clamp((push.deferredSettings >> 16u) & 31u, 1u, 16u);
        radiance = float3(0.0);
        for (uint sample = 0u; sample < samples; ++sample) {
            float ignoredDistance;
            radiance += pathTraceOpenPBRWithPrimaryHit(ray,
                primaryRayConeFromPixel(rayPixel, push), transmissionPush,
                seed, ignoredDistance, true, hit);
        }
        radiance /= float(samples);
    }
#endif
    else {
        radiance = shadeOpenPBRDirectSurface(ray, hit, rayPixel, push, seed, true,
            MATERIAL_CLASS == 1 ? 0 : (MATERIAL_CLASS == 2 ? 1 : -1),
            MATERIAL_CLASS >= 1 && MATERIAL_CLASS <= 3);
    }
#endif
#if METALLIC_REALTIME_DEFERRED
    if (gRealtimeShadowParameters[0].control.w != 0u) {
        float encoded = gRealtimeShadow.Load(int3(pixel, 0));
        radiance = float3(encoded * encoded);
    }
#endif
    if (push.enableAccumulation != 0u) {
        if (push.hasHistory != 0u) {
            radiance = lerp(gHistoryPrevious[pixel].rgb, radiance, 1.0 / float(push.accumulationFrame + 1u));
        }
        gHistoryCurrent[pixel] = float4(radiance, 1.0);
    }
    gOutput[pixel] = float4(max(radiance, float3(0.0)), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void visibilityBufferDeferredMain(uint3 dispatchId : SV_DispatchThreadID)
{
    ScenePathTracePush push = deferredViewPush();
    shadeVisibilityPixel(dispatchId.xy, push);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void visibilityBufferDeferredBinnedMain(uint3 group : SV_GroupID)
{
    ScenePathTracePush push = deferredViewPush();
    MaterialBin bin = gMaterialBins[push.deferredSettings & 0xffffu];
    uint index = group.y * kMaterialBinMaxGroupsX + group.x;
    if (index >= bin.count) { return; }
    MaterialTile tile = gMaterialTiles[bin.offset + index];
    uint lane = WaveGetLaneIndex();
    if ((tile.laneMask & (1u << lane)) == 0u) { return; }
    shadeVisibilityPixel(materialTilePixel(tile, lane, push.width), push);
}
