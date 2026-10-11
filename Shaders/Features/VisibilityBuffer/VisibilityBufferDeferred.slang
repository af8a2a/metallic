import Core;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#ifndef MATERIAL_CLASS
#define MATERIAL_CLASS -1
#endif

#include "VisibilityBufferLighting.slang"

ScenePathTracePush deferredViewPush()
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
#if METALLIC_GLOBAL_VIEW
    ViewConstants view = resolveBuffer<StructuredBuffer<ViewConstants>>(getResourceParameters<SceneResourceParameters>().view)[0];
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

#define gVisibility resolveUniform(getResourceParameters<SceneResourceParameters>().visibility)
#define gVisibilityDepth resolveUniform(getResourceParameters<SceneResourceParameters>().visibilityDepth)
#define gRasterInstances resolveBuffer<StructuredBuffer<GPUDrivenPreviewInstance>>(getResourceParameters<SceneResourceParameters>().rasterInstances)
#define gRasterMaterials resolveBuffer<StructuredBuffer<GPUDrivenPreviewMaterial>>(getResourceParameters<SceneResourceParameters>().rasterMaterials)
#include "VisibilityStreamDecode.slang"
#define gStreamRecords resolveBuffer<StructuredBuffer<CompactStreamVisibleRecord>>(getResourceParameters<SceneResourceParameters>().streamRecords)
#define gStreamGroups resolveBuffer<StructuredBuffer<StreamActiveGroup>>(getResourceParameters<SceneResourceParameters>().streamGroups)
#define gStreamPages resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().streamPages)
#define gStreamPageTable resolveBuffer<StructuredBuffer<uint2>>(getResourceParameters<SceneResourceParameters>().streamPageTable)
#define gRasterFrameInfo resolveBuffer<ByteAddressBuffer>(getResourceParameters<SceneResourceParameters>().rasterFrameInfo)
#define gRasterDomain resolveUniform(getResourceParameters<SceneResourceParameters>().rasterDomain)

float4 rasterDomain(uint2 pixel)
{
    return (gRasterFrameInfo.Load(96) & 1u) != 0u ? gRasterDomain.Load(int3(pixel, 0)) : float4(-1, -1, 0, 0);
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
    float viewDepth = push.upProjection.w > 0.5 ? lerp(push.clipOrtho.y, push.clipOrtho.x, depth)
        : push.clipOrtho.x * push.clipOrtho.y / max(push.clipOrtho.x + depth * (push.clipOrtho.y - push.clipOrtho.x), 1e-6);
    float3 forward = cameraForwardFrom(push.eye, push.center);
    return (viewDepth - dot(ray.origin - push.eye.xyz, forward)) / max(dot(ray.direction, forward), 1e-6);
}
#define gMaterialBins typedBufferSpan<MaterialBin>(getResourceParameters<SceneResourceParameters>().materialBins)
#define gMaterialTiles typedBufferSpan<MaterialTile>(getResourceParameters<SceneResourceParameters>().materialTiles)
#if METALLIC_DEFERRED_UPSCALER_GUIDES
#define gUpscalerMotion resolveUniform(getResourceParameters<SceneResourceParameters>().upscalerMotion)
#define gUpscalerDepth resolveUniform(getResourceParameters<SceneResourceParameters>().upscalerDepth)
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
    VisibleClusterRecord record;
    float3 a, b, c;
    if (!decodeStreamTriangle(recordIndex, triangleIndex, gStreamRecords, gStreamGroups,
        gStreamPages, gStreamPageTable, streamPageWordCount(resolveBuffer<ByteAddressBuffer>(getResourceParameters<SceneResourceParameters>().streamPageCount)), record, a, b, c)) { return false; }
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
    return reconstructStreamSurface(pixel, recordIndex, triangleIndex, ray, push, hit);
}

// Both schedules resolve the same raster surface. The RNG is used only by
// optional stochastic texture filtering, never for light/path sampling.
void shadeVisibilityPixel(uint2 pixel, ScenePathTracePush push)
{
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 samplePixel = float2(pixel) + 0.5;
    float2 rayPixel = samplePixel + float2(push.jitterOffsetX, push.jitterOffsetY);
    Ray ray;
    ray.origin = rayOriginFromPixel(rayPixel, push);
    ray.direction = rayDirectionFromPixel(rayPixel, push);
    uint seed = hash(pixel.x + pixel.y * push.width);
    HitInfo hit;
    float3 radiance;
    bool hasSurface = reconstructVisibilitySurface(pixel, ray, push, hit);
#if METALLIC_DEFERRED_UPSCALER_GUIDES
    float deviceDepth = gVisibilityDepth.Load(int3(pixel, 0));
    // Upscaler guides retain the standard-depth contract.
    gUpscalerDepth[pixel] = 1.0 - deviceDepth;
    gUpscalerMotion[pixel] = hasSurface ? motionVectorToPrevious(hit.position, samplePixel, push)
        : skyMotionVectorToPrevious(ray.direction, samplePixel, push);
#endif
#if MATERIAL_CLASS == 0
    radiance = environmentRadiance(ray.direction, push, true, ray.origin);
#else
    if (!hasSurface) {
        radiance = environmentRadiance(ray.direction, push, true, ray.origin);
    }
    else {
        radiance = shadeVisibilitySurface(ray, hit, rayPixel, push, seed,
            MATERIAL_CLASS == 1 ? 0 : (MATERIAL_CLASS == 2 ? 1 : -1),
            MATERIAL_CLASS >= 1 && MATERIAL_CLASS <= 3);
    }
#endif
    if (push.debugView != kOpenPBRDebugViewFinal && !hasSurface) { radiance = float3(0.0); }
    if (gRealtimeShadowParameters[0].control.w != 0u) {
        float encoded = gRealtimeShadow.Load(int3(pixel, 0));
        radiance = float3(encoded * encoded);
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
    uint binIndex = push.deferredSettings & 0xffffu;
    if (!typedBufferSpan<MaterialBin>(getResourceParameters<SceneResourceParameters>().materialBins).contains(binIndex)) { return; }
    MaterialBin bin = gMaterialBins[binIndex];
    uint index = group.y * kMaterialBinMaxGroupsX + group.x;
    uint tileCount = typedBufferSpan<MaterialTile>(getResourceParameters<SceneResourceParameters>().materialTiles).length();
    if (index >= bin.count || bin.offset >= tileCount || index >= tileCount - bin.offset) { return; }
    MaterialTile tile = gMaterialTiles[bin.offset + index];
    uint lane = WaveGetLaneIndex();
    if ((tile.laneMask & (1u << lane)) == 0u) { return; }
    shadeVisibilityPixel(materialTilePixel(tile, lane, push.width), push);
}
