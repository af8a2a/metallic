#include "../../Libraries/Resources/ComputeResources.slang"

#ifndef MATERIAL_CLASS
#define MATERIAL_CLASS -1
#endif

#include "../Lighting/OpenPBRDirectLighting.slang"
#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../Libraries/GPUDriven/MaterialBinningCommon.slang"
#include "../../Libraries/Camera/ViewConstants.slang"

ScenePathTracePush deferredViewPush()
{
    ScenePathTracePush push = METALLIC_CONSTANTS(ScenePathTracePush);
#if METALLIC_GLOBAL_VIEW
    ViewConstants view = METALLIC_RESOURCE(StructuredBuffer<ViewConstants>, 80)[0];
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

#define gVisibility METALLIC_RESOURCE(Texture2D<uint>, 60)
#define gVisibilityDepth METALLIC_RESOURCE(Texture2D<float>, 61)
#define gRasterVertices METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewVertex>, 62)
#define gRasterMeshlets METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewMeshlet>, 63)
#define gRasterRecords METALLIC_RESOURCE(StructuredBuffer<VisibleClusterRecord>, 64)
#define gRasterMeshletVertices METALLIC_RESOURCE(StructuredBuffer<uint>, 65)
#define gRasterTriangles METALLIC_RESOURCE(StructuredBuffer<uint>, 66)
#define gRasterGeometries METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewGeometry>, 67)
#define gRasterInstances METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewInstance>, 68)
#define gRasterMaterials METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewMaterial>, 69)
#define gMaterialBins METALLIC_RESOURCE(StructuredBuffer<MaterialBin>, 70)
#define gMaterialTiles METALLIC_RESOURCE(StructuredBuffer<MaterialTile>, 71)
#if METALLIC_DEFERRED_UPSCALER_GUIDES
#define gUpscalerMotion METALLIC_RESOURCE(RWTexture2D<float2>, 72)
#define gUpscalerDepth METALLIC_RESOURCE(RWTexture2D<float>, 73)
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

bool reconstructVisibilitySurface(uint2 pixel, Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return false; }
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
    float depth = gVisibilityDepth.Load(int3(pixel, 0));
    float zNear = push.clipOrtho.x, zFar = push.clipOrtho.y;
    float standardDepth = push.clipOrtho.w > 0.5 ? 1.0 - depth : depth;
    float viewDepth = push.upProjection.w > 0.5 ? lerp(zNear, zFar, standardDepth)
        : zNear * zFar / max(zFar - standardDepth * (zFar - zNear), 0.000001);
    hit.t = viewDepth / max(dot(ray.direction, cameraForwardFrom(push.eye, push.center)), 0.000001);
    float3 position = ray.origin + ray.direction * hit.t;
    float3 e0 = p1 - p0, e1 = p2 - p0, offset = position - p0;
    float3 crossEdges = cross(e0, e1);
    float determinant = dot(crossEdges, crossEdges);
    if (determinant < 1e-20) { return false; }
    float b1 = dot(cross(offset, e1), crossEdges) / determinant;
    float b2 = dot(cross(e0, offset), crossEdges) / determinant;
    float3 bary = float3(1.0 - b1 - b2, b1, b2);
    hit.position = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    float3 localNormal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
    float3 localGeometryNormal = safeTriangleNormal(v1.position.xyz - v0.position.xyz,
        v2.position.xyz - v0.position.xyz, safeNormalize(localNormal, float3(0, 1, 0)));
    localNormal = safeNormalize(localNormal, localGeometryNormal);
    hit.geometryNormal = rasterWorldNormal(instance, localGeometryNormal);
    hit.normal = rasterWorldNormal(instance, localNormal);
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;
    hit.texcoord = v0.texcoord * bary.x + v1.texcoord * bary.y + v2.texcoord * bary.z;
    float4 localTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;
    float3 triangleTangent, triangleBitangent;
    computeTriangleTangentSpace(v0.position.xyz, v1.position.xyz, v2.position.xyz,
        v0.texcoord, v1.texcoord, v2.texcoord, localNormal, triangleTangent, triangleBitangent);
    float3 axis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 fallback = safeNormalize(cross(axis, hit.normal), float3(1, 0, 0));
    bool authoredTangent = (v0.flags & 2u) != 0u;
    float3 tangent = rasterWorldNormal(instance, authoredTangent ? localTangent.xyz : triangleTangent);
    tangent = safeNormalize(tangent - hit.normal * dot(hit.normal, tangent), fallback);
    float sign = localTangent.w < 0.0 ? -1.0 : 1.0;
    float3 bitangent;
    if (authoredTangent) {
        bitangent = safeNormalize(cross(hit.normal, tangent) * sign, cross(hit.normal, fallback));
    } else {
        bitangent = rasterWorldNormal(instance, triangleBitangent);
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
    hit.rayConeLodConstant = uvArea > 1e-7 ? 0.5 * log2(uvArea / sqrt(determinant)) : 0.0;
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
#if !METALLIC_REALTIME_DEFERRED && (MATERIAL_CLASS < 0 || MATERIAL_CLASS == 4)
    else if (gMaterials[hit.materialIndex].glassParams.x > 0.0 &&
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
