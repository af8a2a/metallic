module LumenProbeImportance;

import ShaderCore;
import ScenePathMath;
import LumenProbeMath;

using Metallic;

namespace Metallic.Lumen
{
static const uint kImportanceTraceResolution = 8u;
static const uint kImportanceTraceCount = kImportanceTraceResolution * kImportanceTraceResolution;
static const uint kImportanceUniformLevel = 1u;
static const float kImportanceMinPDFToTrace = 0.1f;

groupshared ThreeBandSHVector brdfSphericalHarmonics[kImportanceTraceCount * 2u];
groupshared uint numSphericalHarmonics;
groupshared uint2 raysToRefine[kImportanceTraceCount * 2u];
groupshared uint numRaysToSubdivide;

// UE 5.7.4 LumenScreenProbeTracingCommon.ush, PF_R16_UINT ray info.
uint packProbeRayInfo(uint2 texelCoord, uint level)
{
    return (texelCoord.x & 0x3fu) | ((texelCoord.y & 0x3fu) << 6u) | ((level & 0xfu) << 12u);
}

public void lumenUnpackProbeRayInfo(uint rayInfo, out uint2 texelCoord, out uint level)
{
    texelCoord.x = rayInfo & 0x3fu;
    texelCoord.y = (rayInfo >> 6u) & 0x3fu;
    level = (rayInfo >> 12u) & 0xfu;
}

public uint lumenProbeRayMipSize(uint level)
{
    return (kImportanceTraceResolution << kImportanceUniformLevel) >> level;
}

ThreeBandSHVector addProbeSH(ThreeBandSHVector a, ThreeBandSHVector b)
{
    a.v0 += b.v0;
    a.v1 += b.v1;
    a.v2 += b.v2;
    return a;
}

// UE GetWorldPositionFromScreenUV. Metallic's position guide already stores linear view depth.
float3 probeWorldPositionAtUV(ViewCameraConstants camera, float2 jitter, float2 screenUV, float sceneDepth)
{
    float3 forward = ScenePath.cameraForwardFrom(camera.eye, camera.center);
    float3 right = ScenePath.cameraRightFrom(forward, camera.upProjection.xyz);
    float3 up = ScenePath.cameraUpFrom(forward, right);
    bool orthographic = camera.upProjection.w > 0.5f;
    float halfHeight = orthographic ? camera.clipOrtho.z * 0.5f : tan(camera.viewport.w * 0.5f);
    float2 halfExtent = float2(camera.viewport.x * halfHeight, halfHeight);
    float2 screen = (screenUV - 0.5f) * float2(2.0f, -2.0f);
    float2 unjitteredScreen = screen + jitter / camera.viewport.yz * float2(2.0f, -2.0f);
    float2 offset = unjitteredScreen * halfExtent * (orthographic ? 1.0f : sceneDepth);
    return camera.eye.xyz + forward * sceneDepth + right * offset.x + up * offset.y;
}

// UE 5.7.4 ScreenProbeComputeBRDFProbabilityDensityFunctionCS, opaque hemispherical receivers.
// A full 8x8 workgroup calls this for one valid probe.
public void lumenEvaluateProbeBrdfPdf(Texture2D<float4> position, Texture2D<float4> normal,
    ViewCameraConstants camera, float2 viewJitter, uint2 extent, uint2 probePixel, uint2 lane,
    uint downsampleFactor, out ThreeBandSHVector brdfPdf)
{
    uint threadIndex = lane.y * kImportanceTraceResolution + lane.x;
    if (threadIndex == 0u) {
        numSphericalHarmonics = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 invExtent = 1.0f / float2(extent);
    float2 screenUV = (float2(probePixel) + 0.5f) * invExtent;
    float2 threadOffset = ((float2(lane) + 0.5f) / float(kImportanceTraceResolution) * 2.0f - 1.0f) * float(downsampleFactor);
    bool centerSample = all(lane == kImportanceTraceResolution / 2u);
    if (centerSample) {
        threadOffset = 0.0f;
    }

    float2 pixelScreenUV = screenUV + threadOffset * invExtent;
    pixelScreenUV = clamp(pixelScreenUV, float2(0.0f), float2(extent - 1u) * invExtent);
    // Keep the known-valid center pixel when the viewport-edge UV rounds down on conversion.
    uint2 pixel = centerSample ? probePixel : uint2(pixelScreenUV * float2(extent));
    float pixelSceneDepth = position.Load(int3(pixel, 0)).w;
    // Metallic clears the receiver guide where there is no opaque surface.
    if (pixelSceneDepth > 0.0f) {
        float3 worldNormal = normalize(normal.Load(int3(pixel, 0)).xyz);
        float3 pixelPosition = probeWorldPositionAtUV(camera, viewJitter, pixelScreenUV, pixelSceneDepth);
        float4 pixelPlane = float4(worldNormal, dot(worldNormal, pixelPosition));
        float probeSceneDepth = position.Load(int3(probePixel, 0)).w;
        float3 probeWorldPosition = probeWorldPositionAtUV(camera, viewJitter, screenUV, probeSceneDepth);

        float planeDistance = abs(dot(float4(probeWorldPosition, -1.0f), pixelPlane));
        float relativeDepthDifference = planeDistance / probeSceneDepth;
        float depthWeight = exp2(-10000.0f * (relativeDepthDifference * relativeDepthDifference));
        if (depthWeight > 0.1f || centerSample) {
            uint index;
            InterlockedAdd(numSphericalHarmonics, 1u, index);
            brdfSphericalHarmonics[index] = calcDiffuseTransferSH3(worldNormal, 1.0f);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint numSHToAccumulate = numSphericalHarmonics;
    uint offset = 0u;
    while (numSHToAccumulate > 1u) {
        uint threadBaseIndex = threadIndex * 4u;
        if (threadBaseIndex < numSHToAccumulate) {
            ThreeBandSHVector pdf = brdfSphericalHarmonics[threadBaseIndex + offset];
            if (threadBaseIndex + 1u < numSHToAccumulate) {
                pdf = addProbeSH(pdf, brdfSphericalHarmonics[threadBaseIndex + 1u + offset]);
            }
            if (threadBaseIndex + 2u < numSHToAccumulate) {
                pdf = addProbeSH(pdf, brdfSphericalHarmonics[threadBaseIndex + 2u + offset]);
            }
            if (threadBaseIndex + 3u < numSHToAccumulate) {
                pdf = addProbeSH(pdf, brdfSphericalHarmonics[threadBaseIndex + 3u + offset]);
            }
            brdfSphericalHarmonics[threadIndex + offset + numSHToAccumulate] = pdf;
        }
        offset += numSHToAccumulate;
        numSHToAccumulate = (numSHToAccumulate + 3u) / 4u;
        GroupMemoryBarrierWithGroupSync();
    }

    float normalizeWeight = 1.0f / float(numSphericalHarmonics);
    ThreeBandSHVector pdf = brdfSphericalHarmonics[offset];
    // Preserve UE's PF_R16F BRDF buffer precision with Metallic's structured buffer.
    brdfPdf.v0 = f16tof32(f32tof16(pdf.v0 * normalizeWeight));
    brdfPdf.v1 = f16tof32(f32tof16(pdf.v1 * normalizeWeight));
    brdfPdf.v2 = f16tof32(f32tof16(pdf.v2 * normalizeWeight));
}

uint2 packRaySortInfo(uint2 texelCoord, uint level, float pdf)
{
    return uint2((texelCoord.x & 0xffu) | ((texelCoord.y & 0xffu) << 8u) | ((level & 0xffu) << 16u), asuint(pdf));
}

void unpackRaySortInfo(uint2 raySortInfo, out uint2 texelCoord, out uint level, out float pdf)
{
    texelCoord.x = raySortInfo.x & 0xffu;
    texelCoord.y = (raySortInfo.x >> 8u) & 0xffu;
    level = (raySortInfo.x >> 16u) & 0xffu;
    pdf = asfloat(raySortInfo.y);
}

// UE 5.7.4 ScreenProbeGenerateRaysCS, default parallel path with one subdivision level.
// Lighting PDF has already passed through the reference PF_R16F storage precision.
public uint lumenGenerateProbeRay(uint2 lane, ThreeBandSHVector brdfPdf, float lightingPdf)
{
    uint threadIndex = lane.y * kImportanceTraceResolution + lane.x;
    float2 probeUV = (float2(lane) + 0.5f) / float(kImportanceTraceResolution);
    float3 worldConeDirection = lumenEquiAreaSphericalMapping(probeUV);
    ThreeBandSHVector directionSH = shBasisFunction3(worldConeDirection);
    float pdf = dot(brdfPdf.v0, directionSH.v0);
    pdf += dot(brdfPdf.v1, directionSH.v1);
    pdf += brdfPdf.v2 * directionSH.v2;
    pdf = max(pdf, 0.0f);

    bool notCulledByBrdf = pdf >= kImportanceMinPDFToTrace;
    float lightingPdfScale = lightingPdf * float(kImportanceTraceCount);
    pdf *= lightingPdfScale;
    if (notCulledByBrdf) {
        pdf = max(pdf, kImportanceMinPDFToTrace);
    }
    raysToRefine[threadIndex] = packRaySortInfo(lane, kImportanceUniformLevel, pdf);
    if (threadIndex == 0u) {
        numRaysToSubdivide = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint sortedOffset = kImportanceTraceCount;
    uint2 rayTexelCoord;
    uint rayLevel;
    float sortKey;
    unpackRaySortInfo(raysToRefine[threadIndex], rayTexelCoord, rayLevel, sortKey);

    uint numSmaller = 0u;
    for (uint otherRayIndex = 0u; otherRayIndex < kImportanceTraceCount; ++otherRayIndex) {
        uint2 otherRayTexelCoord;
        uint otherRayLevel;
        float otherSortKey;
        unpackRaySortInfo(raysToRefine[otherRayIndex], otherRayTexelCoord, otherRayLevel, otherSortKey);
        if (otherSortKey < sortKey || (otherSortKey == sortKey && otherRayIndex > threadIndex)) {
            ++numSmaller;
        }
    }
    raysToRefine[sortedOffset + numSmaller] = raysToRefine[threadIndex];
    GroupMemoryBarrierWithGroupSync();

    uint mergeThreadIndex = threadIndex % 3u;
    uint mergeIndex = threadIndex / 3u;
    uint rayIndexToRefine = uint(max(int(kImportanceTraceCount) - int(mergeIndex) - 1, 0));
    uint rayIndexToMerge2 = mergeIndex * 3u + 2u;
    bool reassignRay = false;
    uint2 subdividedRay = 0u;
    if (rayIndexToMerge2 < rayIndexToRefine) {
        uint2 rayTexelCoord2;
        uint rayLevel2;
        float pdf2;
        unpackRaySortInfo(raysToRefine[sortedOffset + rayIndexToMerge2], rayTexelCoord2, rayLevel2, pdf2);
        if (pdf2 < kImportanceMinPDFToTrace) {
            uint2 originalRayTexelCoord;
            uint originalRayLevel;
            float originalPdf;
            unpackRaySortInfo(raysToRefine[sortedOffset + rayIndexToRefine], originalRayTexelCoord, originalRayLevel, originalPdf);
            subdividedRay = packRaySortInfo(originalRayTexelCoord * 2u + uint2((mergeThreadIndex + 1u) % 2u,
                (mergeThreadIndex + 1u) / 2u), originalRayLevel - 1u, 0.0f);
            reassignRay = true;
        }
    }
    // A low-PDF packet can straddle waves. Finish reading its original entries before overwriting them.
    GroupMemoryBarrierWithGroupSync();
    if (reassignRay) {
        raysToRefine[sortedOffset + threadIndex] = subdividedRay;
        if (mergeThreadIndex == 0u) {
            InterlockedAdd(numRaysToSubdivide, 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex < numRaysToSubdivide) {
        uint rayIndexToSubdivide = kImportanceTraceCount - threadIndex - 1u;
        uint2 originalRayTexelCoord;
        uint originalRayLevel;
        float originalPdf;
        unpackRaySortInfo(raysToRefine[sortedOffset + rayIndexToSubdivide], originalRayTexelCoord, originalRayLevel, originalPdf);
        raysToRefine[sortedOffset + rayIndexToSubdivide] = packRaySortInfo(originalRayTexelCoord * 2u, originalRayLevel - 1u, 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    float rayPdf;
    unpackRaySortInfo(raysToRefine[sortedOffset + threadIndex], rayTexelCoord, rayLevel, rayPdf);
    return packProbeRayInfo(rayTexelCoord, rayLevel);
}
} // namespace Metallic.Lumen
