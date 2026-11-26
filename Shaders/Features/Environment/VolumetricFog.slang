module VolumetricFog;

import ShaderCore;
import Core;
import ParameterRoot;
import ScenePathMath;
import Lighting;
import LumenScreenTrace;
import VolumetricFogPunctual;
using Metallic;
using Metallic.ScenePath;
using Metallic.Lighting;

#include "../../../Source/Runtime/Render/Environment/VolumetricFogParameters.h"
__include "LocalFogVolumes.slang";

// Reference: UE nvrtx-5.7 8284a0e654e8, VolumetricFog.usf,
// HeightFogCommon.ush, ParticipatingMediaCommon.ush and GenerateConservativeDepth.usf.
// The renderer camera and TLAS use meters/Y-up. Grid Z and medium coefficients
// retain UE centimeters. Reconstruct camera-relative positions before integration.
float computeDepthFromZSlice(float zSlice, float3 gridZ)
{
    return (exp2(zSlice / gridZ.z) - gridZ.y) / gridZ.x;
}

float computeNormalizedZSlice(float depthCm, VolumetricFogParams p)
{
    return log2(depthCm * p.gridZ.x + p.gridZ.y) * p.gridZ.z / float(p.grid.z);
}

float3 computeCellTranslatedWorldPosition(uint3 coordinate, float3 offset, VolumetricFogParams p)
{
    float2 uv = (coordinate.xy + offset.xy) / float2(p.grid.xy);
    float2 ndc = (uv * 2.0f - 1.0f) * float2(1.0f, -1.0f);
    float sceneDepth = computeDepthFromZSlice(max(coordinate.z + offset.z, 0.0f), p.gridZ.xyz);
    float3 forward = cameraForwardFrom(p.current.eye, p.current.center);
    float3 right = cameraRightFrom(forward, p.current.upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    float halfHeight = p.current.upProjection.w > 0.5f ? p.current.clipOrtho.z * 50.0f :
        sceneDepth * tan(p.current.viewport.w * 0.5f);
    return forward * sceneDepth + (right * (ndc.x * p.current.viewport.x) + up * ndc.y) * halfHeight;
}

float halton(uint index, uint base)
{
    float result = 0.0f;
    float invBase = 1.0f / float(base);
    float fraction = invBase;
    while (index > 0u) {
        result += float(index % base) * fraction;
        index /= base;
        fraction *= invBase;
    }
    return result;
}

float3 frameJitterOffset(uint sampleIndex, VolumetricFogParams p)
{
    if (p.temporal.y == 0u || p.distances.w == 0.0f) { return 0.5f; }
    uint frame = (p.temporal.x - sampleIndex) & 1023u;
    return float3(halton(frame, 2u), halton(frame, 3u), halton(frame, 5u));
}

float2 volumeUVMax(VolumetricFogParams p)
{
    float2 physicalSize = float2(p.grid.xy * p.grid.w);
    return (physicalSize - (p.grid.w / 2u + 1u)) / physicalSize;
}

float reversedDeviceZ(float depthCm, ViewCameraConstants camera)
{
    float viewZ = depthCm * 0.01f;
    float nearZ = camera.clipOrtho.x;
    float farZ = camera.clipOrtho.y;
    return camera.upProjection.w > 0.5f ? (farZ - viewZ) / (farZ - nearZ) :
        nearZ * (farZ - viewZ) / (viewZ * (farZ - nearZ));
}

float3 previousTranslatedPosition(float3 translatedCm, VolumetricFogParams p)
{
    return (p.current.eye.xyz - p.previous.eye.xyz) + translatedCm * 0.01f;
}

float3 computeHistoryVolumeUV(float3 translatedCm, VolumetricFogParams p)
{
    float3 translated = previousTranslatedPosition(translatedCm, p);
    float3 forward = cameraForwardFrom(p.previous.eye, p.previous.center);
    float3 right = cameraRightFrom(forward, p.previous.upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    float viewZ = dot(translated, forward);
    if (viewZ <= 0.0f) { return -1.0f; }
    float halfHeight = p.previous.upProjection.w > 0.5f ? p.previous.clipOrtho.z * 0.5f :
        viewZ * tan(p.previous.viewport.w * 0.5f);
    float2 ndc = float2(dot(translated, right) / (halfHeight * p.previous.viewport.x),
        dot(translated, up) / halfHeight);
    float2 uv = ndc * float2(0.5f, -0.5f) + 0.5f;
    // UE uses the previous/current view-to-padded-grid ratio. The pass
    // invalidates history on an extent change and allocates its exact view grid.
    return min(float3(uv, computeNormalizedZSlice(viewZ * 100.0f, p)), 1.0f);
}

float2 fixupHistoryUV(float2 uv, float prevCellDeviceZ, VolumetricFogParams p, out bool valid)
{
    float2 size = float2(p.grid.xy);
    float2 fullResUV = uv * size;
    float2 screenCoord = floor(fullResUV - 0.5f);
    float2 gatherUV = (screenCoord + 1.0f) / size;
    float2 fullResOffset = fullResUV - screenCoord;
    float4 prevDepths = resolveUniform(p.previousDepth).GatherRed(resolveUniform(p.sampler), gatherUV);
    bool4 validSamples = prevDepths < prevCellDeviceZ;
    valid = true;
    if (all(validSamples)) { return uv; }
    if (all(validSamples.wz)) { return (screenCoord + float2(fullResOffset.x, 0.5f)) / size; }
    if (all(validSamples.xy)) { return (screenCoord + float2(fullResOffset.x, 1.5f)) / size; }
    if (all(validSamples.wx)) { return (screenCoord + float2(0.5f, fullResOffset.y)) / size; }
    if (all(validSamples.zy)) { return (screenCoord + float2(1.5f, fullResOffset.y)) / size; }
    if (validSamples.x) { return (screenCoord + float2(0.5f, 1.5f)) / size; }
    if (validSamples.y) { return (screenCoord + float2(1.5f, 1.5f)) / size; }
    if (validSamples.w) { return (screenCoord + float2(0.5f, 0.5f)) / size; }
    if (validSamples.z) { return (screenCoord + float2(1.5f, 0.5f)) / size; }
    valid = false;
    return uv;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void materialSetup(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    if (any(id >= p.grid.xyz)) { return; }
    float height = p.current.eye.y * 100.0f + computeCellTranslatedWorldPosition(id, 0.5f, p).y;
    float first = p.fogLayer0.x * exp2(-p.fogLayer0.y * (height - p.fogLayer0.z));
    float second = p.fogLayer1.x * exp2(-p.fogLayer1.y * (height - p.fogLayer1.z));
    float globalDensity = (first + second) * 0.5f;
    float extinction = max(globalDensity * p.albedoExtinction.w, 0.0f);
    float3 scattering = p.albedoExtinction.rgb * extinction;
    float3 additionalEmissive = 0.0f;
    if (p.localFog.instances.count != 0u) {
        injectLocalFog(id, p, scattering, extinction, additionalEmissive);
    }
    resolveUniform(p.materialOut)[id] = float4(scattering, extinction);
    if (p.localFog.emissiveOut.index != 0xffffffffu) {
        resolveUniform(p.localFog.emissiveOut)[id] = float4(p.emissive.rgb + additionalEmissive, 0.0f);
    }
}

[shader("compute")]
[numthreads(4, 4, 4)]
void directionalShadow(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    if (any(id >= p.grid.xyz)) { return; }
    if (p.lightColorIntensity.w == 0.0f) {
        resolveUniform(p.shadowOut)[id] = 1.0f;
        return;
    }
    float3 translatedCm = computeCellTranslatedWorldPosition(id, frameJitterOffset(0u, p), p);
    RayDesc ray;
    ray.Origin = p.current.eye.xyz + translatedCm * 0.01f;
    ray.Direction = p.lightDirectionPhase.xyz;
    ray.TMin = 0.0f;
    ray.TMax = 1.0e25f; // UE's 1e27 cm in the Metallic TLAS.
    // This permutation is specifically for opaque casters. The stream TLAS
    // tags opaque instances; alpha-tested geometry needs a material any-hit path.
    const uint flags = RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    RayQuery<flags> query;
    query.TraceRayInline(resolveAccelerationStructure(p.scene), flags, 0xffu, ray);
    while (query.Proceed()) {}
    resolveUniform(p.shadowOut)[id] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void conservativeDepth(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    if (any(id.xy >= p.grid.xy)) { return; }
    uint2 size;
    let depth = resolveUniform(p.depth);
    depth.GetDimensions(size.x, size.y);
    // Same half-tile margin, inclusive pixel centers and 4x4 HZB footprint
    // as UE. Reduce the covered leaves directly from the matching raster depth.
    float2 tileMin = (float2(id.xy) - 0.5f) * float(p.grid.w);
    float2 tileMax = (float2(id.xy) + 1.5f) * float(p.grid.w);
    int2 pixelMin = int2(clamp(tileMin, 0.0f, float2(size)) + 0.5f);
    int2 pixelMax = min(int2(clamp(tileMax, 0.0f, float2(size)) - 0.5f), int2(size) - 1);
    int2 hzbMin = pixelMin >> 1;
    int2 hzbMax = max(pixelMin, pixelMax) >> 1;
    int2 mipXY = firstbithigh(hzbMax - hzbMin);
    int mip = max(max(mipXY.x, mipXY.y) - 1, 0);
    mip += any((hzbMax >> mip) - (hzbMin >> mip) > 3) ? 1 : 0;
    int2 low = (hzbMin >> mip) << (mip + 1);
    int2 high = min(((hzbMax >> mip) + 1) << (mip + 1), int2(size));
    float minDepth = 1.0f;
    for (int y = low.y; y < high.y; ++y) {
        for (int x = low.x; x < high.x; ++x) {
            minDepth = min(minDepth, depth.Load(int3(x, y, 0)));
        }
    }
    resolveUniform(p.currentDepthOut)[id.xy] = minDepth;
    if (p.historyDepthOut.index != 0xffffffffu) { resolveUniform(p.historyDepthOut)[id.xy] = minDepth; }
}

void writeScattering(uint3 id, float4 value, VolumetricFogParams p)
{
    resolveUniform(p.scatteringOut)[id] = value;
    if (p.historyScatteringOut.index != 0xffffffffu) { resolveUniform(p.historyScatteringOut)[id] = value; }
}

// LumenTranslucencyVolumeShared.ush::GetTranslucencyGIVolumeLighting(false)
// and the RotatedHGZonalHarmonic dot in VolumetricFog.usf.
float3 fogIndirectScattering(float3 translatedWorldPositionCm, float3 cameraVector, VolumetricFogParams p)
{
    // Project in camera-relative meters; UE's perspective clip W uses centimeters.
    ViewCameraConstants camera = p.current;
    camera.center.xyz -= camera.eye.xyz;
    camera.eye.xyz = 0.0f;
    Lumen.ScreenTraceProjection view = Lumen.makeScreenTraceProjection(camera,
        float2(p.indirectLight.rasterJitterX, p.indirectLight.rasterJitterY));
    float4 clip = Lumen.screenTraceWorldToClip(view, translatedWorldPositionCm * 0.01f);
    float2 uv = Lumen.screenTraceScreenUV(clip.xy / clip.w);
    float clipW = camera.upProjection.w > 0.5f ? clip.w : clip.w * 100.0f;
    float4 gridZ = p.indirectLight.gridZ;
    float slice = log2(clipW * gridZ.x + gridZ.y) * gridZ.z;
    // The physical texture can have padding after the last active slice.
    float z = min(slice, gridZ.w - 0.5f) / p.indirectLight.textureDepth;
    float3 volumeUV = float3(uv, z);
    float3 ambient = resolveUniform(p.indirectLight.ambient).SampleLevel(resolveUniform(p.sampler), volumeUV, 0).rgb;
    float3 directional = resolveUniform(p.indirectLight.directional).SampleLevel(resolveUniform(p.sampler), volumeUV, 0).rgb;
    float3 normalizedAmbientColor = ambient / (WorkingColor::toXYZ(ambient).y + 0.00001f);
    float4 red = float4(ambient.r, directional * normalizedAmbientColor.r);
    float4 green = float4(ambient.g, directional * normalizedAmbientColor.g);
    float4 blue = float4(ambient.b, directional * normalizedAmbientColor.b);
    // The volume's SH coefficients use the UE axes. Metallic -Z/X/Y map to UE X/Y/Z.
    float3 directionUE = float3(-cameraVector.z, cameraVector.x, cameraVector.y);
    float g = p.lightDirectionPhase.w;
    float4 rotatedHG = float4(1.0f, directionUE.y, directionUE.z, directionUE.x) * float4(1.0f, g, g, g);
    return max(float3(dot(red, rotatedHG), dot(green, rotatedHG), dot(blue, rotatedHG)), 0.0f);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void lightScattering(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    if (any(id >= p.grid.xyz)) { return; }
    float historyAlpha = p.distances.z;
    float3 historyUV = 0.0f;
    if (historyAlpha > 0.0f) {
        historyUV = computeHistoryVolumeUV(computeCellTranslatedWorldPosition(id, 0.5f, p), p);
    }
    if (p.temporal.z != 0u) {
        float3 cell = computeCellTranslatedWorldPosition(id, float3(0.5f, 0.5f, -0.5f), p);
        float frontDepth = computeDepthFromZSlice(max(float(id.z) - 0.5f, 0.0f), p.gridZ.xyz);
        if (resolveUniform(p.currentDepth).Load(int3(id.xy, 0)) > reversedDeviceZ(frontDepth, p.current)) {
            writeScattering(id, 0.0f, p);
            return;
        }
        if (historyAlpha > 0.0f) {
            float3 prevDirection = cameraForwardFrom(p.previous.eye, p.previous.center);
            float prevZ = dot(previousTranslatedPosition(cell, p), prevDirection) * 100.0f;
            bool valid;
            historyUV.xy = fixupHistoryUV(historyUV.xy, reversedDeviceZ(prevZ, p.previous), p, valid);
            if (!valid) { historyAlpha = 0.0f; }
        }
    }
    if (any(historyUV < 0.0f) || any(historyUV >= 1.0f)) { historyAlpha = 0.0f; }
    uint numSamples = p.temporal.y != 0u && historyAlpha < 0.001f ? p.temporal.w : 1u;
    float shadowFactor = p.shadow.index == 0xffffffffu ? 1.0f : resolveUniform(p.shadow).Load(int4(id, 0));
    float3 light = 0.0f;
    for (uint sampleIndex = 0u; sampleIndex < numSamples; ++sampleIndex) {
        float3 offset = frameJitterOffset(sampleIndex, p);
        float3 cell = computeCellTranslatedWorldPosition(id, offset, p);
        float3 cameraVector = p.current.upProjection.w > 0.5f ? cameraForwardFrom(p.current.eye, p.current.center) : normalize(cell);
        float cosTheta = dot(p.lightDirectionPhase.xyz, -cameraVector);
        float phase = fogPhaseFunction(p.lightDirectionPhase.w, cosTheta);
        float3 irradiance = p.lightColorIntensity.rgb;
        if (p.atmosphereParameters.index != 0xffffffffu) {
            irradiance = physicalCelestialIrradiance(resolveUniform(p.atmosphereParameters)[0], resolveUniform(p.transmittance),
                p.cloudShadow, 0u, p.current.eye.xyz + cell * 0.01f);
        }
        light += (irradiance * p.lightColorIntensity.w) * shadowFactor * phase;
        if (p.indirectLight.ambient.index != 0xffffffffu) {
            light += fogIndirectScattering(cell, cameraVector, p);
        }
        if (p.lightGrid.index != 0xffffffffu) {
            float sceneDepthCm = computeDepthFromZSlice(max(float(id.z) + offset.z, 0.0f), p.gridZ.xyz);
            float cellRadius = length(cell - computeCellTranslatedWorldPosition(id + uint3(1, 1, 1), offset, p));
            light += fogPunctualScattering(resolveUniform(p.lightGrid)[0], resolveUniform(p.lights),
                resolveUniform(p.lightCells), resolveUniform(p.lightIndices), resolveUniform(p.lightCandidates),
                p.worldShadowFaces, id.xy * p.grid.w, sceneDepthCm * 0.01f, p.current.eye.xyz, cell, cameraVector,
                cellRadius, p.lightDirectionPhase.w);
        }
    }
    light /= float(numSamples);
    float4 medium = resolveUniform(p.material).Load(int4(id, 0));
    // UE's default PROJECT_EXPFOG_MATCHES_VFOG=0 writes a constant to FP16 VBufferB.
    float3 emissive = p.localFog.emissive.index == 0xffffffffu ? f16tof32(f32tof16(p.emissive.rgb)) :
        resolveUniform(p.localFog.emissive).Load(int4(id, 0)).rgb;
    float4 result = float4(light * medium.rgb + emissive, medium.a);
    if (historyAlpha > 0.0f) {
        float4 previous = resolveUniform(p.history).SampleLevel(resolveUniform(p.sampler),
            min(historyUV, float3(volumeUVMax(p), 1.0f)), 0.0f);
        result = lerp(result, previous, historyAlpha);
    }
    writeScattering(id, select(asuint(result) >= 0x7f800000u, 0.0f, result), p);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void finalIntegration(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    if (any(id.xy >= p.grid.xy)) { return; }
    float3 accumulatedLighting = 0.0f;
    float accumulatedTransmittance = 1.0f;
    float3 previousSlice = computeCellTranslatedWorldPosition(uint3(id.xy, 0), float3(0.5f, 0.5f, 0.0f), p);
    float accumulatedDepth = 0.0f;
    for (uint layerIndex = 0u; layerIndex < p.grid.z; ++layerIndex) {
        uint3 layer = uint3(id.xy, layerIndex);
        float4 source = resolveUniform(p.scattering).Load(int4(layer, 0));
        float3 position = computeCellTranslatedWorldPosition(layer, 0.5f, p);
        float stepLength = length(position - previousSlice);
        previousSlice = position;
        float transmittance = exp(-source.w * stepLength);
        accumulatedDepth += stepLength;
        float fade = saturate(accumulatedDepth * p.distances.y);
        float3 integrated = fade * (source.rgb - source.rgb * transmittance) / max(source.w, 0.00001f);
        accumulatedLighting += integrated * accumulatedTransmittance;
        accumulatedTransmittance *= lerp(1.0f, transmittance, fade);
        resolveUniform(p.integratedOut)[layer] = float4(accumulatedLighting, accumulatedTransmittance);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compositeFog(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    uint2 size;
    resolveUniform(p.colorOut).GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) { return; }
    float deviceZ = resolveUniform(p.depth).Load(int3(id.xy, 0));
    float nearZ = p.current.clipOrtho.x;
    float farZ = p.current.clipOrtho.y;
    float sceneDepth = 100.0f * (p.current.upProjection.w > 0.5f ? lerp(farZ, nearZ, deviceZ) :
        nearZ * farZ / (nearZ + deviceZ * (farZ - nearZ)));
    float zSlice = computeNormalizedZSlice(sceneDepth, p);
    float2 uv = (float2(id.xy) + 0.5f) / float2(p.grid.xy * p.grid.w);
    float4 fog = resolveUniform(p.integrated).SampleLevel(resolveUniform(p.sampler),
        float3(min(uv, volumeUVMax(p)), zSlice), 0.0f);
    fog = lerp(float4(0.0f, 0.0f, 0.0f, 1.0f), fog, saturate((sceneDepth - p.distances.x) * 100000000.0f));
    float4 source = resolveUniform(p.sceneColor).Load(int3(id.xy, 0));
    resolveUniform(p.colorOut)[id.xy] = float4(fog.rgb + fog.a * source.rgb, source.a);
}
