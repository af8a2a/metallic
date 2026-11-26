module LumenTranslucencyVolume;

import ShaderCore;
import Core;
import ParameterRoot;
import ScenePathMath;
import GlobalDistanceField;
import LumenGlobalRay;
import LumenProbeMath;
import LumenSky;
using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;
using Metallic.ScenePath;

#include "../../../Source/Runtime/Render/RenderPass/BuiltinPass/LumenTranslucencyVolumeParameters.h"

// UE 5.7.4 LumenTranslucencyVolumeLighting{,Shared}.ush/usf.
// B and the resulting view depth are in centimeters.
float computeDepthFromZSlice(float z)
{
    return (exp2(z / 4.0f) - 1.0f) / 0.01f;
}

float viewDepthCm(float deviceZ, ViewCameraConstants camera)
{
    float n = camera.clipOrtho.x, f = camera.clipOrtho.y;
    return 100.0f * (camera.upProjection.w > 0.5f ? f - deviceZ * (f - n) :
        n * f / (n + deviceZ * (f - n)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reduceVisibleDepth(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<LumenTranslucencyVolumeParameters>();
    if (any(id.xy >= p.grid.xy)) { return; }
    let depth = resolveUniform(p.depth);
    uint2 size;
    depth.GetDimensions(size.x, size.y);
    // Reduce the leaves of the same point-sampled furthest-HZB texel.
    uint pixelShift = uint(firstbithigh(p.grid.w));
    uint footprint = 1u << (max(int(pixelShift) - 1, 0) + 2);
    uint2 center = min(uint2((float2(id.xy) + 0.5f) * (1u << pixelShift)), size - 1u);
    uint2 low = (center / footprint) * footprint;
    uint2 high = min(low + footprint, size);
    float furthest = 1.0f;
    for (uint y = low.y; y < high.y; ++y) {
        for (uint x = low.x; x < high.x; ++x) {
            furthest = min(furthest, depth.Load(int3(x, y, 0)));
        }
    }
    resolveUniform(p.maxVisibleDepthOut)[id.xy] = viewDepthCm(furthest, p.camera);
}

bool isFroxelVisible(uint3 cell, LumenTranslucencyVolumeParameters p)
{
    return computeDepthFromZSlice(max(float(cell.z) - 1.0f, 0.0f)) <
        resolveUniform(p.maxVisibleDepth).Load(int3(cell.xy, 0));
}

float3 froxelWorldPosition(uint3 cell, LumenTranslucencyVolumeParameters p)
{
    float3 offset = p.frameJitter;
    float2 uv = (float2(cell.xy) + offset.xy) / float2(p.grid.xy);
    let depth = resolveUniform(p.depth);
    uint2 size;
    depth.GetDimensions(size.x, size.y);
    // The GI grid is unjittered. VisibilityBuffer's projection subtracts
    // its pixel jitter (published in the camera W lanes) from raster XY.
    float2 rasterPixel = uv * float2(size) - float2(p.camera.eye.w, p.camera.center.w);
    float rawDepth = depth.Load(int3(clamp(int2(floor(rasterPixel)), int2(0), int2(size) - 1), 0));
    float depthGrid = log2(viewDepthCm(rawDepth, p.camera) * 0.01f + 1.0f) * 4.0f - 0.5f;
    float difference = depthGrid - (float(cell.z) + offset.z);
    if (difference < 0.0f && -difference < 1.0f) { offset.z += difference; }
    float depthM = computeDepthFromZSlice(float(cell.z) + offset.z) * 0.01f;
    float2 ndc = (uv * 2.0f - 1.0f) * float2(1.0f, -1.0f);
    float3 forward = cameraForwardFrom(p.camera.eye, p.camera.center);
    float3 right = cameraRightFrom(forward, p.camera.upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    float halfHeight = p.camera.upProjection.w > 0.5f ? p.camera.clipOrtho.z * 0.5f :
        depthM * tan(p.camera.viewport.w * 0.5f);
    return p.camera.eye.xyz + forward * depthM + (right * (ndc.x * p.camera.viewport.x) + up * ndc.y) * halfHeight;
}

// Store each 3x3 octahedral tile in adjacent Z slices of the same XY froxel.
static const uint kOctResolution = 3u;
static const uint kRayCount = kOctResolution * kOctResolution;

[shader("compute")]
[numthreads(4, 4, 4)]
void traceVolume(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<LumenTranslucencyVolumeParameters>();
    uint3 cell = uint3(id.xy, id.z / kRayCount);
    if (any(id.xy >= p.grid.xy) || cell.z >= p.tracing.z) { return; }
    float3 radiance = 0.0f;
    if (cell.z < p.grid.z && isFroxelVisible(cell, p)) {
        uint oct = id.z % kRayCount;
        uint2 texel = uint2(oct % kOctResolution, oct / kOctResolution);
        float2 center = lumenProbeBlueNoiseVec2(resolveUniform(p.blueNoise), cell.xy, p.tracing.y);
        float3 dirUE = lumenEquiAreaSphericalMapping((float2(texel) + center) / float(kOctResolution));
        float3 direction = float3(dirUE.y, dirUE.z, -dirUE.x);
        LumenGlobalTraceRay ray = {float4(froxelWorldPosition(cell, p), 0.0f), float4(direction, 200.0f)};
        let hit = traceLumenGlobalRay(resolveLumenGlobalTraceScene(p.scene, p.tracing.x), ray);
        radiance = hit.state.x == kGlobalDistanceFieldHit ? hit.radiance.rgb : sampleLumenSky(p.sky, direction);
        float maxLighting = max(radiance.r, max(radiance.g, radiance.b));
        if (maxLighting > p.maxRayIntensity) { radiance *= p.maxRayIntensity / maxLighting; }
    }
    resolveUniform(p.traceOut)[id] = float4(radiance, 0.0f);
}

void filterAxis(uint3 id, int3 direction, bool readFirst)
{
    let p = getParameters<LumenTranslucencyVolumeParameters>();
    int3 cell = int3(id.xy, id.z / kRayCount);
    if (any(id.xy >= p.grid.xy) || uint(cell.z) >= p.tracing.z) { return; }
    float3 filteredRadiance = 0.0f;
    float totalWeight = 0.0f;
    for (int offset = -3; offset <= 3; ++offset) {
        int3 neighbor = cell + direction * offset;
        if (all(neighbor >= 0) && all(neighbor < int3(p.grid.xyz))) {
            int4 coord = int4(neighbor.xy, neighbor.z * int(kRayCount) + int(id.z % kRayCount), 0);
            float3 radiance = readFirst ? resolveUniform(p.trace).Load(coord).rgb : resolveUniform(p.filtered).Load(coord).rgb;
            const float invSigmaSqrt2Pi = 1.0f / (5.0f * sqrt(2.0f * kLumenProbePi));
            float weight = invSigmaSqrt2Pi * exp(-float(offset) * float(offset) * (1.0f / 50.0f));
            filteredRadiance += radiance * weight;
            totalWeight += weight;
        }
    }
    float4 value = float4(filteredRadiance / max(totalWeight, 0.00001f), 0.0f);
    if (readFirst) { resolveUniform(p.filteredOut)[id] = value; }
    else { resolveUniform(p.traceOut)[id] = value; }
}

[shader("compute")] [numthreads(4, 4, 4)]
void filterX(uint3 id : SV_DispatchThreadID) { filterAxis(id, int3(1, 0, 0), true); }
[shader("compute")] [numthreads(4, 4, 4)]
void filterY(uint3 id : SV_DispatchThreadID) { filterAxis(id, int3(0, 1, 0), false); }
[shader("compute")] [numthreads(4, 4, 4)]
void filterZ(uint3 id : SV_DispatchThreadID) { filterAxis(id, int3(0, 0, 1), true); }

[shader("compute")]
[numthreads(4, 4, 4)]
void integrateVolume(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<LumenTranslucencyVolumeParameters>();
    if (any(id.xy >= p.grid.xy) || id.z >= p.tracing.z) { return; }
    float4 r = 0.0f, g = 0.0f, b = 0.0f;
    if (id.z < p.grid.z && isFroxelVisible(id, p)) {
        for (uint y = 0; y < kOctResolution; ++y) {
            for (uint x = 0; x < kOctResolution; ++x) {
                int4 coord = int4(id.xy, id.z * kRayCount + y * kOctResolution + x, 0);
                float3 radiance = p.tracing.w != 0u ? resolveUniform(p.filtered).Load(coord).rgb :
                    resolveUniform(p.trace).Load(coord).rgb;
                float2 uv = (float2(x, y) + 0.5f) * (1.0f / float(kOctResolution));
                float4 basis = shBasisFunction3(lumenEquiAreaSphericalMapping(uv)).v0;
                r += basis * radiance.r;
                g += basis * radiance.g;
                b += basis * radiance.b;
            }
        }
        float normalizeFactor = kLumenProbePi / float(kOctResolution) / float(kOctResolution);
        r *= normalizeFactor;
        g *= normalizeFactor;
        b *= normalizeFactor;
    }
    float3 ambient = float3(r.x, g.x, b.x);
    float3 weights = ambient / (dot(ambient, float3(1.0f)) + 0.00001f);
    resolveUniform(p.ambientOut)[id] = float4(ambient, 0.0f);
    resolveUniform(p.directionalOut)[id] = float4(dot(float3(r.y, g.y, b.y), weights),
        dot(float3(r.z, g.z, b.z), weights), dot(float3(r.w, g.w, b.w), weights), 0.0f);
}
