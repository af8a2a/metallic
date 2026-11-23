module LumenScreenProbeImportance;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
import LumenAdaptiveProbes;
import LumenScreenProbeWork;
import LumenProbeMath;
import LumenProbeImportance;
import LumenRadianceCache;
import LumenRadianceCacheInterpolation;

using Metallic;
using Metallic.Lumen;

struct LumenScreenProbeImportanceParams
{
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float2>> blueNoiseVec2;
    ResourceHandle<StructuredBuffer<uint2>> probePositions;
    ResourceHandle<StructuredBuffer<uint>> probeWorkList;
    ResourceHandle<StructuredBuffer<uint>> probeWorkState;
    ResourceHandle<StructuredBuffer<ViewConstants>> view;
    ResourceHandle<StructuredBuffer<uint>> cacheIndirection;
    ResourceHandle<StructuredBuffer<LumenRadianceCacheClipmap>> cacheClipmaps;
    ResourceHandle<Texture2D<float4>> cacheRadiance;
    ResourceHandle<RWStructuredBuffer<float>> brdfPdf;
    ResourceHandle<RWStructuredBuffer<float>> lightingPdf;
    ResourceHandle<RWStructuredBuffer<uint>> rayInfo;
    ResourceHandle<StructuredBuffer<float4>> filteredRadiance;
    ResourceHandle<StructuredBuffer<float4>> previousRadiance;
    ResourceHandle<StructuredBuffer<float4>> previousGeometry;
    ResourceHandle<RWStructuredBuffer<float4>> savedRadiance;
    ResourceHandle<RWStructuredBuffer<float4>> savedGeometry;
    uint2 viewSize;
    uint2 probeSize;
    uint frameIndex;
    uint rayDirectionFrameIndex;
    uint historyValid;
    SamplerHandle cacheSampler;
};

bool resolveImportanceProbe(LumenScreenProbeImportanceParams params, uint3 group, out uint probeIndex)
{
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    return resolveLumenActiveProbe(group, resolveUniform(params.probeWorkList),
        resolveUniform(params.probeWorkState), probeIndex);
#else
    probeIndex = lumenProbeIndex(group.xy, group.z, params.probeSize.x);
    return true;
#endif
}

uint uniformProbeIndex(uint2 tile, uint probeWidth)
{
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
    return tile.x + tile.y * probeWidth;
#else
    return lumenProbeIndex(tile, 0u, probeWidth);
#endif
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenComputeProbeBrdfPdfMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeImportanceParams params = getParameters<LumenScreenProbeImportanceParams>();
    uint probeIndex;
    if (!resolveImportanceProbe(params, group, probeIndex)) { return; }
    uint2 probePixel = resolveUniform(params.probePositions)[probeIndex];
    if (probePixel.x == kLumenInvalidProbe) { return; }
    ViewConstants view = resolveUniform(params.view)[0];
    ThreeBandSHVector pdf;
    lumenEvaluateProbeBrdfPdf(resolveUniform(params.position), resolveUniform(params.normal), view.current,
        view.jitter.xy, params.viewSize, probePixel, lane.xy, kLumenProbeDownsampleFactor, pdf);
    if (all(lane.xy == 0u)) {
        RWStructuredBuffer<float> output = resolveUniform(params.brdfPdf);
        uint base = probeIndex * 9u;
        [unroll]
        for (uint coefficient = 0u; coefficient < 4u; ++coefficient) {
            output[base + coefficient] = pdf.v0[coefficient];
            output[base + coefficient + 4u] = pdf.v1[coefficient];
        }
        output[base + 8u] = pdf.v2;
    }
}

groupshared float sharedLightingPdf[64 * 4];

// UE 5.7.4 LumenScreenProbeImportanceSampling.usf, incoming-lighting history and PDF.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenComputeProbeLightingPdfMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeImportanceParams params = getParameters<LumenScreenProbeImportanceParams>();
    uint probeIndex;
    if (!resolveImportanceProbe(params, group, probeIndex)) { return; }
    StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
    uint2 probePixel = probes[probeIndex];
    if (probePixel.x == kLumenInvalidProbe) { return; }

    float4 receiver = resolveUniform(params.position).Load(int3(probePixel, 0));
    float3 lighting = 0.0f;
    float transparency = 1.0f;
    if (params.historyValid != 0u) {
        ViewConstants view = resolveUniform(params.view)[0];
        ViewCameraConstants camera = view.previous;
        float2 historyScreenUV;
        if (ScenePath::projectWorldToUv(receiver.xyz, camera.eye, camera.center, camera.upProjection,
                camera.viewport, camera.clipOrtho, historyScreenUV)) {
            historyScreenUV -= view.jitter.zw / float2(params.viewSize);
            if (all(historyScreenUV >= 0.0f) && all(historyScreenUV <= 1.0f)) {
                uint2 jitter = lumenUniformProbePosition(0u, uint2(kLumenProbeDownsampleFactor), params.frameIndex - 1u);
                float2 invViewSize = 1.0f / float2(params.viewSize);
                float2 historyProbeCoord = (historyScreenUV - (float2(jitter) + 0.5f) * invViewSize)
                    / (float(kLumenProbeDownsampleFactor) * invViewSize);
                float3 normal = normalize(resolveUniform(params.normal).Load(int3(probePixel, 0)).xyz);
                float4 plane = float4(normal, dot(receiver.xyz, normal));
                StructuredBuffer<float4> historyGeometry = resolveUniform(params.previousGeometry);
                StructuredBuffer<float4> historyRadiance = resolveUniform(params.previousRadiance);
                float historyWeight = 0.0f;
                for (uint y = 0u; y < 2u; ++y) {
                    for (uint x = 0u; x < 2u; ++x) {
                        uint2 neighbor = uint2(clamp(historyProbeCoord + float2(x, y), 0.0f, float2(params.probeSize - 1u)));
                        uint neighborIndex = neighbor.x + neighbor.y * params.probeSize.x;
                        // UE 5.7.4's GetScreenProbeDepth overload reads the current probe depth atlas.
                        if (probes[uniformProbeIndex(neighbor, params.probeSize.x)].x != kLumenInvalidProbe) {
                            float4 geometry = historyGeometry[neighborIndex];
                            float planeDistance = abs(dot(float4(geometry.xyz, -1), plane));
                            float relativeDepthDifference = planeDistance / receiver.w;
                            float positionWeight = exp2(-10000.0f * (relativeDepthDifference * relativeDepthDifference)) > .1f ? 1.0f : 0.0f;
                            lighting += historyRadiance[lumenProbeGatherAddress(neighborIndex, lane.xy)].xyz * positionWeight;
                            historyWeight += positionWeight;
                        }
                    }
                }
                if (historyWeight > 0.0f) { lighting /= historyWeight; }
                transparency = 1.0f - saturate(historyWeight / 4.0f);
            }
        }
    }

    if (transparency > 0.0f) {
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        uint2 screenTileCoord = lumenScreenTileCoord(probePixel, params.frameIndex);
        float2 center = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoiseVec2), screenTileCoord,
            params.rayDirectionFrameIndex);
        float3 direction = lumenEquiAreaSphericalMapping((float2(lane.xy) + center) / 8.0f);
        LumenRadianceCacheInputs cache = {resolveUniform(params.cacheIndirection), resolveUniform(params.cacheClipmaps),
            resolveUniform(params.cacheRadiance), resolveUniform(params.cacheSampler)};
        LumenRadianceCacheCoverage coverage = getLumenRadianceCacheCoverage(cache, receiver.xyz);
        if (coverage.valid) {
            lighting += sampleLumenRadianceCache(cache, coverage, receiver.xyz, direction) * transparency;
        } else
#endif
        {
            lighting = 1.0f;
        }
    }

    float pdf = WorkingColor::toXYZ(lighting).y;
    uint threadIndex = lane.y * 8u + lane.x;
    sharedLightingPdf[threadIndex] = pdf;
    GroupMemoryBarrierWithGroupSync();
    uint count = 64u;
    uint offset = 0u;
    while (count > 1u) {
        uint base = threadIndex * 4u;
        if (base < count) {
            float localPdf = sharedLightingPdf[base + offset];
            if (base + 1u < count) { localPdf += sharedLightingPdf[base + 1u + offset]; }
            if (base + 2u < count) { localPdf += sharedLightingPdf[base + 2u + offset]; }
            if (base + 3u < count) { localPdf += sharedLightingPdf[base + 3u + offset]; }
            sharedLightingPdf[threadIndex + offset + count] = localPdf;
        }
        offset += count;
        count = (count + 3u) / 4u;
        GroupMemoryBarrierWithGroupSync();
    }
    float sumPdf = sharedLightingPdf[offset];
    resolveUniform(params.lightingPdf)[lumenProbeTraceAddress(probeIndex, lane.xy)] =
        f16tof32(f32tof16(pdf / max(sumPdf, 0.0001f)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenGenerateStructuredProbeRaysMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeImportanceParams params = getParameters<LumenScreenProbeImportanceParams>();
    uint probeIndex;
    if (!resolveImportanceProbe(params, group, probeIndex)) { return; }
    uint address = lumenProbeTraceAddress(probeIndex, lane.xy);
    uint2 probePixel = resolveUniform(params.probePositions)[probeIndex];
    if (probePixel.x == kLumenInvalidProbe) {
        resolveUniform(params.rayInfo)[address] = 0xffffffffu;
        return;
    }
    RWStructuredBuffer<float> input = resolveUniform(params.brdfPdf);
    uint base = probeIndex * 9u;
    ThreeBandSHVector brdfPdf;
    [unroll]
    for (uint coefficient = 0u; coefficient < 4u; ++coefficient) {
        brdfPdf.v0[coefficient] = input[base + coefficient];
        brdfPdf.v1[coefficient] = input[base + coefficient + 4u];
    }
    brdfPdf.v2 = input[base + 8u];
    uint metadata = lumenGenerateProbeRay(lane.xy, brdfPdf, resolveUniform(params.lightingPdf)[address]);
    resolveUniform(params.rayInfo)[address] = metadata;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenSaveProbeLightingHistoryMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeImportanceParams params = getParameters<LumenScreenProbeImportanceParams>();
    uint probeIndex = uniformProbeIndex(group.xy, params.probeSize.x);
    uint historyIndex = group.x + group.y * params.probeSize.x;
    uint2 probePixel = resolveUniform(params.probePositions)[probeIndex];
    bool valid = probePixel.x != kLumenInvalidProbe;
    float4 radiance = 0.0f;
    if (valid) { radiance = resolveUniform(params.filteredRadiance)[lumenProbeGatherAddress(probeIndex, lane.xy)]; }
    resolveUniform(params.savedRadiance)[lumenProbeGatherAddress(historyIndex, lane.xy)] = radiance;
    if (all(lane.xy == 0u)) {
        float4 geometry = float4(0.0f, 0.0f, 0.0f, -1.0f);
        if (valid) { geometry = resolveUniform(params.position).Load(int3(probePixel, 0)); }
        resolveUniform(params.savedGeometry)[historyIndex] = geometry;
    }
}
