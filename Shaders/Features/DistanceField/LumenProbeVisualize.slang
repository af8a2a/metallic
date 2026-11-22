module LumenProbeVisualize;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import LumenAdaptiveProbes;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

typealias LumenProbeDiagnosticImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 10>;
typealias LumenProbeRadianceImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;

struct LumenProbeVisualizeParams
{
    ResourceHandle<StructuredBuffer<uint2>> probePositions;
    ResourceHandle<StructuredBuffer<float4>> probeRadiance;
    ResourceHandle<StructuredBuffer<float4>> filteredProbeRadiance;
    ResourceHandle<StructuredBuffer<float>> probeDistance;
    ResourceHandle<Texture2D<float4>> sceneColor;
    ResourceHandle<Texture2D<float>> historyWeight;
    ResourceHandle<LumenProbeDiagnosticImage> placement;
    ResourceHandle<LumenProbeDiagnosticImage> traceSource;
    ResourceHandle<LumenProbeRadianceImage> rawRadiance;
    ResourceHandle<LumenProbeRadianceImage> filteredRadiance;
    ResourceHandle<LumenProbeDiagnosticImage> hitDistance;
    ResourceHandle<LumenProbeDiagnosticImage> historyWeightView;
    uint width;
    uint height;
    uint probeWidth;
    uint probeHeight;
    uint probeSlot;
    uint traceResolution;
    float distanceRange;
};

// UE 5.7.4 LumenScreenProbeDebug.usf: DrawPoint paints a two-pixel quad,
// yellow for a uniform probe and magenta for an adaptive probe.
bool inProbeMarker(uint2 pixel, uint2 probePixel)
{
    const int2 delta = int2(pixel) - int2(probePixel);
    return probePixel.x != kLumenInvalidProbe && all(delta >= -1) && all(delta < 1);
}

float3 probePlacementColor(uint2 pixel, uint2 tile, uint2 probeSize, StructuredBuffer<uint2> positions,
    float3 sceneColor)
{
    // Adaptive placement can spill one tile forward. The marker can extend
    // one pixel backward from a tile's origin.
    const uint2 lo = max(tile, 1u) - 1u;
    const uint2 hi = min((pixel + 1u) / kLumenProbeDownsampleFactor, probeSize - 1u);
    bool uniformMarker = false;
    for (uint y = lo.y; y <= hi.y; ++y) {
        for (uint x = lo.x; x <= hi.x; ++x) {
            const uint index = lumenProbeIndex(uint2(x, y), 0u, probeSize.x);
            uniformMarker = uniformMarker || inProbeMarker(pixel, positions[index]);
            for (uint slot = 1u; slot < kLumenProbeSlotsPerTile; ++slot) {
                if (inProbeMarker(pixel, positions[index + slot])) {
                    return float3(1.0f, 0.0f, 1.0f);
                }
            }
        }
    }
    const float luminance = max(Metallic.WorkingColor::luminance(sceneColor), 0.0f);
    const float background = 0.45f * luminance / (1.0f + luminance);
    return uniformMarker ? float3(1.0f, 1.0f, 0.0f) : background.xxx;
}

float3 probeTraceSourceColor(uint source)
{
    // UE 5.7.4 LumenScreenProbeTracing.usf: Screen, Mesh SDF, and voxel
    // visualization colors are red, green, and blue. Other states are from
    // Metallic's Gather trace classification.
    switch (source) {
    case 5u: return float3(0.5f, 0.0f, 0.0f);
    case 9u: return float3(0.0f, 0.5f, 0.0f);
    case kGlobalDistanceFieldHit: return float3(0.0f, 0.0f, 0.5f);
    case 6u: return float3(0.0f, 0.5f, 0.5f);
    case kGlobalDistanceFieldUnavailable: return float3(0.95f, 0.28f, 0.08f);
    case kGlobalDistanceFieldPartial: return float3(0.29f, 0.20f, 0.41f);
    case kGlobalDistanceFieldStepLimit: return float3(0.86f, 0.14f, 0.44f);
    case kGlobalDistanceFieldMiss: return float3(0.18f);
    case kGlobalDistanceFieldEmpty: return float3(0.065f);
    case kGlobalDistanceFieldOutside: return float3(0.0f);
    default: return float3(1.0f);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenProbeVisualizeMain(uint3 id : SV_DispatchThreadID)
{
    const LumenProbeVisualizeParams params = getParameters<LumenProbeVisualizeParams>();
    const uint2 pixel = id.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const uint2 tile = pixel / kLumenProbeDownsampleFactor;
    const uint2 probeSize = uint2(params.probeWidth, params.probeHeight);
    StructuredBuffer<uint2> positions = resolveUniform(params.probePositions);
    resolveUniform(params.placement)[pixel] = float4(probePlacementColor(pixel, tile, probeSize, positions,
        resolveUniform(params.sceneColor).Load(int3(pixel, 0)).rgb), 1.0f);
    const float history = resolveUniform(params.historyWeight).Load(int3(pixel, 0));
    resolveUniform(params.historyWeightView)[pixel] = float4(history.xxx, 1.0f);

    const uint probe = lumenProbeIndex(tile, params.probeSlot, probeSize.x);
    const bool placed = positions[probe].x != kLumenInvalidProbe;
    const float missing = ((((pixel.x / 4u) ^ (pixel.y / 4u)) & 1u) != 0u) ? 0.035f : 0.015f;
    float3 sourceColor = missing.xxx;
    float3 distanceColor = missing.xxx;
    float3 raw = 0.0f;
    float3 filtered = 0.0f;
    if (placed) {
        // Preserve the producer's X-fast octahedral lane order. Raw traces
        // can be 8×8 or 16×16; the filtered atlas always has 8×8 directions.
        const uint2 local = pixel % kLumenProbeDownsampleFactor;
        const uint2 traceLane = local * params.traceResolution / kLumenProbeDownsampleFactor;
        const uint traceAddress = (probe * params.traceResolution + traceLane.y) * params.traceResolution + traceLane.x;
        const float4 trace = resolveUniform(params.probeRadiance)[traceAddress];
        const float distance = resolveUniform(params.probeDistance)[traceAddress];
        const uint2 gatherLane = local * kLumenProbeGatherResolution / kLumenProbeDownsampleFactor;
        filtered = resolveUniform(params.filteredProbeRadiance)[lumenProbeGatherAddress(probe, gatherLane)].rgb;
        raw = trace.rgb;
        sourceColor = probeTraceSourceColor(uint(trace.a));
        distanceColor = distance < 0.0f ? sourceColor : saturate(distance / params.distanceRange).xxx;
    }
    resolveUniform(params.traceSource)[pixel] = float4(sourceColor, 1.0f);
    resolveUniform(params.rawRadiance)[pixel] = float4(raw, 1.0f);
    resolveUniform(params.filteredRadiance)[pixel] = float4(filtered, 1.0f);
    resolveUniform(params.hitDistance)[pixel] = float4(distanceColor, 1.0f);
}
