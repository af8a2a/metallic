module AutoExposure;

import ShaderCore;
using Metallic;

// Histogram metering in physical luminance, before exposure and tone mapping.
// UE's extended luminance convention with lens attenuation 0.78: Lmax = 2^EV100.
static const uint kHistogramBins = 64u;
static const uint kHistogramWeight = 256u;
static const uint kReducePartialThreadsY = 4u;
static const uint kReduceMaxPartialGroups = 64u;
static const uint kPartialGroupFloat4Count = kHistogramBins / 4u;

#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"

[[vk::push_constant]] ConstantBuffer<AutoExposureParams> gParams;

#define gSource resolveUniform(gParams.source)
#define gHistogram gParams.histogram
// Element 0 holds persistent temporal adaptation state; [1..] hold per-group partial histogram bins.
#define gHistory gParams.history
// multiplier, adapted EV100, target EV100, metered luminance (cd/m^2).
#define gExposure gParams.exposure
#define gOutput resolveUniform(gParams.output)

groupshared uint sharedHistogram[kHistogramBins];
groupshared uint partialBinCounts[kReducePartialThreadsY][kHistogramBins];
groupshared float4 reducedPartialChunks[kReduceMaxPartialGroups];
groupshared float reducedHistogram[kHistogramBins];

float3 physicalColor(uint2 pixel, AutoExposurePush push)
{
    float3 color = gSource.Load(int3(pixel, 0)).rgb / push.sourceExposure;
    return float3(isfinite(color.x) ? max(color.x, 0.0) : 0.0,
        isfinite(color.y) ? max(color.y, 0.0) : 0.0,
        isfinite(color.z) ? max(color.z, 0.0) : 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void autoExposureHistogramMain(uint3 pixel : SV_DispatchThreadID, uint index : SV_GroupIndex,
    uint3 group : SV_GroupID)
{
    AutoExposurePush push = gParams.display;
    if (index < kHistogramBins) { sharedHistogram[index] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    if (push.bypass == 0u && pixel.x < push.width && pixel.y < push.height) {
        float luminance = Metallic.WorkingColor::luminance(physicalColor(pixel.xy, push));
        // Like UE's default black-bucket influence of zero: an unlit/hidden sky
        // must not dominate the meter and blow out the visible subject.
        if (luminance > exp2(push.histogramMin)) {
            float logLuminance = log2(luminance);
            float bin = saturate((logLuminance - push.histogramMin) /
                (push.histogramMax - push.histogramMin)) * float(kHistogramBins - 1u);
            uint lower = uint(bin);
            uint upperWeight = uint(frac(bin) * float(kHistogramWeight) + 0.5);
            // Interpolate adjacent bins to reduce exposure stepping on uniform skies.
            InterlockedAdd(sharedHistogram[lower], kHistogramWeight - upperWeight);
            InterlockedAdd(sharedHistogram[min(lower + 1u, kHistogramBins - 1u)], upperWeight);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (index < kHistogramBins) {
        uint tile = group.y * ((push.width + 15u) / 16u) + group.x;
        gHistogram.store(tile * kHistogramBins + index, sharedHistogram[index]);
    }
}

// Integrate a constant EV speed outside the transition band and an exponential
// inside it. Splitting a crossing step keeps the slope continuous and makes
// adaptation independent of frame rate, including at the band boundary.
float adaptEV100(float current, float target, AutoExposurePush push)
{
    float difference = target - current;
    float distance = abs(difference);
    float speed = difference > 0.0 ? push.speedUp : push.speedDown;
    float travel = speed * push.deltaSeconds;
    float transition = push.transitionDistance;
    if (transition <= 0.0) { return current + sign(difference) * min(distance, travel); }
    float linearTravel = min(max(distance - transition, 0.0), travel);
    distance -= linearTravel;
    travel -= linearTravel;
    distance *= exp(-travel / transition);
    return target - sign(difference) * distance;
}

uint autoExposurePartialGroupCount(AutoExposurePush push)
{
    if (push.bypass != 0u || push.tileCount == 0u) { return 1u; }
    return min(kReduceMaxPartialGroups,
        max(1u, (push.tileCount + kReducePartialThreadsY - 1u) / kReducePartialThreadsY));
}

[shader("compute")]
[numthreads(64, 4, 1)]
void autoExposureReducePartialMain(uint3 threadId : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    AutoExposurePush push = gParams.display;
    if (push.bypass != 0u || push.tileCount == 0u) { return; }
    uint groupCount = autoExposurePartialGroupCount(push);
    uint tilesPerGroup = (push.tileCount + groupCount - 1u) / groupCount;
    uint tileStart = group.x * tilesPerGroup;
    uint tileEnd = min(push.tileCount, tileStart + tilesPerGroup);
    uint count = 0u;
    for (uint tile = tileStart + threadId.y; tile < tileEnd; tile += kReducePartialThreadsY) {
        count += gHistogram.load(tile * kHistogramBins + threadId.x);
    }
    partialBinCounts[threadId.y][threadId.x] = count;
    GroupMemoryBarrierWithGroupSync();
    if (threadId.y == 0u) {
        partialBinCounts[0][threadId.x] = count + partialBinCounts[1][threadId.x] +
            partialBinCounts[2][threadId.x] + partialBinCounts[3][threadId.x];
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadId.y == 0u && threadId.x < kPartialGroupFloat4Count) {
        uint baseBin = threadId.x * 4u;
        float4 packed = float4(
            float(partialBinCounts[0][baseBin]),
            float(partialBinCounts[0][baseBin + 1u]),
            float(partialBinCounts[0][baseBin + 2u]),
            float(partialBinCounts[0][baseBin + 3u])) / float(kHistogramWeight);
        gHistory.store(1u + group.x * kPartialGroupFloat4Count + threadId.x, packed);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void autoExposureReduceMain(uint index : SV_GroupIndex)
{
    AutoExposurePush push = gParams.display;
    if (push.bypass != 0u) {
        if (index == 0u) {
            gHistory.store(0, float4(1.0, 0.0, 0.0, 0.0));
            gExposure.store(0, float4(1.0, 0.0, 0.0, 0.0));
        }
        return;
    }
    uint groupCount = autoExposurePartialGroupCount(push);
    uint binGroup = index & (kPartialGroupFloat4Count - 1u);
    uint slice = index >> 4u;
    float4 partial = float4(0.0, 0.0, 0.0, 0.0);
    for (uint group = slice; group < groupCount; group += 4u) {
        partial += gHistory.load(1u + group * kPartialGroupFloat4Count + binGroup);
    }
    reducedPartialChunks[index] = partial;
    GroupMemoryBarrierWithGroupSync();
    if (index < kPartialGroupFloat4Count) {
        float4 totalBins = reducedPartialChunks[index] +
            reducedPartialChunks[index + kPartialGroupFloat4Count] +
            reducedPartialChunks[index + kPartialGroupFloat4Count * 2u] +
            reducedPartialChunks[index + kPartialGroupFloat4Count * 3u];
        uint baseBin = index * 4u;
        reducedHistogram[baseBin] = totalBins.x;
        reducedHistogram[baseBin + 1u] = totalBins.y;
        reducedHistogram[baseBin + 2u] = totalBins.z;
        reducedHistogram[baseBin + 3u] = totalBins.w;
    }
    GroupMemoryBarrierWithGroupSync();
    if (index != 0u) { return; }
    float total = 0.0;
    for (uint bin = 0u; bin < kHistogramBins; ++bin) { total += reducedHistogram[bin]; }
    float low = total * push.lowPercent;
    float high = total * push.highPercent;
    float cumulative = 0.0;
    float sumLog = 0.0;
    float weight = 0.0;
    for (uint bin = 0u; bin < kHistogramBins; ++bin) {
        float next = cumulative + reducedHistogram[bin];
        float accepted = max(0.0, min(next, high) - max(cumulative, low));
        float logLuminance = lerp(push.histogramMin, push.histogramMax,
            float(bin) / float(kHistogramBins - 1u));
        sumLog += accepted * logLuminance;
        weight += accepted;
        cumulative = next;
    }
    float averageLog = weight > 0.0 ? sumLog / weight : push.histogramMin;
    // A metered gray card becomes 0.18 in linear display space, prior to the curve.
    float target = push.automatic != 0u
        ? clamp(averageLog - log2(0.18), push.minEV100, push.maxEV100) : push.manualEV100;
    float adapted = target;
    if (push.automatic != 0u && push.resetHistory == 0u && push.minEV100 < push.maxEV100) {
        float4 previous = gHistory.load(0);
        if (isfinite(previous.y) && previous.w > 0.0) { adapted = adaptEV100(previous.y, target, push); }
    }
    float4 exposure = float4(exp2(push.compensation - adapted), adapted, target, exp2(averageLog));
    float4 history = exposure;
    // An empty meter (e.g. while scene uploads finish) cannot seed adaptation.
    if (total == 0.0) { history.w = 0.0; }
    gHistory.store(0, history);
    gExposure.store(0, exposure);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void autoExposureApplyMain(uint3 pixel : SV_DispatchThreadID)
{
    AutoExposurePush push = gParams.display;
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    if (push.bypass != 0u) {
        gOutput[pixel.xy] = float4(gSource.Load(int3(pixel.xy, 0)).rgb, 1.0);
        return;
    }
    float3 color = physicalColor(pixel.xy, push) * gExposure.load(0).x * push.artisticExposure;
    // Display mapping belongs exclusively to FinalBlit.
    gOutput[pixel.xy] = float4(min(color, float3(65504.0)), 1.0);
}
