// Histogram metering in physical luminance, before exposure and tone mapping.
// UE's extended luminance convention with lens attenuation 0.78: Lmax = 2^EV100.
static const uint kHistogramBins = 64u;
static const uint kHistogramWeight = 256u;

struct AutoExposurePush
{
    uint width;
    uint height;
    uint tileCount;
    uint resetHistory;
    float minEV100;
    float maxEV100;
    float compensation;
    float manualEV100;
    float lowPercent;
    float highPercent;
    float histogramMin;
    float histogramMax;
    float speedUp;
    float speedDown;
    float transitionDistance;
    float deltaSeconds;
    uint automatic;
    uint toneCurve;
    float sourceExposure;
    float artisticExposure;
};

[[vk::binding(0, 0)]] Texture2D<float4> gSource;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gHistogram;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gHistory;
// multiplier, adapted EV100, target EV100, metered luminance (cd/m^2).
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gExposure;
[[vk::binding(4, 0)]] RWTexture2D<float4> gOutput;

groupshared uint sharedHistogram[kHistogramBins];
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
    uint3 group : SV_GroupID, uniform AutoExposurePush push)
{
    if (index < kHistogramBins) { sharedHistogram[index] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    if (pixel.x < push.width && pixel.y < push.height) {
        float luminance = dot(physicalColor(pixel.xy, push), float3(0.2126, 0.7152, 0.0722));
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
        gHistogram[tile * kHistogramBins + index] = sharedHistogram[index];
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

[shader("compute")]
[numthreads(64, 1, 1)]
void autoExposureReduceMain(uint index : SV_GroupIndex, uniform AutoExposurePush push)
{
    float count = 0.0;
    for (uint tile = 0u; tile < push.tileCount; ++tile) {
        count += float(gHistogram[tile * kHistogramBins + index]) / float(kHistogramWeight);
    }
    reducedHistogram[index] = count;
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
        float4 previous = gHistory[0];
        if (isfinite(previous.y) && previous.w > 0.0) { adapted = adaptEV100(previous.y, target, push); }
    }
    float4 exposure = float4(exp2(push.compensation - adapted), adapted, target, exp2(averageLog));
    gHistory[0] = exposure;
    // An empty meter (e.g. while scene uploads finish) cannot seed adaptation.
    if (total == 0.0) { gHistory[0].w = 0.0; }
    gExposure[0] = exposure;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void autoExposureApplyMain(uint3 pixel : SV_DispatchThreadID, uniform AutoExposurePush push)
{
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float3 color = physicalColor(pixel.xy, push) * gExposure[0].x * push.artisticExposure;
    color = min(color, float3(1e30));
    if (push.toneCurve == 2u) {
        // Reference LookDev: preserve scene-linear values, then encode sRGB.
        // Values above display white clip; the upstream HDR output remains available.
        color = saturate(color);
        float3 srgb = select(color <= 0.0031308, color * 12.92,
            1.055 * pow(color, 1.0 / 2.4) - 0.055);
        gOutput[pixel.xy] = float4(srgb, 1.0);
        return;
    }
    color = push.toneCurve == 1u ? 1.0 - exp(-color) : color / (color + 1.0);
    gOutput[pixel.xy] = float4(pow(saturate(color), 1.0 / 2.2), 1.0);
}
