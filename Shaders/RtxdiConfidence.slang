// RTXDI temporal lighting gradients converted to NRD history confidence.
// This follows the FullSample gradient strata, A-trous filtering, confidence
// conversion, and short non-linear temporal filtering flow.

[[vk::binding(0, 0)]]
RWTexture2D<float4> gNoisyDiffuse;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gNoisySpecular;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gBaseColorMetalness;

[[vk::binding(3, 0)]]
RWTexture2D<float4> gMotionVectors;

[[vk::binding(4, 0)]]
RWTexture2D<float2> gPreviousLuminance;

[[vk::binding(5, 0)]]
RWTexture2D<float2> gCurrentLuminance;

[[vk::binding(6, 0)]]
RWTexture2D<float4> gGradientA;

[[vk::binding(7, 0)]]
RWTexture2D<float4> gGradientB;

[[vk::binding(8, 0)]]
RWTexture2D<float> gPreviousDiffuseConfidence;

[[vk::binding(9, 0)]]
RWTexture2D<float> gPreviousSpecularConfidence;

[[vk::binding(10, 0)]]
RWTexture2D<float> gDiffuseConfidence;

[[vk::binding(11, 0)]]
RWTexture2D<float> gSpecularConfidence;

[[vk::binding(12, 0)]]
RWTexture2D<float> gCurrentDiffuseConfidence;

[[vk::binding(13, 0)]]
RWTexture2D<float> gCurrentSpecularConfidence;

static const uint kComputeGradient = 0u;
static const uint kFilterAToB = 1u;
static const uint kFilterBToA = 2u;
static const uint kResolveA = 3u;
static const uint kResolveB = 4u;
static const uint kGradientFactor = 3u;
static const float kFp16Maximum = 65504.0;

struct RtxdiConfidencePush
{
    uint mode;
    uint width;
    uint height;
    uint gradientWidth;
    uint gradientHeight;
    uint hasHistory;
    uint filterStep;
    uint padding0;
    float darknessBias;
    float sensitivity;
    float blendFactor;
    float padding1;
};

float luminance(float3 color)
{
    return dot(max(color, 0.0), float3(0.2126, 0.7152, 0.0722));
}

float2 currentLightingLuminance(uint2 pixel)
{
    float4 baseColorMetalness = gBaseColorMetalness[pixel];
    float3 baseColor = saturate(baseColorMetalness.rgb);
    float metalness = saturate(baseColorMetalness.a);
    float3 diffuseAlbedo = baseColor * (1.0 - metalness);
    float3 specularF0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metalness);
    float2 signal = float2(
        luminance(gNoisyDiffuse[pixel].rgb * diffuseAlbedo),
        luminance(gNoisySpecular[pixel].rgb * specularF0));
    return all(isfinite(signal)) ? max(signal, 0.0) : float2(0.0, 0.0);
}

bool previousPixelFor(uint2 pixel, RtxdiConfidencePush push, out int2 previousPixel)
{
    float2 motion = gMotionVectors[pixel].xy;
    float2 previousPosition = float2(pixel) + 0.5 +
        motion * float2(push.width, push.height);
    bool valid = push.hasHistory != 0u &&
        all(isfinite(previousPosition)) &&
        all(previousPosition >= 0.0) &&
        all(previousPosition < float2(push.width, push.height));
    previousPixel = valid ? int2(previousPosition) : int2(-1, -1);
    return valid;
}

void computeGradient(uint2 gradientPixel, RtxdiConfidencePush push)
{
    if (gradientPixel.x >= push.gradientWidth || gradientPixel.y >= push.gradientHeight) {
        return;
    }

    float2 selectedLuminance = float2(0.0, 0.0);
    float2 comparisonLuminance = float2(0.0, 0.0);
    [unroll]
    for (uint yy = 0u; yy < kGradientFactor; ++yy) {
        [unroll]
        for (uint xx = 0u; xx < kGradientFactor; ++xx) {
            uint2 pixel = gradientPixel * kGradientFactor + uint2(xx, yy);
            if (pixel.x >= push.width || pixel.y >= push.height) {
                continue;
            }

            float2 currentLuminance = currentLightingLuminance(pixel);
            gCurrentLuminance[pixel] = currentLuminance;
            int2 previousPixel;
            bool previousValid = previousPixelFor(pixel, push, previousPixel);
            float2 previousLuminance = previousValid
                ? max(gPreviousLuminance[previousPixel], 0.0)
                : float2(0.0, 0.0);

            float currentMaximum = max(currentLuminance.x, currentLuminance.y);
            float previousMaximum = max(previousLuminance.x, previousLuminance.y);
            float selectedMaximum = max(selectedLuminance.x, selectedLuminance.y);
            if (currentMaximum > selectedMaximum && currentMaximum > previousMaximum) {
                selectedLuminance = currentLuminance;
                comparisonLuminance = previousLuminance;
            } else if (previousMaximum > selectedMaximum) {
                selectedLuminance = previousLuminance;
                comparisonLuminance = currentLuminance;
            }
        }
    }

    float4 gradient = float4(0.0, 0.0, 0.0, 0.0);
    if (push.hasHistory != 0u && any(selectedLuminance > 0.0)) {
        gradient.xy = abs(selectedLuminance - comparisonLuminance);
        gradient.zw = max(selectedLuminance, comparisonLuminance);
    }
    gGradientA[gradientPixel] = min(gradient, kFp16Maximum);
}

float4 loadFilterSource(int2 position, bool sourceA, RtxdiConfidencePush push)
{
    if (any(position < 0) ||
        position.x >= int(push.gradientWidth) ||
        position.y >= int(push.gradientHeight)) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    return sourceA ? gGradientA[position] : gGradientB[position];
}

void filterGradient(uint2 gradientPixel, bool sourceA, RtxdiConfidencePush push)
{
    if (gradientPixel.x >= push.gradientWidth || gradientPixel.y >= push.gradientHeight) {
        return;
    }

    float4 accumulated = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [unroll]
    for (int yy = -1; yy <= 1; ++yy) {
        [unroll]
        for (int xx = -1; xx <= 1; ++xx) {
            int2 position = int2(gradientPixel) + int2(xx, yy) * int(push.filterStep);
            if (any(position < 0) ||
                position.x >= int(push.gradientWidth) ||
                position.y >= int(push.gradientHeight)) {
                continue;
            }
            float weight = (xx == 0 ? 1.0 : 0.5) * (yy == 0 ? 1.0 : 0.5);
            accumulated += loadFilterSource(position, sourceA, push) * weight;
            weightSum += weight;
        }
    }
    accumulated /= max(weightSum, 0.000001);
    if (sourceA) {
        gGradientB[gradientPixel] = accumulated;
    } else {
        gGradientA[gradientPixel] = accumulated;
    }
}

float4 loadGradientBorder(int2 position, bool sourceA, RtxdiConfidencePush push)
{
    if (any(position < 0) ||
        position.x >= int(push.gradientWidth) ||
        position.y >= int(push.gradientHeight)) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    return sourceA ? gGradientA[position] : gGradientB[position];
}

float4 sampleGradient(uint2 pixel, bool sourceA, RtxdiConfidencePush push)
{
    float2 position = (float2(pixel) + 0.5) / float(kGradientFactor) - 0.5;
    int2 base = int2(floor(position));
    float2 fraction = frac(position);
    float4 row0 = lerp(
        loadGradientBorder(base, sourceA, push),
        loadGradientBorder(base + int2(1, 0), sourceA, push),
        fraction.x);
    float4 row1 = lerp(
        loadGradientBorder(base + int2(0, 1), sourceA, push),
        loadGradientBorder(base + int2(1, 1), sourceA, push),
        fraction.x);
    return max(lerp(row0, row1, fraction.y), 0.0);
}

void resolveConfidence(uint2 pixel, bool sourceA, RtxdiConfidencePush push)
{
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float4 gradient = sampleGradient(pixel, sourceA, push);
    gradient.zw += max(push.darknessBias, 0.0);
    float diffuseConfidence = saturate(1.0 - gradient.x / max(gradient.z, 0.000001));
    float specularConfidence = saturate(1.0 - gradient.y / max(gradient.w, 0.000001));
    diffuseConfidence = saturate(pow(diffuseConfidence, max(push.sensitivity, 0.000001)));
    specularConfidence = saturate(pow(specularConfidence, max(push.sensitivity, 0.000001)));

    int2 previousPixel;
    if (push.blendFactor < 1.0 && previousPixelFor(pixel, push, previousPixel)) {
        const float power = 0.25;
        float previousDiffuse = saturate(gPreviousDiffuseConfidence[previousPixel]);
        float previousSpecular = saturate(gPreviousSpecularConfidence[previousPixel]);
        diffuseConfidence = pow(saturate(lerp(
            pow(previousDiffuse, power),
            pow(diffuseConfidence, power),
            push.blendFactor)), 1.0 / power);
        specularConfidence = pow(saturate(lerp(
            pow(previousSpecular, power),
            pow(specularConfidence, power),
            push.blendFactor)), 1.0 / power);
    }

    gDiffuseConfidence[pixel] = diffuseConfidence;
    gSpecularConfidence[pixel] = specularConfidence;
    gCurrentDiffuseConfidence[pixel] = diffuseConfidence;
    gCurrentSpecularConfidence[pixel] = specularConfidence;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rtxdiConfidenceMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform RtxdiConfidencePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (push.mode == kComputeGradient) {
        computeGradient(pixel, push);
    } else if (push.mode == kFilterAToB) {
        filterGradient(pixel, true, push);
    } else if (push.mode == kFilterBToA) {
        filterGradient(pixel, false, push);
    } else if (push.mode == kResolveA) {
        resolveConfidence(pixel, true, push);
    } else if (push.mode == kResolveB) {
        resolveConfidence(pixel, false, push);
    }
}
