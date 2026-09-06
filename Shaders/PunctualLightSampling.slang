#ifndef METALLIC_PUNCTUAL_LIGHT_SAMPLING
#define METALLIC_PUNCTUAL_LIGHT_SAMPLING

#include "PhotometricLighting.slang"

static const uint kPunctualReGIRMagic = 0x52454749u;
static const uint kPunctualReGIRHeaderRecordCount = 3u;

uint punctualSamplingHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float punctualSamplingRandom(inout uint state)
{
    state = punctualSamplingHash(state + 0x9e3779b9u);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float punctualIntensityLuminance(GpuPunctualLight light)
{
    float value = dot(max(light.colorIntensity.rgb, 0.0), float3(0.2126, 0.7152, 0.0722)) *
        max(light.colorIntensity.w, 0.0);
    return isfinite(value) ? max(value, 0.0) : 0.0;
}

float punctualLightPower(GpuPunctualLight light)
{
    const float pi = 3.14159265358979323846;
    float intensity = punctualIntensityLuminance(light);
    // Infinite lights have no finite emitted power. Their illuminance provides
    // a positive importance proxy; this never changes physical light evaluation.
    if (light.directionType.w < 0.5) { return intensity; }
    float solidAngle = 4.0 * pi;
    if (light.directionType.w > 1.5) {
        float innerCosine = clamp(light.spot.x, -1.0, 1.0);
        float outerCosine = clamp(light.spot.y, -1.0, innerCosine);
        // Integral of the same squared cosine ramp used by samplePunctualLight.
        solidAngle = 2.0 * pi * ((1.0 - innerCosine) + (innerCosine - outerCosine) / 3.0);
    }
    float power = intensity * solidAngle;
    return isfinite(power) && power > 0.0 ? power : intensity;
}

float punctualLightTargetForVolume(GpuPunctualLight light, float3 center, float radius)
{
    float intensity = punctualIntensityLuminance(light);
    if (light.directionType.w < 0.5) { return intensity; }
    float distance = length(light.positionRange.xyz - center);
    float denominator = max(distance + radius * 1.1547, 1e-6);
    float averageDistance = distance + radius * (radius / denominator) * (radius / denominator);
    // Keep support over the entire volume, including spot/range boundaries.
    // Evaluating the cone or finite range only at its center can discard lights
    // that illuminate a receiver elsewhere in the cell and bias the estimator.
    return intensity / max(averageDistance * averageDistance, 1e-6);
}

uint punctualZCurveToLinearIndex(uint2 position)
{
    uint index = 0u;
    for (uint bit = 0u; bit < 16u; ++bit) {
        index |= ((position.x >> bit) & 1u) << (bit * 2u);
        index |= ((position.y >> bit) & 1u) << (bit * 2u + 1u);
    }
    return index;
}

#ifndef METALLIC_PUNCTUAL_SAMPLING_DISABLE_RESOURCES
[[vk::binding(52, 0)]]
StructuredBuffer<uint4> gPunctualReGIR;

[[vk::binding(53, 0)]]
Texture2D<float> gPunctualLightPdf;

float punctualPdfWeight(uint2 position, uint mipLevel)
{
    uint width, height, mipCount;
    gPunctualLightPdf.GetDimensions(mipLevel, width, height, mipCount);
    if (position.x >= width || position.y >= height) { return 0.0; }
    float value = gPunctualLightPdf.Load(int3(position, mipLevel));
    return isfinite(value) ? max(value, 0.0) : 0.0;
}

bool selectPunctualLightGlobal(inout uint seed, out uint index, out float inverseSourcePdf)
{
    index = 0u;
    inverseSourcePdf = 0.0;
    uint count = punctualLightCount();
    if (count == 0u) { return false; }
    uint width, height, mipCount;
    gPunctualLightPdf.GetDimensions(0u, width, height, mipCount);
    if (width > 0u && height > 0u && mipCount > 0u &&
        punctualPdfWeight(uint2(0u), mipCount - 1u) > 0.0) {
        uint2 position = uint2(0u);
        float pdf = 1.0;
        for (int mip = int(mipCount) - 2; mip >= 0; --mip) {
            position *= 2u;
            float4 weights = float4(
                punctualPdfWeight(position + uint2(0u, 0u), uint(mip)),
                punctualPdfWeight(position + uint2(0u, 1u), uint(mip)),
                punctualPdfWeight(position + uint2(1u, 0u), uint(mip)),
                punctualPdfWeight(position + uint2(1u, 1u), uint(mip)));
            float largest = max(max(weights.x, weights.y), max(weights.z, weights.w));
            if (!(largest > 0.0)) { return false; }
            weights /= largest;
            float total = dot(weights, float4(1.0));
            if (!(total > 0.0) || !isfinite(total)) { return false; }
            weights /= total;
            float random = punctualSamplingRandom(seed);
            if (random < weights.x) { pdf *= weights.x; }
            else if (random < weights.x + weights.y) {
                position += uint2(0u, 1u); pdf *= weights.y;
            } else if (random < weights.x + weights.y + weights.z) {
                position += uint2(1u, 0u); pdf *= weights.z;
            } else {
                position += uint2(1u, 1u); pdf *= weights.w;
            }
        }
        index = punctualZCurveToLinearIndex(position);
        if (index < count && pdf > 0.0 && isfinite(1.0 / pdf)) {
            inverseSourcePdf = 1.0 / pdf;
            return true;
        }
        // A branch has already been drawn. Resampling a failed branch from the
        // uniform proposal would add an unaccounted, selection-dependent term.
        index = 0u;
        return false;
    }
    index = min(uint(punctualSamplingRandom(seed) * float(count)), count - 1u);
    inverseSourcePdf = float(count);
    return true;
}

bool selectPunctualLight(float3 position, inout uint seed, out uint index, out float inverseSourcePdf)
{
    index = 0u;
    inverseSourcePdf = 0.0;
    if (punctualLightCount() == 0u) { return false; }
    uint recordCount, stride;
    gPunctualReGIR.GetDimensions(recordCount, stride);
    if (recordCount < kPunctualReGIRHeaderRecordCount) {
        return selectPunctualLightGlobal(seed, index, inverseSourcePdf);
    }
    uint4 grid = gPunctualReGIR[0];
    float4 centerSize = asfloat(gPunctualReGIR[1]);
    uint4 metadata = gPunctualReGIR[2];
    float jitter = asfloat(metadata.z);
    if (metadata.x != kPunctualReGIRMagic || any(grid.xyz == uint3(0u)) || grid.w == 0u ||
        metadata.y == 0u || metadata.y > recordCount - kPunctualReGIRHeaderRecordCount ||
        !all(isfinite(centerSize)) || !(centerSize.w > 0.0) || !isfinite(jitter) || jitter < 0.0) {
        return selectPunctualLightGlobal(seed, index, inverseSourcePdf);
    }
    float3 random = float3(punctualSamplingRandom(seed), punctualSamplingRandom(seed),
        punctualSamplingRandom(seed));
    float3 samplePosition = position + (random - 0.5) * (jitter * centerSize.w * 2.0);
    float3 origin = centerSize.xyz - float3(grid.xyz) * (centerSize.w * 0.5);
    float3 cellCoordinates = floor((samplePosition - origin) / centerSize.w);
    if (!all(isfinite(cellCoordinates)) || any(cellCoordinates < 0.0) ||
        any(cellCoordinates >= float3(grid.xyz))) {
        return selectPunctualLightGlobal(seed, index, inverseSourcePdf);
    }
    uint3 cell = uint3(cellCoordinates);
    uint cellIndex = cell.x + (cell.y + cell.z * grid.y) * grid.x;
    uint slotBase = cellIndex * grid.w;
    if (slotBase >= metadata.y || grid.w > metadata.y - slotBase) {
        return selectPunctualLightGlobal(seed, index, inverseSourcePdf);
    }
    uint slot = slotBase + min(uint(punctualSamplingRandom(seed) * float(grid.w)), grid.w - 1u);
    uint4 data = gPunctualReGIR[kPunctualReGIRHeaderRecordCount + slot];
    float inversePdf = asfloat(data.y);
    // An empty RIS slot is a zero-weight sample, not permission to draw a second
    // sample from another distribution. Such a fallback changes the estimator.
    if (data.x >= punctualLightCount() || !(inversePdf > 0.0) || !isfinite(inversePdf)) { return false; }
    index = data.x;
    inverseSourcePdf = inversePdf;
    return true;
}
#endif
#endif
