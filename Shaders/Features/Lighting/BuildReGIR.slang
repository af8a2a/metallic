#include "../../Libraries/Resources/ComputeResources.slang"

// GPU construction of a grid ReGIR light selector. Each grid cell owns a
// fixed-size set of RIS light slots, matching the selection flow used by RTXDI.

#define METALLIC_PUNCTUAL_SAMPLING_DISABLE_RESOURCES
#include "../../Libraries/Lighting/PunctualLightSampling.slang"

#define gLocalLightPdf METALLIC_RESOURCE(Texture2D<float>, 0)

#define gReGIRData METALLIC_RESOURCE(RWStructuredBuffer<uint4>, 1)

static const uint kReGIRHeaderRecordCount = 3u;
static const uint kReGIRMagic = 0x52454749u;

struct BuildReGIRPush
{
    uint lightCount;
    uint gridSize;
    uint lightsPerCell;
    uint buildSamples;
    uint frameIndex;
    uint padding0;
    uint lightSlotCount;
    uint padding1;
    float4 sceneCenterRadius;
    float samplingJitter;
    uint padding2;
    uint padding3;
    uint padding4;
};

float randomFloat(inout uint state)
{
    return punctualSamplingRandom(state);
}

uint zCurveToLinearIndex(uint2 position)
{
    uint index = 0u;
    for (uint bit = 0u; bit < 16u; ++bit) {
        index |= ((position.x >> bit) & 1u) << (bit * 2u);
        index |= ((position.y >> bit) & 1u) << (bit * 2u + 1u);
    }
    return index;
}

float loadImportanceWeight(uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gLocalLightPdf.GetDimensions(mipLevel, width, height, mipCount);
    if (position.x >= width || position.y >= height) {
        return 0.0;
    }
    float weight = gLocalLightPdf.Load(int3(position, mipLevel));
    return isfinite(weight) ? max(weight, 0.0) : 0.0;
}

bool sampleLocalLightPdf(inout uint seed, uint lightCount, out uint lightIndex, out float sourcePdf)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gLocalLightPdf.GetDimensions(0u, width, height, mipCount);
    uint2 position = uint2(0u, 0u);
    sourcePdf = 1.0;

    if (width > 0u && height > 0u && mipCount > 0u &&
        loadImportanceWeight(uint2(0u), mipCount - 1u) > 0.0) {
        for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
            position *= 2u;
            float4 weights = float4(
                loadImportanceWeight(position + uint2(0u, 0u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(0u, 1u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(1u, 0u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(1u, 1u), uint(mipLevel)));
            float largest = max(max(weights.x, weights.y), max(weights.z, weights.w));
            if (!(largest > 0.0)) { lightIndex = 0u; sourcePdf = 0.0; return false; }
            weights /= largest;
            float weightSum = weights.x + weights.y + weights.z + weights.w;
            weights /= weightSum;
            float randomValue = randomFloat(seed);
            if (randomValue < weights.x) {
                sourcePdf *= weights.x;
            } else if (randomValue < weights.x + weights.y) {
                position += uint2(0u, 1u);
                sourcePdf *= weights.y;
            } else if (randomValue < weights.x + weights.y + weights.z) {
                position += uint2(1u, 0u);
                sourcePdf *= weights.z;
            } else {
                position += uint2(1u, 1u);
                sourcePdf *= weights.w;
            }
        }
        lightIndex = zCurveToLinearIndex(position);
        if (sourcePdf > 0.0 && lightIndex < lightCount) {
            return true;
        }
        // Preserve a failed selected branch as a zero sample, rather than
        // changing proposal based on the outcome of its random draw.
        sourcePdf = 0.0;
        return false;
    }

    lightIndex = min(uint(randomFloat(seed) * float(lightCount)), lightCount - 1u);
    sourcePdf = 1.0 / float(lightCount);
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void buildReGIRMain(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    BuildReGIRPush push = METALLIC_CONSTANTS(BuildReGIRPush);
    uint lightSlot = dispatchThreadId.x;
    if (lightSlot >= push.lightSlotCount) {
        return;
    }

    float sceneRadius = max(push.sceneCenterRadius.w, 0.01);
    float cellSize = sceneRadius * 2.0 / float(max(push.gridSize, 1u));
    if (lightSlot == 0u) {
        gReGIRData[0] = uint4(push.gridSize, push.gridSize, push.gridSize, push.lightsPerCell);
        gReGIRData[1] = asuint(float4(push.sceneCenterRadius.xyz, cellSize));
        gReGIRData[2] = uint4(
            kReGIRMagic,
            push.lightSlotCount,
            asuint(max(push.samplingJitter, 0.0)),
            push.frameIndex);
    }

    uint cellIndex = lightSlot / push.lightsPerCell;
    uint3 cellPosition;
    cellPosition.x = cellIndex % push.gridSize;
    cellIndex /= push.gridSize;
    cellPosition.y = cellIndex % push.gridSize;
    cellPosition.z = cellIndex / push.gridSize;
    float3 gridOrigin = push.sceneCenterRadius.xyz - sceneRadius;
    float3 cellCenter = gridOrigin + (float3(cellPosition) + 0.5) * cellSize;
    float cellRadius = cellSize * sqrt(3.0) * (max(push.samplingJitter, 0.0) + 1.0);

    uint seed = punctualSamplingHash(
        lightSlot +
        (push.frameIndex + 1u) * 0x9e3779b9u +
        0x68bc21ebu);
    uint selectedLight = 0xffffffffu;
    float selectedTarget = 0.0;
    float weightSum = 0.0;
    uint lightCount = min(push.lightCount, punctualLightCount());
    for (uint sampleIndex = 0u; lightCount > 0u && sampleIndex < push.buildSamples; ++sampleIndex) {
        uint candidateLight;
        float sourcePdf;
        if (!sampleLocalLightPdf(seed, lightCount, candidateLight, sourcePdf)) {
            continue;
        }
        GpuPunctualLight light = gPunctualLights[candidateLight + 1u];
        float target = punctualLightTargetForVolume(light, cellCenter, cellRadius);
        float candidateWeight = target / (sourcePdf * float(push.buildSamples));
        if (!(candidateWeight > 0.0) || !isfinite(candidateWeight)) { continue; }
        weightSum += candidateWeight;
        if (randomFloat(seed) * weightSum < candidateWeight) {
            selectedLight = candidateLight;
            selectedTarget = target;
        }
    }

    float inverseSourcePdf = selectedTarget > 0.0
        ? weightSum / selectedTarget
        : 0.0;
    gReGIRData[kReGIRHeaderRecordCount + lightSlot] = uint4(
        selectedLight,
        asuint(inverseSourcePdf),
        0u,
        0u);
}
