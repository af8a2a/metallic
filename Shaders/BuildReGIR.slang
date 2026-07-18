// GPU construction of a grid ReGIR light selector. Each grid cell owns a
// fixed-size set of RIS light slots, matching the selection flow used by RTXDI.

[[vk::binding(0, 0)]]
Texture2D<float> gLocalLightPdf;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint4> gReGIRData;

static const uint kReGIRHeaderRecordCount = 3u;
static const uint kReGIRMagic = 0x52454749u;
static const float kGoldenAngle = 2.39996322972865332;

struct BuildReGIRPush
{
    uint lightCount;
    uint gridSize;
    uint lightsPerCell;
    uint buildSamples;
    uint frameIndex;
    uint animateLights;
    uint lightSlotCount;
    uint padding0;
    float4 sceneCenterRadius;
    float lightIntensity;
    float samplingJitter;
    uint padding1;
    uint padding2;
};

struct AnalyticLight
{
    float3 position;
    float3 radiance;
};

uint hash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float hashToFloat(uint value)
{
    return float(hash(value) & 0x00ffffffu) / 16777216.0;
}

float randomFloat(inout uint state)
{
    state = hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 hashColor(uint value)
{
    float3 color = float3(
        hashToFloat(value * 3u + 0u),
        hashToFloat(value * 3u + 1u),
        hashToFloat(value * 3u + 2u));
    return 0.2 + color * 0.8;
}

AnalyticLight getAnalyticLight(uint lightIndex, BuildReGIRPush push)
{
    float radius = max(push.sceneCenterRadius.w, 0.01);
    uint ringCount = 16u;
    uint ringIndex = lightIndex % ringCount;
    uint layerIndex = lightIndex / ringCount;
    uint layerCount = max((push.lightCount + ringCount - 1u) / ringCount, 1u);
    float layer = (float(layerIndex) + 0.5) / float(layerCount);
    float animation = push.animateLights != 0u
        ? float(push.frameIndex) * 0.0125
        : 0.0;
    float angle = float(lightIndex) * kGoldenAngle +
        animation * (0.55 + hashToFloat(lightIndex + 19u));
    float radialJitter = 0.55 + hashToFloat(lightIndex * 11u + 7u) * 0.5;
    float radialDistance = radius * radialJitter;
    float vertical = radius * lerp(-0.2, 1.4, layer);
    float wobble = push.animateLights != 0u
        ? sin(animation * (0.7 + hashToFloat(lightIndex + 31u)) + float(ringIndex)) * radius * 0.08
        : 0.0;

    AnalyticLight light;
    light.position = push.sceneCenterRadius.xyz + float3(
        cos(angle) * radialDistance,
        vertical + wobble,
        sin(angle) * radialDistance);
    float3 color = hashColor(lightIndex + 1u);
    color *= 0.65 + hashToFloat(lightIndex * 17u + 3u) * 0.7;
    light.radiance = color * max(push.lightIntensity, 0.0) * radius * radius /
        float(max(push.lightCount, 1u));
    return light;
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
    return max(gLocalLightPdf.Load(int3(position, mipLevel)), 0.0);
}

bool sampleLocalLightPdf(inout uint seed, uint lightCount, out uint lightIndex, out float sourcePdf)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gLocalLightPdf.GetDimensions(0u, width, height, mipCount);
    uint2 position = uint2(0u, 0u);
    sourcePdf = 1.0;

    if (width > 0u && height > 0u && mipCount > 1u) {
        for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
            position *= 2u;
            float4 weights = float4(
                loadImportanceWeight(position + uint2(0u, 0u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(0u, 1u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(1u, 0u), uint(mipLevel)),
                loadImportanceWeight(position + uint2(1u, 1u), uint(mipLevel)));
            float weightSum = weights.x + weights.y + weights.z + weights.w;
            if (weightSum <= 0.0) {
                sourcePdf = 0.0;
                break;
            }
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
    }

    lightIndex = min(uint(randomFloat(seed) * float(lightCount)), lightCount - 1u);
    sourcePdf = 1.0 / float(lightCount);
    return true;
}

float lightTargetForCell(AnalyticLight light, float3 cellCenter, float cellRadius)
{
    float distanceToCenter = length(light.position - cellCenter);
    float nonlinearFactor = 1.1547;
    float distanceToCell = distanceToCenter + cellRadius * cellRadius * cellRadius /
        max(
            (distanceToCenter + cellRadius * nonlinearFactor) *
                (distanceToCenter + cellRadius * nonlinearFactor),
            0.000001);
    return max(luminance(light.radiance), 0.0) /
        max(distanceToCell * distanceToCell, 0.000001);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void buildReGIRMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform BuildReGIRPush push)
{
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

    uint seed = hash(
        lightSlot +
        (push.frameIndex + 1u) * 0x9e3779b9u +
        0x68bc21ebu);
    uint selectedLight = 0xffffffffu;
    float selectedTarget = 0.0;
    float weightSum = 0.0;
    for (uint sampleIndex = 0u; sampleIndex < push.buildSamples; ++sampleIndex) {
        uint candidateLight;
        float sourcePdf;
        if (!sampleLocalLightPdf(seed, push.lightCount, candidateLight, sourcePdf)) {
            continue;
        }
        AnalyticLight light = getAnalyticLight(candidateLight, push);
        float target = lightTargetForCell(light, cellCenter, cellRadius);
        float candidateWeight = target /
            max(sourcePdf * float(push.buildSamples), 0.00000001);
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
