// GPU PDF preparation for RTXDI-style local-light and environment importance sampling.

static const uint kMaxPdfMipCount = 16u;
static const uint kPrepareLocalLights = 0u;
static const uint kPrepareEnvironment = 1u;
static const uint kGenerateLocalMip = 2u;
static const uint kGenerateEnvironmentMip = 3u;
static const float kPi = 3.14159265358979323846;

[[vk::binding(0, 0)]]
Texture2D<float4> gEnvironmentMap;

[[vk::binding(1, 0)]]
RWTexture2D<float> gLocalPdfMips[kMaxPdfMipCount];

[[vk::binding(2, 0)]]
RWTexture2D<float> gEnvironmentPdfMips[kMaxPdfMipCount];

struct PrepareLightsPdfPush
{
    uint mode;
    uint lightCount;
    uint sourceMipLevel;
    uint padding0;
    uint2 sourceSize;
    uint2 destinationSize;
    float localLightIntensity;
    float sceneRadius;
    uint padding1;
    uint padding2;
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

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

uint2 linearIndexToZCurve(uint index)
{
    uint2 position = uint2(0u, 0u);
    for (uint bit = 0u; bit < 16u; ++bit) {
        position.x |= ((index >> (bit * 2u)) & 1u) << bit;
        position.y |= ((index >> (bit * 2u + 1u)) & 1u) << bit;
    }
    return position;
}

float localLightPower(uint lightIndex, PrepareLightsPdfPush push)
{
    uint value = lightIndex + 1u;
    float3 color = 0.2 + float3(
        hashToFloat(value * 3u + 0u),
        hashToFloat(value * 3u + 1u),
        hashToFloat(value * 3u + 2u)) * 0.8;
    color *= 0.65 + hashToFloat(lightIndex * 17u + 3u) * 0.7;
    float commonIntensity = max(push.localLightIntensity, 0.0) *
        max(push.sceneRadius, 0.01) * max(push.sceneRadius, 0.01) /
        float(max(push.lightCount, 1u));
    return max(luminance(color * commonIntensity), 0.0);
}

void prepareLocalLightPdf(uint2 dispatchPosition, PrepareLightsPdfPush push)
{
    if (any(dispatchPosition >= push.destinationSize)) {
        return;
    }

    uint linearIndex = dispatchPosition.y * push.destinationSize.x + dispatchPosition.x;
    uint2 pdfPosition = linearIndexToZCurve(linearIndex);
    if (any(pdfPosition >= push.destinationSize)) {
        return;
    }
    gLocalPdfMips[0][pdfPosition] = linearIndex < push.lightCount
        ? localLightPower(linearIndex, push)
        : 0.0;
}

void prepareEnvironmentPdf(uint2 position, PrepareLightsPdfPush push)
{
    if (any(position >= push.destinationSize)) {
        return;
    }

    float weight = 0.0;
    if (all(position < push.sourceSize)) {
        float3 radiance = gEnvironmentMap.Load(int3(position, 0)).rgb;
        radiance = all(isfinite(radiance)) ? max(radiance, 0.0) : float3(0.0, 0.0, 0.0);
        float theta = (float(position.y) + 0.5) *
            (kPi / float(max(push.sourceSize.y, 1u)));
        weight = luminance(radiance) * max(sin(theta), 0.0);
    }
    gEnvironmentPdfMips[0][position] = weight;
}

float loadLocalPdf(uint mipLevel, uint2 position, uint2 sourceSize)
{
    return all(position < sourceSize) ? max(gLocalPdfMips[mipLevel][position], 0.0) : 0.0;
}

float loadEnvironmentPdf(uint mipLevel, uint2 position, uint2 sourceSize)
{
    return all(position < sourceSize) ? max(gEnvironmentPdfMips[mipLevel][position], 0.0) : 0.0;
}

void generateLocalMip(uint2 position, PrepareLightsPdfPush push)
{
    if (any(position >= push.destinationSize)) {
        return;
    }
    uint2 sourcePosition = position * 2u;
    float sum =
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 0u), push.sourceSize) +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 0u), push.sourceSize) +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 1u), push.sourceSize) +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 1u), push.sourceSize);
    gLocalPdfMips[push.sourceMipLevel + 1u][position] = sum * 0.25;
}

void generateEnvironmentMip(uint2 position, PrepareLightsPdfPush push)
{
    if (any(position >= push.destinationSize)) {
        return;
    }
    uint2 sourcePosition = position * 2u;
    float sum =
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 0u), push.sourceSize) +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 0u), push.sourceSize) +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 1u), push.sourceSize) +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 1u), push.sourceSize);
    gEnvironmentPdfMips[push.sourceMipLevel + 1u][position] = sum * 0.25;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void prepareLightsPdfMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform PrepareLightsPdfPush push)
{
    if (push.mode == kPrepareLocalLights) {
        prepareLocalLightPdf(dispatchThreadId.xy, push);
    } else if (push.mode == kPrepareEnvironment) {
        prepareEnvironmentPdf(dispatchThreadId.xy, push);
    } else if (push.mode == kGenerateLocalMip) {
        generateLocalMip(dispatchThreadId.xy, push);
    } else if (push.mode == kGenerateEnvironmentMip) {
        generateEnvironmentMip(dispatchThreadId.xy, push);
    }
}
