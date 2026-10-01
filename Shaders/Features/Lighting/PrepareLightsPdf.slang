#include "../../../Source/Runtime/Render/Core/LightingKernelParameters.h"
[[vk::push_constant]] ConstantBuffer<PrepareLightsPdfParams> gParams;

import ShaderCore;
using Metallic;
import Lighting;
using Metallic.Lighting;

// GPU PDF preparation for RTXDI-style local-light and environment importance sampling.

static const uint kPrepareLocalLights = 0u;
static const uint kPrepareEnvironment = 1u;
static const uint kGenerateLocalMip = 2u;
static const uint kGenerateEnvironmentMip = 3u;
static const float kPi = 3.14159265358979323846;

#define gEnvironmentMap resolveDescriptor(gParams.environment)

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
    resolveDescriptor(gParams.destinationMip)[pdfPosition] = linearIndex < min(push.lightCount, min(uint(gParams.lights.data[0].positionRange.x), gParams.lights.length() - 1u))
        ? punctualLightPower(gParams.lights.data[linearIndex + 1u])
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
    resolveDescriptor(gParams.destinationMip)[position] = weight;
}

float loadLocalPdf(uint mipLevel, uint2 position, uint2 sourceSize)
{
    return all(position < sourceSize) ? max(resolveDescriptor(gParams.sourceMip)[position], 0.0) : 0.0;
}

float loadEnvironmentPdf(uint mipLevel, uint2 position, uint2 sourceSize)
{
    return all(position < sourceSize) ? max(resolveDescriptor(gParams.sourceMip)[position], 0.0) : 0.0;
}

void generateLocalMip(uint2 position, PrepareLightsPdfPush push)
{
    if (any(position >= push.destinationSize)) {
        return;
    }
    uint2 sourcePosition = position * 2u;
    float sum =
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 0u), push.sourceSize) * 0.25 +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 0u), push.sourceSize) * 0.25 +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 1u), push.sourceSize) * 0.25 +
        loadLocalPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 1u), push.sourceSize) * 0.25;
    resolveDescriptor(gParams.destinationMip)[position] = sum;
}

void generateEnvironmentMip(uint2 position, PrepareLightsPdfPush push)
{
    if (any(position >= push.destinationSize)) {
        return;
    }
    uint2 sourcePosition = position * 2u;
    float sum =
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 0u), push.sourceSize) * 0.25 +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 0u), push.sourceSize) * 0.25 +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(0u, 1u), push.sourceSize) * 0.25 +
        loadEnvironmentPdf(push.sourceMipLevel, sourcePosition + uint2(1u, 1u), push.sourceSize) * 0.25;
    resolveDescriptor(gParams.destinationMip)[position] = sum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void prepareLightsPdfMain(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    PrepareLightsPdfPush push = gParams.settings;
    if (push.mode == kPrepareLocalLights) {
        if (gParams.lights.length() == 0u) { return; }
        prepareLocalLightPdf(dispatchThreadId.xy, push);
    } else if (push.mode == kPrepareEnvironment) {
        prepareEnvironmentPdf(dispatchThreadId.xy, push);
    } else if (push.mode == kGenerateLocalMip) {
        generateLocalMip(dispatchThreadId.xy, push);
    } else if (push.mode == kGenerateEnvironmentMip) {
        generateEnvironmentMip(dispatchThreadId.xy, push);
    }
}
