// GPU projection and cosine convolution: irradiance SH through l=2.
#include "SphericalHarmonics.slang"
#include "ProceduralEnvironment.slang"

static const uint kIntegratePartials = 0u;
static const uint kFinalizeCoefficients = 1u;
static const uint kThreadCount = 128u;
static const uint kCoefficientCount = SH::L2RGB::kCoefficientCount;
static const float kPi = 3.14159265358979323846;

[[vk::binding(0, 0)]]
Texture2D<float4> gEnvironmentMap;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float4> gPartials;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float4> gCoefficients;

struct EnvironmentLightingPrecomputePush
{
    uint mode;
    uint width;
    uint height;
    uint partialCount;
    uint dispatchWidth;
    uint procedural;
    uint padding1;
    uint padding2;
};

groupshared float4 gShared[kCoefficientCount * kThreadCount];

void integratePartials(
    uint3 groupId,
    uint groupThreadIndex,
    EnvironmentLightingPrecomputePush push)
{
    uint partialIndex = groupId.y * push.dispatchWidth + groupId.x;
    uint texelIndex = partialIndex * kThreadCount + groupThreadIndex;
    float3 weightedRadiance = float3(0.0);
    float3 direction = float3(0.0, 1.0, 0.0);
    uint texelCount = push.width * push.height;
    if (partialIndex < push.partialCount && texelIndex < texelCount) {
        uint2 texel = uint2(texelIndex % push.width, texelIndex / push.width);
        float2 uv = (float2(texel) + 0.5) / float2(push.width, push.height);
        float theta = uv.y * kPi;
        float phi = (uv.x - 0.5) * (2.0 * kPi);
        float sineTheta = sin(theta);
        direction = float3(cos(phi) * sineTheta, cos(theta), sin(phi) * sineTheta);
        float solidAngle = (2.0 * kPi / float(push.width)) *
            (cos(float(texel.y) * kPi / float(push.height)) -
                cos(float(texel.y + 1u) * kPi / float(push.height)));
        float3 radiance = push.procedural != 0u ? proceduralEnvironmentLocalRadiance(direction)
            : gEnvironmentMap.Load(int3(texel, 0)).rgb;
        radiance = all(isfinite(radiance)) ? max(radiance, 0.0) : float3(0.0);
        weightedRadiance = radiance * solidAngle;
    }

    SH::L2RGB projected = SH::projectOntoL2(direction, weightedRadiance);
    for (uint coefficientIndex = 0u; coefficientIndex < kCoefficientCount; ++coefficientIndex) {
        gShared[coefficientIndex * kThreadCount + groupThreadIndex] =
            float4(projected.c[coefficientIndex], 0.0);
    }
    GroupMemoryBarrierWithGroupSync();

    if (partialIndex < push.partialCount && groupThreadIndex < kCoefficientCount) {
        float4 sum = float4(0.0);
        uint baseIndex = groupThreadIndex * kThreadCount;
        for (uint threadIndex = 0u; threadIndex < kThreadCount; ++threadIndex) {
            sum += gShared[baseIndex + threadIndex];
        }
        gPartials[partialIndex * kCoefficientCount + groupThreadIndex] = sum;
    }
}

void finalizeCoefficient(
    uint3 groupId,
    uint groupThreadIndex,
    EnvironmentLightingPrecomputePush push)
{
    uint coefficientIndex = groupId.x;
    float4 sum = float4(0.0);
    if (coefficientIndex < kCoefficientCount) {
        for (uint partialIndex = groupThreadIndex;
             partialIndex < push.partialCount;
             partialIndex += kThreadCount) {
            sum += gPartials[partialIndex * kCoefficientCount + coefficientIndex];
        }
    }
    gShared[groupThreadIndex] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = kThreadCount / 2u; stride > 0u; stride /= 2u) {
        if (groupThreadIndex < stride) {
            gShared[groupThreadIndex] += gShared[groupThreadIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadIndex == 0u && coefficientIndex < kCoefficientCount) {
        // Each workgroup owns one coefficient; retain the parallel reduction
        // and pack the same cosine-convolved irradiance ABI consumed by SH.
        gCoefficients[coefficientIndex] = gShared[0] * SH::cosineConvolutionWeight(coefficientIndex);
    }
}

[shader("compute")]
[numthreads(kThreadCount, 1, 1)]
void environmentLightingPrecomputeMain(
    uint3 groupId : SV_GroupID,
    uint groupThreadIndex : SV_GroupIndex,
    uniform EnvironmentLightingPrecomputePush push)
{
    if (push.mode == kIntegratePartials) {
        integratePartials(groupId, groupThreadIndex, push);
    } else if (push.mode == kFinalizeCoefficients) {
        finalizeCoefficient(groupId, groupThreadIndex, push);
    }
}
