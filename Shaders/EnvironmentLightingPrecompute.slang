// GPU precomputation for environment spherical harmonics.

static const uint kIntegratePartials = 0u;
static const uint kFinalizeCoefficients = 1u;
static const uint kThreadCount = 128u;
static const uint kCoefficientCount = 9u;
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
    uint padding0;
    uint padding1;
    uint padding2;
};

groupshared float4 gShared[kCoefficientCount * kThreadCount];

float shBasis(uint coefficientIndex, float3 direction)
{
    switch (coefficientIndex) {
    case 0u: return 0.282095;
    case 1u: return 0.488603 * direction.y;
    case 2u: return 0.488603 * direction.z;
    case 3u: return 0.488603 * direction.x;
    case 4u: return 1.092548 * direction.x * direction.y;
    case 5u: return 1.092548 * direction.y * direction.z;
    case 6u: return 0.315392 * (3.0 * direction.z * direction.z - 1.0);
    case 7u: return 1.092548 * direction.x * direction.z;
    default: return 0.546274 * (direction.x * direction.x - direction.y * direction.y);
    }
}

void integratePartials(
    uint3 groupId,
    uint groupThreadIndex,
    EnvironmentLightingPrecomputePush push)
{
    uint partialIndex = groupId.y * push.dispatchWidth + groupId.x;
    uint texelIndex = partialIndex * kThreadCount + groupThreadIndex;
    float4 weightedRadiance = float4(0.0);
    float3 direction = float3(0.0, 1.0, 0.0);
    uint texelCount = push.width * push.height;
    if (partialIndex < push.partialCount && texelIndex < texelCount) {
        uint2 texel = uint2(texelIndex % push.width, texelIndex / push.width);
        float2 uv = (float2(texel) + 0.5) / float2(push.width, push.height);
        float theta = uv.y * kPi;
        float phi = (uv.x - 0.5) * (2.0 * kPi);
        float sineTheta = sin(theta);
        direction = float3(cos(phi) * sineTheta, cos(theta), sin(phi) * sineTheta);
        float solidAngle = sineTheta *
            (2.0 * kPi / float(push.width)) *
            (kPi / float(push.height));
        float3 radiance = gEnvironmentMap.Load(int3(texel, 0)).rgb;
        radiance = all(isfinite(radiance)) ? max(radiance, 0.0) : float3(0.0);
        weightedRadiance = float4(radiance * solidAngle, 0.0);
    }

    for (uint coefficientIndex = 0u; coefficientIndex < kCoefficientCount; ++coefficientIndex) {
        gShared[coefficientIndex * kThreadCount + groupThreadIndex] =
            weightedRadiance * shBasis(coefficientIndex, direction);
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
        gCoefficients[coefficientIndex] = gShared[0];
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
