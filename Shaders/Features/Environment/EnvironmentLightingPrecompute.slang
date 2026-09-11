#include "../../Libraries/Resources/ComputeResources.slang"

// GPU projection and cosine convolution: irradiance SH through l=2.
#include "../../Libraries/Math/SphericalHarmonics.slang"
#include "../../Libraries/Lighting/ProceduralEnvironment.slang"
#include "../../Libraries/Lighting/EnvironmentPrefilter.slang"

static const uint kIntegratePartials = 0u;
static const uint kFinalizeCoefficients = 1u;
static const uint kThreadCount = 128u;
static const uint kCoefficientCount = SH::L2RGB::kCoefficientCount;
static const float kPi = 3.14159265358979323846;

#define gEnvironmentMap METALLIC_RESOURCE(Texture2D<float4>, 0)

#define gPartials METALLIC_RESOURCE(RWStructuredBuffer<float4>, 1)

#define gCoefficients METALLIC_RESOURCE(RWStructuredBuffer<float4>, 2)
#define gSpecular METALLIC_RESOURCE(RWStructuredBuffer<float4>, 3)

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

// The split-sum radiance convolution is independent of camera, exposure and
// environment rotation. Rebuild only when the source HDRI changes.
void prefilterSpecular(uint index, EnvironmentLightingPrecomputePush push)
{
    uint layerSize = kEnvironmentFilterWidth * kEnvironmentFilterHeight;
    if (index >= layerSize * kEnvironmentFilterLayers) { return; }
    uint layer = index / layerSize;
    uint pixel = index % layerSize;
    float3 normal = environmentFilterDirection((float2(pixel % kEnvironmentFilterWidth,
        pixel / kEnvironmentFilterWidth) + 0.5) / float2(kEnvironmentFilterWidth, kEnvironmentFilterHeight));
    float roughness = float(layer) / float(kEnvironmentFilterLayers - 1u);
    float alpha = max(roughness * roughness, 0.0001);
    float3 tangent = normalize(cross(abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0), normal));
    float3 bitangent = cross(normal, tangent);
    float3 radiance = float3(0.0);
    float weight = 0.0;
    const uint samples = 256u;
    for (uint sample = 0u; sample < samples; ++sample) {
        float2 xi = float2((float(sample) + 0.5) / float(samples), float(reversebits(sample)) * 2.32830643654e-10);
        float phi = 6.28318530718 * xi.x;
        float cosine = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
        float sine = sqrt(max(1.0 - cosine * cosine, 0.0));
        float3 halfVector = tangent * (cos(phi) * sine) + bitangent * (sin(phi) * sine) + normal * cosine;
        float3 direction = reflect(-normal, halfVector);
        float nl = saturate(dot(normal, direction));
        if (nl > 0.0) {
            float3 value = push.procedural != 0u ? proceduralEnvironmentLocalRadiance(direction)
                : environmentFilterSource(gEnvironmentMap, direction);
            radiance += (all(isfinite(value)) ? max(value, 0.0) : float3(0.0)) * nl;
            weight += nl;
        }
    }
    gSpecular[index] = float4(radiance / max(weight, 1e-6), 1.0);
}

[shader("compute")]
[numthreads(kThreadCount, 1, 1)]
void environmentLightingPrecomputeMain(
    uint3 groupId : SV_GroupID,
    uint groupThreadIndex : SV_GroupIndex)
{
    EnvironmentLightingPrecomputePush push = METALLIC_CONSTANTS(EnvironmentLightingPrecomputePush);
    if (push.mode == kIntegratePartials) {
        integratePartials(groupId, groupThreadIndex, push);
    } else if (push.mode == kFinalizeCoefficients) {
        finalizeCoefficient(groupId, groupThreadIndex, push);
    } else if (push.mode == 2u) {
        prefilterSpecular(groupId.x * kThreadCount + groupThreadIndex, push);
    }
}
