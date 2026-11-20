module LumenScreenProbeFilter;

import ShaderCore;
import ParameterRoot;
import LumenAdaptiveProbes;
import LumenProbeMath;

using Metallic;
using Metallic.Lumen;

static const float kSpatialFilterMaxRadianceHitAngle = 10.0f * kLumenProbePi / 180.0f;
static const float kSpatialFilterPositionWeightScale = 1000.0f;

struct LumenScreenProbeFilterParams
{
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<RWStructuredBuffer<uint2>> probeOutput;
    ResourceHandle<StructuredBuffer<uint2>> probeInput;
    ResourceHandle<StructuredBuffer<float4>> inputRadiance;
    ResourceHandle<StructuredBuffer<float>> inputDistance;
    ResourceHandle<RWStructuredBuffer<float4>> outputRadiance;
    uint width;
    uint height;
    uint probeWidth;
    uint probeHeight;
    uint frameIndex;
};

[shader("compute")]
[numthreads(9, 1, 1)]
void lumenPlaceAdaptiveProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeFilterParams params = getParameters<LumenScreenProbeFilterParams>();
    Texture2D<float4> position = resolveUniform(params.position);
    Texture2D<float4> normal = resolveUniform(params.normal);
    RWStructuredBuffer<uint2> probeOutput = resolveUniform(params.probeOutput);
    uint index = lumenProbeIndex(group.xy, lane.x, params.probeWidth);
    probeOutput[index] = placeLumenAdaptiveProbe(group.xy, lane.x, uint2(params.width, params.height),
        uint2(params.probeWidth, params.probeHeight), position, normal, params.frameIndex);
}

// UE 5.7.4 FastMathThirdParty.ush:108.
float probeAcosFast(float inputValue)
{
    float x = abs(inputValue);
    float result = -0.156583f * x + (0.5f * kLumenProbePi);
    result *= sqrt(1.0f - x);
    return inputValue >= 0.0f ? result : kLumenProbePi - result;
}

float getFilterPositionWeight(float probeDepth, float sceneDepth)
{
    float depthDifference = abs(probeDepth - sceneDepth);
    float relativeDepthDifference = depthDifference / sceneDepth;
    return exp2(-kSpatialFilterPositionWeightScale * (relativeDepthDifference * relativeDepthDifference));
}

// UE 5.7.4 GatherNeighborRadiance, static geometry without adaptive neighbor search or extra AO.
void gatherNeighborRadiance(int2 neighborTile, uint2 probeSize, uint2 ray, float4 receiver, float3 coneDirection,
    float hitDistance, StructuredBuffer<uint2> probes, Texture2D<float4> position,
    StructuredBuffer<float4> inputRadiance, StructuredBuffer<float> inputDistance,
    inout float3 totalRadiance, inout float totalWeight)
{
    if (any(neighborTile < 0) || any(neighborTile >= int2(probeSize))) {
        return;
    }
    uint neighborIndex = lumenProbeIndex(uint2(neighborTile), 0u, probeSize.x);
    uint2 neighborPixel = probes[neighborIndex];
    if (neighborPixel.x == kLumenInvalidProbe) {
        return;
    }

    float4 neighbor = position.Load(int3(neighborPixel, 0));
    float positionWeight = getFilterPositionWeight(neighbor.w, receiver.w);
    if (positionWeight <= 0.0f) {
        return;
    }

    uint neighborAddress = lumenProbeRayAddress(neighborIndex, ray);
    float neighborRadianceDepth = inputDistance[neighborAddress];
    if (neighborRadianceDepth < 0.0f) {
        return;
    }
    if (hitDistance >= 0.0f) {
        neighborRadianceDepth = min(neighborRadianceDepth, hitDistance);
    }

    float3 neighborHitPosition = neighbor.xyz + coneDirection * neighborRadianceDepth;
    float3 toNeighborHit = neighborHitPosition - receiver.xyz;
    // Coincident endpoints have no direction. Constrain the cosine to the domain of sqrt in acosFast.
    float neighborHitLength = length(toNeighborHit);
    if (neighborHitLength == 0.0f) {
        return;
    }
    float neighborAngle = probeAcosFast(clamp(dot(toNeighborHit, coneDirection) / neighborHitLength, -1.0f, 1.0f));
    float angleWeight = 1.0f - saturate(neighborAngle / kSpatialFilterMaxRadianceHitAngle);
    float weight = positionWeight * angleWeight;
    totalRadiance += inputRadiance[neighborAddress].xyz * weight;
    totalWeight += weight;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenFilterScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeFilterParams params = getParameters<LumenScreenProbeFilterParams>();
    StructuredBuffer<uint2> probes = resolveUniform(params.probeInput);
    RWStructuredBuffer<float4> outputRadiance = resolveUniform(params.outputRadiance);
    uint probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
    uint address = lumenProbeRayAddress(probeIndex, lane.xy);
    uint2 probePixel = probes[probeIndex];
    if (probePixel.x == kLumenInvalidProbe) {
        outputRadiance[address] = 0.0f;
        return;
    }

    Texture2D<float4> position = resolveUniform(params.position);
    StructuredBuffer<float4> inputRadiance = resolveUniform(params.inputRadiance);
    StructuredBuffer<float> inputDistance = resolveUniform(params.inputDistance);
    float4 receiver = position.Load(int3(probePixel, 0));
    float3 coneDirection = lumenEquiAreaSphericalMapping((float2(lane.xy) + 0.5f) / float(kLumenProbeRayResolution));
    float hitDistance = inputDistance[address];
    float4 trace = inputRadiance[address];
    float3 totalRadiance = hitDistance >= 0.0f ? trace.xyz : float3(0.0f);
    float totalWeight = hitDistance >= 0.0f ? 1.0f : 0.0f;

    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    [loop]
    for (uint offsetIndex = 0u; offsetIndex < 4u; ++offsetIndex) {
        gatherNeighborRadiance(int2(group.xy) + offsets[offsetIndex], uint2(params.probeWidth, params.probeHeight),
            lane.xy, receiver, coneDirection, hitDistance, probes, position, inputRadiance, inputDistance,
            totalRadiance, totalWeight);
    }
    if (totalWeight > 0.0f) {
        totalRadiance /= totalWeight;
    }

    // Float32 storage preserves the radiance without the UE low-precision atlas's blue-noise rounding.
    outputRadiance[address] = float4(totalRadiance, trace.w);
}
