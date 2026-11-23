module LumenScreenProbeFilter;

import ShaderCore;
import ParameterRoot;
import LumenAdaptiveProbes;
import LumenProbeMath;
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
import LumenScreenProbeWork;
#endif

using Metallic;
using Metallic.Lumen;

static const float kSpatialFilterMaxRadianceHitAngle = 10.0f * kLumenProbePi / 180.0f;
static const float kSpatialFilterPositionWeightScale = 1000.0f;

// Slang image format r11f_g11f_b10f, matching the UE default lighting data format.
typealias ProbeIrradianceImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 5>;

typealias SHDirectionalImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

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
    ResourceHandle<Texture2D<float2>> blueNoiseVec2;
    ResourceHandle<RWStructuredBuffer<float>> outputDistance;
    ResourceHandle<ProbeIrradianceImage> irradianceOutput;
    ResourceHandle<SHDirectionalImage> shDirectionalOutput;
    uint rayDirectionFrameIndex;
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    ResourceHandle<StructuredBuffer<uint>> probeWorkList;
    ResourceHandle<StructuredBuffer<uint>> probeWorkState;
#endif
};

#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
bool resolveFilterProbeIndex(LumenScreenProbeFilterParams params, uint3 group, out uint probeIndex)
{
    return resolveLumenActiveProbe(group, resolveUniform(params.probeWorkList),
        resolveUniform(params.probeWorkState), probeIndex);
}
#endif

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

#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
static const uint kRaysPerProbe = kLumenProbeGatherResolution * kLumenProbeGatherResolution;
static const uint kTraceSamples = kLumenProbeTraceResolution * kLumenProbeTraceResolution;
groupshared uint sharedLighting[kRaysPerProbe][3];
groupshared uint sharedMinHitDistance[kRaysPerProbe];

// UE 5.7.4 ScreenProbeCompositeTracesWithScatterCS, uniform trace directions scattered to the gather grid.
[shader("compute")]
[numthreads(kLumenProbeTraceResolution, kLumenProbeTraceResolution, 1)]
void lumenCompositeScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeFilterParams params = getParameters<LumenScreenProbeFilterParams>();
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    uint probeIndex;
    if (!resolveFilterProbeIndex(params, group, probeIndex)) { return; }
#else
    uint probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
#endif
    uint traceAddress = lumenProbeTraceAddress(probeIndex, lane.xy);
    bool writeGather = all(lane.xy < kLumenProbeGatherResolution);
    uint gatherThread = lane.y * kLumenProbeGatherResolution + lane.x;
    uint address = lumenProbeGatherAddress(probeIndex, lane.xy);
    RWStructuredBuffer<float4> radiance = resolveUniform(params.outputRadiance);
    RWStructuredBuffer<float> distance = resolveUniform(params.outputDistance);
    uint2 probePixel = resolveUniform(params.probeInput)[probeIndex];
#if !defined(METALLIC_LUMEN_ACTIVE_PROBES)
    if (probePixel.x == kLumenInvalidProbe) {
        if (writeGather) {
            radiance[address] = 0.0f;
            distance[address] = -1.0f;
        }
        return;
    }
#endif

    // Metallic stores unencoded meters; +infinity distinguishes a gather bin with no valid ray.
    const uint emptyDistance = 0x7f800000u;
    if (writeGather) {
        sharedLighting[gatherThread][0] = 0u;
        sharedLighting[gatherThread][1] = 0u;
        sharedLighting[gatherThread][2] = 0u;
        sharedMinHitDistance[gatherThread] = emptyDistance;
    }
    GroupMemoryBarrierWithGroupSync();

    float hitDistance = resolveUniform(params.inputDistance)[traceAddress];
    if (hitDistance >= 0.0f) {
        Texture2D<float2> noise = resolveUniform(params.blueNoiseVec2);
        uint2 screenTileCoord = lumenScreenTileCoord(probePixel, params.frameIndex);
        uint rayDirectionFrameIndex = params.rayDirectionFrameIndex;
        float2 probeTexelCenter = lumenProbeBlueNoiseVec2(noise, screenTileCoord, rayDirectionFrameIndex);
        float2 gatherTexelCoord = (float2(lane.xy) + probeTexelCenter) * kLumenProbeGatherResolution /
            kLumenProbeTraceResolution - 0.5f;
        float2 random = lumenProbeBlueNoiseVec2(noise, screenTileCoord, rayDirectionFrameIndex + 63u / 2u);
        int2 quantizedCoord = int2(floor(gatherTexelCoord)) +
            select(frac(gatherTexelCoord) < random, int2(0), int2(1));
        uint2 gatherCoord = wrapOctahedralBorder(uint2(quantizedCoord + 1), kLumenProbeGatherResolution + 2u, 1u);
        uint dest = gatherCoord.y * kLumenProbeGatherResolution + gatherCoord.x;
        float sampleWeight = float(kLumenProbeGatherResolution) / kLumenProbeTraceResolution
            * kLumenProbeGatherResolution / kLumenProbeTraceResolution;
        uint3 lighting = lumenQuantizeProbeLighting(resolveUniform(params.inputRadiance)[traceAddress].xyz * sampleWeight, kTraceSamples);
        InterlockedAdd(sharedLighting[dest][0], lighting.x);
        InterlockedAdd(sharedLighting[dest][1], lighting.y);
        InterlockedAdd(sharedLighting[dest][2], lighting.z);
        InterlockedMin(sharedMinHitDistance[dest], asuint(hitDistance));
    }
    GroupMemoryBarrierWithGroupSync();

    if (!writeGather) { return; }
    uint3 sum = uint3(sharedLighting[gatherThread][0], sharedLighting[gatherThread][1], sharedLighting[gatherThread][2]);
    radiance[address] = float4(lumenDecodeProbeLighting(sum, kTraceSamples), 0.0f);
    uint minDistance = sharedMinHitDistance[gatherThread];
    distance[address] = minDistance == emptyDistance ? -1.0f : asfloat(minDistance);
}
#endif

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

    uint neighborAddress = lumenProbeGatherAddress(neighborIndex, ray);
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
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    uint probeIndex;
    if (!resolveFilterProbeIndex(params, group, probeIndex)) { return; }
#else
    uint probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
#endif
    uint address = lumenProbeGatherAddress(probeIndex, lane.xy);
    uint2 probePixel = probes[probeIndex];
#if !defined(METALLIC_LUMEN_ACTIVE_PROBES)
    if (probePixel.x == kLumenInvalidProbe) {
        outputRadiance[address] = 0.0f;
        return;
    }
#endif

    Texture2D<float4> position = resolveUniform(params.position);
    StructuredBuffer<float4> inputRadiance = resolveUniform(params.inputRadiance);
    StructuredBuffer<float> inputDistance = resolveUniform(params.inputDistance);
    float4 receiver = position.Load(int3(probePixel, 0));
#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
    uint2 screenTileCoord = lumenScreenTileCoord(probePixel, params.frameIndex);
    float2 probeTexelCenter = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoiseVec2), screenTileCoord,
        params.rayDirectionFrameIndex);
#else
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    uint2 screenTileCoord = lumenProbeTileForIndex(probeIndex, params.probeWidth);
#else
    uint2 screenTileCoord = group.xy;
#endif
    float2 probeTexelCenter = 0.5f;
#endif
    float3 coneDirection = lumenEquiAreaSphericalMapping((float2(lane.xy) + probeTexelCenter) / float(kLumenProbeGatherResolution));
    float hitDistance = inputDistance[address];
    float4 trace = inputRadiance[address];
    float3 totalRadiance = hitDistance >= 0.0f ? trace.xyz : float3(0.0f);
    float totalWeight = hitDistance >= 0.0f ? 1.0f : 0.0f;

    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    [loop]
    for (uint offsetIndex = 0u; offsetIndex < 4u; ++offsetIndex) {
        gatherNeighborRadiance(int2(screenTileCoord) + offsets[offsetIndex], uint2(params.probeWidth, params.probeHeight),
            lane.xy, receiver, coneDirection, hitDistance, probes, position, inputRadiance, inputDistance,
            totalRadiance, totalWeight);
    }
    if (totalWeight > 0.0f) {
        totalRadiance /= totalWeight;
    }

    // Float32 storage preserves the radiance without the UE low-precision atlas's blue-noise rounding.
    outputRadiance[address] = float4(totalRadiance, trace.w);
}

#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
static const uint kNumLightingSHCoefficients = 9u;
groupshared float sharedSHBasisFunctions[kRaysPerProbe][kNumLightingSHCoefficients];
groupshared float4 sharedLightingRadiance[kRaysPerProbe];
groupshared float sharedSums[kNumLightingSHCoefficients * 4u];

// UE 5.7.4 ScreenProbeConvertToIrradianceCS, shared-memory SH reduction.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenConvertScreenProbesToIrradianceMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID,
    uint threadIndex : SV_GroupIndex)
{
    LumenScreenProbeFilterParams params = getParameters<LumenScreenProbeFilterParams>();
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    uint probeIndex;
    if (!resolveFilterProbeIndex(params, group, probeIndex)) { return; }
#else
    uint probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
#endif
    uint2 probeCoord = lumenProbeAtlasCoord(probeIndex, params.probeWidth);
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
    uint atlasWidth = params.probeWidth * kLumenProbeSlotsPerTileAxis;
    SHDirectionalImage directionalOutput = resolveUniform(params.shDirectionalOutput);
#else
    uint2 outputCoord = probeCoord * kLumenIrradianceProbeWithBorderResolution + lane.xy;
#endif
    ProbeIrradianceImage irradianceOutput = resolveUniform(params.irradianceOutput);
#if !defined(METALLIC_LUMEN_ACTIVE_PROBES)
    if (resolveUniform(params.probeInput)[probeIndex].x == kLumenInvalidProbe) {
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
        if (threadIndex == 0u) { irradianceOutput[probeCoord] = 0.0f; }
        if (threadIndex < 6u) { directionalOutput[probeCoord + uint2(threadIndex * atlasWidth, 0u)] = 0.0f; }
#else
        irradianceOutput[outputCoord] = 0.0f;
#endif
        return;
    }
#endif

    float2 probeUV = (float2(lane.xy) + 0.5f) / float(kLumenProbeGatherResolution);
    ThreeBandSHVector basis = shBasisFunction3(lumenEquiAreaSphericalMapping(probeUV));
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        sharedSHBasisFunctions[threadIndex][i] = basis.v0[i];
        sharedSHBasisFunctions[threadIndex][i + 4u] = basis.v1[i];
    }
    sharedSHBasisFunctions[threadIndex][8] = basis.v2;
    uint address = lumenProbeGatherAddress(probeIndex, lane.xy);
    sharedLightingRadiance[threadIndex] = float4(resolveUniform(params.inputRadiance)[address].xyz, 1.0f);
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex < kNumLightingSHCoefficients * 4u) {
        uint shCoefficientIndex = threadIndex % kNumLightingSHCoefficients;
        uint radianceCoefficientIndex = threadIndex % 4u;
        float sum = 0.0f;
        for (uint i = 0u; i < kRaysPerProbe; ++i) {
            float coefficient = radianceCoefficientIndex == 3u ? 1.0f : sharedSHBasisFunctions[i][shCoefficientIndex];
            sum += coefficient * sharedLightingRadiance[i][radianceCoefficientIndex];
        }
        sharedSums[radianceCoefficientIndex * kNumLightingSHCoefficients + shCoefficientIndex] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    float normalizeWeight = 1.0f / sharedSums[3u * kNumLightingSHCoefficients];
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
    if (threadIndex == 0u) {
        float3 ambientRGB = float3(sharedSums[0], sharedSums[kNumLightingSHCoefficients],
            sharedSums[2u * kNumLightingSHCoefficients]) * normalizeWeight;
        irradianceOutput[probeCoord] = float4(ambientRGB, 0.0f);
    }
    if (threadIndex < 3u) {
        uint i = threadIndex * kNumLightingSHCoefficients;
        float4 directionalSH0 = float4(sharedSums[i + 1u], sharedSums[i + 2u], sharedSums[i + 3u], sharedSums[i + 4u]) * normalizeWeight;
        float4 directionalSH1 = float4(sharedSums[i + 5u], sharedSums[i + 6u], sharedSums[i + 7u], sharedSums[i + 8u]) * normalizeWeight;
        directionalOutput[probeCoord + uint2((threadIndex * 2u) * atlasWidth, 0u)] = directionalSH0;
        directionalOutput[probeCoord + uint2((threadIndex * 2u + 1u) * atlasWidth, 0u)] = directionalSH1;
    }
#else
    uint2 wrapped = wrapOctahedralBorder(lane.xy, kLumenIrradianceProbeWithBorderResolution, 1u);
    float2 irradianceProbeUV = (float2(wrapped) + 0.5f) / float(kLumenIrradianceProbeResolution);
    ThreeBandSHVector diffuseTransferSH = calcDiffuseTransferSH3(lumenEquiAreaSphericalMapping(irradianceProbeUV), 1.0f);
    float3 dotSH;
    [unroll]
    for (uint channel = 0u; channel < 3u; ++channel) {
        uint i = channel * kNumLightingSHCoefficients;
        float4 v0 = float4(sharedSums[i], sharedSums[i + 1u], sharedSums[i + 2u], sharedSums[i + 3u]) * normalizeWeight;
        float4 v1 = float4(sharedSums[i + 4u], sharedSums[i + 5u], sharedSums[i + 6u], sharedSums[i + 7u]) * normalizeWeight;
        float v2 = sharedSums[i + 8u] * normalizeWeight;
        dotSH[channel] = dot(v0, diffuseTransferSH.v0);
        dotSH[channel] += dot(v1, diffuseTransferSH.v1);
        dotSH[channel] += v2 * diffuseTransferSH.v2;
    }
    irradianceOutput[outputCoord] = float4(4.0f * kLumenProbePi * dotSH, 0.0f);
#endif
}
#endif
