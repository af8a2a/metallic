module LumenScreenProbeGather;

import ShaderCore;
import ParameterRoot;
import LumenGlobalRay;
import LumenProbeMath;
import GlobalDistanceField;
#if defined(METALLIC_LUMEN_FINAL_GATHER)
import LumenAdaptiveProbes;
#endif
#if defined(METALLIC_LUMEN_SCREEN_TRACES)
import LumenScreenTrace;
#endif

using Metallic;
using Metallic.Lumen;
using Metallic.DistanceField;

static const uint kProbeDownsampleFactor = 16u;
static const uint kProbeRayResolution = 8u;
static const uint kRaysPerProbe = kProbeRayResolution * kProbeRayResolution;
static const float kSurfaceBias = 0.05f;
static const float kMinTraceDistance = 0.0001f;
static const float kMaxRayIntensity = 10.0f;
static const float kInterpolationDepthWeight = -200.0f;
static const float kMinInterpolationFallbackWeight = 0.0001f;

typealias LumenSceneColorTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;
typealias LumenCompositeTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;

struct LumenScreenProbeGatherParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float4>> diffuse;
    ResourceHandle<Texture2D<float4>> directColor;
    ResourceHandle<RWStructuredBuffer<float4>> traceOutput;
    ResourceHandle<StructuredBuffer<float4>> traceInput;
    ResourceHandle<LumenSceneColorTexture> indirectDiffuse;
    ResourceHandle<LumenCompositeTexture> color;
    uint width;
    uint height;
    uint probeWidth;
    uint probeHeight;
    uint clipmapCount;
    float maxTraceDistance;
    ResourceHandle<StructuredBuffer<uint2>> probePositions;
    ResourceHandle<RWStructuredBuffer<float>> traceDistance;
    ResourceHandle<StructuredBuffer<float>> closestHZB;
    ResourceHandle<Texture2D<float>> previousDepth;
    ResourceHandle<Texture2D<float4>> previousColor;
#if defined(METALLIC_LUMEN_SCREEN_TRACES)
    ResourceHandle<StructuredBuffer<LumenScreenTraceView>> traceView;
#else
    uint traceView;
#endif
    uint frameIndex;
};

// UE 5.7.4 GetUniformScreenProbeScreenPosition at Hammersley phase zero.
uint2 uniformProbePosition(uint2 tile, uint2 viewSize)
{
    return min(tile * kProbeDownsampleFactor, viewSize - 1u);
}

uint probeRayAddress(uint probeIndex, uint2 ray)
{
    return probeIndex * kRaysPerProbe + ray.y * kProbeRayResolution + ray.x;
}

float3 probeRayDirection(uint2 ray)
{
    // GetProbeTracingUV with one sample and the fixed center branch of GetProbeTexelCenter.
    return lumenEquiAreaSphericalMapping((float2(ray) + 0.5f) / float(kProbeRayResolution));
}

// UE 5.7.4 LumenScreenProbeFiltering.usf:77. Equal trace/gather resolutions and a centered ray
// send each sample to its own texel, so the scatter has one writer per destination.
float3 compositeProbeTrace(float3 lighting)
{
    float maxLighting = max(lighting.x, max(lighting.y, lighting.z));
    if (maxLighting > kMaxRayIntensity) {
        lighting *= kMaxRayIntensity / maxLighting;
    }
    const float maxValuePerThread = float(0xffffffffu) / float(kRaysPerProbe);
    const float lightingQuantizeScale = maxValuePerThread / kMaxRayIntensity;
    uint3 quantizedLighting = uint3(lighting * lightingQuantizeScale);
    const float invMaxValuePerThread = float(kRaysPerProbe) / float(0xffffffffu);
    const float invLightingQuantizeScale = kMaxRayIntensity * invMaxValuePerThread;
    return float3(quantizedLighting) * invLightingQuantizeScale;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenTraceScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeGatherParams params = getParameters<LumenScreenProbeGatherParams>();
    Texture2D<float4> position = resolveUniform(params.position);
    RWStructuredBuffer<float4> traceOutput = resolveUniform(params.traceOutput);
    uint probeIndex = group.y * params.probeWidth + group.x;
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
    StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
    RWStructuredBuffer<float> distances = resolveUniform(params.traceDistance);
    uint2 probePixel = probes[probeIndex];
    uint address = probeRayAddress(probeIndex, lane.xy);
    if (probePixel.x == kLumenInvalidProbe) {
        traceOutput[address] = 0.0f;
        distances[address] = -1.0f;
        return;
    }
#else
    uint2 probePixel = uniformProbePosition(group.xy, uint2(params.width, params.height));
    uint address = probeRayAddress(probeIndex, lane.xy);
#endif
    float4 receiver = position.Load(int3(probePixel, 0));
    float4 trace = 0.0f;
    if (receiver.w > 0.0f) {
        Texture2D<float4> normal = resolveUniform(params.normal);
        float3 rayDirection = probeRayDirection(lane.xy);
        float3 receiverNormal = normalize(normal.Load(int3(probePixel, 0)).xyz);
        // UE 5.7.4 TraceVoxels, with neither screen nor Mesh SDF trace continuation.
        float3 samplePosition = receiver.xyz + kSurfaceBias * rayDirection;
        samplePosition += kSurfaceBias * receiverNormal;
#if defined(METALLIC_LUMEN_SCREEN_TRACES)
        LumenScreenTraceInputs screenInputs = {resolveUniform(params.closestHZB), resolveUniform(params.previousDepth),
            resolveUniform(params.previousColor), resolveUniform(params.traceView)};
        float3 screenRadiance;
        float screenDistance;
        uint2 traceBufferCoord = uint2(group.x * kLumenProbeSlotsPerTile + group.z, group.y) * kProbeRayResolution + lane.xy;
        if (traceLumenScreenRay(screenInputs, receiver.xyz, receiverNormal, rayDirection, params.maxTraceDistance, traceBufferCoord,
                screenRadiance, screenDistance)) {
            traceOutput[address] = float4(compositeProbeTrace(screenRadiance), 5.0f);
            distances[address] = screenDistance;
            return;
        }
#endif
        LumenGlobalTraceRay ray = {float4(samplePosition, kMinTraceDistance), float4(rayDirection, params.maxTraceDistance)};
        LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
        LumenGlobalTraceResult hit = traceLumenGlobalRay(scene, ray);
        trace = float4(compositeProbeTrace(hit.radiance.xyz), float(hit.state.x));
#if defined(METALLIC_LUMEN_FINAL_GATHER)
        distances[address] = hit.state.x == kGlobalDistanceFieldHit ? hit.metrics.x :
            (hit.state.x == kGlobalDistanceFieldUnavailable ? -1.0f : params.maxTraceDistance);
#endif
    } else {
#if defined(METALLIC_LUMEN_FINAL_GATHER)
        distances[address] = -1.0f;
#endif
    }
    traceOutput[address] = trace;
}

struct ScreenProbeSample
{
    uint2 atlasCoords[4];
    float4 weights;
};

// UE 5.7.4 CalculateUniformUpsampleInterpolationWeights, opaque plane weighting.
ScreenProbeSample calculateProbeInterpolation(uint2 pixel, uint2 viewSize, uint2 probeSize, float4 receiver,
    float3 normal, Texture2D<float4> position)
{
    uint2 screenTileCoord00 = min(pixel / kProbeDownsampleFactor, max(probeSize, 2u) - 2u);
    const uint bilinearExpand = 1u;
    float2 bilinear = float2(pixel - screenTileCoord00 * kProbeDownsampleFactor + bilinearExpand) /
        float(kProbeDownsampleFactor + 2u * bilinearExpand);
    ScreenProbeSample sample;
    sample.weights = float4((1.0f - bilinear.y) * (1.0f - bilinear.x), (1.0f - bilinear.y) * bilinear.x,
        bilinear.y * (1.0f - bilinear.x), bilinear.y * bilinear.x);

    float4 scenePlane = float4(normal, dot(receiver.xyz, normal));
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
        uint2 tile = min(screenTileCoord00 + uint2(corner % 2u, corner / 2u), probeSize - 1u);
        sample.atlasCoords[corner] = tile;
        float4 probe = position.Load(int3(uniformProbePosition(tile, viewSize), 0));
        float planeDistance = abs(dot(float4(probe.xyz, -1.0f), scenePlane));
        float relativeDepthDifference = abs(planeDistance / receiver.w);
        sample.weights[corner] *= probe.w > 0.0f ? exp2(kInterpolationDepthWeight * relativeDepthDifference) : 0.0f;
    }

    float weightSum = dot(sample.weights, 1.0f);
    if (weightSum >= kMinInterpolationFallbackWeight) {
        sample.weights /= max(weightSum, kMinInterpolationFallbackWeight);
    } else {
        sample.weights = 0.0f;
    }
    return sample;
}

// UE 5.7.4 DIFFUSE_INTEGRATION_NUMERICAL_INTEGRAL, opaque receivers.
float3 integrateProbeDiffuse(uint4 probeIndices, float4 weights, float3 normal,
    StructuredBuffer<float4> radiance)
{
    int numValidSamples = 0;
    float3 diffuseLighting = 0.0f;
    for (uint y = 0u; y < kProbeRayResolution; ++y) {
        for (uint x = 0u; x < kProbeRayResolution; ++x) {
            uint2 ray = uint2(x, y);
            float nDotL = dot(probeRayDirection(ray), normal);
            if (nDotL > 0.0f) {
                float sampleWeight = saturate(nDotL);
                float3 interpolatedRadiance = 0.0f;
                [unroll]
                for (uint corner = 0u; corner < 4u; ++corner) {
                    if (weights[corner] > 0.0f) {
                        uint address = probeRayAddress(probeIndices[corner], ray);
                        interpolatedRadiance += radiance[address].xyz * weights[corner];
                    }
                }
                diffuseLighting += interpolatedRadiance * sampleWeight;
                ++numValidSamples;
            }
        }
    }
    if (numValidSamples > 0) {
        float normalizeTerm = 2.0f * kLumenProbePi / float(numValidSamples);
        diffuseLighting *= normalizeTerm;
    }
    return diffuseLighting * (1.0f / kLumenProbePi);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenIntegrateScreenProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenScreenProbeGatherParams params = getParameters<LumenScreenProbeGatherParams>();
    uint2 viewSize = uint2(params.width, params.height);
    if (any(thread.xy >= viewSize)) {
        return;
    }
    Texture2D<float4> position = resolveUniform(params.position);
    Texture2D<float4> directColor = resolveUniform(params.directColor);
    LumenSceneColorTexture indirectDiffuse = resolveUniform(params.indirectDiffuse);
    LumenCompositeTexture color = resolveUniform(params.color);
    float4 receiver = position.Load(int3(thread.xy, 0));
    float3 indirect = 0.0f;
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    float encodedAlpha = 0.0f;
#else
    const float encodedAlpha = 1.0f;
#endif
    if (receiver.w > 0.0f) {
        Texture2D<float4> normalTexture = resolveUniform(params.normal);
        float3 normal = normalize(normalTexture.Load(int3(thread.xy, 0)).xyz);
        Texture2D<float4> diffuse = resolveUniform(params.diffuse);
#if defined(METALLIC_LUMEN_FINAL_GATHER)
        StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
        LumenProbeSample sample = calculateLumenAdaptiveInterpolation(thread.xy, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position, probes, params.frameIndex);
        uint4 probeIndices = sample.indices;
        // UE 5.7.4 InterpolateAndIntegrate, static lighting with zero moving fraction.
        encodedAlpha = sample.lightingValid ? 0.004f : 0.0f;
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
        // UE 5.7.4 STOCHASTIC_PROBE_INTERPOLATION and the rendering-commandlet BlackDummy.
        bool sampleProbes = any(sample.weights > 0.0f);
        float randomValue = min(0.0f, 0.99f);
        float weightSum = dot(sample.weights, 1.0f);
        randomValue *= weightSum;
        uint pickedProbe;
        if (randomValue >= sample.weights[0] + sample.weights[1] + sample.weights[2]) {
            pickedProbe = sample.indices[3];
        } else if (randomValue >= sample.weights[0] + sample.weights[1]) {
            pickedProbe = sample.indices[2];
        } else if (randomValue >= sample.weights[0]) {
            pickedProbe = sample.indices[1];
        } else {
            pickedProbe = sample.indices[0];
        }
        probeIndices = uint4(pickedProbe);
        sample.weights = float4(sampleProbes ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
#endif
#else
        ScreenProbeSample sample = calculateProbeInterpolation(thread.xy, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position);
        uint4 probeIndices;
        [unroll]
        for (uint corner = 0u; corner < 4u; ++corner) {
            probeIndices[corner] = sample.atlasCoords[corner].y * params.probeWidth + sample.atlasCoords[corner].x;
        }
#endif
        if (any(sample.weights > 0.0f)) {
            StructuredBuffer<float4> radiance = resolveUniform(params.traceInput);
            float3 irradianceOverPi = integrateProbeDiffuse(probeIndices, sample.weights, normal, radiance);
            indirect = irradianceOverPi * diffuse.Load(int3(thread.xy, 0)).rgb;
        }
    }
    indirectDiffuse[thread.xy] = float4(indirect, encodedAlpha);
    color[thread.xy] = float4(directColor.Load(int3(thread.xy, 0)).rgb + indirect, 1.0f);
}
