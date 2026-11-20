module LumenScreenProbeGather;

import ShaderCore;
import ParameterRoot;
import LumenGlobalRay;
import LumenProbeMath;

using Metallic;
using Metallic.Lumen;

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
};

// UE 5.7.4 GetUniformScreenProbeScreenPosition at Hammersley phase zero.
uint2 uniformProbePosition(uint2 tile, uint2 viewSize)
{
    return min(tile * kProbeDownsampleFactor, viewSize - 1u);
}

uint probeRayAddress(uint2 tile, uint2 ray, uint probeWidth)
{
    return (tile.y * probeWidth + tile.x) * kRaysPerProbe + ray.y * kProbeRayResolution + ray.x;
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
    uint2 probePixel = uniformProbePosition(group.xy, uint2(params.width, params.height));
    float4 receiver = position.Load(int3(probePixel, 0));
    float4 trace = 0.0f;
    if (receiver.w > 0.0f) {
        Texture2D<float4> normal = resolveUniform(params.normal);
        float3 rayDirection = probeRayDirection(lane.xy);
        // UE 5.7.4 TraceVoxels, with neither screen nor Mesh SDF trace continuation.
        float3 samplePosition = receiver.xyz + kSurfaceBias * rayDirection;
        samplePosition += kSurfaceBias * normalize(normal.Load(int3(probePixel, 0)).xyz);
        LumenGlobalTraceRay ray = {float4(samplePosition, kMinTraceDistance), float4(rayDirection, params.maxTraceDistance)};
        LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
        LumenGlobalTraceResult hit = traceLumenGlobalRay(scene, ray);
        trace = float4(compositeProbeTrace(hit.radiance.xyz), float(hit.state.x));
    }
    RWStructuredBuffer<float4> traceOutput = resolveUniform(params.traceOutput);
    traceOutput[probeRayAddress(group.xy, lane.xy, params.probeWidth)] = trace;
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
float3 integrateProbeDiffuse(ScreenProbeSample sample, float3 normal, uint probeWidth,
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
                    if (sample.weights[corner] > 0.0f) {
                        uint address = probeRayAddress(sample.atlasCoords[corner], ray, probeWidth);
                        interpolatedRadiance += radiance[address].xyz * sample.weights[corner];
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
    if (receiver.w > 0.0f) {
        Texture2D<float4> normalTexture = resolveUniform(params.normal);
        float3 normal = normalize(normalTexture.Load(int3(thread.xy, 0)).xyz);
        Texture2D<float4> diffuse = resolveUniform(params.diffuse);
        ScreenProbeSample sample = calculateProbeInterpolation(thread.xy, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position);
        if (any(sample.weights > 0.0f)) {
            StructuredBuffer<float4> radiance = resolveUniform(params.traceInput);
            float3 irradianceOverPi = integrateProbeDiffuse(sample, normal, params.probeWidth, radiance);
            indirect = irradianceOverPi * diffuse.Load(int3(thread.xy, 0)).rgb;
        }
    }
    indirectDiffuse[thread.xy] = float4(indirect, 1.0f);
    color[thread.xy] = float4(directColor.Load(int3(thread.xy, 0)).rgb + indirect, 1.0f);
}
