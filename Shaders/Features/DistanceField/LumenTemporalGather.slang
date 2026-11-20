module LumenTemporalGather;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;

using Metallic;
using Metallic.ScenePath;

typealias DiffuseTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;
typealias FullPrecisionTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;
typealias ScalarTexture = _Texture<float, __Shape2D, 0, 0, 0, 1, 0, 0, 6>;

struct LumenTemporalGatherParams
{
    ResourceHandle<Texture2D<float4>> indirectDiffuse;
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float4>> directColor;
    ResourceHandle<Texture2D<float4>> previousDiffuse;
    ResourceHandle<Texture2D<float4>> previousGeometry;
    ResourceHandle<DiffuseTexture> indirectOutput;
    ResourceHandle<FullPrecisionTexture> color;
    ResourceHandle<ScalarTexture> historyWeight;
    ResourceHandle<DiffuseTexture> currentDiffuse;
    ResourceHandle<FullPrecisionTexture> currentGeometry;
    uint historyValid;
    uint width;
    uint height;
    float2 previousJitter;
    ViewCameraConstants currentCamera;
    ViewCameraConstants previousCamera;
    ResourceHandle<Texture2D<float>> sceneDepth;
    ResourceHandle<ScalarTexture> screenHistoryDepth;
    ResourceHandle<FullPrecisionTexture> screenHistoryColor;
    uint publishScreenTraceHistory;
};

static const float kLumenHistoryDistanceThreshold = 0.01f;
static const float kLumenHistoryNormalCosThreshold = 0.7071067811865476f;
static const float kMaxFramesAccumulated = 10.0f;

// UE 5.7.4 LumenScreenProbeGatherTemporal.usf, GetBilinearCustomWeights.
float4 getBilinearCustomWeights(float2 fraction, float4 customWeights)
{
    float4 weights;
    weights.x = (1.0f - fraction.x) * (1.0f - fraction.y);
    weights.y = fraction.x * (1.0f - fraction.y);
    weights.z = (1.0f - fraction.x) * fraction.y;
    weights.w = fraction.x * fraction.y;
    return weights * customWeights;
}

float3 weightedAverage(float3 v00, float3 v10, float3 v01, float3 v11, float4 weights)
{
    float3 result = v00 * weights.x + v10 * weights.y + v01 * weights.z + v11 * weights.w;
    return result / max(dot(weights, 1.0f), 0.00001f);
}

float weightedAverage(float4 values, float4 weights)
{
    return dot(values, weights) / max(dot(weights, 1.0f), 0.00001f);
}

// StochasticLighting.cpp GetHistoryScreenCoordCodec and
// StochasticLightingCommon.ush Encode/DecodeHistoryScreenCoord.
float4 decodeHistoryFootprint(float2 historyScreenUV, uint2 viewSize)
{
    uint2 historyScreenCoordShift = uint2(firstbithigh(65535u / viewSize));
    float2 subPixelGridSize = float2(1u << historyScreenCoordShift);
    float2 invSubPixelGridSize = 1.0f / subPixelGridSize;
    float2 guardBandSize = 0.5f + max(invSubPixelGridSize, 1.0f / 512.0f);
    float2 historyGatherUVMin = min(guardBandSize / float2(viewSize), 0.5f);
    historyScreenUV = clamp(historyScreenUV, historyGatherUVMin, 1.0f - historyGatherUVMin);
    float2 historyScreenCoord = historyScreenUV * float2(viewSize) - 0.5f;
    float2 encodeCoord = clamp(historyScreenCoord * subPixelGridSize + 0.5f, 0.0f, 65535.0f);
    uint encoded = (uint(encodeCoord.y) << 16u) | uint(encodeCoord.x);
    uint2 decodedCoord = uint2(encoded & 0xFFFFu, encoded >> 16u);
    uint2 decodedScreenCoord = decodedCoord >> historyScreenCoordShift;
    uint2 subPixelCoord = decodedCoord - (decodedScreenCoord << historyScreenCoordShift);
    float2 bilinearWeights = float2(subPixelCoord) * invSubPixelGridSize;
    return float4(decodedScreenCoord, bilinearWeights);
}

// LumenScreenProbeGatherTemporal.usf Pack/UnpackFastUpdateModeAmountAndNumFramesAccumulated,
// specialized to the static-lighting branch where the high nibble is zero.
float packNumFramesAccumulated(float numFramesAccumulated)
{
    float localMaxFramesAccumulated = min(kMaxFramesAccumulated, 15.0f);
    float encodedNumFramesAccumulated = round(saturate(numFramesAccumulated / localMaxFramesAccumulated) * 15.0f);
    return encodedNumFramesAccumulated / 255.0f;
}

float4 unpackNumFramesAccumulated(float4 packed)
{
    float4 encodedNumFramesAccumulated = round(packed * 255.0f);
    return select(encodedNumFramesAccumulated == 15.0f, kMaxFramesAccumulated,
        encodedNumFramesAccumulated / 15.0f * min(kMaxFramesAccumulated, 15.0f));
}

// Reprojection for static opaque surfaces in
// UE 5.7.4 StochasticLightingTileClassification.usf, the linear-depth branch.
void accumulateDiffuse(LumenTemporalGatherParams params, float3 worldPosition, float3 normal, float4 newDiffuseLighting,
    out float3 outDiffuseIndirect, out float newNumFramesAccumulated, out float outHistoryWeight)
{
    outDiffuseIndirect = newDiffuseLighting.rgb;
    newNumFramesAccumulated = 0.0f;
    outHistoryWeight = 0.0f;
    ViewCameraConstants previousCamera = params.previousCamera;
    float3 previousForward = cameraForwardFrom(previousCamera.eye, previousCamera.center);
    float reprojectedSceneDepth = dot(worldPosition - previousCamera.eye.xyz, previousForward);
    float2 historyScreenUV;
    if (reprojectedSceneDepth < previousCamera.clipOrtho.x || reprojectedSceneDepth > previousCamera.clipOrtho.y
        || !projectWorldToUv(worldPosition, previousCamera.eye, previousCamera.center,
            previousCamera.upProjection, previousCamera.viewport, previousCamera.clipOrtho, historyScreenUV)) {
        return;
    }
    uint2 viewSize = uint2(params.width, params.height);
    // The previous primary ray used pixel center + previousJitter.
    historyScreenUV -= params.previousJitter / float2(viewSize);
    if (any(historyScreenUV < 0.0f) || any(historyScreenUV > 1.0f)) {
        return;
    }
    float4 historyFootprint = decodeHistoryFootprint(historyScreenUV, viewSize);
    int2 p00 = int2(historyFootprint.xy);
    int2 maxCoord = int2(viewSize) - 1;
    int2 p10 = min(p00 + int2(1, 0), maxCoord);
    int2 p01 = min(p00 + int2(0, 1), maxCoord);
    int2 p11 = min(p00 + int2(1, 1), maxCoord);

    Texture2D<float4> previousGeometry = resolveUniform(params.previousGeometry);
    float4 geometry00 = previousGeometry.Load(int3(p00, 0));
    float4 geometry10 = previousGeometry.Load(int3(p10, 0));
    float4 geometry01 = previousGeometry.Load(int3(p01, 0));
    float4 geometry11 = previousGeometry.Load(int3(p11, 0));
    float4 historySampleSceneDepth = float4(geometry00.w, geometry10.w, geometry01.w, geometry11.w);
    float4 distanceToHistoryValue = abs(historySampleSceneDepth - reprojectedSceneDepth);
    // UE BlueNoise.cpp GetBlueNoiseDummyParameters uses BlackDummy for rendering
    // commandlets. The sample uses that branch without importing Epic's texture.
    const float randomScalar = 0.0f;
    float lumenDisocclusionDistanceThreshold = kLumenHistoryDistanceThreshold;
    lumenDisocclusionDistanceThreshold *= lerp(0.5f, 1.5f, randomScalar);
    float3 v = normalize(params.currentCamera.eye.xyz - worldPosition);
    lumenDisocclusionDistanceThreshold /= clamp(saturate(dot(v, normal)), 0.1f, 1.0f);
    float4 lumenDepthWeights = select(distanceToHistoryValue >= reprojectedSceneDepth * lumenDisocclusionDistanceThreshold,
        0.0f, 1.0f);
    float4 historyNormalWeights = select(float4(dot(geometry00.xyz, normal), dot(geometry10.xyz, normal),
        dot(geometry01.xyz, normal), dot(geometry11.xyz, normal)) < kLumenHistoryNormalCosThreshold, 0.0f, 1.0f);
    float2 historyFraction = historyFootprint.zw;
    float4 lumenHistoryWeights = getBilinearCustomWeights(historyFraction, lumenDepthWeights);
    lumenHistoryWeights *= historyNormalWeights;
    float4 visibilityWeights = select(lumenHistoryWeights > 0.01f, 1.0f, 0.0f);
    if (!any(visibilityWeights > 0.0f)) {
        return;
    }

    // LumenScreenProbeGatherTemporal.usf full-resolution diffuse history.
    float4 finalWeights = getBilinearCustomWeights(historyFraction, visibilityWeights);
    Texture2D<float4> previousDiffuse = resolveUniform(params.previousDiffuse);
    float4 history00 = previousDiffuse.Load(int3(p00, 0));
    float4 history10 = previousDiffuse.Load(int3(p10, 0));
    float4 history01 = previousDiffuse.Load(int3(p01, 0));
    float4 history11 = previousDiffuse.Load(int3(p11, 0));
    float3 historyDiffuseIndirect = weightedAverage(history00.rgb, history10.rgb, history01.rgb, history11.rgb, finalWeights);
    float4 historyNumFramesAccumulated = unpackNumFramesAccumulated(float4(history00.w, history10.w, history01.w, history11.w));
    float numFramesAccumulated = min(weightedAverage(historyNumFramesAccumulated + 1.0f, finalWeights), kMaxFramesAccumulated);
    newNumFramesAccumulated = numFramesAccumulated;
    float alpha = 1.0f / (1.0f + newNumFramesAccumulated);
    if (newDiffuseLighting.w <= 0.0f && newNumFramesAccumulated >= 1.0f) {
        alpha = 1.0f / (1.0f + 4.0f * newNumFramesAccumulated);
    }
    outDiffuseIndirect = lerp(historyDiffuseIndirect, newDiffuseLighting.rgb, alpha);
    outHistoryWeight = 1.0f - alpha;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenTemporalGatherMain(uint3 thread : SV_DispatchThreadID)
{
    LumenTemporalGatherParams params = getParameters<LumenTemporalGatherParams>();
    if (any(thread.xy >= uint2(params.width, params.height))) {
        return;
    }
    int3 texel = int3(thread.xy, 0);
    Texture2D<float4> position = resolveUniform(params.position);
    Texture2D<float4> normal = resolveUniform(params.normal);
    Texture2D<float4> indirectDiffuse = resolveUniform(params.indirectDiffuse);
    Texture2D<float4> directColor = resolveUniform(params.directColor);
    float4 surface = position.Load(texel);
    float3 surfaceNormal = surface.w > 0.0f ? normalize(normal.Load(texel).xyz) : 0.0f;
    float4 newDiffuseLighting = indirectDiffuse.Load(texel);
    float3 outDiffuseIndirect = newDiffuseLighting.rgb;
    float newNumFramesAccumulated = 0.0f;
    float outHistoryWeight = 0.0f;
    if (params.historyValid != 0u && surface.w > 0.0f) {
        accumulateDiffuse(params, surface.xyz, surfaceNormal, newDiffuseLighting, outDiffuseIndirect,
            newNumFramesAccumulated, outHistoryWeight);
    }

    DiffuseTexture indirectOutput = resolveUniform(params.indirectOutput);
    FullPrecisionTexture color = resolveUniform(params.color);
    ScalarTexture historyWeight = resolveUniform(params.historyWeight);
    DiffuseTexture currentDiffuse = resolveUniform(params.currentDiffuse);
    FullPrecisionTexture currentGeometry = resolveUniform(params.currentGeometry);
    indirectOutput[thread.xy] = float4(outDiffuseIndirect, newDiffuseLighting.w);
    float4 scene = directColor.Load(texel);
    float4 sceneColor = float4(scene.rgb + outDiffuseIndirect, scene.a);
    color[thread.xy] = sceneColor;
    // UE 5.7.4 ScreenSpaceRayTracingInput extracts resolved color and depth before postprocessing.
    if (params.publishScreenTraceHistory != 0u) {
        resolveUniform(params.screenHistoryDepth)[thread.xy] = resolveUniform(params.sceneDepth).Load(texel);
        resolveUniform(params.screenHistoryColor)[thread.xy] = sceneColor;
    }
    historyWeight[thread.xy] = outHistoryWeight;
    currentDiffuse[thread.xy] = float4(outDiffuseIndirect, packNumFramesAccumulated(newNumFramesAccumulated));
    currentGeometry[thread.xy] = float4(surfaceNormal, surface.w);
}
