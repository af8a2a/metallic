module LumenTemporalGather;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
#if defined(METALLIC_LUMEN_DITHER_HISTORY) || defined(METALLIC_LUMEN_SHORT_RANGE_AO)
import LumenProbeMath;
#endif
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
import LumenShortRangeAOPacking;
import LumenShortRangeAOComposite;
#endif

using Metallic;
using Metallic.ScenePath;
#if defined(METALLIC_LUMEN_DITHER_HISTORY) || defined(METALLIC_LUMEN_SHORT_RANGE_AO)
using Metallic.Lumen;
#endif

typealias DiffuseTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;
typealias FullPrecisionTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;
typealias ScalarTexture = _Texture<float, __Shape2D, 0, 0, 0, 1, 0, 0, 6>;
typealias PackedShortRangeAOTexture = _Texture<uint, __Shape2D, 0, 0, 0, 1, 0, 0, 37>; // R32Uint

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
    float maxFramesAccumulated;
    ResourceHandle<Texture2D<float>> blueNoiseScalar;
    uint frameIndex;
    ResourceHandle<Texture2D<float4>> previousPosition;
    ResourceHandle<Texture2D<uint>> shortRangeAORaw;
    ResourceHandle<Texture2D<uint>> previousShortRangeAO;
    ResourceHandle<PackedShortRangeAOTexture> shortRangeAO;
    ResourceHandle<PackedShortRangeAOTexture> currentShortRangeAO;
    ResourceHandle<Texture2D<float4>> diffuseAlbedo;
    float2 currentJitter;
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    ResourceHandle<ScalarTexture> fastUpdateAmount;
#else
    uint shortRangeAOPadding;
#endif
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    ResourceHandle<Texture2D<float4>> roughSpecular;
    ResourceHandle<Texture2D<float4>> previousRoughSpecular;
    ResourceHandle<DiffuseTexture> roughSpecularOutput;
    ResourceHandle<DiffuseTexture> currentRoughSpecular;
#endif
};

static const float kLumenHistoryDistanceThreshold = 0.01f;
static const float kLumenHistoryNormalCosThreshold = 0.7071067811865476f;

#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
static const uint kShortRangeAOSharedTileSize = 6u;
groupshared half4 sharedShortRangeAO[kShortRangeAOSharedTileSize][kShortRangeAOSharedTileSize];

// UE 5.7.4 LumenScreenProbeGatherTemporal.usf. Denoise cone direction and
// openness separately. The packed history and raw AO share the RGB111110 layout.
uint packSharedShortRangeAO(half4 bentNormalAndAO)
{
    half3 bentNormal = normalize(bentNormalAndAO.xyz) * bentNormalAndAO.w;
    bentNormal = saturate(bentNormal * 0.5 + 0.5);
    uint packed = 0u;
    packed |= uint(bentNormal.x * 2047.0 + 0.5) << 21u;
    packed |= uint(bentNormal.y * 2047.0 + 0.5) << 10u;
    packed |= uint(bentNormal.z * 1023.0 + 0.5);
    return packed;
}

half4 unpackSharedShortRangeAO(uint packed)
{
    half3 bentNormal;
    bentNormal.x = half(packed >> 21u) / 2047.0;
    bentNormal.y = half((packed >> 10u) & 2047u) / 2047.0;
    bentNormal.z = half(packed & 1023u) / 1023.0;
    bentNormal = bentNormal * 2.0 - 1.0;
    half len = sqrt(dot(bentNormal, bentNormal));
    return half4(bentNormal / max(len, half(0.0001)), len);
}

void loadSharedShortRangeAO(Texture2D<uint> shortRangeAO, uint2 group, uint2 lane, uint2 shortRangeAOSize)
{
    if (all(lane < kShortRangeAOSharedTileSize)) {
        int2 loadCoord = int2(group * 4u + lane) - 1;
        loadCoord = clamp(loadCoord, int2(0), int2(shortRangeAOSize) - 1);
        sharedShortRangeAO[lane.x][lane.y] = unpackSharedShortRangeAO(shortRangeAO.Load(int3(loadCoord, 0)));
    }
    GroupMemoryBarrierWithGroupSync();
}

float shortRangeAONormalWeight(float3 sceneWorldNormal, float3 sampleWorldNormal)
{
    // UE StochasticLightingTileClassification.usf GetNormalWeight and FastMathThirdParty.ush acosFast.
    float x = saturate(dot(sampleWorldNormal, sceneWorldNormal));
    float angleBetweenNormals = (-0.156583f * x + (0.5f * kLumenProbePi)) * sqrt(1.0f - x);
    float normalWeight = 1.0f - saturate(angleBetweenNormals);
    return normalWeight * normalWeight;
}

// UE 5.7.4 StochasticLightingTileClassification.usf ComputeUpsampleWeights,
// opaque 2x2 downsample branch. Read guides through the same 4-rooks sample
// locations rather than allocating a second copy of the full-resolution guides.
float4 computeShortRangeAOUpsampleWeights(LumenTemporalGatherParams params, uint2 pixel, uint2 shortRangeAOSize,
    float4 surface, float3 worldNormal, Texture2D<float4> position, Texture2D<float4> normal)
{
    const int2 sampleOffsets[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    const uint2 viewSize = uint2(params.width, params.height);
    const float2 invViewSize = 1.0f / float2(viewSize);
    const float2 screenUV = (float2(pixel) + 0.5f) * invViewSize;
    const int2 downsampledCoord00 = int2(floor(screenUV * float2(viewSize) / 2.0f - 0.5f));
    const int2 screenCoordOffset = int2(pixel) - downsampledCoord00 * 2;

    ViewCameraConstants camera = params.currentCamera;
    float3 forward = cameraForwardFrom(camera.eye, camera.center);
    float3 right = cameraRightFrom(forward, camera.upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    bool orthographic = camera.upProjection.w > 0.5f;
    float halfHeight = orthographic ? camera.clipOrtho.z * 0.5f : tan(camera.viewport.w * 0.5f);
    float2 halfExtent = float2(camera.viewport.x * halfHeight, halfHeight);
    float2 screenPosition = (screenUV - 0.5f) * float2(2.0f, -2.0f);
    float2 projectionJitter = params.currentJitter * invViewSize * float2(2.0f, -2.0f);
    float4 scenePlane = float4(worldNormal, dot(surface.xyz, worldNormal));

    float4 upsampleWeights;
    float cornerDepth00 = -1.0f;
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
        int2 downsampledCoord = downsampledCoord00 + sampleOffsets[corner];
        int2 sampleScreenOffset = int2(lumenShortRangeAOJitter(uint2(downsampledCoord), params.frameIndex)) +
            sampleOffsets[corner] * 2 - screenCoordOffset;
        uint2 sampleScreenCoord = uint2(int2(pixel) + sampleScreenOffset);
        float cornerDepth = -1.0f;
        float3 cornerNormal = 0.0f;
        if (all(sampleScreenCoord < viewSize)) {
            cornerDepth = position.Load(int3(sampleScreenCoord, 0)).w;
            cornerNormal = normal.Load(int3(sampleScreenCoord, 0)).xyz;
        }
        if (corner == 0u) { cornerDepth00 = cornerDepth; }

        float weight = (2.0f - abs(sampleScreenOffset.x)) * (2.0f - abs(sampleScreenOffset.y));
        float2 cornerScreenPosition = screenPosition + float2(sampleScreenOffset) * invViewSize * float2(2.0f, -2.0f);
        float2 offset = (cornerScreenPosition + projectionJitter) * halfExtent * (orthographic ? 1.0f : cornerDepth);
        // The reference's 01 corner uses the 00 depth for the view-space Z component.
        float viewDepth = corner == 2u ? cornerDepth00 : cornerDepth;
        float3 cornerPosition = camera.eye.xyz + forward * viewDepth + right * offset.x + up * offset.y;
        float planeDistance = abs(dot(float4(cornerPosition, -1.0f), scenePlane));
        float relativeDepthDifference = planeDistance / surface.w;
        float depthWeight = cornerDepth > 0.0f ? exp2(-5000.0f * (relativeDepthDifference * relativeDepthDifference)) : 0.0f;
        weight *= depthWeight;
        weight *= shortRangeAONormalWeight(worldNormal, lumenShortRangeAODecodeNormal(cornerNormal));
        upsampleWeights[corner] = all(uint2(downsampledCoord) < shortRangeAOSize) ? weight : 0.0f;
    }
    return upsampleWeights;
}

// UE 5.7.4 StochasticLightingUpsample.ush, 2x2 stochastic bilinear offsets.
int2 getShortRangeAOStochasticBilinearOffset(float randomScalar, float4 upsampleWeights)
{
    float4 preSum;
    preSum.x = upsampleWeights.x;
    preSum.y = preSum.x + upsampleWeights.y;
    preSum.z = preSum.y + upsampleWeights.z;
    preSum.w = preSum.z + upsampleWeights.w;
    preSum /= max(preSum.w, 0.00001f);
    if (randomScalar <= preSum.x) { return int2(0, 0); }
    if (randomScalar <= preSum.y) { return int2(1, 0); }
    if (randomScalar <= preSum.z) { return int2(0, 1); }
    return int2(1, 1);
}

struct ShortRangeAONeighborhood {
    half4 center, extent;
};

ShortRangeAONeighborhood getShortRangeAONeighborhood(uint2 sharedCoord, half4 centerShortRangeAO)
{
    half4 aoSum = centerShortRangeAO;
    half4 aoSqSum = centerShortRangeAO * centerShortRangeAO;
    half weightSum = 1.0;
    const int kernelSize = 1;
    for (int neighborOffsetY = -kernelSize; neighborOffsetY <= kernelSize; ++neighborOffsetY) {
        for (int neighborOffsetX = -kernelSize; neighborOffsetX <= kernelSize; ++neighborOffsetX) {
            bool center = neighborOffsetX == 0 && neighborOffsetY == 0;
            uint2 neighborSharedCoord = uint2(int2(sharedCoord) + int2(neighborOffsetX, neighborOffsetY));
            if (!center && all(neighborSharedCoord < kShortRangeAOSharedTileSize)) {
                half4 neighborAO = sharedShortRangeAO[neighborSharedCoord.x][neighborSharedCoord.y];
                aoSum += neighborAO;
                aoSqSum += neighborAO * neighborAO;
                weightSum += 1.0;
            }
        }
    }
    half4 m1 = aoSum / weightSum;
    half4 m2 = aoSqSum / weightSum;
    half4 variance = max(m2 - m1 * m1, half(0.0));
    half4 stdDev = sqrt(variance);
    return {m1, stdDev}; // Epic default neighborhood clamp scale is 1.
}

half4 weightedAverageShortRangeAO(half4 v00, half4 v10, half4 v01, half4 v11, half4 weights)
{
    half4 result = v00 * weights.x + v10 * weights.y + v01 * weights.z + v11 * weights.w;
    return result / max(dot(weights, half(1.0)), half(0.0001));
}
#endif

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

// UE 5.7.4 Pack/UnpackFastUpdateModeAmountAndNumFramesAccumulated.
float packHistoryState(float fastUpdateAmount, float numFramesAccumulated, float maxFramesAccumulated)
{
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    // Only a zero amount is encoded as not moving.
    float encodedFastUpdateAmount = ceil(fastUpdateAmount * 15.0f);
#endif
    float localMaxFramesAccumulated = min(maxFramesAccumulated, 15.0f);
    float encodedNumFramesAccumulated = round(saturate(numFramesAccumulated / localMaxFramesAccumulated) * 15.0f);
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    return (encodedFastUpdateAmount * 16.0f + encodedNumFramesAccumulated) / 255.0f;
#else
    return encodedNumFramesAccumulated / 255.0f;
#endif
}

float4 unpackHistoryState(float4 packed, float maxFramesAccumulated, out float4 fastUpdateAmount)
{
    packed = round(packed * 255.0f);
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    float4 encodedFastUpdateAmount = floor(packed / 16.0f);
    float4 encodedNumFramesAccumulated = packed - encodedFastUpdateAmount * 16.0f;
    fastUpdateAmount = encodedFastUpdateAmount / 15.0f;
#else
    float4 encodedNumFramesAccumulated = packed;
    fastUpdateAmount = 0.0f;
#endif
    return select(encodedNumFramesAccumulated == 15.0f, maxFramesAccumulated,
        encodedNumFramesAccumulated / 15.0f * min(maxFramesAccumulated, 15.0f));
}

float3 quantizeHistoryLighting(float3 lighting, float randomScalar)
{
#if defined(METALLIC_LUMEN_DITHER_HISTORY)
    // UE QuantizeForFloatRenderTarget / QuantizeFloatColor, RGBA16F has ten mantissa bits.
    float3 error = lighting * (1.0f / 1024.0f);
    error = asfloat(asuint(error) & ~0x007fffffu);
    return lighting + error * randomScalar;
#else
    return lighting;
#endif
}

// UE 5.7.4 StochasticLightingTileClassification.usf, the opaque linear-depth branch.
// With instance motion, the previous world point supplies the same two values
// as GetHistoryScreenPosition: previous screen UV and previous scene depth.
void accumulateLighting(LumenTemporalGatherParams params, float3 worldPosition, float3 previousWorldPosition,
    float3 normal, float4 newDiffuseLighting, float randomScalar,
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    inout half4 outShortRangeAO, uint2 shortRangeAOSharedCoord, bool validReconstruction,
#endif
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    inout float3 outRoughSpecular,
#endif
    out float3 outDiffuseIndirect, out float newNumFramesAccumulated, out float outHistoryWeight,
    out float outFastUpdateAmount)
{
    outDiffuseIndirect = newDiffuseLighting.rgb;
    newNumFramesAccumulated = 0.0f;
    outHistoryWeight = 0.0f;
    outFastUpdateAmount = 0.0f;
    ViewCameraConstants previousCamera = params.previousCamera;
    float3 previousForward = cameraForwardFrom(previousCamera.eye, previousCamera.center);
    float reprojectedSceneDepth = dot(previousWorldPosition - previousCamera.eye.xyz, previousForward);
    float2 historyScreenUV;
    if (reprojectedSceneDepth < previousCamera.clipOrtho.x || reprojectedSceneDepth > previousCamera.clipOrtho.y
        || !projectWorldToUv(previousWorldPosition, previousCamera.eye, previousCamera.center,
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
    float4 historyFastUpdateAmount;
    float4 historyNumFramesAccumulated = unpackHistoryState(float4(history00.w, history10.w, history01.w, history11.w),
        params.maxFramesAccumulated, historyFastUpdateAmount);
    const float numFramesAccumulated = min(weightedAverage(historyNumFramesAccumulated + 1.0f, finalWeights),
        params.maxFramesAccumulated);
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    // UE 5.7.4 defaults for FractionOfLightingMovingForFastUpdateMode and MaxFastUpdateModeAmount.
    const float invFractionOfLightingMovingForFastUpdateMode = 1.0f / 0.1f;
    const float maxFastUpdateAmount = 0.9f;
    const float lightingIsMoving = abs(newDiffuseLighting.w);
    float fastUpdateAmount = saturate(lightingIsMoving * invFractionOfLightingMovingForFastUpdateMode);
    fastUpdateAmount = saturate(min((fastUpdateAmount - 0.2f) / 0.8f, maxFastUpdateAmount));
    outFastUpdateAmount = fastUpdateAmount;
    float fastUpdateHistory = weightedAverage(historyFastUpdateAmount, finalWeights);
    fastUpdateHistory = min(fastUpdateHistory, maxFastUpdateAmount);
    fastUpdateAmount = max(fastUpdateAmount, fastUpdateHistory);
    newNumFramesAccumulated = min(numFramesAccumulated, (1.0f - fastUpdateAmount) * params.maxFramesAccumulated);
#else
    newNumFramesAccumulated = numFramesAccumulated;
#endif
    float alpha = 1.0f / (1.0f + newNumFramesAccumulated);
    bool lightingIsInvalid = newDiffuseLighting.w <= 0.0f;
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    lightingIsInvalid = lightingIsInvalid || !validReconstruction;
#endif
    if (lightingIsInvalid && newNumFramesAccumulated >= 1.0f) {
        alpha = 1.0f / (1.0f + 4.0f * newNumFramesAccumulated);
    }
    outDiffuseIndirect = lerp(historyDiffuseIndirect, newDiffuseLighting.rgb, alpha);
    outHistoryWeight = 1.0f - alpha;
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    // UE LumenScreenProbeGatherTemporal.usf reuses the diffuse geometry taps and age.
    Texture2D<float4> previousRoughSpecular = resolveUniform(params.previousRoughSpecular);
    float3 historyRoughSpecular = weightedAverage(previousRoughSpecular.Load(int3(p00, 0)).rgb,
        previousRoughSpecular.Load(int3(p10, 0)).rgb, previousRoughSpecular.Load(int3(p01, 0)).rgb,
        previousRoughSpecular.Load(int3(p11, 0)).rgb, finalWeights);
    outRoughSpecular = lerp(historyRoughSpecular, outRoughSpecular, alpha);
#endif
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    Texture2D<uint> previousShortRangeAO = resolveUniform(params.previousShortRangeAO);
    half4 historyShortRangeAO00 = unpackSharedShortRangeAO(previousShortRangeAO.Load(int3(p00, 0)));
    half4 historyShortRangeAO10 = unpackSharedShortRangeAO(previousShortRangeAO.Load(int3(p10, 0)));
    half4 historyShortRangeAO01 = unpackSharedShortRangeAO(previousShortRangeAO.Load(int3(p01, 0)));
    half4 historyShortRangeAO11 = unpackSharedShortRangeAO(previousShortRangeAO.Load(int3(p11, 0)));
    half4 historyShortRangeAO = weightedAverageShortRangeAO(historyShortRangeAO00, historyShortRangeAO10,
        historyShortRangeAO01, historyShortRangeAO11, half4(finalWeights));
    ShortRangeAONeighborhood neighborhood = getShortRangeAONeighborhood(shortRangeAOSharedCoord, outShortRangeAO);
    historyShortRangeAO = clamp(historyShortRangeAO, neighborhood.center - neighborhood.extent,
        neighborhood.center + neighborhood.extent);
    outShortRangeAO = lerp(historyShortRangeAO, outShortRangeAO, half(alpha));
#endif
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenTemporalGatherMain(uint3 thread : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenTemporalGatherParams params = getParameters<LumenTemporalGatherParams>();
    uint2 viewSize = uint2(params.width, params.height);
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    uint2 shortRangeAOSize = (viewSize + 1u) / 2u;
    loadSharedShortRangeAO(resolveUniform(params.shortRangeAORaw), group.xy, lane.xy, shortRangeAOSize);
#endif
    if (any(thread.xy >= viewSize)) {
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
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    float3 outRoughSpecular = surface.w > 0.0f ? resolveUniform(params.roughSpecular).Load(texel).rgb : 0.0f;
#endif
    float newNumFramesAccumulated = 0.0f;
    float outHistoryWeight = 0.0f;
    float outFastUpdateAmount = 0.0f;
#if defined(METALLIC_LUMEN_DITHER_HISTORY) || defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    // UE 5.7.4 BlueNoiseScalar(ScreenCoord, ScreenProbeGatherStateFrameIndex).
    float randomScalar = lumenProbeBlueNoiseScalar(resolveUniform(params.blueNoiseScalar), thread.xy, params.frameIndex);
#else
    const float randomScalar = 0.0f;
#endif
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    // UE ShortRangeAO invalid material initialization and full-resolution reconstruction.
    half4 outShortRangeAO = half4(0.0, 0.0, 1.0, 1.0);
    uint2 shortRangeAOSharedCoord = 0u;
    bool validReconstruction = false;
    if (surface.w > 0.0f) {
        float4 upsampleWeights = computeShortRangeAOUpsampleWeights(params, thread.xy, shortRangeAOSize,
            surface, surfaceNormal, position, normal);
        int2 stochasticBilinearOffset = getShortRangeAOStochasticBilinearOffset(randomScalar, upsampleWeights);
        validReconstruction = dot(upsampleWeights, 1.0f) >= 0.01f;
        int2 sampleCoord = (int2(thread.xy) - 1) / 2 + stochasticBilinearOffset;
        int2 shortRangeAOCoord = clamp(sampleCoord, int2(0), int2(shortRangeAOSize) - 1);
        shortRangeAOSharedCoord = uint2((2 + int2(lane.xy) - 1) / 2 + stochasticBilinearOffset +
            shortRangeAOCoord - sampleCoord);
        outShortRangeAO = sharedShortRangeAO[shortRangeAOSharedCoord.x][shortRangeAOSharedCoord.y];
    }
#endif
    if (params.historyValid != 0u && surface.w > 0.0f) {
        float4 previousSurface = surface;
#if defined(METALLIC_LUMEN_DYNAMIC_REPROJECTION)
        previousSurface = resolveUniform(params.previousPosition).Load(texel);
#endif
        if (previousSurface.w > 0.0f) {
            accumulateLighting(params, surface.xyz, previousSurface.xyz, surfaceNormal, newDiffuseLighting, randomScalar,
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
                outShortRangeAO, shortRangeAOSharedCoord, validReconstruction,
#endif
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
                outRoughSpecular,
#endif
                outDiffuseIndirect, newNumFramesAccumulated, outHistoryWeight, outFastUpdateAmount);
        }
    }

    DiffuseTexture indirectOutput = resolveUniform(params.indirectOutput);
    FullPrecisionTexture color = resolveUniform(params.color);
    ScalarTexture historyWeight = resolveUniform(params.historyWeight);
    DiffuseTexture currentDiffuse = resolveUniform(params.currentDiffuse);
    FullPrecisionTexture currentGeometry = resolveUniform(params.currentGeometry);
    float3 storedDiffuse = quantizeHistoryLighting(outDiffuseIndirect, randomScalar);
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    float3 storedRoughSpecular = quantizeHistoryLighting(-min(-outRoughSpecular, 0.0f), randomScalar);
    resolveUniform(params.roughSpecularOutput)[thread.xy] = float4(storedRoughSpecular, 0.0f);
    resolveUniform(params.currentRoughSpecular)[thread.xy] = float4(storedRoughSpecular, 0.0f);
#endif
    float3 sceneIndirect = outDiffuseIndirect;
#if defined(METALLIC_LUMEN_SHORT_RANGE_AO)
    uint packedShortRangeAO = packSharedShortRangeAO(outShortRangeAO);
    resolveUniform(params.shortRangeAO)[thread.xy] = packedShortRangeAO;
    resolveUniform(params.currentShortRangeAO)[thread.xy] = packedShortRangeAO;
    float4 diffuseAlbedo = resolveUniform(params.diffuseAlbedo).Load(texel);
    // UE composites from its quantized history output. Reproduce the Metallic
    // RGBA16F storage conversion in this fused composition.
    float3 historyTexel = float3(half3(storedDiffuse));
    sceneIndirect = applyLumenShortRangeAO(historyTexel, diffuseAlbedo.rgb, packedShortRangeAO) * diffuseAlbedo.a;
    indirectOutput[thread.xy] = float4(sceneIndirect, newDiffuseLighting.w);
#else
    indirectOutput[thread.xy] = float4(storedDiffuse, newDiffuseLighting.w);
#endif
    float4 scene = directColor.Load(texel);
    float4 sceneColor = float4(scene.rgb + sceneIndirect, scene.a);
    color[thread.xy] = sceneColor;
    // UE 5.7.4 ScreenSpaceRayTracingInput extracts resolved color and depth before postprocessing.
    if (params.publishScreenTraceHistory != 0u) {
        resolveUniform(params.screenHistoryDepth)[thread.xy] = resolveUniform(params.sceneDepth).Load(texel);
        resolveUniform(params.screenHistoryColor)[thread.xy] = sceneColor;
    }
    historyWeight[thread.xy] = outHistoryWeight;
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    resolveUniform(params.fastUpdateAmount)[thread.xy] = outFastUpdateAmount;
#endif
    currentDiffuse[thread.xy] = float4(storedDiffuse, packHistoryState(outFastUpdateAmount, newNumFramesAccumulated,
        params.maxFramesAccumulated));
    currentGeometry[thread.xy] = float4(surfaceNormal, surface.w);
}
