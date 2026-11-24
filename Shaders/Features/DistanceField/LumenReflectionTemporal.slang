implementing LumenReflection;

import ShaderCore;
import ParameterRoot;
import LumenScreenTrace;
import LumenReflectionSampling;

using Metallic;
using Metallic.Lumen;

struct LumenReflectionTemporalParams
{
    ResourceHandle<Texture2D<float4>> position, normal, specular, previousPosition;
    ResourceHandle<Texture2D<float4>> resolvedReflections;
    ResourceHandle<Texture2D<float>> resolvedHitDistance;
    ResourceHandle<Texture2D<float4>> previousLighting;
    ResourceHandle<Texture2D<float>> previousDepth, previousAge;
    ResourceHandle<RWTexture2D<float4>> temporalReflections;
    ResourceHandle<RWTexture2D<float2>> temporalState;
    ResourceHandle<RWTexture2D<float4>> currentLighting;
    ResourceHandle<RWTexture2D<float>> currentDepth, currentAge;
    uint width, height;
    ViewCameraConstants currentCamera, previousCamera;
    uint historyValid, dynamicReprojection, frameIndex;
};

static const uint kReflectionTemporalTileSize = 8u;
static const int kReflectionTemporalTileBorder = 2;
static const uint kReflectionTemporalSharedSize = kReflectionTemporalTileSize + 2u * kReflectionTemporalTileBorder;
static const float kReflectionTemporalMaxFrames = 12.0f;
static const float kReflectionHistoryDistanceThreshold = 0.03f;
groupshared float3 sharedResolvedSpecularYCoCg[kReflectionTemporalSharedSize][kReflectionTemporalSharedSize];

bool isReflectionLightingValid(float3 lighting)
{
    return !all(lighting == kInvalidReflectionLighting);
}

// UE 5.7.4 LumenReflectionDenoiserCommon.ush.
float3 reflectionRGBToYCoCg(float3 rgb)
{
    float y = dot(rgb, float3(0.25f, 0.5f, 0.25f));
    float co = dot(rgb, float3(0.5f, 0.0f, -0.5f));
    float cg = dot(rgb, float3(-0.25f, 0.5f, -0.25f));
    return float3(y, co, cg);
}

float3 reflectionYCoCgToRGB(float3 ycocg)
{
    float y = ycocg.x;
    float co = ycocg.y;
    float cg = ycocg.z;
    return float3(y + co - cg, y + cg, y - co - cg);
}

float getReflectionMaxFramesAccumulated(float roughness)
{
#if defined(METALLIC_LUMEN_REFLECTION_DOWNSAMPLE_2X)
    return kReflectionTemporalMaxFrames;
#else
    return lerp(2.0f, kReflectionTemporalMaxFrames, saturate(roughness / 0.05f));
#endif
}

struct ReflectionNeighborhood
{
    float3 center, extent;
};

ReflectionNeighborhood getReflectionNeighborhood(uint2 localCoord, float3 centerSample)
{
    centerSample = reflectionRGBToYCoCg(centerSample);
    float weightSum = 1.0f;
    float3 sampleSum = centerSample;
    float3 sampleSqSum = centerSample * centerSample;

    const int kernelSize = 2;
    for (int y = -kernelSize; y <= kernelSize; ++y) {
        for (int x = -kernelSize; x <= kernelSize; ++x) {
            bool center = x == 0 && y == 0;
            bool corner = abs(x) == kernelSize && abs(y) == kernelSize;
            if (!center && !corner) {
                float3 neighbor = sharedResolvedSpecularYCoCg[kReflectionTemporalTileBorder + int(localCoord.x) + x]
                    [kReflectionTemporalTileBorder + int(localCoord.y) + y];
                if (isReflectionLightingValid(neighbor)) {
                    sampleSum += neighbor;
                    sampleSqSum += neighbor * neighbor;
                    weightSum += 1.0f;
                }
            }
        }
    }

    float3 m1 = sampleSum / weightSum;
    float3 m2 = sampleSqSum / weightSum;
    float3 variance = max(m2 - m1 * m1, 0.0f);
    return {m1, sqrt(variance)};
}

struct ReflectionLightingHistory
{
    bool valid;
    float3 specular;
    float secondMoment, numFramesAccumulated;
};

// UE LumenReflectionDenoiserTemporal.usf, GetLightingHistory. Load the four taps in Gather.wzxy order.
ReflectionLightingHistory getReflectionLightingHistory(LumenReflectionTemporalParams params,
    ScreenTraceProjection previous, float3 historyScreenPosition, bool fromReflectHit,
    float3 translatedWorldPosition, float3 sceneWorldNormal, float noise)
{
    ReflectionLightingHistory history = {};
    float2 historyUV = screenTraceScreenUV(historyScreenPosition.xy);
    if (!(all(historyUV >= 0.0f) && all(historyUV <= 1.0f))) { return history; }

    int2 extent = int2(params.width, params.height);
    float2 invExtent = 1.0f / float2(extent);
    // A one-pixel axis has only one point-clamped sample.
    float2 inset = min(0.51f, float2(extent) * 0.5f);
    historyUV = clamp(historyUV, inset * invExtent, (float2(extent) - inset) * invExtent);
    float2 historyTexel = historyUV * float2(extent) - 0.5f;
    int2 pixel = int2(floor(historyTexel));
    float2 bilinear = frac(historyTexel);
    float4 weights = float4(
        (1.0f - bilinear.y) * (1.0f - bilinear.x),
        (1.0f - bilinear.y) * bilinear.x,
        bilinear.y * (1.0f - bilinear.x),
        bilinear.y * bilinear.x);
    int2 taps[4] = {
        clamp(pixel, 0, extent - 1), clamp(pixel + int2(1, 0), 0, extent - 1),
        clamp(pixel + int2(0, 1), 0, extent - 1), clamp(pixel + int2(1, 1), 0, extent - 1)
    };

    // The reference disables REFLECT_HIT_DEPTH_TEST. Only surface history compares scene depth.
    if (!fromReflectHit) {
        let previousDepth = resolveUniform(params.previousDepth);
        float4 sampleDepth = float4(previousDepth.Load(int3(taps[0], 0)), previousDepth.Load(int3(taps[1], 0)),
            previousDepth.Load(int3(taps[2], 0)), previousDepth.Load(int3(taps[3], 0)));
        float depth = screenTraceConvertFromDeviceZ(previous, historyScreenPosition.z);
        float threshold = kReflectionHistoryDistanceThreshold * lerp(0.5f, 1.5f, noise);
        float3 v = normalize(-translatedWorldPosition);
        threshold /= clamp(saturate(dot(v, sceneWorldNormal)), 0.1f, 1.0f);
        float4 distanceToHistory = abs(sampleDepth - depth);
        weights *= select(distanceToHistory >= depth * threshold, 0.0f, 1.0f);
    }

    if (!any(weights > 0.01f)) { return history; }

    let previousLighting = resolveUniform(params.previousLighting);
    float4 specularHistory[4];
    for (uint i = 0u; i < 4u; ++i) {
        specularHistory[i] = previousLighting.Load(int3(taps[i], 0));
        if (isReflectionLightingValid(specularHistory[i].rgb)) {
            specularHistory[i].rgb = reflectionRGBToDenoiserSpace(specularHistory[i].rgb);
        } else {
            weights[i] = 0.0f;
        }
    }

    history.valid = any(weights > 0.01f);
    if (history.valid) {
        float weightSum = dot(weights, 1.0f);
        float4 specularAndSecondMoment = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            specularAndSecondMoment += specularHistory[i] * weights[i];
        }
        specularAndSecondMoment = specularAndSecondMoment / weightSum;
        history.specular = specularAndSecondMoment.rgb;
        history.secondMoment = specularAndSecondMoment.w;

        let previousAge = resolveUniform(params.previousAge);
        float4 age = float4(previousAge.Load(int3(taps[0], 0)), previousAge.Load(int3(taps[1], 0)),
            previousAge.Load(int3(taps[2], 0)), previousAge.Load(int3(taps[3], 0)));
        history.numFramesAccumulated = (dot(age, weights) * 32.0f) / weightSum;
    }
    return history;
}

float3 previousReflectionScreenPosition(ScreenTraceProjection previous, float3 worldPosition)
{
    float4 previousClip = screenTraceWorldToClip(previous, worldPosition);
    return previousClip.xyz / previousClip.w;
}

// UE 5.7.4 LumenReflectionDenoiserTemporal.usf, full-resolution opaque permutation.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionTemporalMain(uint3 thread : SV_DispatchThreadID, uint3 local : SV_GroupThreadID,
    uint3 group : SV_GroupID)
{
    LumenReflectionTemporalParams params = getParameters<LumenReflectionTemporalParams>();
    uint2 pixel = thread.xy;
    uint2 extent = uint2(params.width, params.height);
    let resolved = resolveUniform(params.resolvedReflections);
    for (uint y = local.y; y < kReflectionTemporalSharedSize; y += kReflectionTemporalTileSize) {
        for (uint x = local.x; x < kReflectionTemporalSharedSize; x += kReflectionTemporalTileSize) {
            int2 coord = int2(group.xy * kReflectionTemporalTileSize + uint2(x, y)) - kReflectionTemporalTileBorder;
            float3 ycocg = kInvalidReflectionLighting;
            bool corner = (x == 0u || x + 1u == kReflectionTemporalSharedSize) &&
                (y == 0u || y + 1u == kReflectionTemporalSharedSize);
            if (!corner && all(coord >= 0) && all(coord < int2(extent))) {
                float3 sample = resolved.Load(int3(coord, 0)).rgb;
                if (isReflectionLightingValid(sample)) {
                    ycocg = reflectionRGBToYCoCg(reflectionRGBToDenoiserSpace(sample));
                }
            }
            sharedResolvedSpecularYCoCg[x][y] = ycocg;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (any(pixel >= extent)) { return; }
    float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    float3 specularLighting = sharedResolvedSpecularYCoCg[local.x + kReflectionTemporalTileBorder]
        [local.y + kReflectionTemporalTileBorder];
    float secondMoment = 0.0f;
    float numFramesAccumulated = 0.0f;
    uint historyMethod = 0u;

    if (isReflectionLightingValid(specularLighting)) {
        specularLighting = reflectionYCoCgToRGB(specularLighting);
        float luminance = WorkingColor.luminance(specularLighting);
        secondMoment = luminance * luminance;

        if (params.historyValid != 0u) {
            ScreenTraceProjection current = makeScreenTraceProjection(params.currentCamera, 0.0f);
            ScreenTraceProjection previous = makeScreenTraceProjection(params.previousCamera, 0.0f);
            float2 screenUV = (float2(pixel) + 0.5f) / float2(extent);
            float2 screenPosition = screenTraceScreenPosition(screenUV);
            float sceneDepth = position.w;
            float3 worldPosition = screenTraceScreenToWorld(current, screenPosition, sceneDepth);
            float3 translatedWorldPosition = worldPosition - params.currentCamera.eye.xyz;
            float3 motionOffset = 0.0f;
            if (params.dynamicReprojection != 0u) {
                float4 previousPosition = resolveUniform(params.previousPosition).Load(int3(pixel, 0));
                if (previousPosition.w > 0.0f) {
                    // Apply only the object-motion part of receiver velocity to the virtual hit projection.
                    motionOffset = previousReflectionScreenPosition(previous, previousPosition.xyz) -
                        previousReflectionScreenPosition(previous, position.xyz);
                }
            }

            float roughness = resolveUniform(params.specular).Load(int3(pixel, 0)).w;
            float s = saturate(1.0f - roughness);
            float dominantDirectionFactor = s * (sqrt(s) + roughness);
            float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
            float noise = screenTraceNoise(float2(pixel), float(params.frameIndex % 8u));
            ReflectionLightingHistory history = {};
            bool preferredHistoryFailed = false;
            if (dominantDirectionFactor >= 0.5f) {
                float hitDistance = resolveUniform(params.resolvedHitDistance).Load(int3(pixel, 0)) * 0.01f;
                float3 virtualHit = screenTraceScreenToWorld(current, screenPosition, sceneDepth + hitDistance);
                float3 historyScreen = previousReflectionScreenPosition(previous, virtualHit) + motionOffset;
                history = getReflectionLightingHistory(params, previous, historyScreen, true,
                    translatedWorldPosition, normal, noise);
                preferredHistoryFailed = !history.valid;
                if (history.valid) { historyMethod = 1u; }
            }

            if (!history.valid) {
                float3 historyScreen = previousReflectionScreenPosition(previous, worldPosition) + motionOffset;
                history = getReflectionLightingHistory(params, previous, historyScreen, false,
                    translatedWorldPosition, normal, noise);
                if (history.valid) { historyMethod = 2u; }
            }

            float maxFramesAccumulated = getReflectionMaxFramesAccumulated(roughness);
            if (history.valid) {
                float3 historyLighting = reflectionRGBToYCoCg(history.specular);
                ReflectionNeighborhood neighborhood = getReflectionNeighborhood(local.xy, specularLighting);
#if defined(METALLIC_LUMEN_REFLECTION_DOWNSAMPLE_2X)
                // UE 5.7.4 LumenReflectionDenoiserTemporal.usf reconstructed sample clamp.
                if (any(pixel % 2u != lumenReflectionScreenTileJitter(pixel / 2u, params.frameIndex))) {
                    neighborhood.extent *= 2.0f;
                }
#endif
                float3 clampedHistory = clamp(historyLighting, neighborhood.center - neighborhood.extent,
                    neighborhood.center + neighborhood.extent);
                float normalizedDistance = length(abs(clampedHistory - historyLighting) / max(neighborhood.extent, 0.1f));
                float confidence = saturate(1.0f - normalizedDistance);
                history.specular = reflectionYCoCgToRGB(clampedHistory);

                if (preferredHistoryFailed) { confidence = 0.0f; }
                confidence = 0.75f * confidence + 0.25f;
                numFramesAccumulated = min(history.numFramesAccumulated * confidence + 1.0f, maxFramesAccumulated);
                float alpha = 1.0f / numFramesAccumulated;
                specularLighting = lerp(history.specular, specularLighting, alpha);
                secondMoment = lerp(history.secondMoment, secondMoment, alpha);
            }
        }

        specularLighting = max(WorkingColor.sanitizeRadiance(specularLighting), 0.0f);
        secondMoment = max(isfinite(secondMoment) ? secondMoment : 0.0f, 0.0f);
        specularLighting = reflectionDenoiserSpaceToRGB(specularLighting);
    }

    float fade = resolved.Load(int3(pixel, 0)).a;
    resolveUniform(params.temporalReflections)[pixel] = float4(specularLighting, fade);
    resolveUniform(params.temporalState)[pixel] = float2(numFramesAccumulated, float(historyMethod));
    resolveUniform(params.currentLighting)[pixel] = float4(specularLighting, secondMoment);
    resolveUniform(params.currentDepth)[pixel] = position.w;
    resolveUniform(params.currentAge)[pixel] = numFramesAccumulated / 32.0f + 0.5f / 255.0f;
}
