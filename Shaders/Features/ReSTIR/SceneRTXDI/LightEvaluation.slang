#language slang 2026
implementing SceneRTXDI;

import Core;
import Lighting;
using Metallic;
using Metallic.Lighting;

#define gEnvironmentMap resolveUniform(getResourceParameters<SceneResourceParameters>().environmentMap)

LightEvaluation evaluateDeltaLight(PunctualLightSample light, Surface surface)
{
    float3 lightDirection = light.direction;
    float nDotL = saturate(dot(surface.normal, lightDirection));
    float3 halfVector = safeNormalize(lightDirection + surface.viewDirection, surface.normal);
    float nDotH = saturate(dot(surface.normal, halfVector));
    float specularPower = lerp(128.0, 4.0, surface.roughness);
    float3 incidentRadiance = light.illuminance;

    LightEvaluation evaluation;
    evaluation.diffuseRadiance = incidentRadiance * (nDotL / kPi);
    evaluation.specularRadiance = incidentRadiance * nDotL *
        ((specularPower + 2.0) / (2.0 * kPi)) *
        pow(nDotH, specularPower);
    evaluation.contribution = evaluation.diffuseRadiance * surface.diffuseAlbedo +
        evaluation.specularRadiance * surface.specularF0;
    evaluation.direction = lightDirection;
    evaluation.distance = light.distance;
    evaluation.target = max(luminance(evaluation.contribution), 0.0);
    return evaluation;
}

LightEvaluation evaluateLight(uint lightIndex, Surface surface, SceneRTXDIPush push)
{
    if (lightIndex >= punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights))) {
        return emptyLightEvaluation();
    }
    return evaluateDeltaLight(samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(
        getResourceParameters<SceneResourceParameters>().lights), lightIndex, surface.position), surface);
}

bool isEnvironmentSample(uint lightIndex)
{
    return (lightIndex & kEnvironmentSampleBit) != 0u;
}

uint environmentTexelIndex(uint lightIndex)
{
    return lightIndex & kEnvironmentSampleMask;
}

uint encodeEnvironmentSample(uint texelIndex)
{
    return kEnvironmentSampleBit | (texelIndex & kEnvironmentSampleMask);
}

float3 environmentLocalDirectionToWorld(float3 direction, float rotationRadians)
{
    float sine = sin(rotationRadians);
    float cosine = cos(rotationRadians);
    return float3(
        direction.x * cosine + direction.z * sine,
        direction.y,
        -direction.x * sine + direction.z * cosine);
}

float3 environmentLocalDirectionFromUv(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = uv.y * kPi;
    float sinTheta = sin(theta);
    return float3(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
}

float2 environmentUvFromDirection(float3 direction, float rotationRadians)
{
    float sine = sin(-rotationRadians);
    float cosine = cos(-rotationRadians);
    float3 localDirection = float3(
        direction.x * cosine + direction.z * sine,
        direction.y,
        -direction.x * sine + direction.z * cosine);
    float phi = atan2(localDirection.z, localDirection.x);
    float u = frac(phi / (2.0 * kPi) + 0.5);
    float v = acos(clamp(localDirection.y, -1.0, 1.0)) / kPi;
    return float2(u, v);
}

float environmentTexelSolidAngle(uint y, uint width, uint height)
{
    float theta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    return max(
        (2.0 * kPi / float(max(width, 1u))) *
            (kPi / float(max(height, 1u))) *
            sin(theta),
        0.0000001);
}

float3 environmentRadiance(float3 direction, SceneRTXDIPush push)
{
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || push.environmentIntensity <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return float3(0.0, 0.0, 0.0);
    }
    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    return max(gEnvironmentMap.Load(int3(texel, 0)).rgb, float3(0.0, 0.0, 0.0)) *
        push.environmentIntensity;
}

LightEvaluation evaluateEnvironmentSample(uint encodedSample, Surface surface, SceneRTXDIPush push)
{
    LightEvaluation evaluation = emptyLightEvaluation();
    evaluation.distance = max(push.clipOrtho.y, push.sceneCenterRadius.w * 16.0);

    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    uint texelIndex = environmentTexelIndex(encodedSample);
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || width == 0u || height == 0u ||
        texelIndex >= width * height) {
        return evaluation;
    }

    uint2 texel = uint2(texelIndex % width, texelIndex / width);
    if (texel.y >= height) {
        return evaluation;
    }
    float2 uv = (float2(texel) + 0.5) / float2(width, height);
    evaluation.direction = safeNormalize(
        environmentLocalDirectionToWorld(
            environmentLocalDirectionFromUv(uv),
            push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
    float nDotL = saturate(dot(surface.normal, evaluation.direction));
    float3 halfVector = safeNormalize(evaluation.direction + surface.viewDirection, surface.normal);
    float nDotH = saturate(dot(surface.normal, halfVector));
    float specularPower = lerp(128.0, 4.0, surface.roughness);
    float3 incidentRadiance = max(
        gEnvironmentMap.Load(int3(texel, 0)).rgb,
        float3(0.0, 0.0, 0.0)) * push.environmentIntensity;
    evaluation.diffuseRadiance = incidentRadiance * (nDotL / kPi);
    evaluation.specularRadiance = incidentRadiance * nDotL *
        ((specularPower + 2.0) / (2.0 * kPi)) *
        pow(nDotH, specularPower);
    evaluation.contribution = evaluation.diffuseRadiance * surface.diffuseAlbedo +
        evaluation.specularRadiance * surface.specularF0;
    evaluation.target = max(luminance(evaluation.contribution), 0.0);
    return evaluation;
}

LightEvaluation evaluateLightSample(uint lightIndex, Surface surface, SceneRTXDIPush push)
{
    return isEnvironmentSample(lightIndex)
        ? evaluateEnvironmentSample(lightIndex, surface, push)
        : evaluateLight(lightIndex, surface, push);
}
