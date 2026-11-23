implementing LumenReflection;

import ShaderCore;
import ParameterRoot;
import LumenProbeMath;

using Metallic;
using Metallic.Lumen;

struct LumenReflectionBilateralParams
{
    ViewCameraConstants camera;
    ResourceHandle<Texture2D<float4>> position, normal, specular, temporalReflections, currentLighting;
    ResourceHandle<Texture2D<float>> currentAge;
    ResourceHandle<RWTexture2D<float4>> bilateralReflections;
    uint width, height, frameIndex;
};

static const uint kReflectionBilateralNumSamples = 4u;
static const float kReflectionBilateralKernelRadius = 8.0f;
static const float kReflectionBilateralDepthWeightScale = 10000.0f;

// UE 5.7.4 Common.ush, AtanFast (x >= 0), and LumenReflectionDenoiserCommon.ush.
float reflectionAtanFast(float x)
{
    float3 a = x < 1.0f ? float3(x, 0.0f, 1.0f) : float3(1.0f / x, 0.5f * kLumenProbePi, -1.0f);
    return a.y + a.z * (((-0.130234f * a.x - 0.0954105f) * a.x + 1.00712f) * a.x - 0.00001203333f);
}

float getReflectionLobeHalfAngle(float roughness)
{
    const float energy = 0.75f;
    return reflectionAtanFast(((roughness * roughness) * energy) / (1.0f - energy));
}

// Eberly's degree-1 approximation for the saturated dot product, as used by
// UE FastMathThirdParty.ush (Shader Fast Math Lib, MIT, copyright 2014 Michal Drobot).
float reflectionNormalAngle(float normalDot)
{
    float x = saturate(normalDot);
    float result = -0.156583f * x + (0.5f * kLumenProbePi);
    return result * sqrt(1.0f - x);
}

float3 tonemapReflectionDisocclusion(float3 lighting, float disocclusionFactor)
{
    return lighting / (1.0f + disocclusionFactor * WorkingColor.luminance(lighting));
}

float3 inverseTonemapReflectionDisocclusion(float3 lighting, float disocclusionFactor)
{
    return lighting / (1.0f - disocclusionFactor * WorkingColor.luminance(lighting));
}

// UE 5.7.4 LumenReflectionDenoiserSpatial.usf, full-resolution opaque permutation.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionBilateralMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionBilateralParams params = getParameters<LumenReflectionBilateralParams>();
    uint2 pixel = thread.xy;
    uint2 extent = uint2(params.width, params.height);
    if (any(pixel >= extent)) { return; }

    let lightingAndMoment = resolveUniform(params.currentLighting);
    float4 center = lightingAndMoment.Load(int3(pixel, 0));
    float3 specularLighting = center.rgb;
    if (isReflectionLightingValid(specularLighting)) {
        center.rgb = reflectionRGBToDenoiserSpace(center.rgb);
        float centerLuminance = WorkingColor.luminance(center.rgb);
        float centerStdDev = sqrt(max(center.w - centerLuminance * centerLuminance, 0.0f));
        float roughness = resolveUniform(params.specular).Load(int3(pixel, 0)).w;

        float disocclusionFactor = 0.0f;
        float numFramesAccumulated = resolveUniform(params.currentAge).Load(int3(pixel, 0)) * 32.0f;
        uint maxFramesAccumulated = uint(getReflectionMaxFramesAccumulated(roughness));
        if (maxFramesAccumulated > 1u) {
            disocclusionFactor = numFramesAccumulated > 1.0f ? 0.0f : 1.0f;
        }

        float3 specularLightingSum = tonemapReflectionDisocclusion(center.rgb, disocclusionFactor);
        float specularLightingWeightSum = 1.0f;
        float kernelRadiusInPixels = kReflectionBilateralKernelRadius * saturate(roughness * 8.0f);
        bool filterSpecular = centerStdDev / max(centerLuminance, 0.1f) > 0.5f || disocclusionFactor > 0.0f;

        if (kernelRadiusInPixels > 1.0f && filterSpecular) {
            uint2 randomSeed = lumenRand3DPCG16(int3(pixel, params.frameIndex % 8u)).xy;
            let positions = resolveUniform(params.position);
            let normals = resolveUniform(params.normal);
            float4 position = positions.Load(int3(pixel, 0));
            float3 normal = normalize(normals.Load(int3(pixel, 0)).xyz);
            float3 translatedWorldPosition = position.xyz - params.camera.eye.xyz;
            float4 scenePlane = float4(normal, dot(translatedWorldPosition, normal));
            float lobeHalfAngle = getReflectionLobeHalfAngle(roughness) * lerp(1.0f, 2.0f, disocclusionFactor);

            for (uint neighborIndex = 0u; neighborIndex < kReflectionBilateralNumSamples; ++neighborIndex) {
                float2 neighborOffsetInRect = lumenHammersley16(neighborIndex, kReflectionBilateralNumSamples, randomSeed);
                float2 neighborOffset = uniformSampleDiskConcentric(neighborOffsetInRect) * kernelRadiusInPixels;
                int2 neighborPixel = int2(float2(pixel) + neighborOffset);
                if (all(neighborPixel >= 0) && all(neighborPixel < int2(extent))) {
                    float3 neighborLighting = lightingAndMoment.Load(int3(neighborPixel, 0)).rgb;
                    if (isReflectionLightingValid(neighborLighting)) {
                        float3 neighborTranslatedWorldPosition = positions.Load(int3(neighborPixel, 0)).xyz - params.camera.eye.xyz;
                        float planeDistance = abs(dot(float4(neighborTranslatedWorldPosition, -1.0f), scenePlane));
                        float relativeDepthDifference = planeDistance / position.w;
                        float depthWeight = exp2(-kReflectionBilateralDepthWeightScale *
                            (relativeDepthDifference * relativeDepthDifference));

                        float3 neighborNormal = normalize(normals.Load(int3(neighborPixel, 0)).xyz);
                        float angleBetweenNormals = reflectionNormalAngle(dot(scenePlane.xyz, neighborNormal));
                        float normalWeight = 1.0f - saturate(angleBetweenNormals / (lobeHalfAngle + 0.01f));

                        neighborLighting = reflectionRGBToDenoiserSpace(neighborLighting);
                        float luminanceDelta = abs(centerLuminance - WorkingColor.luminance(neighborLighting));
                        float luminanceWeight = exp2(-luminanceDelta / max(centerStdDev, 0.001f));
                        luminanceWeight = lerp(luminanceWeight, 1.0f, disocclusionFactor);

                        float specularNeighborWeight = depthWeight * normalWeight * luminanceWeight;
                        specularLightingSum += tonemapReflectionDisocclusion(neighborLighting, disocclusionFactor) *
                            specularNeighborWeight;
                        specularLightingWeightSum += specularNeighborWeight;
                    }
                }
            }
        }

        specularLighting = inverseTonemapReflectionDisocclusion(
            specularLightingSum / specularLightingWeightSum, disocclusionFactor);
        specularLighting = reflectionDenoiserSpaceToRGB(specularLighting);
    }

    float fade = resolveUniform(params.temporalReflections).Load(int3(pixel, 0)).a;
    resolveUniform(params.bilateralReflections)[pixel] = float4(specularLighting, fade);
}
