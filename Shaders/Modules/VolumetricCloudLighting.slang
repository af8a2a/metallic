#language slang 2026
module VolumetricCloudLighting;

import AtmosphereTypes;
import Clouds;

namespace Metallic.Clouds
{
// UE 5.7.4 m_SimpleVolumetricCloud_Inst, default non-storm material.
// Each lane of an order vector is a scattering octave of the grey medium.
public static const uint kCloudShadowSampleCount = 10u;
public static const float kCloudMaterialAmbientOcclusion = 0.5f;
static const float kCloudPi = 3.14159265358979323846f;

float henyeyGreensteinPhase(float g, float cosTheta)
{
    float numer = 1.0f - g * g;
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return numer / (4.0f * kCloudPi * denom * sqrt(denom));
}

public float3 cloudPhaseOrders(float phaseCosTheta, float phaseG)
{
    // VolumetricCloud.usf SamplePhaseFunction uses the inward view-ray convention.
    float phaseA = henyeyGreensteinPhase(clamp(phaseG, -0.999f, 0.999f), -phaseCosTheta);
    float phaseB = henyeyGreensteinPhase(0.166667f, -phaseCosTheta);
    float phase = phaseA + 0.575f * (phaseB - phaseA);
    float isotropic = 1.0f / (4.0f * kCloudPi);
    float msPhaseFactor = 0.18f;
    return float3(phase, lerp(isotropic, phase, msPhaseFactor), lerp(isotropic, phase, msPhaseFactor * msPhaseFactor));
}

// VolumetricCloud.usf SetupParticipatingMediaContext. Each successive octave
// multiplies the preceding coefficient by the squared previous factor.
float3 cloudCoefficientOrders(float base, float msFactor)
{
    float3 coefficients;
    coefficients[0] = base;
    [unroll]
    for (int ms = 1; ms < 3; ++ms) {
        coefficients[ms] = coefficients[ms - 1] * msFactor;
        msFactor *= msFactor;
    }
    return coefficients;
}

public float3 cloudScatteringOrders(float albedo, float extinctionPerMeter)
{
    return cloudCoefficientOrders(albedo * extinctionPerMeter, 0.666667f);
}

public float3 cloudExtinctionOrders(float extinctionPerMeter)
{
    return cloudCoefficientOrders(extinctionPerMeter, 0.25f);
}

public float3 cloudShadowTransmittanceOrders(GPUAtmosphereParameters p, float3 positionKm, float3 lightDir,
    uint sampleCount)
{
    // VolumetricCloud.usf, MATERIAL_VOLUMETRIC_ADVANCED_RAYMARCH_VOLUME_SHADOW.
    const float shadowLengthKm = 15.0f;
    float invShadowStepCount = 1.0f / float(sampleCount);
    float previousNormT = 0.0f;
    float3 extinctionAcc = 0.0f;
    [loop]
    for (float shadowT = invShadowStepCount; shadowT <= 1.00001f; shadowT += invShadowStepCount) {
        float currentNormT = shadowT * shadowT;
        float deltaNormT = currentNormT - previousNormT;
        float shadowSampleDistance = shadowLengthKm * (previousNormT + deltaNormT * 0.5f);
        float3 sampleKm = positionKm + lightDir * shadowSampleDistance;
        previousNormT = currentNormT;

        float h = saturate(cloudNormalizedHeight(p, sampleKm));
        float3 cameraRelativeKm = sampleKm - p.observerPlanetBottom.xyz;
        float4 conservative = cloudConservativeDensity(p, cameraRelativeKm, h);
        if (conservative.x <= 0.0f) { continue; }
        if (h <= 0.0f || h >= 1.0f) { break; }
        if (cloudMaterialContainsHeight(h)) {
            float extinction = cloudMaterialExtinction(p, conservative, cameraRelativeKm);
            extinctionAcc += cloudExtinctionOrders(extinction) * deltaNormT;
        }
    }
    return exp(-extinctionAcc * (shadowLengthKm * 1000.0f));
}

// Analytic integral of one scattering order across a single path segment.
public float3 integrateCloudScattering(float3 extinctionPerMeter, float3 scatteredLuminancePerMeter,
    float stepMeters, out float3 segmentTransmittance)
{
    float3 safeExtinction = max(0.000001f, extinctionPerMeter);
    segmentTransmittance = exp(-safeExtinction * stepMeters);
    return (scatteredLuminancePerMeter - scatteredLuminancePerMeter * segmentTransmittance) / safeExtinction;
}
} // namespace Metallic.Clouds
