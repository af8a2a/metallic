#language slang 2026
implementing Lighting;

import Atmosphere;
import AtmosphereTypes;
import Core;
using Metallic;

namespace Metallic.Lighting
{
internal float celestialDiskSineHalf(float angularRadius)
{
    // Device fast trigonometry can round very small sin arguments to zero.
    // For radius <=1e-3, sin(radius/2) ~= radius/2 has relative error <=4.2e-8,
    // below FP32 epsilon, and retains every normal positive tiny radius.
    return angularRadius <= 1e-3 ? angularRadius * 0.5 : sin(0.5 * angularRadius);
}

public float celestialDiskSolidAngle(float angularRadius)
{
    // Stable for small physical radii, where subtracting cos(radius) from one
    // loses significant bits. Uniform cone density is 1 / solidAngle.
    float sineHalf = celestialDiskSineHalf(angularRadius);
    return 4.0 * 3.14159265358979323846 * sineHalf * sineHalf;
}

public float3 sampleCelestialDiskDirection(float3 sourceDirection, float angularRadius, float2 uniform)
{
    float3 axis = normalize(sourceDirection);
    float3 reference = abs(axis.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(reference, axis)), top = cross(axis, right);
    float sineHalf = celestialDiskSineHalf(angularRadius);
    float offset = saturate(uniform.x) * (2.0 * sineHalf * sineHalf);
    float cosine = 1.0 - offset;
    float sine = sqrt(max(offset * (2.0 - offset), 0.0));
    float azimuth = uniform.y * (2.0 * 3.14159265358979323846);
    return normalize(axis * cosine + (right * cos(azimuth) + top * sin(azimuth)) * sine);
}

public PunctualLightSample samplePhysicalCelestialDisk(GPUAtmosphereParameters atmosphere,
    Texture2D<float4> transmittance, uint slot, float3 positionKm, float2 uniform)
{
    PunctualLightSample result;
    float4 source = slot == 0u ? atmosphere.sunDirectionRadius : atmosphere.moonDirectionRadius;
    result.distance = 1e30;
    result.direction = source.xyz;
    if (source.w <= 0.0) {
        result.illuminance = Metallic.Atmosphere::solarIrradiance(atmosphere, transmittance, positionKm, slot);
        return result;
    }
    result.direction = sampleCelestialDiskDirection(source.xyz, source.w, uniform);
    // The result is Li / p(direction), before the surface projected BSDF.
    // A uniform disk gives E * 2/(1+cos(radius)); its cosine-weighted integral
    // remains E. Moon Li contains the resolved terminator and Lambert profile.
    // Extinction and planet/cloud visibility use this actual sampled ray.
    result.illuminance = Metallic.Atmosphere::sourceDiskSolidAngleEstimate(atmosphere, transmittance,
        slot, positionKm, result.direction);
    return result;
}

public PunctualLightSample samplePathCelestialLightAtPosition(StructuredBuffer<GPUCelestialLight> lights,
    uint slot, float3 worldPosition, inout uint seed)
{
    let result = sampleCelestialLight(lights, slot);
    if (!any(result.illuminance > 0.0)) { return result; }
    let resources = getResourceParameters<SceneResourceParameters>().environment;
    GPUAtmosphereParameters atmosphere;
    if (resources.transmittance.index == 0xffffffffu || !loadPhysicalAtmosphere(resources, atmosphere)) {
        // HDRI compatibility retains both the authored delta source and RNG.
        return sampleCelestialLightAtPosition(lights, slot, worldPosition);
    }
    float2 uniform = float2(punctualSamplingRandom(seed), punctualSamplingRandom(seed));
    return samplePhysicalCelestialDisk(atmosphere, resolveUniform(resources.transmittance), slot,
        Metallic.Atmosphere::worldToAtmosphereKm(atmosphere, worldPosition), uniform);
}

public float3 physicalDeltaMissDisks(GPUAtmosphereParameters atmosphere, Texture2D<float4> transmittance,
    float3 direction, float3 positionKm)
{
    if (atmosphere.settings.y <= 0.5) { return float3(0.0); }
    return Metallic.Atmosphere::visibleSourceDisks(atmosphere, transmittance, positionKm, direction);
}

public float3 physicalDeltaMissDisks(EnvironmentResourceParameters resources, float3 direction, float3 worldOrigin)
{
    GPUAtmosphereParameters atmosphere;
    if (resources.transmittance.index == 0xffffffffu || !loadPhysicalAtmosphere(resources, atmosphere)) {
        return float3(0.0);
    }
    // An ideal BSDF delta cannot overlap the finite-light NEE estimator; its
    // projected evaluator has no ordinary solid-angle density. Restore disks
    // on this miss only, with weight one, and retain ordinary environment light.
    return physicalDeltaMissDisks(atmosphere, resolveUniform(resources.transmittance), direction,
        Metallic.Atmosphere::worldToAtmosphereKm(atmosphere, worldOrigin));
}
} // namespace Metallic.Lighting
