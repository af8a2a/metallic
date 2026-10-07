#language slang 2026
implementing Lighting;

import Core;
import Atmosphere;
using Metallic;

namespace Metallic.Lighting
{
public bool loadPhysicalAtmosphere(EnvironmentResourceParameters resources, out GPUAtmosphereParameters parameters)
{
    parameters = {};
    if (resources.atmosphereParameters.index == 0xffffffffu) { return false; }
    let buffer = resolveBuffer<StructuredBuffer<GPUAtmosphereParameters>>(resources.atmosphereParameters);
    uint count, stride;
    buffer.GetDimensions(count, stride);
    if (count == 0u) { return false; }
    parameters = buffer[0];
    return parameters.settings.y > 0.5;
}

public PunctualLightSample sampleCelestialLightAtPosition(StructuredBuffer<GPUCelestialLight> lights,
    uint slot, float3 worldPosition)
{
    PunctualLightSample result = sampleCelestialLight(lights, slot);
    GPUAtmosphereParameters atmosphere;
    let resources = getResourceParameters<SceneResourceParameters>().environment;
    if (resources.transmittance.index != 0xffffffffu && loadPhysicalAtmosphere(resources, atmosphere)) {
        result.illuminance = Metallic.Atmosphere::solarIrradiance(atmosphere,
            resolveUniform(resources.transmittance),
            Metallic.Atmosphere::worldToAtmosphereKm(atmosphere, worldPosition), slot);
    }
    return result;
}

public float3 physicalVisibleEnvironment(EnvironmentResourceParameters resources, float3 direction, float3 primaryWorldOrigin)
{
    GPUAtmosphereParameters atmosphere;
    if (resources.skyView.index == 0xffffffffu || resources.transmittance.index == 0xffffffffu ||
        !loadPhysicalAtmosphere(resources, atmosphere)) { return float3(0.0); }
    if (any(primaryWorldOrigin != atmosphere.observerWorldTop.xyz)) {
        if (resources.multiScattering.index == 0xffffffffu) { return float3(0.0); }
        float3 positionKm = Metallic.Atmosphere::worldToAtmosphereKm(atmosphere, primaryWorldOrigin);
        let integrated = Metallic.Atmosphere::integrateEnvironment(atmosphere, resolveUniform(resources.transmittance),
            resolveUniform(resources.multiScattering), positionKm, direction, 1e9, true);
        return max(Metallic.Atmosphere::spectralToWorking(integrated.radiance), 0.0) +
            Metallic.Atmosphere::visibleSourceDisks(atmosphere, resolveUniform(resources.transmittance), positionKm, direction);
    }
    return Metallic.Atmosphere::visibleSky(atmosphere, resolveUniform(resources.skyView),
        resolveUniform(resources.transmittance), direction, true);
}

public bool physicalEnvironmentEnabled(EnvironmentResourceParameters resources)
{
    GPUAtmosphereParameters atmosphere;
    return loadPhysicalAtmosphere(resources, atmosphere);
}

public float physicalNrdInversePreExposure(EnvironmentResourceParameters resources)
{
    GPUAtmosphereParameters atmosphere;
    if (!loadPhysicalAtmosphere(resources, atmosphere)) { return 1.0; }
    float3 spectrum = max(atmosphere.sunIrradianceEnabled.xyz, 0.0) * atmosphere.sunIrradianceEnabled.w +
        max(atmosphere.moonIrradianceEnabled.xyz, 0.0) * atmosphere.moonIrradianceEnabled.w;
    // Absolute transform coefficients bound every attenuated three-band
    // spectrum, including cases where extinction removes a negative RGB term.
    float3 irradianceBound =
        abs(WorkingColor::toLinearRec709(Metallic.Atmosphere::spectralToWorking(float3(1.0, 0.0, 0.0)))) * spectrum.x +
        abs(WorkingColor::toLinearRec709(Metallic.Atmosphere::spectralToWorking(float3(0.0, 1.0, 0.0)))) * spectrum.y +
        abs(WorkingColor::toLinearRec709(Metallic.Atmosphere::spectralToWorking(float3(0.0, 0.0, 1.0)))) * spectrum.z;
    float peakIrradiance = max(irradianceBound.x, max(irradianceBound.y, irradianceBound.z));
    // SceneRTXDI.evaluateDeltaLight uses a normalized power lobe with exponent
    // <=128, so nDotL * (power+2)/(2*pi) * pow(nDotH,power) <=130/(2*pi).
    // RELAX also stores second moments in FP16; External/NRD/README.md's HDR
    // guidance recommends RGB <=250. A target of 64 leaves fourfold radiance
    // headroom, keeping its square below 65504. Update this bound with the BRDF.
    const float peakBsdf = 130.0 / (2.0 * 3.14159265358979323846);
    return max(1.0, peakIrradiance * peakBsdf / 64.0);
}

public Metallic.Atmosphere::AtmosphereSegment physicalAerialSegment(EnvironmentResourceParameters resources,
    float3 worldEndpoint, float3 primaryWorldOrigin)
{
    Metallic.Atmosphere::AtmosphereSegment result;
    result.radiance = float3(0.0);
    result.transmittance = float3(1.0);
    GPUAtmosphereParameters atmosphere;
    if (!loadPhysicalAtmosphere(resources, atmosphere)) { return result; }
    float3 deltaKm = (worldEndpoint - primaryWorldOrigin) * 0.001;
    float distanceKm = length(deltaKm);
    if (distanceKm < 1e-7) { return result; }
    bool differentOrigin = any(primaryWorldOrigin != atmosphere.observerWorldTop.xyz);
    if ((differentOrigin || distanceKm > atmosphere.settings.x) && resources.transmittance.index != 0xffffffffu &&
        resources.multiScattering.index != 0xffffffffu) {
        // Orthographic pixels have their own primary origins. A camera-centred
        // spherical LUT cannot represent those parallel segments accurately.
        // A far observer can traverse vacuum before entering the atmosphere.
        // The integral clips the actual camera segment to the shell; clamping
        // the LUT distance would miss the entire medium in that case.
        return Metallic.Atmosphere::segmentIntegration(atmosphere, resolveUniform(resources.transmittance),
            resolveUniform(resources.multiScattering), Metallic.Atmosphere::worldToAtmosphereKm(atmosphere, primaryWorldOrigin),
            deltaKm / distanceKm, distanceKm);
    }
    if (resources.aerialPerspective.index == 0xffffffffu) { return result; }
    let buffer = resolveBuffer<StructuredBuffer<float4>>(resources.aerialPerspective);
    uint count, stride;
    buffer.GetDimensions(count, stride);
    if (count < 32u * 32u * 32u * 2u) { return result; }
    BufferSpan<float4> span = {resources.aerialPerspective, 0u, count};
    return Metallic.Atmosphere::aerialPerspective(atmosphere, span, worldEndpoint);
}

public float3 applyPrimaryAerialPerspective(float3 radiance, float3 worldEndpoint, float3 primaryWorldOrigin)
{
    let segment = physicalAerialSegment(getResourceParameters<SceneResourceParameters>().environment, worldEndpoint, primaryWorldOrigin);
    return segment.transmittance * radiance + segment.radiance;
}
} // namespace Metallic.Lighting
