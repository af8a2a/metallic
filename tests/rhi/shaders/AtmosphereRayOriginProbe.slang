module AtmosphereRayOriginProbe;

import Atmosphere;
import Lighting;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Lighting;

struct AtmosphereProbeParams
{
    AtmospherePrecomputeParams resources;
    BufferSpan<float4> inputAerial;
    uint reserved;
};
[[vk::push_constant]] ConstantBuffer<AtmosphereProbeParams> gProbe;

EnvironmentResourceParameters atmosphereProbeEnvironment()
{
    EnvironmentResourceParameters environment = {};
    environment.atmosphereParameters.index = gProbe.resources.parameters.resource.index;
    environment.transmittance = gProbe.resources.transmittance;
    environment.multiScattering = gProbe.resources.multiScattering;
    environment.skyView = gProbe.resources.skyView;
    environment.aerialPerspective = gProbe.inputAerial.resource;
    environment.cloudShadow.index = 0xffffffffu;
    return environment;
}

// Exercise the production consumer wrapper, including its named resource
// handles. Parallel orthographic rays have origins displaced from camera eye.
[numthreads(1, 1, 1)]
void atmosphereRayOriginProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0; index < 32; ++index) {
        gProbe.resources.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0));
    }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    EnvironmentResourceParameters environment = atmosphereProbeEnvironment();
    float3 direction = float3(1.0, 0.0, 0.0);
    for (uint ray = 0; ray < 2; ++ray) {
        float3 origin = p.observerWorldTop.xyz + float3(0.0, 20000.0 * float(ray + 1u), 0.0);
        float3 endpoint = origin + direction * 50000.0;
        AtmosphereSegment actual = physicalAerialSegment(environment, endpoint, origin);
        AtmosphereSegment reference = segmentIntegration(p, resolveUniform(environment.transmittance),
            resolveUniform(environment.multiScattering), worldToAtmosphereKm(p, origin), direction, 50.0);
        float3 wrongDeltaKm = (endpoint - p.observerWorldTop.xyz) * 0.001;
        float wrongDistanceKm = length(wrongDeltaKm);
        AtmosphereSegment eyeReference = segmentIntegration(p, resolveUniform(environment.transmittance),
            resolveUniform(environment.multiScattering), p.observerPlanetBottom.xyz,
            wrongDeltaKm / wrongDistanceKm, wrongDistanceKm);
        gProbe.resources.aerial.store(ray * 4u, float4(actual.radiance, 1.0));
        gProbe.resources.aerial.store(ray * 4u + 1u, float4(actual.transmittance, 1.0));
        gProbe.resources.aerial.store(ray * 4u + 2u, float4(reference.radiance, 1.0));
        gProbe.resources.aerial.store(ray * 4u + 3u, float4(reference.transmittance, 1.0));
        gProbe.resources.aerial.store(8u + ray * 2u, float4(eyeReference.radiance, 1.0));
        gProbe.resources.aerial.store(9u + ray * 2u, float4(eyeReference.transmittance, 1.0));
        gProbe.resources.aerial.store(18u + ray, float4(origin, 1.0));
    }
    float3 offsetOrigin = p.observerWorldTop.xyz + float3(0.0, 20000.0, 0.0);
    AtmosphereSegment zero = physicalAerialSegment(environment, offsetOrigin, offsetOrigin);
    gProbe.resources.aerial.store(12, float4(zero.radiance, 1.0));
    gProbe.resources.aerial.store(13, float4(zero.transmittance, 1.0));
    float3 perspectiveEndpoint = p.observerWorldTop.xyz + float3(0.0, 1000.0, 0.0);
    AtmosphereSegment perspective = physicalAerialSegment(environment, perspectiveEndpoint, p.observerWorldTop.xyz);
    AtmosphereSegment lutReference = aerialPerspective(p, gProbe.inputAerial, perspectiveEndpoint);
    gProbe.resources.aerial.store(14, float4(perspective.radiance, 1.0));
    gProbe.resources.aerial.store(15, float4(perspective.transmittance, 1.0));
    gProbe.resources.aerial.store(16, float4(lutReference.radiance, 1.0));
    gProbe.resources.aerial.store(17, float4(lutReference.transmittance, 1.0));
    float3 offsetPosition = worldToAtmosphereKm(p, offsetOrigin);
    SpectralSegment skyReference = integrateEnvironment(p, resolveUniform(environment.transmittance),
        resolveUniform(environment.multiScattering), offsetPosition, direction, 1e9, true);
    float3 offsetSky = physicalVisibleEnvironment(environment, direction, offsetOrigin);
    float3 eyeSky = physicalVisibleEnvironment(environment, direction, p.observerWorldTop.xyz);
    gProbe.resources.aerial.store(20, float4(offsetSky, 1.0));
    gProbe.resources.aerial.store(21, float4(max(spectralToWorking(skyReference.radiance), 0.0), 1.0));
    gProbe.resources.aerial.store(22, float4(eyeSky, 1.0));
    float3 sourceDirection = p.sunDirectionRadius.xyz;
    SpectralSegment sourceSkyReference = integrateEnvironment(p, resolveUniform(environment.transmittance),
        resolveUniform(environment.multiScattering), offsetPosition, sourceDirection, 1e9, true);
    float3 sourceReference = max(spectralToWorking(sourceSkyReference.radiance), 0.0) +
        visibleSourceDisks(p, resolveUniform(environment.transmittance), offsetPosition, sourceDirection);
    gProbe.resources.aerial.store(23, float4(physicalVisibleEnvironment(environment, sourceDirection, offsetOrigin), 1.0));
    gProbe.resources.aerial.store(24, float4(sourceReference, 1.0));
    gProbe.resources.aerial.store(25, float4(physicalVisibleEnvironment(environment, sourceDirection, p.observerWorldTop.xyz), 1.0));
}

[numthreads(1, 1, 1)]
void atmosphereAerialStartDepthProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    EnvironmentResourceParameters environment = atmosphereProbeEnvironment();
    float distancesMeters[6] = {20.0, 100.0, 101.0, 102.0, 1000.0, 2000.0};
    float3 direction = float3(0.0, 1.0, 0.0);
    for (uint ray = 0; ray < 2; ++ray) {
        // The eye takes the LUT path while an offset orthographic origin needs direct integration.
        float3 origin = p.observerWorldTop.xyz + direction * (2000.0 * float(ray));
        for (uint sampleIndex = 0; sampleIndex < 6; ++sampleIndex) {
            float distanceMeters = distancesMeters[sampleIndex];
            float3 endpoint = origin + direction * distanceMeters;
            AtmosphereSegment actual = physicalAerialSegment(environment, endpoint, origin);
            uint index = sampleIndex * 4u + ray * 2u;
            gProbe.resources.aerial.store(index, float4(actual.radiance, 1.0));
            gProbe.resources.aerial.store(index + 1u, float4(actual.transmittance, 1.0));
            if (sampleIndex >= 4u) {
                // Independent reference for the authored 100 m start, measured along this ray.
                float3 start = worldToAtmosphereKm(p, origin) + direction * 0.1;
                AtmosphereSegment reference = segmentIntegration(p, resolveUniform(environment.transmittance),
                    resolveUniform(environment.multiScattering), start, direction, distanceMeters * 0.001 - 0.1);
                gProbe.resources.aerial.store(index + 8u, float4(reference.radiance, 1.0));
                gProbe.resources.aerial.store(index + 9u, float4(reference.transmittance, 1.0));
            }
        }
    }
}

[numthreads(1, 1, 1)]
void atmosphereOffsetLimbProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0; index < 32; ++index) {
        gProbe.resources.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0));
    }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    EnvironmentResourceParameters environment = atmosphereProbeEnvironment();
    float3 origin = p.observerWorldTop.xyz + float3(0.0, 20000.0, 0.0);
    float3 position = worldToAtmosphereKm(p, origin);
    GPUAtmosphereParameters offsetParameters = p;
    offsetParameters.observerPlanetBottom.xyz = position;
    float horizon = planetLimbZenith(offsetParameters);
    float3 upperDirection = float3(sin(horizon - 0.002), cos(horizon - 0.002), 0.0);
    float3 lowerDirection = float3(sin(horizon + 0.002), cos(horizon + 0.002), 0.0);
    for (uint ray = 0; ray < 2; ++ray) {
        float3 direction = ray == 0u ? upperDirection : lowerDirection;
        SpectralSegment skyReference = integrateEnvironment(p, resolveUniform(environment.transmittance),
            resolveUniform(environment.multiScattering), position, direction, 1e9, true);
        float3 expectedSky = max(spectralToWorking(skyReference.radiance), 0.0);
        float3 actual = physicalVisibleEnvironment(environment, direction, origin);
        gProbe.resources.aerial.store(ray, float4(actual - expectedSky, 1.0));
    }
    gProbe.resources.aerial.store(2, float4(visibleSourceDisks(p,
        resolveUniform(environment.transmittance), position, upperDirection), 1.0));
    gProbe.resources.aerial.store(3, float4(visibleSourceDisks(p,
        resolveUniform(environment.transmittance), p.observerPlanetBottom.xyz, upperDirection), 1.0));
    gProbe.resources.aerial.store(4, float4(transmittanceToSource(p,
        resolveUniform(environment.transmittance), position, lowerDirection), 1.0));
    gProbe.resources.aerial.store(5, float4(transmittanceToSource(p,
        resolveUniform(environment.transmittance), position, upperDirection), 1.0));
    gProbe.resources.aerial.store(6, float4(origin, 1.0));
    gProbe.resources.aerial.store(7, float4(p.sunDirectionRadius.xyz, 1.0));
    gProbe.resources.aerial.store(8, float4(upperDirection, 1.0));
    gProbe.resources.aerial.store(9, float4(lowerDirection, 1.0));
}

// Uniform opaque blockers isolate each celestial contribution in the transport equation.
public struct ProbeAtmosphereShadow : IAtmosphereLightShadow
{
    public uint blockedLights;
    public float factor;

    [mutating] public void selectLight(uint slot, float3 direction, float startDepthKm)
    {
        factor = (blockedLights & (1u << slot)) != 0u ? 0.0 : 1.0;
    }
    public float visibility(float rayDistanceKm) { return factor; }
};

[numthreads(1, 1, 1)]
void atmosphereDirectionalShadowProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0u; index < 32u; ++index) { gProbe.resources.aerial.store(index, float4(0, 0, 0, 1)); }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    // Give the Moon a distinct, measurable spectrum for this transport test.
    p.moonIrradianceEnabled = float4(p.sunIrradianceEnabled.xyz * float3(0.15, 0.3, 0.6), 1.0);
    p.moonDirectionRadius.xyz = normalize(float3(0.0, 0.6, 0.8));
    p.settings.z = 0.1;
    let trans = resolveUniform(gProbe.resources.transmittance);
    let multi = resolveUniform(gProbe.resources.multiScattering);
    float3 direction = normalize(float3(0.9, 0.15, 0.2));
    float depth = 1.0;
    for (uint mask = 0u; mask < 4u; ++mask) {
        ProbeAtmosphereShadow shadow = {mask, 1.0};
        let result = integrateAerialPerspective(p, trans, multi, p.observerPlanetBottom.xyz, direction, depth, shadow);
        gProbe.resources.aerial.store(2u * mask, float4(result.radiance, 1));
        gProbe.resources.aerial.store(2u * mask + 1u, float4(result.transmittance, 1));
    }
    for (uint slot = 0u; slot < 2u; ++slot) {
        // Independent direct-only transport. Occluding this source must subtract exactly this term.
        float3 start = p.observerPlanetBottom.xyz + p.settings.z * direction;
        let single = integrateSpectral(p, trans, multi, start, direction, depth,
            slot == 0u ? p.sunDirectionRadius.xyz : p.moonDirectionRadius.xyz,
            slot == 0u ? p.sunIrradianceEnabled.xyz : p.moonIrradianceEnabled.xyz,
            false, false, false, UnshadowedAtmosphere());
        gProbe.resources.aerial.store(8u + slot, float4(spectralToWorking(single.radiance), 1));
    }
}
