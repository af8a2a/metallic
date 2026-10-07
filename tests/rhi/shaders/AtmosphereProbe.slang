import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

// The nested production ABI remains 64 B. The probe adds a read-only input
// aerial span; the production aerial field holds 32 float4 output records.
struct AtmosphereProbeParams
{
    AtmospherePrecomputeParams resources;
    BufferSpan<float4> inputAerial;
    uint reserved;
};
[[vk::push_constant]] ConstantBuffer<AtmosphereProbeParams> gProbe;
#define gParams gProbe.resources

[numthreads(1, 1, 1)]
void atmosphereProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0; index < 32; ++index) { gParams.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0)); }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float3 surface = float3(0.0, p.observerPlanetBottom.w + 0.002, 0.0);
    float3 up = float3(0.0, 1.0, 0.0);
    gParams.aerial.store(0, float4(solarIrradiance(p, resolveUniform(gParams.transmittance), surface, 0), 1.0));
    gParams.aerial.store(1, float4(solarIrradiance(p, resolveUniform(gParams.transmittance), surface + up * 25.0, 0), 1.0));
    GPUAtmosphereParameters night = p;
    night.sunDirectionRadius.xyz = -up;
    gParams.aerial.store(2, float4(solarIrradiance(night, resolveUniform(gParams.transmittance), surface, 0), 1.0));
    gParams.aerial.store(3, float4(visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), up, false), 1.0));
    AtmosphereSegment near = segmentIntegration(p, resolveUniform(gParams.transmittance), resolveUniform(gParams.multiScattering), surface, up, 1.0);
    gParams.aerial.store(4, float4(near.radiance, 1.0));
    gParams.aerial.store(5, float4(near.transmittance, 1.0));
    AtmosphereSegment far = segmentIntegration(p, resolveUniform(gParams.transmittance), resolveUniform(gParams.multiScattering), surface, up, 50.0);
    gParams.aerial.store(6, float4(far.radiance, 1.0));
    gParams.aerial.store(7, float4(far.transmittance, 1.0));
    GPUAtmosphereParameters vacuum = p;
    vacuum.rayleighScaleHeight.xyz = float3(0.0); vacuum.mieScatteringScaleHeight.xyz = float3(0.0);
    vacuum.mieExtinctionAnisotropy.xyz = float3(0.0); vacuum.ozoneCenter.xyz = float3(0.0);
    AtmosphereSegment empty = segmentIntegration(vacuum, resolveUniform(gParams.transmittance), resolveUniform(gParams.multiScattering), surface, up, 50.0);
    gParams.aerial.store(8, float4(empty.radiance, 1.0));
    gParams.aerial.store(9, float4(empty.transmittance, 1.0));
    float3 orbital = up * (p.observerWorldTop.w + 50.0);
    gParams.aerial.store(10, float4(integrateTransmittance(p, orbital, normalize(float3(1.0, -0.15, 0.0))), 1.0));
    gParams.aerial.store(11, float4(integrateTransmittance(p, orbital, up), 1.0));
    gParams.aerial.store(12, float4(sampleMulti(p, resolveUniform(gParams.multiScattering), surface, up), 1.0));
    gParams.aerial.store(13, float4(spectralToWorking(p.sunIrradianceEnabled.xyz), 1.0));
    gParams.aerial.store(14, float4(integrateTransmittance(p, surface, up), 1.0));
    gParams.aerial.store(15, float4(transmittanceToSource(p, resolveUniform(gParams.transmittance), surface, up), 1.0));
    AtmosphereSegment sampledNear = aerialPerspective(p, gProbe.inputAerial, p.observerWorldTop.xyz + up * 1000.0);
    AtmosphereSegment sampledFar = aerialPerspective(p, gProbe.inputAerial, p.observerWorldTop.xyz + up * 50000.0);
    gParams.aerial.store(16, float4(sampledNear.radiance, 1.0));
    gParams.aerial.store(17, float4(sampledNear.transmittance, 1.0));
    gParams.aerial.store(18, float4(sampledFar.radiance, 1.0));
    gParams.aerial.store(19, float4(sampledFar.transmittance, 1.0));
    float3 sourceDirection = p.sunDirectionRadius.xyz;
    gParams.aerial.store(20, float4(visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), sourceDirection, true), 1.0));
    gParams.aerial.store(21, float4(visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), sourceDirection, false), 1.0));
    gParams.aerial.store(22, float4(sampleLinear(resolveUniform(gParams.sourceMip), directionLatLong(sourceDirection), true).xyz, 1.0));
    AtmosphereSegment exactMetres = segmentIntegration(p, resolveUniform(gParams.transmittance), resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, up, 0.01);
    AtmosphereSegment sampledMetres = aerialPerspective(p, gProbe.inputAerial, p.observerWorldTop.xyz + up * 10.0);
    gParams.aerial.store(23, float4(exactMetres.radiance, 1.0));
    gParams.aerial.store(24, float4(exactMetres.transmittance, 1.0));
    gParams.aerial.store(25, float4(sampledMetres.radiance, 1.0));
    gParams.aerial.store(26, float4(sampledMetres.transmittance, 1.0));
    AtmosphereSegment zero = aerialPerspective(p, gProbe.inputAerial, p.observerWorldTop.xyz);
    gParams.aerial.store(27, float4(zero.radiance, 1.0));
    gParams.aerial.store(28, float4(zero.transmittance, 1.0));
}

[numthreads(1, 1, 1)]
void atmosphereLimbProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0; index < 32; ++index) { gParams.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0)); }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float3 above = normalize(float3(1.0, 0.001, 0.0));
    float3 below = normalize(float3(1.0, -0.004, 0.0));
    float3 upperDisk = visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), above, true) -
        visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), above, false);
    float3 lowerDisk = visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), below, true) -
        visibleSky(p, resolveUniform(gParams.skyView), resolveUniform(gParams.transmittance), below, false);
    gParams.aerial.store(0, float4(upperDisk, 1.0));
    gParams.aerial.store(1, float4(lowerDisk, 1.0));
    gParams.aerial.store(2, float4(solarIrradiance(p, resolveUniform(gParams.transmittance), p.observerPlanetBottom.xyz, 0), 1.0));
    gParams.aerial.store(3, float4(transmittanceToSource(p, resolveUniform(gParams.transmittance), p.observerPlanetBottom.xyz, above), 1.0));
    gParams.aerial.store(4, float4(transmittanceToSource(p, resolveUniform(gParams.transmittance), p.observerPlanetBottom.xyz, below), 1.0));
}

[numthreads(1, 1, 1)]
void atmosphereHorizonProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0; index < 32; ++index) { gParams.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0)); }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float offsets[5] = {-0.01, -0.003, -0.001, 0.001, 0.003};
    float horizon = planetLimbZenith(p);
    for (uint index = 0; index < 5; ++index) {
        float3 direction = directionForZenithAzimuth(p, horizon + offsets[index], 0.0);
        float3 actual = sampleSkyView(p, resolveUniform(gParams.skyView), direction);
        SpectralSegment reference = integrateEnvironment(p, resolveUniform(gParams.transmittance),
            resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, 1e9, true);
        float3 roundtrip = skyDirectionFromUV(p, skyDirectionUV(p, direction));
        gParams.aerial.store(index, float4(actual, 1.0));
        gParams.aerial.store(index + 5u, float4(spectralToWorking(reference.radiance), 1.0));
        gParams.aerial.store(index + 10u, float4(length(roundtrip - direction), abs(length(direction) - 1.0), horizon, 1.0));
    }
}

[numthreads(8, 8, 1)]
void atmosphereHorizonImageMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float2 uv = (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy);
    float zenith = planetLimbZenith(p) + (uv.y - 0.5) * 0.2;
    float azimuth = (uv.x - 0.5) * 0.6;
    float3 direction = directionForZenithAzimuth(p, zenith, azimuth);
    float3 radiance;
    if (gParams.dimensions.w == 0u) {
        radiance = sampleSkyView(p, resolveUniform(gParams.skyView), direction);
    } else {
        SpectralSegment reference = integrateEnvironment(p, resolveUniform(gParams.transmittance),
            resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, 1e9, true);
        radiance = spectralToWorking(reference.radiance);
    }
    gParams.aerial.store(id.y * gParams.dimensions.x + id.x, float4(radiance, 1.0));
}
