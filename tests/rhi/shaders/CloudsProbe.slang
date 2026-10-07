import Atmosphere;
import Clouds;
import Lighting;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Lighting;

struct CloudsProbeParams
{
    AtmospherePrecomputeParams resources;
    BufferSpan<float4> inputAerial;
    ResourceHandle<Texture2D<float4>> radiance;
};
[[vk::push_constant]] ConstantBuffer<CloudsProbeParams> gProbe;

void store(uint index, float3 value)
{
    gProbe.resources.aerial.store(index, float4(value, 1.0));
}

[numthreads(1, 1, 1)]
void cloudsProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    for (uint index = 0u; index < 47u; ++index) { store(index, float3(0.0)); }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    let trans = resolveUniform(gProbe.resources.transmittance);
    let multi = resolveUniform(gProbe.resources.multiScattering);
    let sky = resolveUniform(gProbe.resources.skyView);
    let shadow = resolveUniform(gProbe.resources.sourceMip);
    let capture = resolveUniform(gProbe.radiance);
    float3 ground = p.cloudShadowCentre.xyz;
    float3 up = normalize(ground);
    float3 sun = p.sunDirectionRadius.xyz;
    GPUAtmosphereParameters clear = p;
    clear.cloudOptics.w = 0.0;
    float cloudT = Metallic.Clouds::transmittance(p, ground, sun, 1e9, 48u);
    store(0, float3(cloudT));
    store(1, float3(cloudShadowTransmittance(p, shadow, ground, 0u)));
    float4 map = sampleLinear(shadow, float2(0.5));
    store(2, map.xyz); store(3, float3(map.w, p.cloudShadowBudget.z, p.cloudShadowBudget.w));
    store(4, solarIrradiance(p, trans, ground, 0u));
    store(5, solarIrradiance(clear, trans, ground, 0u));
    float3 above = up * (p.observerPlanetBottom.w + p.cloudLayer.y + 1.0);
    store(6, solarIrradiance(p, trans, above, 0u));
    store(7, solarIrradiance(clear, trans, above, 0u));
    AtmosphereSegment nearSegment = segmentIntegration(p, trans, multi, ground, up, 1.0);
    AtmosphereSegment farSegment = segmentIntegration(p, trans, multi, ground, up, 8.0);
    store(8, nearSegment.radiance); store(9, nearSegment.transmittance);
    store(10, farSegment.radiance); store(11, farSegment.transmittance);
    AtmosphereSegment lut = aerialPerspective(p, gProbe.inputAerial, p.observerWorldTop.xyz + up * 8000.0);
    store(12, lut.radiance); store(13, lut.transmittance);
    store(14, visibleSky(p, sky, trans, sun, false));
    store(15, visibleSky(p, sky, trans, sun, true));
    store(16, sampleLinear(capture, directionLatLong(sun), true).xyz);
    float3 outside = normalize(ground + float3(100.0, 0.0, 0.0)) * (p.observerPlanetBottom.w + 0.002);
    store(17, float3(Metallic.Clouds::transmittance(p, outside, sun),
        cloudShadowTransmittance(p, shadow, outside, 0u), 0.0));
    store(18, float3(Metallic.Clouds::transmittance(p, above, sun),
        cloudShadowTransmittance(p, shadow, above, 0u), 0.0));
    float3 samplePoint = ground + up * 2.5 + float3(6.0, 0.0, 3.0);
    GPUAtmosphereParameters shifted = p;
    shifted.cloudAdvection.x += 1.0;
    store(19, float3(Metallic.Clouds::density(p, samplePoint),
        Metallic.Clouds::density(shifted, samplePoint + float3(1.0, 0.0, 0.0)),
        Metallic.Clouds::density(shifted, samplePoint)));
    float densityMean = 0.0, transMean = 0.0, minT = 1.0;
    for (uint sample = 0; sample < 64u; ++sample) {
        float3 point = normalize(ground + float3(float(sample % 8u) * 2.0 - 7.0, 0.0,
            float(sample / 8u) * 2.0 - 7.0)) * (p.observerPlanetBottom.w + 2.5);
        densityMean += Metallic.Clouds::density(p, point) / 64.0;
        float3 receiver = normalize(point) * (p.observerPlanetBottom.w + 0.002);
        float t = Metallic.Clouds::transmittance(p, receiver, sun, 1e9, 48u);
        transMean += t / 64.0; minT = min(minT, t);
    }
    store(20, float3(densityMean, transMean, minT));
    float directPhase = 0.0, higherPhase = 0.0;
    for (uint sample = 0; sample < 512u; ++sample) {
        float cosine = (float(sample) + 0.5) / 256.0 - 1.0;
        directPhase += Metallic.Clouds::scatteringPhase(cosine, p.cloudOptics.z, p.cloudOptics.y, 1.0) * 4.0 * kPi / 512.0;
        higherPhase += Metallic.Clouds::scatteringPhase(cosine, p.cloudOptics.z, p.cloudOptics.y, 0.0) * 4.0 * kPi / 512.0;
    }
    store(21, float3(directPhase, higherPhase, p.cloudOptics.y));
    GPUAtmosphereParameters zero = p; zero.cloudLayer.w = 0.0;
    AtmosphereSegment zeroSegment = segmentIntegration(zero, trans, multi, ground, up, 8.0);
    AtmosphereSegment clearSegment = segmentIntegration(clear, trans, multi, ground, up, 8.0);
    store(22, zeroSegment.radiance); store(23, zeroSegment.transmittance);
    store(24, clearSegment.radiance); store(25, clearSegment.transmittance);
    ResourceHandle<ByteAddressBuffer> parameterHandle;
    parameterHandle.index = gProbe.resources.parameters.resource.index;
    let structured = resolveBuffer<StructuredBuffer<GPUAtmosphereParameters>>(parameterHandle);
    uint count, stride; structured.GetDimensions(count, stride);
    store(26, float3(count, stride, p.cloudLayer.w));
    store(27, transmittanceToSource(p, trans, ground, up));
    store(28, sampleMulti(p, multi, ground, up));
    store(29, visibleSourceDisks(p, trans, ground, sun));
    store(30, float3(p.cloudShadowBudget.x, p.cloudShadowBudget.y, p.cloudShadowBudget.z));
    store(31, float3(Metallic.Clouds::transmittance(zero, ground, up),
        Metallic.Clouds::transmittance(clear, ground, up), 0.0));
    EnvironmentResourceParameters resources = {};
    resources.atmosphereParameters.index = gProbe.resources.parameters.resource.index;
    resources.transmittance = gProbe.resources.transmittance;
    resources.multiScattering = gProbe.resources.multiScattering;
    resources.skyView = gProbe.resources.skyView;
    resources.aerialPerspective = gProbe.inputAerial.resource;
    resources.cloudShadow = gProbe.resources.sourceMip;
    float3 origin = p.observerWorldTop.xyz + float3(2000.0, 2000.0, 0.0);
    float3 endpoint = origin + up * 6000.0;
    AtmosphereSegment actual = physicalAerialSegment(resources, endpoint, origin);
    AtmosphereSegment reference = segmentIntegration(p, trans, multi, worldToAtmosphereKm(p, origin), up, 6.0);
    store(32, actual.radiance); store(33, actual.transmittance);
    store(34, reference.radiance); store(35, reference.transmittance);
    store(36, solarIrradiance(p, trans, ground, 1u));
    store(37, solarIrradiance(clear, trans, ground, 1u));
    store(38, float3(cloudShadowTransmittance(p, shadow, ground, 1u),
        Metallic.Clouds::transmittance(p, ground, p.moonDirectionRadius.xyz), 0.0));
    store(39, float3(p.cloudAdvection.x, p.cloudAdvection.z, p.weatherComposition.w));
    float mapError = 0.0, terrainParallax = 0.0, elevatedMeanT = 0.0;
    for (uint receiver = 0u; receiver < 16u; ++receiver) {
        float3 elevated = normalize(ground + float3(float(receiver) - 7.5, 0.0, 2.0)) *
            (p.observerPlanetBottom.w + 0.3);
        float mapped = cloudShadowTransmittance(p, shadow, elevated, 0u);
        float reference = Metallic.Clouds::transmittance(p, elevated, sun, 1e9, 48u);
        float3 radialGround = normalize(elevated) * (p.observerPlanetBottom.w + 0.002);
        float unprojected = cloudShadowTransmittance(p, shadow, radialGround, 0u);
        mapError = max(mapError, abs(mapped - reference));
        terrainParallax = max(terrainParallax, abs(mapped - unprojected));
        elevatedMeanT += mapped / 16.0;
    }
    store(40, float3(mapError, terrainParallax, elevatedMeanT));
    float downwardDistance = max(length(p.observerPlanetBottom.xyz) - p.observerPlanetBottom.w, 0.0);
    AtmosphereSegment orbital = segmentIntegration(p, trans, multi, p.observerPlanetBottom.xyz, -up, downwardDistance);
    AtmosphereSegment orbitalClear = segmentIntegration(clear, trans, multi, p.observerPlanetBottom.xyz, -up, downwardDistance);
    store(41, orbital.radiance); store(42, orbital.transmittance);
    store(43, float3(Metallic.Clouds::transmittance(p, p.observerPlanetBottom.xyz, -up, downwardDistance, 48u)));
    store(44, orbitalClear.transmittance); store(45, orbitalClear.radiance);
    SpectralSegment sourceSky = integrateEnvironment(p, trans, multi, p.observerPlanetBottom.xyz, sun, 1e9, true);
    store(46, spectralToWorking(sourceSky.radiance));
}
