module CloudsProbe;

import Atmosphere;
import Clouds;
import CloudShadowMap;
import CloudLightEnvironment;
import VolumetricCloudLighting;
import VolumetricCloudViewTrace;
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
    for (uint index = 0u; index < 64u; ++index) { store(index, float3(0.0)); }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    let trans = resolveUniform(gProbe.resources.transmittance);
    let multi = resolveUniform(gProbe.resources.multiScattering);
    let sky = resolveUniform(gProbe.resources.skyView);
    let shadow = resolveUniform(gProbe.resources.sourceMip);
    let capture = resolveUniform(gProbe.radiance);
    float3 ground = normalize(p.observerPlanetBottom.xyz) * (p.observerPlanetBottom.w + 0.002f);
    float3 up = normalize(ground);
    float3 sun = p.sunDirectionRadius.xyz;
    GPUAtmosphereParameters clear = p;
    clear.cloudOptics.w = 0.0;
    float cloudT = Metallic.Clouds::transmittance(p, ground, sun, 1e9, 48u);
    store(0, float3(cloudT));
    store(1, float3(cloudShadowTransmittance(p, shadow, ground, 0u)));
    // Sample both tile centres independently. An unscheduled emitter has zero optical depth.
    store(2, sampleLinear(shadow, float2(0.25, 0.5)).xyz);
    store(3, sampleLinear(shadow, float2(0.75, 0.5)).xyz);
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
    // The two noise layers wrap at integer multiples of the 256 km layout.
    // Both must remain continuous across the CPU's 2048 km phase reduction.
    shifted.cloudAdvection.x += 2048.0;
    store(19, float3(Metallic.Clouds::density(p, samplePoint),
        Metallic.Clouds::density(shifted, samplePoint), 0.0));
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
    float3 integratedPhase = 0.0f;
    for (uint sample = 0; sample < 512u; ++sample) {
        float cosine = (float(sample) + 0.5) / 256.0 - 1.0;
        integratedPhase += Metallic.Clouds::cloudPhaseOrders(cosine, p.cloudOptics.z) * (4.0f * kPi / 512.0f);
    }
    store(21, integratedPhase);
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
    // In the overhead Moon fixture the ground centre interpolates four published texels,
    // each filtering four traced columns. Average optical depths before exponentiation.
    float filteredMoon = 1.0f;
    if (Metallic.Clouds::enabled(p) && (uint(p.cloudShadowBudget.z) & 2u)) {
        uint width, height;
        shadow.GetDimensions(width, height);
        let projection = Metallic.Clouds::CloudShadowProjection::forLight(p, 1u);
        float opticalDepth = 0.0f;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                float2 offset = (float2(x, y) - 1.5f) * (projection.halfWidthKm / float(height));
                float3 point = ground + projection.right * offset.x - projection.up * offset.y;
                opticalDepth += Metallic.Clouds::opticalDepth(p, point, p.moonDirectionRadius.xyz, 1e9f, uint(p.cloudShadowBudget.y));
            }
        }
        filteredMoon = exp(-opticalDepth / 16.0f);
    }
    store(38, float3(cloudShadowTransmittance(p, shadow, ground, 1u),
        Metallic.Clouds::transmittance(p, ground, p.moonDirectionRadius.xyz), filteredMoon));
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

    // A 100 m vacuum followed by a homogeneous 500 m slab. Record the production
    // integral independently of the procedural density field and atmospheric LUTs.
    let extinctionPerMeter = 0.002f;
    Metallic.Clouds::CloudViewIntegral slab = Metallic.Clouds::CloudViewIntegral::empty();
    slab.integrateSample(50.0f, 100.0f, 0.0f, float3(0.0f));
    for (uint step = 0u; step < 5u; ++step) {
        slab.integrateSample(150.0f + 100.0f * step, 100.0f, extinctionPerMeter,
            extinctionPerMeter * float3(1, 2, 4));
    }
    store(47, slab.luminance);
    store(48, float3(slab.transmittance, slab.representativeDepthMeters(), slab.depthWeightsSum));

    store(51, Metallic.Clouds::cloudPhaseOrders(1.0f, p.cloudOptics.z));
    store(52, Metallic.Clouds::cloudPhaseOrders(0.0f, p.cloudOptics.z));
    store(53, Metallic.Clouds::cloudPhaseOrders(-1.0f, p.cloudOptics.z));
    // Illuminate each scattering order in a different channel through one homogeneous slab.
    float3 scattering = Metallic.Clouds::cloudScatteringOrders(p.cloudOptics.y, extinctionPerMeter);
    float3 extinction = Metallic.Clouds::cloudExtinctionOrders(extinctionPerMeter);
    Metallic.Clouds::CloudViewIntegral orders = Metallic.Clouds::CloudViewIntegral::empty();
    for (int order = 2; order >= 0; --order) {
        float3 source = 0.0f;
        source[order] = scattering[order];
        orders.integrateOrder(350.0f, 500.0f, extinction[order], source, order);
    }
    store(54, orders.luminance);
    store(55, float3(orders.transmittance, orders.representativeDepthMeters(), orders.depthWeightsSum));

    if (Metallic.Clouds::enabled(p) && (uint(p.cloudShadowBudget.z) & 1u)) {
        float groundT;
        bool valid = tryCloudShadowMap(p, shadow, ground, 0u, groundT);
        store(62, float3(valid ? 1.0f : 0.0f, cloudShadowTransmittance(p, shadow, ground, 0u),
            Metallic.Clouds::transmittance(p, ground, sun)));
        // Bilinear sampling at the Sun tile's right edge must not mix in the Moon tile.
        uint width, height;
        shadow.GetDimensions(width, height);
        uint row = height / 2u;
        float4 edge = Metallic.Clouds::sampleShadowColumn(p, shadow, float2(1, (float(row) + 0.5f) / float(height)), 0u);
        store(63, abs(edge.xyz - shadow.Load(int3(height - 1u, row, 0)).xyz));
        let projection = Metallic.Clouds::CloudShadowProjection::forLight(p, 0u);
        float3 nearPoint = projection.nearPlanePosition(float2(0.5f));
        float3 column = Metallic.Clouds::sampleShadowColumn(p, shadow, float2(0.5f), 0u).xyz;
        float3 receiverT, mapped;
        float halfDepthKm = column.z * 0.5f / max(column.y, 1e-8f) / 1000.0f;
        for (uint i = 0; i < 3; ++i) {
            float depthKm = column.x + (i == 0 ? -0.1f : (i == 1 ? halfDepthKm : 2.0f * halfDepthKm + 1.0f));
            float value;
            mapped[i] = tryCloudShadowMap(p, shadow, nearPoint + projection.forward * depthKm, 0u, value) ? 1.0f : 0.0f;
            receiverT[i] = value;
        }
        store(60, receiverT);
        store(61, mapped);
    }
    if (Metallic.Clouds::enabled(p)) {
        // The UE instance reference coordinates use absolute world km and its original placement.
        GPUAtmosphereParameters shape = p;
        shape.cloudLayer = float4(1.5, 4.0, 0.5, 1.0);
        shape.cloudOptics.x = 8.0;
        shape.cloudAdvection = 0.0;
        float3 a = worldToAtmosphereKm(shape, float3(-13000, 3000, 16000));
        float3 b = worldToAtmosphereKm(shape, float3(4000, 3750, 18000));
        float3 c = worldToAtmosphereKm(shape, float3(6000, 2500, 3000));
        store(49, float3(Metallic.Clouds::density(shape, a), Metallic.Clouds::density(shape, b), Metallic.Clouds::density(shape, c)));
        store(56, cloudDistantSkyLuminance(p));
        store(57, cloudAtmosphereLightIlluminanceOnGround(p, 0u));
        store(58, cloudAtmosphereLightIlluminanceOnGround(p, 1u));
        GPUAtmosphereParameters thin = shape;
        thin.cloudOptics.x = 0.8f;
        store(59, Metallic.Clouds::cloudShadowTransmittanceOrders(thin, a, -up, 10u));
        // Hold the world-space material sample still while moving the camera to 250 m and 0 m.
        shape.observerPlanetBottom.xyz = a - float3(0.25, 0, 0);
        shape.observerWorldTop.xyz = float3(-13250, 3000, 16000);
        float halfFade = Metallic.Clouds::density(shape, a);
        shape.observerPlanetBottom.xyz = a;
        shape.observerWorldTop.xyz = float3(-13000, 3000, 16000);
        store(50, float3(halfFade, Metallic.Clouds::density(shape, a), 0));
    }
}
