import Atmosphere;
import Lighting;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Lighting;

struct AtmosphereSeamParams
{
    AtmospherePrecomputeParams resources;
    BufferSpan<float4> inputAerial;
    uint reserved;
};
[[vk::push_constant]] ConstantBuffer<AtmosphereSeamParams> gProbe;

EnvironmentResourceParameters seamEnvironment()
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

// Construct each pair without sin(pi): the +/- z components are exact
// mirrors around the global -X longitude seam at the ground observer.
float3 seamDirection(float elevation, float longitudeOffset)
{
    return float3(-cos(longitudeOffset) * cos(elevation), sin(elevation),
        sin(longitudeOffset) * cos(elevation));
}

[numthreads(64, 1, 1)]
void atmosphereSeamProbeMain(uint3 id : SV_DispatchThreadID)
{
    const uint pairCount = 16u, sweepWidth = 256u, bandCount = 8u;
    if (id.x >= pairCount + sweepWidth * bandCount) { return; }
    float elevations[8] = {-0.0872664626, 0.0, 0.0349065850, 0.1745329252,
        0.4363323130, 0.9599310886, 1.3089969390, 1.5358897420};
    uint band = id.x < pairCount ? id.x / 2u : (id.x - pairCount) / sweepWidth;
    float offset = id.x < pairCount ? (id.x % 2u == 0u ? -1e-5 : 1e-5) :
        ((float((id.x - pairCount) % sweepWidth) + 0.5) / float(sweepWidth) - 0.5) * (2.0 * kPi);
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    float3 direction = seamDirection(elevations[band], offset);
    float2 uv = skyDirectionUV(p, direction);
    float3 roundtrip = skyDirectionFromUV(p, uv);
    let sky = resolveUniform(gProbe.resources.skyView);
    let transmittance = resolveUniform(gProbe.resources.transmittance);
    let reference = integrateEnvironment(p, transmittance, resolveUniform(gProbe.resources.multiScattering),
        p.observerPlanetBottom.xyz, direction, 1e9, true);
    float3 disks = visibleSourceDisks(p, transmittance, p.observerPlanetBottom.xyz, direction);
    float3 consumer = physicalVisibleEnvironment(seamEnvironment(), direction, p.observerWorldTop.xyz);
    uint width, height;
    sky.GetDimensions(width, height);
    uint row = min(uint(saturate(uv.y) * float(height)), height - 1u);
    uint base = id.x * 15u;
    gProbe.resources.aerial.store(base, float4(direction, elevations[band]));
    gProbe.resources.aerial.store(base + 1u, float4(uv, length(roundtrip - direction), 1.0));
    gProbe.resources.aerial.store(base + 2u, float4(sampleSkyView(p, sky, direction), 1.0));
    gProbe.resources.aerial.store(base + 3u, float4(max(spectralToWorking(reference.radiance), 0.0), 1.0));
    gProbe.resources.aerial.store(base + 4u, float4(sampleLinear(resolveUniform(gProbe.resources.sourceMip),
        directionLatLong(direction), true).xyz, 1.0));
    gProbe.resources.aerial.store(base + 5u, float4(consumer, 1.0));
    gProbe.resources.aerial.store(base + 6u, float4(consumer - disks, 1.0));
    gProbe.resources.aerial.store(base + 7u, sky.Load(int3(0, row, 0)));
    gProbe.resources.aerial.store(base + 8u, sky.Load(int3(int(width - 1u), int(row), 0)));
    // Sampler diagnostics distinguish the inverse mapping from negative
    // fractional coordinates and signed remainder lowering at longitude zero.
    float2 pixel = float2(uv.x, saturate(uv.y)) * float2(width, height) - 0.5;
    int2 pixelBase = int2(floor(pixel));
    float2 helperFraction = frac(pixel);
    int wrappedFirst = (pixelBase.x % int(width) + int(width)) % int(width);
    int wrappedSecond = ((pixelBase.x + 1) % int(width) + int(width)) % int(width);
    float2 expectedFraction = pixel - floor(pixel);
    float4 explicitSample = float4(0.0);
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = pixelBase.x + dx;
            // UV lies in [0,1], so one branch correction is sufficient and
            // avoids the historical signed-remainder wrap entirely.
            if (x < 0) { x += int(width); }
            if (x >= int(width)) { x -= int(width); }
            int y = clamp(pixelBase.y + dy, 0, int(height) - 1);
            float weight = (dx == 0 ? 1.0 - expectedFraction.x : expectedFraction.x) *
                (dy == 0 ? 1.0 - expectedFraction.y : expectedFraction.y);
            explicitSample += sky.Load(int3(x, y, 0)) * weight;
        }
    }
    gProbe.resources.aerial.store(base + 9u, float4(pixel, float2(pixelBase)));
    gProbe.resources.aerial.store(base + 10u, float4(helperFraction, float(wrappedFirst), float(wrappedSecond)));
    gProbe.resources.aerial.store(base + 11u, explicitSample);
    gProbe.resources.aerial.store(base + 12u,
        sampleLinear(sky, float2(uv.x + 1.0, uv.y), true));
    // The production 32^3 aerial interpolation shares longitude topology,
    // but uses its own buffer indexing rather than the sky image sampler.
    AtmosphereSegment aerial = aerialPerspective(p, gProbe.inputAerial,
        p.observerWorldTop.xyz + direction * 10000.0);
    gProbe.resources.aerial.store(base + 13u, float4(aerial.radiance, 1.0));
    gProbe.resources.aerial.store(base + 14u, float4(aerial.transmittance, 1.0));
}

[numthreads(8, 8, 1)]
void atmosphereSeamImageMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gProbe.resources.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gProbe.resources.parameters);
    float2 uv = (float2(id.xy) + 0.5) / float2(gProbe.resources.dimensions.xy);
    float elevation = atan2(1.0, 0.84) + (0.5 - uv.y) * 0.8;
    float3 direction = seamDirection(elevation, (uv.x - 0.5) * 1.1);
    float3 radiance;
    if (gProbe.resources.dimensions.w == 0u) {
        radiance = sampleSkyView(p, resolveUniform(gProbe.resources.skyView), direction);
    } else if (gProbe.resources.dimensions.w == 1u) {
        let reference = integrateEnvironment(p, resolveUniform(gProbe.resources.transmittance),
            resolveUniform(gProbe.resources.multiScattering), p.observerPlanetBottom.xyz, direction, 1e9, true);
        radiance = max(spectralToWorking(reference.radiance), 0.0);
    } else {
        radiance = physicalVisibleEnvironment(seamEnvironment(), direction, p.observerWorldTop.xyz);
    }
    gProbe.resources.aerial.store(id.y * gProbe.resources.dimensions.x + id.x, float4(radiance, 1.0));
}
