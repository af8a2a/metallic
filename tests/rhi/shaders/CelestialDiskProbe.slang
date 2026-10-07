import Atmosphere;
import AtmosphereTypes;
import Lighting;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Lighting;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

float radicalInverseTwo(uint value)
{
    return float(reversebits(value)) * (1.0 / 4294967296.0);
}

[numthreads(16, 1, 1)]
void celestialDiskProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 9u) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    if (id.x == 8u) {
        p.sunDirectionRadius.xyz = float3(0.0, 1.0, 0.0);
        p.moonIrradianceEnabled.w = 0.0;
        float3 direction = p.sunDirectionRadius.xyz;
        let transmittance = resolveUniform(gParams.transmittance);
        gParams.aerial.store(64u, float4(physicalDeltaMissDisks(p, transmittance, direction, p.observerPlanetBottom.xyz), 1.0));
        float denominator = kPi * sin(p.sunDirectionRadius.w) * sin(p.sunDirectionRadius.w);
        gParams.aerial.store(65u, float4(spectralToWorking(p.sunIrradianceEnabled.xyz) / denominator, 1.0));
        gParams.aerial.store(66u, float4(physicalDeltaMissDisks(p, transmittance, float3(1.0, 0.0, 0.0), p.observerPlanetBottom.xyz), 1.0));
        p.settings.y = 0.0;
        gParams.aerial.store(67u, float4(physicalDeltaMissDisks(p, transmittance, direction, p.observerPlanetBottom.xyz), 1.0));
        for (uint index = 68u; index <= 70u; ++index) { gParams.aerial.store(index, float4(0.0, 0.0, 0.0, 1.0)); }
        gParams.aerial.store(71u, float4(8.0, 1.0, 0.0, 1.0));
        return;
    }
    const uint count = 8192u;
    uint slot = id.x >= 1u && id.x <= 3u || id.x == 7u ? 1u : 0u;
    float phase = id.x == 2u ? kPi * 0.5 : id.x == 3u ? kPi * 0.8 : id.x == 7u ? kPi : 0.0;
    float lambert = phase >= kPi ? 0.0 : (sin(phase) + (kPi - phase) * cos(phase)) / kPi;
    p.sunDirectionRadius.xyz = float3(0.0, 1.0, 0.0);
    p.moonDirectionRadius.xyz = float3(0.0, 1.0, 0.0);
    p.moonToSun = float4(sin(phase), -cos(phase), 0.0, 1.0);
    p.moonPhase = float4(phase, 0.5 * (1.0 + cos(phase)), lambert, 0.0);
    p.moonIrradianceEnabled.xyz *= lambert;
    if (id.x == 4u) {
        // A non-axis direction exposes dot/cos membership cancellation when
        // this radius is below FP32 angular direction resolution.
        p.sunDirectionRadius = float4(normalize(float3(-0.5142664, 0.83096236, 0.21220636)), 1e-8);
    }
    if (id.x == 5u) { p.sunDirectionRadius.w = 0.0; }
    if (id.x == 6u) { p.sunDirectionRadius.xyz = float3(1.0, 0.0, 0.0); }
    float4 directionRadius = slot == 0u ? p.sunDirectionRadius : p.moonDirectionRadius;
    float4 source = slot == 0u ? p.sunIrradianceEnabled : p.moonIrradianceEnabled;
    float3 irradiance = float3(0.0), projected = float3(0.0);
    float cosineMin = 1.0, cosineMax = 0.0, cosineLossMean = 0.0;
    float positive = 0.0, totalWeight = 0.0, centroidX = 0.0, centroidZ = 0.0;
    float3 firstDirection = float3(0.0);
    float3 positionKm = p.observerPlanetBottom.xyz;
    for (uint sample = 0u; sample < count; ++sample) {
        float2 uniform = float2((float(sample) + 0.5) / float(count), radicalInverseTwo(sample));
        let light = samplePhysicalCelestialDisk(p, resolveUniform(gParams.transmittance), slot, positionKm, uniform);
        if (sample == 0u) { firstDirection = light.direction; }
        float cosine = dot(light.direction, directionRadius.xyz);
        irradiance += light.illuminance / float(count);
        projected += light.illuminance * cosine / float(count);
        cosineMin = min(cosineMin, cosine); cosineMax = max(cosineMax, cosine);
        // Sum small losses rather than 8192 near-one terms: the latter loses
        // their differences to FP32 rounding and biases this distribution test.
        cosineLossMean += (1.0 - cosine) / float(count);
        float weight = max(light.illuminance.y, 0.0);
        positive += weight > 0.0 ? 1.0 / float(count) : 0.0;
        totalWeight += weight;
        if (directionRadius.w > 1e-6) {
            centroidX += weight * light.direction.x / sin(directionRadius.w);
            centroidZ += weight * light.direction.z / sin(directionRadius.w);
        }
    }
    const uint base = id.x * 8u;
    gParams.aerial.store(base + 0u, float4(projected, 1.0));
    gParams.aerial.store(base + 1u, float4(spectralToWorking(source.xyz * source.w), 1.0));
    gParams.aerial.store(base + 2u, float4(irradiance, 1.0));
    gParams.aerial.store(base + 3u, float4(cosineMin, 1.0 - cosineLossMean, cosineMax, positive));
    gParams.aerial.store(base + 4u, float4(centroidX / max(totalWeight, 1e-20), centroidZ / max(totalWeight, 1e-20), 0.0, 1.0));
    gParams.aerial.store(base + 5u, float4(directionRadius.w, celestialDiskSolidAngle(directionRadius.w), lambert, 1.0));
    gParams.aerial.store(base + 6u, float4(firstDirection, 1.0));
    gParams.aerial.store(base + 7u, float4(float(id.x), float(count), 0.0, 1.0));
}

[numthreads(8, 8, 1)]
void moonPhaseAtlasMain(uint3 id : SV_DispatchThreadID)
{
    const uint width = 256u, height = 64u;
    if (id.x >= width || id.y >= height) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    uint panel = id.x / 64u;
    float phase = panel == 0u ? 0.0 : panel == 1u ? kPi * 0.5 : panel == 2u ? kPi * 0.8 : kPi;
    float lambert = phase >= kPi ? 0.0 : (sin(phase) + (kPi - phase) * cos(phase)) / kPi;
    p.moonDirectionRadius.xyz = float3(0.0, 1.0, 0.0);
    p.moonToSun = float4(sin(phase), -cos(phase), 0.0, 1.0);
    p.moonPhase = float4(phase, 0.5 * (1.0 + cos(phase)), lambert, 0.0);
    p.moonIrradianceEnabled.xyz *= lambert;
    float2 disk = ((float2(id.x % 64u, 63u - id.y) + 0.5) / 32.0 - 1.0) / 0.9;
    float radiusSquared = dot(disk, disk);
    float4 value = float4(0.0);
    if (radiusSquared <= 1.0) {
        float sineRadius = sin(p.moonDirectionRadius.w);
        float3 direction = float3(disk.x * sineRadius, sqrt(max(1.0 - radiusSquared * sineRadius * sineRadius, 0.0)),
            disk.y * sineRadius);
        value = float4(sourceDiskRadiance(p, resolveUniform(gParams.transmittance), 1u,
            p.observerPlanetBottom.xyz, direction), 1.0);
    }
    gParams.aerial.store(id.x + id.y * width, value);
}
