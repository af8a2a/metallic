#language slang 2026
module CloudLightingLut;

import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;
groupshared float3 skyLuminanceSamples[64];

float randomStreamFraction(inout uint seed)
{
    seed = seed * 196314165u + 907633515u;
    return asfloat(0x3f800000u | (seed >> 9)) - 1.0f;
}

float3 uniformSphereSample(uint2 id)
{
    // UE FUniformSphereSamplesBuffer fills column first and writes to j * 8 + i.
    uint seed = 0xde4dc0deu;
    for (uint preceding = 0; preceding < 2u * (8u * id.x + id.y); ++preceding) {
        seed = seed * 196314165u + 907633515u;
    }
    float u0 = (float(id.x) + randomStreamFraction(seed)) * (1.0f / 8.0f);
    float u1 = (float(id.y) + randomStreamFraction(seed)) * (1.0f / 8.0f);
    float a = 1.0f - 2.0f * u0;
    float b = sqrt(1.0f - a * a);
    float phi = 2.0f * kPi * u1;
    return float3(b * cos(phi), a, b * sin(phi)); // UE Z-up to Metallic Y-up.
}

float3 skyLightSource(GPUAtmosphereParameters p, float3 point, float3 direction, float4 irradiance,
    float3 scattering, float3 isotropicScattering)
{
    if (irradiance.w == 0.0f) { return float3(0.0f); }
    float3 up = normalize(point);
    float2 planet = sphereRoots(point - 0.001f * up, direction, p.observerPlanetBottom.w);
    float planetShadow = any(planet >= 0.0f) ? 0.0f : 1.0f;
    float3 transmittance = transmittanceToSource(p, resolveUniform(gParams.transmittance), point, direction);
    float3 multiple = sampleMulti(p, resolveUniform(gParams.multiScattering), point, direction);
    return irradiance.xyz * (planetShadow * transmittance * isotropicScattering + multiple * scattering);
}

// RenderDistantSkyLightLutCS: isotropic gas scattering at 6 km, no ground bounce, disks, or clouds.
float3 distantSkySample(GPUAtmosphereParameters p, float3 direction)
{
    float3 position = float3(0.0f, p.observerPlanetBottom.w + 6.0f, 0.0f);
    float2 bottom = sphereRoots(position, direction, p.observerPlanetBottom.w);
    float2 top = sphereRoots(position, direction, p.observerWorldTop.w);
    if (all(top < 0.0f)) { return float3(0.0f); }
    float tMax = all(bottom < 0.0f) ? max(top.x, top.y) : max(0.0f, min(bottom.x, bottom.y));
    float dt = tMax / 10.0f;
    float3 luminance = float3(0.0f);
    float3 throughput = float3(1.0f);
    for (uint i = 0; i < 10u; ++i) {
        float t = tMax * (float(i) + 0.3f) / 10.0f;
        float3 point = position + t * direction;
        Medium medium = sampleMedium(p, point);
        float3 transmittance = exp(-medium.extinction * dt);
        float3 scattering = medium.rayleigh + medium.mie;
        float3 isotropic = scattering * (1.0f / (4.0f * kPi));
        float3 source = skyLightSource(p, point, p.sunDirectionRadius.xyz, p.sunIrradianceEnabled, scattering, isotropic);
        source += skyLightSource(p, point, p.moonDirectionRadius.xyz, p.moonIrradianceEnabled, scattering, isotropic);
        float3 integral = (source - source * transmittance) / max(medium.extinction, float3(1e-9f));
        luminance += throughput * integral;
        throughput *= transmittance;
    }
    return luminance;
}

// FAtmosphereSetup::GetTransmittanceAtGroundLevel: trace from 500 m using the
// default minimum solar elevation of -90 degrees, including subsurface optical depth at night.
float3 groundIlluminance(GPUAtmosphereParameters p, float3 lightDirection, float4 irradiance)
{
    if (irradiance.w == 0.0f) { return float3(0.0f); }
    float elevation = asin(clamp(lightDirection.y, -1.0f, 1.0f));
    float3 direction = float3(cos(elevation), sin(elevation), 0.0f);
    float3 position = float3(0.0f, p.observerPlanetBottom.w + 0.5f, 0.0f);
    float2 roots = sphereRoots(position, direction, p.observerWorldTop.w);
    float tMax = roots.x >= 0.0f ? roots.x : roots.y;
    float3 opticalDepth = float3(0.0f);
    if (tMax > 0.0f) {
        const float sampleStep = 1.0f / 15.0f;
        float sampleLength = sampleStep * tMax;
        for (float sampleT = 0.0f; sampleT < 1.0f; sampleT += sampleStep) {
            float height = length(position + direction * (tMax * sampleT)) - p.observerPlanetBottom.w;
            float densityMie = exp(-height / p.mieScatteringScaleHeight.w);
            float densityRay = exp(-height / p.rayleighScaleHeight.w);
            float densityOzone = saturate(1.0f - abs(height - p.ozoneCenter.w) / p.groundOzoneWidth.w);
            // Below the horizon densities may overflow. A zero coefficient still describes vacuum.
            float3 mie = select(p.mieExtinctionAnisotropy.xyz > 0.0f, densityMie * p.mieExtinctionAnisotropy.xyz, float3(0.0f));
            float3 ray = select(p.rayleighScaleHeight.xyz > 0.0f, densityRay * p.rayleighScaleHeight.xyz, float3(0.0f));
            opticalDepth += sampleLength * (mie + ray + densityOzone * p.ozoneCenter.xyz);
        }
    }
    return irradiance.xyz * exp(-opticalDepth);
}

[numthreads(8, 8, 1)]
void cloudLightingLutMain(uint3 id : SV_GroupThreadID)
{
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    uint index = id.y * 8u + id.x;
    skyLuminanceSamples[index] = distantSkySample(p, uniformSphereSample(id.xy));
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 32u; stride > 1u; stride >>= 1u) {
        if (index < stride) { skyLuminanceSamples[index] += skyLuminanceSamples[index + stride]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (index == 0u) {
        float3 sum = skyLuminanceSamples[0] + skyLuminanceSamples[1];
        float3 illuminance = sum * (4.0f * kPi / 64.0f);
        resolveUniform(gParams.output)[uint2(0, 0)] = float4(illuminance * (1.0f / (4.0f * kPi)), 0.0f);
    } else if (index == 1u) {
        resolveUniform(gParams.output)[uint2(1, 0)] = float4(groundIlluminance(p, p.sunDirectionRadius.xyz, p.sunIrradianceEnabled), 0.0f);
    } else if (index == 2u) {
        resolveUniform(gParams.output)[uint2(2, 0)] = float4(groundIlluminance(p, p.moonDirectionRadius.xyz, p.moonIrradianceEnabled), 0.0f);
    }
}
