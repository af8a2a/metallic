#language slang 2026
module Clouds;

import AtmosphereTypes;
import SimpleVolumetricCloud;

namespace Metallic.Clouds
{
public bool enabled(GPUAtmosphereParameters p)
{
    return p.cloudOptics.w > 0.5 && p.cloudLayer.z > 0.0 &&
        p.cloudLayer.w > 0.0 && p.cloudOptics.x > 0.0;
}
uint hash(uint value)
{
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
public float cloudNormalizedHeight(GPUAtmosphereParameters p, float3 planetPositionKm)
{
    float altitude = length(planetPositionKm) - p.observerPlanetBottom.w;
    return (altitude - p.cloudLayer.x) / (p.cloudLayer.y - p.cloudLayer.x);
}

public float4 cloudConservativeDensity(GPUAtmosphereParameters p, float3 cameraRelativeKm, float h)
{
    float3 worldKm = cameraRelativeKm + p.observerWorldTop.xyz * 0.001f;
    float periodKm = p.cloudAdvection.y;
    float3 skyTexScale = float3(worldKm.xz / periodKm + 0.5f, worldKm.y / 256.0f);
    // Seed zero preserves the reference placement. Other seeds translate the same field.
    uint seed = hash(asuint(p.cloudAdvection.w));
    skyTexScale.xy += float2(seed & 0xffffu, seed >> 16u) * (1.0f / 65536.0f);
    float3 wind = float3(-p.cloudAdvection.xz / periodKm, 0.0f);
    return simpleCloudConservativeDensity(p.cloudShapeResources, skyTexScale, h, wind,
        p.cloudLayer.z - 0.7f);
}

// UE material If_1 uses AEqualsB for abs(h - 0.5) <= EqualsThreshold.
public bool cloudMaterialContainsHeight(float h)
{
    return abs(h - 0.5f) <= 0.48f;
}

public float cloudMaterialExtinction(GPUAtmosphereParameters p, float4 conservative, float3 cameraRelativeKm)
{
    float extinctionPerMetre = p.cloudOptics.x * 0.001f;
    return simpleCloudExtinction(conservative, extinctionPerMetre * p.cloudLayer.w, length(cameraRelativeKm));
}

// Called within an active cloud layer, where extinction and the shape textures are valid.
public float cloudExtinctionPerMeter(GPUAtmosphereParameters p, float3 planetPositionKm)
{
    float h = cloudNormalizedHeight(p, planetPositionKm);
    if (!cloudMaterialContainsHeight(h)) { return 0.0f; }
    float3 cameraRelativeKm = planetPositionKm - p.observerPlanetBottom.xyz;
    float4 conservative = cloudConservativeDensity(p, cameraRelativeKm, h);
    return cloudMaterialExtinction(p, conservative, cameraRelativeKm);
}

public float density(GPUAtmosphereParameters p, float3 planetPositionKm)
{
    if (!enabled(p)) { return 0.0; }
    // The transport callers multiply dimensionless density by the authored extinction per km.
    return (cloudExtinctionPerMeter(p, planetPositionKm) * 1000.0f) / p.cloudOptics.x;
}
float2 roots(float3 position, float3 direction, float radius)
{
    float b = dot(position, direction), r = length(position);
    float c = (r - radius) * (r + radius), discriminant = b * b - c;
    if (discriminant < 0.0) { return float2(-1.0); }
    float root = sqrt(max(discriminant, 0.0));
    float near = -b - root, far = -b + root;
    if (b < 0.0 && abs(far) > 1e-8) { near = c / far; }
    if (b > 0.0 && abs(near) > 1e-8) { far = c / near; }
    return float2(near, far);
}
// A ray may cross the spherical layer twice when it misses the solid planet.
// Each pair is a clipped, disjoint interval; negative pairs mean no interval.
public float4 intervals(GPUAtmosphereParameters p, float3 position, float3 direction, float maximumDistance)
{
    float4 result = float4(-1.0);
    if (!enabled(p) || maximumDistance <= 0.0) { return result; }
    float2 outer = roots(position, direction, p.observerPlanetBottom.w + p.cloudLayer.y);
    float begin = max(outer.x, 0.0), end = min(outer.y, maximumDistance);
    if (end <= begin || outer.y < 0.0) { return result; }
    float2 inner = roots(position, direction, p.observerPlanetBottom.w + p.cloudLayer.x);
    if (inner.y <= begin || inner.x >= end || inner.y < 0.0) { return float4(begin, end, -1.0, -1.0); }
    float firstEnd = min(end, inner.x), secondBegin = max(begin, inner.y);
    if (firstEnd > begin) { result.xy = float2(begin, firstEnd); }
    if (end > secondBegin) {
        if (result.x < 0.0) { result.xy = float2(secondBegin, end); }
        else { result.zw = float2(secondBegin, end); }
    }
    return result;
}
public float opticalDepth(GPUAtmosphereParameters p, float3 position, float3 direction,
    float maximumDistance = 1e9, uint samples = 16u)
{
    float4 ranges = intervals(p, position, direction, maximumDistance);
    float depth = 0.0;
    samples = clamp(samples, 1u, 64u);
    for (uint interval = 0u; interval < 2u; ++interval) {
        float begin = interval == 0u ? ranges.x : ranges.z;
        float end = interval == 0u ? ranges.y : ranges.w;
        if (begin < 0.0 || end <= begin) { continue; }
        float stepLength = (end - begin) / float(samples);
        for (uint step = 0u; step < samples; ++step) {
            depth += density(p, position + direction * (begin + (float(step) + 0.5) * stepLength)) * stepLength;
        }
    }
    return min(depth * p.cloudOptics.x, 80.0);
}
public float transmittance(GPUAtmosphereParameters p, float3 position, float3 direction,
    float maximumDistance = 1e9, uint samples = 16u)
{
    return exp(-opticalDepth(p, position, direction, maximumDistance, samples));
}
} // namespace Metallic.Clouds
