#language slang 2026
module Clouds;

import AtmosphereTypes;

namespace Metallic.Clouds
{
// Original analytic cloud field and bounded scattering approximation. This is
// a prescribed weather field, not a fluid simulation or droplet Mie solver.
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
float lattice(int3 cell, uint seed)
{
    uint3 wrapped = uint3(cell) & 255u;
    return float(hash(wrapped.x ^ hash(wrapped.y ^ hash(wrapped.z ^ seed)))) * (1.0 / 4294967296.0);
}
float noise(float3 position, uint seed)
{
    int3 cell = int3(floor(position));
    float3 f = frac(position); f = f * f * (3.0 - 2.0 * f);
    float lower = lerp(lerp(lattice(cell, seed), lattice(cell + int3(1, 0, 0), seed), f.x),
        lerp(lattice(cell + int3(0, 1, 0), seed), lattice(cell + int3(1, 1, 0), seed), f.x), f.y);
    float upper = lerp(lerp(lattice(cell + int3(0, 0, 1), seed), lattice(cell + int3(1, 0, 1), seed), f.x),
        lerp(lattice(cell + int3(0, 1, 1), seed), lattice(cell + int3(1, 1, 1), seed), f.x), f.y);
    return lerp(lower, upper, f.z);
}
public float density(GPUAtmosphereParameters p, float3 planetPositionKm)
{
    if (!enabled(p)) { return 0.0; }
    float altitude = length(planetPositionKm) - p.observerPlanetBottom.w;
    float h = (altitude - p.cloudLayer.x) / max(p.cloudLayer.y - p.cloudLayer.x, 1e-4);
    if (h <= 0.0 || h >= 1.0) { return 0.0; }
    // Coordinate subtraction represents continuous field advection. The CPU
    // wraps its double-precision displacement by the common 2048 km period.
    float3 position = planetPositionKm - p.cloudAdvection.xyz;
    uint seed = asuint(p.cloudAdvection.w);
    float shape = 0.65 * noise(position / 8.0, seed) + 0.35 * noise(position / 2.0, seed + 17u);
    float erosion = noise(position / 0.5, seed + 71u) * 0.12;
    float occupancy = smoothstep(1.0 - p.cloudLayer.z, 1.0 - p.cloudLayer.z + 0.2, shape - erosion);
    float profile = smoothstep(0.0, 0.12, h) * (1.0 - smoothstep(0.72, 1.0, h));
    return occupancy * (0.35 + 0.65 * shape) * profile * p.cloudLayer.w;
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
public float phase(float cosine, float g)
{
    float square = g * g;
    return (1.0 - square) / (12.566370614359172 * pow(max(1.0 + square - 2.0 * g * cosine, 1e-4), 1.5));
}
public float scatteringPhase(float cosine, float g, float albedo, float sourceTransmittance)
{
    // Three higher orders broaden the HG lobe. Their integrated angular
    // weights form a geometric series in albedo/2, plus the transmitted
    // direct term. Thus the sum never exceeds one for 0<=albedo<=1: cloud
    // multiple scattering cannot manufacture incident celestial energy.
    float result = sourceTransmittance * phase(cosine, g);
    float orderWeight = 1.0 - sourceTransmittance, orderG = g;
    for (uint order = 1u; order <= 3u; ++order) {
        orderWeight *= saturate(albedo) * 0.5; orderG *= g;
        result += orderWeight * phase(cosine, orderG);
    }
    return result;
}
} // namespace Metallic.Clouds
