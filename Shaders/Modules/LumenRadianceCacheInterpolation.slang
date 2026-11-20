module LumenRadianceCacheInterpolation;

import LumenProbeMath;
import LumenRadianceCache;

namespace Metallic.Lumen
{
public struct LumenRadianceCacheInputs
{
    public StructuredBuffer<uint> indirection;
    public StructuredBuffer<LumenRadianceCacheClipmap> clipmaps;
    public Texture2D<float4> radiance;
    public SamplerState sampler;
};

public struct LumenRadianceCacheCoverage
{
    public float minTraceDistance;
    public bool valid;
    uint clipmapIndex;
    uint3 corner;
    float3 lerpAlphas;
    uint probeIndices[8];
};

// UE 5.7.4 LumenRadianceCacheInterpolation.ush:197,381. An incomplete
// neighborhood can occur at the physical atlas limit. It cannot cap geometry tracing.
public LumenRadianceCacheCoverage getLumenRadianceCacheCoverage(LumenRadianceCacheInputs inputs, float3 worldPosition)
{
    LumenRadianceCacheCoverage coverage = {};
    coverage.minTraceDistance = 100000.0f;
    uint clipmapCount = inputs.clipmaps[0].layout.y;
    uint clipmapIndex = lumenRadianceClipmapForPosition(worldPosition, inputs.clipmaps, 0.0f);
    if (clipmapIndex >= clipmapCount) {
        return coverage;
    }

    LumenRadianceCacheClipmap clipmap = inputs.clipmaps[clipmapIndex];
    float3 cornerProbeCoordFloat = lumenRadianceProbeCoordFloat(worldPosition, clipmap) - 0.5f;
    coverage.clipmapIndex = clipmapIndex;
    coverage.corner = uint3(floor(cornerProbeCoordFloat));
    coverage.lerpAlphas = frac(cornerProbeCoordFloat);

    uint maxNumProbes = clipmap.layout.w * clipmap.layout.w;
    for (uint index = 0u; index < 8u; ++index) {
        uint3 offset = uint3(index >> 2u, (index >> 1u) & 1u, index & 1u);
        uint address = lumenRadianceIndirectionAddress(coverage.corner + offset, clipmapIndex, clipmap.layout.x, clipmapCount);
        uint probeIndex = inputs.indirection[address];
        if (probeIndex >= maxNumProbes) {
            return coverage;
        }
        coverage.probeIndices[index] = probeIndex;
    }

    float cellOcclusionDistance = clipmap.cornerAndCellSize.w * sqrt(3.0f);
    coverage.minTraceDistance = clipmap.traceSettings.x + cellOcclusionDistance;
    coverage.valid = true;
    return coverage;
}

// UE 5.7.4 Common.ush RayIntersectSphere, far intersection.
float radianceCacheRaySphereExit(float3 rayOrigin, float3 rayDirection, float3 sphereCenter, float sphereRadius)
{
    float3 localPosition = rayOrigin - sphereCenter;
    float localPositionSqr = dot(localPosition, localPosition);
    float3 quadraticCoef;
    quadraticCoef.x = dot(rayDirection, rayDirection);
    quadraticCoef.y = 2.0f * dot(rayDirection, localPosition);
    quadraticCoef.z = localPositionSqr - sphereRadius * sphereRadius;
    float discriminant = quadraticCoef.y * quadraticCoef.y - 4.0f * quadraticCoef.x * quadraticCoef.z;
    float intersection = -1.0f;
    if (discriminant >= 0.0f) {
        float sqrtDiscriminant = sqrt(discriminant);
        intersection = (-quadraticCoef.y + sqrtDiscriminant) / (2.0f * quadraticCoef.x);
    }
    return intersection;
}

// UE 5.7.4 LumenRadianceCacheInterpolation.ush:277. The one-mip diffuse
// specialization uses the bordered atlas at mip zero without sky visibility.
float3 sampleRadianceCacheProbeWithParallaxCorrection(LumenRadianceCacheInputs inputs, LumenRadianceCacheClipmap clipmap,
    uint3 probeCoord, uint probeIndex, float3 worldPosition, float3 worldDirection)
{
    float3 probeWorldPosition = lumenRadianceProbeWorldPosition(probeCoord, clipmap);
    float reprojectionRadius = clipmap.traceSettings.z * clipmap.traceSettings.x;
    float t = radianceCacheRaySphereExit(worldPosition, worldDirection, probeWorldPosition, reprojectionRadius);
    float3 intersectionPosition = worldPosition + worldDirection * t;
    float3 reprojectedDirection = intersectionPosition - probeWorldPosition;
    float correctionFactor = t * t / (reprojectionRadius * dot(reprojectedDirection, worldDirection));

    float2 probeUV = lumenInverseEquiAreaSphericalMapping(reprojectedDirection);
    uint probeResolution = clipmap.layout.z;
    uint atlasProbesPerAxis = clipmap.layout.w;
    uint finalProbeResolution = probeResolution + 2u;
    uint2 probeAtlasCoord = finalProbeResolution * uint2(probeIndex % atlasProbesPerAxis, probeIndex / atlasProbesPerAxis);
    float2 probeTexelCoord = probeUV * float(probeResolution) + 1.0f;
    float invProbeFinalRadianceAtlasResolution = 1.0f / float(atlasProbesPerAxis * finalProbeResolution);
    float2 probeAtlasUV = (float2(probeAtlasCoord) + probeTexelCoord) * invProbeFinalRadianceAtlasResolution;
    float3 radiance = inputs.radiance.SampleLevel(inputs.sampler, probeAtlasUV, 0.0f).xyz;
    return radiance * correctionFactor;
}

// UE 5.7.4 LumenRadianceCacheInterpolation.ush:418, default trilinear interpolation.
public float3 sampleLumenRadianceCache(LumenRadianceCacheInputs inputs, LumenRadianceCacheCoverage coverage,
    float3 worldPosition, float3 worldDirection)
{
    LumenRadianceCacheClipmap clipmap = inputs.clipmaps[coverage.clipmapIndex];
    uint3 corner = coverage.corner;
    float3 sample000 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(0, 0, 0),
        coverage.probeIndices[0], worldPosition, worldDirection);
    float3 sample001 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(0, 0, 1),
        coverage.probeIndices[1], worldPosition, worldDirection);
    float3 sample010 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(0, 1, 0),
        coverage.probeIndices[2], worldPosition, worldDirection);
    float3 sample011 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(0, 1, 1),
        coverage.probeIndices[3], worldPosition, worldDirection);
    float3 sample100 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(1, 0, 0),
        coverage.probeIndices[4], worldPosition, worldDirection);
    float3 sample101 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(1, 0, 1),
        coverage.probeIndices[5], worldPosition, worldDirection);
    float3 sample110 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(1, 1, 0),
        coverage.probeIndices[6], worldPosition, worldDirection);
    float3 sample111 = sampleRadianceCacheProbeWithParallaxCorrection(inputs, clipmap, corner + uint3(1, 1, 1),
        coverage.probeIndices[7], worldPosition, worldDirection);

    float3 lerpAlphas = coverage.lerpAlphas;
    float3 zLerp00 = lerp(sample000, sample001, lerpAlphas.z);
    float3 zLerp01 = lerp(sample010, sample011, lerpAlphas.z);
    float3 zLerp10 = lerp(sample100, sample101, lerpAlphas.z);
    float3 zLerp11 = lerp(sample110, sample111, lerpAlphas.z);
    float3 yLerp0 = lerp(zLerp00, zLerp01, lerpAlphas.y);
    float3 yLerp1 = lerp(zLerp10, zLerp11, lerpAlphas.y);
    return lerp(yLerp0, yLerp1, lerpAlphas.x);
}
} // namespace Metallic.Lumen
