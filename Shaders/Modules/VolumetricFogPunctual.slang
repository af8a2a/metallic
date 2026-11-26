module VolumetricFogPunctual;

import Lighting;
import VirtualShadowMapWorld;
using Metallic.Lighting;
using Metallic.Shadow;

public float fogPhaseFunction(float g, float cosTheta)
{
    float numer = 1.0f - g * g;
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return numer / (4.0f * 3.14159265358979323846f * denom * sqrt(denom));
}

// FSpotLightSceneProxy clamps authored cones before building the two attenuation constants.
// Metallic records store cosines. Common cones already satisfying both bounds retain these values.
public float2 fogSpotAngles(float2 coneCosines)
{
    static const float maxInner = 89.0f * 3.14159265358979323846f / 180.0f;
    const float minWidth = 0.001f;
    if (coneCosines.x - coneCosines.y > minWidth && coneCosines.y >= cos(maxInner + minWidth)
        && coneCosines.x >= cos(maxInner)) {
        return float2(coneCosines.y, rcp(coneCosines.x - coneCosines.y));
    }
    float inner = clamp(acos(coneCosines.x), 0.0f, maxInner);
    float outer = clamp(acos(coneCosines.y), inner + minWidth, maxInner + minWidth);
    float cosOuter = cos(outer);
    return float2(cosOuter, rcp(cos(inner) - cosOuter));
}

// LightGridInjection.cpp::PackLightColor followed by LightData.ush::UnpackLightColor.
// The shared Metallic light record retains full-precision surface lighting.
float3 fogGridLightColor(float3 lightColor)
{
    float colorLength = length(lightColor);
    float3 direction = colorLength > 1e-8f ? lightColor * rcp(colorLength) : 0.0f;
    uint3 packedDirection = uint3(direction * 1023.0f) & 1023u;
    return (colorLength / 1023.0f) * float3(packedDirection);
}

// UE VolumetricFog.usf: the clustered and shadowed branches share attenuation
// and phase for point/spot lights with zero source length and inverse-square falloff.
public float3 fogPunctualScattering(ClusterLightGridParams grid, StructuredBuffer<ClusterLightData> lights,
    StructuredBuffer<ClusterLightGridCell> cells, StructuredBuffer<uint> indices, StructuredBuffer<uint> candidates,
    WorldShadowFaces worldShadowFaces, uint2 pixel, float sceneDepthMeters, float3 eyeMeters, float3 cellCm, float3 cameraVector,
    float cellRadius, float phaseG)
{
    ClusterLightGridLookup lookup = clusterLightGridLookup(grid, cells, pixel, sceneDepthMeters);
    float distanceBiasSqr = max(cellRadius, 1.0f);
    distanceBiasSqr *= distanceBiasSqr;
    float3 worldPosition = eyeMeters + cellCm * 0.01f;
    float3 lightScattering = 0.0f;
    for (uint i = 0; i < lookup.count + clusterLightGridGlobalLightCount(grid); ++i) {
        uint index = i < lookup.count ? clusterLightGridLocalLightIndex(lookup, indices, candidates, i)
            : clusterLightGridGlobalLightIndex(grid, candidates, i - lookup.count);
        if (index >= grid.counts.y) { continue; } // Rectangles follow the punctual source slots.
        ClusterLightData light = lights[index];
        bool isSpotLight = light.directionType.w > 1.5f;
        let face = getLocalWorldShadowFace(worldShadowFaces, index, isSpotLight, worldPosition - light.positionRange.xyz);
        bool hasShadow = face.pageSize != 0u;
        // DeferredLightUniforms supply full precision for shadowed injection.
        // Only UE's unshadowed forward light grid packs color and intensity.
        float intensity = hasShadow ? light.spot.w : f16tof32(f32tof16(light.spot.w));
        if (intensity > 0.0f) {
            float3 toLight = (light.positionRange.xyz - eyeMeters) * 100.0f - cellCm;
            float distanceSqr = dot(toLight, toLight);
            float3 L = toLight * rsqrt(distanceSqr);
            float invRadius = light.positionRange.w > 0.0f ? rcp(max(0.00001f, light.positionRange.w * 100.0f)) : 0.0f;
            float radiusRatioSqr = distanceSqr * (invRadius * invRadius);
            float lightMask = saturate(1.0f - radiusRatioSqr * radiusRatioSqr);
            lightMask *= lightMask;
            if (isSpotLight) {
                float2 angles = fogSpotAngles(light.spot.xy);
                float cone = saturate((dot(L, -light.directionType.xyz) - angles.x) * angles.y);
                lightMask *= cone * cone;
            }
            lightMask = applyPunctualIESProfile(light, -L, lightMask);
            float lighting = rcp(distanceSqr + distanceBiasSqr);
            float combinedAttenuation = lighting * lightMask;
            // Metallic stores candela. UE applies its cd-to-cm² conversion at light upload.
            float3 lightColor = light.colorIntensity.rgb * (light.colorIntensity.w * 10000.0f);
            if (!hasShadow) { lightColor = fogGridLightColor(lightColor); }
            float shadowFactor = 1.0f;
            if (hasShadow && combinedAttenuation > 0.0f) {
                shadowFactor = sampleLocalVirtualShadowMap(face, worldPosition);
            }
            lightScattering += lightColor * (fogPhaseFunction(phaseG, dot(L, -cameraVector)) * combinedAttenuation * shadowFactor * intensity);
        }
    }
    return lightScattering;
}
