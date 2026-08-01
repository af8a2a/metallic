// This software contains source code that calls interfaces provided by NVIDIA Corporation.
#include <rtxcr/HairChiangBSDF.hlsli>
#include <rtxcr/HairFarFieldBCSDF.hlsli>
#include <rtxcr/SubsurfaceScattering.hlsli>

struct RtxcrMaterialSamplePush
{
    uint width;
    uint height;
    uint viewMode;
    uint padding0;

    float exposure;
    float lightAzimuthDegrees;
    float hairMelanin;
    float hairMelaninRedness;

    float hairLongitudinalRoughness;
    float hairAzimuthalRoughness;
    float hairCuticleAngleDegrees;
    float hairIor;

    float sssScale;
    float sssAnisotropy;
    float sssMaxSampleRadius;
    float padding1;
};

[[vk::binding(0, 0)]] RWTexture2D<float4> gOutput;

static const float kPi = 3.14159265358979323846;

float hash11(float value)
{
    return frac(sin(value * 127.1 + 311.7) * 43758.5453123);
}

float2 lowDiscrepancy2(uint index, uint2 pixel)
{
    float rotation = hash11(float(pixel.x * 1973u + pixel.y * 9277u));
    return float2(
        frac((float(index) + 0.5) * 0.61803398875 + rotation),
        frac((float(index) + 0.5) * 0.75487766625 + rotation * 0.5));
}

bool intersectSphere(
    float3 rayOrigin,
    float3 rayDirection,
    float3 center,
    float radius,
    out float hitT)
{
    float3 offset = rayOrigin - center;
    float b = dot(offset, rayDirection);
    float c = dot(offset, offset) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        hitT = 0.0;
        return false;
    }
    float root = sqrt(discriminant);
    float nearT = -b - root;
    float farT = -b + root;
    hitT = nearT > 0.001 ? nearT : farT;
    return hitT > 0.001;
}

float3 sphereFiberTangent(float3 normal)
{
    float3 tangent = float3(0.0, 1.0, 0.0) - normal * normal.y;
    if (dot(tangent, tangent) < 0.0001) {
        tangent = float3(1.0, 0.0, 0.0) - normal * normal.x;
    }
    return normalize(tangent);
}

RTXCR_HairMaterialData makeHairMaterial(RtxcrMaterialSamplePush push)
{
    RTXCR_HairMaterialData material;
    material.baseColor = float3(0.12, 0.035, 0.012);
    material.longitudinalRoughness = push.hairLongitudinalRoughness;
    material.azimuthalRoughness = push.hairAzimuthalRoughness;
    material.ior = push.hairIor;
    material.eta = 1.0 / push.hairIor;
    material.fresnelApproximation = 1u;
    material.absorptionModel = RTXCR_HairAbsorptionModel_Normalized;
    material.melanin = push.hairMelanin;
    material.melaninRedness = push.hairMelaninRedness;
    material.cuticleAngleInDegrees = push.hairCuticleAngleDegrees;
    return material;
}

void makeHairFrame(
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    out RTXCR_HairInteractionSurface surface,
    out float3 wi,
    out float3 wo)
{
    float3 tangent = sphereFiberTangent(normal);
    float3 bitangent = normalize(cross(normal, tangent));
    float3x3 worldToFiber = float3x3(tangent, bitangent, normal);
    wi = normalize(mul(worldToFiber, lightDirection));
    wo = normalize(mul(worldToFiber, viewDirection));
    surface.incidentRayDirection = wo;
    surface.shadingNormal = float3(0.0, 0.0, 1.0);
    surface.tangent = float3(1.0, 0.0, 0.0);
}

float3 shadeChiangHair(
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    RtxcrMaterialSamplePush push)
{
    RTXCR_HairInteractionSurface surface;
    float3 wi;
    float3 wo;
    makeHairFrame(normal, viewDirection, lightDirection, surface, wi, wo);
    RTXCR_HairMaterialData material = makeHairMaterial(push);
    RTXCR_HairMaterialInteraction interaction =
        RTXCR_CreateHairMaterialInteraction(material, surface);
    float3 bcsdf = RTXCR_HairChiangBsdfEval(interaction, wi, wo);
    float grazing = pow(1.0 - saturate(dot(normal, viewDirection)), 3.0);
    return min(bcsdf * 3.0, 32.0) + float3(0.018, 0.006, 0.003) * (1.0 + grazing);
}

float3 shadeFarFieldHair(
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    RtxcrMaterialSamplePush push)
{
    RTXCR_HairInteractionSurface surface;
    float3 wi;
    float3 wo;
    makeHairFrame(normal, viewDirection, lightDirection, surface, wi, wo);
    RTXCR_HairMaterialData material = makeHairMaterial(push);
    RTXCR_HairMaterialInteractionBcsdf interaction =
        RTXCR_CreateHairMaterialInteractionBcsdf(
            material,
            float3(0.11, 0.025, 0.008),
            0.08,
            push.hairLongitudinalRoughness);
    float3 bcsdf;
    float3 diffuse;
    float pdf;
    RTXCR_HairFarFieldBcsdfEval(surface, interaction, wi, wo, bcsdf, diffuse, pdf);
    float grazing = pow(1.0 - saturate(dot(normal, viewDirection)), 2.0);
    return min((bcsdf + diffuse) * 4.0, 32.0) +
        float3(0.014, 0.004, 0.002) * (1.0 + grazing);
}

float3 shadeSubsurface(
    float3 position,
    float3 normal,
    float3 viewDirection,
    float3 lightPosition,
    float sphereRadius,
    uint2 pixel,
    RtxcrMaterialSamplePush push)
{
    RTXCR_SubsurfaceMaterialData material = RTXCR_CreateDefaultSubsurfaceMaterialData();
    material.transmissionColor = float3(0.82, 0.32, 0.22);
    material.scatteringColor = float3(1.0, 0.48, 0.32);
    material.scale = push.sssScale;
    material.g = push.sssAnisotropy;

    float3 tangent = sphereFiberTangent(normal);
    float3 bitangent = normalize(cross(normal, tangent));
    RTXCR_SubsurfaceInteraction interaction = RTXCR_CreateSubsurfaceInteraction(
        position,
        normal,
        tangent,
        bitangent);

    float3 diffusion = float3(0.0, 0.0, 0.0);
    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < 8u; ++sampleIndex) {
        RTXCR_SubsurfaceSample sample;
        float2 randomSample = clamp(
            lowDiscrepancy2(sampleIndex, pixel),
            float2(0.001, 0.001),
            float2(0.999, 0.999));
        RTXCR_EvalBurleyDiffusionProfile(
            material,
            interaction,
            push.sssMaxSampleRadius,
            false,
            randomSample,
            sample);
        float3 toLight = lightPosition - sample.samplePosition;
        float distanceSquared = max(dot(toLight, toLight), 0.05);
        float3 sampleLightDirection = normalize(toLight);
        float noL = saturate(dot(normal, sampleLightDirection));
        float3 incidentRadiance = float3(16.0, 12.5, 10.0) / distanceSquared;
        diffusion += RTXCR_EvalBssrdf(sample, incidentRadiance, noL);
    }
    diffusion /= 8.0;

    RTXCR_SubsurfaceMaterialCoefficients coefficients =
        RTXCR_ComputeSubsurfaceMaterialCoefficients(material);
    float noV = saturate(dot(normal, viewDirection));
    float pathLengthMeters = max(2.0 * sphereRadius * noV * 0.01, 0.0001);
    float3 transmission = exp(-coefficients.sigma_t * pathLengthMeters);
    float3 backDirection = normalize(lightPosition - position);
    float backLighting = pow(saturate(dot(-normal, backDirection)), 2.0);
    float fresnel = pow(1.0 - noV, 5.0);
    float3 surfaceSpecular = float3(0.04, 0.035, 0.03) * (0.08 + fresnel);
    return max(diffusion, 0.0) +
        transmission * float3(2.6, 0.45, 0.2) * backLighting +
        surfaceSpecular;
}

float3 background(float2 ndc, float aspect)
{
    float3 top = float3(0.035, 0.052, 0.082);
    float3 bottom = float3(0.006, 0.009, 0.015);
    float horizon = saturate(ndc.y * 0.45 + 0.52);
    float3 color = lerp(bottom, top, horizon);
    float vignette = saturate(1.0 - 0.18 * dot(float2(ndc.x * aspect, ndc.y), float2(ndc.x * aspect, ndc.y)));
    return color * vignette;
}

float3 acesApprox(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rtxcrMaterialSampleMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform RtxcrMaterialSamplePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float2 dimensions = float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = (float2(pixel) + 0.5) / dimensions * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float aspect = dimensions.x / dimensions.y;
    float3 rayOrigin = float3(0.0, 0.0, 5.2);
    float3 rayDirection = normalize(float3(ndc.x * aspect, ndc.y, -2.8));

    float angle = radians(push.lightAzimuthDegrees);
    float3 lightPosition = float3(sin(angle) * 4.5, 3.4, cos(angle) * 2.5 + 2.0);
    float3 color = background(ndc, aspect);

    float nearestT = 1e30;
    uint materialIndex = 0xffffffffu;
    float3 sphereCenter = float3(0.0, 0.0, 0.0);
    float sphereRadius = 0.0;
    uint firstMaterial = push.viewMode == 0u ? 0u : push.viewMode - 1u;
    uint materialCount = push.viewMode == 0u ? 3u : 1u;
    for (uint sphereIndex = 0u; sphereIndex < materialCount; ++sphereIndex) {
        uint candidateMaterial = firstMaterial + sphereIndex;
        float candidateRadius = push.viewMode == 0u ? 0.82 : 1.28;
        float candidateX = push.viewMode == 0u ? (float(sphereIndex) - 1.0) * 1.75 : 0.0;
        float3 candidateCenter = float3(candidateX, 0.0, 0.0);
        float hitT;
        if (intersectSphere(rayOrigin, rayDirection, candidateCenter, candidateRadius, hitT) &&
            hitT < nearestT) {
            nearestT = hitT;
            materialIndex = candidateMaterial;
            sphereCenter = candidateCenter;
            sphereRadius = candidateRadius;
        }
    }

    if (materialIndex != 0xffffffffu) {
        float3 position = rayOrigin + rayDirection * nearestT;
        float3 normal = normalize(position - sphereCenter);
        float3 viewDirection = normalize(-rayDirection);
        float3 lightDirection = normalize(lightPosition - position);
        float lightDistanceSquared = max(dot(lightPosition - position, lightPosition - position), 0.1);
        float lightScale = 32.0 / lightDistanceSquared;

        if (materialIndex == 0u) {
            color = shadeChiangHair(normal, viewDirection, lightDirection, push) * lightScale;
        } else if (materialIndex == 1u) {
            color = shadeFarFieldHair(normal, viewDirection, lightDirection, push) * lightScale;
        } else {
            color = shadeSubsurface(
                position,
                normal,
                viewDirection,
                lightPosition,
                sphereRadius,
                pixel,
                push);
        }

        float rim = pow(1.0 - saturate(dot(normal, viewDirection)), 4.0);
        color += float3(0.025, 0.04, 0.07) * rim;
    }

    if (push.viewMode == 0u && ndc.y < -0.78 && ndc.y > -0.81) {
        float normalizedX = ndc.x * aspect;
        if (abs(normalizedX + 1.75) < 0.55) {
            color += float3(0.7, 0.18, 0.06);
        } else if (abs(normalizedX) < 0.55) {
            color += float3(0.9, 0.42, 0.08);
        } else if (abs(normalizedX - 1.75) < 0.55) {
            color += float3(0.9, 0.24, 0.18);
        }
    }

    color = acesApprox(max(color * push.exposure, 0.0));
    color = pow(color, 1.0 / 2.2);
    gOutput[pixel] = float4(color, 1.0);
}
