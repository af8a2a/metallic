implementing SceneSurface;

import MaterialTextureDecode;
import AlphaCoverage;
import Core;
import Material;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import MaterialProgram;
import Lighting;
import RTXCRHair;
import ScenePathTypes;
import ScenePathMath;
import RTXCRMaterialAdapter;
import MaterialValueTextureSampling;
import RayMaterialExecution;
import StreamTextureSampling;
import WorkingColor;

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.ScenePath;

// Keep path math unambiguous when programs also import GPUDriven helpers.
public float3 safeNormalize(float3 value, float3 fallback)
{
    return Metallic.ScenePath.safeNormalize(value, fallback);
}
public float3 sampleCosineHemisphere(float3 normal, inout uint seed)
{
    float r0 = rand(seed);
    float r1 = rand(seed);
    float phi = 2.0 * kPi * r0;
    float cosTheta = sqrt(1.0 - r1);
    float sinTheta = sqrt(r1);

    float3 tangent = abs(normal.z) < 0.999
        ? safeNormalize(cross(float3(0.0, 0.0, 1.0), normal), float3(1.0, 0.0, 0.0))
        : safeNormalize(cross(float3(0.0, 1.0, 0.0), normal), float3(1.0, 0.0, 0.0));
    float3 bitangent = cross(normal, tangent);
    return safeNormalize(
        tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normal * cosTheta,
        normal);
}
public float3 transformObjectNormalToWorld(float3 normal, float3x4 worldToObject)
{
    return mul(normal, worldToObject).xyz;
}

public float3 faceForwardShadingNormal(float3 normal, float3 geometryNormal, float3 rayDirection)
{
    float3 facingGeometryNormal = dot(geometryNormal, -rayDirection) < 0.0
        ? -geometryNormal
        : geometryNormal;
    if (dot(normal, facingGeometryNormal) < 0.0) {
        normal = -normal;
    }
    return safeNormalize(normal, facingGeometryNormal);
}

public float luminanceMax(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

public float fresnelDielectric(float ior, float cosTheta)
{
    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;
    float x = 1.0 - saturate(cosTheta);
    return saturate(f0 + (1.0 - f0) * x * x * x * x * x);
}

public bool refractDirection(float3 incident, float3 normal, float eta, out float3 transmitted)
{
    float cosTheta = saturate(dot(-incident, normal));
    float sin2Theta = max(0.0, 1.0 - cosTheta * cosTheta);
    float sin2Transmitted = eta * eta * sin2Theta;
    if (sin2Transmitted >= 1.0) {
        transmitted = float3(0.0, 0.0, 0.0);
        return false;
    }

    float cosTransmitted = sqrt(max(0.0, 1.0 - sin2Transmitted));
    transmitted = safeNormalize(eta * incident + (eta * cosTheta - cosTransmitted) * normal, incident);
    return true;
}

public float3 powColor(float3 value, float exponent)
{
    float3 clamped = max(value, float3(0.001, 0.001, 0.001));
    return float3(pow(clamped.x, exponent), pow(clamped.y, exponent), pow(clamped.z, exponent));
}

public float3 volumeTransmittance(float3 baseColor, float transmission, float thickness, float attenuationDistance, float3 attenuationColor, float distance)
{
    float exponent = 0.0;
    if (attenuationDistance > 0.0001) {
        exponent = max(distance, 0.0) / attenuationDistance;
    }

    float3 attenuation = powColor(max(attenuationColor, float3(0.001, 0.001, 0.001)), exponent);
    float3 baseTint = lerp(float3(1.0, 1.0, 1.0), powColor(baseColor, 0.5), transmission);
    return saturate(attenuation * baseTint);
}

public float3 srgbToLinear(float3 color)
{
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}

public float fallbackBsdfPdf(
    float3 incidentDirection,
    float3 normal,
    float3 sampledDirection,
    float metallic,
    float roughness,
    float diffuseTransmission)
{
    float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
    float transmissionChance = saturate(diffuseTransmission * (1.0 - metallic));
    float frontCos = saturate(dot(normal, sampledDirection));
    float backCos = saturate(dot(-normal, sampledDirection));
    float diffusePdf = frontCos / kPi;
    float transmissionPdf = backCos / kPi;

    float3 reflectedDirection = safeNormalize(reflect(incidentDirection, normal), normal);
    float specularPower = lerp(96.0, 8.0, roughness);
    float specularPdf = ((specularPower + 1.0) / (2.0 * kPi)) *
        pow(saturate(dot(reflectedDirection, sampledDirection)), specularPower);
    return specularChance * specularPdf +
        (1.0 - specularChance) *
            ((1.0 - transmissionChance) * diffusePdf + transmissionChance * transmissionPdf);
}

public float3 applyNormalMap(
    float3 normal,
    float3 geometryNormal,
    float4 tangent,
    float3 bitangent,
    float4 normalSample,
    float normalScale)
{
    normal = safeNormalize(normal, geometryNormal);
    float3 referenceAxis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
    float3 tangentDirection = tangent.xyz - normal * dot(normal, tangent.xyz);
    tangentDirection = safeNormalize(tangentDirection, fallbackTangent);
    float tangentSign = tangent.w >= 0.0 ? 1.0 : -1.0;
    float3 bitangentFallback = cross(normal, tangentDirection) * tangentSign;
    float3 bitangentDirection = safeNormalize(bitangent - normal * dot(normal, bitangent), bitangentFallback);

    float3 tangentNormal = float3(normalSample.xy * 2.0 - 1.0, normalSample.z * 2.0 - 1.0);
    tangentNormal.xy *= max(normalScale, 0.0);
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    return safeNormalize(
        tangentDirection * tangentNormal.x +
        bitangentDirection * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

} // namespace Metallic.SceneSurface
