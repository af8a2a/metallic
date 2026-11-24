module LumenReflectionSampling;

import LumenProbeMath;

namespace Metallic.Lumen
{
// UE 5.7.4 LumenReflectionCommon.ush, 2x2 four-rooks pattern.
public uint2 lumenReflectionScreenTileJitter(uint2 downsampledScreenCoord, uint reflectionsStateFrameIndex)
{
    uint2 cellIndex = downsampledScreenCoord % 2u;
    uint linearIndex = cellIndex.x + cellIndex.y * 2u;
    linearIndex = (linearIndex + reflectionsStateFrameIndex) % 4u;
    uint2 jitter;
    jitter.x = (linearIndex & 0x02u) != 0u ? 1u : 0u;
    jitter.y = (linearIndex & 0x01u) != 0u ? 0u : 1u;
    return jitter;
}

// UE 5.7.4 MonteCarlo.ush, isotropic material tangent basis.
public float3x3 getReflectionTangentBasis(float3 tangentZ)
{
    const float signZ = tangentZ.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -rcp(signZ + tangentZ.z);
    const float b = tangentZ.x * tangentZ.y * a;
    float3 tangentX = float3(1.0f + signZ * a * (tangentZ.x * tangentZ.x), signZ * b, -signZ * tangentZ.x);
    float3 tangentY = float3(b, signZ + a * (tangentZ.y * tangentZ.y), -tangentZ.y);
    return float3x3(tangentX, tangentY, tangentZ);
}

// UE 5.7.4 MonteCarlo.ush, VisibleGGXPDF_aniso overload with
// GGX_BOUNDED_VNDF_SAMPLING and bLimitVDNFToReflection enabled.
float visibleReflectionGGXPDF(float3 v, float3 h, float2 alpha)
{
    float noV = v.z;
    float noH = h.z;
    float voH = dot(v, h);
    float a2 = alpha.x * alpha.y;
    float3 hs = float3(alpha.y * h.x, alpha.x * h.y, a2 * noH);
    float s = dot(hs, hs);
    float d = (1.0f / kLumenProbePi) * a2 * ((a2 / s) * (a2 / s));
    float lenV = length(float3(v.x * alpha.x, v.y * alpha.y, noV));
    float a = saturate(min(alpha.x, alpha.y));
    float bound = 1.0f + length(v.xy);
    float ka2 = a * a, s2 = bound * bound;
    float k = (s2 - ka2 * s2) / (s2 + ka2 * v.z * v.z);
    return (2.0f * d * voH) / (k * noV + lenV);
}

// UE 5.7.4 MonteCarlo.ush, ImportanceSampleVisibleGGX with the bounded reflection sampling range.
public float4 importanceSampleVisibleReflectionGGX(float2 e, float2 alpha, float3 v)
{
    float3 vh = normalize(float3(alpha * v.xy, v.z));
    float phi = (2.0f * kLumenProbePi) * e.x;
    float a = saturate(min(alpha.x, alpha.y));
    float s = 1.0f + length(v.xy);
    float a2 = a * a, s2 = s * s;
    float k = (s2 - a2 * s2) / (s2 + a2 * v.z * v.z);
    float z = lerp(1.0f, -k * vh.z, e.y);
    float sinTheta = sqrt(saturate(1.0f - z * z));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 h = float3(x, y, z) + vh;
    h = normalize(float3(alpha * h.xy, max(0.0f, h.z)));
    return float4(h, visibleReflectionGGXPDF(v, h, alpha));
}
} // namespace Metallic.Lumen
