#language slang 2026
module RectLightLTC;

// Copyright (c) 2017 Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt.
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, SIGGRAPH 2016.
// Horizon-clipped quad from selfshadow/ltc_code, webgl/shaders/ltc/ltc_quad.fs.
// See Shaders/Licenses/PolygonalLightLTC.txt.

import OpenPBR;

namespace Metallic.Lighting
{
using Metallic.OpenPBR;

public struct GPURectLight
{
    public float4 positionRange;
    public float4 direction;
    public float4 halfWidth;
    public float4 halfHeight;
    public float4 radiance;
};

float3 integrateEdgeVector(float3 v1, float3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206*y)*y;
    float b = 3.4175940 + (4.1616724 + y)*y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5*rsqrt(max(1.0 - x*x, 1e-7)) - v;

    return cross(v1, v2)*theta_sintheta;
}

float integrateEdge(float3 v1, float3 v2)
{
    return integrateEdgeVector(v1, v2).z;
}

void clipQuadToHorizon(inout float3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0) {
        // clip all
    } else if (config == 1) { // V1 clip V2 V3 V4
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    } else if (config == 2) { // V2 clip V1 V3 V4
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if (config == 3) { // V1 V2 clip V3 V4
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    } else if (config == 4) { // V3 clip V1 V2 V4
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    } else if (config == 5) { // V1 V3 clip V2 V4; impossible
        n = 0;
    } else if (config == 6) { // V2 V3 clip V1 V4
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if (config == 7) { // V1 V2 V3 clip V4
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    } else if (config == 8) { // V4 clip V1 V2 V3
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    } else if (config == 9) { // V1 V4 clip V2 V3
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    } else if (config == 10) { // V2 V4 clip V1 V3; impossible
        n = 0;
    } else if (config == 11) { // V1 V2 V4 clip V3
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    } else if (config == 12) { // V3 V4 clip V1 V2
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    } else if (config == 13) { // V1 V3 V4 clip V2
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    } else if (config == 14) { // V2 V3 V4 clip V1
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    } else if (config == 15) { // V1 V2 V3 V4
        n = 4;
    }

    if (n == 3) { L[3] = L[0]; }
    if (n == 4) { L[4] = L[0]; }
}

float evaluateLTC(float3x3 basis, float3x3 Minv, float3 points[4])
{
    // points are already relative to the receiver. Slang vectors initialize matrix rows.
    Minv = mul(Minv, basis);

    float3 L[5];
    L[0] = mul(Minv, points[0]);
    L[1] = mul(Minv, points[1]);
    L[2] = mul(Minv, points[2]);
    L[3] = mul(Minv, points[3]);

    int n;
    clipQuadToHorizon(L, n);
    if (n == 0) { return 0.0; }

    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);

    float sum = 0.0;
    sum += integrateEdge(L[0], L[1]);
    sum += integrateEdge(L[1], L[2]);
    sum += integrateEdge(L[2], L[3]);
    if (n >= 4) {
        L[4] = normalize(L[4]);
        sum += integrateEdge(L[3], L[4]);
    }
    if (n == 5) { sum += integrateEdge(L[4], L[0]); }
    return max(0.0, sum);
}

public struct RectLightLTCReceiver
{
    public float3x3 basis;
    public float3x3 inverseTransform;
    public float3 specularScale;
    public float3 diffuseColor;
};

public RectLightLTCReceiver prepareRectLightLTC(float3 N, float3 V, float3 diffuseColor,
    float3 specularColor, float roughness, Texture2D<float4> matrixLUT,
    Texture2D<float4> amplitudeLUT, SamplerState samplerState)
{
    float ndotv = saturate(dot(N, V));
    float2 uv = float2(roughness, sqrt(1.0 - ndotv));
    uv = uv * (63.0 / 64.0) + (0.5 / 64.0);
    float4 t1 = matrixLUT.SampleLevel(samplerState, uv, 0);
    float4 t2 = amplitudeLUT.SampleLevel(samplerState, uv, 0);

    RectLightLTCReceiver result;
    // Transpose the constructor's column arrangement from the GLSL reference.
    result.inverseTransform = float3x3(
        float3(t1.x, 0, t1.z),
        float3(   0, 1,    0),
        float3(t1.y, 0, t1.w));
    result.specularScale = specularColor * t2.x + (1.0 - specularColor) * t2.y;
    result.diffuseColor = diffuseColor;

    float3 T1 = V - N * dot(V, N);
    T1 = dot(T1, T1) > 1e-12 ? normalize(T1) : openPBRMakeBasis(N).t;
    float3 T2 = cross(N, T1);
    result.basis = float3x3(T1, T2, N);
    return result;
}

public float rectLightAttenuation(GPURectLight light, float3 center)
{
    // The emitter faces away from the normal of its LTC winding.
    if (dot(center, light.direction.xyz) >= 0.0) { return 0.0; }
    if (light.positionRange.w > 0.0) {
        float ratio = length(center) / light.positionRange.w;
        return saturate(1.0 - ratio * ratio * ratio * ratio);
    }
    return 1.0;
}

static const float3x3 kRectLTCIdentity = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);

void makeRectLightCorners(GPURectLight light, float3 center, out float3 points[4])
{
    float3 ex = light.halfWidth.xyz;
    float3 ey = light.halfHeight.xyz;
    points[0] = center - ex - ey;
    points[1] = center + ex - ey;
    points[2] = center + ex + ey;
    points[3] = center - ex + ey;
}

public float3x3 rectLightLambertBasis(float3 normal)
{
    float3 tangent = openPBRMakeBasis(normal).t;
    return float3x3(tangent, cross(normal, tangent), normal);
}

public float3 evaluateRectLightDiffuseIrradiance(GPURectLight light, float3 position, float3x3 basis)
{
    float3 center = light.positionRange.xyz - position;
    float attenuation = rectLightAttenuation(light, center);
    if (attenuation == 0.0) { return float3(0.0); }
    float3 points[4];
    makeRectLightCorners(light, center, points);
    // The reference integrates cosine / pi. Surface Cache stores incident irradiance.
    float irradiance = 3.1415926535897932f * evaluateLTC(basis, kRectLTCIdentity, points);
    return light.radiance.rgb * irradiance * attenuation;
}

public float3 evaluateRectLightLTC(RectLightLTCReceiver receiver, GPURectLight light,
    float3 center, float attenuation)
{
    float3 points[4];
    makeRectLightCorners(light, center, points);
    float3 spec = evaluateLTC(receiver.basis, receiver.inverseTransform, points) * receiver.specularScale;
    float diff = evaluateLTC(receiver.basis, kRectLTCIdentity, points);
    return light.radiance.rgb * (spec + receiver.diffuseColor * diff) * attenuation;
}

} // namespace Metallic.Lighting
