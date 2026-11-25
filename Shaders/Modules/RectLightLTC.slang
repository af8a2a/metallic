#language slang 2026
module RectLightLTC;

// Copyright (c) 2017 Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt.
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, SIGGRAPH 2016.
// Horizon-clipped quad from selfshadow/ltc_code, webgl/shaders/ltc/ltc_quad.fs.
// See Shaders/Licenses/PolygonalLightLTC.txt.

import Core;
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

public float3 evaluateRectLights(float3 position, float3 N, float3 V,
    float3 diffuseColor, float3 specularColor, float roughness)
{
    let rect = getResourceParameterField<SceneRealtimeResourceParameters, RectLightResourceParameters>(
        kSceneRectLightsOffset);
    if (rect.lights.resource.index == 0xffffffffu || rect.lights.count == 0u) { return float3(0.0); }
    // The named resource encoder stores span counts in 32-bit words.
    uint lightCount = rect.lights.count / (sizeof(GPURectLight) / 4u);

    float ndotv = saturate(dot(N, V));
    float2 uv = float2(roughness, sqrt(1.0 - ndotv));
    uv = uv * (63.0 / 64.0) + (0.5 / 64.0);
    let samplerState = resolveUniform(getSceneResourceParameterField<SamplerHandle>(kSceneOpenPBRLUTSamplerOffset));
    float4 t1 = resolveUniform(rect.matrixLUT).SampleLevel(samplerState, uv, 0);
    float4 t2 = resolveUniform(rect.amplitudeLUT).SampleLevel(samplerState, uv, 0);

    // Transpose the constructor's column arrangement from the GLSL reference.
    float3x3 Minv = float3x3(
        float3(t1.x, 0, t1.z),
        float3(   0, 1,    0),
        float3(t1.y, 0, t1.w));
    float3 specularScale = specularColor * t2.x + (1.0 - specularColor) * t2.y;

    float3 T1 = V - N * dot(V, N);
    T1 = dot(T1, T1) > 1e-12 ? normalize(T1) : openPBRMakeBasis(N).t;
    float3 T2 = cross(N, T1);
    float3x3 basis = float3x3(T1, T2, N);
    float3x3 identity = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);

    float3 radiance = float3(0.0);
    let buffer = resolveUniform(rect.lights.resource);
    for (uint index = 0u; index < lightCount; ++index) {
        let light = buffer.Load<GPURectLight>(rect.lights.byteOffset + index * sizeof(GPURectLight));
        float3 center = light.positionRange.xyz - position;
        // Convention: the emitter faces away from the normal of its LTC winding.
        if (dot(center, light.direction.xyz) >= 0.0) { continue; }
        float attenuation = 1.0;
        if (light.positionRange.w > 0.0) {
            float ratio = length(center) / light.positionRange.w;
            attenuation = saturate(1.0 - ratio * ratio * ratio * ratio);
            if (attenuation == 0.0) { continue; }
        }
        float3 ex = light.halfWidth.xyz;
        float3 ey = light.halfHeight.xyz;
        float3 points[4] = {center - ex - ey, center + ex - ey, center + ex + ey, center - ex + ey};

        float3 spec = evaluateLTC(basis, Minv, points) * specularScale;
        float diff = evaluateLTC(basis, identity, points);
        radiance += light.radiance.rgb * (spec + diffuseColor * diff) * attenuation;
    }
    return radiance;
}

} // namespace Metallic.Lighting
