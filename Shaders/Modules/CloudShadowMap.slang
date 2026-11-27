#language slang 2026
module CloudShadowMap;

import AtmosphereTypes;
import Clouds;
import ShaderCore;

namespace Metallic.Clouds
{
// Orthographic FLookAtMatrix / FReversedZOrthoMatrix from UE's PrepareCloudShadowMapLightData,
// expressed in planet-relative km. The two independent light projections occupy adjacent tiles.
public struct CloudShadowProjection
{
    public float3 anchor;
    public float3 forward;
    public float3 right;
    public float3 up;
    public float halfWidthKm;

    public static CloudShadowProjection forLight(GPUAtmosphereParameters p, uint slot)
    {
        CloudShadowProjection v;
        v.anchor = slot == 0u ? p.cloudShadowCentre.xyz : p.cloudShadowMoonAnchor.xyz;
        v.forward = -(slot == 0u ? p.sunDirectionRadius.xyz : p.moonDirectionRadius.xyz);
        float3 up = abs(v.forward.y) > 0.99f ? float3(1, 0, 0) : float3(0, 1, 0);
        v.right = normalize(cross(v.forward, up));
        v.up = cross(v.right, v.forward);
        v.halfWidthKm = p.cloudShadowCentre.w;
        return v;
    }

    public float3 nearPlanePosition(float2 uv)
    {
        float2 clip = (uv - 0.5f) * float2(2.0f, -2.0f);
        return anchor + (right * clip.x + up * clip.y - forward) * halfWidthKm;
    }

    public float3 project(float3 positionKm)
    {
        float3 delta = positionKm - anchor;
        float2 uv = 0.5f + float2(dot(delta, right), -dot(delta, up)) * (0.5f / halfWidthKm);
        return float3(uv, dot(delta, forward) + halfWidthKm);
    }
}

// VolumetricCloudCommon.ush GetCloudVolumetricShadow: front depth (km),
// mean positive extinction (/m), and maximum integrated optical depth.
public float receiverTransmittance(float3 data, float sampleDepthKm)
{
    float opticalDepth = data.y * (max(0.0f, sampleDepthKm - data.x) * 1000.0f);
    return saturate(exp(-min(data.z, opticalDepth)));
}

public float4 sampleShadowColumn(GPUAtmosphereParameters p, Texture2D<float4> atlas, float2 uv, uint slot)
{
    uint width, height;
    atlas.GetDimensions(width, height);
    // Clamp to this light's texel centres before applying atlas coordinates.
    float inset = 0.5f / float(height);
    float2 tileUV = clamp(uv, inset, 1.0f - inset);
    return atlas.SampleLevel(resolveUniform(SamplerHandle(p.cloudShapeResources.w)),
        float2((float(slot) + tileUV.x) * 0.5f, tileUV.y), 0.0f);
}
} // namespace Metallic.Clouds
