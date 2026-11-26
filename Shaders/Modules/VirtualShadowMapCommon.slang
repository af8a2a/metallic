#language slang 2026
module VirtualShadowMapCommon;

namespace Metallic.Shadow
{
// GPU projection buffer shared by the VSM rasterizer, surface receivers and
// world-space receivers. Local lights use the same matrix for every page mip.
public struct VirtualShadowMapProjection
{
    public float4 rowX, rowY, rowZ, rowW, metric;
    public uint4 page; // Virtual-page offset, pages per axis, level, coarse flag.
    public float4 originNear;
};

public float4 virtualShadowMapProjectClip(float3 position, VirtualShadowMapProjection projection)
{
    float4 p = float4(position, 1.0);
    return float4(dot(projection.rowX, p), dot(projection.rowY, p), dot(projection.rowZ, p), dot(projection.rowW, p));
}

public uint2 virtualShadowMapLocalTexel(float2 ndc, uint resolution)
{
    uint2 texel = uint2(floor((ndc * 0.5 + 0.5) * float(resolution)));
    return min(texel, uint2(resolution - 1u));
}

// Canonical cubemap order: +X, -X, +Y, -Y, +Z, -Z. Exact edge/corner
// ties belong to X before Y before Z, including the light origin (+X).
public uint virtualShadowMapPointFace(float3 delta)
{
    float3 magnitude = abs(delta);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) { return delta.x >= 0.0 ? 0u : 1u; }
    if (magnitude.y >= magnitude.z) { return delta.y >= 0.0 ? 2u : 3u; }
    return delta.z >= 0.0 ? 4u : 5u;
}

// All faces keep right x up = forward, matching the directional raster winding.
public void virtualShadowMapPointBasis(uint face, out float3 right, out float3 up, out float3 forward)
{
    switch (face) {
    case 0u: right = float3(0, 0, -1); up = float3(0, 1, 0); forward = float3(1, 0, 0); break;
    case 1u: right = float3(0, 0, 1); up = float3(0, 1, 0); forward = float3(-1, 0, 0); break;
    case 2u: right = float3(1, 0, 0); up = float3(0, 0, -1); forward = float3(0, 1, 0); break;
    case 3u: right = float3(1, 0, 0); up = float3(0, 0, 1); forward = float3(0, -1, 0); break;
    case 4u: right = float3(1, 0, 0); up = float3(0, 1, 0); forward = float3(0, 0, 1); break;
    default: right = float3(-1, 0, 0); up = float3(0, 1, 0); forward = float3(0, 0, -1); break;
    }
}
} // namespace Metallic.Shadow
