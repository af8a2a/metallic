#language slang 2026
module VirtualShadowMapCommon;

namespace Metallic.Shadow
{
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
