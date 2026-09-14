#language slang 2026
module NrdEncoding;

// Canonical owner of NRD's frontend headers. Denoiser passes have their own
// program-local macro permutations and never import this encoding module.
#define NRD_HEADER_ONLY
#include "Denoising/NRD.hlsli"

namespace Metallic.Interop.NrdEncoding
{
public static const float kNrdFp16Max = NRD_FP16_MAX;

public float4 packNrdNormalRoughness(float3 normal, float roughness, float materialId)
{
    return NRD_FrontEnd_PackNormalAndRoughness(normal, roughness, materialId);
}

public float4 packNrdRadianceHitDistance(float3 radiance, float hitDistance, bool sanitize)
{
    return RELAX_FrontEnd_PackRadianceAndHitDist(radiance, hitDistance, sanitize);
}

public float packNrdPenumbra(float hitDistance, float tanAngularRadius)
{
    return SIGMA_FrontEnd_PackPenumbra(hitDistance, tanAngularRadius);
}

public float packNrdPenumbra(float hitDistance, float lightDistance, float lightDiameter)
{
    return SIGMA_FrontEnd_PackPenumbra(hitDistance, lightDistance, lightDiameter);
}
} // namespace Metallic.Interop.NrdEncoding
