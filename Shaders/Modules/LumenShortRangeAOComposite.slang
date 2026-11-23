module LumenShortRangeAOComposite;

import LumenShortRangeAOPacking;

namespace Metallic.Lumen
{

// UE 5.7.4 DeferredShadingCommon.ush, AOMultiBounce in the non-native-half composite.
float3 aoMultiBounce(float3 baseColor, float ao)
{
    float3 a = 2.0404f * baseColor - 0.3324f;
    float3 b = -4.7951f * baseColor + 0.6417f;
    float3 c = 2.7552f * baseColor + 0.6903f;
    return max(ao, ((ao * a + b) * ao + c) * ao);
}

// UE 5.7.4 DiffuseIndirectComposite.usf, opaque diffuse with ShortRangeGI disabled.
public float3 applyLumenShortRangeAO(float3 irradianceOverPi, float3 diffuseAlbedo, uint packedBentNormal)
{
    const float maxAOMultibounceAlbedo = 0.5f;
    const float3 bentNormal = unpackLumenScreenBentNormal(packedBentNormal);
    const float ao = saturate(length(bentNormal));
    const float3 diffuseOcclusion = aoMultiBounce(min(diffuseAlbedo, maxAOMultibounceAlbedo), ao);
    return (irradianceOverPi * diffuseAlbedo) * diffuseOcclusion;
}

} // namespace Metallic.Lumen
