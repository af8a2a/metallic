module AerialPerspective;

import Atmosphere;
import AtmosphereVirtualShadowMap;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

void buildAerialPerspectiveCell<TShadow : IAtmosphereLightShadow>(uint3 id, GPUAtmosphereParameters p, TShadow shadow,
    bool includeViewCloudMedium = true)
{
    float3 direction = skyDirectionFromUV(p, (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy));
    float fraction = float(id.z) / float(gParams.dimensions.z - 1u);
    float depth = aerialDepthFromSlice(max(p.settings.x - p.settings.z, 0.0), fraction);
    AtmosphereSegment segment = integrateAerialPerspective(p, resolveUniform(gParams.transmittance),
        resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, depth, shadow, includeViewCloudMedium);
    uint index = ((id.z * gParams.dimensions.y + id.y) * gParams.dimensions.x + id.x) * 2u;
    gParams.aerial.store(index, float4(segment.radiance, p.settings.z + depth));
    gParams.aerial.store(index + 1u, float4(segment.transmittance, 1.0));
}

[numthreads(8, 8, 1)]
void aerialPerspectiveMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gParams.dimensions.xyz)) { return; }
    buildAerialPerspectiveCell(id, loadParameters(gParams.parameters), UnshadowedAtmosphere());
}

[numthreads(8, 8, 1)]
void shadowedAerialPerspectiveMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gParams.dimensions.xyz)) { return; }
    if (gParams.parameters.count == 0u) {
        // An HDRI has no intervening atmospheric medium.
        uint index = ((id.z * gParams.dimensions.y + id.y) * gParams.dimensions.x + id.x) * 2u;
        gParams.aerial.store(index, float4(0.0));
        gParams.aerial.store(index + 1u, float4(1.0));
        return;
    }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    let shadow = DirectionalAtmosphereShadow::forRay(gParams.worldShadowFaces, gParams.dimensions.w, p.observerWorldTop.xyz);
    buildAerialPerspectiveCell(id, p, shadow, (gParams.dimensions.w & kAtmosphereSeparateViewClouds) == 0u);
}
