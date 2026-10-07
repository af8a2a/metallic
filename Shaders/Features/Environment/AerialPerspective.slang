import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void aerialPerspectiveMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gParams.dimensions.xyz)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float3 direction = skyDirectionFromUV(p, (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy));
    float fraction = float(id.z) / float(gParams.dimensions.z - 1u);
    float distance = aerialDistanceFromSlice(p, fraction);
    AtmosphereSegment segment = segmentIntegration(p, resolveUniform(gParams.transmittance),
        resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, distance);
    uint index = ((id.z * gParams.dimensions.y + id.y) * gParams.dimensions.x + id.x) * 2u;
    gParams.aerial.store(index, float4(segment.radiance, distance));
    gParams.aerial.store(index + 1u, float4(segment.transmittance, 1.0));
}
