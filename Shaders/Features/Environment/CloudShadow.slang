import Atmosphere;
import Clouds;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void cloudShadowMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float3 up, east, north; radialBasis(p, up, east, north);
    float2 offset = ((float2(id.xy) + 0.5) / float2(gParams.dimensions.xy) * 2.0 - 1.0) * p.cloudShadowCentre.w;
    float3 point = normalize(p.cloudShadowCentre.xyz + east * offset.x + north * offset.y) *
        (p.observerPlanetBottom.w + 0.002);
    float4 result = float4(1.0, 1.0, 0.0, 0.0);
    uint mask = uint(p.cloudShadowBudget.z);
    if (Metallic.Clouds::enabled(p)) {
        if ((mask & 1u) != 0u) {
            result.x = Metallic.Clouds::transmittance(p, point, p.sunDirectionRadius.xyz, 1e9, uint(p.cloudShadowBudget.x));
            result.z = 1.0;
        }
        if ((mask & 2u) != 0u) {
            result.y = Metallic.Clouds::transmittance(p, point, p.moonDirectionRadius.xyz, 1e9, uint(p.cloudShadowBudget.y));
            result.w = 1.0;
        }
    }
    resolveUniform(gParams.output)[id.xy] = result;
}
