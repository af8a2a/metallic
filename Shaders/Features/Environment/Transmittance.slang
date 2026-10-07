import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void transmittanceMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float2 uv = float2(id.xy) / float2(gParams.dimensions.xy - 1u);
    float radius = lerp(p.observerPlanetBottom.w, p.observerWorldTop.w, uv.y);
    float cosine = uv.x * 2.0 - 1.0;
    float3 direction = float3(sqrt(max(1.0 - cosine * cosine, 0.0)), cosine, 0.0);
    float3 transmission = integrateTransmittance(p, float3(0.0, radius, 0.0), direction);
    resolveUniform(gParams.output)[id.xy] = float4(transmission, 1.0);
}
