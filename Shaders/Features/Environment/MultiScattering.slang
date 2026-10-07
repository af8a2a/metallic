import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void multiScatteringMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float2 uv = float2(id.xy) / float2(gParams.dimensions.xy - 1u);
    float radius = lerp(p.observerPlanetBottom.w + 0.002, p.observerWorldTop.w, uv.y);
    float cosine = uv.x * 2.0 - 1.0;
    float3 sourceDirection = float3(sqrt(max(1.0 - cosine * cosine, 0.0)), cosine, 0.0);
    float3 secondOrder = float3(0.0), response = float3(0.0);
    // Equal-area Fibonacci quadrature. The first integral uses an isotropic
    // phase; closure integrates all higher orders as Psi_2 / (1 - f_ms).
    for (uint sample = 0; sample < 32; ++sample) {
        float y = 1.0 - 2.0 * (float(sample) + 0.5) / 32.0;
        float azimuth = float(sample) * 2.399963229728653;
        float horizontal = sqrt(max(1.0 - y * y, 0.0));
        float3 direction = float3(cos(azimuth) * horizontal, y, sin(azimuth) * horizontal);
        SpectralSegment integrated = integrateSpectral(p, resolveUniform(gParams.transmittance),
            resolveUniform(gParams.transmittance), float3(0.0, radius, 0.0), direction, 1e9,
            sourceDirection, float3(1.0), true, false, true);
        secondOrder += integrated.radiance / 32.0;
        response += integrated.scatteringResponse / 32.0;
    }
    float3 multiple = secondOrder / max(float3(1.0) - min(response, float3(0.999)), float3(0.001));
    resolveUniform(gParams.output)[id.xy] = float4(min(multiple, float3(65504.0)), 1.0);
}
