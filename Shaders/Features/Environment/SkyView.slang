import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void skyViewMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    float3 direction = skyDirectionFromUV(p, (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy));
    SpectralSegment integrated = integrateEnvironment(p, resolveUniform(gParams.transmittance),
        resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, 1e9, true);
    float3 radiance = spectralToWorking(integrated.radiance);
    // No source disks: sky capture, SH, PDF and GGX own scattering only.
    resolveUniform(gParams.output)[id.xy] = float4(clamp(radiance, 0.0, 65504.0), 1.0);
}
