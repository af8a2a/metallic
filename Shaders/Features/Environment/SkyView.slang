module SkyView;

import Atmosphere;
import AtmosphereVirtualShadowMap;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

void buildSkyViewTexel<TShadow : IAtmosphereLightShadow>(uint2 id, GPUAtmosphereParameters p, TShadow shadow,
    bool includeViewCloudMedium = true)
{
    float3 direction = skyDirectionFromUV(p, (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy));
    SpectralSegment integrated = integrateEnvironment(p, resolveUniform(gParams.transmittance),
        resolveUniform(gParams.multiScattering), p.observerPlanetBottom.xyz, direction, 1e9, true, shadow, 0.0, includeViewCloudMedium);
    float3 radiance = spectralToWorking(integrated.radiance);
    // No source disks: sky capture, SH, PDF and GGX own scattering only.
    resolveUniform(gParams.output)[id.xy] = float4(max(radiance, 0.0), 1.0);
}

[numthreads(8, 8, 1)]
void skyViewMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    buildSkyViewTexel(id.xy, loadParameters(gParams.parameters), UnshadowedAtmosphere());
}

[numthreads(8, 8, 1)]
void shadowedSkyViewMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    let shadow = DirectionalAtmosphereShadow::forRay(gParams.worldShadowFaces, gParams.dimensions.w, p.observerWorldTop.xyz);
    buildSkyViewTexel(id.xy, p, shadow, (gParams.dimensions.w & kAtmosphereSeparateViewClouds) == 0u);
}
