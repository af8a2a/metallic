#language slang 2026
module CloudLightEnvironment;

import AtmosphereTypes;
import ShaderCore;

namespace Metallic.Atmosphere
{
// All three texels are unexposed spectra in the same units as the atmosphere transport.
public float3 cloudDistantSkyLuminance(GPUAtmosphereParameters p)
{
    return resolveUniform(ResourceHandle<Texture2D<float4>>(asuint(p.settings.w))).Load(int3(0, 0, 0)).xyz;
}

public float3 cloudAtmosphereLightIlluminanceOnGround(GPUAtmosphereParameters p, uint lightIndex)
{
    return resolveUniform(ResourceHandle<Texture2D<float4>>(asuint(p.settings.w))).Load(int3(1u + lightIndex, 0, 0)).xyz;
}
}
