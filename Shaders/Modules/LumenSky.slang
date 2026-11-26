module LumenSky;

import ShaderCore;
import Lighting;
using Metallic;

namespace Metallic.Lumen
{
public struct LumenSkyParameters
{
    ResourceHandle<Texture2D<float4>> radiance;
    SamplerHandle sampler;
    float rotationRadians;
    float intensity;
};

// UE ApplySkylightToTraceResult queries the unblurred skylight (zero cone angle).
public float3 sampleLumenSky(LumenSkyParameters sky, float3 worldDirection)
{
    if (sky.intensity <= 0.0f) { return 0.0f; }
    float sine, cosine;
    sincos(sky.rotationRadians, sine, cosine);
    float3 localDirection = float3(cosine * worldDirection.x - sine * worldDirection.z, worldDirection.y,
        sine * worldDirection.x + cosine * worldDirection.z);
    if (sky.radiance.index == 0xffffffffu) {
        return Metallic.Lighting::proceduralEnvironmentLocalRadiance(localDirection) * sky.intensity;
    }
    return resolveUniform(sky.radiance).SampleLevel(resolveUniform(sky.sampler),
        Metallic.Lighting::environmentFilterUv(localDirection), 0.0f).rgb * sky.intensity;
}
} // namespace Metallic.Lumen
