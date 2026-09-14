#language slang 2026
implementing Lighting;

namespace Metallic.Lighting
{
public float3 proceduralEnvironmentLocalRadiance(float3 direction)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    float sun = pow(saturate(dot(direction, normalize(float3(0.35, 0.88, 0.22)))), 256.0);
    return lerp(float3(0.62, 0.70, 0.78), float3(0.12, 0.24, 0.48), t) * 0.45 +
        sun * float3(7.0, 6.4, 5.3);
}
} // namespace Metallic.Lighting
