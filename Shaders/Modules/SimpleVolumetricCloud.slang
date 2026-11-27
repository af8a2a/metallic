#language slang 2026
module SimpleVolumetricCloud;

import ShaderCore;

namespace Metallic.Clouds
{
// m_SimpleVolumetricCloud_Inst, UE 5.7.4. StormClouds=0, UseNoise3=false,
// CloudPerTypeScale.rgb=CloudType.rgb=1, CloudTypeMask.rgb=0, placement=0.
// Global coverage and density, wind, position and layer height are supplied by the host.
public float4 simpleCloudConservativeDensity(uint4 resources, float3 skyTexScale, float h,
    float3 wind, float globalCoverage)
{
    let patternTexture = resolveUniform(ResourceHandle<Texture2D<float4>>(resources.x));
    let profileTexture = resolveUniform(ResourceHandle<Texture2D<float4>>(resources.y));
    let noiseTexture = resolveUniform(ResourceHandle<Texture3D<float4>>(resources.z));
    let wrapSampler = resolveUniform(SamplerHandle(resources.w));

    float3 pattern = patternTexture.SampleLevel(wrapSampler, skyTexScale.xy - 0.5f, 0.0f).rgb;
    float maxPattern = max(max(pattern.r, pattern.g), pattern.b);
    // VolumetricCloud.usf skips the material evaluation when conservative density X is zero.
    if (maxPattern <= 0.0f) { return 0.0f; }

    float3 profiles;
    profiles.r = profileTexture.SampleLevel(wrapSampler, float2(h, pattern.r), 0.0f).r;
    profiles.g = profileTexture.SampleLevel(wrapSampler, float2(h, pattern.g), 0.0f).g;
    profiles.b = profileTexture.SampleLevel(wrapSampler, float2(h, pattern.b), 0.0f).b;
    float profile = max(max(profiles.r, profiles.g), profiles.b) - 0.5f;

    float3 uvNoise2 = float3(60.0f, 60.0f, 75.0f) * skyTexScale + -6.0f * wind;
    float noise2 = (noiseTexture.SampleLevel(wrapSampler, uvNoise2, 0.0f).r - 0.8f) * 0.08f;
    float distortionXY = h * (1.0f - 0.05f) + 0.05f;
    float distortionZ = abs(distortionXY - 0.5f) * 2.0f;
    float3 distortion = float3(distortionXY, distortionXY, distortionZ * distortionZ);
    float3 uvNoise1 = float3(4.167f, 4.444f, 9.091f) * skyTexScale + 7.0f * wind + distortion * noise2;
    float noise1 = ((noiseTexture.SampleLevel(wrapSampler, uvNoise1, 0.0f).g - 0.5f) + globalCoverage) * 0.8f;
    return float4(maxPattern, maxPattern, profile, noise1);
}

public float simpleCloudExtinction(float4 conservative, float globalDensityPerMetre, float cameraDistanceKm)
{
    // SubsurfaceColor / Extinction. The caller applies If_1's layer-height test.
    float weightedProfile = conservative.y * (conservative.z + conservative.w);
    float cameraFade = saturate(cameraDistanceKm * 2.0f);
    return saturate((weightedProfile * globalDensityPerMetre) * cameraFade);
}
} // namespace Metallic.Clouds
