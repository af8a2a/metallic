module VolumetricCloudViewTrace;

import VolumetricCloudLighting;

namespace Metallic.Clouds
{
// UE 5.7.4 VolumetricCloud.usf, tAP and improved scattering integration.
// The material adapter supplies extinction and scattered radiance per metre.
public struct CloudViewIntegral
{
    public float3 luminance;
    public float transmittance;
    public float depthWeightedSum;
    public float depthWeightsSum;

    public static CloudViewIntegral empty()
    {
        CloudViewIntegral result = {};
        result.transmittance = 1.0f;
        return result;
    }

    [mutating] public void integrateOrder(float distanceMeters, float stepMeters,
        float extinctionPerMeter, float3 scatteredLuminancePerMeter, uint order)
    {
        float3 segmentTransmittance;
        float3 luminanceIntegral = integrateCloudScattering(extinctionPerMeter, scatteredLuminancePerMeter,
            stepMeters, segmentTransmittance);
        luminance += transmittance * luminanceIntegral;
        // Accumulate higher octaves first so all three see the same incoming view transmittance.
        if (order == 0u) {
            depthWeightedSum += distanceMeters * transmittance;
            depthWeightsSum += transmittance;
            transmittance *= segmentTransmittance.x;
        }
    }

    [mutating] public void integrateSample(float distanceMeters, float stepMeters,
        float extinctionPerMeter, float3 scatteredLuminancePerMeter)
    {
        if (extinctionPerMeter > 0.0f) {
            integrateOrder(distanceMeters, stepMeters, extinctionPerMeter, scatteredLuminancePerMeter, 0u);
        }
    }

    public float representativeDepthMeters()
    {
        // Zero is the explicit empty-ray marker in the full-resolution graph output.
        return depthWeightsSum == 0.0f ? 0.0f : depthWeightedSum / max(0.0000000001f, depthWeightsSum);
    }
};
} // namespace Metallic.Clouds
