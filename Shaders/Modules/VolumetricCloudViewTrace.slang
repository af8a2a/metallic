module VolumetricCloudViewTrace;

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

    [mutating] public void integrateSample(float distanceMeters, float stepMeters,
        float extinctionPerMeter, float3 scatteredLuminancePerMeter)
    {
        if (extinctionPerMeter > 0.0f) {
            float weight = transmittance;
            depthWeightedSum += distanceMeters * weight;
            depthWeightsSum += weight;
            float safeExtinction = max(0.000001f, extinctionPerMeter);
            float stepTransmittance = exp(-safeExtinction * stepMeters);
            float3 luminanceIntegral = (scatteredLuminancePerMeter -
                scatteredLuminancePerMeter * stepTransmittance) / safeExtinction;
            luminance += transmittance * luminanceIntegral;
            transmittance *= stepTransmittance;
        }
    }

    public float representativeDepthMeters()
    {
        // Zero is the explicit empty-ray marker in the full-resolution graph output.
        return depthWeightsSum == 0.0f ? 0.0f : depthWeightedSum / max(0.0000000001f, depthWeightsSum);
    }
};
} // namespace Metallic.Clouds
