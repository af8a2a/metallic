#language slang 2026
module VkGltfBSDF;

// Isolated full glTF BSDF used by vk_gltf_renderer / nvpro_core2.
// No OpenPBR, renderer resources, LUTs, or scene-material imports.
__include "VkGltfBSDF/Constants.slang";
__include "VkGltfBSDF/Math.slang";
__include "VkGltfBSDF/Material.slang";
__include "VkGltfBSDF/Types.slang";
__include "VkGltfBSDF/Microfacet.slang";
__include "VkGltfBSDF/Lobes.slang";

namespace Metallic.VkGltf
{
public static const int kEventAbsorb = 0;
public static const int kEventDiffuse = 1;
public static const int kEventGlossy = 2;
public static const int kEventImpulse = 4;
public static const int kEventReflection = 8;
public static const int kEventTransmission = 16;
public static const float kDiracPDF = -1.0f;

public struct BSDFEvaluation
{
    // Each projected contribution already includes the rendering-equation cosine.
    public float3 projectedDiffuse;
    public float3 projectedGlossy;
    public float pdf;
};

public struct BSDFSample
{
    public float3 direction;
    public float3 weight;
    public float pdf;
    public int eventType;
};

// k1 points toward the incoming ray origin; k2 points toward the sampled light.
// xi has three uniform [0,1) random numbers: full upstream evaluation is stochastic.
// Its pdf is the selected lobe's PDF, as in the upstream MIS estimator.
public BSDFEvaluation evaluateBSDF(PreparedMaterial mat, float3 k1, float3 k2, float3 xi)
{
    EvaluationData data = {};
    data.k1 = k1;
    data.k2 = k2;
    data.xi = xi;
    bsdfEvaluate(data, mat);
    BSDFEvaluation result;
    result.projectedDiffuse = data.bsdf_diffuse;
    result.projectedGlossy = data.bsdf_glossy;
    result.pdf = data.pdf;
    return result;
}

// weight is the upstream bsdf_over_pdf and must not be divided by pdf again.
// Absorption is indicated by eventType == kEventAbsorb. A valid impulse uses
// pdf == kDiracPDF and should receive MIS weight 1 against a continuous sampler.
public BSDFSample sampleBSDF(PreparedMaterial mat, float3 k1, float3 xi)
{
    SampleData data = {};
    data.k1 = k1;
    data.xi = xi;
    bsdfSample(data, mat);
    BSDFSample result;
    result.direction = data.k2;
    result.weight = data.bsdf_over_pdf;
    result.pdf = data.pdf;
    result.eventType = data.event_type;
    return result;
}
} // namespace Metallic.VkGltf
