#language slang 2026
module SceneSurfaceMaterial;

import ScenePathTypes;
import MaterialProgram;
import RTXCRMaterialParameters;
import OpenPBR;
import OpenPBRClosure;
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif

namespace Metallic
{
using Metallic.Material;
using Metallic.OpenPBR;

// Renderer diagnostics and environment approximations are independent of the
// scattering model. These values never stand in for a model's resolved closure.
public struct SceneSurfaceSummary
{
    public float3 baseColor;
    public float3 albedo;
    public float3 diffuseColor;
    public float3 specularReflectance;
    public float3 guideSpecularReflectance;
    public float3 dielectricReflectance;
    public float3 transmissionColor;
    public float conductorFraction;
    public float diffuseWeight;
    public float surfaceTransmission;
    public float roughness;
    public float ior;
    public float transmissionWeight;
    public float occlusion;
    public bool unlit;
    public bool numericalEnvironment;
};

public interface ISceneSurfaceMaterialProgram : ISurfaceMaterialProgram
{
    public HitInfo evaluatedHit();
    public uint evaluatedSeed();
    public float4 sampledNormal();
    public SceneSurfaceSummary surfaceSummary();
};

// Fiber guides are bounded diagnostic approximations, independent of Surface
// scattering. They use the same parameter limits as the native Fiber adapter.
public SceneSurfaceSummary summarizeFiberGuides(RTXCRHairMaterialParameters parameters)
{
    let guides = fiberGuideParameters(parameters);
    SceneSurfaceSummary summary = (SceneSurfaceSummary)0;
    summary.baseColor = guides.albedo;
    summary.albedo = guides.albedo;
    summary.roughness = guides.roughness;
    summary.ior = guides.ior;
    summary.guideSpecularReflectance = guides.specularReflectance;
    summary.occlusion = 1.0;
    return summary;
}

public SceneSurfaceSummary summarizeOpenPBRSurface<TContext : IOpenPBRContext>(OpenPBRClosure<TContext> closure, bool unlit)
{
    let inputs = closure.inputs;
    SceneSurfaceSummary summary = (SceneSurfaceSummary)0;
    summary.baseColor = inputs.base_color;
    float metalness = saturate(inputs.base_metalness);
    summary.albedo = saturate(inputs.base_color) * (1.0 - metalness);
    summary.diffuseColor = inputs.base_color * inputs.base_weight * (1.0 - metalness) *
        (1.0 - inputs.transmission_weight);
    summary.ior = max(inputs.specular_ior, 1.0);
    float dielectricF0 = (summary.ior - 1.0) / (summary.ior + 1.0);
    summary.dielectricReflectance = dielectricF0 * dielectricF0 * inputs.specular_color * inputs.specular_weight;
    summary.conductorFraction = metalness;
    summary.diffuseWeight = inputs.base_weight;
    summary.surfaceTransmission = inputs.transmission_weight;
    summary.specularReflectance = lerp(summary.dielectricReflectance, inputs.base_color, metalness);
    summary.guideSpecularReflectance = lerp(dielectricF0 * dielectricF0 * saturate(inputs.specular_color),
        saturate(inputs.base_color), metalness);
    summary.transmissionWeight = inputs.transmission_weight * (1.0 - metalness);
    summary.transmissionColor = inputs.transmission_color;
    summary.roughness = inputs.specular_roughness;
    summary.occlusion = closure.occlusion;
    summary.unlit = unlit;
    summary.numericalEnvironment = inputs.coat_weight > 0.0 || inputs.fuzz_weight > 0.0 ||
        inputs.specular_roughness_anisotropy > 0.0;
    return summary;
}

#if METALLIC_CUSTOM_MATERIALS
public SceneSurfaceSummary summarizeSlabSurface(SlabMaterialInputs inputs)
{
    SceneSurfaceSummary summary = (SceneSurfaceSummary)0;
    summary.baseColor = inputs.first.reflectance.rgb;
    if (inputs.control.x == 1.0) {
        summary.baseColor = lerp(inputs.first.reflectance.rgb, inputs.second.reflectance.rgb, inputs.control.y);
    } else if (inputs.control.x == 2.0) {
        float3 transmission = 1.0 - inputs.first.reflectance.rgb;
        summary.baseColor += transmission * transmission * inputs.second.reflectance.rgb *
            exp(-2.0 * inputs.first.opticalDepth.rgb);
    }
    summary.diffuseColor = summary.baseColor;
    summary.albedo = summary.baseColor;
    summary.roughness = 1.0;
    summary.ior = 1.0;
    summary.occlusion = 1.0;
    summary.numericalEnvironment = true;
    return summary;
}
#endif

} // namespace Metallic
