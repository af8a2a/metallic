#language slang 2026
module RTXCRMaterialParameters;

import Material;

namespace Metallic.Material
{
public struct RTXCRHairMaterialParameters
{
    public float4 baseColor;
    public float4 roughnessIOR;
    public float4 melanin;
    public float4 diffuseTint;
};

// Bounded renderer-guide approximation. It never evaluates Surface inputs or
// textures and uses the same color/roughness/IOR limits as the Fiber adapter.
public struct FiberGuideParameters
{
    public float3 albedo;
    public float3 specularReflectance;
    public float roughness;
    public float ior;
};

public FiberGuideParameters fiberGuideParameters(RTXCRHairMaterialParameters parameters)
{
    FiberGuideParameters guides;
    guides.albedo = saturate(parameters.baseColor.rgb);
    guides.roughness = clamp(parameters.roughnessIOR.x, 0.02, 1.0);
    guides.ior = clamp(parameters.roughnessIOR.z, 1.01, 3.0);
    float f0 = (guides.ior - 1.0) / (guides.ior + 1.0);
    guides.specularReflectance = float3(f0 * f0);
    return guides;
}

public RTXCRHairMaterialParameters sceneFiberParameters(SceneMaterialData data)
{
    RTXCRHairMaterialParameters result;
    result.baseColor = data.rtxcrHairBaseColor;
    result.roughnessIOR = data.rtxcrHairParams0;
    result.melanin = data.rtxcrHairParams1;
    result.diffuseTint = data.rtxcrHairDiffuseTint;
    return result;
}
} // namespace Metallic.Material
