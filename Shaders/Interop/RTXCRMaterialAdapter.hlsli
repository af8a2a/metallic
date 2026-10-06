#ifndef METALLIC_RTXCR_MATERIAL_ADAPTER
#define METALLIC_RTXCR_MATERIAL_ADAPTER

// Program-owned interop: vendor types and safeNormalize are provided by the owner.
Metallic.RTXCR.HairMaterialData makeRTXCRHairMaterial(PathTraceMaterial material)
{
    Metallic.RTXCR.HairMaterialData hair;
    hair.baseColor = saturate(material.rtxcrHairBaseColor.rgb);
    hair.longitudinalRoughness = clamp(material.rtxcrHairParams0.x, 0.02, 1.0);
    hair.azimuthalRoughness = clamp(material.rtxcrHairParams0.y, 0.02, 1.0);
    hair.ior = clamp(material.rtxcrHairParams0.z, 1.01, 3.0);
    hair.eta = 1.0 / hair.ior;
    hair.fresnelApproximation = 1u;
    hair.absorptionModel = Metallic.RTXCR.kHairAbsorptionModelNormalized;
    hair.melanin = saturate(material.rtxcrHairParams1.x);
    hair.melaninRedness = saturate(material.rtxcrHairParams1.y);
    hair.cuticleAngleInDegrees = clamp(material.rtxcrHairParams0.w, -10.0, 10.0);
    return hair;
}

Metallic.RTXCR.HairInteractionSurface makeRTXCRHairSurface(
    float3 stableNormal,
    float3 tangent,
    float3 viewDirection)
{
    Metallic.RTXCR.HairInteractionSurface surface;
    surface.shadingNormal = safeNormalize(stableNormal, float3(0.0, 0.0, 1.0));
    float3 tangentFallback = abs(surface.shadingNormal.y) < 0.99
        ? cross(float3(0.0, 1.0, 0.0), surface.shadingNormal)
        : cross(float3(1.0, 0.0, 0.0), surface.shadingNormal);
    surface.tangent = safeNormalize(
        tangent - surface.shadingNormal * dot(surface.shadingNormal, tangent),
        tangentFallback);
    surface.incidentRayDirection = safeNormalize(viewDirection, -surface.shadingNormal);
    return surface;
}

import MaterialProgram;
import FiberMaterial;
import RTXCRFiber;
using Metallic.Material;

struct SceneRTXCRHairParameters : IRTXCRHairParameterProvider
{
    PathTraceMaterial value;
    // The scene has already selected this immutable instance record. Offset 0
    // is relative to that record, as required by MaterialInstanceRef.
    [mutating] Metallic.RTXCR.HairMaterialData load(MaterialInstanceRef instance)
    {
        return makeRTXCRHairMaterial(value);
    }
};

FiberMaterialResult<RTXCRChiangClosure> evaluateSceneFiberMaterial(
    PathTraceMaterial parameters, FiberMaterialContext context)
{
    let surface = makeRTXCRHairSurface(context.normalWS, context.tangentWS, context.normalWS);
    context.normalWS = surface.shadingNormal;
    context.tangentWS = surface.tangent;
    context.bitangentWS = cross(context.normalWS, context.tangentWS);
    RTXCRChiangMaterialProgram<SceneRTXCRHairParameters> program;
    program.parameters.value = parameters;
    return program.evaluate(context, (MaterialInstanceRef)0);
}

float sceneFiberMISRoughness(PathTraceMaterial parameters)
{
    return clamp(parameters.rtxcrHairParams0.x, 0.02, 1.0);
}

#endif
