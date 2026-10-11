#language slang 2026
module RTXCRMaterialAdapter;


import Material;
import RTXCRHair;
import MaterialProgram;
import FiberMaterial;
import RTXCRFiber;
using Metallic.Material;

internal float3 normalizeHairVector(float3 value, float3 fallback, float minimumLengthSquared)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > minimumLengthSquared ? value * rsqrt(lengthSquared) : fallback;
}
public Metallic.RTXCR.HairMaterialData makeRTXCRHairMaterial(PathTraceMaterial material)
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

public Metallic.RTXCR.HairInteractionSurface makeRTXCRHairSurface(
    float3 stableNormal,
    float3 tangent,
    float3 viewDirection,
    float minimumLengthSquared = 1e-8)
{
    Metallic.RTXCR.HairInteractionSurface surface;
    surface.shadingNormal = normalizeHairVector(stableNormal, float3(0.0, 0.0, 1.0), minimumLengthSquared);
    float3 tangentFallback = abs(surface.shadingNormal.y) < 0.99
        ? cross(float3(0.0, 1.0, 0.0), surface.shadingNormal)
        : cross(float3(1.0, 0.0, 0.0), surface.shadingNormal);
    surface.tangent = normalizeHairVector(
        tangent - surface.shadingNormal * dot(surface.shadingNormal, tangent),
        tangentFallback, minimumLengthSquared);
    surface.incidentRayDirection = normalizeHairVector(viewDirection, -surface.shadingNormal, minimumLengthSquared);
    return surface;
}

public struct SceneRTXCRHairParameters : IRTXCRHairParameterProvider
{
    public PathTraceMaterial value;
    // The scene has already selected this immutable instance record. Offset 0
    // is relative to that record, as required by MaterialInstanceRef.
    [mutating] public Metallic.RTXCR.HairMaterialData load(MaterialInstanceRef instance)
    {
        return makeRTXCRHairMaterial(value);
    }
};

public FiberMaterialResult<RTXCRChiangClosure> evaluateSceneFiberMaterial(
    PathTraceMaterial parameters, FiberMaterialContext context, float minimumLengthSquared = 1e-8)
{
    let surface = makeRTXCRHairSurface(context.normalWS, context.tangentWS, context.normalWS, minimumLengthSquared);
    context.normalWS = surface.shadingNormal;
    context.tangentWS = surface.tangent;
    context.bitangentWS = cross(context.normalWS, context.tangentWS);
    RTXCRChiangMaterialProgram<SceneRTXCRHairParameters> program;
    program.parameters.value = parameters;
    return program.evaluate(context, (MaterialInstanceRef)0);
}

public float sceneFiberMISRoughness(PathTraceMaterial parameters)
{
    return clamp(parameters.rtxcrHairParams0.x, 0.02, 1.0);
}

