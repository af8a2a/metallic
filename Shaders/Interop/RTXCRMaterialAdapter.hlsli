#ifndef METALLIC_RTXCR_MATERIAL_ADAPTER
#define METALLIC_RTXCR_MATERIAL_ADAPTER

// Program-owned interop: vendor types and safeNormalize are provided by the owner.
RTXCR_HairMaterialData makeRtxcrHairMaterial(PathTraceMaterial material)
{
    RTXCR_HairMaterialData hair;
    hair.baseColor = saturate(material.rtxcrHairBaseColor.rgb);
    hair.longitudinalRoughness = clamp(material.rtxcrHairParams0.x, 0.02, 1.0);
    hair.azimuthalRoughness = clamp(material.rtxcrHairParams0.y, 0.02, 1.0);
    hair.ior = clamp(material.rtxcrHairParams0.z, 1.01, 3.0);
    hair.eta = 1.0 / hair.ior;
    hair.fresnelApproximation = 1u;
    hair.absorptionModel = RTXCR_HairAbsorptionModel_Normalized;
    hair.melanin = saturate(material.rtxcrHairParams1.x);
    hair.melaninRedness = saturate(material.rtxcrHairParams1.y);
    hair.cuticleAngleInDegrees = clamp(material.rtxcrHairParams0.w, -10.0, 10.0);
    return hair;
}

RTXCR_HairInteractionSurface makeRtxcrHairSurface(
    float3 stableNormal,
    float3 tangent,
    float3 viewDirection)
{
    RTXCR_HairInteractionSurface surface;
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

struct RTXCRPreparedMaterial
{
    RTXCR_HairMaterialData parameters;
    RTXCR_HairInteractionSurface interaction;
};

RTXCRPreparedMaterial prepareRTXCRMaterial(PathTraceMaterial parameters, FiberInteraction interaction)
{
    RTXCRPreparedMaterial prepared;
    prepared.parameters = makeRtxcrHairMaterial(parameters);
    prepared.interaction = makeRtxcrHairSurface(interaction.authoredNormal,
        interaction.tangent, interaction.outgoingDirection);
    return prepared;
}

// Chiang's fiber measure is preserved. Do not apply a Surface N dot L here.
float3 evalRTXCRMaterial(RTXCRPreparedMaterial prepared, float3 wi)
{
    return RTXCR_HairChiangBsdfEval(prepared.parameters, prepared.interaction, wi);
}

// weight is the vendor numerator; the existing integrator divides by pdf once.
bool sampleRTXCRMaterial(RTXCRPreparedMaterial prepared, float2 random[2],
    out float3 wi, out float pdf, out float3 weight, out RTXCR_HairLobeType eventType)
{
    return RTXCR_SampleChiangBsdf(prepared.parameters, prepared.interaction,
        random, wi, pdf, weight, eventType);
}

#endif
