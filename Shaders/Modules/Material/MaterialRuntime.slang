#language slang 2026
implementing Material;

namespace Metallic.Material
{
// Stable built-in IDs shared with MaterialRuntime.h. Zero denotes a legacy upload.
public static const uint kMaterialProgramOpenPBRComposite = 1u;
public static const uint kMaterialProgramRTXCRChiang = 2u;

public uint materialProgramId(PathTraceMaterial parameters)
{
    if (parameters.textureParams.z == float(kMaterialProgramOpenPBRComposite)) {
        return kMaterialProgramOpenPBRComposite;
    }
    if (parameters.textureParams.z == float(kMaterialProgramRTXCRChiang)) {
        return kMaterialProgramRTXCRChiang;
    }
    // Existing probes and legacy scene producers have no explicit identity.
    return parameters.rtxcrHairBaseColor.w > 0.5
        ? kMaterialProgramRTXCRChiang : kMaterialProgramOpenPBRComposite;
}

// Minimal DOTS interaction. Authored normal/tangent stay independent of the ray;
// outgoingDirection is the only view-dependent field. No invented curve radius,
// strand ID or cross-sectional coordinate is inferred from a triangle hit.
public struct FiberInteraction
{
    public float3 authoredNormal;
    public float3 tangent;
    public float3 outgoingDirection;
};

} // namespace Metallic.Material
