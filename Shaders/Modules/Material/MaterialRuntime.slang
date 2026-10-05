#language slang 2026
implementing Material;

import MaterialProgram;

namespace Metallic.Material
{
// Stable built-in IDs shared with MaterialRuntime.h. Zero denotes a legacy upload.
public static const uint kMaterialProgramOpenPBRComposite = 1u;
public static const uint kMaterialProgramRTXCRChiang = 2u;
public static const uint kMaterialProgramSingleSlab = 3u;
public static const uint kMaterialProgramDualSlab = 4u;

// Compatibility storage program. The existing upload embeds resources in the
// same 720-byte record, so both offsets must reference that same record. This
// adapter is not a new scattering model or a per-instance shader specialization.
public static const uint kLegacyMaterialRecordBytes = 720u;
public MaterialInstanceRef legacyMaterialInstanceRef(uint materialIndex)
{
    MaterialInstanceRef instance;
    instance.parameterOffset = materialIndex * kLegacyMaterialRecordBytes;
    instance.resourceOffset = instance.parameterOffset;
    return instance;
}

public struct LegacyMaterialProgram
{
    public PathTraceMaterial loadParameters(StructuredBuffer<PathTraceMaterial> parameters, MaterialInstanceRef instance)
    {
        return parameters[instance.parameterOffset / kLegacyMaterialRecordBytes];
    }
};

public uint materialProgramId(PathTraceMaterial parameters)
{
    if (parameters.textureParams.z == float(kMaterialProgramSingleSlab)) { return kMaterialProgramSingleSlab; }
    if (parameters.textureParams.z == float(kMaterialProgramDualSlab)) { return kMaterialProgramDualSlab; }
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

public static const uint kMaterialDomainSurface = 0u;
public static const uint kMaterialDomainFiber = 1u;

// Model registration belongs to the material runtime, never the integrator.
public uint materialDomain(PathTraceMaterial parameters)
{
    return materialProgramId(parameters) == kMaterialProgramRTXCRChiang ? kMaterialDomainFiber : kMaterialDomainSurface;
}

} // namespace Metallic.Material
