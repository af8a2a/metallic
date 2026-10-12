#language slang 2026
implementing Material;

import MaterialProgram;

namespace Metallic.Material
{
// Stable built-in IDs shared with MaterialRuntime.h. CPU publication assigns IDs.
public static const uint kMaterialProgramOpenPBRComposite = 1u;
public static const uint kMaterialProgramRTXCRChiang = 2u;
public static const uint kMaterialProgramSingleSlab = 3u;
public static const uint kMaterialProgramDualSlab = 4u;

// Scene transport offsets are independent of each model's resolved parameters.
public static const uint kSceneMaterialRecordBytes = sizeof(SceneMaterialData);
public MaterialInstanceRef sceneMaterialInstanceRef(uint materialIndex)
{
    MaterialInstanceRef instance;
    instance.parameterOffset = materialIndex * kSceneMaterialRecordBytes;
    instance.resourceOffset = instance.parameterOffset;
    return instance;
}

public uint materialProgramId(SceneMaterialData parameters)
{
    return uint(parameters.textureParams.z);
}

public static const uint kMaterialDomainSurface = 0u;
public static const uint kMaterialDomainFiber = 1u;

// Model registration belongs to the material runtime, never the integrator.
public uint materialDomain(SceneMaterialData parameters)
{
    return materialProgramId(parameters) == kMaterialProgramRTXCRChiang ? kMaterialDomainFiber : kMaterialDomainSurface;
}

} // namespace Metallic.Material
