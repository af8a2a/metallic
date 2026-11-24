module LumenProbeIntegration;

namespace Metallic.Lumen
{
public static const uint kProbeIntegrationTileSize = 8u;
public static const uint kProbeIntegrationModeCount = 2u;
public static const uint kProbeIntegrationSimple = 0u;
public static const uint kProbeIntegrationRoughSpecular = 1u;

// The Metallic opaque, isotropic receiver adapter used by both integration and its tile mask.
public float lumenProbeRoughSpecularDiffuseLerp(float roughness)
{
    const float fadeLength = 0.2f;
    const float maxRoughnessToEvaluateRoughSpecular = 0.8f;
    return saturate((roughness - maxRoughnessToEvaluateRoughSpecular + fadeLength) / fadeLength);
}

public uint2 lumenProbeIntegrationTileDimensions(uint2 viewSize)
{
    return (viewSize + kProbeIntegrationTileSize - 1u) / kProbeIntegrationTileSize;
}
}
