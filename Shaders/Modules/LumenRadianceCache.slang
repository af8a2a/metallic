module LumenRadianceCache;

namespace Metallic.Lumen
{

public static const uint kLumenRadianceInvalidProbe = 0xffffffffu;
public static const uint kLumenRadianceUsedProbe = 0xfffffffeu;

public struct LumenRadianceCacheClipmap
{
    public float4 cornerAndCellSize;
    // Minimum ray distance, reciprocal fade size, reprojection radius scale, squared supersample distance.
    public float4 traceSettings;
    // Grid resolution, clipmap count, trace resolution, number of atlas probes on one axis.
    public uint4 layout;
};

// UE 5.7.4 LumenRadianceCacheInterpolation.ush, world coordinates in meters.
public float3 lumenRadianceProbeCoordFloat(float3 world, LumenRadianceCacheClipmap clipmap)
{
    return (world - clipmap.cornerAndCellSize.xyz) / clipmap.cornerAndCellSize.w;
}

public float3 lumenRadianceProbeWorldPosition(uint3 cell, LumenRadianceCacheClipmap clipmap)
{
    const float3 cornerToProbe = (float3(cell) + 0.5f) * clipmap.cornerAndCellSize.w;
    return clipmap.cornerAndCellSize.xyz + cornerToProbe;
}

// The StructuredBuffer retains UE's X-packed Texture3D ordering.
public uint lumenRadianceIndirectionAddress(uint3 cell, uint clipmap, uint gridResolution, uint clipmapCount)
{
    return cell.x + clipmap * gridResolution + (cell.y + cell.z * gridResolution) * gridResolution * clipmapCount;
}

public uint lumenRadianceClipmapForPosition(float3 world, StructuredBuffer<LumenRadianceCacheClipmap> clipmaps,
    float dither)
{
    const uint count = clipmaps[0].layout.y;
    for (uint index = 0; index < count; ++index) {
        LumenRadianceCacheClipmap clipmap = clipmaps[index];
        float3 cell = lumenRadianceProbeCoordFloat(world, clipmap);
        float3 bottomEdgeFades = saturate((cell - 0.5f) * clipmap.traceSettings.y);
        float3 topEdgeFades = saturate((float(clipmap.layout.x) - 0.5f - cell) * clipmap.traceSettings.y);
        float edgeFade = min(min(bottomEdgeFades.x, min(bottomEdgeFades.y, bottomEdgeFades.z)),
            min(topEdgeFades.x, min(topEdgeFades.y, topEdgeFades.z)));
        if (edgeFade > dither) {
            return index;
        }
    }
    return count;
}

} // namespace Metallic.Lumen
