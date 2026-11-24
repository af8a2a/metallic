module LumenMeshSDFGrid;

import ShaderCore;

namespace Metallic.Lumen
{

public static const uint kLumenMeshSDFGridPixelSizeShift = 6u;
public static const uint kLumenMeshSDFGridPixelSize = 1u << kLumenMeshSDFGridPixelSizeShift;
public static const uint kLumenMeshSDFDenseInstanceOffset = ~0u;

// UE 5.7.4 LumenDiffuseIndirect.cpp:132, converted from centimeters to meters.
static const float3 kCardGridZParams = float3(1.0f, 1.0f, 4.0f);

public uint lumenMeshSDFGridDepthSlice(float sceneDepth, uint sliceCount)
{
    // UE 5.7.4 LumenMeshSDFCulling.usf:347 and LumenScreenProbeTracing.usf:534.
    uint zSlice = uint(max(0.0f, log2(sceneDepth * kCardGridZParams.x + kCardGridZParams.y) * kCardGridZParams.z));
    return min(zSlice, sliceCount - 1u);
}

public float lumenMeshSDFGridSliceDepth(float zSlice)
{
    return (exp2(zSlice / kCardGridZParams.z) - kCardGridZParams.y) / kCardGridZParams.x;
}

public uint lumenMeshSDFGridIndex(uint3 gridCoordinate, uint3 gridSize)
{
    return (gridCoordinate.z * gridSize.y + gridCoordinate.y) * gridSize.x + gridCoordinate.x;
}

public uint lumenMeshSDFGridCellIndex(uint2 pixel, float sceneDepth, uint3 gridSize)
{
    uint3 gridCoordinate = uint3(pixel >> kLumenMeshSDFGridPixelSizeShift,
        lumenMeshSDFGridDepthSlice(sceneDepth, gridSize.z));
    return lumenMeshSDFGridIndex(gridCoordinate, gridSize);
}

} // namespace Metallic.Lumen
