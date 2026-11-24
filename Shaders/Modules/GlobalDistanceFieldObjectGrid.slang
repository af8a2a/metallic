module GlobalDistanceFieldObjectGrid;

import GlobalDistanceField;
import GlobalDistanceFieldPages;
import MeshDistanceField;
import ShaderCore;

namespace Metallic.DistanceField
{
// UE 5.7.4 GlobalDistanceFieldObjectGrid.ush.
public static const uint kGlobalDistanceFieldInvalidObjectGridEntry = 0xffffffffu;
public static const uint kGlobalDistanceFieldObjectGridResolution = 4u;
public static const uint kGlobalDistanceFieldObjectGridPageStride = 64u;
public static const float kGlobalDistanceFieldObjectGridCardRangeInVoxels = 3.0f;
static const uint kCellSize = 4u;
static const uint kPackedInstanceMask = 0x00ffffffu;
static const uint kMaxCulledObjects = 511u;

public struct GlobalDistanceFieldObjectGridSource
{
    public float3 center;
    public uint object;
    public float3 extent;
    public uint padding;
};

groupshared uint sharedObjectGridCulledCount;
groupshared uint sharedObjectGridCulledIndices[kMaxCulledObjects];

// UE 5.7.4 Common.ush:2373. Each coordinate spans two bits.
uint encodeObjectGridCell(uint3 coordinate)
{
    uint index = 0u;
    [unroll]
    for (uint i = 0u; i < 2u; ++i) {
        index |= ((coordinate.x >> i) & 1u) << (3u * i + 0u);
        index |= ((coordinate.y >> i) & 1u) << (3u * i + 1u);
        index |= ((coordinate.z >> i) & 1u) << (3u * i + 2u);
    }
    return index;
}

uint packObjectGridCellIndex(uint instanceIndex, float distance, float cellWorldExtent, float maxQueryDistance)
{
    float normalizedPackedDistance = saturate((distance + 0.25f * cellWorldExtent) /
        (maxQueryDistance + 0.25f * cellWorldExtent));
    uint packedDistance = uint(clamp(normalizedPackedDistance * (0xffu + 1.0f), 0.0f, 0xffu));
    return (instanceIndex & kPackedInstanceMask) | (packedDistance << 24u);
}

void addToObjectGridCell(inout uint4 cell, uint instanceIndex, float distance, float cellWorldExtent,
    float maxQueryDistance)
{
    if (abs(distance) < maxQueryDistance) {
        uint toInsert = packObjectGridCellIndex(instanceIndex, distance, cellWorldExtent, maxQueryDistance);
        uint maxPackedIndex = 0u;
        for (uint index = 0u; index < kCellSize; ++index) {
            maxPackedIndex = max(maxPackedIndex, cell[index]);
        }
        if (toInsert < maxPackedIndex) {
            for (uint index = 0u; index < kCellSize; ++index) {
                if (cell[index] == maxPackedIndex) {
                    cell[index] = toInsert;
                    break;
                }
            }
        }
    }
}

void sortObjectGridCell(inout uint4 cell)
{
    for (uint passIndex = 0u; passIndex < kCellSize - 1u; ++passIndex) {
        for (uint elementIndex = 0u; elementIndex < kCellSize - passIndex - 1u; ++elementIndex) {
            uint minIndex = min(cell[elementIndex], cell[elementIndex + 1u]);
            uint maxIndex = max(cell[elementIndex], cell[elementIndex + 1u]);
            cell[elementIndex] = minIndex;
            cell[elementIndex + 1u] = maxIndex;
        }
    }
}

float computeSquaredDistanceBetweenAABBs(float3 centerA, float extentA, float3 centerB, float3 extentB)
{
    float3 axisDistances = max(abs(centerB - centerA) - (extentA + extentB), 0.0f);
    return dot(axisDistances, axisDistances);
}

// UE 5.7.4 GlobalDistanceFieldCompositeObjects.usf:262.
public void compositeGlobalDistanceFieldObjectGridPage(GlobalDistanceFieldDomain domain, uint3 pageCoordinate,
    uint3 cellCoordinate, uint threadIndex, uint physicalPage, MeshDistanceFieldResources meshes,
    StructuredBuffer<MeshDistanceFieldInstance> instances, StructuredBuffer<GlobalDistanceFieldObjectGridSource> objects,
    uint instanceCount, ResourceHandle<StructuredBuffer<uint>> cullGrid, uint2 candidates,
    RWStructuredBuffer<uint4> objectGrid)
{
    const float voxelSize = domain.minAndVoxelSize.w;
    const float voxelExtent = 0.5f * voxelSize;
    const float3 pageCoordToWorldCenterBias = domain.minAndVoxelSize.xyz + voxelExtent;
    const float3 cellPageCoordinate = float3(pageCoordinate) * kGlobalDistanceFieldPageCells +
        float3(cellCoordinate) * 2.0f + 0.5f - 0.5f;
    const float3 cellWorldCenter = cellPageCoordinate * voxelSize + pageCoordToWorldCenterBias;
    const float cellWorldExtent = voxelExtent * kGlobalDistanceFieldPageCells /
        float(kGlobalDistanceFieldObjectGridResolution);
    const float cardInterpolationRange = kGlobalDistanceFieldObjectGridCardRangeInVoxels * voxelExtent;
    const float maxQueryDistance = 1.44f * cellWorldExtent + cardInterpolationRange;

    const float3 tilePageCoordinate = float3(pageCoordinate) * kGlobalDistanceFieldPageCells +
        2.0f * (float(kGlobalDistanceFieldPageCells) / 4.0f);
    const float3 tileWorldCenter = tilePageCoordinate * voxelSize + pageCoordToWorldCenterBias;
    const float tileWorldExtent = voxelExtent * kGlobalDistanceFieldPageCells;
    const float maxTileQueryDistance = 1.44f * tileWorldExtent + cardInterpolationRange;

    uint4 cell = uint4(kGlobalDistanceFieldInvalidObjectGridEntry);
    const bool useCullGrid = candidates.y != kGlobalDistanceFieldFullCullScan;
    const uint candidateCount = useCullGrid ? candidates.y : instanceCount;
    const uint passes = candidateCount == 0u ? 0u : 1u + (candidateCount - 1u) / kMaxCulledObjects;
    for (uint passIndex = 0u; passIndex < passes; ++passIndex) {
        uint passStartObject = passIndex * kMaxCulledObjects;
        uint passEndObject = passStartObject + min(candidateCount - passStartObject, kMaxCulledObjects);
        if (threadIndex == 0u) {
            sharedObjectGridCulledCount = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint index = passStartObject + threadIndex; index < passEndObject;
            index += kGlobalDistanceFieldObjectGridPageStride) {
            uint instanceIndex = index;
            if (useCullGrid) {
                instanceIndex = resolveUniform(cullGrid)[candidates.x + index];
            }
            GlobalDistanceFieldObjectGridSource object = objects[instanceIndex];
            float distanceSquared = computeSquaredDistanceBetweenAABBs(tileWorldCenter, tileWorldExtent,
                object.center, object.extent);
            if (distanceSquared < maxTileQueryDistance * maxTileQueryDistance) {
                uint destination;
                InterlockedAdd(sharedObjectGridCulledCount, 1u, destination);
                sharedObjectGridCulledIndices[destination] = instanceIndex;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        uint count = sharedObjectGridCulledCount;
        for (uint listIndex = 0u; listIndex < count; ++listIndex) {
            uint instanceIndex = sharedObjectGridCulledIndices[listIndex];
            GlobalDistanceFieldObjectGridSource object = objects[instanceIndex];
            float distanceSquared = computeSquaredDistanceBetweenAABBs(cellWorldCenter, cellWorldExtent,
                object.center, object.extent);
            if (distanceSquared <= cardInterpolationRange * cardInterpolationRange) {
                float distance = distanceToNearestMeshSurface(meshes, instances[instanceIndex],
                    cellWorldCenter, maxQueryDistance);
                addToObjectGridCell(cell, instanceIndex, distance, cellWorldExtent, maxQueryDistance);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    sortObjectGridCell(cell);
    // The UE sort key uses dense instance indices. Export the matching full
    // Metallic scene identity to retain entity generation bits for card lookup.
    for (uint index = 0u; index < kCellSize; ++index) {
        if (cell[index] < kGlobalDistanceFieldInvalidObjectGridEntry) {
            cell[index] = objects[cell[index] & kPackedInstanceMask].object;
        }
    }
    objectGrid[kGlobalDistanceFieldObjectGridPageStride * physicalPage + encodeObjectGridCell(cellCoordinate)] = cell;
}

public uint4 loadGlobalDistanceFieldObjectGridCell(Texture3D<uint> pageTable, StructuredBuffer<uint4> objectGrid,
    GlobalDistanceFieldDomain domain, uint clipmapIndex, float3 worldPosition)
{
    if (any(worldPosition < domain.minAndVoxelSize.xyz) || any(worldPosition > domain.maxAndInfluenceRadius.xyz)) {
        return uint4(kGlobalDistanceFieldInvalidObjectGridEntry);
    }
    float3 gridPosition = (worldPosition - domain.minAndVoxelSize.xyz) / domain.minAndVoxelSize.w;
    float3 pagePosition;
    uint page = loadGlobalDistanceFieldPage(pageTable, gridPosition, pagePosition, clipmapIndex);
    if (page >= kGlobalDistanceFieldOverflowPage) {
        return uint4(kGlobalDistanceFieldInvalidObjectGridEntry);
    }
    // The non-wrapping page table owns the positive endpoint as part of its last page.
    uint3 cell = min(uint3((pagePosition / float(kGlobalDistanceFieldPageCells)) *
        kGlobalDistanceFieldObjectGridResolution), kGlobalDistanceFieldObjectGridResolution - 1u);
    return objectGrid[kGlobalDistanceFieldObjectGridPageStride * page + encodeObjectGridCell(cell)];
}
} // namespace Metallic.DistanceField
