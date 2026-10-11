/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 * Ported from RTXGI v2 (Libraries/Sharc/include/HashGridCommon.h) for Metallic's
 * Slang/SPIR-V path tracer. The 64-bit key insertion path uses SPIR-V Int64
 * atomics (verified against Slang codegen); this module uses the renderer
 * configuration with 64-bit atomics and no lock-buffer fallback.
 */

implementing SHARC;

namespace Metallic.SHARC
{
// Computes logarithm with an arbitrary base.
// The base controls how frequently the hash grid level changes with distance
// (i.e., the rate at which voxel size increases across levels).
// SHaRC typically uses base 2.0 as a good default
internal float hashGridLogBase(float x, float base)
{
    return log2(x) * rcp(log2(base));
}

// http://burtleburtle.net/bob/hash/integer.html
internal uint hashGridHashJenkins32(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);

    return a;
}

internal uint hashGridHash32(HashGridKey hashKey)
{
    return hashGridHashJenkins32(uint(hashKey)) ^ hashGridHashJenkins32(uint(hashKey >> 32));
}

// Computes the first slot for the probe bucket.
// The bucket is kept contiguous and does not wrap around the end of the table,
// which simplifies lookup / insertion logic. Applications may allocate
// kHashGridHashMapBucketSize - 1 extra slots beyond capacity to avoid
// clamping the base slot near the end
public uint hashGridGetBaseSlot(const HashGridKey hashKey, uint capacity)
{
    const uint baseSlotCount = capacity - kHashGridHashMapBucketSize + 1;

    return hashGridHash32(hashKey) % baseSlotCount;
}

public uint hashGridGetLevel(float3 samplePosition, HashGridParameters gridParameters)
{
    float3 cameraOffset = gridParameters.cameraPosition - samplePosition;
    float distance2 = dot(cameraOffset, cameraOffset);
    distance2 = max(distance2, 1e-10f);
    float gridLevel = 0.5f * hashGridLogBase(distance2, gridParameters.logarithmBase) + gridParameters.levelBias;

    return uint(clamp(gridLevel, 1.0f, float(kHashGridLevelBitMask)));
}

public float hashGridGetVoxelSize(uint gridLevel, HashGridParameters gridParameters)
{
    float exponent = log2(gridParameters.logarithmBase) * (float(gridLevel) - gridParameters.levelBias);

    return exp2(exponent) * rcp(gridParameters.sceneScale);
}

// Based on logarithmic caching by Johannes Jendersie
internal int4 hashGridCalculatePositionLogWithVoxelSize(float3 samplePosition, HashGridParameters gridParameters, out float voxelSize)
{
    samplePosition += float3(kHashGridPositionBias, kHashGridPositionBias, kHashGridPositionBias);

    uint gridLevel      = hashGridGetLevel(samplePosition, gridParameters);
    voxelSize           = hashGridGetVoxelSize(gridLevel, gridParameters);
    int3 gridPosition   = int3(floor(samplePosition * rcp(voxelSize)));

    return int4(gridPosition.xyz, gridLevel);
}

internal int4 hashGridCalculatePositionLog(float3 samplePosition, HashGridParameters gridParameters)
{
    float voxelSize;
    return hashGridCalculatePositionLogWithVoxelSize(samplePosition, gridParameters, voxelSize);
}

internal HashGridKey hashGridComputeSpatialHashFromGridPosition(uint4 gridPosition, float3 sampleNormal)
{
    HashGridKey hashKey =
        ((HashGridKey(gridPosition.x) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0)) |
        ((HashGridKey(gridPosition.y) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1)) |
        ((HashGridKey(gridPosition.z) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2)) |
        ((HashGridKey(gridPosition.w) & kHashGridLevelBitMask) << kHashGridLevelBitOffset);

    uint normalBits =
        (sampleNormal.x + kHashGridNormalBias >= 0 ? 0 : 1) +
        (sampleNormal.y + kHashGridNormalBias >= 0 ? 0 : 2) +
        (sampleNormal.z + kHashGridNormalBias >= 0 ? 0 : 4);

    hashKey |= (HashGridKey(normalBits) << kHashGridNormalBitOffset);

    return hashKey;
}

internal HashGridKey hashGridComputeSpatialHashWithVoxelSize(float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters, out float voxelSize)
{
    uint4 gridPosition = uint4(hashGridCalculatePositionLogWithVoxelSize(samplePosition, gridParameters, voxelSize));
    return hashGridComputeSpatialHashFromGridPosition(gridPosition, sampleNormal);
}

public HashGridKey hashGridComputeSpatialHash(float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters)
{
    float voxelSize;
    return hashGridComputeSpatialHashWithVoxelSize(samplePosition, sampleNormal, gridParameters, voxelSize);
}

internal float3 hashGridGetPositionFromKey(const HashGridKey hashKey, HashGridParameters gridParameters)
{
    int3 gridPosition;
    gridPosition.x = int((hashKey >> (kHashGridPositionBitNum * 0)) & kHashGridPositionBitMask);
    gridPosition.y = int((hashKey >> (kHashGridPositionBitNum * 1)) & kHashGridPositionBitMask);
    gridPosition.z = int((hashKey >> (kHashGridPositionBitNum * 2)) & kHashGridPositionBitMask);

    // Sign-extend packed coordinates without divergent branches.
    gridPosition = (gridPosition << (32 - kHashGridPositionBitNum)) >> (32 - kHashGridPositionBitNum);

    uint   gridLevel        = uint((hashKey >> kHashGridLevelBitOffset) & kHashGridLevelBitMask);
    float  voxelSize        = hashGridGetVoxelSize(gridLevel, gridParameters);
    float3 samplePosition   = (gridPosition + 0.5f) * voxelSize;

    return samplePosition;
}

public struct HashGridData
{
    public uint capacity;

    public RWStructuredBuffer<HashGridKey> hashEntriesBuffer;
};

internal void hashGridAtomicCompareExchange(in HashGridData hashData, in uint dstOffset, in HashGridKey compareValue, in HashGridKey value, out HashGridKey originalValue)
{
    InterlockedCompareExchange(hashData.hashEntriesBuffer[dstOffset], compareValue, value, originalValue);
}

public bool hashGridInsert(in HashGridData hashData, const HashGridKey hashKey, uint baseSlot, uint probeRange, out HashGridIndex cacheIndex, out uint bucketOffset)
{
    cacheIndex = kHashGridInvalidCacheIndex;
    probeRange = min(probeRange, kHashGridHashMapBucketSize);

    [loop]
    for (bucketOffset = 0; bucketOffset < probeRange; ++bucketOffset) {
        HashGridKey prevHashKey;
        hashGridAtomicCompareExchange(hashData, baseSlot + bucketOffset, kHashGridInvalidHashKey, hashKey, prevHashKey);

        if (prevHashKey == kHashGridInvalidHashKey || prevHashKey == hashKey) {
            cacheIndex = baseSlot + bucketOffset;
            return true;
        }
    }

    return false;
}

public bool hashGridFind(in HashGridData hashData, const HashGridKey hashKey, uint baseSlot, uint probeRange, out HashGridIndex cacheIndex, out uint bucketOffset)
{
    cacheIndex = kHashGridInvalidCacheIndex;
    probeRange = min(probeRange, kHashGridHashMapBucketSize);
    uint emptyEntries = 0;

    [loop]
    for (bucketOffset = 0; bucketOffset < probeRange; ++bucketOffset) {
        HashGridKey storedHashKey = hashData.hashEntriesBuffer[baseSlot + bucketOffset];
        if (storedHashKey == kHashGridInvalidHashKey) {
            if (emptyEntries > kHashGridLimitEmptySlots)
                break;

            ++emptyEntries;
        }

        if (storedHashKey == hashKey) {
            cacheIndex = baseSlot + bucketOffset;
            return true;
        }
    }

    return false;
}

internal bool hashGridInsertEntry(in HashGridData hashData, float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters, out HashGridKey hashKey, out HashGridIndex cacheIndex)
{
    hashKey                     = hashGridComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    uint baseSlot               = hashGridGetBaseSlot(hashKey, hashData.capacity);
    uint bucketOffset;

    return hashGridInsert(hashData, hashKey, baseSlot, kHashGridHashMapBucketSize, cacheIndex, bucketOffset);
}

internal HashGridIndex hashGridFindEntry(in HashGridData hashData, float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters, out HashGridKey hashKey)
{
    HashGridIndex cacheIndex    = kHashGridInvalidCacheIndex;
    hashKey                     = hashGridComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    uint baseSlot               = hashGridGetBaseSlot(hashKey, hashData.capacity);
    uint bucketOffset;
    hashGridFind(hashData, hashKey, baseSlot, kHashGridHashMapBucketSize, cacheIndex, bucketOffset);

    return cacheIndex;
}

// Debug functions
internal float3 hashGridGetColorFromHash32(uint hash)
{
    float3 color;
    color.x = ((hash >>  0) & 0x3ff) / 1023.0f;
    color.y = ((hash >> 11) & 0x7ff) / 2047.0f;
    color.z = ((hash >> 22) & 0x7ff) / 2047.0f;

    return color;
}

// Debug visualization
internal float3 hashGridDebugColoredHash(float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters)
{
    HashGridKey hashKey  = hashGridComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    uint gridLevel              = hashGridGetLevel(samplePosition, gridParameters);
    float3 color                = hashGridGetColorFromHash32(hashGridHash32(hashKey)) * hashGridGetColorFromHash32(hashGridHashJenkins32(gridLevel)).xyz;

    return color;
}

internal float3 hashGridDebugOccupancy(uint2 pixelPosition, uint2 screenSize, HashGridData hashData)
{
    const uint elementSize = 7;
    const uint borderSize = 1;
    const uint blockSize = elementSize + borderSize;

    uint rowNum = screenSize.y / blockSize;
    uint rowIndex = pixelPosition.y / blockSize;
    uint columnIndex = pixelPosition.x / blockSize;
    uint elementIndex = (columnIndex / kHashGridHashMapBucketSize) * (rowNum * kHashGridHashMapBucketSize) + rowIndex * kHashGridHashMapBucketSize + (columnIndex % kHashGridHashMapBucketSize);

    if (elementIndex < hashData.capacity && ((pixelPosition.x % blockSize) < elementSize && (pixelPosition.y % blockSize) < elementSize)) {
        HashGridKey hashGridKey = hashData.hashEntriesBuffer[elementIndex];
        if (hashGridKey != kHashGridInvalidHashKey)
            return (hashGridKey >> (kHashGridKeyBitNum - 1)) != 0 ? float3(1.0f, 0.75f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    }

    return float3(0.0f, 0.0f, 0.0f);
}

internal float3 hashGridDebugHashCollisions(float3 samplePosition, float3 sampleNormal, HashGridParameters gridParameters, HashGridData hashData)
{
    HashGridKey hashKey  = hashGridComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    HashGridIndex cacheIndex    = kHashGridInvalidCacheIndex;
    uint baseSlot               = hashGridGetBaseSlot(hashKey, hashData.capacity);
    uint bucketOffset;
    hashGridFind(hashData, hashKey, baseSlot, kHashGridHashMapBucketSize, cacheIndex, bucketOffset);

    float3 debugColor;
    if (bucketOffset == 0)
        debugColor = float3(0.0f, 0.0f, 1.0f);
    else if (bucketOffset == 1)
        debugColor = float3(0.0f, 0.5f, 0.5f);
    else if (bucketOffset == 2)
        debugColor = float3(0.0f, 1.0f, 0.0f);
    else if (bucketOffset == 3)
        debugColor = float3(1.0f, 1.0f, 0.0f);
    else if (bucketOffset == 4)
        debugColor = float3(0.75f, 0.25f, 0.0f);
    else
        debugColor = float3(1.0f, 0.0f, 0.0f);

    return debugColor;
}

} // namespace Metallic.SHARC
