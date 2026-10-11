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
 * Ported from RTXGI v2 (Libraries/Sharc/include/SharcCommon.h, v1.8) for
 * Metallic's Slang/SPIR-V path tracer. Differences from upstream:
 *  - resolved radiance is stored as fp16 halves packed into uint2 (see Types.slang)
 *  - responsive lighting and spherical-harmonics encoding are not ported
 * Everything else (hash grid usage, accumulation, resolve, resampling,
 * adjacent-level blending) follows the reference implementation.
 */

implementing SHARC;

namespace Metallic.SHARC
{
public struct SHARCParameters
{
    public HashGridParameters hashGridParameters;
    public HashGridData hashGridData;
    public float radianceScale;            // quantization factor for atomic radiance accumulation (u32 per channel during SHARC_UPDATE). Start with 1e3f; reduce for large radiance values to prevent overflow

    public RWStructuredBuffer<SHARCAccumulationData> accumulationBuffer;
    public RWStructuredBuffer<SHARCPackedData> resolvedBuffer;
};

public struct SHARCState
{
    public HashGridIndex cacheIndices[kSHARCPropagationDepth];
    public float3 sampleWeights[kSHARCPropagationDepth];
    public uint pathLength;
};

public struct SHARCHitData
{
    public float3 positionWorld;
    public float3 normalWorld;             // geometry normal in world space. Shading or object-space normals should work, but are not generally recommended
};

public struct SHARCRadianceData
{
    public float3 radiance;
};

public struct SHARCResolveParameters
{
    public float3 cameraPositionPrev;      // previous camera position
    public uint accumulationFrameNum;      // maximum number of frames for the temporal accumulation window
    public uint staleFrameNumMax;          // maximum number of frames without new samples before the cache entry is evicted
    public uint frameIndex;
};

internal SHARCRadianceData sharcZeroRadianceData()
{
    SHARCRadianceData radianceData;
    radianceData.radiance = float3(0, 0, 0);

    return radianceData;
}

internal SHARCRadianceData sharcAddRadianceData(SHARCRadianceData a, SHARCRadianceData b)
{
    SHARCRadianceData result;
    result.radiance = a.radiance + b.radiance;

    return result;
}

internal SHARCRadianceData sharcScaleRadianceData(SHARCRadianceData radianceData, float scale)
{
    SHARCRadianceData result;
    result.radiance = radianceData.radiance * scale;

    return result;
}

internal uint sharcPackFloat16(float value)
{
    return f32tof16(value) & 0xFFFFu;
}

internal float sharcUnpackFloat16(uint value)
{
    return f16tof32(value & 0xFFFFu);
}

internal uint sharcPackFloat16Pair(float2 value)
{
    return sharcPackFloat16(value.x) | (sharcPackFloat16(value.y) << 16);
}

internal float2 sharcUnpackFloat16Pair(uint value)
{
    return float2(sharcUnpackFloat16(value), sharcUnpackFloat16(value >> 16));
}

public SHARCPackedData sharcZeroPackedData()
{
    SHARCPackedData packedData;
    packedData.radianceData = uint2(0, 0);
    packedData.sampleData = 0;
    packedData.sampleDataExt = 0;

    return packedData;
}

public SHARCAccumulationData sharcZeroAccumulationData()
{
    SHARCAccumulationData accumulatedData;
    accumulatedData.data = uint4(0, 0, 0, 0);

    return accumulatedData;
}

internal float sharcGetAccumulatedSampleNum(SHARCAccumulationData accumulatedData)
{
    return float(accumulatedData.data.w);
}

internal SHARCRadianceData sharcGetAccumulatedRadianceData(SHARCAccumulationData accumulatedData, float radianceScale, float sampleNum)
{
    SHARCRadianceData radianceData;
    float scale = rcp(radianceScale * max(sampleNum, 1e-6f));
    radianceData.radiance = float3(accumulatedData.data.xyz) * scale;

    return radianceData;
}

internal SHARCPackedData sharcPackVoxelData(SHARCRadianceData radianceData, float sampleNum, uint accumulatedFrameNum, uint staleFrameNum, uint sampleDataExt)
{
    const float float16Max = 65504.0f;

    SHARCPackedData packedData;
    packedData.radianceData.x = sharcPackFloat16Pair(
        clamp(radianceData.radiance.xy, float2(-float16Max, -float16Max), float2(float16Max, float16Max)));
    packedData.radianceData.y = sharcPackFloat16Pair(
        clamp(float2(radianceData.radiance.z, sampleNum), float2(-float16Max, -float16Max), float2(float16Max, float16Max)));
    packedData.sampleData = accumulatedFrameNum | (staleFrameNum << kSHARCStaleFrameNumBitOffset);
    packedData.sampleDataExt = sampleDataExt;

    return packedData;
}

public struct SHARCVoxelData
{
    public SHARCRadianceData accumulatedRadiance;
    public float accumulatedSampleNum;
    public uint accumulatedFrameNum;
    public uint staleFrameNum;
    public uint sampleDataExt;
};

internal SHARCVoxelData sharcUnpackVoxelData(SHARCPackedData packedData)
{
    SHARCVoxelData voxelData;
    float2 radianceXY = sharcUnpackFloat16Pair(packedData.radianceData.x);
    float2 radianceZAndSamples = sharcUnpackFloat16Pair(packedData.radianceData.y);
    voxelData.accumulatedRadiance.radiance = float3(radianceXY.x, radianceXY.y, radianceZAndSamples.x);
    voxelData.accumulatedSampleNum = radianceZAndSamples.y;
    voxelData.accumulatedFrameNum = (packedData.sampleData >> kSHARCAccumulatedFrameNumBitOffset) & kSHARCAccumulatedFrameNumBitMask;
    voxelData.staleFrameNum = (packedData.sampleData >> kSHARCStaleFrameNumBitOffset) & kSHARCStaleFrameNumBitMask;
    voxelData.sampleDataExt = packedData.sampleDataExt;

    return voxelData;
}

public SHARCVoxelData sharcGetVoxelData(RWStructuredBuffer<SHARCPackedData> voxelDataBuffer, HashGridIndex hashGridIndex)
{
    SHARCPackedData packedData = voxelDataBuffer[hashGridIndex];

    return sharcUnpackVoxelData(packedData);
}

internal float sharcLuma(float3 color)
{
    const float3 luma = float3(0.213f, 0.715f, 0.072f);

    return dot(color, luma);
}

public void sharcAddVoxelData(in SHARCParameters sharcParameters, HashGridIndex hashGridIndex, float3 sampleValue, float3 sampleWeight, uint sampleData)
{
    uint3 scaledRadiance = uint3(max(sampleValue * sampleWeight * sharcParameters.radianceScale, float3(0.0f, 0.0f, 0.0f)));

    if (scaledRadiance.x != 0) InterlockedAdd(sharcParameters.accumulationBuffer[hashGridIndex].data.x, scaledRadiance.x);
    if (scaledRadiance.y != 0) InterlockedAdd(sharcParameters.accumulationBuffer[hashGridIndex].data.y, scaledRadiance.y);
    if (scaledRadiance.z != 0) InterlockedAdd(sharcParameters.accumulationBuffer[hashGridIndex].data.z, scaledRadiance.z);
    if (sampleData != 0) InterlockedAdd(sharcParameters.accumulationBuffer[hashGridIndex].data.w, sampleData);
}

public void sharcInit(inout SHARCState sharcState)
{
    sharcState.pathLength = 0;
}

public void sharcUpdateMiss(in SHARCParameters sharcParameters, in SHARCState sharcState, float3 radiance)
{
    // This is an update operation in every importer. Query passes never call it.
    for (int i = 0; i < int(sharcState.pathLength); ++i) {
        HashGridIndex hashGridIndex = sharcState.cacheIndices[i];
        sharcAddVoxelData(sharcParameters, hashGridIndex, radiance, sharcState.sampleWeights[i], 1);
    }
}

public bool sharcUpdateHit(in SHARCParameters sharcParameters, inout SHARCState sharcState, SHARCHitData sharcHitData, float3 directLighting, float random)
{
    bool continueTracing = true;
    HashGridKey hashGridKey;
    HashGridIndex hashGridIndex;
    if (!hashGridInsertEntry(sharcParameters.hashGridData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters, hashGridKey, hashGridIndex))
        return false;

    float3 sharcRadiance = directLighting;

    uint resamplingDepth = uint(round(lerp(float(kSHARCResamplingDepthMin), float(kSHARCPropagationDepth), random)));
    if (resamplingDepth <= sharcState.pathLength) {
        SHARCVoxelData voxelData = sharcGetVoxelData(sharcParameters.resolvedBuffer, hashGridIndex);
        if (voxelData.accumulatedSampleNum > kSHARCSampleNumThreshold) {
            sharcRadiance = voxelData.accumulatedRadiance.radiance;
            continueTracing = false;
        }
    }

    if (continueTracing) {
        sharcAddVoxelData(sharcParameters, hashGridIndex, directLighting, float3(1.0f, 1.0f, 1.0f), 1);
    }

    uint i;
    for (i = 0; i < sharcState.pathLength; ++i) {
        HashGridIndex tempHashGridIndex = sharcState.cacheIndices[i];
        if (tempHashGridIndex == kHashGridInvalidCacheIndex)
            continue;

        sharcAddVoxelData(sharcParameters, tempHashGridIndex, sharcRadiance, sharcState.sampleWeights[i], 0);
    }

    for (i = min(sharcState.pathLength, kSHARCPropagationDepth - 1); i > 0; --i) {
        sharcState.cacheIndices[i] = sharcState.cacheIndices[i - 1];
        sharcState.sampleWeights[i] = sharcState.sampleWeights[i - 1];
    }

    sharcState.cacheIndices[0] = hashGridIndex;
    sharcState.sampleWeights[0] = float3(1.0f, 1.0f, 1.0f);
    sharcState.pathLength = min(++sharcState.pathLength, kSHARCPropagationDepth);
    return continueTracing;
}

public void sharcSetThroughput(inout SHARCState sharcState, float3 throughput)
{
    for (uint i = 0; i < sharcState.pathLength; ++i)
        sharcState.sampleWeights[i] *= throughput;
}

internal bool sharcGetCachedRadianceFromHash(in SHARCParameters sharcParameters, in SHARCHitData sharcHitData, HashGridKey hashGridKey, out float3 radiance)
{
    radiance = float3(0.0f, 0.0f, 0.0f);

    HashGridIndex hashGridIndex;
    uint baseSlot = hashGridGetBaseSlot(hashGridKey, sharcParameters.hashGridData.capacity);
    uint bucketOffset;
    if (!hashGridFind(sharcParameters.hashGridData, hashGridKey, baseSlot, kHashGridHashMapBucketSize, hashGridIndex, bucketOffset))
        return false;

    SHARCVoxelData voxelData = sharcGetVoxelData(sharcParameters.resolvedBuffer, hashGridIndex);
    if (voxelData.accumulatedSampleNum > kSHARCSampleNumThreshold) {
        radiance = voxelData.accumulatedRadiance.radiance;
        return true;
    }

    return false;
}

public bool sharcGetCachedRadiance(in SHARCParameters sharcParameters, in SHARCHitData sharcHitData, out float3 radiance)
{
    HashGridKey hashGridKey = hashGridComputeSpatialHash(sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters);
    return sharcGetCachedRadianceFromHash(sharcParameters, sharcHitData, hashGridKey, radiance);
}

internal int sharcGetGridDistance2(int3 position)
{
    return position.x * position.x + position.y * position.y + position.z * position.z;
}

internal HashGridKey sharcGetAdjacentLevelHashKey(HashGridKey hashGridKey, HashGridParameters gridParameters, float3 cameraPositionPrev)
{
    int3 gridPosition;
    gridPosition.x = int((hashGridKey >> kHashGridPositionBitNum * 0) & kHashGridPositionBitMask);
    gridPosition.y = int((hashGridKey >> kHashGridPositionBitNum * 1) & kHashGridPositionBitMask);
    gridPosition.z = int((hashGridKey >> kHashGridPositionBitNum * 2) & kHashGridPositionBitMask);

    // Sign-extend packed coordinates without divergent branches.
    gridPosition = (gridPosition << (32 - kHashGridPositionBitNum)) >> (32 - kHashGridPositionBitNum);

    int level = int((hashGridKey >> kHashGridLevelBitOffset) & kHashGridLevelBitMask);

    float voxelSize = hashGridGetVoxelSize(uint(level), gridParameters);
    float inverseVoxelSize = rcp(voxelSize);
    int3 cameraGridPosition = int3(floor(gridParameters.cameraPosition * inverseVoxelSize));
    int3 cameraVector = cameraGridPosition - gridPosition;
    int cameraDistance = sharcGetGridDistance2(cameraVector);

    int3 cameraGridPositionPrev = int3(floor(cameraPositionPrev * inverseVoxelSize));
    int3 cameraVectorPrev = cameraGridPositionPrev - gridPosition;
    int cameraDistancePrev = sharcGetGridDistance2(cameraVectorPrev);

    if (cameraDistance < cameraDistancePrev) {
        gridPosition = int3(floor(gridPosition / gridParameters.logarithmBase));
        level = min(level + 1, int(kHashGridLevelBitMask));
    }
    else // this may be inaccurate
    {
        gridPosition = int3(floor(gridPosition * gridParameters.logarithmBase));
        level = max(level - 1, 1);
    }

    HashGridKey modifiedHashGridKey = ((HashGridKey(gridPosition.x) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 0))
        | ((HashGridKey(gridPosition.y) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 1))
        | ((HashGridKey(gridPosition.z) & kHashGridPositionBitMask) << (kHashGridPositionBitNum * 2))
        | ((HashGridKey(level) & kHashGridLevelBitMask) << kHashGridLevelBitOffset);

    modifiedHashGridKey |= hashGridKey & (HashGridKey(kHashGridNormalBitMask) << kHashGridNormalBitOffset);

    return modifiedHashGridKey;
}

public void sharcResolveEntry(uint entryIndex, SHARCParameters sharcParameters, SHARCResolveParameters resolveParameters)
{
    if (entryIndex >= sharcParameters.hashGridData.capacity)
        return;

    HashGridKey hashGridKey = sharcParameters.hashGridData.hashEntriesBuffer[entryIndex];
    if (hashGridKey == kHashGridInvalidHashKey)
        return;

    SHARCAccumulationData accumulatedData = sharcParameters.accumulationBuffer[entryIndex];
    SHARCPackedData resolvedData = sharcParameters.resolvedBuffer[entryIndex];
    SHARCVoxelData sharcVoxelData = sharcUnpackVoxelData(resolvedData);

    float sampleNum = sharcGetAccumulatedSampleNum(accumulatedData);
    float sampleNumPrev = sharcVoxelData.accumulatedSampleNum;
    uint accumulatedFrameNum = sharcVoxelData.accumulatedFrameNum + 1;
    uint staleFrameNum = sharcVoxelData.staleFrameNum;

    staleFrameNum = (sampleNum != 0) ? 0 : staleFrameNum + 1;
    uint staleFrameNumMax = clamp(resolveParameters.staleFrameNumMax, kSHARCStaleFrameNumMin, kSHARCStaleFrameNumMax);

    bool isValidElement = (staleFrameNum < staleFrameNumMax) ? true : false;
    if (!isValidElement) {
        SHARCAccumulationData zeroAccumulationData = sharcZeroAccumulationData();
        SHARCPackedData zeroPackedData = sharcZeroPackedData();

        sharcParameters.hashGridData.hashEntriesBuffer[entryIndex] = kHashGridInvalidHashKey;
        sharcParameters.accumulationBuffer[entryIndex] = zeroAccumulationData;
        sharcParameters.resolvedBuffer[entryIndex] = zeroPackedData;
        return;
    } else if (sampleNum == 0) {
        // Resolve dispatches exactly one thread per entry, so no atomic is needed.
        sharcParameters.resolvedBuffer[entryIndex].sampleData = resolvedData.sampleData + ((1 << kSHARCAccumulatedFrameNumBitOffset) | (1 << kSHARCStaleFrameNumBitOffset));
        return;
    }

    // Performs hash map lookup to find existing entries in case previous insertions
    // encountered collisions and a different slot was assigned.
    // Uses a fixed-size linear probe window
    if (sampleNumPrev == 0) {
        uint searchEnd = min(entryIndex + 1 + kSHARCLinearProbeWindowSize, sharcParameters.hashGridData.capacity);
        for (uint i = entryIndex + 1; i < searchEnd; ++i) {
            HashGridKey hashKeyOld = sharcParameters.hashGridData.hashEntriesBuffer[i];
            if (hashKeyOld == hashGridKey) {
                resolvedData = sharcParameters.resolvedBuffer[i];
                sharcVoxelData = sharcUnpackVoxelData(resolvedData);
                sampleNumPrev = sharcVoxelData.accumulatedSampleNum;
                accumulatedFrameNum = sharcVoxelData.accumulatedFrameNum + 1;
                staleFrameNum = 0;
                break;
            }
        }
    }

    SHARCRadianceData accumulatedRadiance = sharcGetAccumulatedRadianceData(accumulatedData, sharcParameters.radianceScale, sampleNum);
    SHARCRadianceData accumulatedRadiancePrev = sharcVoxelData.accumulatedRadiance;
    uint accumulationFrameNum = clamp(resolveParameters.accumulationFrameNum, kSHARCAccumulatedFrameNumMin, kSHARCAccumulatedFrameNumMax);
    if (accumulatedFrameNum > accumulationFrameNum) {
        float normalizationScale = float(accumulationFrameNum) / float(accumulatedFrameNum);
        accumulatedFrameNum = accumulationFrameNum;
        sampleNumPrev *= normalizationScale;
    }

    float sampleTotalInv = rcp(sampleNumPrev + sampleNum);
    accumulatedRadiance = sharcAddRadianceData(sharcScaleRadianceData(accumulatedRadiancePrev, sampleNumPrev * sampleTotalInv), sharcScaleRadianceData(accumulatedRadiance, sampleNum * sampleTotalInv));
    float accumulatedSampleNum = sampleNumPrev + sampleNum;

    // Reproject sample from adjacent level
    float3 cameraOffset = sharcParameters.hashGridParameters.cameraPosition.xyz - resolveParameters.cameraPositionPrev.xyz;
    if ((dot(cameraOffset, cameraOffset) > 1e-6f) && (accumulatedFrameNum <= 2)) {
        HashGridKey adjacentLevelHashKey = sharcGetAdjacentLevelHashKey(hashGridKey, sharcParameters.hashGridParameters, resolveParameters.cameraPositionPrev);

        HashGridIndex hashGridIndex;
        uint baseSlot = hashGridGetBaseSlot(adjacentLevelHashKey, sharcParameters.hashGridData.capacity);
        uint hashCollisionsNum;
        if (hashGridFind(sharcParameters.hashGridData, adjacentLevelHashKey, baseSlot, kHashGridHashMapBucketSize, hashGridIndex, hashCollisionsNum)) {
            SHARCPackedData adjacentPackedDataPrev = sharcParameters.resolvedBuffer[hashGridIndex];
            SHARCVoxelData adjacentVoxelDataPrev = sharcUnpackVoxelData(adjacentPackedDataPrev);
            float adjacentSampleNum = adjacentVoxelDataPrev.accumulatedSampleNum;
            if (adjacentSampleNum > kSHARCSampleNumThreshold) {
                float blendWeight = rcp(adjacentSampleNum + accumulatedSampleNum);
                accumulatedRadiance = sharcAddRadianceData(
                    sharcScaleRadianceData(adjacentVoxelDataPrev.accumulatedRadiance, adjacentSampleNum * blendWeight),
                    sharcScaleRadianceData(accumulatedRadiance, accumulatedSampleNum * blendWeight));
                accumulatedSampleNum += adjacentSampleNum;
            }
        }
    }

    sharcParameters.resolvedBuffer[entryIndex] = sharcPackVoxelData(accumulatedRadiance, accumulatedSampleNum, accumulatedFrameNum, staleFrameNum, sharcVoxelData.sampleDataExt);

    // Clear buffer entry for the next frame
    SHARCAccumulationData zeroAccumulationData = sharcZeroAccumulationData();
    sharcParameters.accumulationBuffer[entryIndex] = zeroAccumulationData;
}

} // namespace Metallic.SHARC
