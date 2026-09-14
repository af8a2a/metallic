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
 *  - resolved radiance is stored as fp16 halves packed into uint2 (see SharcTypes.h)
 *  - responsive lighting and spherical-harmonics encoding are not ported
 * Everything else (hash grid usage, accumulation, resolve, resampling,
 * adjacent-level blending) follows the reference implementation.
 */

// SHaRC usage overview
//
// SHaRC uses a world-space radiance cache to reduce path tracing cost. The
// integration consists of three render-time passes:
//
// 1. SHaRC Update
//    Sparse RT pass, compiled with SHARC_UPDATE=1, that traces a subset of
//    paths and writes newly observed radiance into the accumulation cache.
//    Call SharcInit() for each sampled path, SharcUpdateHit() on hits, and
//    SharcUpdateMiss() on misses. If SharcUpdateHit() returns false, the path
//    must be terminated early. Each path segment is treated independently:
//    after selecting the next ray, call SharcSetThroughput() with the segment
//    throughput, then reset path throughput to 1.0 for the next segment.
//
// 2. SHaRC Resolve
//    Compute pass that calls SharcResolveEntry() for each cache entry. This
//    combines per-frame accumulation data with previously resolved data,
//    updates temporal accumulation, handles stale entry eviction, and clears
//    accumulation data for the next frame.
//
// 3. SHaRC Render / Query
//    Main render pass, compiled with SHARC_QUERY=1, that traces normally but
//    queries the resolved cache on eligible non-primary hits using
//    SharcGetCachedRadiance(). On a successful query, cached radiance is used
//    and the path can terminate early.
//
// Required resources are hash entries, accumulation, and resolved buffers.
// All buffers must have the same number of entries and be zero-initialized.
// Proper barriers are required between passes to guarantee correctness.

// Version
#define SHARC_VERSION_MAJOR                     1
#define SHARC_VERSION_MINOR                     8
#define SHARC_VERSION_BUILD                     3
#define SHARC_VERSION_REVISION                  0

// Normalize the update/query mode defines so preprocessor conditionals never
// see undefined identifiers (mirrors Nrc.hlsli's handling).
#if defined(SHARC_UPDATE) && defined(SHARC_QUERY)
// both defined by the application; nothing to do
#elif defined(SHARC_UPDATE)
#define SHARC_QUERY 0
#elif defined(SHARC_QUERY)
#define SHARC_UPDATE 0
#else
#define SHARC_UPDATE 0
#define SHARC_QUERY 0
#endif

// Constants
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET  0
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM     16
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK    ((1 << SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM) - 1)
#define SHARC_STALE_FRAME_NUM_BIT_OFFSET        16
#define SHARC_STALE_FRAME_NUM_BIT_NUM           16
#define SHARC_STALE_FRAME_NUM_BIT_MASK          ((1 << SHARC_STALE_FRAME_NUM_BIT_NUM) - 1)
#define SHARC_GRID_LOGARITHM_BASE               2.0f
#define SHARC_ACCUMULATED_FRAME_NUM_MIN         1       // minimum number of frames to use for data accumulation
#define SHARC_ACCUMULATED_FRAME_NUM_MAX         1024    // maximum number of frames to use for data accumulation
#define SHARC_STALE_FRAME_NUM_MAX               1024    // maximum number of frames without new samples before the cache entry is evicted

// Tweakable parameters
#ifndef SHARC_SAMPLE_NUM_THRESHOLD
#define SHARC_SAMPLE_NUM_THRESHOLD              0       // elements with sample count above this threshold will be used for early-out/resampling
#endif

#ifndef SHARC_LINEAR_PROBE_WINDOW_SIZE
#define SHARC_LINEAR_PROBE_WINDOW_SIZE          8       // size of the linear search window for probe lookups
#endif

#ifndef SHARC_ENABLE_CACHE_RESAMPLING
#define SHARC_ENABLE_CACHE_RESAMPLING           SHARC_UPDATE // resamples the cache during update step
#endif

#ifndef SHARC_PROPAGATION_DEPTH
#if SHARC_ENABLE_CACHE_RESAMPLING
#define SHARC_PROPAGATION_DEPTH                 2       // controls the amount of vertices stored in memory for signal backpropagation with cache resampling
#else // !SHARC_ENABLE_CACHE_RESAMPLING
#define SHARC_PROPAGATION_DEPTH                 4       // controls the amount of vertices stored in memory for signal backpropagation
#endif // SHARC_ENABLE_CACHE_RESAMPLING
#endif

#ifndef SHARC_BLEND_ADJACENT_LEVELS
#define SHARC_BLEND_ADJACENT_LEVELS             1       // combine the data from adjacent levels on camera movement
#endif

#ifndef SHARC_RESAMPLING_DEPTH_MIN
#define SHARC_RESAMPLING_DEPTH_MIN              1       // controls minimum path depth which can be used with cache resampling
#endif

#ifndef SHARC_STALE_FRAME_NUM_MIN
#define SHARC_STALE_FRAME_NUM_MIN               8       // minimum number of frames to keep the element in the cache
#endif

#ifndef SHARC_GRID_LEVEL_BIAS
#define SHARC_GRID_LEVEL_BIAS                   0       // LOD bias - positive adds extra magnified levels, negative reduces levels
#endif

#ifndef HASH_GRID_COMPACT
#define HASH_GRID_COMPACT                       0
#endif

#ifndef HASH_GRID_LIMIT_EMPTY_SLOTS
#define HASH_GRID_LIMIT_EMPTY_SLOTS             2
#endif

#ifndef RW_STRUCTURED_BUFFER
#define RW_STRUCTURED_BUFFER(name, type)        RWStructuredBuffer<type> name
#endif

#ifndef BUFFER_AT_OFFSET
#define BUFFER_AT_OFFSET(name, offset)          name[offset]
#endif

#define HashGridKey uint64_t
#define HASH_GRID_KEY_TYPE HashGridKey
#define HASH_GRID_PREFIX HashGrid
#define HASH_GRID_CONST_PREFIX HASH_GRID
#include "HashGridTypes.h"
#include "HashGridCommon.h"
#undef HASH_GRID_KEY_TYPE
#undef HASH_GRID_PREFIX
#undef HASH_GRID_CONST_PREFIX

#include "SharcTypes.h"

struct SharcParameters
{
    HashGridParameters hashGridParameters;
    HashGridData hashGridData;
    float radianceScale;            // quantization factor for atomic radiance accumulation (u32 per channel during SHARC_UPDATE). Start with 1e3f; reduce for large radiance values to prevent overflow

    RW_STRUCTURED_BUFFER(accumulationBuffer, SharcAccumulationData);
    RW_STRUCTURED_BUFFER(resolvedBuffer, SharcPackedData);
};

struct SharcState
{
#if SHARC_UPDATE
    HashGridIndex cacheIndices[SHARC_PROPAGATION_DEPTH];
    float3 sampleWeights[SHARC_PROPAGATION_DEPTH];
    uint pathLength;
#else // !SHARC_UPDATE
    uint placeholder;               // prevents empty-struct compilation issues
#endif // SHARC_UPDATE
};

struct SharcHitData
{
    float3 positionWorld;
    float3 normalWorld;             // geometry normal in world space. Shading or object-space normals should work, but are not generally recommended
};

struct SharcRadianceData
{
    float3 radiance;
};

struct SharcResolveParameters
{
    float3 cameraPositionPrev;      // previous camera position
    uint accumulationFrameNum;      // maximum number of frames for the temporal accumulation window
    uint staleFrameNumMax;          // maximum number of frames without new samples before the cache entry is evicted
    uint frameIndex;
};

SharcRadianceData SharcZeroRadianceData()
{
    SharcRadianceData radianceData;
    radianceData.radiance = float3(0, 0, 0);

    return radianceData;
}

SharcRadianceData SharcAddRadianceData(SharcRadianceData a, SharcRadianceData b)
{
    SharcRadianceData result;
    result.radiance = a.radiance + b.radiance;

    return result;
}

SharcRadianceData SharcScaleRadianceData(SharcRadianceData radianceData, float scale)
{
    SharcRadianceData result;
    result.radiance = radianceData.radiance * scale;

    return result;
}

uint SharcPackFloat16(float value)
{
    return f32tof16(value) & 0xFFFFu;
}

float SharcUnpackFloat16(uint value)
{
    return f16tof32(value & 0xFFFFu);
}

uint SharcPackFloat16Pair(float2 value)
{
    return SharcPackFloat16(value.x) | (SharcPackFloat16(value.y) << 16);
}

float2 SharcUnpackFloat16Pair(uint value)
{
    return float2(SharcUnpackFloat16(value), SharcUnpackFloat16(value >> 16));
}

SharcPackedData SharcZeroPackedData()
{
    SharcPackedData packedData;
    packedData.radianceData = uint2(0, 0);
    packedData.sampleData = 0;
    packedData.sampleDataExt = 0;

    return packedData;
}

SharcAccumulationData SharcZeroAccumulationData()
{
    SharcAccumulationData accumulatedData;
    accumulatedData.data = uint4(0, 0, 0, 0);

    return accumulatedData;
}

float SharcGetAccumulatedSampleNum(SharcAccumulationData accumulatedData)
{
    return float(accumulatedData.data.w);
}

SharcRadianceData SharcGetAccumulatedRadianceData(SharcAccumulationData accumulatedData, float radianceScale, float sampleNum)
{
    SharcRadianceData radianceData;
    float scale = rcp(radianceScale * max(sampleNum, 1e-6f));
    radianceData.radiance = float3(accumulatedData.data.xyz) * scale;

    return radianceData;
}

SharcPackedData SharcPackVoxelData(SharcRadianceData radianceData, float sampleNum, uint accumulatedFrameNum, uint staleFrameNum, uint sampleDataExt)
{
    const float float16Max = 65504.0f;

    SharcPackedData packedData;
    packedData.radianceData.x = SharcPackFloat16Pair(
        clamp(radianceData.radiance.xy, float2(-float16Max, -float16Max), float2(float16Max, float16Max)));
    packedData.radianceData.y = SharcPackFloat16Pair(
        clamp(float2(radianceData.radiance.z, sampleNum), float2(-float16Max, -float16Max), float2(float16Max, float16Max)));
    packedData.sampleData = accumulatedFrameNum | (staleFrameNum << SHARC_STALE_FRAME_NUM_BIT_OFFSET);
    packedData.sampleDataExt = sampleDataExt;

    return packedData;
}

struct SharcVoxelData_SharcCommon
{
    SharcRadianceData accumulatedRadiance;
    float accumulatedSampleNum;
    uint accumulatedFrameNum;
    uint staleFrameNum;
    uint sampleDataExt;
};

SharcVoxelData_SharcCommon SharcUnpackVoxelData(SharcPackedData packedData)
{
    SharcVoxelData_SharcCommon voxelData;
    float2 radianceXY = SharcUnpackFloat16Pair(packedData.radianceData.x);
    float2 radianceZAndSamples = SharcUnpackFloat16Pair(packedData.radianceData.y);
    voxelData.accumulatedRadiance.radiance = float3(radianceXY.x, radianceXY.y, radianceZAndSamples.x);
    voxelData.accumulatedSampleNum = radianceZAndSamples.y;
    voxelData.accumulatedFrameNum = (packedData.sampleData >> SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) & SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK;
    voxelData.staleFrameNum = (packedData.sampleData >> SHARC_STALE_FRAME_NUM_BIT_OFFSET) & SHARC_STALE_FRAME_NUM_BIT_MASK;
    voxelData.sampleDataExt = packedData.sampleDataExt;

    return voxelData;
}

SharcVoxelData_SharcCommon SharcGetVoxelData(RW_STRUCTURED_BUFFER(voxelDataBuffer, SharcPackedData), HashGridIndex hashGridIndex)
{
    SharcPackedData packedData = BUFFER_AT_OFFSET(voxelDataBuffer, hashGridIndex);

    return SharcUnpackVoxelData(packedData);
}

float SharcLuma(float3 color)
{
    const float3 luma = float3(0.213f, 0.715f, 0.072f);

    return dot(color, luma);
}

void SharcAddVoxelData(in SharcParameters sharcParameters, HashGridIndex hashGridIndex, float3 sampleValue, float3 sampleWeight, uint sampleData)
{
    uint3 scaledRadiance = uint3(max(sampleValue * sampleWeight * sharcParameters.radianceScale, float3(0.0f, 0.0f, 0.0f)));

    if (scaledRadiance.x != 0) InterlockedAdd(BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, hashGridIndex).data.x, scaledRadiance.x);
    if (scaledRadiance.y != 0) InterlockedAdd(BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, hashGridIndex).data.y, scaledRadiance.y);
    if (scaledRadiance.z != 0) InterlockedAdd(BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, hashGridIndex).data.z, scaledRadiance.z);
    if (sampleData != 0) InterlockedAdd(BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, hashGridIndex).data.w, sampleData);
}

void SharcInit(inout SharcState sharcState)
{
#if SHARC_UPDATE
    sharcState.pathLength = 0;
#endif // SHARC_UPDATE
}

void SharcUpdateMiss(in SharcParameters sharcParameters, in SharcState sharcState, float3 radiance)
{
#if SHARC_UPDATE
    for (int i = 0; i < int(sharcState.pathLength); ++i)
    {
        HashGridIndex hashGridIndex = sharcState.cacheIndices[i];
        SharcAddVoxelData(sharcParameters, hashGridIndex, radiance, sharcState.sampleWeights[i], 1);
    }
#endif // SHARC_UPDATE
}

bool SharcUpdateHit(in SharcParameters sharcParameters, inout SharcState sharcState, SharcHitData sharcHitData, float3 directLighting, float random)
{
    bool continueTracing = true;
#if SHARC_UPDATE
    HashGridKey hashGridKey;
    HashGridIndex hashGridIndex;
    if (!HashGridInsertEntry(sharcParameters.hashGridData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters, hashGridKey, hashGridIndex))
        return false;

    float3 sharcRadiance = directLighting;

#if SHARC_ENABLE_CACHE_RESAMPLING
    uint resamplingDepth = uint(round(lerp(float(SHARC_RESAMPLING_DEPTH_MIN), float(SHARC_PROPAGATION_DEPTH), random)));
    if (resamplingDepth <= sharcState.pathLength)
    {
        SharcVoxelData_SharcCommon voxelData = SharcGetVoxelData(sharcParameters.resolvedBuffer, hashGridIndex);
        if (voxelData.accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
        {
            sharcRadiance = voxelData.accumulatedRadiance.radiance;
            continueTracing = false;
        }
    }
#endif // SHARC_ENABLE_CACHE_RESAMPLING

    if (continueTracing)
    {
        SharcAddVoxelData(sharcParameters, hashGridIndex, directLighting, float3(1.0f, 1.0f, 1.0f), 1);
    }

    uint i;
    for (i = 0; i < sharcState.pathLength; ++i)
    {
        HashGridIndex tempHashGridIndex = sharcState.cacheIndices[i];
        if (tempHashGridIndex == HASH_GRID_INVALID_CACHE_INDEX)
            continue;

        SharcAddVoxelData(sharcParameters, tempHashGridIndex, sharcRadiance, sharcState.sampleWeights[i], 0);
    }

    for (i = min(sharcState.pathLength, SHARC_PROPAGATION_DEPTH - 1); i > 0; --i)
    {
        sharcState.cacheIndices[i] = sharcState.cacheIndices[i - 1];
        sharcState.sampleWeights[i] = sharcState.sampleWeights[i - 1];
    }

    sharcState.cacheIndices[0] = hashGridIndex;
    sharcState.sampleWeights[0] = float3(1.0f, 1.0f, 1.0f);
    sharcState.pathLength = min(++sharcState.pathLength, SHARC_PROPAGATION_DEPTH);
#endif // SHARC_UPDATE
    return continueTracing;
}

void SharcSetThroughput(inout SharcState sharcState, float3 throughput)
{
#if SHARC_UPDATE
    for (uint i = 0; i < sharcState.pathLength; ++i)
        sharcState.sampleWeights[i] *= throughput;
#endif // SHARC_UPDATE
}

bool SharcGetCachedRadianceFromHash(in SharcParameters sharcParameters, in SharcHitData sharcHitData, HashGridKey hashGridKey, out float3 radiance)
{
    radiance = float3(0.0f, 0.0f, 0.0f);

    HashGridIndex hashGridIndex;
    uint baseSlot = HashGridGetBaseSlot(hashGridKey, sharcParameters.hashGridData.capacity);
    uint bucketOffset;
    if (!HashGridFind(sharcParameters.hashGridData, hashGridKey, baseSlot, HASH_GRID_HASH_MAP_BUCKET_SIZE, hashGridIndex, bucketOffset))
        return false;

    SharcVoxelData_SharcCommon voxelData = SharcGetVoxelData(sharcParameters.resolvedBuffer, hashGridIndex);
    if (voxelData.accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
    {
        radiance = voxelData.accumulatedRadiance.radiance;
        return true;
    }

    return false;
}

bool SharcGetCachedRadiance(in SharcParameters sharcParameters, in SharcHitData sharcHitData, out float3 radiance)
{
    HashGridKey hashGridKey = HashGridComputeSpatialHash(sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters);
    return SharcGetCachedRadianceFromHash(sharcParameters, sharcHitData, hashGridKey, radiance);
}

int SharcGetGridDistance2(int3 position)
{
    return position.x * position.x + position.y * position.y + position.z * position.z;
}

HashGridKey SharcGetAdjacentLevelHashKey(HashGridKey hashGridKey, HashGridParameters gridParameters, float3 cameraPositionPrev)
{
    int3 gridPosition;
    gridPosition.x = int((hashGridKey >> HASH_GRID_POSITION_BIT_NUM * 0) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashGridKey >> HASH_GRID_POSITION_BIT_NUM * 1) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashGridKey >> HASH_GRID_POSITION_BIT_NUM * 2) & HASH_GRID_POSITION_BIT_MASK);

    // Sign-extend packed coordinates without divergent branches.
    gridPosition = (gridPosition << (32 - HASH_GRID_POSITION_BIT_NUM)) >> (32 - HASH_GRID_POSITION_BIT_NUM);

    int level = int((hashGridKey >> HASH_GRID_LEVEL_BIT_OFFSET) & HASH_GRID_LEVEL_BIT_MASK);

    float voxelSize = HashGridGetVoxelSize(uint(level), gridParameters);
    float inverseVoxelSize = rcp(voxelSize);
    int3 cameraGridPosition = int3(floor(gridParameters.cameraPosition * inverseVoxelSize));
    int3 cameraVector = cameraGridPosition - gridPosition;
    int cameraDistance = SharcGetGridDistance2(cameraVector);

    int3 cameraGridPositionPrev = int3(floor(cameraPositionPrev * inverseVoxelSize));
    int3 cameraVectorPrev = cameraGridPositionPrev - gridPosition;
    int cameraDistancePrev = SharcGetGridDistance2(cameraVectorPrev);

    if (cameraDistance < cameraDistancePrev)
    {
        gridPosition = int3(floor(gridPosition / gridParameters.logarithmBase));
        level = min(level + 1, int(HASH_GRID_LEVEL_BIT_MASK));
    }
    else // this may be inaccurate
    {
        gridPosition = int3(floor(gridPosition * gridParameters.logarithmBase));
        level = max(level - 1, 1);
    }

    HashGridKey modifiedHashGridKey = ((HashGridKey(gridPosition.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
        | ((HashGridKey(gridPosition.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
        | ((HashGridKey(gridPosition.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
        | ((HashGridKey(level) & HASH_GRID_LEVEL_BIT_MASK) << HASH_GRID_LEVEL_BIT_OFFSET);

#if HASH_GRID_USE_NORMALS
    modifiedHashGridKey |= hashGridKey & (HashGridKey(HASH_GRID_NORMAL_BIT_MASK) << HASH_GRID_NORMAL_BIT_OFFSET);
#endif // HASH_GRID_USE_NORMALS

    return modifiedHashGridKey;
}

void SharcResolveEntry(uint entryIndex, SharcParameters sharcParameters, SharcResolveParameters resolveParameters)
{
    if (entryIndex >= sharcParameters.hashGridData.capacity)
        return;

    HashGridKey hashGridKey = BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, entryIndex);
    if (hashGridKey == HASH_GRID_INVALID_HASH_KEY)
        return;

    SharcAccumulationData accumulatedData = BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, entryIndex);
    SharcPackedData resolvedData = BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, entryIndex);
    SharcVoxelData_SharcCommon sharcVoxelData = SharcUnpackVoxelData(resolvedData);

    float sampleNum = SharcGetAccumulatedSampleNum(accumulatedData);
    float sampleNumPrev = sharcVoxelData.accumulatedSampleNum;
    uint accumulatedFrameNum = sharcVoxelData.accumulatedFrameNum + 1;
    uint staleFrameNum = sharcVoxelData.staleFrameNum;

    staleFrameNum = (sampleNum != 0) ? 0 : staleFrameNum + 1;
    uint staleFrameNumMax = clamp(resolveParameters.staleFrameNumMax, SHARC_STALE_FRAME_NUM_MIN, SHARC_STALE_FRAME_NUM_MAX);

    bool isValidElement = (staleFrameNum < staleFrameNumMax) ? true : false;
    if (!isValidElement)
    {
        SharcAccumulationData zeroAccumulationData = SharcZeroAccumulationData();
        SharcPackedData zeroPackedData = SharcZeroPackedData();

        BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
        BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, entryIndex) = zeroAccumulationData;
        BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, entryIndex) = zeroPackedData;
        return;
    }
    else if (sampleNum == 0)
    {
        // Resolve dispatches exactly one thread per entry, so no atomic is needed.
        BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, entryIndex).sampleData = resolvedData.sampleData + ((1 << SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) | (1 << SHARC_STALE_FRAME_NUM_BIT_OFFSET));
        return;
    }

    // Performs hash map lookup to find existing entries in case previous insertions
    // encountered collisions and a different slot was assigned.
    // Uses a fixed-size linear probe window
    if (sampleNumPrev == 0)
    {
        uint searchEnd = min(entryIndex + 1 + SHARC_LINEAR_PROBE_WINDOW_SIZE, sharcParameters.hashGridData.capacity);
        for (uint i = entryIndex + 1; i < searchEnd; ++i)
        {
            HashGridKey hashKeyOld = BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, i);
            if (hashKeyOld == hashGridKey)
            {
                resolvedData = BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, i);
                sharcVoxelData = SharcUnpackVoxelData(resolvedData);
                sampleNumPrev = sharcVoxelData.accumulatedSampleNum;
                accumulatedFrameNum = sharcVoxelData.accumulatedFrameNum + 1;
                staleFrameNum = 0;
                break;
            }
        }
    }

    SharcRadianceData accumulatedRadiance = SharcGetAccumulatedRadianceData(accumulatedData, sharcParameters.radianceScale, sampleNum);
    SharcRadianceData accumulatedRadiancePrev = sharcVoxelData.accumulatedRadiance;
    uint accumulationFrameNum = clamp(resolveParameters.accumulationFrameNum, SHARC_ACCUMULATED_FRAME_NUM_MIN, SHARC_ACCUMULATED_FRAME_NUM_MAX);
    if (accumulatedFrameNum > accumulationFrameNum)
    {
        float normalizationScale = float(accumulationFrameNum) / float(accumulatedFrameNum);
        accumulatedFrameNum = accumulationFrameNum;
        sampleNumPrev *= normalizationScale;
    }

    float sampleTotalInv = rcp(sampleNumPrev + sampleNum);
    accumulatedRadiance = SharcAddRadianceData(SharcScaleRadianceData(accumulatedRadiancePrev, sampleNumPrev * sampleTotalInv), SharcScaleRadianceData(accumulatedRadiance, sampleNum * sampleTotalInv));
    float accumulatedSampleNum = sampleNumPrev + sampleNum;

#if SHARC_BLEND_ADJACENT_LEVELS
    // Reproject sample from adjacent level
    float3 cameraOffset = sharcParameters.hashGridParameters.cameraPosition.xyz - resolveParameters.cameraPositionPrev.xyz;
    if ((dot(cameraOffset, cameraOffset) > 1e-6f) && (accumulatedFrameNum <= 2))
    {
        HashGridKey adjacentLevelHashKey = SharcGetAdjacentLevelHashKey(hashGridKey, sharcParameters.hashGridParameters, resolveParameters.cameraPositionPrev);

        HashGridIndex hashGridIndex;
        uint baseSlot = HashGridGetBaseSlot(adjacentLevelHashKey, sharcParameters.hashGridData.capacity);
        uint hashCollisionsNum;
        if (HashGridFind(sharcParameters.hashGridData, adjacentLevelHashKey, baseSlot, HASH_GRID_HASH_MAP_BUCKET_SIZE, hashGridIndex, hashCollisionsNum))
        {
            SharcPackedData adjacentPackedDataPrev = BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, hashGridIndex);
            SharcVoxelData_SharcCommon adjacentVoxelDataPrev = SharcUnpackVoxelData(adjacentPackedDataPrev);
            float adjacentSampleNum = adjacentVoxelDataPrev.accumulatedSampleNum;
            if (adjacentSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
            {
                float blendWeight = rcp(adjacentSampleNum + accumulatedSampleNum);
                accumulatedRadiance = SharcAddRadianceData(
                    SharcScaleRadianceData(adjacentVoxelDataPrev.accumulatedRadiance, adjacentSampleNum * blendWeight),
                    SharcScaleRadianceData(accumulatedRadiance, accumulatedSampleNum * blendWeight));
                accumulatedSampleNum += adjacentSampleNum;
            }
        }
    }
#endif // SHARC_BLEND_ADJACENT_LEVELS

    BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, entryIndex) = SharcPackVoxelData(accumulatedRadiance, accumulatedSampleNum, accumulatedFrameNum, staleFrameNum, sharcVoxelData.sampleDataExt);

    // Clear buffer entry for the next frame
    SharcAccumulationData zeroAccumulationData = SharcZeroAccumulationData();
    BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, entryIndex) = zeroAccumulationData;
}
