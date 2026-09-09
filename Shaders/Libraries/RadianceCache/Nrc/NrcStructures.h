/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * Ported from the NRC SDK (Include/NrcStructures.h) for Metallic's Slang/SPIR-V
 * path tracer. Shader-side declarations only; the C++ side uses the SDK
 * headers directly. Field order/widths must match the SDK structures exactly
 * (Slang's scalar structured-buffer layout matches the SDK's element sizes).
 */

#ifndef __METALLIC_NRC_STRUCTURES_H__
#define __METALLIC_NRC_STRUCTURES_H__

// Set to 0: TCNN is expecting UNORM positions in fp32 format
#define TCNN_USES_FIXED_POINT_POSITIONS 0

typedef uint nrc_uint;
typedef uint2 nrc_uint2;
typedef float2 nrc_float2;
typedef float3 nrc_float3;
typedef float4 nrc_float4;

#if TCNN_USES_FIXED_POINT_POSITIONS
typedef nrc_uint3 NrcEncodedPosition;
#else
typedef nrc_float3 NrcEncodedPosition;
#endif

/** Enumeration of the atomic counters used.
 */
enum class NrcCounter
{
    Queries = 0,
    TrainingRecords = 1,

    // Must be last
    Count
};

/**
 *  Additional information about radiance evaluated/cached at certain point
 */
struct NrcRadianceParams
{
    NrcEncodedPosition encodedPosition;
    float roughness;
    nrc_float2 normal; // Shading normal. If unavailable, a geometry normal should be used instead
    nrc_float2 viewDirection; // Direction towards the viewer or opposite direction of the incident ray

    nrc_float3 albedo; // Diffuse albedo of the hit surface
    nrc_float3 specular; // Specular albedo of the hit surface
};

/**
 *  Information about the path being traced, needed to reconstruct the path and resolve radiance to create the final image
 */
struct NrcTrainingPathInfo
{
    uint packedData;
    uint queryBufferIndex;
};

/**
 *  Packed version of NrcTrainingPathInfo (right now it looks just the same)
 */
struct NrcPackedTrainingPathInfo
{
    uint packedData;
    uint queryBufferIndex;
};

/**
 *  Information about the path being traced, needed to resolve radiance to create the final image
 */
struct NrcQueryPathInfo
{
    nrc_float3 prefixThroughput;
    uint queryBufferIndex;
};

/**
 *  Packed version of NrcQueryPathInfo
 */
struct NrcPackedQueryPathInfo
{
    uint prefixThroughput;
    uint queryBufferIndex;
};

/**
 *  Struct holding path vertex data for training the NRC.
 */
struct NrcPathVertex
{
    nrc_float3 radiance; ///< Reflected radiance
    nrc_float3 throughput; ///< Throughput to the next vertex

    NrcEncodedPosition encodedPosition; ///< World space position squashed into unorm range
    float linearRoughness; ///< Material roughness

    nrc_float3 normal; ///< Sampled direction
    nrc_float3 viewDirection; ///< Direction towards the previous path vertex

    nrc_float3 albedo; ///< Base diffuse reflectance
    nrc_float3 specular; ///< Base specular reflectance
};

/**
 *  Packed version of NrcPathVertex
 */
struct NrcPackedPathVertex
{
    uint data[7];
    uint pad0;

    NrcEncodedPosition encodedPosition;
    uint pad1;
};

/**
 *  Debug Path Termination Reasons
 */
enum class NrcDebugPathTerminationReason
{
    Unset = 0,
    PathMissExit,
    CreateQueryImmediate,
    MaxPathVertices,
    CreateQueryAfterDirectLighting,
    RussianRoulette,
    BRDFAbsorption,
    Count
};

enum class NrcResolveMode
{
    // The default behaviour.
    // This takes the query result and adds it to the output buffer.
    AddQueryResultToOutput = 0,

    // A debug mode that overwrites the output buffer with the query results
    ReplaceOutputWithQueryResult,

    // A debug mode that shows a heatmap for the number of training bounces.
    TrainingBounceHeatMap,

    // The same as TrainingBounceHeatMap, but smoothed over time
    TrainingBounceHeatMapSmoothed,

    // A debug mode that shows the reconstructed radiance for the primary
    // ray segment.
    PrimaryVertexTrainingRadiance,

    // The same as PrimaryVertexTrainingRadiance, but smoothed over time
    PrimaryVertexTrainingRadianceSmoothed,

    // As PrimaryVertexTrainingRadiance, but for the secondary ray segment
    SecondaryVertexTrainingRadiance,

    // The same as SecondaryVertexTrainingRadiance, but smoothed over time
    SecondaryVertexTrainingRadianceSmoothed,

    // A debug mode that shows a random colour that's a hash of the query index.
    QueryIndex,

    // Same as QueryIndex, but for the training pass's self-training records.
    TrainingQueryIndex,

    // Direct visualization of the cache (equivalent of querying at vertex zero).
    DirectCacheView,
};

/**
 *  Holds common parameters needed by NRC functions called from the path tracer.
 *  The app should fill this in using Context::PopulateShaderConstants and then
 *  pass it up to the path tracing shader.
 *  Layout must match ::NrcConstants from the SDK headers byte for byte.
 */
struct NrcConstants
{
    nrc_uint2 frameDimensions;
    nrc_uint2 trainingDimensions;

    nrc_float3 scenePosScale;
    nrc_uint samplesPerPixel;

    NrcEncodedPosition scenePosBias;
    nrc_uint maxPathVertices;

    nrc_uint learnIrradiance;
    nrc_uint radianceCacheDirect;
    float radianceUnpackMultiplier; // See NrcUnpackQueryRadiance
    int resolveMode;

    nrc_uint enableTerminationHeuristic;
    nrc_uint skipDeltaVertices;
    float terminationHeuristicThreshold;
    float trainingTerminationHeuristicThreshold;

    float proportionUnbiased;
    nrc_uint pad0;
    nrc_uint pad1;
    nrc_uint pad2;
};

/**
 *  Holds state data about path being traced.
 */
struct NrcPathState
{
    uint packedPrefixThroughput;
    uint queryBufferIndex;

    float primarySpreadRadius; ///< Approximated as `d^2 / cos` at primary hit.
    float cumulSpreadRadius; ///< Square root of the cumulative area spread at the current path vertex.

    uint packedData; ///< The number of vertices processed, flags and termination reason
    float brdfPdf;
};

/**
 *  Attributes of the surface hit by the ray, needed to evaluate NRC query and training data
 */
struct NrcSurfaceAttributes
{
    NrcEncodedPosition encodedPosition; // Use NrcEncodePosition
    float roughness;
    nrc_float3 specularF0;
    nrc_float3 diffuseReflectance;
    nrc_float3 shadingNormal;
    nrc_float3 viewVector;
    bool isDeltaLobe;
};

/**
 *  Holds common parameters needed by NRC functions; called from the path tracer
 */
struct NrcBuffers
{
    RWStructuredBuffer<NrcPackedQueryPathInfo> queryPathInfo;
    RWStructuredBuffer<NrcPackedTrainingPathInfo> trainingPathInfo;
    RWStructuredBuffer<NrcPackedPathVertex> trainingPathVertices;
    RWStructuredBuffer<NrcRadianceParams> queryRadianceParams;
    RWStructuredBuffer<uint> countersData;
};

/**
 *  Create an NrcContext with NrcCreateState.
 *  This structure is used by nearly all Nrc functions in Nrc.hlsli
 */
struct NrcContext
{
    NrcConstants constants;
    NrcBuffers buffers;

    nrc_uint2 pixelIndex;
    uint sampleIndex;
};

#endif // __METALLIC_NRC_STRUCTURES_H__
