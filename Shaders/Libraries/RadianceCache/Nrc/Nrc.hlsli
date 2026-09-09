/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * Ported from the NRC SDK (Include/Nrc.hlsli) for Metallic's Slang/SPIR-V
 * path tracer. The Update/Query mode selection is compile-time via the
 * NRC_UPDATE / NRC_QUERY macro defines, matching the SDK usage model.
 */

#ifndef __METALLIC_NRC_HLSL__
#define __METALLIC_NRC_HLSL__

#include "NrcHelpers.hlsli"

// NRC can be configured to be in Update or Query modes, or can be
// entirely disabled:
//
// define either NRC_UPDATE=1 or NRC_QUERY=1
//     Forces NRC into either Update or Query mode.
//
// define ENABLE_NRC=0, or leave it undefined
//     Disables NRC; stub functions keep call sites unchanged.

enum class NrcMode
{
    Disabled,
    Update,
    Query
};

#if defined(ENABLE_NRC)
#if !((ENABLE_NRC == 0) || (ENABLE_NRC == 1))
#error "If you #define ENABLE_NRC, please set it to 0 or 1 to disable or enable NRC respectively"
#endif
#else
#if (defined(NRC_UPDATE) || defined(NRC_QUERY))
#define ENABLE_NRC 1
#else
#define ENABLE_NRC 0
#endif
#endif

#if ENABLE_NRC && defined(NRC_UPDATE) && defined(NRC_QUERY)
#if NRC_UPDATE == NRC_QUERY
#error "NRC_UPDATE and NRC_QUERY are mutually exclusive. Please only #define one of them to 1."
#endif
#endif

#if (defined(NRC_UPDATE) || defined(NRC_QUERY))
#if defined(NRC_UPDATE)
#if defined(NRC_QUERY)
#define NRC_QUERY (!NRC_UPDATE)
#else
#define NRC_QUERY 0
#endif
#else
#define NRC_UPDATE 0
#endif

#if NRC_UPDATE
static const NrcMode g_nrcMode = NrcMode::Update;
#else
static const NrcMode g_nrcMode = NrcMode::Query;
#endif

#else

// Disable NRC
#define NRC_QUERY 0
#define NRC_UPDATE 0
static const NrcMode g_nrcMode = NrcMode::Disabled;

#endif

bool NrcIsEnabled()
{
    return g_nrcMode != NrcMode::Disabled;
}

bool NrcIsUpdateMode()
{
    return g_nrcMode == NrcMode::Update;
}

bool NrcIsQueryMode()
{
    return g_nrcMode == NrcMode::Query;
}

enum class NrcProgressState
{
    Continue,
    TerminateImmediately,
    TerminateAfterDirectLighting,
};

#if ENABLE_NRC

// -------------------------------------------------------------------------
// Internal NRC Helpers
// -------------------------------------------------------------------------

/** Internal helper.
    Returns true when we termination heuristic is satisfied and we can terminate path and create the NRC query point
 */
bool NrcEvaluateTerminationHeuristic(const NrcPathState pathState, float threshold)
{
    return (pathState.primarySpreadRadius > 0.0f) && (pathState.cumulSpreadRadius > (threshold * pathState.primarySpreadRadius));
}

// Layout of pathState.packedData
//   +-----------+-------------+--------------+
//   | 15     12 | 11        7 | 6          0 |
//   +-----------+-------------+--------------+
//   |   Flags   | Termination |    Vertex    |
//   |           |   Reason    |     Count    |
//   +-----------+-------------+--------------+
static const uint nrcTerminationReasonShift          = 7;
static const uint nrcPathFlagsShift                  = 12;
static const uint nrcPathFlagHasExitedScene          = (1U << (nrcPathFlagsShift + 0U)); //< Training paths that exited the scene should not be "Q-learned"
static const uint nrcPathFlagIsUnbiased              = (1U << (nrcPathFlagsShift + 1U)); //< Some of the training paths are marked as "unbiased" to be extended through the entire scene
static const uint nrcPathFlagPreviousHitWasDeltaLobe = (1U << (nrcPathFlagsShift + 2U));
static const uint nrcPathFlagHeuristicReset          = (1U << (nrcPathFlagsShift + 3U));
static const uint nrcVertexCountMask                 = ((1U << nrcTerminationReasonShift) - 1);
static const uint nrcTerminationReasonMask           = (((1U << nrcPathFlagsShift) - 1U) & ~nrcVertexCountMask);

void NrcSetFlag(inout uint packedData, in uint flag)
{
    packedData |= flag;
}

void NrcSetFlag(inout uint packedData, in uint flag, in bool value)
{
    packedData &= ~flag;
    packedData |= value ? flag : 0;
}

bool NrcGetFlag(in uint packedData, in uint flag)
{
    return (packedData & flag) ? true : false;
}

uint NrcGetVertexCount(in uint packedData)
{
    return (packedData & nrcVertexCountMask);
}

void NrcSetVertexCount(inout uint packedData, uint vertexCount)
{
    packedData &= ~nrcVertexCountMask;
    packedData |= vertexCount;
}

// -------------------------------------------------------------------------
// Public NRC Shader API
// -------------------------------------------------------------------------

/** When a path is terminated, call this to specify the reason.
    Used for debugging.
*/
void NrcSetDebugPathTerminationReason(inout NrcPathState pathState, NrcDebugPathTerminationReason reason)
{
    pathState.packedData &= ~nrcTerminationReasonMask;
    pathState.packedData |= ((uint)reason) << nrcTerminationReasonShift;
}

/** Creates a new NrcContext
    \param[in] constants. NrcConstants passed from the Nrc SDK library.
    \param[in] buffers. NrcBuffers struct that should be filled in by the app.
    \param[in] pixelIndex. The pixel coordinate.
*/
NrcContext NrcCreateContext(in NrcConstants constants, in NrcBuffers buffers, in uint2 pixelIndex)
{
    NrcContext context;
    context.constants = constants;
    context.buffers = buffers;
    context.pixelIndex = pixelIndex;
    context.sampleIndex = 0;

    return context;
}

/** Creates a fresh NrcPathState for a new path.
    Call this before entering the bounce loop.
    \param[in] constants. The NRC constants.
    \param[in] rand0to1. A random number between 0 and 1.
*/
NrcPathState NrcCreatePathState(in NrcConstants constants, float rand0to1)
{
    NrcPathState pathState = (NrcPathState)0;
    pathState.queryBufferIndex = 0xFFFFFFFF;
    pathState.packedPrefixThroughput = 0;
    pathState.cumulSpreadRadius = 0.0f;
    pathState.primarySpreadRadius = 0.0f;
    pathState.packedData = 0;

    // Get a pseudorandom selection of "unbiased" training paths. Unbiased means that the paths are traced to their full length.
    const bool isUnbiased = NrcIsUpdateMode() && (rand0to1 < constants.proportionUnbiased);
    NrcSetFlag(pathState.packedData, nrcPathFlagIsUnbiased, isUnbiased);

    return pathState;
}

/** Sets the sample index of the path that we're going to trace.
    \param[inout] context
    \param[in]    sampleIndex. Optional sample index when rendering multiple paths per pixel
*/
void NrcSetSampleIndex(inout NrcContext context, in uint sampleIndex)
{
    context.sampleIndex = sampleIndex;
}

/** Determines whether application can use termination by russian roulette for this path.
    RR should not be used when we need to trace long unbiased paths to train the NRC.

    \param[in] pathState. NrcPathState structure created during path tracing
    \return Returns true when application can use russian roulette (RR) for this path.
*/
bool NrcCanUseRussianRoulette(in NrcPathState pathState)
{
    return !NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased);
}

/** This should be called when the path traced ray segment is a 'hit'.
    Note that throughput and radiance will be modified by this function when NRC
    is in Update mode.  This is because NRC needs to know the throughput and direct
    light for each path segment.

    \param[in] context. NrcContext.
    \param[inout] pathState. NrcPathState to be updated.
    \param[in] surfaceAttributes. Information about the surface that was hit.
    \param[in] hitDistance. Distance from the previous hit, or the path length so far if this is
                            the first call with bounce > 0 when using Primary Surface Replacement.
    \param[in] bounce. The index of this bounce (the primary hit is bounce 0)
    \param[inout] throughput. The path tracer's accumulated throughput.
    \param[inout] radiance. The path tracer's accumulated radiance.
    \return Returns NrcProgressState to determine if and when this path should be terminated.
*/
NrcProgressState NrcUpdateOnHit(
    in NrcContext context,
    inout NrcPathState pathState,
    NrcSurfaceAttributes surfaceAttributes,
    float hitDistance,
    uint bounce,
    inout float3 throughput,
    inout float3 radiance)
{
    if (!NrcIsEnabled())
    {
        return NrcProgressState::Continue;
    }

    // Update the path spread approximation, used to trigger the path termination heuristic.
    // The heuristic prevents querying the NRC before the signal has been sufficiently blurred by the path spread.
    // This needs to be calculated even when heuristics is disabled, because it's still used for training paths
    const float cosGamma = abs(dot(surfaceAttributes.viewVector, surfaceAttributes.shadingNormal));
    if (pathState.primarySpreadRadius == 0.f)
    {
        const float kOneOverFourPI = 0.079577471545947667884f; // 1/4pi
        pathState.primarySpreadRadius = hitDistance / sqrt(cosGamma * kOneOverFourPI);
    }
    else if (!NrcGetFlag(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe))
    {
        pathState.cumulSpreadRadius += hitDistance / sqrt(cosGamma * pathState.brdfPdf /* The BRDF PDF of the previous hit */);
    }
    NrcSetFlag(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe, surfaceAttributes.isDeltaLobe);

    // Determine if we want to skip querying NRC at this bounce (e.g., we want skip mirrors)
    const bool skipVertex = (context.constants.skipDeltaVertices != 0u || context.constants.enableTerminationHeuristic != 0u) && surfaceAttributes.isDeltaLobe;
    if (skipVertex)
    {
        return NrcProgressState::Continue;
    }

    uint vertexCount = NrcGetVertexCount(pathState.packedData);
    if (NrcIsUpdateMode())
    {
        // Write training path vertex information
        const uint trainingPathVertexIndex = NrcCalculateTrainingPathVertexIndex(context.constants.trainingDimensions, context.pixelIndex, vertexCount, context.constants.maxPathVertices);
        if (vertexCount > 0)
        {
            // Finalize the previous vertex with the radiance and throughput that the path tracer accumulated
            // during its previous iteration
            const uint previousTrainingPathVertexIndex = trainingPathVertexIndex - 1;
            context.buffers.trainingPathVertices[previousTrainingPathVertexIndex] =
                NrcUpdateTrainingPathVertex(context.buffers.trainingPathVertices[previousTrainingPathVertexIndex], radiance, throughput);
        }

        // Always update vertex counts. The pathState vertexCount variable mostly mirrors 'bounce' variable,
        // but it does not count specular vertices if these were marked to be skipped.
        // This is needed to ensure that a surface scene in a mirror is handled similarly to surfaces seen directly.
        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        // Reset the path tracer's throughput and radiance for the next
        // path segment.
        throughput = float3(1.0f, 1.0f, 1.0f);
        radiance = float3(0.0f, 0.0f, 0.0f);

        // Store path vertex
        context.buffers.trainingPathVertices[trainingPathVertexIndex] = NrcInitializePackedPathVertex(
            surfaceAttributes.roughness, surfaceAttributes.shadingNormal, surfaceAttributes.viewVector, surfaceAttributes.diffuseReflectance, surfaceAttributes.specularF0, surfaceAttributes.encodedPosition);

        bool terminate = (bounce == context.constants.maxPathVertices - 1); //< Is this path at last vertex already? If yes, we can terminate.
        if (!NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased))
        {
            if (NrcEvaluateTerminationHeuristic(pathState, context.constants.trainingTerminationHeuristicThreshold))
            {
                // We should run the path to its normal termination, then reset the spread radius and run again
                // until we hit the termination criteria a second time
                terminate = terminate || NrcGetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                NrcSetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                pathState.cumulSpreadRadius = 0.f;
            }
        }

        if (terminate)
        {
            return (context.constants.radianceCacheDirect != 0u) ? NrcProgressState::TerminateImmediately : NrcProgressState::TerminateAfterDirectLighting;
        }
    }
    else
    {
        // Check if we can query the cache at the current vertex (terminating the path)
        bool createQuery = false;
        if (context.constants.enableTerminationHeuristic != 0u)
        {
            // This evaluates more complex heuristic based on the spread of the ray cone approximating the ray along the path
            createQuery = NrcEvaluateTerminationHeuristic(pathState, context.constants.terminationHeuristicThreshold);
        }
        else
        {
            // Termination criterion enabling debug visualization of the cache by querying at vertex index zero.
            createQuery = vertexCount == 0;
        }

        // Always update vertex counts. The pathState vertexCount variable mostly mirrors 'bounce' variable,
        // but it does not count specular vertices if these were marked to be skipped.
        // This is needed to ensure that a surface scene in a mirror is handled similarly to surfaces seen directly.
        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        // Create query record
        if (createQuery)
        {
            float3 prefixThroughput = (context.constants.learnIrradiance != 0u) ? (throughput * (surfaceAttributes.specularF0 + surfaceAttributes.diffuseReflectance)) : throughput;
            prefixThroughput = max(0.0f, NrcSanitizeNansInfs(prefixThroughput));
            pathState.packedPrefixThroughput = NrcEncodeLogLuvHdr(prefixThroughput);

            pathState.queryBufferIndex = NrcIncrementCounter(context.buffers.countersData, NrcCounter::Queries);

            NrcRadianceParams params;
            params.encodedPosition = surfaceAttributes.encodedPosition;
            params.roughness = surfaceAttributes.roughness;
            params.normal = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.shadingNormal);
            params.viewDirection = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.viewVector);
            params.albedo = surfaceAttributes.diffuseReflectance;
            params.specular = surfaceAttributes.specularF0;

            context.buffers.queryRadianceParams[pathState.queryBufferIndex] = params;

            // Terminate now if the cache already includes direct reflected radiance.
            // Otherwise, we will terminate later, after NEE and the scatter ray has been computed.
            if (context.constants.radianceCacheDirect != 0u)
            {
                NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::CreateQueryImmediate);
                return NrcProgressState::TerminateImmediately;
            }
            else
            {
                NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::CreateQueryAfterDirectLighting);
                return NrcProgressState::TerminateAfterDirectLighting;
            }
        }
    }
    return NrcProgressState::Continue;
}

/** This should be called when the path traced ray segment is a 'miss'.

    \param[inout] pathState. NrcPathState to be updated.
*/
void NrcUpdateOnMiss(inout NrcPathState pathState)
{
    NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::PathMissExit);
    NrcSetFlag(pathState.packedData, nrcPathFlagHasExitedScene);
}

/** Inform NRC of the PDF of the BRDF.
    NRC uses the PDF of the BRDF for its termination heuristic.  A path tracer usually
    evaluates this when figuring out what direction to shoot the next ray.
    Call this function at that point to pass this information to NRC.

    \param[inout] pathState. NrcPathState structure to be updated.
    \param[in] brdfPdf. The PDF of the BRDF.
*/
void NrcSetBrdfPdf(inout NrcPathState pathState, in float brdfPdf)
{
    pathState.brdfPdf = brdfPdf;
}

/** Write out whatever information is required when the path is finished.
    Call this after the path tracer's bounce loop.

    \param[in] context. NrcContext.
    \param[inout] pathState. NrcPathState to be updated.
    \param[in] throughput. The final throughput.
    \param[in] radiance. The final radiance.
*/
void NrcWriteFinalPathInfo(in    NrcContext context,
                           inout NrcPathState pathState,
                           in    float3 throughput,
                           in    float3 radiance)
{
    if (!NrcIsEnabled())
    {
        return;
    }
    if (NrcIsUpdateMode())
    {
        // Training pass

        uint vertexCount = NrcGetVertexCount(pathState.packedData);
        // Only create cache query for self-training if the last vertex throughput is non-zero
        if (vertexCount > 0)
        {
            const uint vertexIndex = vertexCount - 1;
            const uint arrayIndex = NrcCalculateTrainingPathVertexIndex(
                context.constants.trainingDimensions, context.pixelIndex, vertexIndex, context.constants.maxPathVertices);
            context.buffers.trainingPathVertices[arrayIndex] =
                NrcUpdateTrainingPathVertex(context.buffers.trainingPathVertices[arrayIndex], radiance, throughput);

            // Create self-training records for _all_ training paths, including unbiased ones.
            // Without self-training, each training vertex position within the path would matter.
            // Vertices closer to the tail end would receive less indirect illumination, since there
            // are less following vertices, than those closer to the head.
            // An alternative would be to condition the network prediction on the vertex index, but
            // this complicates the task of the network.
            if (!NrcGetFlag(pathState.packedData, nrcPathFlagHasExitedScene) && (context.constants.maxPathVertices > 1))
            {
                NrcPathVertex vertex = NrcUnpackPathVertex(context.buffers.trainingPathVertices[arrayIndex], radiance, throughput);
                pathState.queryBufferIndex = NrcIncrementCounter(context.buffers.countersData, NrcCounter::Queries);

                context.buffers.queryRadianceParams[pathState.queryBufferIndex] = NrcCreateRadianceParams(vertex);
            }
        }

        NrcTrainingPathInfo unpackedPathInfo = (NrcTrainingPathInfo)0;
        unpackedPathInfo.packedData = pathState.packedData;
        unpackedPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        const uint trainingPathIndex = NrcCalculateTrainingPathIndex(context.constants.trainingDimensions, context.pixelIndex);
        context.buffers.trainingPathInfo[trainingPathIndex] = NrcPackTrainingPathInfo(unpackedPathInfo);
    }
    else
    {
        // Query pass

        const uint queryPathIndex = NrcCalculateQueryPathIndex(context.constants.frameDimensions, context.pixelIndex, context.sampleIndex, context.constants.samplesPerPixel);

        NrcPackedQueryPathInfo packedQueryPathInfo = (NrcPackedQueryPathInfo)0;

        // The prefix throughput was saved separately in case
        // radianceCacheDirect is set to false, in which case path throughput would also include
        // the BSDF weight of the query vertex, not just the prefix throughput.
        packedQueryPathInfo.prefixThroughput = pathState.packedPrefixThroughput;
        packedQueryPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        context.buffers.queryPathInfo[queryPathIndex] = packedQueryPathInfo;
    }
}

#else // !ENABLE_NRC

//
// Define stub functions for when NRC is disabled so that the caller does not
// need to guard code with preprocessor macros if they don't want to.
//

void NrcSetDebugPathTerminationReason(NrcPathState pathState, NrcDebugPathTerminationReason reason)
{}

NrcContext NrcCreateContext(NrcConstants constants, NrcBuffers buffers, uint2 pixelIndex)
{
    NrcContext context;
    context.constants = constants;
    context.buffers = buffers;
    context.pixelIndex = pixelIndex;
    context.sampleIndex = 0;
    return context;
}

NrcPathState NrcCreatePathState(NrcConstants constants, float rand0to1)
{
    NrcPathState pathState = (NrcPathState)0;
    pathState.queryBufferIndex = 0xFFFFFFFF;
    return pathState;
}

void NrcSetSampleIndex(NrcContext context, uint sampleIndex)
{}

bool NrcCanUseRussianRoulette(NrcPathState pathState)
{
    return true;
}

NrcProgressState NrcUpdateOnHit(NrcContext context, inout NrcPathState pathState, NrcSurfaceAttributes surfaceAttributes, float hitDistance, uint bounce, inout float3 throughput, inout float3 radiance)
{
    return NrcProgressState::Continue;
}

void NrcUpdateOnMiss(inout NrcPathState pathState)
{}

void NrcSetBrdfPdf(inout NrcPathState pathState, float brdfPdf)
{}

void NrcWriteFinalPathInfo(NrcContext context, inout NrcPathState pathState, float3 throughput, float3 radiance)
{}

#endif // !ENABLE_NRC

#endif // __METALLIC_NRC_HLSL__
