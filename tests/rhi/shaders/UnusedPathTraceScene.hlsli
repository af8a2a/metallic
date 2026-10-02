#pragma once
#include "../../../Source/Runtime/Render/Core/PathTraceParameters.h"
// Pure helper probes never access scene resources. Initialize handles explicitly:
// Slang descriptor handles do not support aggregate zero initialization.
static const PathTraceParameters kUnusedSceneParameters = {
    (PathTraceSettings)0,
    PathTraceScene(uint2(0, 0)),
    PathTraceOutput(uint2(0, 0)),
    (PathTraceVertices)0,
    (PathTraceIndices)0,
    (PathTracePrimitives)0,
    (PathTraceInstances)0,
    (PathTracePositions)0,
    PathTraceMaterials(uint2(0, 0)),
    PathTraceHistoryCurrent(uint2(0, 0)),
    PathTraceHistoryPrevious(uint2(0, 0)),
    (PathTraceMaterialTextures)0,
    PathTraceEnvironment(uint2(0, 0)),
    PathTraceEnvironmentPdf(uint2(0, 0)),
    (PathTraceLUT2D)0,
    (PathTraceLUT3D)0,
    PathTraceLights(uint2(0, 0)),
    PathTraceReGIR(uint2(0, 0)),
    PathTracePunctualPdf(uint2(0, 0)),
    PathTraceAlbedo(uint2(0, 0)),
    PathTraceSpecularAlbedo(uint2(0, 0)),
    PathTraceNormalRoughness(uint2(0, 0)),
    PathTraceMotionVectors(uint2(0, 0)),
    PathTraceLinearDepth(uint2(0, 0)),
    PathTraceSpecularHitDistance(uint2(0, 0)),
    PathTraceDepth(uint2(0, 0)),
    PathTraceMaterialValues(uint2(0, 0)),
    (PathTraceNTCLatents)0,
    PathTraceNTCConstants(uint2(0, 0)),
    PathTraceNTCWeights(uint2(0, 0)),
    PathTraceNTCInfo(uint2(0, 0)),
    PathTraceNTCSampler(uint2(0, 0))
};
#define gPathTraceParameters kUnusedSceneParameters
