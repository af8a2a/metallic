import Core;
using Metallic;

#include "../TestResourceParameters.h"
#include "../../../Shaders/Features/PathTracing/SceneSurface.slang"

#define gMisInput resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<FrameCopyProbeResources>().input)
#define gMisOutput resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<FrameCopyProbeResources>().output)

[shader("compute")]
[numthreads(1, 1, 1)]
void misPowerHeuristicProbeMain(uint3 id : SV_DispatchThreadID)
{
    // Load PDFs from the host so the regression exercises runtime arithmetic.
    float2 pdfs = gMisInput[id.x].xy;
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.environmentMode = kEnvironmentModeMap;
    gMisOutput[id.x * 2] = float4(
        misPowerHeuristic(pdfs.x, pdfs.y),
        misPowerHeuristic(pdfs.y, pdfs.x),
        environmentLightMisWeight(push, pdfs.x, pdfs.y),
        environmentLightMisWeight(push, pdfs.y, pdfs.x));

    // Delta paths bypass environment PDF lookup. Procedural/disabled lighting
    // likewise retains unit MIS weight without scene resources.
    float deltaWeight = environmentBsdfMisWeight(push, pdfs.x, float3(0, 1, 0), true);
    push.environmentMode = kEnvironmentModeProcedural;
    float proceduralLightWeight = environmentLightMisWeight(push, pdfs.x, pdfs.y);
    float proceduralBsdfWeight = environmentBsdfMisWeight(push, pdfs.x, float3(0, 1, 0), false);
    push.environmentMode = kEnvironmentModeDisabled;
    float disabledLightWeight = environmentLightMisWeight(push, pdfs.x, pdfs.y);
    gMisOutput[id.x * 2 + 1] = float4(
        deltaWeight, proceduralLightWeight, proceduralBsdfWeight, disabledLightWeight);
}
