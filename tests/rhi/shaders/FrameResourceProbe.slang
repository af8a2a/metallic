import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct ProbePush
{
    uint index;
};

#define gInput resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<FrameCopyProbeResources>().input)
#define gOutput resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<FrameCopyProbeResources>().output)

[shader("compute")]
[numthreads(1, 1, 1)]
void copyValue(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    gOutput[push.index] = gInput[0];
}

#define gPrevious resolveUniform(getResourceParameters<FrameHistoryProbeResources>().previous)
#define gCurrent resolveUniform(getResourceParameters<FrameHistoryProbeResources>().current)
#define gHistoryResults resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<FrameHistoryProbeResources>().output)

[shader("compute")]
[numthreads(1, 1, 1)]
void accumulateHistory(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    float value = push.index == 0 ? 41.0 : gPrevious[uint2(0, 0)].x + 1.0;
    gCurrent[uint2(0, 0)] = float4(value, 0.0, 0.0, 1.0);
    gHistoryResults[push.index] = uint(value);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void sampleImages(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    float a = resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<FrameImagesProbeResources>().images.load(0))).Load(int3(0, 0, 0)).x;
    float b = resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<FrameImagesProbeResources>().images.load(1))).Load(int3(0, 0, 0)).x;
    resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<FrameImagesProbeResources>().output)[push.index] = uint(a + b);
}
