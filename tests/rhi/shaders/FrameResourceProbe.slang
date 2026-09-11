#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

struct ProbePush
{
    uint index;
};

#define gInput METALLIC_RESOURCE(StructuredBuffer<uint>, 0)
#define gOutput METALLIC_RESOURCE(RWStructuredBuffer<uint>, 1)

[shader("compute")]
[numthreads(1, 1, 1)]
void copyValue(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = METALLIC_CONSTANTS(ProbePush);
    gOutput[push.index] = gInput[0];
}

#define gPrevious METALLIC_RESOURCE(RWTexture2D<float4>, 0)
#define gCurrent METALLIC_RESOURCE(RWTexture2D<float4>, 1)
#define gHistoryResults METALLIC_RESOURCE(RWStructuredBuffer<uint>, 2)

[shader("compute")]
[numthreads(1, 1, 1)]
void accumulateHistory(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = METALLIC_CONSTANTS(ProbePush);
    float value = push.index == 0 ? 41.0 : gPrevious[uint2(0, 0)].x + 1.0;
    gCurrent[uint2(0, 0)] = float4(value, 0.0, 0.0, 1.0);
    gHistoryResults[push.index] = uint(value);
}
