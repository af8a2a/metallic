import Core;
using Metallic;

struct ProbePush
{
    uint index;
};

#define gInput getResource<StructuredBuffer<uint>>(0)
#define gOutput getResource<RWStructuredBuffer<uint>>(1)

[shader("compute")]
[numthreads(1, 1, 1)]
void copyValue(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    gOutput[push.index] = gInput[0];
}

#define gPrevious getResource<RWTexture2D<float4>>(0)
#define gCurrent getResource<RWTexture2D<float4>>(1)
#define gHistoryResults getResource<RWStructuredBuffer<uint>>(2)

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
    float a = getResourceArray<Texture2D<float4>>(0, 0).Load(int3(0, 0, 0)).x;
    float b = getResourceArray<Texture2D<float4>>(0, 1).Load(int3(0, 0, 0)).x;
    gOutput[push.index] = uint(a + b);
}
