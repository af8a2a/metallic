struct ProbePush
{
    uint index;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> gInput;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gOutput;

[shader("compute")]
[numthreads(1, 1, 1)]
void copyValue(uint3 thread : SV_DispatchThreadID, uniform ProbePush push)
{
    gOutput[push.index] = gInput[0];
}

[[vk::binding(0, 0)]] RWTexture2D<float4> gPrevious;
[[vk::binding(1, 0)]] RWTexture2D<float4> gCurrent;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gHistoryResults;

[shader("compute")]
[numthreads(1, 1, 1)]
void accumulateHistory(uint3 thread : SV_DispatchThreadID, uniform ProbePush push)
{
    float value = push.index == 0 ? 41.0 : gPrevious[uint2(0, 0)].x + 1.0;
    gCurrent[uint2(0, 0)] = float4(value, 0.0, 0.0, 1.0);
    gHistoryResults[push.index] = uint(value);
}
