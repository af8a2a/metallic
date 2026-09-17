#ifndef DGC_ADD
#define DGC_ADD 0
#endif

struct ProbePush
{
    uint index;
    uint value;
};

[[vk::push_constant]] ConstantBuffer<ProbePush> probe;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> outputValues;

[shader("compute")]
[numthreads(1, 1, 1)]
void generatedCommandsProbeMain()
{
    outputValues[probe.index] = probe.value + DGC_ADD;
}
