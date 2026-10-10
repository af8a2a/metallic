import ShaderCore;
using Metallic;

#ifndef DGC_ADD
#define DGC_ADD 0
#endif

struct ProbePush
{
    uint index;
    uint value;
    ResourceHandle<RWStructuredBuffer<uint>> output;
};

[[vk::push_constant]] ConstantBuffer<ProbePush> probe;

[shader("compute")]
[numthreads(1, 1, 1)]
void generatedCommandsProbeMain()
{
    resolveUniform(probe.output)[probe.index] = probe.value + DGC_ADD;
}
