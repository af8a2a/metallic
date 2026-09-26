import Core;
import ParameterRoot;
using Metallic;

struct RegistryProbeParams {
    DescriptorHandle<StructuredBuffer<uint>> source;
    DescriptorHandle<RWStructuredBuffer<uint>> output;
    uint add;
    uint index;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void registryProbeMain()
{
    RegistryProbeParams params = getParameters<RegistryProbeParams>();
    resolveDescriptor(params.output)[params.index] = resolveDescriptor(params.source)[0] + params.add;
}
