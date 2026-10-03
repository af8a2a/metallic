import ShaderCore;
#ifndef INLINE_PARAMETERS
#define INLINE_PARAMETERS 0
#endif
#if !INLINE_PARAMETERS
import ParameterRoot;
#endif
using Metallic;

struct RegistryProbeParams {
    ResourceHandle<StructuredBuffer<uint>> source;
    ResourceHandle<RWStructuredBuffer<uint>> output;
    uint add;
    uint index;
};

#if INLINE_PARAMETERS
[[vk::push_constant]] ConstantBuffer<RegistryProbeParams> gParams;
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void registryProbeMain()
{
#if INLINE_PARAMETERS
    RegistryProbeParams params = gParams;
#else
    RegistryProbeParams params = getParameters<RegistryProbeParams>();
#endif
    resolveUniform(params.output)[params.index] = resolveUniform(params.source)[0] + params.add;
}
