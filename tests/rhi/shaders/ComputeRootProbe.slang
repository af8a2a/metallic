#language slang 2026
module ComputeRootProbe;

import Core;
using Metallic;

#ifndef METALLIC_INLINE_COMPUTE_ROOT
#define METALLIC_INLINE_COMPUTE_ROOT 0
#endif

struct ComputeRootProbeConstants
{
    ResourceHandle<ByteAddressBuffer> output;
    uint marker;
    uint dynamicOffset;
    uint padding;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // The output comes from constants, independently of a complete resource
    // packet, so a short packet can prove that all guarded reads return zero.
    let constants = getConstants<ComputeRootProbeConstants>();
    let output = resolveBuffer<RWByteAddressBuffer>(constants.output);
    let root = getComputeResources();
    let packet = getResourceParameters<GPUProbeResourceParameters>();
    output.Store(0, uint(sizeof(ComputeResourceParameters)));
    output.Store(4, root.resources.byteOffset);
    output.Store(8, root.resources.count);
    output.Store(12, root.constants.byteOffset);
    output.Store(16, root.constants.count);
    output.Store(20, constants.marker);
    output.Store(24, packet.input.index);
    output.Store(28, packet.output.index);
    output.Store(32, getResourceParameterField<GPUProbeResourceParameters, uint>(0));
    output.Store(36, getResourceParameterField<GPUProbeResourceParameters, uint>(4));
    output.Store(40, getResourceParameterField<GPUProbeResourceParameters, uint>(8));
    output.Store(44, getResourceParameterField<GPUProbeResourceParameters, uint>(1));
    output.Store(48, getResourceParameterField<GPUProbeResourceParameters, uint>(0xfffffffcu));
    output.Store(52, getResourceParameterField<GPUProbeResourceParameters, uint4>(0).x);
    output.Store(56, getResourceParameterField<GPUProbeResourceParameters, uint>(constants.dynamicOffset));
    output.Store(60, uint(METALLIC_INLINE_COMPUTE_ROOT));
}
