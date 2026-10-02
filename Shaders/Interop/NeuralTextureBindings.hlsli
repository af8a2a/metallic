// Caller-side bridge: the inference module has no parameter-root dependency.
NeuralTextureBindings sceneNeuralTextures()
{
#if !METALLIC_HAS_NTC
    return (NeuralTextureBindings)0;
#elif METALLIC_PATH_TRACE_TYPED
    return {gPathTraceParameters.ntcLatents, resolveDescriptor(gPathTraceParameters.ntcConstants),
        resolveDescriptor(gPathTraceParameters.ntcWeights), resolveDescriptor(gPathTraceParameters.ntcInfo),
        resolveDescriptor(gPathTraceParameters.ntcSampler)};
#else
    uint2 address = getComputeResources().resources[29].payload;
    return {(DescriptorHandle<Texture2DArray<float4>>*)(uint64_t(address.x) | (uint64_t(address.y) << 32)),
        getResource<NeuralTextureConstantBuffer>(30), getResource<ByteAddressBuffer>(31),
        getResource<StructuredBuffer<uint4>>(32), getResource<SamplerState>(33)};
#endif
}
