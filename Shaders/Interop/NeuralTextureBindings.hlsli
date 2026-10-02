// Caller-side bridge: the inference module has no parameter-root dependency.
NeuralTextureBindings sceneNeuralTextures()
{
#if !METALLIC_HAS_NTC
    return (NeuralTextureBindings)0;
#else
    return {gPathTraceParameters.ntcLatents, resolveDescriptor(gPathTraceParameters.ntcConstants),
        resolveDescriptor(gPathTraceParameters.ntcWeights), resolveDescriptor(gPathTraceParameters.ntcInfo),
        resolveDescriptor(gPathTraceParameters.ntcSampler)};
#endif
}
