struct RenderGraphBufferPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint inputBuffer;
    uint outputBuffer;
    uint passIndex;
    uint padding;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferWriteMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform RenderGraphBufferPush push)
{
    DescriptorHandle<RWByteAddressBuffer> outputHandle =
        DescriptorHandle<RWByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.outputBuffer, 0));

    outputHandle->Store4(0, uint4(0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferCopyMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform RenderGraphBufferPush push)
{
    DescriptorHandle<ByteAddressBuffer> inputHandle =
        DescriptorHandle<ByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.inputBuffer, 0));
    DescriptorHandle<RWByteAddressBuffer> outputHandle =
        DescriptorHandle<RWByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.outputBuffer, 0));

    const uint4 value = inputHandle->Load4(0);
    outputHandle->Store4(0, value);
}
