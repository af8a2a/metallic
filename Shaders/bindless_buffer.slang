struct BindlessBufferPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint inputBuffer;
    uint outputBuffer;
    uint passIndex;
    uint padding;
};

struct ConstantData
{
    uint x;
    uint y;
    uint z;
    uint w;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void bindlessBufferConstantMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    DescriptorHandle<ConstantBuffer<ConstantData>> inputHandle =
        DescriptorHandle<ConstantBuffer<ConstantData>>(uint2(push.bufferShaderIndexBase + push.inputBuffer, 0));
    DescriptorHandle<RWByteAddressBuffer> outputHandle =
        DescriptorHandle<RWByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.outputBuffer, 0));

    outputHandle->Store4(0, uint4((*inputHandle).x, (*inputHandle).y, (*inputHandle).z, (*inputHandle).w));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bindlessBufferRwByteAddressMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    DescriptorHandle<RWByteAddressBuffer> inputHandle =
        DescriptorHandle<RWByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.inputBuffer, 0));
    DescriptorHandle<RWByteAddressBuffer> outputHandle =
        DescriptorHandle<RWByteAddressBuffer>(uint2(push.bufferShaderIndexBase + push.outputBuffer, 0));

    if (push.passIndex == 0) {
        inputHandle->Store(0, 0xDEADBEEFu);
        inputHandle->Store2(4, uint2(0x11223344u, 0xAABBCCDDu));
        inputHandle->Store4(12, uint4(1u, 2u, 3u, 4u));
    } else {
        const uint v0 = inputHandle->Load(0);
        const uint2 v1 = inputHandle->Load2(4);
        const uint4 v2 = inputHandle->Load4(12);
        outputHandle->Store(0, v0);
        outputHandle->Store2(4, v1);
        outputHandle->Store4(12, v2);
    }
}
