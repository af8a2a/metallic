import ShaderCore;
using Metallic;

struct RenderGraphBufferPush
{
    uint inputBuffer;
    uint outputBuffer;
    uint passIndex;
    uint padding;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferWriteMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform RenderGraphBufferPush push)
{
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    outputHandle.Store4(0, uint4(0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferCopyMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform RenderGraphBufferPush push)
{
    ByteAddressBuffer inputHandle =
        resolveUniform(ResourceHandle<ByteAddressBuffer>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    const uint4 value = inputHandle.Load4(0);
    outputHandle.Store4(0, value);
}
