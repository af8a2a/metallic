import ShaderCore;
using Metallic;

struct BindlessBufferPush
{
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

struct Element
{
    uint x;
    uint y;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void bindlessBufferConstantMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    ConstantBuffer<ConstantData> inputHandle =
        resolveUniform(ResourceHandle<ConstantBuffer<ConstantData>>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    outputHandle.Store4(0, uint4(inputHandle.x, inputHandle.y, inputHandle.z, inputHandle.w));
}

[shader("compute")]
[numthreads(4, 1, 1)]
void bindlessBufferStructuredMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    StructuredBuffer<Element> inputHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<Element>>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    const uint index = dispatchThreadId.x;
    const Element element = inputHandle[index];
    outputHandle.Store2(index * 8u, uint2(element.x, element.y));
}

[shader("compute")]
[numthreads(4, 1, 1)]
void bindlessBufferRwStructuredMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    RWStructuredBuffer<Element> inputHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<Element>>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    const uint index = dispatchThreadId.x;
    if (push.passIndex == 0) {
        Element element;
        element.x = index * 2u;
        element.y = index * 2u + 1u;
        inputHandle[index] = element;
    } else {
        const Element element = inputHandle[index];
        outputHandle.Store2(index * 8u, uint2(element.x, element.y));
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bindlessBufferByteAddressMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    ByteAddressBuffer inputHandle =
        resolveUniform(ResourceHandle<ByteAddressBuffer>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    const uint v0 = inputHandle.Load(0);
    const uint2 v1 = inputHandle.Load2(4);
    const uint4 v2 = inputHandle.Load4(12);
    outputHandle.Store(0, v0);
    outputHandle.Store2(4, v1);
    outputHandle.Store4(12, v2);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bindlessBufferRwByteAddressMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    RWByteAddressBuffer inputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.inputBuffer));
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    if (push.passIndex == 0) {
        inputHandle.Store(0, 0xDEADBEEFu);
        inputHandle.Store2(4, uint2(0x11223344u, 0xAABBCCDDu));
        inputHandle.Store4(12, uint4(1u, 2u, 3u, 4u));
    } else {
        const uint v0 = inputHandle.Load(0);
        const uint2 v1 = inputHandle.Load2(4);
        const uint4 v2 = inputHandle.Load4(12);
        outputHandle.Store(0, v0);
        outputHandle.Store2(4, v1);
        outputHandle.Store4(12, v2);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bindlessBufferAtomicsMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform BindlessBufferPush push)
{
    RWByteAddressBuffer outputHandle =
        resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));

    const uint threadId = dispatchThreadId.x + dispatchThreadId.y * 8u;
    uint original = 0;
    outputHandle.InterlockedAdd(0, 1u, original);
    outputHandle.InterlockedMin(4, threadId, original);
    outputHandle.InterlockedMax(8, threadId, original);
    outputHandle.InterlockedCompareExchange(12, 0xDEADBEEFu, 0xCAFEBABEu, original);
}
