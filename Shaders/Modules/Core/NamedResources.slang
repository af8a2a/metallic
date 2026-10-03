#language slang 2026
implementing Core;

namespace Metallic
{
#include "../../../Source/Runtime/Render/Core/NamedResourceParameters.h"

public T getResourceParameters<T>()
{
    let span = getComputeResources().resources;
    if (span.count < (sizeof(T) + 3u) / 4u) { T zero = {}; return zero; }
    return resolveUniform(span.resource).Load<T>(span.byteOffset);
}

// Buffer data layout is selected at the use site, while identity stays a 32-bit
// engine handle. No logical slot lookup or descriptor representation escapes Core.
public T resolveBuffer<T : IOpaqueDescriptor>(ResourceHandle<ByteAddressBuffer> handle)
{
    return resolveUniform(ResourceHandle<T>(handle.index));
}

public RaytracingAccelerationStructure resolveAccelerationStructure(uint64_t address)
{
    return resolveDescriptor(DescriptorHandle<RaytracingAccelerationStructure>(
        uint2(uint(address), uint(address >> 32))));
}

public RWBufferSpan<T> typedBufferSpan<T>(BufferSpan<uint> words)
{
    uint stride = sizeof(T) / 4u;
    return {ResourceHandle<RWByteAddressBuffer>(words.resource.index), words.byteOffset,
        stride != 0u && words.count % stride == 0u ? words.count / stride : 0u};
}
} // namespace Metallic
