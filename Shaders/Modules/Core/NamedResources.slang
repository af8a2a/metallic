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

// Load only the requested field/view, while retaining the full packet validity
// contract of getResourceParameters. Read just the resource span from the root;
// constants are not needed here. Offsets are bytes in the scalar wire layout.
public TField getResourceParameterField<TPacket, TField>(uint byteOffset)
{
    TField zero = {};
    if (gParameterRoot.count < (sizeof(ComputeResourceParameters) + 3u) / 4u) { return zero; }
    let span = resolveUniform(gParameterRoot.resource).Load<BufferSpan<uint>>(gParameterRoot.byteOffset);
    if (span.count < (sizeof(TPacket) + 3u) / 4u ||
        sizeof(TField) > sizeof(TPacket) || byteOffset > sizeof(TPacket) - sizeof(TField) ||
        (byteOffset & 3u) != 0u) { return zero; }
    return resolveUniform(span.resource).Load<TField>(span.byteOffset + byteOffset);
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
