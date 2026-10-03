#language slang 2026
implementing Core;

namespace Metallic
{
// Shared with ComputeProgram. Only acceleration structures retain a 64-bit
// address; images, buffers, resource arrays and constants all use DR.
public struct ComputeResourceSlot
{
    public uint2 handle;
    public BufferSpan<uint> payload;
    public uint dataStride;
};
public struct ComputeResourceParameters
{
    public BufferSpan<ComputeResourceSlot> resources;
    public BufferSpan<uint> constants;
};

public ComputeResourceParameters getComputeResources()
{
    return getParameters<ComputeResourceParameters>();
}

public T getResource<T : IOpaqueDescriptor>(uint slot)
{
    uint2 handle = getComputeResources().resources.load(slot).handle;
    if (T.kind == DescriptorKind.AccelerationStructure) {
        return resolveDescriptor(DescriptorHandle<T>(handle));
    }
    return resolveUniform(ResourceHandle<T>(handle.x));
}

public T getResourceArray<T : IOpaqueDescriptor>(uint slot, uint index)
{
    uint handle = getComputeResources().resources.load(slot).payload.load(index);
    return resolveNonUniform(ResourceHandle<T>(handle));
}

public RWBufferSpan<T> getData<T>(uint slot)
{
    ComputeResourceSlot entry = getComputeResources().resources.load(slot);
    BufferSpan<uint> span = entry.payload;
    return {ResourceHandle<RWByteAddressBuffer>(span.resource.index), span.byteOffset,
        entry.dataStride == sizeof(T) ? span.count : 0u};
}

public T getConstants<T>()
{
    BufferSpan<uint> span = getComputeResources().constants;
    return resolveUniform(span.resource).Load<T>(span.byteOffset);
}
} // namespace Metallic
