#language slang 2026
implementing Core;

namespace Metallic
{
// Shared with ComputeProgram. Arrays contain canonical handles; descriptor slots
// do not have to be contiguous in the device registry.
public struct ComputeResourceSlot
{
    public uint2 handle;
    // Descriptor-array address, or ordinary-data element count and stride.
    public uint2 payload;
};
public struct ComputeResourceParameters
{
    public ComputeResourceSlot* resources;
    public uint* constants;
};

// Resource tables are one payload of the shared ParameterRoot ABI.
public ComputeResourceParameters getComputeResources()
{
    return getParameters<ComputeResourceParameters>();
}

public T getResource<T : IOpaqueDescriptor>(uint slot)
{
    return resolveDescriptor(DescriptorHandle<T>(getComputeResources().resources[slot].handle));
}

public T getResourceArray<T : IOpaqueDescriptor>(uint slot, uint index)
{
    uint2 payload = getComputeResources().resources[slot].payload;
    uint2* handles = (uint2*)(uint64_t(payload.x) | (uint64_t(payload.y) << 32));
    return resolveDescriptor(nonuniform(DescriptorHandle<T>(handles[index])));
}

public DataSpan<T> getData<T>(uint slot)
{
    ComputeResourceSlot resource = getComputeResources().resources[slot];
    return {(T*)(uint64_t(resource.handle.x) | (uint64_t(resource.handle.y) << 32)),
        resource.payload.x, resource.payload.y};
}

public T getConstants<T>()
{
    return *((T*)getComputeResources().constants);
}
} // namespace Metallic
