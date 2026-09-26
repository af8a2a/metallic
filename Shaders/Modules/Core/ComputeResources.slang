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
public struct ComputeResourcePush
{
    public ComputeResourceSlot* resources;
    public uint* constants;
};

// Canonical owner of the common compute ABI; importing Core through multiple
// modules still declares this push constant exactly once. Pass-specific bindings
// remain in the program, expressed through the typed accessors below.
[[vk::push_constant]] public ComputeResourcePush gComputeResources;

public T getResource<T : IOpaqueDescriptor>(uint slot)
{
    return resolveDescriptor(DescriptorHandle<T>(gComputeResources.resources[slot].handle));
}

public T getResourceArray<T : IOpaqueDescriptor>(uint slot, uint index)
{
    uint2 payload = gComputeResources.resources[slot].payload;
    uint2* handles = (uint2*)(uint64_t(payload.x) | (uint64_t(payload.y) << 32));
    return resolveDescriptor(nonuniform(DescriptorHandle<T>(handles[index])));
}

public DataSpan<T> getData<T>(uint slot)
{
    ComputeResourceSlot resource = gComputeResources.resources[slot];
    return {(T*)(uint64_t(resource.handle.x) | (uint64_t(resource.handle.y) << 32)),
        resource.payload.x, resource.payload.y};
}

public T getConstants<T>()
{
    return *((T*)gComputeResources.constants);
}
} // namespace Metallic
