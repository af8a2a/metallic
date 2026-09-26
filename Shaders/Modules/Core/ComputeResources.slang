#language slang 2026
implementing Core;

namespace Metallic
{
// Shared with ComputeProgram. Arrays contain canonical handles; descriptor slots
// do not have to be contiguous in the device registry.
public struct ComputeResourceSlot
{
    public uint2 handle;
    public uint2* array;
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
    return resolveDescriptor(nonuniform(DescriptorHandle<T>(gComputeResources.resources[slot].array[index])));
}

public T getConstants<T>()
{
    return *((T*)gComputeResources.constants);
}
} // namespace Metallic
