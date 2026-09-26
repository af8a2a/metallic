#language slang 2026
implementing Core;

namespace Metallic
{
// Shared with ComputeProgram. Slots contain absolute descriptor indices or RTAS addresses.
public struct ComputeResourcePush
{
    public uint2* resources;
    public uint* constants;
};

// Canonical owner of the common compute ABI; importing Core through multiple
// modules still declares this push constant exactly once. Pass-specific bindings
// remain in the program, expressed through the typed accessors below.
[[vk::push_constant]] public ComputeResourcePush gComputeResources;

public T getResource<T : IOpaqueDescriptor>(uint slot)
{
    return resolveDescriptor(DescriptorHandle<T>(gComputeResources.resources[slot]));
}

public T getResourceArray<T : IOpaqueDescriptor>(uint slot, uint index)
{
    return resolveDescriptor(nonuniform(DescriptorHandle<T>(uint2(gComputeResources.resources[slot].x + index, 0))));
}

public T getConstants<T>()
{
    return *((T*)gComputeResources.constants);
}
} // namespace Metallic
