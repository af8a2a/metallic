#language slang 2026
implementing ShaderCore;

namespace Metallic
{
// Stable wire ABI: Slang's DescriptorHandle representation stays inside resolvers.
public struct ResourceHandle<T : IOpaqueDescriptor>
{
    public uint index;
};

public struct SamplerHandle
{
    public uint index;
};

[ForceInline]
public T resolveUniform<T : IOpaqueDescriptor>(ResourceHandle<T> handle)
{
    return defaultGetDescriptorFromHandle(DescriptorHandle<T>(uint2(handle.index, 0)));
}

[ForceInline]
public T resolveNonUniform<T : IOpaqueDescriptor>(ResourceHandle<T> handle)
{
    return defaultGetDescriptorFromHandle(nonuniform(DescriptorHandle<T>(uint2(handle.index, 0))));
}

[ForceInline]
public SamplerState resolveUniform(SamplerHandle handle)
{
    return SamplerDescriptorHeap[handle.index];
}

[ForceInline]
public SamplerState resolveNonUniform(SamplerHandle handle)
{
    return SamplerDescriptorHeap[NonUniformResourceIndex(handle.index)];
}

public struct BufferSpan<T>
{
    public ResourceHandle<ByteAddressBuffer> resource;
    public uint byteOffset;
    public uint count;

    public uint length() { return count; }
    public bool contains(uint index) { return index < count; }
    public T load(uint index)
    {
        if (!contains(index)) { T zero = {}; return zero; }
        return resolveUniform(resource).Load<T>(byteOffset + index * sizeof(T));
    }
    public T loadNonUniform(uint index)
    {
        if (!contains(index)) { T zero = {}; return zero; }
        return resolveNonUniform(resource).Load<T>(byteOffset + index * sizeof(T));
    }
};

public struct RWBufferSpan<T>
{
    public ResourceHandle<RWByteAddressBuffer> resource;
    public uint byteOffset;
    public uint count;

    public uint length() { return count; }
    public bool contains(uint index) { return index < count; }
    public T load(uint index)
    {
        if (!contains(index)) { T zero = {}; return zero; }
        return resolveUniform(resource).Load<T>(byteOffset + index * sizeof(T));
    }
    public void store(uint index, T value)
    {
        if (contains(index)) { resolveUniform(resource).Store<T>(byteOffset + index * sizeof(T), value); }
    }
    public T loadNonUniform(uint index)
    {
        if (!contains(index)) { T zero = {}; return zero; }
        return resolveNonUniform(resource).Load<T>(byteOffset + index * sizeof(T));
    }
    public void storeNonUniform(uint index, T value)
    {
        if (contains(index)) { resolveNonUniform(resource).Store<T>(byteOffset + index * sizeof(T), value); }
    }
};

// Explicit pointer semantics. Ordinary resource-relative access uses BufferSpan.
public struct PhysicalPtr<T>
{
    public T* address;
};
} // namespace Metallic
