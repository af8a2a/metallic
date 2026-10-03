#language slang 2026
implementing Core;

namespace Metallic
{
// Shared with ComputeProgram. Only acceleration structures retain a 64-bit
// address; images, buffers, resource arrays and constants all use DR.
public struct ComputeResourceParameters
{
    public BufferSpan<uint> resources;
    public BufferSpan<uint> constants;
};

public ComputeResourceParameters getComputeResources()
{
    return getParameters<ComputeResourceParameters>();
}

public T getConstants<T>()
{
    BufferSpan<uint> span = getComputeResources().constants;
    return resolveUniform(span.resource).Load<T>(span.byteOffset);
}
} // namespace Metallic
