#language slang 2026
implementing Core;

#ifndef METALLIC_INLINE_COMPUTE_ROOT
#define METALLIC_INLINE_COMPUTE_ROOT 0
#endif

namespace Metallic
{
public ComputeResourceParameters getComputeResources()
{
#if METALLIC_INLINE_COMPUTE_ROOT
    return gInlineComputeRoot;
#else
    return getParameters<ComputeResourceParameters>();
#endif
}

public T getConstants<T>()
{
    BufferSpan<uint> span = getComputeResources().constants;
    return resolveUniform(span.resource).Load<T>(span.byteOffset);
}
} // namespace Metallic
