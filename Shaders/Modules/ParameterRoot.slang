#language slang 2026
module ParameterRoot;

import ShaderCore;

#ifdef METALLIC_INLINE_COMPUTE_ROOT
#if METALLIC_INLINE_COMPUTE_ROOT
#error ParameterRoot cannot be imported with METALLIC_INLINE_COMPUTE_ROOT=1
#endif
#endif

namespace Metallic
{
// The immutable parameter packet is an ordinary DR buffer. The push ABI is
// {resource index, byte offset, word count}, with no physical pointer in shaders.
[[vk::push_constant]] public ConstantBuffer<BufferSpan<uint>> gParameterRoot;

public T getParameters<T>()
{
    if (gParameterRoot.count < (sizeof(T) + 3u) / 4u) { T zero = {}; return zero; }
    return resolveUniform(gParameterRoot.resource).Load<T>(gParameterRoot.byteOffset);
}
} // namespace Metallic
