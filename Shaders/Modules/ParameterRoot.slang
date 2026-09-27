#language slang 2026
module ParameterRoot;

namespace Metallic
{
// One immutable parameter packet per dispatch; the RHI owns the heap header.
[[vk::push_constant]] public uint* gParameterRoot;

public T getParameters<T>()
{
    return *((T*)gParameterRoot);
}
} // namespace Metallic
