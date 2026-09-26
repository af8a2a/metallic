#language slang 2026
module ParameterRoot;

namespace Metallic
{
// One immutable typed packet per dispatch; no resource-slot table or heap header.
[[vk::push_constant]] public uint* gParameterRoot;

public T getParameters<T>()
{
    return *((T*)gParameterRoot);
}
} // namespace Metallic
