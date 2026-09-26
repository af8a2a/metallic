#language slang 2026
implementing Core;

namespace Metallic
{
// Matches ShaderDataSpan. BDA accesses have no descriptor bounds checking;
// consumers must check length() before indexing data, including derived indices.
public struct DataSpan<T>
{
    public T* data;
    public uint count;
    public uint stride;

    public uint length()
    {
        return stride == sizeof(T) ? count : 0u;
    }
    public bool contains(uint index)
    {
        return index < length();
    }
};
} // namespace Metallic
