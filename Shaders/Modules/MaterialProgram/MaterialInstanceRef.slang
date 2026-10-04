#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
// Byte offsets relative to the parameter/resource tables bound for one program.
// Not descriptors, material IDs, shader variants or absolute GPU addresses.
// The program's schema defines record alignment, size and table bounds.
public struct MaterialInstanceRef
{
    public uint parameterOffset;
    public uint resourceOffset;
};
} // namespace Metallic.Material
