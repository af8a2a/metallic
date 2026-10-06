#language slang 2026
implementing OpenPBR;

namespace Metallic.OpenPBR
{
// Statically specialized by the caller. The module owns no descriptors, push
// constants, material textures, or renderer state. Coordinates follow Adobe's
// normalized, clamped LUT contract; providers must preserve table orientation.
public interface IOpenPBRContext
{
    public static bool enableSheenAndCoat();
    public static bool enableDispersion();
    public static bool enableTranslucency();
    public static bool enableMetallic();
    public float4 sample2D(int id, float2 uv);
    public float4 sample3D(int id, float3 uvw);
};
} // namespace Metallic.OpenPBR
