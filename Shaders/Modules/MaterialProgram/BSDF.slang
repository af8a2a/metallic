#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
public static const uint kBSDFReflection = 1u;
public static const uint kBSDFTransmission = 2u;
public static const uint kBSDFDiffuse = 4u;
public static const uint kBSDFGlossy = 8u;
public static const uint kBSDFDelta = 16u;

// f is a scene-linear BSDF value, WITHOUT cosine or division by pdf.
// pdf is the solid-angle density of the complete sampling strategy (including
// lobe selection). Delta samples use discrete probability and kBSDFDelta;
// arbitrary-direction eval of a delta event is zero. Zero pdf means invalid.
public struct BSDFEval
{
    public float3 f;
    public float pdf;
    public uint flags;
};

// Directions point away from the surface, in world space. eta is eta_i/eta_t
// for transmission, 1 for reflection. Transport-mode scaling belongs to the
// future prepared closure; these records do not silently apply it.
public struct BSDFSample
{
    public float3 wiWS;
    public float3 f;
    public float pdf;
    public float eta;
    public uint flags;
};
} // namespace Metallic.Material
