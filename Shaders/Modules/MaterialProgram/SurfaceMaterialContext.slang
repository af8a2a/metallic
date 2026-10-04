#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
public static const uint kTextureFootprintGradients0 = 1u;
public static const uint kTextureFootprintGradients1 = 2u;
public static const uint kTextureFootprintIsotropic0 = 4u;
public static const uint kTextureFootprintIsotropic1 = 8u;

// Texture-independent normalized UV footprint, before a texture's UV transform.
// Gradients are per render pixel. Isotropic LOD is log2 of the normalized UV
// footprint, before adding 0.5 * log2(width * height). It can be negative.
// A producer may project a ray cone into this scalar; no view/light enters the
// context. Flags explicitly distinguish unavailable data from a zero footprint.
public struct TextureFootprint
{
    public float2 uv0Dx;
    public float2 uv0Dy;
    public float2 uv1Dx;
    public float2 uv1Dy;
    public float isotropicLod0;
    public float isotropicLod1;
    public uint flags;
};

public struct SurfaceMaterialContext
{
    public float3 positionWS;
    public float3 geometryNormalWS;
    // Authored/interpolated normal, before normal mapping or BSDF face-forward.
    public float3 shadingNormalWS;
    public float3 tangentWS;
    public float3 bitangentWS;
    public float2 texCoord0;
    public float2 texCoord1;
    public TextureFootprint footprint;
    public uint texCoordMask; // bit 0/1: UV0/UV1 available; absent coordinates are zero
    public uint frontFace; // 0 or 1; explicit uint gives a stable storage representation
};
} // namespace Metallic.Material
