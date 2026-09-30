implementing Material;

namespace Metallic.Material
{

// The newly resident tail contains the previous tail's mips. Hold sampling at
// that old detail level on publication, then release the floor independently
// of the source-resolution demand used by the streaming scheduler.
public float4 sampleResidentMaterialTexture(
    Texture2D<float4> texture,
    SamplerState sampler,
    float2 uv,
    float lod,
    float minimumLod)
{
    return texture.SampleLevel(sampler, uv, max(lod, minimumLod));
}

} // namespace Metallic.Material
