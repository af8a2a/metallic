#language slang 2026
module MaterialValueTextureSampling;

import Material;
import MaterialTextureDecode;
import WorkingColor;
import ScenePathMath;
import StreamTextureSampling;

namespace Metallic
{
using Metallic.Material;

public float2 transformMaterialUv(MaterialTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

public float textureTransformLodBias(MaterialTextureInfo textureInfo)
{
    float determinant =
        textureInfo.transform0.x * textureInfo.transform1.y -
        textureInfo.transform0.y * textureInfo.transform1.x;
    return 0.5 * Metallic.ScenePath.safeLog2(abs(determinant));
}

internal float4 materialValueTexel(Texture2D<float4> texture, MaterialTextureInfo info, int2 texel, uint mip)
{
    return decodeMaterialTextureSample(texture.Load(int3(texel, mip)), info.transform0.w);
}

internal float4 materialValueBilinear(Texture2D<float4> texture, MaterialTextureInfo info, float2 uv, uint mip)
{
    uint2 dimensions; uint count;
    texture.GetDimensions(mip, dimensions.x, dimensions.y, count);
    dimensions = max(dimensions, uint2(1));
    float2 position = frac(uv) * float2(dimensions) - 0.5;
    int2 base = int2(floor(position));
    int2 lo = (base % int2(dimensions) + int2(dimensions)) % int2(dimensions);
    int2 hi = (lo + 1) % int2(dimensions);
    float2 f = frac(position);
    return lerp(lerp(materialValueTexel(texture, info, lo, mip), materialValueTexel(texture, info, int2(hi.x, lo.y), mip), f.x),
        lerp(materialValueTexel(texture, info, int2(lo.x, hi.y), mip), materialValueTexel(texture, info, hi, mip), f.x), f.y);
}

public float4 sampleMaterialValueTextureResource(Texture2D<float4> texture,
    StructuredBuffer<TextureSamplingMetadata> metadata, MaterialTextureInfo info, float2 uv,
    uint policy, float4 dxLod, float4 dy, float normalizedRayLod, bool linearColor)
{
    uint2 dimensions; uint mipCount;
    texture.GetDimensions(0u, dimensions.x, dimensions.y, mipCount);
    float lod = dxLod.x;
    if (policy == 1u) {
        float2 dx = float2(dot(info.transform0.xy, dxLod.xy), dot(info.transform1.xy, dxLod.xy)) * float2(dimensions);
        float2 ddy = float2(dot(info.transform0.xy, dy.xy), dot(info.transform1.xy, dy.xy)) * float2(dimensions);
        lod = 0.5 * log2(max(max(dot(dx, dx), dot(ddy, ddy)), 1e-20));
    } else if (policy == 2u) {
        lod = normalizedRayLod + 0.5 * log2(float(max(dimensions.x, 1u)) * float(max(dimensions.y, 1u))) + textureTransformLodBias(info) + dxLod.x;
    }
    lod = max(lod, streamTextureSamplingFloor(metadata, info.textureIndex));
    lod = clamp(lod, 0.0, float(max(mipCount, 1u) - 1u));
    const uint lo = uint(floor(lod)), hi = min(lo + 1u, max(mipCount, 1u) - 1u);
    float2 transformed = transformMaterialUv(info, uv);
    float4 sample = lerp(materialValueBilinear(texture, info, transformed, lo),
        materialValueBilinear(texture, info, transformed, hi), frac(lod));
    // IR v1 arithmetic retains its original untyped Rec.709 sample semantics.
    // Explicitly tagged wide-gamut linear colors enter that authoring basis.
    if ((uint(info.transform0.w) & 4u) == 0u && (linearColor || ((uint(info.transform0.w) >> 3u) & 3u) != 0u)) {
        sample.rgb = Metallic.WorkingColor::toLinearRec709(materialTextureColor(sample.rgb, info.transform0.w));
    }
    return sample;
}
} // namespace Metallic
