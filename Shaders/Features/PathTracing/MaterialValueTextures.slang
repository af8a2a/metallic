// Included after the production texture adapters and the generated Value IR
// declarations. Explicit footprints work in compute bins without quad derivatives.
#if METALLIC_CUSTOM_MATERIALS
float4 materialValueTexel(MaterialTextureInfo info, int2 texel, uint mip)
{
    return decodeMaterialTextureSample(gMaterialTextures(info.textureIndex).Load(int3(texel, mip)), info.transform0.w);
}

float4 materialValueBilinear(MaterialTextureInfo info, float2 uv, uint mip)
{
    uint2 dimensions; uint count;
    gMaterialTextures(info.textureIndex).GetDimensions(mip, dimensions.x, dimensions.y, count);
    dimensions = max(dimensions, uint2(1));
    float2 position = frac(uv) * float2(dimensions) - 0.5;
    int2 base = int2(floor(position));
    int2 lo = (base % int2(dimensions) + int2(dimensions)) % int2(dimensions);
    int2 hi = (lo + 1) % int2(dimensions);
    float2 f = frac(position);
    return lerp(lerp(materialValueTexel(info, lo, mip), materialValueTexel(info, int2(hi.x, lo.y), mip), f.x),
        lerp(materialValueTexel(info, int2(lo.x, hi.y), mip), materialValueTexel(info, hi, mip), f.x), f.y);
}

float4 sampleMaterialValueTexture(MaterialValueParameters material, uint slot, float2 uv,
    uint policy, float4 dxLod, float4 dy, MaterialValueTextureContext context, bool linearColor)
{
    MaterialTextureInfo info = material.baseColorTexture;
    if (slot == 1u) { info = material.metallicRoughnessTexture; }
    else if (slot == 2u) { info = material.normalTexture; }
    else if (slot == 3u) { info = material.occlusionTexture; }
    else if (slot == 4u) { info = material.emissiveTexture; }
    else if (slot == 5u) { info = material.specularTexture; }
    uint2 dimensions; uint mipCount;
    // The Graph sampler has semantic defaults. Preserve legacy raw-IR fallback.
    float4 fallback = linearColor && slot == 2u ? float4(0.5, 0.5, 1.0, 1.0) : float4(1.0);
    if (info.textureIndex >= context.textureCount || info.textureIndex == 0xffffffffu) { return fallback; }
    gMaterialTextures(info.textureIndex).GetDimensions(0u, dimensions.x, dimensions.y, mipCount);
    float lod = dxLod.x;
    if (policy == 1u) {
        float2 dx = float2(dot(info.transform0.xy, dxLod.xy), dot(info.transform1.xy, dxLod.xy)) * float2(dimensions);
        float2 ddy = float2(dot(info.transform0.xy, dy.xy), dot(info.transform1.xy, dy.xy)) * float2(dimensions);
        lod = 0.5 * log2(max(max(dot(dx, dx), dot(ddy, ddy)), 1e-20));
    } else if (policy == 2u) {
        lod = context.normalizedRayLod + 0.5 * log2(float(max(dimensions.x, 1u)) * float(max(dimensions.y, 1u))) + textureTransformLodBias(info) + dxLod.x;
    }
    lod = max(lod, materialTextureSamplingFloor(info.textureIndex));
    lod = clamp(lod, 0.0, float(max(mipCount, 1u) - 1u));
    const uint lo = uint(floor(lod)), hi = min(lo + 1u, max(mipCount, 1u) - 1u);
    float2 transformed = transformMaterialUv(info, uv);
    float4 sample = lerp(materialValueBilinear(info, transformed, lo),
        materialValueBilinear(info, transformed, hi), frac(lod));
    // IR v1 arithmetic retains its original untyped Rec.709 sample semantics.
    // Explicitly tagged wide-gamut linear colors enter that authoring basis.
    if ((uint(info.transform0.w) & 4u) == 0u && (linearColor || ((uint(info.transform0.w) >> 3u) & 3u) != 0u)) {
        sample.rgb = Metallic.WorkingColor::toLinearRec709(materialTextureColor(sample.rgb, info.transform0.w));
    }
    return sample;
}
#endif
