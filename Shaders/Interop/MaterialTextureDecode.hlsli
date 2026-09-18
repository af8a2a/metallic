#ifndef METALLIC_MATERIAL_TEXTURE_DECODE
#define METALLIC_MATERIAL_TEXTURE_DECODE
// TextureInfo.transform0.w is a numeric bit mask, shared with the resource uploader.
// Bit 0: sampled RGB is linear (hardware sRGB decode or linear KTX2).
// Bit 1: BC5 normal XY.
float3 materialTextureColor(float3 color, float flags)
{
    if ((uint(flags) & 1u) != 0u) { return color; }
    return float3(color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}
float4 decodeMaterialTextureSample(float4 sample, float flags)
{
    if ((uint(flags) & 2u) != 0u) {
        float2 xy = sample.xy * 2.0 - 1.0;
        sample.z = sqrt(saturate(1.0 - dot(xy,xy))) * 0.5 + 0.5;
    }
    return sample;
}
#endif
