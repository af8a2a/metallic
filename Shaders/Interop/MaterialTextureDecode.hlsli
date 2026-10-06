#ifndef METALLIC_MATERIAL_TEXTURE_DECODE
#define METALLIC_MATERIAL_TEXTURE_DECODE
import WorkingColor;
import ColorSpace;
// Numeric mask in TextureInfo.transform0.w: bit 0 = decoded linear RGB,
// bit 1 = BC5 normal XY, bit 2 = Data, bits 3-4 = source primaries.
float3 materialTextureLinearSource(float3 color, float flags)
{
    if ((uint(flags) & 1u) != 0u || (uint(flags) & 4u) != 0u) { return color; }
    return float3(color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}
float3 materialTextureSourceToWorking(float3 color, float flags)
{
    if ((uint(flags) & 4u) != 0u) { return color; }
    uint primaries = (uint(flags) >> 3u) & 3u;
    if (primaries == 1u) { return Metallic.WorkingColor::fromACEScg(color); }
    if (primaries == 2u) {
        return Metallic.WorkingColor::fromACEScg(mul(Metallic.ColorSpace::kXYZToAP1, mul(Metallic.ColorSpace::kAP0ToXYZ, color)));
    }
    if (primaries == 3u) {
        return Metallic.WorkingColor::fromACEScg(mul(Metallic.ColorSpace::kXYZToAP1,
            mul(Metallic.ColorSpace::kD65ToD60, mul(Metallic.ColorSpace::kRec2020ToXYZ, color))));
    }
    return Metallic.WorkingColor::fromLinearRec709(color);
}
float3 materialTextureColor(float3 color, float flags)
{
    return materialTextureSourceToWorking(materialTextureLinearSource(color, flags), flags);
}
// Preserve factor*texture modulation in the declared source basis.
float3 materialTextureWorkingToSource(float3 working, float flags)
{
    uint primaries = (uint(flags) >> 3u) & 3u;
    if (primaries == 0u) { return Metallic.WorkingColor::toLinearRec709(working); }
    float3 ap1 = Metallic.WorkingColor::toACEScg(working);
    if (primaries == 1u) { return ap1; }
    float3 xyz = mul(Metallic.ColorSpace::kAP1ToXYZ, ap1);
    if (primaries == 2u) { return mul(Metallic.ColorSpace::kXYZToAP0, xyz); }
    return mul(Metallic.ColorSpace::kXYZToRec2020, mul(Metallic.ColorSpace::kD60ToD65, xyz));
}
float3 materialTextureModulate(float3 workingFactor, float3 sample, float flags)
{
    float3 linear = materialTextureLinearSource(sample, flags);
    if ((uint(flags) & 4u) != 0u) { return workingFactor * linear; }
    return materialTextureSourceToWorking(materialTextureWorkingToSource(workingFactor, flags) * linear, flags);
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
