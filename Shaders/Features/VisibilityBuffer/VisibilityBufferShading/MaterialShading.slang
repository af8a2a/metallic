implementing VisibilityBufferShading;

import GPUDriven;
using Metallic.GPUDriven;
import Core;
import MaterialTextureDecode;
using Metallic;
import Material;
using Metallic.Material;

import OpenPBR;
import OpenPBRClosure;
import OpenPBRTextureLUT;
import MaterialProgram;
using Metallic.OpenPBR;

#ifndef GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE
#define GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE 0
#endif
#ifndef GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE
#define GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE 0
#endif

// Legacy preview callers must bind a linear clamp sampler at this handle.
#ifndef GPU_DRIVEN_OPENPBR_LUT_SAMPLER
#define GPU_DRIVEN_OPENPBR_LUT_SAMPLER 0
#endif

float4 sampleOpenPBRLut2D(int lutId, float2 uv)
{
    return sampleOpenPBRTexture2D(
        resolveUniform(ResourceHandle<Texture2D<float4>>(GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE + openPBRLut2DSlot(lutId))),
        resolveUniform(SamplerHandle(GPU_DRIVEN_OPENPBR_LUT_SAMPLER)), uv);
}

float4 sampleOpenPBRLut3D(int lutId, float3 uvw)
{
    return sampleOpenPBRTexture3D(
        resolveUniform(ResourceHandle<Texture3D<float4>>(GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE + openPBRLut3DSlot(lutId))),
        resolveUniform(SamplerHandle(GPU_DRIVEN_OPENPBR_LUT_SAMPLER)), uvw);
}

public struct PreviewOpenPBRContext : IOpenPBRContext
{
    public static const bool kEnableSheenAndCoat = false;
    public static const bool kEnableDispersion = false;
    public static const bool kEnableThinFilm = false;
    public static const bool kEnableSampling = true;
    static bool enableSheenAndCoat() { return false; }
    static bool enableDispersion() { return false; }
    static bool enableTranslucency() { return true; }
    static bool enableMetallic() { return true; }
    float4 sample2D(int id, float2 uv) { return sampleOpenPBRLut2D(id, uv); }
    float4 sample3D(int id, float3 uvw) { return sampleOpenPBRLut3D(id, uvw); }
};
typealias SceneOpenPBRClosure = OpenPBRClosure<PreviewOpenPBRContext>;
float3 srgbToLinear(float3 color)
{
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, float3(2.4));
    return float3(
        color.x <= 0.04045 ? low.x : high.x,
        color.y <= 0.04045 ? low.y : high.y,
        color.z <= 0.04045 ? low.z : high.z);
}

float3 linearToSrgb(float3 color)
{
    color = max(color, float3(0.0));
    float3 low = color * 12.92;
    float3 high = 1.055 * pow(color, float3(1.0 / 2.4)) - 0.055;
    return float3(
        color.x <= 0.0031308 ? low.x : high.x,
        color.y <= 0.0031308 ? low.y : high.y,
        color.z <= 0.0031308 ? low.z : high.z);
}

float2 transformMaterialUv(GPUDrivenPreviewTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float4 sampleImageBilinear(Texture2D<float4> texture, float2 uv, bool wrapY)
{
    uint width = 1u;
    uint height = 1u;
    texture.GetDimensions(width, height);
    float2 addressedUv = float2(frac(uv.x), wrapY ? frac(uv.y) : saturate(uv.y));
    float2 texel = addressedUv * float2(width, height) - 0.5;
    int2 baseTexel = int2(floor(texel));
    float2 fraction = frac(texel);
    int x0 = (baseTexel.x % int(width) + int(width)) % int(width);
    int x1 = (x0 + 1) % int(width);
    int y0 = wrapY
        ? (baseTexel.y % int(height) + int(height)) % int(height)
        : clamp(baseTexel.y, 0, int(height) - 1);
    int y1 = wrapY ? (y0 + 1) % int(height) : min(y0 + 1, int(height) - 1);
    float4 v00 = texture.Load(int3(x0, y0, 0));
    float4 v10 = texture.Load(int3(x1, y0, 0));
    float4 v01 = texture.Load(int3(x0, y1, 0));
    float4 v11 = texture.Load(int3(x1, y1, 0));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float4 sampleMaterialTexture(
    GPUDrivenPreviewTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    if (textureInfo.textureIndex == kInvalidTextureIndex ||
        textureInfo.textureIndex >= params.materialTextureCount) {
        return fallback;
    }
    StructuredBuffer<uint> remapHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.materialTextureRemapBuffer));
    StructuredBuffer<uint> remap = remapHandle;
    Texture2D<float4> textureHandle =
        resolveNonUniform(ResourceHandle<Texture2D<float4>>(remap[textureInfo.textureIndex]));
    Texture2D<float4> texture = textureHandle;
    return sampleImageBilinear(texture, transformMaterialUv(textureInfo, uv), true);
}

float2 environmentUv(float3 direction, float rotationRadians)
{
    float phi = atan2(direction.z, direction.x) + rotationRadians;
    return float2(frac(phi / (2.0 * kPi) + 0.5), acos(clamp(direction.y, -1.0, 1.0)) / kPi);
}

float3 proceduralSky(float3 direction, float rotationRadians)
{
    float sine = sin(rotationRadians);
    float cosine = cos(rotationRadians);
    float3 sunDirection = safeNormalize(
        float3(0.35 * cosine + 0.22 * sine, 0.88, -0.35 * sine + 0.22 * cosine),
        float3(0.0, 1.0, 0.0));
    float t = saturate(direction.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.62, 0.70, 0.78), float3(0.12, 0.24, 0.48), t) * 0.45;
    return Metallic.WorkingColor::fromLinearRec709(sky + pow(saturate(dot(direction, sunDirection)), 256.0) * float3(7.0, 6.4, 5.3));
}

float3 environmentRadiance(float3 direction, GPUDrivenPreviewParams params, GPUDrivenPreviewPush push)
{
    if (params.environmentMode == kEnvironmentModeDisabled || params.environmentIntensity <= 0.0) {
        return float3(0.0);
    }
    if (params.environmentMode == kEnvironmentModeProcedural) {
        return proceduralSky(direction, params.environmentRotationRadians) * params.environmentIntensity;
    }
    Texture2D<float4> environmentHandle =
        resolveUniform(ResourceHandle<Texture2D<float4>>(push.environmentImage));
    Texture2D<float4> environmentMap = environmentHandle;
    return max(
        sampleImageBilinear(
            environmentMap,
            environmentUv(safeNormalize(direction, float3(0.0, 1.0, 0.0)), params.environmentRotationRadians),
            false).rgb,
        float3(0.0)) * params.environmentIntensity;
}

float3 environmentSHIrradiance(
    float3 direction,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    float sine = sin(params.environmentRotationRadians);
    float cosine = cos(params.environmentRotationRadians);
    float3 rotatedDirection = float3(
        direction.x * cosine - direction.z * sine,
        direction.y,
        direction.x * sine + direction.z * cosine);
    StructuredBuffer<float4> coefficientsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<float4>>(push.environmentSHBuffer));
    StructuredBuffer<float4> coefficients = coefficientsHandle;
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(coefficients);
    return max(environment.evaluate(rotatedDirection), float3(0.0)) * params.environmentIntensity;
}

float3 hashColor(uint seed)
{
    float3 value = frac(sin(float3(seed, seed * 17u + 13u, seed * 31u + 7u) * 0.017453292) * 43758.5453);
    return lerp(float3(0.18, 0.34, 0.62), float3(0.95, 0.82, 0.35), value);
}

float3 acesToneMap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

uint packColor(float3 color)
{
    uint3 rgb = uint3(saturate(color) * 255.0 + 0.5);
    return rgb.x | (rgb.y << 8u) | (rgb.z << 16u) | 0xff000000u;
}

uint packHdrColor(float3 color)
{
    return packColor(linearToSrgb(acesToneMap(max(Metallic.WorkingColor::toLinearRec709(color), float3(0.0)))));
}

uint shadeDeferredSurface(
    GPUDrivenDeferredSurface surface,
    float2 pixelCenter,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    float3 geometryNormal = surface.geometryNormal;
    float3 normal = surface.normal;
    float3 tangent = safeNormalize(
        surface.tangent - normal * dot(normal, surface.tangent),
        abs(normal.z) < 0.999
            ? safeNormalize(
                cross(float3(0.0, 0.0, 1.0), normal),
                float3(1.0, 0.0, 0.0))
            : safeNormalize(
                cross(float3(0.0, 1.0, 0.0), normal),
                float3(1.0, 0.0, 0.0)));
    float tangentSign = surface.tangentSign < 0.0 ? -1.0 : 1.0;
    float3 bitangent = safeNormalize(
        cross(normal, tangent) * tangentSign,
        float3(0.0, 1.0, 0.0));
    float3 viewDirection = safeNormalize(
        params.renderEye.xyz - surface.worldPosition,
        -cameraRayDirection(pixelCenter, params));

    uint materialIndex = min(surface.materialIndex, max(params.materialCount, 1u) - 1u);
    StructuredBuffer<GPUDrivenPreviewMaterial> materialHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(push.materialBuffer));
    StructuredBuffer<GPUDrivenPreviewMaterial> materials = materialHandle;
    GPUDrivenPreviewMaterial material = materials[materialIndex];

    float4 normalSample = sampleMaterialTexture(
        material.normalTexture,
        surface.texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        params,
        push);
    float3 tangentNormal = normalSample.xyz * 2.0 - 1.0;
    tangentNormal.xy *= material.textureParams.x;
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    normal = safeNormalize(
        tangent * tangentNormal.x + bitangent * tangentNormal.y + normal * tangentNormal.z,
        normal);
    if (dot(normal, geometryNormal) < 0.0) {
        normal = -normal;
    }
    if (dot(normal, viewDirection) < 0.0) {
        normal = -normal;
    }
    tangent = safeNormalize(tangent - normal * dot(normal, tangent), tangent);
    bitangent = safeNormalize(cross(normal, tangent) * tangentSign, bitangent);

    float4 baseColorSample = sampleMaterialTexture(
        material.baseColorTexture,
        surface.texcoord,
        float4(1.0),
        params,
        push);
    float4 metallicRoughnessSample = sampleMaterialTexture(
        material.metallicRoughnessTexture,
        surface.texcoord,
        float4(1.0),
        params,
        push);
    float4 occlusionSample = sampleMaterialTexture(
        material.occlusionTexture,
        surface.texcoord,
        float4(1.0),
        params,
        push);
    float4 emissiveSample = sampleMaterialTexture(
        material.emissiveTexture,
        surface.texcoord,
        float4(1.0),
        params,
        push);
    float4 transmissionSample = sampleMaterialTexture(
        material.transmissionTexture,
        surface.texcoord,
        float4(1.0),
        params,
        push);

    float3 baseColor = saturate(
        materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
    if (params.mode == kModeBaseColor) {
        return packColor(linearToSrgb(Metallic.WorkingColor::toLinearRec709(baseColor)));
    }
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(
        material.params.y * metallicRoughnessSample.g,
        0.001,
        1.0);
    float transmission = saturate(
        material.glassParams.x * transmissionSample.r) * (1.0 - metallic);
    float3 emissive = max(
        materialTextureModulate(material.emissive.rgb, emissiveSample.rgb, material.emissiveTexture.transform0.w),
        float3(0.0));
    float occlusion = lerp(
        1.0,
        saturate(occlusionSample.r),
        saturate(material.textureParams.y));

    OpenPBRResolvedInputs inputs = openPBRMakeDefaultResolvedInputs();
    inputs.base_weight = 1.0;
    inputs.base_color = baseColor;
    // glTF specifies a Lambertian diffuse lobe, independently of GGX roughness.
    inputs.base_diffuse_roughness = 0.0;
    inputs.base_metalness = metallic;
    inputs.specular_weight = 1.0;
    inputs.specular_color = float3(1.0);
    inputs.specular_roughness = roughness;
    inputs.specular_roughness_anisotropy = 0.0;
    inputs.specular_ior = clamp(material.glassParams.y, 1.0, 3.0);
    inputs.specular_anisotropy_rotation_cos_sin = float2(1.0, 0.0);
    inputs.transmission_weight = transmission;
    inputs.transmission_color = sqrt(max(baseColor, float3(0.0)));
    inputs.transmission_depth = 0.0;
    inputs.subsurface_weight = 0.0;
    float emissionLuminance = max(emissive.x, max(emissive.y, emissive.z));
    inputs.emission_luminance = emissionLuminance;
    inputs.emission_color = emissionLuminance > 0.0
        ? emissive / emissionLuminance
        : float3(1.0);
    inputs.geometry_opacity = saturate(material.baseColor.a * baseColorSample.a);
    inputs.geometry_thin_walled = false;
    inputs.geometry_basis = openPBRMakeBasis(normal, tangent, bitangent);
    inputs.geometry_coat_basis = inputs.geometry_basis;

    SceneOpenPBRClosure closure;
    closure.context = PreviewOpenPBRContext();
    closure.inputs = inputs;
    closure.occlusion = occlusion;
    let prepared = closure.prepare(viewDirection, TransportMode.Radiance);
    float3 reflectedEnvironment = params.environmentMode == kEnvironmentModeDisabled ? float3(0.0)
        : environmentSHIrradiance(normal, params, push) * baseColor *
            (1.0 - metallic) * (1.0 - transmission) / kPi;
    float3 radiance = max(prepared.emission(), float3(0.0)) +
        reflectedEnvironment * occlusion;
    return packHdrColor(radiance);
}
