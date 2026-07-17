#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0
#define OPENPBR_FEATURE_EnableSheenAndCoat false
#define OPENPBR_FEATURE_EnableDispersion false
#define OPENPBR_FEATURE_EnableTranslucency true
#define OPENPBR_FEATURE_EnableMetallic true
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) OPENPBR_FEATURE_##name

[[vk::binding(11, 0)]]
Texture2D<float4> gOpenPBRLut2D[6];

[[vk::binding(12, 0)]]
Texture3D<float4> gOpenPBRLut3D[2];

uint openPBRLut2DSlot(int lutId)
{
    if (lutId == 1) {
        return 0u;
    }
    if (lutId == 2) {
        return 1u;
    }
    if (lutId == 4) {
        return 2u;
    }
    if (lutId == 5) {
        return 3u;
    }
    if (lutId == 6) {
        return 4u;
    }
    return 5u;
}

uint openPBRLut3DSlot(int lutId)
{
    return lutId == 3 ? 1u : 0u;
}

float4 sampleOpenPBRLut2D(int lutId, float2 uv)
{
    Texture2D<float4> texture = gOpenPBRLut2D[openPBRLut2DSlot(lutId)];
    uint width;
    uint height;
    texture.GetDimensions(width, height);
    float2 size = float2(max(width, 1u), max(height, 1u));
    float2 texel = clamp(uv * size - 0.5, float2(0.0, 0.0), size - 1.0);
    int2 baseTexel = int2(floor(texel));
    int2 nextTexel = min(baseTexel + int2(1, 1), int2(width - 1u, height - 1u));
    float2 frac = texel - float2(baseTexel);
    float4 v00 = texture.Load(int3(baseTexel.x, baseTexel.y, 0));
    float4 v10 = texture.Load(int3(nextTexel.x, baseTexel.y, 0));
    float4 v01 = texture.Load(int3(baseTexel.x, nextTexel.y, 0));
    float4 v11 = texture.Load(int3(nextTexel.x, nextTexel.y, 0));
    return lerp(lerp(v00, v10, frac.x), lerp(v01, v11, frac.x), frac.y);
}

float4 sampleOpenPBRLut3D(int lutId, float3 uvw)
{
    Texture3D<float4> texture = gOpenPBRLut3D[openPBRLut3DSlot(lutId)];
    uint width;
    uint height;
    uint depth;
    texture.GetDimensions(width, height, depth);
    float3 size = float3(max(width, 1u), max(height, 1u), max(depth, 1u));
    float3 texel = clamp(uvw * size - 0.5, float3(0.0, 0.0, 0.0), size - 1.0);
    int3 baseTexel = int3(floor(texel));
    int3 nextTexel = min(baseTexel + int3(1, 1, 1), int3(width - 1u, height - 1u, depth - 1u));
    float3 frac = texel - float3(baseTexel);
    float4 v000 = texture.Load(int4(baseTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v100 = texture.Load(int4(nextTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v010 = texture.Load(int4(baseTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v110 = texture.Load(int4(nextTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v001 = texture.Load(int4(baseTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v101 = texture.Load(int4(nextTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v011 = texture.Load(int4(baseTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 v111 = texture.Load(int4(nextTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 z0 = lerp(lerp(v000, v100, frac.x), lerp(v010, v110, frac.x), frac.y);
    float4 z1 = lerp(lerp(v001, v101, frac.x), lerp(v011, v111, frac.x), frac.y);
    return lerp(z0, z1, frac.z);
}

#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) sampleOpenPBRLut2D(lut_id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) sampleOpenPBRLut3D(lut_id, uvw)

#include "../External/openpbr-bsdf/openpbr.h"
#include "ScenePathTrace.slang"

struct OpenPBRMaterialEval
{
    OpenPBR_ResolvedInputs inputs;
    float occlusion;
};

static const uint kOpenPBRInteriorStackCapacity = 8u;

struct OpenPBRInteriorStackEntry
{
    uint materialIndex;
    float ior;
};

struct OpenPBRInteriorStack
{
    OpenPBRInteriorStackEntry entries[kOpenPBRInteriorStackCapacity];
    uint count;
};

OpenPBRInteriorStack makeOpenPBRInteriorStack()
{
    OpenPBRInteriorStack stack;
    stack.count = 0u;
    return stack;
}

float openPbrMaterialIor(PathTraceMaterial material)
{
    return clamp(material.glassParams.y, 1.0, 3.0);
}

float openPbrInteriorStackTopIor(OpenPBRInteriorStack stack)
{
    return stack.count > 0u ? stack.entries[stack.count - 1u].ior : OpenPBR_VacuumIor;
}

float openPbrInteriorStackNextIor(OpenPBRInteriorStack stack)
{
    return stack.count > 1u ? stack.entries[stack.count - 2u].ior : OpenPBR_VacuumIor;
}

float openPbrExteriorIorForHit(OpenPBRInteriorStack stack, HitInfo hit)
{
    if (hit.frontFace) {
        return openPbrInteriorStackTopIor(stack);
    }
    if (stack.count > 0u && stack.entries[stack.count - 1u].materialIndex == hit.materialIndex) {
        return openPbrInteriorStackNextIor(stack);
    }
    return openPbrInteriorStackTopIor(stack);
}

float3 openPbrVolumeTransmittance(OpenPBRInteriorStack stack, float distance)
{
    if (stack.count == 0u || distance <= 0.0) {
        return float3(1.0, 1.0, 1.0);
    }

    PathTraceMaterial mediumMaterial = gMaterials[stack.entries[stack.count - 1u].materialIndex];
    float attenuationDistance = max(mediumMaterial.glassParams.w, 0.0);
    if (attenuationDistance <= 0.0001) {
        return float3(1.0, 1.0, 1.0);
    }

    float3 attenuationColor = clamp(
        mediumMaterial.attenuationColor.rgb,
        float3(0.0000001, 0.0000001, 0.0000001),
        float3(1.0, 1.0, 1.0));
    float exponent = max(distance, 0.0) / attenuationDistance;
    return float3(
        pow(attenuationColor.x, exponent),
        pow(attenuationColor.y, exponent),
        pow(attenuationColor.z, exponent));
}

void openPbrInteriorStackPush(inout OpenPBRInteriorStack stack, uint materialIndex, float ior)
{
    if (stack.count > 0u && stack.entries[stack.count - 1u].materialIndex == materialIndex) {
        stack.entries[stack.count - 1u].ior = ior;
        return;
    }
    if (stack.count >= kOpenPBRInteriorStackCapacity) {
        stack.entries[kOpenPBRInteriorStackCapacity - 1u].materialIndex = materialIndex;
        stack.entries[kOpenPBRInteriorStackCapacity - 1u].ior = ior;
        return;
    }

    stack.entries[stack.count].materialIndex = materialIndex;
    stack.entries[stack.count].ior = ior;
    stack.count += 1u;
}

void openPbrInteriorStackPopMaterial(inout OpenPBRInteriorStack stack, uint materialIndex)
{
    if (stack.count == 0u) {
        return;
    }
    if (stack.entries[stack.count - 1u].materialIndex == materialIndex) {
        stack.count -= 1u;
        return;
    }

    for (uint i = stack.count; i > 0u; --i) {
        uint entryIndex = i - 1u;
        if (stack.entries[entryIndex].materialIndex == materialIndex) {
            for (uint j = entryIndex; j + 1u < stack.count; ++j) {
                stack.entries[j] = stack.entries[j + 1u];
            }
            stack.count -= 1u;
            return;
        }
    }
}

void openPbrUpdateInteriorStackAfterTransmission(
    inout OpenPBRInteriorStack stack,
    HitInfo hit,
    PathTraceMaterial material)
{
    if (hit.frontFace) {
        openPbrInteriorStackPush(stack, hit.materialIndex, openPbrMaterialIor(material));
    } else {
        openPbrInteriorStackPopMaterial(stack, hit.materialIndex);
    }
}

OpenPBRMaterialEval evaluateOpenPBRMaterial(
    HitInfo hit,
    PathTraceMaterial material,
    ScenePathTracePush push,
    RayCone rayCone,
    float3 rayDirection,
    inout uint seed)
{
    bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
    float4 baseColorSample = loadMaterialTextureForMaterial(
        material.baseColorTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed);
    float4 metallicRoughnessSample = loadMaterialTextureForMaterial(
        material.metallicRoughnessTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed);
    float4 emissiveSample = loadMaterialTextureForMaterial(
        material.emissiveTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed);
    float4 occlusionSample = loadMaterialTextureForMaterial(
        material.occlusionTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed);
    float4 transmissionSample = loadMaterialTextureForMaterial(
        material.transmissionTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed);
    float3 baseColor = saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.001, 1.0);
    float transmission = saturate(material.glassParams.x * transmissionSample.r);
    float ior = clamp(material.glassParams.y, 1.0, 3.0);
    float3 emissive = max(material.emissive.rgb * srgbToLinear(emissiveSample.rgb), float3(0.0, 0.0, 0.0));

    OpenPBRMaterialEval result;
    result.inputs = openpbr_make_default_resolved_inputs();
    result.occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

    result.inputs.base_weight = 1.0;
    result.inputs.base_color = baseColor;
    result.inputs.base_diffuse_roughness = roughness;
    result.inputs.base_metalness = metallic;

    result.inputs.specular_weight = 1.0;
    result.inputs.specular_color = float3(1.0, 1.0, 1.0);
    result.inputs.specular_roughness = roughness;
    result.inputs.specular_roughness_anisotropy = 0.0;
    result.inputs.specular_ior = ior;
    result.inputs.specular_anisotropy_rotation_cos_sin = float2(1.0, 0.0);

    result.inputs.transmission_weight = transmission * (1.0 - metallic);
    result.inputs.transmission_color = powColor(baseColor, 0.5);
    result.inputs.transmission_depth = 0.0;
    result.inputs.transmission_scatter = float3(0.0, 0.0, 0.0);
    result.inputs.transmission_scatter_anisotropy = 0.0;

    result.inputs.subsurface_weight = 0.0;

    float emissionLuminance = luminanceMax(emissive);
    result.inputs.emission_luminance = emissionLuminance;
    result.inputs.emission_color = emissionLuminance > 0.0 ? emissive / emissionLuminance : float3(1.0, 1.0, 1.0);

    result.inputs.geometry_opacity = saturate(material.baseColor.a * baseColorSample.a);
    result.inputs.geometry_thin_walled = false;
    result.inputs.geometry_basis = openpbr_make_basis(
        safeNormalize(hit.normal, hit.geometryNormal),
        safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)),
        safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)));
    result.inputs.geometry_coat_basis = result.inputs.geometry_basis;
    return result;
}

float3 openPbrRayOffsetNormal(float3 direction, float3 geometryNormal)
{
    return dot(direction, geometryNormal) >= 0.0 ? geometryNormal : -geometryNormal;
}

float3 pathTraceOpenPBR(Ray ray, RayCone rayCone, ScenePathTracePush push, inout uint seed)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    OpenPBRInteriorStack interiorStack = makeOpenPBRInteriorStack();

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosest(ray, push, hit)) {
            float misWeight = depth == 0u
                ? 1.0
                : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
            radiance += throughput * environmentRadiance(ray.direction, push, depth == 0u) * misWeight;
            break;
        }

        rayCone = propagateRayCone(rayCone, hit.t);
        throughput *= openPbrVolumeTransmittance(interiorStack, hit.t);
        if (luminanceMax(throughput) <= 0.000001) {
            break;
        }

        PathTraceMaterial material = gMaterials[hit.materialIndex];
        float4 normalSample = loadMaterialTextureForMaterial(
            material.normalTexture,
            hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            materialNeedsDeterministicTextureMip0(material),
            seed);
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);

        OpenPBRMaterialEval materialEval = evaluateOpenPBRMaterial(
            hit,
            material,
            push,
            rayCone,
            ray.direction,
            seed);
        float3 viewDirection = safeNormalize(-ray.direction, hit.normal);
        OpenPBR_PreparedBsdf prepared = openpbr_prepare(
            materialEval.inputs,
            throughput,
            OpenPBR_BaseRgbWavelengths_nm,
            openPbrExteriorIorForHit(interiorStack, hit),
            viewDirection);

        radiance += throughput * max(prepared.emission, float3(0.0, 0.0, 0.0));

        EnvironmentLightSample lightSample;
        if (sampleEnvironmentLight(push, seed, lightSample)) {
            float lightBsdfPdf = openpbr_pdf(prepared, lightSample.direction);
            OpenPBR_DiffuseSpecular lightBsdf = openpbr_eval(prepared, lightSample.direction);
            float3 lightWeight = openpbr_get_sum_of_diffuse_specular(lightBsdf);
            bool occluded = lightBsdfPdf <= 0.0 ||
                luminanceMax(lightWeight) <= 0.0 ||
                traceShadow(
                    offsetRay(hit.position, openPbrRayOffsetNormal(lightSample.direction, hit.geometryNormal)),
                    lightSample.direction,
                    10000.0,
                    push);
            if (!occluded) {
                float lightMisWeight = environmentLightMisWeight(
                    push,
                    lightSample.pdfSolidAngle,
                    lightBsdfPdf);
                radiance += throughput *
                    lightWeight *
                    materialEval.occlusion *
                    (lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001));
            }
        }

        float3 sampledDirection;
        OpenPBR_DiffuseSpecular sampledWeight;
        float sampledPdf;
        OpenPBR_BsdfLobeType sampledType;
        openpbr_sample(
            prepared,
            float3(rand(seed), rand(seed), rand(seed)),
            sampledDirection,
            sampledWeight,
            sampledPdf,
            sampledType);

        if (sampledPdf <= 0.0 || sampledType == OpenPBR_BsdfLobeTypeNone) {
            break;
        }

        sampledDirection = safeNormalize(sampledDirection, hit.normal);
        throughput *= max(openpbr_get_sum_of_diffuse_specular(sampledWeight), float3(0.0, 0.0, 0.0));
        previousBsdfPdf = sampledPdf;
        previousScatterDelta = bool(sampledType & OpenPBR_BsdfLobeTypeSpecular);
        if (!previousScatterDelta) {
            rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
        }
        if (bool(sampledType & OpenPBR_BsdfLobeTypeTransmission)) {
            openPbrUpdateInteriorStackAfterTransmission(interiorStack, hit, material);
        }
        ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(sampledDirection, hit.geometryNormal));
        ray.direction = sampledDirection;

        if (depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

    return radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                                  uniform ScenePathTracePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint samples = clamp(push.samples, 1u, 16u);
    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.accumulationFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed));
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        color += pathTraceOpenPBR(ray, rayCone, push, seed);
    }
    color /= float(samples);

    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
    color = saturate(color);

    if (push.enableAccumulation != 0u) {
        if (push.hasHistory != 0u) {
            float accumulationWeight = 1.0 / float(push.accumulationFrame + 1u);
            float3 previousColor = gHistoryPrevious[pixel].rgb;
            color = lerp(previousColor, color, accumulationWeight);
        }
        gHistoryCurrent[pixel] = float4(color, 1.0);
    }

    gOutput[pixel] = float4(color, 1.0);
}
