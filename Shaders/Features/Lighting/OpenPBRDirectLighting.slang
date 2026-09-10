#include "../PathTracing/OpenPBRRayQueryPathTrace.slang"
#include "../../Libraries/Math/SphericalHarmonics.slang"

[[vk::binding(51, 0)]] StructuredBuffer<float4> gEnvironmentIrradianceSH;

float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0.0); }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gEnvironmentIrradianceSH);
    float3 irradiance = environment.evaluate(localNormal);
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

float deferredRadicalInverse(uint index, uint base)
{
    float value = 0.0;
    float scale = 1.0 / float(base);
    while (index != 0u) {
        value += float(index % base) * scale;
        index /= base;
        scale /= float(base);
    }
    return value;
}

float3 openPbrDirectVisibility(float3 origin, float3 direction, float distance, ScenePathTracePush push)
{
    if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) { return float3(1.0); }
    if (openPbrDebugFlag(push, kOpenPBRDebugUseOpaqueShadows)) {
        return traceShadow(origin, direction, distance, push) ? float3(0.0) : float3(1.0);
    }
    return openPbrShadowTransmittance(origin, direction, distance, makeOpenPBRInteriorStack(), push);
}

float3 shadeOpenPBRDirectSurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, bool sampledEnvironment,
    int fixedMetalness = -1, bool opaquePrimary = false)
{
    RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
    PathTraceMaterial material = gMaterials[hit.materialIndex];
    float3 authoredNormal = hit.normal;
    float4 normalSample = loadMaterialTextureForMaterial(material.normalTexture, hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0), push, cone, hit, ray.direction, true, seed);
    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent,
        hit.bitangent, normalSample, material.textureParams.x);
    // TBN and geometric normal remain authored; face-forward only the evaluated normal.
    hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
    OpenPBRMaterialEval surface = evaluateOpenPBRMaterial(hit, material, push, cone, ray.direction, seed);
    // Conservative tile classification supplies compile-time constants only
    // when the source factors/textures guarantee these exact values. Secondary
    // hits and shadow transmittance retain the unrestricted OpenPBR evaluator.
    if (fixedMetalness >= 0) { surface.inputs.base_metalness = float(fixedMetalness); }
    if (opaquePrimary) { surface.inputs.transmission_weight = 0.0; }
    if (push.debugView != kOpenPBRDebugViewFinal) {
        return openPbrPrimaryDebugColor(push, hit, authoredNormal, normalSample, surface, ray.direction);
    }
    float3 viewDirection = -ray.direction;
    OpenPBR_PreparedBsdf prepared = openpbr_prepare(surface.inputs, float3(1.0),
        OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, viewDirection);
    float3 radiance = max(prepared.emission, float3(0.0));
    if (!openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) {
        for (uint i = 0u; i < punctualLightCount(); ++i) {
            PunctualLightSample light = samplePunctualLight(i, hit.position);
            float3 bsdf = max(openpbr_get_sum_of_diffuse_specular(
                openpbr_eval(prepared, light.direction)), float3(0.0));
            if (luminanceMax(bsdf * light.illuminance) <= 0.0) { continue; }
            float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
            float distance = max(light.distance - length(origin - hit.position), 0.0);
            radiance += bsdf * light.illuminance * openPbrDirectVisibility(origin, light.direction, distance, push);
        }
    }
    if (sampledEnvironment && push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
        float3 environment = float3(0.0);
        float3 rotation = float3(rand(seed), rand(seed), rand(seed));
        uint count = max(push.samples, 1u);
        for (uint i = 0u; i < count; ++i) {
            float3 random = frac(float3((float(i) + 0.5) / float(count),
                deferredRadicalInverse(i, 2u), deferredRadicalInverse(i, 3u)) + rotation);
            float3 direction;
            OpenPBR_DiffuseSpecular weight;
            float pdf;
            OpenPBR_BsdfLobeType type;
            openpbr_sample(prepared, random, direction, weight, pdf, type);
            if (pdf <= 0.0 || type == OpenPBR_BsdfLobeTypeNone) { continue; }
            float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(direction, hit.geometryNormal));
            float3 visibility = openPbrDirectVisibility(origin, direction, max(push.clipOrtho.y, 1.0), push);
            environment += max(openpbr_get_sum_of_diffuse_specular(weight), float3(0.0)) *
                environmentRadiance(direction, push, false) * visibility;
        }
        radiance += environment * surface.occlusion / float(count);
    } else if (!sampledEnvironment) {
        float3 normal = dot(hit.normal, viewDirection) >= 0.0 ? hit.normal : -hit.normal;
        float diffuseWeight = surface.inputs.base_weight * (1.0 - surface.inputs.base_metalness) *
            (1.0 - surface.inputs.transmission_weight);
        radiance += surface.inputs.base_color * diffuseWeight * surface.occlusion *
            environmentDiffuseIrradiance(normal, push) / kPi;
    }
    return radiance;
}
