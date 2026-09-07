#include "OpenPBRRayQueryPathTrace.slang"
#include "SphericalHarmonics.slang"

[[vk::binding(51, 0)]]
StructuredBuffer<float4> gEnvironmentIrradianceSH;

float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) {
        return float3(0.0);
    }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    float3 irradiance = float3(0.0);
    for (uint i = 0u; i < 9u; ++i) {
        irradiance += gEnvironmentIrradianceSH[i].rgb * shBasis(i, localNormal);
    }
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

// One primary visibility ray and one shadow ray per contributing punctual light.
// Environment GI is a deterministic SH lookup, independent of the PDF sampler.
[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRealtimeLightingMain(uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform ScenePathTracePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 samplePixel = float2(pixel) + 0.5;
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    HitInfo hit;
    float3 radiance = float3(0.0);
    uint seed = hash(pixel.x + pixel.y * push.width);
    if (!traceClosest(ray, push, hit)) {
        radiance = environmentRadiance(ray.direction, push, true);
    } else {
        RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
        PathTraceMaterial material = gMaterials[hit.materialIndex];
        float4 normalSample = loadMaterialTextureForMaterial(material.normalTexture, hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0), push, cone, hit, ray.direction, true, seed);
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent,
            hit.bitangent, normalSample, material.textureParams.x);
        OpenPBRMaterialEval surface = evaluateOpenPBRMaterial(hit, material, push, cone, ray.direction, seed);
        float3 viewDirection = -ray.direction;
        OpenPBR_PreparedBsdf prepared = openpbr_prepare(surface.inputs, float3(1.0),
            OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, viewDirection);
        radiance = max(prepared.emission, float3(0.0));
        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) {
            for (uint i = 0u; i < punctualLightCount(); ++i) {
                PunctualLightSample light = samplePunctualLight(i, hit.position);
                float3 bsdf = max(openpbr_get_sum_of_diffuse_specular(
                    openpbr_eval(prepared, light.direction)), float3(0.0));
                if (luminanceMax(bsdf * light.illuminance) <= 0.0) { continue; }
                float3 origin = offsetRay(hit.position,
                    openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
                float distance = max(light.distance - length(origin - hit.position), 0.0);
                bool shadowed = !openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) &&
                    traceShadow(origin, light.direction, distance, push);
                if (!shadowed) { radiance += bsdf * light.illuminance; }
            }
        }
        float3 normal = dot(hit.normal, viewDirection) >= 0.0 ? hit.normal : -hit.normal;
        float diffuseWeight = surface.inputs.base_weight * (1.0 - surface.inputs.base_metalness) *
            (1.0 - surface.inputs.transmission_weight);
        radiance += surface.inputs.base_color * diffuseWeight * surface.occlusion *
            environmentDiffuseIrradiance(normal, push) / kPi;
    }
    if (push.outputLinear != 0u) {
        gOutput[pixel] = float4(max(radiance, float3(0.0)), 1.0);
        return;
    }
    float3 exposed = max(radiance * lightingExposure(), float3(0.0));
    float3 color = pow(exposed / (exposed + 1.0), 1.0 / 2.2);
    gOutput[pixel] = float4(saturate(color), 1.0);
}
