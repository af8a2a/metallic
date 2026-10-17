#language slang 2026
module VkGltfPathTracing;

import Core;
import Material;
import MaterialTextureDecode;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Lighting;
import Atmosphere;
import WorkingColor;
import ExperimentalRayTraversal;
import VkGltfBSDF;
#if METALLIC_SIMPLE_BSDF
import SimpleBSDF;
#endif

namespace Metallic.ExperimentalPath
{
using Metallic.Material;
using Metallic.ScenePath;
using Metallic.SceneSurface;
using Metallic.Lighting;
using Metallic.VkGltf;

// These flags share the existing pass UI and parameter ABI, not its BSDF.
bool flag(ScenePathTracePush push, uint bit) { return (push.debugFlags & (1u << bit)) != 0u; }
float3 random3(inout uint seed) { return float3(rand(seed), rand(seed), rand(seed)); }
float3 offsetNormal(float3 direction, float3 normal) { return dot(direction, normal) >= 0.0 ? normal : -normal; }

float4 materialTexture(MaterialTextureInfo texture, float4 fallback, HitInfo hit,
    RayCone cone, float3 direction, ScenePathTracePush push, bool deterministic, inout uint seed)
{
    return flag(push, 2) ? fallback : loadMaterialTextureForMaterial(texture, hit.texcoord,
        fallback, push, cone, hit, direction, deterministic, seed);
}

// Only the builtin scene payload is mapped. Custom material programs and Fiber
// are rejected by the CPU contract; they must never silently become glTF lobes.
#if METALLIC_SIMPLE_BSDF
PreparedMaterial resolveSimpleBSDFInputs(HitInfo hit, RayCone cone, float3 direction,
#else
PreparedMaterial evaluateMaterial(HitInfo hit, RayCone cone, float3 direction,
#endif
    ScenePathTracePush push, bool guides, inout uint seed)
{
    let source = loadSceneMaterial(hit.materialIndex);
    bool deterministic = guides || source.glassParams.x > 0.01;
    let base = materialTexture(source.baseColorTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let mr = materialTexture(source.metallicRoughnessTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let emission = materialTexture(source.emissiveTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let normal = materialTexture(source.normalTexture, float4(0.5, 0.5, 1, 1), hit, cone, direction, push, deterministic, seed);
    let occlusion = materialTexture(source.occlusionTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let transmission = materialTexture(source.transmissionTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let thickness = materialTexture(source.thicknessTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let specular = materialTexture(source.specularTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let specularColor = materialTexture(source.specularColorTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let diffuseTransmission = materialTexture(source.diffuseTransmissionTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    let diffuseTransmissionColor = materialTexture(source.diffuseTransmissionColorTexture, float4(1), hit, cone, direction, push, deterministic, seed);
    PreparedMaterial material = defaultPreparedMaterial();
    material.baseColor = saturate(materialTextureModulate(source.baseColor.rgb, base.rgb, source.baseColorTexture.transform0.w));
    material.opacity = saturate(source.baseColor.a * base.a);
    material.metallic = saturate(source.params.x * mr.b);
    float roughness = clamp(source.params.y * mr.g, 0.0014142, 1.0);
    material.roughness = float2(roughness * roughness);
    material.emissive = max(materialTextureModulate(source.emissive.rgb, emission.rgb, source.emissiveTexture.transform0.w), float3(0));
    material.occlusion = lerp(1.0, saturate(occlusion.r), saturate(source.textureParams.y));
    material.Ng = hit.frontFace ? hit.geometryNormal : -hit.geometryNormal;
    material.N = flag(push, 1) ? material.Ng : faceForwardShadingNormal(hit.normal, material.Ng, direction);
    if (!flag(push, 0) && !flag(push, 1) && !flag(push, 2)) {
        material.N = applyNormalMap(material.N, material.Ng, hit.tangent, hit.bitangent, normal, source.textureParams.x);
        material.N = faceForwardShadingNormal(material.N, material.Ng, direction);
    }
    float3 fallbackTangent = cross(abs(material.N.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0), material.N);
    material.T = Metallic.ScenePath::safeNormalize(hit.tangent.xyz - material.N * dot(material.N, hit.tangent.xyz), normalize(fallbackTangent));
    material.B = cross(material.N, material.T) * (hit.tangent.w >= 0.0 ? 1.0 : -1.0);
    material.Nc = material.N;
    material.transmission = flag(push, 5) ? 0.0 : saturate(source.glassParams.x * transmission.r);
    material.thickness = max(source.glassParams.z * thickness.g, 0.0);
    material.attenuationColor = clamp(source.attenuationColor.rgb, float3(1e-7), float3(1));
    material.attenuationDistance = source.glassParams.w > 0.0 ? source.glassParams.w : 1e30;
    material.ior1 = 1.0;
    material.ior2 = clamp(source.glassParams.y, 1.0, 3.0);
    if (!hit.frontFace && material.thickness > 0.0) {
        material.ior1 = material.ior2;
        material.ior2 = 1.0;
    }
    material.specular = saturate(source.specular.w * specular.a);
    material.specularColor = saturate(materialTextureModulate(source.specular.rgb, specularColor.rgb, source.specularColorTexture.transform0.w));
    material.diffuseTransmissionFactor = saturate(source.diffuseTransmission.w * diffuseTransmission.a);
    material.diffuseTransmissionColor = saturate(materialTextureModulate(source.diffuseTransmission.rgb,
        diffuseTransmissionColor.rgb, source.diffuseTransmissionColorTexture.transform0.w));
    return material;
}

#if METALLIC_SIMPLE_BSDF
// Material resources and mutable texture RNG belong to the program's source,
// never to the resolved closure or its per-view prepared value.
public struct SceneSimpleBSDFSource : ISimpleBSDFSource
{
    public HitInfo hit;
    public RayCone cone;
    public float3 direction;
    public ScenePathTracePush push;
    public bool guides;
    public uint seed;

    [mutating]
    public PreparedMaterial resolve(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        hit.position = context.positionWS;
        hit.geometryNormal = context.geometryNormalWS;
        hit.normal = context.shadingNormalWS;
        hit.tangent.xyz = context.tangentWS;
        hit.bitangent = context.bitangentWS;
        hit.texcoord = context.texCoord0;
        hit.frontFace = context.frontFace != 0u;
        hit.materialIndex = instance.parameterOffset / kSceneMaterialRecordBytes;
        return resolveSimpleBSDFInputs(hit, cone, direction, push, guides, seed);
    }
};

SimpleBSDFMaterialResult<SimpleBSDFClosure> evaluateSimpleMaterial(
    HitInfo hit, RayCone cone, float3 direction, ScenePathTracePush push, bool guides, inout uint seed)
{
    Material<SimpleBSDFMaterialProgram<SceneSimpleBSDFSource>> material = {};
    material.program.source.hit = hit;
    material.program.source.cone = cone;
    material.program.source.direction = direction;
    material.program.source.push = push;
    material.program.source.guides = guides;
    material.program.source.seed = seed;
    let result = material.evaluate(sceneSurfaceMaterialContextForRay(hit, cone, direction),
        sceneMaterialInstanceRef(hit.materialIndex));
    seed = material.program.source.seed;
    return result;
}

PreparedMaterial evaluateMaterial(HitInfo hit, RayCone cone, float3 direction,
    ScenePathTracePush push, bool guides, inout uint seed)
{
    return evaluateSimpleMaterial(hit, cone, direction, push, guides, seed).closure.inputs;
}
#endif

struct MediumStack
{
    uint materialIds[8];
    float3 extinction[8];
    uint count;
};

void crossMedium(inout MediumStack stack, HitInfo hit, PreparedMaterial material)
{
    if (material.thickness <= 0.0) { return; }
    if (hit.frontFace) {
        uint index = min(stack.count, 7u);
        stack.materialIds[index] = hit.materialIndex;
        stack.extinction[index] = volumeExtinctionCoefficient(material);
        stack.count = min(stack.count + 1u, 8u);
    } else {
        for (uint i = stack.count; i > 0u; --i) {
            if (stack.materialIds[i - 1u] == hit.materialIndex) {
                for (uint j = i; j < stack.count; ++j) {
                    stack.materialIds[j - 1u] = stack.materialIds[j];
                    stack.extinction[j - 1u] = stack.extinction[j];
                }
                stack.count--;
                break;
            }
        }
    }
}

float3 attenuation(MediumStack stack, float distance, ScenePathTracePush push)
{
    return stack.count == 0u || flag(push, 7) ? float3(1) : exp(-stack.extinction[stack.count - 1u] * max(distance, 0.0));
}

// An ordered straight-through shadow approximation, independent of BSDF
// sampling. Alpha BLEND is deterministic transmittance on shadow rays.
float3 shadowTransmittance(float3 origin, float3 direction, float distance,
    MediumStack stack, ScenePathTracePush push)
{
    if (flag(push, 6)) { return float3(1); }
    Ray ray = {origin, direction};
    float3 weight = float3(1);
    for (uint layer = 0u; layer < 16u; ++layer) {
        HitInfo hit;
        if (!traceClosest(ray, 0.001, max(distance, 0.001), push, hit)) { return weight * attenuation(stack, distance, push); }
        if (flag(push, 4)) { return float3(0); }
        let source = loadSceneMaterial(hit.materialIndex);
        weight *= attenuation(stack, hit.t, push);
        if (source.textureParams.w > 1.5) {
            weight *= 1.0 - evaluateSceneCoverage(hit.materialIndex, sceneMaterialCoverageParameters(source), hit.texcoord, push).opacity;
        } else {
            if (source.glassParams.x <= 0.0) { return float3(0); }
            uint shadowSeed = 0u;
            PreparedMaterial material = evaluateMaterial(hit, makeRayCone(0, 0), direction, push, true, shadowSeed);
            weight *= material.baseColor * material.transmission * (1.0 - material.metallic) *
                (1.0 - fresnelDielectric(max(material.ior1, material.ior2), abs(dot(direction, hit.geometryNormal))));
            crossMedium(stack, hit, material);
        }
        distance -= hit.t;
        if (distance <= 0.001 || luminanceMax(weight) <= 0.0001) { return weight; }
        ray.origin = offsetRay(hit.position, offsetNormal(direction, hit.geometryNormal));
    }
    return float3(0); // Conservative bound after the same 16 ordered layers.
}

#if METALLIC_SIMPLE_BSDF
float3 directLighting<TPrepared : IPreparedSimpleBSDFClosure>(TPrepared prepared,
    PreparedMaterial material, HitInfo hit, float3 view, MediumStack stack,
#else
float3 directLighting(PreparedMaterial material, HitInfo hit, float3 view, MediumStack stack,
#endif
    ScenePathTracePush push, inout uint seed)
{
    if (flag(push, 3)) { return float3(0); }
    float3 sum = float3(0);
    // Local, celestial and environment proposals share exactly one BSDF
    // evaluation site. Keep the loop rolled to avoid expansion by source kind.
    [loop]
    for (uint source = 0u; source < kCelestialLightCount + 2u; ++source) {
        float3 direction = float3(0), incident = float3(0);
        float distance = 10000.0, environmentPdf = 0.0;
        bool shadow = true;
        if (source < kCelestialLightCount) {
            let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(getSceneEnvironmentResources().celestialLights);
            let sample = samplePathCelestialLightAtPosition(celestial, source, hit.position, seed);
            direction = sample.direction; incident = sample.illuminance; distance = sample.distance;
            shadow = celestialLightCastsShadow(celestial, source);
        } else if (source == kCelestialLightCount) {
            uint lightIndex; float inversePdf;
            if (selectPunctualLight(PunctualSamplingResources(sceneLights(), sceneReGIRGrid(), sceneLightsPdf()),
                hit.position, seed, lightIndex, inversePdf)) {
                let sample = samplePunctualLight(sceneLights(), lightIndex, hit.position);
                direction = sample.direction; incident = sample.illuminance * inversePdf; distance = sample.distance;
            }
        } else {
            EnvironmentLightSample sample;
            if (sampleEnvironmentLight(push, seed, sample)) {
                direction = sample.direction; environmentPdf = sample.pdfSolidAngle;
                incident = sample.radiance / max(sample.pdfSolidAngle, 1e-6);
            }
        }
        if (!any(incident > 0.0)) { continue; }
#if METALLIC_SIMPLE_BSDF
        let evaluated = evaluateSimpleBSDF(prepared, direction, random3(seed));
#else
        let evaluated = evaluateBSDF(material, view, direction, random3(seed));
#endif
        if (evaluated.pdf <= 0.0) { continue; }
        float3 projected = evaluated.projectedDiffuse + evaluated.projectedGlossy;
        if (!any(projected > 0.0)) { continue; }
        MediumStack shadowStack = stack;
        if (dot(view, hit.geometryNormal) * dot(direction, hit.geometryNormal) < 0.0) { crossMedium(shadowStack, hit, material); }
        float3 origin = offsetRay(hit.position, offsetNormal(direction, hit.geometryNormal));
        float3 visibility = shadow ? shadowTransmittance(origin, direction,
            max(distance - length(origin - hit.position), 0.001), shadowStack, push) : float3(1);
        float mis = environmentPdf > 0.0 ? environmentLightMisWeight(push, environmentPdf, evaluated.pdf) : 1.0;
        sum += projected * incident * visibility * mis;
    }
    return sum;
}

float guideHardwareDepth(float linearDepth, ScenePathTracePush push)
{
    // Slang 2026.18.2 drops source-level precise on SPIR-V. Keep the depth
    // projection's rounding boundaries explicit across TraceRay/SER codegen.
    float nearPlane = max(push.clipOrtho.x, 0.0001);
    float farPlane = max(push.clipOrtho.y, nearPlane + 0.0001);
    float viewDepth = clamp(linearDepth, nearPlane, farPlane);
    float range = spirv_asm {
        %range : $$float = OpFSub $farPlane $nearPlane;
        OpDecorate %range NoContraction;
        result : $$float = OpCopyObject %range
    };
    float result;
    if (push.upProjection.w > 0.5) {
        result = spirv_asm {
            %offset : $$float = OpFSub $viewDepth $nearPlane;
            %depth : $$float = OpFDiv %offset $range;
            OpDecorate %offset NoContraction;
            OpDecorate %depth NoContraction;
            result : $$float = OpCopyObject %depth
        };
    } else {
        result = spirv_asm {
            %numerator : $$float = OpFMul $nearPlane $farPlane;
            %denominator : $$float = OpFMul $range $viewDepth;
            %inverse : $$float = OpFDiv $(1.0) %denominator;
            %scale : $$float = OpFDiv $farPlane $range;
            %negative : $$float = OpFNegate %numerator;
            OpDecorate %numerator NoContraction;
            OpDecorate %denominator NoContraction;
            OpDecorate %inverse NoContraction;
            OpDecorate %scale NoContraction;
            %depth : $$float = OpExtInst glsl450 Fma %negative %inverse %scale;
            OpDecorate %depth NoContraction;
            result : $$float = OpCopyObject %depth
        };
    }
    return saturate(result);
}

void writePrimaryGuides(uint2 pixel, Ray ray, RayCone cone, bool found, HitInfo hit, ScenePathTracePush push)
{
#if METALLIC_VKGLTF_GUIDES
    float3 albedo = float3(0), specular = float3(0), normal = float3(0, 1, 0);
    float roughness = 1.0, linearDepth = max(push.clipOrtho.y, 0.0);
    float2 motion = skyMotionVectorToPrevious(ray.direction, float2(pixel) + 0.5, push);
    if (found) {
        uint seed = hash(pixel.x + pixel.y * max(push.width, 1u) + (push.sampleFrame + 1u) * 0x85ebca6bu);
        let material = evaluateMaterial(hit, propagateRayCone(cone, hit.t), ray.direction, push, true, seed);
        roughness = sqrt(material.roughness.x);
        albedo = material.baseColor * (1.0 - material.metallic);
        normal = material.N;
        float ior = max(material.ior1, material.ior2);
        float f0 = (ior - 1.0) / (ior + 1.0); f0 *= f0;
        specular = saturate(approxSpecularIntegralGGX(lerp(f0 * material.specular * material.specularColor, material.baseColor,
            material.metallic), material.roughness.x, saturate(dot(normal, -ray.direction))));
        motion = motionVectorToPrevious(hit.position, float2(pixel) + 0.5, push);
        linearDepth = cameraLinearDepth(hit.position, push);
    }
    let resources = getSceneResourceParameterField<SceneGuideResourceParameters>(kSceneGuidesOffset);
    resolveUniform(resources.guideAlbedo)[pixel] = float4(saturate(albedo), 1);
    resolveUniform(resources.guideSpecularAlbedo)[pixel] = float4(specular, 1);
    resolveUniform(resources.guideNormalRoughness)[pixel] = float4(normal, roughness);
    resolveUniform(resources.guideMotion)[pixel] = motion;
    resolveUniform(resources.guideLinearDepth)[pixel] = linearDepth;
    resolveUniform(resources.guideDepth)[pixel] = guideHardwareDepth(linearDepth, push);
#endif
}

float3 tracePath(uint2 pixel, Ray ray, RayCone cone, ScenePathTracePush push,
    inout uint seed, bool writeGuides, out float specularDistance)
{
    float3 radiance = float3(0), throughput = float3(1);
    float previousPdf = 0.0;
    bool previousDelta = true, captureSpecular = false, primarySurface = false;
    float3 primaryOrigin = ray.origin, primaryEndpoint = float3(0);
    MediumStack stack = (MediumStack)0;
    specularDistance = 0.0;
    uint alphaLayers = 0u, depth = 0u;
    bool firstQuery = true;
    [loop]
    while (depth < clamp(push.maxDepth, 1u, 32u)) {
        HitInfo hit;
        bool found = traceClosest(ray, max(push.clipOrtho.x, 0.0001), max(push.clipOrtho.y, 0.001), push, hit);
        if (firstQuery) {
            if (writeGuides) { writePrimaryGuides(pixel, ray, cone, found, hit, push); }
            firstQuery = false;
        }
        if (!found) {
            if (captureSpecular) { specularDistance = 65504.0; }
            float mis = depth == 0u ? 1.0 : environmentBsdfMisWeight(push, previousPdf, ray.direction, previousDelta);
            float3 miss = environmentRadiance(ray.direction, push, depth == 0u, primaryOrigin) * mis;
            if (depth > 0u && previousDelta && push.environmentMode != kEnvironmentModeDisabled) {
                miss += physicalDeltaMissDisks(getSceneEnvironmentResources(), ray.direction, ray.origin) * max(push.environmentIntensity, 0.0);
            }
            radiance += throughput * miss;
            break;
        }
        if (captureSpecular) { specularDistance = hit.t; captureSpecular = false; }
        if (depth == 0u) { primarySurface = true; primaryEndpoint = hit.position; }
        throughput *= attenuation(stack, hit.t, push);
        cone = propagateRayCone(cone, hit.t);
        let source = loadSceneMaterial(hit.materialIndex);
        if (source.textureParams.w > 1.5) {
            float opacity = evaluateSceneCoverage(hit.materialIndex, sceneMaterialCoverageParameters(source), hit.texcoord, push).opacity;
            if (rand(seed) >= opacity) {
                if (alphaLayers >= 16u) { break; }
                alphaLayers++;
                if (depth == 0u) { primarySurface = false; }
                ray.origin = offsetRay(hit.position, offsetNormal(ray.direction, hit.geometryNormal));
                continue;
            }
        }
#if METALLIC_SIMPLE_BSDF
        let resolved = evaluateSimpleMaterial(hit, cone, ray.direction, push, false, seed);
        let material = resolved.closure.inputs;
        let prepared = prepareSimpleBSDF(resolved.closure, -ray.direction, TransportMode.Radiance);
#else
        let material = evaluateMaterial(hit, cone, ray.direction, push, false, seed);
#endif
        if (source.emissive.w != 0.0) { radiance += throughput * material.baseColor; break; }
#if METALLIC_SIMPLE_BSDF
        radiance += throughput * resolved.emission;
        radiance += throughput * directLighting(prepared, material, hit, -ray.direction, stack, push, seed);
        let scatter = sampleSimpleBSDF(prepared, random3(seed));
#else
        radiance += throughput * material.emissive;
        radiance += throughput * directLighting(material, hit, -ray.direction, stack, push, seed);
        let scatter = sampleBSDF(material, -ray.direction, random3(seed));
#endif
        if (scatter.eventType == kEventAbsorb || !any(scatter.weight > 0.0)) { break; }
        previousDelta = (scatter.eventType & kEventImpulse) != 0;
        previousPdf = scatter.pdf;
        throughput *= scatter.weight;
        if (depth == 0u) { captureSpecular = (scatter.eventType & (2 | 4)) != 0 && (scatter.eventType & kEventTransmission) == 0; }
        if ((scatter.eventType & kEventTransmission) != 0) { crossMedium(stack, hit, material); }
        if (!previousDelta) { cone = expandRayConeAfterScatter(cone, max(scatter.pdf, 0.0)); }
        ray.direction = Metallic.ScenePath::safeNormalize(scatter.direction, material.N);
        ray.origin = offsetRay(hit.position, offsetNormal(ray.direction, hit.geometryNormal));
        if (depth >= 2u) {
            float probability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > probability) { break; }
            throughput /= max(probability, 0.001);
        }
        depth++;
    }
    return primarySurface ? applyPrimaryAerialPerspective(radiance, primaryEndpoint, primaryOrigin) : radiance;
}

#if METALLIC_VKGLTF_GUIDES
// Experimental guides-only camera consistency: preserve the shared camera basis
// and projection inputs, but make the final affine/normalization rounding explicit.
float guidesCameraMultiply(float left, float right)
{
    return spirv_asm {
        %product : $$float = OpFMul $left $right;
        OpDecorate %product NoContraction;
        result : $$float = OpCopyObject %product
    };
}

float guidesCameraAffineComponent(float forward, float right, float up, float horizontal, float vertical)
{
    return spirv_asm {
        %horizontalPart : $$float = OpFMul $right $horizontal;
        %base : $$float = OpFAdd $forward %horizontalPart;
        %verticalPart : $$float = OpFMul $up $vertical;
        %component : $$float = OpFAdd %base %verticalPart;
        OpDecorate %horizontalPart NoContraction;
        OpDecorate %base NoContraction;
        OpDecorate %verticalPart NoContraction;
        OpDecorate %component NoContraction;
        result : $$float = OpCopyObject %component
    };
}

float3 guidesCameraNormalize(float3 value, float3 fallback)
{
    float lengthSquared = spirv_asm {
        %xx : $$float = OpFMul $(value.x) $(value.x);
        %yy : $$float = OpFMul $(value.y) $(value.y);
        %zz : $$float = OpFMul $(value.z) $(value.z);
        %xy : $$float = OpFAdd %xx %yy;
        %sum : $$float = OpFAdd %xy %zz;
        OpDecorate %xx NoContraction;
        OpDecorate %yy NoContraction;
        OpDecorate %zz NoContraction;
        OpDecorate %xy NoContraction;
        OpDecorate %sum NoContraction;
        result : $$float = OpCopyObject %sum
    };
    if (lengthSquared > 0.00000001) {
        float inverseLength = spirv_asm {
            %inverse : $$float = OpExtInst glsl450 InverseSqrt $lengthSquared;
            OpDecorate %inverse NoContraction;
            result : $$float = OpCopyObject %inverse
        };
        return float3(guidesCameraMultiply(value.x, inverseLength),
            guidesCameraMultiply(value.y, inverseLength), guidesCameraMultiply(value.z, inverseLength));
    }
    return fallback;
}

float3 guidesCameraDirectionFromPixel(float2 pixel, ScenePathTracePush push)
{
    float aspect = max(push.viewport.x, 0.001);
    float2 uv = pixel / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 forward = Metallic.ScenePath::safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = Metallic.ScenePath::safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = Metallic.ScenePath::safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = Metallic.ScenePath::safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    if (push.upProjection.w > 0.5) {
        return forward;
    }

    float fovRadians = clamp(push.viewport.w, 0.017453292, 3.12413936);
    float halfHeight = tan(fovRadians * 0.5);
    float horizontal = guidesCameraMultiply(guidesCameraMultiply(ndc.x, aspect), halfHeight);
    float vertical = guidesCameraMultiply(ndc.y, halfHeight);
    float3 direction = float3(
        guidesCameraAffineComponent(forward.x, right.x, up.x, horizontal, vertical),
        guidesCameraAffineComponent(forward.y, right.y, up.y, horizontal, vertical),
        guidesCameraAffineComponent(forward.z, right.z, up.z, horizontal, vertical));
    return guidesCameraNormalize(direction, forward);
}
#endif
public void dispatchVkGltf(uint2 pixel)
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    if (any(pixel >= uint2(push.width, push.height))) { return; }
    uint seed = hash(pixel.x + pixel.y * push.width + (push.sampleFrame + 1u) * 0x9e3779b9u);
    uint samples = clamp(push.samples, 1u, 16u);
#if METALLIC_PT_SINGLE_SAMPLE
    samples = 1u;
#endif
    float3 color = float3(0);
    float firstSpecularDistance = 0.0;
    for (uint sample = 0u; sample < samples; ++sample) {
#if METALLIC_VKGLTF_GUIDES
        float2 jitter = float2(0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
#else
        float2 jitter = push.temporalJitter != 0u ? float2(0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
            : (samples == 1u ? float2(0.5) : float2(rand(seed), rand(seed)));
#endif
        float2 samplePixel = float2(pixel) + jitter;
#if METALLIC_VKGLTF_GUIDES
        Ray ray = {rayOriginFromPixel(samplePixel, push), guidesCameraDirectionFromPixel(samplePixel, push)};
#else
        Ray ray = {rayOriginFromPixel(samplePixel, push), rayDirectionFromPixel(samplePixel, push)};
#endif
        RayCone cone = primaryRayConeFromPixel(samplePixel, push);
        float specularDistance;
        color += tracePath(pixel, ray, cone, push, seed, sample == 0u, specularDistance);
        if (sample == 0u) { firstSpecularDistance = specularDistance; }
    }
    color = max(color / float(samples), float3(0));
#if !METALLIC_VKGLTF_GUIDES
    if (push.outputLinear == 0u) {
        color = Metallic.WorkingColor::toLinearRec709(color) * lightingExposure(sceneLights());
        color = saturate(pow(color / (color + 1.0), 1.0 / 2.2));
    }
#endif
    if (push.enableAccumulation != 0u) { color = accumulate(pixel, color, push); }
#if METALLIC_VKGLTF_GUIDES
    if (push.outputLinear == 0u) { color *= lightingExposure(sceneLights()); }
    let resources = getSceneResourceParameterField<SceneGuideResourceParameters>(kSceneGuidesOffset);
    resolveUniform(resources.guideSpecularHitDistance)[pixel] = max(firstSpecularDistance, 0.0);
#endif
    sceneOutput()[pixel] = float4(color, 1);
}
} // namespace Metallic.ExperimentalPath
