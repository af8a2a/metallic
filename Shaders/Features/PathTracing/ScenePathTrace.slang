#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY 0
#endif

import Atmosphere;
#include "SceneSurface.slang"
#include "SceneFiberPath.slang"

// ---------------------------------------------------------------------------
// Radiance cache integration (reference: RTXGI v2 SDK)
//
// The same entry point is compiled into several permutations:
//   (no defines)      plain path tracing
//   SHARC_UPDATE=1    sparse SHaRC update pass (populates the hash-grid cache)
//   SHARC_QUERY=1     full-resolution pass with SHaRC early termination
// ---------------------------------------------------------------------------
#if SHARC_UPDATE || SHARC_QUERY
#define METALLIC_PT_SHARC 1
#else
#define METALLIC_PT_SHARC 0
#endif

#if METALLIC_PT_SHARC
import SHARC;
using Metallic.SHARC;
#endif

// Cache mode pushed through the per-frame parameter buffer.
#define METALLIC_CACHE_MODE_OFF 0
#define METALLIC_CACHE_MODE_SHARC 1

// Per-frame radiance-cache parameters (binding 24). Layout must match
// ScenePathTraceCacheParams in BuiltinPassCommon.h byte for byte.
struct ScenePathTraceCacheParams
{
    float4 sharcCameraPosition;
    float4 sharcCameraPositionPrev;
    float sharcSceneScale;
    uint sharcEntriesNum;
    uint sharcAccumulationFrameNum;
    uint sharcStaleFrameNumMax;
    uint frameIndex;
    uint cacheMode;
    uint sharcUpdateStride;
    uint width;
    uint height;
    uint sharcQueryMinDepth;
    float sharcMinRoughness;
    uint sharcLobeMask;
};

// Match the tightly packed 80-byte CPU constants.
typedef StructuredBuffer<ScenePathTraceCacheParams, ScalarDataLayout> ScenePathTraceCacheBuffer;
// Keep sparse training code unchanged: hash insertion/atomic order can affect
// temporal cache contents. Narrow resource reads are for the shading/query path.
#if SHARC_UPDATE
#define gCacheParams resolveBuffer<ScenePathTraceCacheBuffer>(getResourceParameters<SceneResourceParameters>().cacheParams)
#else
#define gCacheParams resolveBuffer<ScenePathTraceCacheBuffer>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneCacheParamsOffset))
#endif

#if SHARC_UPDATE
#define gScene resolveAccelerationStructure(getResourceParameters<SceneResourceParameters>().scene)
#else
#define gScene resolveAccelerationStructure(getResourceParameterField<SceneResourceParameters, uint64_t>(kSceneSceneOffset))
#endif

#if !METALLIC_REALTIME_DEFERRED
#include "StreamSceneRayQuery.slang"
#endif

bool traceShadow(float3 origin, float3 direction, float maxDistance, ScenePathTracePush push)
{
#if METALLIC_REALTIME_DEFERRED
    return false;
#else
    Ray ray; ray.origin = origin; ray.direction = direction;
    HitInfo hit;
    return traceStreamClosest(ray, 0.001, max(maxDistance, 0.001), push, hit, true);
#endif
}

bool traceClosestImpl(Ray ray, ScenePathTracePush push, bool classifyShadow,
    out bool opaqueShadow, out HitInfo hit)
{
    opaqueShadow = false;
#if METALLIC_REALTIME_DEFERRED
    hit = (HitInfo)0;
    return false;
#else
    bool found = traceStreamClosest(ray, max(push.clipOrtho.x, 0.0001), max(push.clipOrtho.y, 0.001), push, hit);
    if (found && classifyShadow) {
        PathTraceMaterial material = loadSceneMaterial(hit.materialIndex);
        opaqueShadow = material.textureParams.w <= 1.5 && material.glassParams.x == 0.0;
    }
    return found;
#endif
}

bool traceClosest(Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    bool unusedOpaqueShadow;
    return traceClosestImpl(ray, push, false, unusedOpaqueShadow, hit);
}

bool traceClosestShadow(Ray ray, ScenePathTracePush push, out bool opaqueShadow, out HitInfo hit)
{
    // Ordered traversal is essential for nested media and transparent layers.
    // Streamed geometry deliberately retains its existing full decode path.
    return traceClosestImpl(ray, push, true, opaqueShadow, hit);
}

// Per-path radiance-cache state. Members exist only in permutations that use
// them; the plain permutation carries just the glossy cone tracker.
struct PathTraceCacheContext
{
#if METALLIC_PT_SHARC
    SHARCParameters sharc;
#if SHARC_UPDATE
    SHARCState sharcState;
#endif
#endif
    float materialRoughnessPrev;
};

float3 pathTrace(
    Ray ray,
    RayCone rayCone,
    ScenePathTracePush push,
    inout PathTraceCacheContext cache,
    inout uint seed,
    out float specularHitDistance,
    out Metallic.Atmosphere::AtmosphereSegment primaryAerial)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    cache.materialRoughnessPrev = 1.0;
    specularHitDistance = 0.0;
    bool captureSpecularHitDistance = false;
    float3 primaryEndpoint = float3(0.0);
    float3 primaryOrigin = ray.origin;
    bool primarySurface = false;
    primaryAerial.transmittance = float3(1.0);
    primaryAerial.radiance = float3(0.0);

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosest(ray, push, hit)) {
            if (captureSpecularHitDistance) {
                specularHitDistance = 65504.0;
                captureSpecularHitDistance = false;
            }
            float misWeight = depth == 0u
                ? 1.0
                : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
            float3 missRadiance = environmentRadiance(ray.direction, push, depth == 0u, primaryOrigin) * misWeight;
            if (depth > 0u && previousScatterDelta && push.environmentMode != kEnvironmentModeDisabled) {
                missRadiance += physicalDeltaMissDisks(getResourceParameters<SceneResourceParameters>().environment,
                    ray.direction, ray.origin) * max(push.environmentIntensity, 0.0);
            }
            radiance += throughput * missRadiance;
#if METALLIC_PT_SHARC && SHARC_UPDATE
            sharcUpdateMiss(cache.sharc, cache.sharcState, missRadiance);
#endif
            break;
        }
        if (captureSpecularHitDistance) {
            specularHitDistance = hit.t;
            captureSpecularHitDistance = false;
        }
        if (depth == 0u) { primaryEndpoint = hit.position; primarySurface = true; }

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // SHaRC accumulates each path segment independently: drop the signal
        // gathered for the previous segment before evaluating this hit.
        radiance = float3(0.0, 0.0, 0.0);
        throughput = float3(1.0, 1.0, 1.0);
#endif

        rayCone = propagateRayCone(rayCone, hit.t);
        PathTraceMaterial material = loadSceneMaterial(hit.materialIndex);
        bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
        float4 baseColorSample = loadMaterialTextureForMaterial(
            material.baseColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 metallicRoughnessSample = loadMaterialTextureForMaterial(
            material.metallicRoughnessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 emissiveSample = loadMaterialTextureForMaterial(
            material.emissiveTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 normalSample = loadMaterialTextureForMaterial(
            material.normalTexture,
            hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 occlusionSample = loadMaterialTextureForMaterial(
            material.occlusionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 transmissionSample = loadMaterialTextureForMaterial(
            material.transmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 thicknessSample = loadMaterialTextureForMaterial(
            material.thicknessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionColorSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
        float3 stableShadingNormal = hit.normal;
        hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
        float occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

        float3 baseColor = saturate(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
        float metallic = saturate(material.params.x * metallicRoughnessSample.b);
        float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.02, 1.0);
        float transmission = saturate(material.glassParams.x * transmissionSample.r);
        float ior = clamp(material.glassParams.y, 1.0, 3.0);
        float thickness = max(material.glassParams.z * thicknessSample.g, 0.0);
        float attenuationDistance = max(material.glassParams.w, 0.0);
        float3 attenuationColor = max(saturate(material.attenuationColor.rgb), float3(0.001, 0.001, 0.001));
        float diffuseTransmission = saturate(material.diffuseTransmission.a * diffuseTransmissionSample.a);
        float3 diffuseTransmissionColor = max(
            saturate(materialTextureModulate(material.diffuseTransmission.rgb, diffuseTransmissionColorSample.rgb, material.diffuseTransmissionColorTexture.transform0.w)),
            float3(0.001, 0.001, 0.001));
        float dielectricF0 = (ior - 1.0) / (ior + 1.0);
        dielectricF0 *= dielectricF0;
        float3 f0 = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);

        float3 emissive = max(materialTextureModulate(material.emissive.rgb, emissiveSample.rgb, material.emissiveTexture.transform0.w), float3(0.0, 0.0, 0.0));
        radiance += throughput * emissive;

        if (materialDomain(material) == kMaterialDomainFiber) {
            if (!shadeSceneFiber(material, hit, stableShadingNormal, push, depth,
                    ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta)) { break; }
            continue;
        }

        if (transmission > 0.01) {
            float eta = hit.frontFace ? 1.0 / max(ior, 1.001) : max(ior, 1.001);
            float3 refractedDirection = float3(0.0, 0.0, 0.0);
            bool canRefract = refractDirection(ray.direction, hit.normal, eta, refractedDirection);
            float reflectProbability = canRefract
                ? fresnelDielectric(ior, saturate(dot(-ray.direction, hit.normal)))
                : 1.0;
            if (!canRefract || rand(seed) < reflectProbability) {
                ray.direction = safeNormalize(reflect(ray.direction, hit.normal), hit.normal);
                if (depth == 0u) {
                    captureSpecularHitDistance = true;
                }
            } else {
                ray.direction = safeNormalize(refractedDirection, ray.direction);
                throughput *= volumeTransmittance(
                    baseColor,
                    transmission,
                    thickness,
                    attenuationDistance,
                    attenuationColor,
                    max(thickness, 0.001)) * transmission;
            }
            previousBsdfPdf = 1.0;
            previousScatterDelta = true;
            ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);
            continue;
        }

        // ------------------------------------------------------------------
        // Radiance cache (RTXGI SHaRC reference integration).
        // Delta-lobe segments (ideal transmission, hair) skip the cache,
        // mirroring the reference treatment of skipped delta vertices.
        // ------------------------------------------------------------------
#if METALLIC_PT_SHARC
        SHARCHitData sharcHitData;
        sharcHitData.positionWorld = hit.position;
        sharcHitData.normalWorld = hit.geometryNormal;

#if SHARC_QUERY
        // Camera hits must retain exact material/normal and direct-light detail.
        // The cache is only filtered enough to query after a surface bounce.
        if (depth >= max(gCacheParams[0].sharcQueryMinDepth, 1u)) {
            uint gridLevel = hashGridGetLevel(hit.position, cache.sharc.hashGridParameters);
            float voxelSize = hashGridGetVoxelSize(gridLevel, cache.sharc.hashGridParameters);
            bool isValidHit = hit.t > voxelSize * sqrt(3.0);

            float roughnessPrev = min(cache.materialRoughnessPrev, 0.99);
            float alpha = roughnessPrev * roughnessPrev;
            float footprint = hit.t * sqrt(max(0.5 * alpha * alpha / max(1.0 - alpha * alpha, 1e-6), 0.0));
            isValidHit &= footprint > voxelSize;

            float3 sharcRadiance;
            if (isValidHit && sharcGetCachedRadiance(cache.sharc, sharcHitData, sharcRadiance)) {
                radiance += sharcRadiance * throughput;
                break; // Terminate the path once we've looked up into the cache
            }
        }
#endif
#endif

        EnvironmentLightSample lightSample;
        // Sun and Moon remain fixed sources outside local RIS; physical mode
        // samples each finite disk before its projected BSDF and visibility.
        for (uint celestialSlot = 0u; celestialSlot < kCelestialLightCount; ++celestialSlot) {
            let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights);
            PunctualLightSample light = samplePathCelestialLightAtPosition(celestial, celestialSlot, hit.position, seed);
            float3 bsdf = punctualStandardBsdf(hit.normal, -ray.direction, light.direction,
                baseColor, f0, metallic, roughness, max(transmission, diffuseTransmission));
            if (luminanceMax(bsdf * light.illuminance) > 0.0) {
                float3 origin = offsetRay(hit.position,
                    dot(light.direction, hit.geometryNormal) >= 0.0 ? hit.geometryNormal : -hit.geometryNormal);
                if (!celestialLightCastsShadow(celestial, celestialSlot) ||
                    !traceShadow(origin, light.direction, light.distance, push)) {
                    radiance += throughput * bsdf * light.illuminance;
                }
            }
        }
        uint punctualLightIndex;
        float punctualInverseSourcePdf;
        if (selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), hit.position, seed, punctualLightIndex, punctualInverseSourcePdf)) {
            PunctualLightSample light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), punctualLightIndex, hit.position);
            float3 bsdf = punctualStandardBsdf(hit.normal, -ray.direction, light.direction,
                baseColor, f0, metallic, roughness, max(transmission, diffuseTransmission));
            if (luminanceMax(bsdf * light.illuminance) > 0.0) {
                float3 origin = offsetRay(hit.position,
                    dot(light.direction, hit.geometryNormal) >= 0.0 ? hit.geometryNormal : -hit.geometryNormal);
                if (!traceShadow(origin, light.direction,
                        max(light.distance - length(origin - hit.position), 0.0), push)) {
                    // This is the ReGIR RIS inverse source weight, not an
                    // environment solid-angle PDF. Delta-light MIS is one.
                    radiance += throughput * bsdf * light.illuminance * punctualInverseSourcePdf;
                }
            }
        }
        if (sampleEnvironmentLight(push, seed, lightSample)) {
            float3 lightDirection = lightSample.direction;
            float lightMisWeight = environmentLightMisWeight(
                push,
                lightSample.pdfSolidAngle,
                fallbackBsdfPdf(ray.direction, hit.normal, lightDirection, metallic, roughness, diffuseTransmission));
            float3 lightRadianceOverPdf =
                lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001);
            float nDotL = saturate(dot(hit.normal, lightDirection));
            if (nDotL > 0.0) {
                bool occluded = traceShadow(offsetRay(hit.position, hit.geometryNormal), lightDirection, 10000.0, push);
                if (!occluded) {
                    float3 viewDirection = -ray.direction;
                    float3 halfVector = safeNormalize(lightDirection + viewDirection, hit.normal);
                    float nDotH = saturate(dot(hit.normal, halfVector));
                    float specularPower = lerp(96.0, 8.0, roughness);
                    float3 diffuse = baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission) * (nDotL / kPi);
                    float3 specular = f0 * pow(nDotH, specularPower) * nDotL;
                    radiance += throughput * (diffuse + specular) * lightRadianceOverPdf;
                }
            }
            if (diffuseTransmission > 0.01) {
                float backNDotL = saturate(dot(-hit.normal, lightDirection));
                if (backNDotL > 0.0) {
                    float3 transmittedDiffuse =
                        baseColor *
                        diffuseTransmissionColor *
                        diffuseTransmission *
                        volumeTransmittance(
                            baseColor,
                            transmission,
                            thickness,
                            attenuationDistance,
                            attenuationColor,
                            max(thickness, 0.01)) *
                        (backNDotL / kPi);
                    radiance += throughput * transmittedDiffuse * lightRadianceOverPdf;
                }
            }
        }

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Feed the signal accumulated for this segment (emissive + direct
        // lighting) into the cache. Returning false means the path must stop.
        if (!sharcUpdateHit(cache.sharc, cache.sharcState, sharcHitData, radiance, rand(seed))) {
            break;
        }
#endif

        float3 diffuseDirection = sampleCosineHemisphere(hit.normal, seed);
        float3 incidentDirection = ray.direction;
        float3 reflectedDirection = reflect(incidentDirection, hit.normal);
        float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
        bool sampledSpecularLobe = false;
        if (rand(seed) < specularChance) {
            sampledSpecularLobe = true;
            float glossMix = roughness * roughness;
            ray.direction = safeNormalize(lerp(reflectedDirection, diffuseDirection, glossMix), reflectedDirection);
            throughput *= lerp(f0, baseColor, metallic * 0.35);
        } else if (rand(seed) < diffuseTransmission * (1.0 - metallic)) {
            ray.direction = sampleCosineHemisphere(-hit.normal, seed);
            throughput *=
                baseColor *
                diffuseTransmissionColor *
                diffuseTransmission *
                volumeTransmittance(
                    baseColor,
                    transmission,
                    thickness,
                    attenuationDistance,
                    attenuationColor,
                    max(thickness, 0.01));
        } else {
            ray.direction = diffuseDirection;
            throughput *= baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission);
        }
        if (depth == 0u) {
            captureSpecularHitDistance = sampledSpecularLobe;
        }
        previousBsdfPdf = max(
            fallbackBsdfPdf(incidentDirection, hit.normal, ray.direction, metallic, roughness, diffuseTransmission),
            0.000001);
        previousScatterDelta = false;
        rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
        ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);

#if METALLIC_PT_SHARC
        // Track the effective cone spread of the path so far for the glossy
        // footprint test performed on the next hit.
        cache.materialRoughnessPrev += sampledSpecularLobe ? roughness : 1.0;
#endif
#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Hand the segment throughput to the cache; each segment is
        // accumulated independently afterwards.
        sharcSetThroughput(cache.sharcState, throughput);
#endif

        if (depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

    if (primarySurface) {
        primaryAerial = physicalAerialSegment(getResourceParameters<SceneResourceParameters>().environment, primaryEndpoint, primaryOrigin);
    }
#if SHARC_UPDATE
    // Cache radiance stays local to the shaded point.
    return radiance;
#else
    return primaryAerial.transmittance * radiance + primaryAerial.radiance;
#endif
}

PathTraceCacheContext makeCacheContext()
{
    PathTraceCacheContext cache;
    ScenePathTraceCacheParams params = gCacheParams[0];
#if METALLIC_PT_SHARC
#if !SHARC_UPDATE
    let resources = getResourceParameterField<SceneResourceParameters, SceneSharcResourceParameters>(kSceneSharcOffset);
#endif
    cache.sharc.hashGridParameters.cameraPosition = params.sharcCameraPosition.xyz;
    cache.sharc.hashGridParameters.logarithmBase = kSHARCGridLogarithmBase;
    cache.sharc.hashGridParameters.sceneScale = max(params.sharcSceneScale, 0.0001);
    cache.sharc.hashGridParameters.levelBias = kSHARCGridLevelBias;

    cache.sharc.hashGridData.capacity = params.sharcEntriesNum;
#if SHARC_UPDATE
    cache.sharc.hashGridData.hashEntriesBuffer = resolveBuffer<RWStructuredBuffer<uint64_t>>(getResourceParameters<SceneResourceParameters>().sharcHashEntries);
    cache.sharc.accumulationBuffer = resolveBuffer<RWStructuredBuffer<SHARCAccumulationData>>(getResourceParameters<SceneResourceParameters>().sharcAccumulation);
    cache.sharc.resolvedBuffer = resolveBuffer<RWStructuredBuffer<SHARCPackedData>>(getResourceParameters<SceneResourceParameters>().sharcResolved);
#else
    cache.sharc.hashGridData.hashEntriesBuffer = resolveBuffer<RWStructuredBuffer<uint64_t>>(resources.sharcHashEntries);
    cache.sharc.accumulationBuffer = resolveBuffer<RWStructuredBuffer<SHARCAccumulationData>>(resources.sharcAccumulation);
    cache.sharc.resolvedBuffer = resolveBuffer<RWStructuredBuffer<SHARCPackedData>>(resources.sharcResolved);
#endif
    cache.sharc.radianceScale = 1.0e3;
#if SHARC_UPDATE
    sharcInit(cache.sharcState);
#endif
#endif
    cache.materialRoughnessPrev = 1.0;
    return cache;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    uint2 pixel = dispatchThreadId.xy;
    uint samples = clamp(push.samples, 1u, 16u);
    uint2 outputPixel = pixel;
    uint seed = 0u;

#if SHARC_UPDATE
    // Sparse update: one random sample per stride x stride block of pixels,
    // with coverage distributed over time through the jitter and frame index.
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    uint stride = max(cacheParams.sharcUpdateStride, 1u);
    if (pixel.x >= (push.width + stride - 1u) / stride || pixel.y >= (push.height + stride - 1u) / stride) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 updateJitter = float2(rand(seed), rand(seed));
    float2 samplePixel = (float2(pixel) + updateJitter) * float(stride);
    if (samplePixel.x >= float(push.width) || samplePixel.y >= float(push.height)) {
        return;
    }
    samples = 1u;
#else
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel);
#endif

    PathTraceCacheContext cache = makeCacheContext();

    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
#if SHARC_UPDATE
        float2 jitter = float2(0.0, 0.0);
#else
        float2 jitter = push.temporalJitter != 0u
            ? float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
            : (samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed)));
#endif
#if !SHARC_UPDATE
        samplePixel = float2(outputPixel) + jitter;
#endif

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        float ignoredSpecularHitDistance;
        Metallic.Atmosphere::AtmosphereSegment primaryAerial;
        color += pathTrace(ray, rayCone, push, cache, seed, ignoredSpecularHitDistance, primaryAerial);
    }
#if SHARC_UPDATE
    // Cache update passes produce no image output.
    return;
#endif
    color /= float(samples);

    if (push.outputLinear == 0u) {
        color = Metallic.WorkingColor::toLinearRec709(color);
        color *= lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
        color = color / (color + float3(1.0, 1.0, 1.0));
        color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
        color = saturate(color);
    }

    if (push.enableAccumulation != 0u) {
        color = accumulateStreamPathTrace(pixel, color, push);
    }

    gOutput[pixel] = float4(color, 1.0);
}
