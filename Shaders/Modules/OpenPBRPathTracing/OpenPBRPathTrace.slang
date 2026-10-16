implementing OpenPBRPathTracing;

import Core;
import MaterialTextureDecode;
import RTXCRMaterialAdapter;
import RayMaterialExecution;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import SceneSurfaceMaterial;
import SurfaceLighting;
import ScenePathTracing;
import SHARC;
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.ScenePathTracing;
using Metallic.SHARC;

namespace Metallic.OpenPBRPathTracing
{

public static const uint kOpenPBRInteriorStackCapacity = 8u;

public struct OpenPBRInteriorStackEntry
{
    public uint materialIndex;
    public float ior;
};

public struct OpenPBRInteriorStack
{
    public OpenPBRInteriorStackEntry entries[kOpenPBRInteriorStackCapacity];
    public uint count;
};


public OpenPBRInteriorStack makeOpenPBRInteriorStack()
{
    OpenPBRInteriorStack stack;
    stack.count = 0u;
    return stack;
}

public float openPbrMaterialIor(OpenPBRMaterialParameters material)
{
    return clamp(material.glassParams.y, 1.0, 3.0);
}

public float openPbrInteriorStackTopIor(OpenPBRInteriorStack stack)
{
    return stack.count > 0u ? stack.entries[stack.count - 1u].ior : OpenPBRVacuumIor;
}

public float openPbrInteriorStackNextIor(OpenPBRInteriorStack stack)
{
    return stack.count > 1u ? stack.entries[stack.count - 2u].ior : OpenPBRVacuumIor;
}

public float openPbrExteriorIorForHit(OpenPBRInteriorStack stack, HitInfo hit)
{
    if (hit.frontFace) {
        return openPbrInteriorStackTopIor(stack);
    }
    if (stack.count > 0u && stack.entries[stack.count - 1u].materialIndex == hit.materialIndex) {
        return openPbrInteriorStackNextIor(stack);
    }
    return openPbrInteriorStackTopIor(stack);
}

public float3 openPbrVolumeTransmittance(OpenPBRInteriorStack stack, float distance)
{
    if (stack.count == 0u || distance <= 0.0) {
        return float3(1.0, 1.0, 1.0);
    }

    OpenPBRMaterialParameters mediumMaterial = loadSceneOpenPBRMaterial(stack.entries[stack.count - 1u].materialIndex);
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

public void openPbrInteriorStackPush(inout OpenPBRInteriorStack stack, uint materialIndex, float ior)
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

public void openPbrInteriorStackPopMaterial(inout OpenPBRInteriorStack stack, uint materialIndex)
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

public void openPbrUpdateInteriorStackAfterTransmission(
    inout OpenPBRInteriorStack stack,
    HitInfo hit,
    OpenPBRMaterialParameters material)
{
    if (hit.frontFace) {
        openPbrInteriorStackPush(stack, hit.materialIndex, openPbrMaterialIor(material));
    } else {
        openPbrInteriorStackPopMaterial(stack, hit.materialIndex);
    }
}

public float3 openPbrRayOffsetNormal(float3 direction, float3 geometryNormal)
{
    return dot(direction, geometryNormal) >= 0.0 ? geometryNormal : -geometryNormal;
}

public static const uint kOpenPBRShadowTransmissionMaxHits = 16u;

public float3 openPbrShadowTransmittance(
    float3 origin,
    float3 direction,
    float maxDistance,
    OpenPBRInteriorStack interiorStack,
    ScenePathTracePush push)
{
    Ray ray;
    ray.origin = origin;
    ray.direction = Metallic.ScenePath::safeNormalize(direction, float3(0.0, 1.0, 0.0));
    float remainingDistance = max(maxDistance, 0.001);
    float3 transmittance = float3(1.0, 1.0, 1.0);

    for (uint hitIndex = 0u; hitIndex < kOpenPBRShadowTransmissionMaxHits; ++hitIndex) {
        ScenePathTracePush shadowPush = push;
        shadowPush.clipOrtho.x = 0.001;
        shadowPush.clipOrtho.y = remainingDistance;
        HitInfo shadowHit;
        bool opaqueShadow;
        if (!traceClosestShadow(ray, shadowPush, opaqueShadow, shadowHit) || shadowHit.t >= remainingDistance) {
            return transmittance;
        }
        if (opaqueShadow) { return float3(0.0); }

        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableVolumeAttenuation)) {
            transmittance *= openPbrVolumeTransmittance(interiorStack, shadowHit.t);
        }
        let shadowTransport = loadSceneMaterial(shadowHit.materialIndex);
        if (shadowTransport.textureParams.w > 1.5) {
            float alpha = evaluateSceneCoverage(shadowHit.materialIndex, sceneMaterialCoverageParameters(shadowTransport), shadowHit.texcoord, push).opacity;
            transmittance *= 1.0 - alpha;
            remainingDistance -= shadowHit.t;
            ray.origin = offsetRay(shadowHit.position, openPbrRayOffsetNormal(ray.direction, shadowHit.geometryNormal));
            if (remainingDistance <= 0.001 || luminanceMax(transmittance) <= 0.0001) { return transmittance; }
            continue;
        }
        // Slab and Fiber have independent scattering contracts and do not
        // participate in this OpenPBR straight-through transmission policy.
        if (materialDomain(shadowTransport) == kMaterialDomainFiber ||
            sceneSurfaceMaterialFamily(shadowHit.materialIndex) != 0u) { return float3(0); }
        let shadowMaterial = sceneOpenPBRParameters(shadowTransport);
        float4 transmissionSample = loadMaterialTextureMip0(
            shadowMaterial.transmissionTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        float4 metallicRoughnessSample = loadMaterialTextureMip0(
            shadowMaterial.metallicRoughnessTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures)) {
            transmissionSample = float4(1.0, 1.0, 1.0, 1.0);
            metallicRoughnessSample = float4(1.0, 1.0, 1.0, 1.0);
        }
        float metallic = saturate(shadowMaterial.params.x * metallicRoughnessSample.b);
        float surfaceTransmission =
            saturate(shadowMaterial.glassParams.x * transmissionSample.r) *
            (1.0 - metallic);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableTransmission)) {
            surfaceTransmission = 0.0;
        }
        if (surfaceTransmission <= 0.01) {
            return float3(0.0, 0.0, 0.0);
        }

        float4 baseColorSample = loadMaterialTextureMip0(
            shadowMaterial.baseColorTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures)) {
            baseColorSample = float4(1.0, 1.0, 1.0, 1.0);
        }
        float3 baseColor = saturate(
            materialTextureModulate(shadowMaterial.baseColor.rgb, baseColorSample.rgb, shadowMaterial.baseColorTexture.transform0.w));
        transmittance *= surfaceTransmission * powColor(baseColor, 0.5);
        if (luminanceMax(transmittance) <= 0.0001) {
            return float3(0.0, 0.0, 0.0);
        }

        // This NEE visibility term uses straight-through transmission; fully
        // refracted connections remain the responsibility of BSDF-sampled paths.
        // An exit interface must not collapse visibility to an opaque black band.
        openPbrUpdateInteriorStackAfterTransmission(
            interiorStack,
            shadowHit,
            shadowMaterial);
        remainingDistance -= shadowHit.t;
        ray.origin = offsetRay(
            shadowHit.position,
            openPbrRayOffsetNormal(ray.direction, shadowHit.geometryNormal));
    }

    return float3(0.0, 0.0, 0.0);
}

public struct PathSurfaceLights : ISurfaceLightProvider
{
    public ScenePathTracePush push;
    public HitInfo hit;
    public float surfaceIor;
    public bool transmittingSurface;
    public OpenPBRInteriorStack interiorStack;
    public float3 viewDirection;
    public uint seed;
    public uint count() { return openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting) ? 0u : kCelestialLightCount + 1u; }
    [mutating]
    public SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result = (SurfaceLightSample)0;
        if (index < kCelestialLightCount) {
            PunctualLightSample light = samplePathCelestialLightAtPosition(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getSceneEnvironmentResources().celestialLights), index, hit.position, seed);
            result.direction = light.direction;
            result.illuminance = light.illuminance;
            result.distance = light.distance;
            result.inverseSourcePdf = any(light.illuminance > 0.0) ? 1.0 : 0.0;
            result.source = celestialSourceIdentity(index);
            return result;
        }
        uint source;
        float inversePdf;
        if (selectPunctualLight(PunctualSamplingResources(sceneLights(), sceneReGIRGrid(), sceneLightsPdf()), hit.position, seed, source, inversePdf)) {
            PunctualLightSample light = samplePunctualLight(sceneLights(), source, hit.position);
            result.direction = light.direction;
            result.illuminance = light.illuminance;
            result.distance = light.distance;
            result.inverseSourcePdf = inversePdf;
            result.source = source;
        }
        return result;
    }
    public float3 visibility(SurfaceLightSample light)
    {
        if ((light.source & kCelestialSourceBit) != 0u &&
            !celestialLightCastsShadow(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getSceneEnvironmentResources().celestialLights),
                light.source & ~kCelestialSourceBit)) { return float3(1.0); }
        float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
        OpenPBRInteriorStack shadowStack = interiorStack;
        if (transmittingSurface && dot(viewDirection, hit.geometryNormal) * dot(light.direction, hit.geometryNormal) < 0.0) {
            if (hit.frontFace) { openPbrInteriorStackPush(shadowStack, hit.materialIndex, surfaceIor); }
            else { openPbrInteriorStackPopMaterial(shadowStack, hit.materialIndex); }
        }
        float distance = max(light.distance - length(origin - hit.position), 0.0);
        return openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) ? float3(1)
            : openPbrShadowTransmittance(origin, light.direction, distance, shadowStack, push);
    }
};

// Keep the first authored hit separate from the mutable shading hit. The first
// sample fills this cache at the normal path-query site; guides and subsequent
// samples reuse it without moving hit reconstruction out of the path loop.
public struct OpenPBRPrimaryQuery
{
    public HitInfo hit;
    public bool queried;
    public bool found;
};


public float3 pathTraceOpenPBRWithPrimaryQuery(
    Ray ray,
    RayCone rayCone,
    ScenePathTracePush push,
    inout uint seed,
    out float specularHitDistance,
    inout OpenPBRPrimaryQuery primaryQuery,
    uint2 guidePixel = uint2(0xffffffffu)
)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    OpenPBRInteriorStack interiorStack = makeOpenPBRInteriorStack();
    specularHitDistance = 0.0;
    bool captureSpecularHitDistance = false;
    uint alphaLayers = 0u;
    float3 primaryEndpoint = float3(0.0);
    float3 primaryOrigin = ray.origin;
    bool primarySurface = false;
    bool primaryQueryConsumed = false;
#if (SHARC_UPDATE || SHARC_QUERY)
#if METALLIC_PT_CACHE_QUERY_SPECIALIZATION
    // Select only numeric values and engine handle indices across the policy branch.
    // Materials is mandatory for every accepted production dispatch, including misses.
    let inactiveMaterials = getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset);
    float3 cacheCameraPosition = float3(0.0, 0.0, 0.0);
    float cacheLogarithmBase = 0.0;
    float cacheSceneScale = 0.0;
    float cacheLevelBias = 0.0;
    uint cacheCapacity = 0u;
    float cacheRadianceScale = 0.0;
    ResourceHandle<ByteAddressBuffer> cacheHashEntries = inactiveMaterials;
    ResourceHandle<ByteAddressBuffer> cacheAccumulation = inactiveMaterials;
    ResourceHandle<ByteAddressBuffer> cacheResolved = inactiveMaterials;
    if (kOpenPBRCacheQueryEnabled) {
        // Keep these expressions and their order in sync with makeCacheContext().
        ScenePathTraceCacheParams params = scenePathCacheParams()[0];
        let resources = getResourceParameterField<SceneResourceParameters, SceneSharcResourceParameters>(kSceneSharcOffset);
        cacheCameraPosition = params.sharcCameraPosition.xyz;
        cacheLogarithmBase = kSHARCGridLogarithmBase;
        cacheSceneScale = max(params.sharcSceneScale, 0.0001);
        cacheLevelBias = kSHARCGridLevelBias;
        cacheCapacity = params.sharcEntriesNum;
        cacheHashEntries = resources.sharcHashEntries;
        cacheAccumulation = resources.sharcAccumulation;
        cacheResolved = resources.sharcResolved;
        cacheRadianceScale = 1.0e3;
    }
    // Resolve after the merge; no opaque descriptor value crosses the branch.
    SHARCParameters selectedSharc;
    selectedSharc.hashGridParameters.cameraPosition = cacheCameraPosition;
    selectedSharc.hashGridParameters.logarithmBase = cacheLogarithmBase;
    selectedSharc.hashGridParameters.sceneScale = cacheSceneScale;
    selectedSharc.hashGridParameters.levelBias = cacheLevelBias;
    selectedSharc.hashGridData.capacity = cacheCapacity;
    selectedSharc.hashGridData.hashEntriesBuffer = resolveBuffer<RWStructuredBuffer<uint64_t>>(cacheHashEntries);
    selectedSharc.accumulationBuffer = resolveBuffer<RWStructuredBuffer<SHARCAccumulationData>>(cacheAccumulation);
    selectedSharc.resolvedBuffer = resolveBuffer<RWStructuredBuffer<SHARCPackedData>>(cacheResolved);
    selectedSharc.radianceScale = cacheRadianceScale;
    OpenPBRCacheContext cache = { selectedSharc };
#else
    OpenPBRCacheContext cache = makeOpenPbrCacheContext();
#endif
#endif

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        bool found;
        if (primaryQueryConsumed || !primaryQuery.queried) {
            found = traceClosest(ray, push, hit);
        } else {
            hit = primaryQuery.hit;
            found = primaryQuery.found;
        }
        if (!primaryQueryConsumed) {
            // BLEND continuation can revisit depth zero from another origin.
            primaryQueryConsumed = true;
            if (!primaryQuery.queried) {
                primaryQuery.hit = hit;
                primaryQuery.found = found;
                primaryQuery.queried = true;
#if !SHARC_UPDATE
                // Keep reconstruction in the original loop. Consume only an
                // independent guide evaluation here, before any path closure.
                if (guidePixel.x < push.width && guidePixel.y < push.height) {
                    evaluateAndWriteOpenPbrPrimaryGuides(guidePixel, ray, rayCone,
                        push, found, hit);
                }
#endif
            }
        }
        let control = shadeOpenPBRPathVertex(found, hit, push,
            ray, rayCone, seed, depth, radiance, throughput,
            previousBsdfPdf, previousScatterDelta, interiorStack,
            specularHitDistance, captureSpecularHitDistance,
            alphaLayers, primaryEndpoint, primaryOrigin, primarySurface
#if (SHARC_UPDATE || SHARC_QUERY)
            , cache
#endif
        );
        if (control == OpenPBRPathControl.Return) { return radiance; }
        if (control == OpenPBRPathControl.Stop) { break; }
    }

#if SHARC_UPDATE
    return radiance; // Cache entries are local scene radiance, without primary aerial perspective.
#else
    return primarySurface && push.debugView == kOpenPBRDebugViewFinal
        ? applyPrimaryAerialPerspective(radiance, primaryEndpoint, primaryOrigin) : radiance;
#endif
}

public float3 pathTraceOpenPBRWithPrimaryHit(Ray ray, RayCone rayCone, ScenePathTracePush push,
    inout uint seed, out float specularHitDistance, bool hasPrimaryHit, HitInfo primaryHit)
{
    OpenPBRPrimaryQuery primaryQuery;
    primaryQuery.hit = primaryHit;
    primaryQuery.queried = hasPrimaryHit;
    primaryQuery.found = hasPrimaryHit;
    return pathTraceOpenPBRWithPrimaryQuery(ray, rayCone, push, seed, specularHitDistance, primaryQuery);
}

// Preserve the ray-primary reference API; the raster path supplies an authored
// first hit while sharing BSDF sampling, nested IORs and volume attenuation.
public float3 pathTraceOpenPBR(Ray ray, RayCone rayCone, ScenePathTracePush push,
    inout uint seed, out float specularHitDistance)
{
    HitInfo unusedHit = {};
    return pathTraceOpenPBRWithPrimaryHit(ray, rayCone, push, seed,
        specularHitDistance, false, unusedHit);
}

#if METALLIC_PT_GUIDE_SPECIALIZATION
#if !SHARC_UPDATE
// One trace core serves both dispatch-uniform policies. The public guide entry
// remains a compatibility caller; the shared entry reads the guide specialization.
public void dispatchOpenPbrRayQueryPathTracePolicy(
    uint3 dispatchThreadId, ScenePathTracePush push, bool exportGuides)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    bool debugViewEnabled = push.debugView != kOpenPBRDebugViewFinal;
    uint samples;
    uint seed;
    float3 color;
    Ray guideRay;
    RayCone guideRayCone;
    float firstSampleSpecularHitDistance;
    OpenPBRPrimaryQuery primaryQuery;
    if (exportGuides) {
        samples = clamp(push.samples, 1u, 16u);
        seed = hash(
            pixel.x +
            pixel.y * push.width +
            (push.sampleFrame + 1u) * 0x9e3779b9u);
        float2 jitter = float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
        float2 samplePixel = float2(pixel) + jitter;
        guideRay.origin = rayOriginFromPixel(samplePixel, push);
        guideRay.direction = rayDirectionFromPixel(samplePixel, push);
        guideRayCone = primaryRayConeFromPixel(samplePixel, push);
        color = float3(0.0);
        firstSampleSpecularHitDistance = 0.0;
        primaryQuery = (OpenPBRPrimaryQuery)0;
    } else {
        samples = debugViewEnabled ? 1u : clamp(push.samples, 1u, 16u);
        // Keep diagnostic environment/shadow samples stable between frames.
        uint sampleFrame = debugViewEnabled ? 0u : push.sampleFrame;
        seed = hash(
            pixel.x +
            pixel.y * push.width +
            (sampleFrame + 1u) * 0x9e3779b9u);
        color = float3(0.0, 0.0, 0.0);
    }
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        Ray ray;
        RayCone rayCone;
        uint2 guidePixel = uint2(0xffffffffu);
        if (exportGuides) {
            ray = guideRay;
            rayCone = guideRayCone;
            guidePixel = pixel;
        } else {
            float2 jitter = push.temporalJitter != 0u
                ? float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
                : (samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed)));
            float2 samplePixel = float2(pixel) + jitter;

            ray.origin = rayOriginFromPixel(samplePixel, push);
            ray.direction = rayDirectionFromPixel(samplePixel, push);
            rayCone = primaryRayConeFromPixel(samplePixel, push);
            // Match pathTraceOpenPBR -> pathTraceOpenPBRWithPrimaryHit(false):
            // each ordinary sample owns a fresh zero hit/query.
            HitInfo unusedHit = {};
            primaryQuery.hit = unusedHit;
            primaryQuery.queried = false;
            primaryQuery.found = false;
        }
        float sampleSpecularHitDistance;
        color += pathTraceOpenPBRWithPrimaryQuery(ray, rayCone, push, seed,
            sampleSpecularHitDistance, primaryQuery, guidePixel);
        if (exportGuides && sampleIndex == 0u) {
            firstSampleSpecularHitDistance = sampleSpecularHitDistance;
        }
    }
    color /= float(samples);
    if (exportGuides) {
        writeOpenPbrPathColor(pixel, color, push);
        let resources = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
        resolveUniform(resources.guideSpecularHitDistance)[pixel] = max(firstSampleSpecularHitDistance, 0.0);
    } else {
        if (debugViewEnabled) {
            color = openPbrDisplayDebugColor(push, color);
        } else if (push.outputLinear == 0u) {
            color = Metallic.WorkingColor::toLinearRec709(color);
            color *= lightingExposure(sceneLights());
            color = color / (color + float3(1.0, 1.0, 1.0));
            color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
            color = saturate(color);
        }

        if (push.enableAccumulation != 0u && !debugViewEnabled) {
            color = accumulateStreamPathTrace(pixel, color, push);
        }

        sceneOutput()[pixel] = float4(color, 1.0);
    }
}
#endif

#endif
public void dispatchOpenPbrRayQueryPathTrace(uint3 dispatchThreadId)
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    uint2 pixel = dispatchThreadId.xy;
#if METALLIC_PT_GUIDE_SPECIALIZATION
    dispatchOpenPbrRayQueryPathTracePolicy(dispatchThreadId, push, kOpenPBRGuideExportEnabled);
#else
#if SHARC_UPDATE
    ScenePathTraceCacheParams cacheParams = scenePathCacheParams()[0];
    uint stride = max(cacheParams.sharcUpdateStride, 1u);
    if (any(pixel >= (uint2(push.width, push.height) + stride - 1u) / stride)) { return; }
    uint seed = hash(pixel.x + pixel.y * push.width + (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 samplePixel = (float2(pixel) + float2(rand(seed), rand(seed))) * float(stride);
    if (any(samplePixel >= float2(push.width, push.height))) { return; }
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    float ignoredHitDistance;
    pathTraceOpenPBR(ray, primaryRayConeFromPixel(samplePixel, push), push, seed, ignoredHitDistance);
#else
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    bool debugViewEnabled = push.debugView != kOpenPBRDebugViewFinal;
    uint samples = debugViewEnabled ? 1u : clamp(push.samples, 1u, 16u);
    // Keep diagnostic environment/shadow samples stable between frames.
    uint sampleFrame = debugViewEnabled ? 0u : push.sampleFrame;
    uint seed = hash(
        pixel.x +
        pixel.y * push.width +
        (sampleFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = push.temporalJitter != 0u
            ? float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
            : (samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed)));
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        float ignoredSpecularHitDistance;
        color += pathTraceOpenPBR(ray, rayCone, push, seed, ignoredSpecularHitDistance);
    }
    color /= float(samples);

    if (debugViewEnabled) {
        color = openPbrDisplayDebugColor(push, color);
    } else if (push.outputLinear == 0u) {
        color = Metallic.WorkingColor::toLinearRec709(color);
        color *= lightingExposure(sceneLights());
        color = color / (color + float3(1.0, 1.0, 1.0));
        color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
        color = saturate(color);
    }

    if (push.enableAccumulation != 0u && !debugViewEnabled) {
        color = accumulateStreamPathTrace(pixel, color, push);
    }

    sceneOutput()[pixel] = float4(color, 1.0);
#endif
#endif
}

} // namespace Metallic.OpenPBRPathTracing
